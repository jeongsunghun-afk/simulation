#pragma once
// sim HAL(실물 배선) — MuJoCo 백엔드 + quad/cpp `QuadControl` 코어 소유(재기어/GEARBOX/솔버 셋업 재사용).
//   read():  q.d → LowState(관절·IMU·접촉).   write(): LowCmd.tau_ff → q.d->ctrl + mj_step(1스텝).
//   ★sim2real 에뮬레이션(실기 센서/버스 모델, trot_sim EST_CTRL와 동일 모델·seed2024·draw순서):
//     센서 노이즈(가우시안: ENCQ_N·ENCDQ_N·QUAT_N·GYRO_N·ACC_N)·센서 지연(SENSE_LAT_MS)·구동 지연(ACT_LAT_MS).
//     real_hal은 이 에뮬레이션 대신 실제 노이즈/지연을 가짐 → 컨트롤러/estimator 불변. 모든 param 0 = clean(회귀).
//   ★이관 1·2단계: 검증된 컨트롤러를 재작성 없이 HAL 경계 뒤로. real_hal은 read/write만 실센서/모터로 교체.
#include "hal/robot_interface.hpp"
#include <mujoco/mujoco.h>
#include "quad_control.hpp"        // quad/cpp/src (CMake include_dir) — ::QuadControl
#include "trot_controller.hpp"     // apply_env_gains
#include <random>
#include <vector>
#include <cmath>
#include <cstdlib>

namespace qc {

class MujocoHal : public RobotInterface {
  ::QuadControl q_;                 // m,d 소유 + 재기어/게인/q_home LUT/MPC 셋업(quad/cpp 그대로)
  // ── sim2real 센서/구동 에뮬레이션 상태(trot_sim와 동일) ──
  std::mt19937 rng_{2024};
  std::normal_distribution<double> nd_{0.0, 1.0};
  double gyro_n_ = 0, quat_n_ = 0, encq_n_ = 0, encdq_n_ = 0, acc_n_ = 0;  // 센서 노이즈
  int Lsense_ = 0, Lact_ = 0;                                              // 지연 스텝(센서·구동)
  int NJ_ = 0, fsz_ = 0;                                                   // 관절수·센서프레임 크기
  std::vector<std::vector<double>> sring_, cring_;                         // 센서·구동 지연 링(고정 크기)
  std::vector<std::vector<double>> pring_;                                 // ★드라이버 PD 목표 지연 링 [q_des|q̇_des|kp|kd]
  long step_ = 0;                                                          // 현재 틱(read/write 공유)
 public:
  explicit MujocoHal(const char* mjcf) {
    q_.load(mjcf); apply_env_gains(q_); q_.crouch_home(); q_.build_qhome_lut(); q_.setup_mpc();
    NJ_ = q_.m->nq - 7;  fsz_ = 2 * NJ_ + 11;                    // 프레임: [qn(NJ),dqn(NJ),quat(4),gyro(3),cts(4)]
    auto ev = [](const char* k) { return getenv(k) ? atof(getenv(k)) : 0.0; };
    gyro_n_ = ev("GYRO_N"); quat_n_ = ev("QUAT_N"); encq_n_ = ev("ENCQ_N"); encdq_n_ = ev("ENCDQ_N"); acc_n_ = ev("ACC_N");
    double dt = q_.m->opt.timestep;
    Lsense_ = (int)std::lround(ev("SENSE_LAT_MS") * 1e-3 / dt);
    Lact_   = (int)std::lround(ev("ACT_LAT_MS")   * 1e-3 / dt);
    sring_.assign(Lsense_ + 1, std::vector<double>(fsz_, 0.0));
    cring_.assign(Lact_ + 1,   std::vector<double>(q_.nu, 0.0));
    pring_.assign(Lact_ + 1,   std::vector<double>(4 * q_.nu, 0.0));
  }
  ::QuadControl& core() { return q_; }            // 컨트롤러 브리지가 같은 코어 공유(TrotCtrl(q_))

  int    nu() const override { return q_.nu; }
  double dt() const override { return q_.m->opt.timestep; }

  bool read(LowState& s) override {
    mjData* d = q_.d; const int nu = q_.nu;
    s.time = d->time;
    s.q.resize(nu); s.dq.resize(nu); s.tau_est.resize(nu);
    for (int i = 0; i < nu; ++i) s.tau_est[i] = d->actuator_force[i];   // 추정토크(지연 미적용, 현재 미소비)
    // 현재 발 접촉(dist<0.002, trot_sim EST와 동일 판정, 노이즈 없음)
    double cts0[4];
    for (int i = 0; i < 4; ++i) { bool con = false;
      for (int ci = 0; ci < d->ncon; ++ci) { const auto& c = d->contact[ci];
        if ((c.geom1 == q_.fgid[i] || c.geom2 == q_.fgid[i]) && c.dist < 0.002) { con = true; break; } }
      cts0[i] = con ? 1.0 : 0.0; }
    // 노이즈 센서프레임을 지연 링에 기록(trot_sim와 동일 draw 순서: ENCQ→ENCDQ→QUAT→GYRO→cts)
    { auto& fr = sring_[step_ % (long)sring_.size()]; int o = 0;
      for (int j = 0; j < NJ_; j++) fr[o++] = d->qpos[7 + j] + encq_n_  * nd_(rng_);
      for (int j = 0; j < NJ_; j++) fr[o++] = d->qvel[6 + j] + encdq_n_ * nd_(rng_);
      double dqp[4] = {1, 0.5 * quat_n_ * nd_(rng_), 0.5 * quat_n_ * nd_(rng_), 0.5 * quat_n_ * nd_(rng_)};
      mju_normalize4(dqp); double qq[4]; mju_mulQuat(qq, &d->qpos[3], dqp); for (int a = 0; a < 4; a++) fr[o++] = qq[a];
      for (int a = 0; a < 3; a++) fr[o++] = d->qvel[3 + a] + gyro_n_ * nd_(rng_);
      for (int i = 0; i < 4; i++) fr[o++] = cts0[i]; }
    // world 선가속(노이즈, 지연 없음=현재; trot_sim aw와 동일). real_hal=R·f_body+g
    for (int a = 0; a < 3; a++) s.imu_acc[a] = d->qacc[a] + acc_n_ * nd_(rng_);
    // 지연된 센서프레임 언팩(SENSE_LAT_MS 전) → LowState
    auto& df = sring_[std::max(0L, step_ - Lsense_) % (long)sring_.size()];
    { int o = 0; for (int j = 0; j < NJ_; j++) s.q[j] = df[o++]; for (int j = 0; j < NJ_; j++) s.dq[j] = df[o++];
      for (int a = 0; a < 4; a++) s.imu_quat[a] = df[o++]; for (int a = 0; a < 3; a++) s.imu_gyro[a] = df[o++];
      for (int i = 0; i < 4; i++) s.foot_force[i] = df[o++]; }
    return true;
  }

  bool write(const LowCmd& c) override {
    mjData* d = q_.d;
    // 구동 지연: 현재 토크를 링에 기록 → 지연된 토크(ACT_LAT_MS 전) 적용
    { auto& cf = cring_[step_ % (long)cring_.size()]; for (int i = 0; i < q_.nu; i++) cf[i] = c.tau_ff[i]; }
    auto& dc = cring_[std::max(0L, step_ - Lact_) % (long)cring_.size()];
    for (int i = 0; i < q_.nu; i++) d->ctrl[i] = dc[i];          // TrotCtrl이 tau 계산(kp/kd=0 규약)
    // ★2026-09-30 드라이버 PD(LowCmd kp/kd>0 일 때): 목표·게인도 토크와 같은 구동지연 → 현재 실상태로 PD. trot_sim DRV_TRACK 과 동일.
    { auto& pf = pring_[step_ % (long)pring_.size()]; pf.assign(4 * q_.nu, 0.0);
      for (int i = 0; i < q_.nu; i++) { pf[i] = c.q_des[i]; pf[q_.nu + i] = c.dq_des[i]; pf[2 * q_.nu + i] = c.kp[i]; pf[3 * q_.nu + i] = c.kd[i]; }
      const auto& pd = pring_[std::max(0L, step_ - Lact_) % (long)pring_.size()];
      if ((int)pd.size() == 4 * q_.nu)
        for (int i = 0; i < q_.nu; i++) if (pd[2 * q_.nu + i] != 0.0 || pd[3 * q_.nu + i] != 0.0)
          d->ctrl[i] += pd[2 * q_.nu + i] * (pd[i] - d->qpos[7 + i]) + pd[3 * q_.nu + i] * (pd[q_.nu + i] - d->qvel[6 + i]); }
    mj_step(q_.m, d);
    step_++;
    return true;
  }
};

}  // namespace qc
