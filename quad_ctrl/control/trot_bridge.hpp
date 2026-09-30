#pragma once
// 컨트롤러 브리지 — quad/cpp `TrotCtrl`(검증된 MPC+WBIC+gait+getup)을 qc HAL 경계 뒤로.
//   set_command(HighCmd) → TrotCtrl 세팅.  step(LowCmd) → ctrl.control()(q.d->ctrl 설정) → LowCmd.tau_ff 추출.
//   ★재작성 아님(wrap). 2단계에서 estimator(d_est) 주입, 3단계에서 HAL만 real 교체.
#include "trot_controller.hpp"        // ::TrotCtrl, ::QuadControl
#include "command/sport_client.hpp"   // qc::HighCmd, qc::Mode
#include "hal/robot_interface.hpp"    // qc::LowCmd

namespace qc {

class TrotBridge {
  ::QuadControl& q_;
  ::TrotCtrl ctrl_;
 public:
  explicit TrotBridge(::QuadControl& q) : q_(q), ctrl_(q) {}

  void set_command(const HighCmd& hc) {
    ctrl_.set_gait(hc.gait);
    ctrl_.V = hc.vx; ctrl_.VY = hc.vy; ctrl_.WZ = hc.wz;
    if (hc.body_h > 0.01) ctrl_.body_h = hc.body_h;   // 0=모델 기본(base_z0) 유지
    switch (hc.mode) {
      case Mode::Walk:      ctrl_.mode = "move";                       break;   // gait(trot/walk/run)로 보행
      case Mode::Stand:     ctrl_.mode = "move"; ctrl_.V = 0;          break;   // 정지 서기 = move V0
      case Mode::Off:       ctrl_.mode = "off";                        break;
      case Mode::Sit:       ctrl_.mode = "sit";                        break;
      case Mode::StandUp:   ctrl_.mode = "stand_up";                   break;
      case Mode::StandDown: ctrl_.mode = "stand_down";                 break;
    }
  }

  // ctrl_data=nullptr: q_.d(=d_phys, GT)로 계산(1단계). ctrl_data=d_est: 추정상태로 계산(2단계 EST) 후 복원.
  void step(LowCmd& cmd, mjData* ctrl_data = nullptr) {
    mjData* saved = q_.d;
    if (ctrl_data) q_.d = ctrl_data;       // 추정상태(d_est)에서 Jacobian/MPC/WBIC 계산
    ctrl_.control();                       // q_.d->ctrl 설정(mj_step은 HAL 몫)
    const int nu = q_.nu;
    cmd.tau_ff.resize(nu); cmd.q_des.setZero(nu); cmd.dq_des.setZero(nu);
    cmd.kp.setZero(nu);    cmd.kd.setZero(nu);
    for (int i = 0; i < nu; ++i) cmd.tau_ff[i] = q_.d->ctrl[i];   // 컨트롤러 tau → LowCmd(kp/kd=0)
    // ★2026-09-30 DRV_TRACK=1: 하이브리드 KinWBC 계획을 드라이버 PD 목표로(τ_ff + kp(q_des−q) + kd(q̇_des−q̇)).
    //   기본 0 = 종전 순수토크. ⚠실기 kp 단위(real_hal "quad TBD") 확인 전에는 켜지 말 것. 게인 TRK_KP_Q/TRK_KD_Q(trot_sim 과 동일).
    { static const bool DTRK = getenv("DRV_TRACK") && atoi(getenv("DRV_TRACK"));
      static const double TKP = getenv("TRK_KP_Q") ? atof(getenv("TRK_KP_Q")) : 20.0, TKD = getenv("TRK_KD_Q") ? atof(getenv("TRK_KD_Q")) : 1.0;
      if (DTRK && q_.mit_valid && q_.mit_qdes.size() == nu) {
        for (int i = 0; i < nu; ++i) { cmd.q_des[i] = q_.mit_qdes[i]; cmd.dq_des[i] = q_.mit_dqdes[i]; cmd.kp[i] = TKP; cmd.kd[i] = TKD; } }
      q_.mit_valid = false; }
    q_.d = saved;                          // ★HAL의 d_phys로 복원(write의 mj_step이 실물리를 밟도록)
  }

  ::TrotCtrl& raw() { return ctrl_; }      // (검증용: tiltdeg 등)
};

}  // namespace qc
