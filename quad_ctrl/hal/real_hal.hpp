#pragma once
// real HAL(실기, Pi 전용) — RGA RobotSharedMem(Gait) 백엔드. `RobotInterface` 구현.
//   ★참조: ~/문서/jsh/RobotTestGait (실제 SHM API·RT골격) + ★biped/emb/pace/RESULTS.md (제어법칙·단위·필드 실측확정).
//   read():  RobotMemGait_GetMotorStatus16 + GetIMU (관절축 deg) → LowState(rad).  write(): LowCmd → MotGeneral_t → SetMotorCommand16.
//   ★컨트롤러·estimator 불변 — sim(MujocoHal) 대비 read/write만 실센서/모터로 교체(이관 3단계).
//   ※RobotSharedMem.h는 Pi에만 존재 → 데스크톱 빌드는 자동 skip(아래 가드). robot_main도 이 가드로 Pi에서만.
//
// ★★PACE 실측 확정(2026-08-05): τ=fGainKp·(fPos−q)[rad]+fGainKd·(fVel−q̇)[rad/s]+fTorque(MIT)·torque_frame=joint(관절축)·단위 deg.
//   MotGeneral_t: fAccel=1.0(미사용)·fCurrent=fTorque(중복)·float16 전필드.
// ★★★배포 결정 = 해결(2026-08-05 실기): **순수토크 가능**(드라이버 Kp=Kd=0·fTorque 수용, 0.45Nm서 ~1%). → A 직결(fGainKp=fGainKd=0·fTorque=tau_ff).
//   ⚠2차 확인: 고토크·동적 토크추종(0.45Nm=저토크·마찰floor 근처).
// ★확정 SHM API(RobotTestGait): read=RobotMemGait_{IsUpdatedMotorStatus16,GetMotorStatus16,IsUpdatedIMU,GetIMU} · write=RobotMemGait_SetMotorCommand16(ch별).
//   IMU=float[LEN_OF_IMU_DATA=17]: [QUAT]=0(4) [ACCL]=4(3) [GYRO]=7(3) [ARPY]=10(3) [MAGN]=13(3) [TEMP]=16(1).
//
// ★2026-08-05 Pi 실기 확인 — 아래 3건은 **SDK 헤더(/usr/include/RobotSharedMem.h)로 확정**됐다(추측 아님):
//   ① gyro 인덱스 = IDX_OF_IMU_GYRO(=7). biped/emb/hal/shm_bridge.cpp 는 심볼명을 `IDX_OF_IMU_AVEL`로
//      찾다 실패해 **gyro를 0으로 채우고 있었다**(그 파일 L16-18·L77-80). 실 심볼은 GYRO다.
//   ② 자세는 **네이티브 쿼터니언** IDX_OF_IMU_QUAT(=0, 4개)이 따로 있다 → RPY→quat 변환(오일러 순서·
//      부호 convention 추측)이 애초에 불필요. RPY는 폴백으로만 둔다.
//   ③ MotGeneral_t·ENUM_RESULT_* 는 SDK 헤더에 **없다**(RobotTestGait/inc 사설 정의).
//      MotGeneral_t 는 SDK의 MotorParam16_t 와 필드가 완전히 동일 → MotorParam16_t 로 직접 쓴다
//      (사설 헤더 의존 제거). ENUM_RESULT_SUCCESS(=0x0)만 아래에 로컬 상수로 둔다.
//   ⚠ 남은 실기 확정 대상: quat 성분순서(wxyz/xyzw)·gyro 단위(deg/s 가정)·accel 프레임/중력 포함 여부.
//      → `tools/imu_probe` 로 실측 후 확정. 확정 전에는 read()가 자세/각속도를 **거부**(안전).
#include "hal/robot_interface.hpp"
#include <vector>

#if defined(QC_HAVE_ROBOT_SHM) || __has_include("/usr/include/RobotSharedMem.h")
#include "/usr/include/RobotSharedMem.h"    // MotorParam16_t·float16·RobotMemGait_*·IDX_OF_IMU_*·LEN_OF_IMU_DATA
#include <cmath>
#include <algorithm>
#include <cstring>
#include <ctime>

namespace qc {

// SDK 헤더에 없는 반환코드(RobotTestGait/inc/define/defineGeneral.h 정의를 그대로 인용).
//   ENUM_RESULT_SUCCESS=0x00000000 · RESUMED=0x10000000 · FAILURE=0x20000000.
static constexpr unsigned int QC_SHM_OK = 0x00000000u;

// 관절맵/한계 — config(config/joint_map_16dof.hpp). chan=Gait SHM 채널, sign/zero=MJCF관절↔실모터 부호·오프셋(★축별 JOG 실측 TODO).
struct GaitJointCfg { int chan; int sign; double zero_deg, min_deg, max_deg, vel_max_dps; };

class RealHal : public RobotInterface {
  int nu_; double dt_;
  std::vector<GaitJointCfg> jc_;                 // nu_개 (config)
  std::vector<MotorParam16_t> cmd_;              // 채널별 명령 버퍼(=RobotTestGait MotGeneral_t와 동일 레이아웃)
  bool tx_enabled_ = false;                      // ★안전: 상태 수신 확인 전엔 명령 전송 금지(robot_main이 enable)
  bool imu_ok_ = false;                          // IMU convention 실측 확정 여부(미확정=자세/각속도 미갱신)
  static constexpr double D2R = M_PI / 180.0, R2D = 180.0 / M_PI;

  // RPY(rad) → quat wxyz (ZYX). ★네이티브 quat(IDX_OF_IMU_QUAT)이 있으면 그걸 쓰고 이건 폴백.
  static void rpy2quat(double r, double p, double y, Vector4d& q) {
    double cr=std::cos(r*0.5), sr=std::sin(r*0.5), cp=std::cos(p*0.5), sp=std::sin(p*0.5), cy=std::cos(y*0.5), sy=std::sin(y*0.5);
    q[0]=cr*cp*cy+sr*sp*sy; q[1]=sr*cp*cy-cr*sp*sy; q[2]=cr*sp*cy+sr*cp*sy; q[3]=cr*cp*sy-sr*sp*cy;
  }
 public:
  RealHal(int nu, double dt, std::vector<GaitJointCfg> jc)
      : nu_(nu), dt_(dt), jc_(std::move(jc)), cmd_(nu) {}
  int    nu() const override { return nu_; }
  double dt() const override { return dt_; }
  // robot_main: 상태 N프레임 수신 후 true(=RobotTestGait 안전패턴).
  // ★off 는 "전송 중단"이 아니라 **명시적 limp 전송**이다 — 전송만 끊으면 드라이버는
  //   마지막 토크 명령을 계속 물고 있는다(통신두절 시 다리가 뻗은 채 굳는다).
  //   biped/emb/hal/shm_bridge.cpp:write_mit_impl 이 검증한 패턴(kp=kd=tau=0)을 따른다.
  void enable(bool on) {
    const bool was = tx_enabled_;
    tx_enabled_ = on;
    if (!on && was) send_limp();               // 무장 해제 순간 1회 강제 limp
  }

  // 전 채널 kp=kd=tau=0(=자유) 전송. 통신두절·E-stop·종료 경로에서 호출.
  void send_limp() {
    MotorParam16_t m;
    for (int i = 0; i < nu_; ++i) {
      if (jc_[i].chan < 0) continue;                              // 미배선 축은 건너뜀
      std::memset(&m, 0, sizeof(m));
      m.ucDevID = (unsigned char)jc_[i].chan; m.ucMode = 1; m.ucCommand = 0;
      m.fAccelrationOrTemperture = (float16)1.0f;                 // 나머지 필드(위치/속도/토크/게인)=0
      RobotMemGait_SetMotorCommand16(&m, jc_[i].chan);            // ★tx_enabled_ 무관 — 안전정지는 항상 나가야 한다
    }
  }

  // IMU convention(quat 성분순서·gyro 단위·accel 프레임)이 실측 확정되면 robot_main이 켠다.
  //   ★기본 off — 미확정 자세를 KF에 먹이면 조용히 틀린 추정으로 넘어진다. 명시적 opt-in.
  //   ※2026-08-05 현재 IMU는 **펌웨어단 미배선**(SHM 17필드 전부 0) → 켤 수 없다.
  void trust_imu(bool on) { imu_ok_ = on; }
  bool imu_trusted() const { return imu_ok_; }

  // 전 축이 실채널에 매핑됐는가. false = 미배선 축 존재 → 모델기반 제어(stand/walk) 금지.
  //   ★컨트롤러는 17축 전체의 q/dq 로 M·h·Jacobian 을 계산한다. 미배선 축을 0으로 채우면
  //     계산은 돌아가지만 그 결과는 **실제 로봇이 아닌 다른 로봇의 토크**다. 캡으로 숨기지 않고 거부한다.
  bool fully_mapped() const {
    for (const auto& j : jc_) if (j.chan < 0) return false;
    return true;
  }
  int n_mapped() const {
    int n = 0; for (const auto& j : jc_) if (j.chan >= 0) ++n; return n;
  }

  // ★SHM 연결(RobotMemGait_InitComm) + RobotEmbedded 기동 핸드셰이크.
  //   biped/emb/hal/shm_bridge.cpp:bridge_init 과 동일 절차(실기 검증된 경로).
  //   반환 false = SHM 미연결 또는 임베디드 모터컨트롤러 미기동 → 호출부가 구동 금지.
  bool init(int recv_wait_ms = 2000) {
    if (RobotMemGait_InitComm() != QC_SHM_OK) return false;
    for (int waited = 0; waited < recv_wait_ms; waited += 5) {
      if (RobotMemGait_IsUpdatedMotorStatus16() == 1) return true;
      struct timespec ts{0, 5 * 1000 * 1000L}; nanosleep(&ts, nullptr);
    }
    return false;
  }

  bool read(LowState& s) override {
    if (RobotMemGait_IsUpdatedMotorStatus16() != 1) return false;   // ★fresh 상태 없음 → 안전정지(호출부가 hold/estop)
    s.q.resize(nu_); s.dq.resize(nu_); s.tau_est.resize(nu_);
    MotorParam16_t st;
    for (int i = 0; i < nu_; ++i) {
      const auto& j = jc_[i];
      if (j.chan < 0) {                              // ★미배선 축: 읽을 게 없다. 0으로 두되
        s.q[i] = s.dq[i] = s.tau_est[i] = 0.0;       //   이 값을 컨트롤러가 쓰면 안 된다
        continue;                                    //   → fully_mapped() 게이트가 막는다
      }
      if (RobotMemGait_GetMotorStatus16(&st, j.chan) != QC_SHM_OK) return false;
      s.q[i]       = j.sign * ((double)st.fPosition - j.zero_deg) * D2R;   // 관절축 deg → MJCF rad(부호·오프셋)
      s.dq[i]      = j.sign *  (double)st.fVelocity * D2R;
      s.tau_est[i] = j.sign *  (double)st.fTorque;                          // ★PACE: 드라이버 산출토크(독립 힘측정 아님)
    }
    // ── IMU (몸통=ForeC 블록) ──────────────────────────────────────────────
    //   인덱스는 SDK 헤더로 확정: QUAT=0(4)·ACCL=4(3)·GYRO=7(3)·ARPY=10(3).
    //   ★단, 성분순서/단위/프레임은 **아직 실측 미확정** → imu_ok_ 전에는 채우지 않는다.
    //     (미확정 자세를 KF에 넣으면 조용히 틀린 base 추정 → 넘어짐. 0으로 두면 컨트롤러가
    //      "자세 평평·각속도 0"으로 보므로 그것도 틀리다. 그래서 stand 이전 단계에서만 허용하고,
    //      robot_main 이 trust_imu(true) 를 명시적으로 켠 뒤에야 보행을 허가한다.)
    if (imu_ok_ && RobotMemGait_IsUpdatedIMU() == 1) {
      float imu[LEN_OF_IMU_DATA] = {0};
      if (RobotMemGait_GetIMU(&imu[0], IDX_OF_IMU_ForeC_START, LEN_OF_IMU_DATA) == QC_SHM_OK) {
        // 자세: 네이티브 quat 우선(노름≈1 이면 유효), 아니면 RPY(deg) 폴백.
        const double qn = std::sqrt(
            (double)imu[IDX_OF_IMU_QUAT+0]*imu[IDX_OF_IMU_QUAT+0] + (double)imu[IDX_OF_IMU_QUAT+1]*imu[IDX_OF_IMU_QUAT+1] +
            (double)imu[IDX_OF_IMU_QUAT+2]*imu[IDX_OF_IMU_QUAT+2] + (double)imu[IDX_OF_IMU_QUAT+3]*imu[IDX_OF_IMU_QUAT+3]);
        if (std::fabs(qn - 1.0) < 0.05) {
          for (int a = 0; a < 4; ++a) s.imu_quat[a] = imu[IDX_OF_IMU_QUAT+a] / qn;   // ★성분순서 wxyz 가정(imu_probe 확정)
        } else {
          rpy2quat(imu[IDX_OF_IMU_ARPY+0]*D2R, imu[IDX_OF_IMU_ARPY+1]*D2R, imu[IDX_OF_IMU_ARPY+2]*D2R, s.imu_quat);
        }
        for (int a = 0; a < 3; ++a) s.imu_gyro[a] = imu[IDX_OF_IMU_GYRO+a] * D2R;    // ★deg/s 가정(RPY가 deg이므로)
        for (int a = 0; a < 3; ++a) s.imu_acc[a]  = imu[IDX_OF_IMU_ACCL+a];          // ★프레임/중력=imu_probe 확정 후 변환
      }
    }
    // ★foot_force: 이 로봇엔 발 힘센서가 없다. 0으로 두면 KF가 "항상 비접촉"으로 보고
    //   base 위치/속도를 앵커링하지 못한다(=추정 발산). 관절토크+기구학 접촉추정이 필요.
    //   → stand 단계 전 필수. 미구현 상태를 캡으로 숨기지 않고 여기 명시한다.
    // TODO(Pi): tau_est + Jacobian 으로 발 수직력 추정 → 임계 >0.5 지시자.
    return true;
  }

  bool write(const LowCmd& c) override {
    for (int i = 0; i < nu_; ++i) {
      const auto& j = jc_[i];
      if (j.chan < 0) continue;                                                // ★미배선 축은 절대 명령하지 않는다
      double q_des_deg = j.zero_deg + j.sign * c.q_des[i] * R2D;               // MJCF rad → 관절축 deg
      q_des_deg = std::min(j.max_deg, std::max(j.min_deg, q_des_deg));         // ★한계 clip(안전)
      MotorParam16_t& m = cmd_[i];                                            // PACE 확정 필드 매핑(순수토크: Kp=Kd=0)
      m.ucDevID  = (unsigned char)j.chan; m.ucMode = 1; m.ucCommand = 0;
      m.fPosition = (float16) q_des_deg;                                       // 관절축 deg
      m.fVelocity = (float16)( j.sign * c.dq_des[i] * R2D );                   // deg/s
      m.fTorque   = (float16)( j.sign * c.tau_ff[i] );                         // 관절 토크(WBIC) — 순수토크 배포경로
      m.fGainKp   = (float16) c.kp[i];   // ★A=0(순수토크). 임피던스 시 driver단위 환산(biped 1≈1.0027 Nm/rad, quad TBD)
      m.fGainKd   = (float16) c.kd[i];   // ★A=0
      m.fGainKi   = (float16) 0.0f;
      m.fAccelrationOrTemperture = (float16) 1.0f;                             // PACE: 미사용 상수 1.0
      if (tx_enabled_) RobotMemGait_SetMotorCommand16(&m, j.chan);              // ★안전게이트 후 전송
    }
    return true;
  }
};

}  // namespace qc
#endif   // RobotSharedMem.h 존재(Pi)
