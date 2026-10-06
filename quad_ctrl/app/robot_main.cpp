// robot_main — 실기(Pi) 진입점: RealHal + Estimator + Controller 를 RT 루프로.
//   ★sim(sim_bridge)과 동일 구조 — HAL만 RealHal(실모터/센서)로 교체. 컨트롤러·estimator 불변(=이관 핵심).
//     루프: RealHal.read → EkfEstimator.update(→d_est) → TrotBridge.step(d_est) → RealHal.write(tau_ff, kp=kd=0 순수토크).
//   ※Pi 전용(RobotSharedMem). 데스크톱은 자동 skip(#else 스텁).
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <csignal>
#include "hal/real_hal.hpp"
#include "common/rt.hpp"

#if defined(QC_HAVE_ROBOT_SHM) || __has_include("/usr/include/RobotSharedMem.h")
#include "quad_control.hpp"                 // ::QuadControl (모델=M/h/Jac 계산기)
#include "trot_controller.hpp"              // apply_env_gains
#include "estimator/ekf_estimator.hpp"
#include "control/trot_bridge.hpp"
#include "command/sport_client.hpp"
#include "common/config.hpp"
#include "config/joint_map_16dof.hpp"

using namespace qc;
static volatile sig_atomic_t g_run = 1;
static void on_sigint(int) { g_run = 0; }

int main(int argc, char** argv) {
  load_config_env();                          // QC_CONFIG(게인·물리·관절범위)
  // 기본 경로는 이 저장소 기준 상대(Pi 체크아웃 위치 무관). argv[1]로 덮어쓸 수 있다.
  const char* mjcf = argc > 1 ? argv[1]
      : "../quad/mjcf/quad_real_16dof_sphere.mjcf";

  // ★모델(동역학 계산기) — sim과 동일 셋업. 실기서도 MuJoCo를 M/h/Jac 계산에 씀(물리엔진 아님, mj_step 안함).
  ::QuadControl q;
  q.load(mjcf); apply_env_gains(q); q.crouch_home(); q.build_qhome_lut(); q.setup_mpc();

  EkfEstimator ekf(q);                        // KF 추정 → d_est
  TrotBridge   ctrl(q);                        // 검증된 MPC+WBIC(순수토크)
  // ★2026-10-06 관절맵 길이 = 모델 nu 검사. RealHal 은 i<nu 로 맵을 인덱싱하므로 어긋나면(예: 17-DOF 모델+16 맵)
  //   축이 밀려 다른 채널로 명령이 나간다 → 구동 전에 중단.
  const auto jmap = joint_map_16dof();
  if ((int)jmap.size() != q.nu) {
    std::fprintf(stderr, "[robot_main] 관절맵 %zu축 ≠ 모델 nu=%d (%s) — 모델/맵 불일치, 구동 중단.\n", jmap.size(), q.nu, mjcf);
    return 1;
  }
  RealHal      hal(q.nu, q.m->opt.timestep, jmap);   // ★관절맵=축별 JOG 실측 확정 필요

  // ★SHM 연결 + RobotEmbedded 기동 핸드셰이크. 이게 없으면 SHM이 아예 붙지 않는다(스캐폴드 누락분).
  if (!hal.init()) {
    std::fprintf(stderr, "[robot_main] SHM 연결 실패 — RobotEmbedded(모터 컨트롤러) 기동 확인 필요. 구동 중단.\n");
    return 1;
  }
  std::printf("[robot_main] SHM 연결 OK (nu=%d, dt=%.4fs, 매핑축 %d/%d)\n",
              q.nu, q.m->opt.timestep, hal.n_mapped(), q.nu);

  // ★★배포 게이트 — 모델기반 제어(stand/walk)의 전제조건. 하나라도 못 지키면 구동하지 않는다.
  //   여기서 막지 않으면 컨트롤러는 "평평한 자세·정지·17축 전부 정상"이라는 거짓 상태를 믿고
  //   토크를 뿜는다. sim 에서 falls=0 이었던 건 그 전제가 참일 때의 이야기다.
  //   (2026-08-05 Pi 실측 기준 셋 다 미충족 — config/joint_map_16dof.hpp 주석 참조)
  bool ok = true;
  if (!hal.fully_mapped()) {
    std::fprintf(stderr, "[robot_main] ✗ 미배선 축 존재(%d/%d 매핑) — 모델기반 제어 불가.\n"
                         "    컨트롤러는 17축 전체 q/dq 로 M·h·Jacobian 을 계산한다. 빠진 축을 0으로\n"
                         "    채우면 계산은 되지만 그 토크는 이 로봇의 것이 아니다.\n", hal.n_mapped(), q.nu);
    ok = false;
  }
  if (!hal.imu_trusted()) {
    std::fprintf(stderr, "[robot_main] ✗ IMU 미확정/미배선 — base 자세·각속도 없음 = KF 추정 불가.\n"
                         "    2026-08-05 실측: SHM IMU 17필드 전부 0(펌웨어 미배선).\n");
    ok = false;
  }
  if (!ok) {
    std::fprintf(stderr, "[robot_main] 구동 거부. 축별 JOG(tools/) 로 관절맵 실측부터 진행할 것.\n");
    hal.send_limp();
    return 2;
  }

  HighCmd hc; hc.mode = Mode::Stand;           // TODO(nav/teleop): HighCmd(cmd_vel)를 SportClient 경계서 수신
  ctrl.set_command(hc);

  std::signal(SIGINT, on_sigint);
  // ★RT 우선순위 — 일반 우선순위면 RobotEmbedded(CPU ~90%)에 선점당해 주기가 밀린다.
  //   rtprio ulimit=0 인 기기에서는 sudo 로 실행해야 걸린다. 실패해도 진행하되 사실은 보고한다.
  rt_report(rt_setup(getenv("RT_PRIO") ? atoi(getenv("RT_PRIO")) : 80));
  const long period_ns = (long)(q.m->opt.timestep * 1e9);       // 제어주기(=모델 timestep, 통상 1ms)
  struct timespec next; clock_gettime(CLOCK_MONOTONIC, &next);

  LowState ls; LowCmd cmd; long status_cnt = 0, miss = 0;
  while (g_run) {
    if (hal.read(ls)) {                        // 센서(신선한 상태)
      if (++status_cnt > 100) hal.enable(true);// ★안전: 상태 100프레임 수신 후 명령 전송 허용(RobotTestGait 패턴)
      ekf.update(ls);                          // 추정 → d_est
      ctrl.step(cmd, ekf.d_est());             // 제어 → LowCmd(순수토크 tau_ff, kp=kd=0)
      hal.write(cmd);                          // 명령(전송 enable 시)
      miss = 0;
    } else if (++miss > 10) {
      hal.enable(false);                       // ★통신 두절 → limp(안전정지)
      hal.send_limp();                         //   두절이 지속되는 동안 매 주기 재전송(1회로는 놓칠 수 있다)
    }
    next.tv_nsec += period_ns;                 // 다음 주기 절대시각
    while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; next.tv_sec++; }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
  }
  hal.enable(false);                           // 종료 시 limp
  std::printf("[robot_main] 종료 (status_cnt=%ld)\n", status_cnt);
  return 0;
}
#else
int main() { std::printf("[robot_main] Pi 전용(RobotSharedMem.h 없음) — 데스크톱 빌드 skip\n"); return 0; }
#endif
