/* hal_smoke.cpp — RealHal 순수토크 경로 end-to-end 실기 검증 (단일 채널, 벤치 모터).
 *
 *   ★왜 필요한가: 순수토크(Kp=Kd=0·fTorque) 가능하다는 건 확정됐지만, 그건 biped/emb 의
 *     **Python+C 브리지**로 검증한 것이다. 실제 배포 경로인 quad_ctrl 의
 *     `RealHal::write()` 는 아직 한 번도 실모터에 닿은 적이 없다.
 *     이 도구는 재구현하지 않고 **RealHal 클래스 그대로** 를 써서 그 경로를 검증한다.
 *
 *   측정 항목
 *     ① limp 기저 노이즈 (위치/속도/토크) — 정지 상태 산포
 *     ② 토크 추종 정확도 — 명령 tau_ff vs 드라이버 보고 tau  (MIGRATION.md "2차 확인" 항목)
 *     ③ 왕복지연 — 명령 발행 → 보고토크 반응까지 (예산 12ms, PACE 실측 8.39ms 와 비교)
 *
 *   ★★안전 설계 (다리 미장착 벤치 모터 전제)
 *     · 무부하 모터는 토크를 계속 주면 가속만 한다 → **짧은 펄스**(기본 400ms) + 사이 limp.
 *     · **속도 가드**: |dq| > VEL_ABORT 이면 즉시 토크 0 + 중단.
 *     · 토크 하드캡: PACE 실측 마찰 floor(JFRIC=0.38Nm) 근처인 0.45Nm 를 상한으로 강제.
 *     · SIGINT·정상종료·이상종료 모두 limp 로 끝난다.
 *     · 단일 채널만 건드린다(기본 ch0). 나머지 채널엔 명령을 보내지 않는다.
 *
 *   빌드: cmake --build build --target hal_smoke      (MuJoCo 불필요, Eigen+RobotSharedMem 만)
 *   실행: ./build/hal_smoke [chan] [tau_Nm] [pulse_ms]
 *         예) ./build/hal_smoke 0 0.30 400
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <csignal>
#include <ctime>
#include <vector>
#include <algorithm>

#include "hal/real_hal.hpp"
#include "config/joint_map_16dof.hpp"
#include "common/rt.hpp"

#if !(defined(QC_HAVE_ROBOT_SHM) || __has_include("/usr/include/RobotSharedMem.h"))
int main(){ std::printf("[hal_smoke] Pi 전용(RobotSharedMem.h 없음)\n"); return 0; }
#else
using namespace qc;

static constexpr double TAU_CAP    = 0.45;   // [Nm] 하드캡 — PACE 검증범위(마찰 floor 근처) 밖으로 안 나간다
static constexpr double VEL_ABORT  = 60.0;   // [deg/s] 이 속도 넘으면 즉시 중단(무부하 폭주 방지)
static constexpr int    ARM_FRAMES = 100;    // 상태 N프레임 수신 후 무장(RobotTestGait 패턴)

static volatile sig_atomic_t g_run = 1;
static void on_sigint(int){ g_run = 0; }

static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

struct Stat {
  int n=0; double sum=0, sum2=0, mn=1e18, mx=-1e18;
  void add(double v){ n++; sum+=v; sum2+=v*v; mn=std::min(mn,v); mx=std::max(mx,v); }
  double mean() const { return n? sum/n : 0; }
  double sd()   const { return n>1? std::sqrt(std::max(0.0, sum2/n - mean()*mean())) : 0; }
};

int main(int argc, char** argv){
  const int    chan     = argc>1? atoi(argv[1]) : 0;
  double       tau_cmd  = argc>2? atof(argv[2]) : 0.30;
  const int    pulse_ms = argc>3? atoi(argv[3]) : 400;

  if (std::fabs(tau_cmd) > TAU_CAP){
    std::printf("[hal_smoke] 토크 %.3f Nm → 하드캡 %.2f Nm 로 제한\n", tau_cmd, TAU_CAP);
    tau_cmd = tau_cmd>0? TAU_CAP : -TAU_CAP;
  }

  // 검증 대상 축의 실제 config 를 관절맵에서 가져온다(부호·한계 규약 동일 적용).
  const auto full = joint_map_16dof();
  GaitJointCfg jc{ chan, +1, 0.0, -180, 180, 300 };
  for (const auto& j : full) if (j.chan == chan) { jc = j; break; }
  if (jc.chan < 0){ std::printf("[hal_smoke] ch%d 는 미배선(chan=-1) — 중단\n", chan); return 1; }

  std::printf("[hal_smoke] ch%d · tau=%+.3f Nm · 펄스 %dms · 속도가드 %.0f deg/s\n",
              chan, tau_cmd, pulse_ms, VEL_ABORT);
  std::printf("            sign=%+d zero=%.2f deg 한계=[%.0f, %.0f] deg\n",
              jc.sign, jc.zero_deg, jc.min_deg, jc.max_deg);

  // ★RT A/B: RT=1 이면 SCHED_FIFO+mlockall 시도. 미지정=일반 우선순위(기존 측정과 동일 조건).
  //   같은 도구·같은 절차에서 이 스위치만 바꿔 비교 → 지연차의 원인을 스케줄링으로 귀속할 수 있다.
  if (getenv("RT") && atoi(getenv("RT")) != 0)
    rt_report(rt_setup(getenv("RT_PRIO") ? atoi(getenv("RT_PRIO")) : 80));
  else
    std::printf("[rt] 일반 우선순위(RT=1 로 SCHED_FIFO 시도)\n");

  RealHal hal(1, 0.001, { jc });                 // nu=1 — 이 채널 하나만
  if (!hal.init()){ std::printf("[hal_smoke] ✗ SHM 연결/핸드셰이크 실패 — RobotEmbedded 확인\n"); return 1; }
  std::printf("[hal_smoke] SHM 연결 OK\n");

  std::signal(SIGINT, on_sigint);

  LowState ls; LowCmd cmd;
  cmd.q_des.setZero(1); cmd.dq_des.setZero(1);
  cmd.kp.setZero(1);    cmd.kd.setZero(1);       // ★순수토크: Kp=Kd=0
  cmd.tau_ff.setZero(1);

  // ── 무장: 상태 프레임 누적 ────────────────────────────────────────────────
  int frames = 0;
  for (int k=0; k<3000 && g_run && frames<ARM_FRAMES; ++k){
    if (hal.read(ls)) frames++;
    struct timespec ts{0, 1000*1000L}; nanosleep(&ts,nullptr);
  }
  if (frames < ARM_FRAMES){ std::printf("[hal_smoke] ✗ 상태 %d/%d 프레임 — 체인 미생존. 중단\n", frames, ARM_FRAMES); return 1; }
  std::printf("[hal_smoke] 상태 %d프레임 수신 → 무장\n\n", frames);

  Stat jit;                                        // ★루프 실주기[ms] — 스케줄링 지터 관측
  auto hold = [&](double tau, int ms, Stat* sq, Stat* sdq, Stat* stau, double* t_react)->bool {
    const double t0 = now_s();
    double tprev = t0;
    bool reacted = false;
    cmd.tau_ff[0] = tau;
    while (g_run && (now_s()-t0)*1000.0 < ms){
      { const double tn = now_s(); jit.add((tn - tprev)*1000.0); tprev = tn; }
      if (hal.read(ls)){
        const double dq_dps = ls.dq[0] * 180.0 / M_PI;
        if (std::fabs(dq_dps) > VEL_ABORT){            // ★속도 가드
          cmd.tau_ff[0] = 0.0; hal.write(cmd); hal.enable(false);
          std::printf("\n[hal_smoke] ✗ 속도 가드 발동 (%.1f deg/s > %.0f) — 즉시 limp·중단\n", dq_dps, VEL_ABORT);
          return false;
        }
        if (sq){ sq->add(ls.q[0]*180.0/M_PI); sdq->add(dq_dps); stau->add(ls.tau_est[0]); }
        if (t_react && !reacted && std::fabs(ls.tau_est[0]) > std::fabs(tau)*0.5 && tau != 0.0){
          *t_react = (now_s()-t0)*1000.0; reacted = true;
        }
      }
      hal.write(cmd);
      struct timespec ts{0, 1000*1000L}; nanosleep(&ts,nullptr);
    }
    return g_run;
  };

  // ── ① limp 기저(무장 전 토크 0) ───────────────────────────────────────────
  Stat bq, bdq, btau;
  std::printf("① limp 기저 측정(토크 0, 800ms)…\n");
  if (!hold(0.0, 800, &bq, &bdq, &btau, nullptr)){ hal.enable(false); return 1; }
  std::printf("   위치 %.3f±%.3f deg · 속도 %.2f±%.2f deg/s · 토크 %.4f±%.4f Nm\n\n",
              bq.mean(), bq.sd(), bdq.mean(), bdq.sd(), btau.mean(), btau.sd());

  hal.enable(true);                                  // ★여기서부터 실제 명령이 나간다
  std::printf("② 순수토크 펄스 (Kp=Kd=0, tau_ff 만)\n");

  struct Step { const char* name; double tau; };
  const Step steps[] = { {"+tau", +tau_cmd}, {"limp", 0.0}, {"-tau", -tau_cmd}, {"limp", 0.0} };
  bool okrun = true;
  for (const auto& s : steps){
    Stat q,dq,tq; double react = -1;
    std::printf("   %-5s 명령 %+.3f Nm … ", s.name, s.tau);
    std::fflush(stdout);
    if (!hold(s.tau, s.tau==0.0? 300 : pulse_ms, &q,&dq,&tq, &react)){ okrun=false; break; }
    std::printf("보고 %+.4f±%.4f Nm · 이동 %.2f deg · 속도 %.1f deg/s",
                tq.mean(), tq.sd(), q.mx-q.mn, dq.mx);
    if (s.tau != 0.0){
      const double err = tq.mean() - s.tau;
      std::printf(" · 오차 %+.4f Nm (%.1f%%)", err, 100.0*err/std::fabs(s.tau));
      if (react >= 0) std::printf(" · 반응 %.1f ms", react);
      else            std::printf(" · 반응 미검출");
    }
    std::printf("\n");
  }

  cmd.tau_ff[0] = 0.0; hal.write(cmd);
  hal.enable(false);                                  // ★종료 = 명시적 limp

  // ★루프 지터 — 목표 1ms 대비 실주기. 최대값이 크면 선점당한 것 = 지연의 스케줄링 성분.
  std::printf("\n③ 루프 주기: 평균 %.3f ms · sd %.3f · 최대 %.3f ms (목표 1.0, n=%d)\n",
              jit.mean(), jit.sd(), jit.mx, jit.n);
  std::printf("[hal_smoke] 종료 — limp 전송 완료 (%s)\n", okrun? "정상" : "중단됨");
  return okrun? 0 : 1;
}
#endif
