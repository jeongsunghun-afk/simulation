// biped 전체 컨트롤러 (C++) — Python biped_mpc_wbic+biped_step+biped_wbic 통합 이식.
#include <cstdlib>
// MuJoCo C API로 M·h·jac·com 계산 → event-DCM 게이트 + base-frame 발배치 + MPC(50Hz) + WBIC.
#pragma once
#include <mujoco/mujoco.h>
#include "biped_mpc.hpp"
#include "biped_wbic.hpp"
#include "biped_zmp.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <cstdio>
using namespace Eigen;

struct BipedControl {
  mjModel* m; mjData* d;
  int nv, nu, NF=2;
  int sph[2], fbody[2];               // 발 sphere geom(tip)·contact body
  int sph2[2]={-1,-1}; bool has_heel=false;   // ★평발 heel 접촉구(foot_link 원점)
  int cmode=0;                        // 접촉모드: 0=1점(점발 stepping)·1=2점(평발 정적). 통합모델 기본 평발.
  // ★평발 home(발목 눕힘, CoM 밑창중심). **2026-08-13 새 CAD 로 재산출** — 구값
  //   {0.25, −0.50, −1.14626} 은 구 CAD 유래이고, 2026-08-12 Qhome8 재산출(커밋 37a517e)
  //   때 **여기가 빠졌다**(그 검증이 점발 8조건뿐이었다). 결과: 새 CAD 에서 CoM 이 밑창중심보다
  //   **4.5cm 앞**(밑창 반길이 7.3cm → 전방여유 37.9%) → 전방 폭주로 1.29s 낙상.
  //   ⚠밑창 기울기는 구값도 0 이었다 — 눈으로는 멀쩡해 보인다. 깨진 건 **전후 정렬**뿐이다.
  //   재산출: cpp/src/flat_home.cpp (밑창수평 · CoM=밑창중심 · CoM높이 유지 3조건 Newton).
  //   결과 CoM−밑창중심 0.00000 m(여유 100%) · CoM z 0.3649 유지 · base z 0.4451.
  double Qflat8[8]={0,0.064256,-0.416657,-1.043858, 0,0.064256,-0.416657,-1.043858};
  // ── 파라미터 (Python 동일) ──
  // ★T_STEP 0.24 → 0.32 (2026-08-05). 실측 ROTOR_I(7.4e-4, 구 placeholder 의 7.4배)를
  //   넣으면 반사관성이 7.4배가 되어 0.24s 스텝의 스윙 가속에 필요한 토크가 드라이브 한계(drv_peak)를
  //   넘어 QP 가 포화 → 2.18s 낙상. 필요가속도 ∝ 1/T² 이므로 스텝을 늦추는 것이 해법이다.
  //   ⚠ 스윙게인을 올리는 것은 역효과였다(SW_KP 800→1600/3200/5920 = 1.16/1.10/0.65s 낙상)
  //     — 대역 부족이 아니라 토크 포화이기 때문. 실측 스윕:
  //       ROTOR_I 1e-4/2e-4/4e-4/5e-4 = 15s 무낙상 · 6e-4 = 9.4s · 7.4e-4 = 2.18s 낙상
  //       7.4e-4 + T_STEP 0.32 = 15s 무낙상 tilt 2.7°(전 설정 중 최량) · 0.40/0.50 = 낙상
  //   ⚠ vx=0.15 단일 조건의 4점 스윕으로 잡은 값이다. 속도대역 전반 재검증 필요.
  // ★2026-08-05 재튜닝: T_STEP 0.32→0.38, K_RETURN 0.45→0.15.
  //   leg-odom 야코비안 편향(구중심 vs 접촉점)을 제거하자 기존 튜닝이 성립하지 않았다 —
  //   기존 값은 그 편향을 전제로 맞춰져 있었다. 실측 센서노이즈까지 넣고 재스윕한 결과다.
  //   상세: cpp/STABILITY_MAP.md
  // ★2026-08-12 재스윕 0.38 → **0.34** (새 CAD 몸통). 위 "0.32→0.38"은 **구 CAD 기준**이다.
  //   실측 몸통(2.8kg, CoM x=−0.0727)·고관절 26cm 이동 후 0.38 은 후진이 6.5초에 깨진다.
  //   Q_HOME 재산출 후 15s 스윕(biped_from_quad 점발):
  //       0.38  vx −0.10 = 6.54s 낙상 · 0.20 = 2.13s 낙상 (전진 0~0.15 만 생존)
  //       0.42  vx −0.10 = 5.45s · 0.15 = 2.56s 낙상
  //       0.34  **−0.15/−0.05/0/0.05/0.10/0.15/0.20 전부 15s 무낙상, tilt 0.2~1.9°**
  //   ⚠Q_HOME 을 먼저 고쳐야 이 값이 산다(아래 Qhome8 주석). 순서를 바꾸면 안 된다.
  // ★2026-08-14 재스윕 0.34 → **0.30** (PACE 최종 파라미터 반영 후).
  //   0.34 는 **구 파라미터(JFRIC 0.38 스칼라)로 잡은 값**이다. 마찰을 kind 별 실측값으로
  //   바꾸자(1.3~2.2배) 그 지점이 안정영역 **밖으로** 밀려났다.
  //   재스윕 7속도(−0.15~0.20) × 8스텝 = 56조건, 15s:
  //       T      0.24 0.26 0.28 0.30 0.32 | 0.34 0.36 0.38 0.40 0.42
  //       통과   7/7  7/7  7/7  7/7  7/7  | 5/7  6/7  5/7  4/7  0/7
  //   ⇒ 안정영역 = **[≤0.24, 0.32]**. 상단을 가르는 건 **후진(vx −0.15)** 이다
  //     (0.34 부터 6.8s 에 깨진다 — 구 파라미터 때도 후진이 먼저 깨졌던 것과 같은 양상).
  //   0.30 을 고른 이유: 평균 tilt 가 가장 낮고(0.28→1.31° · **0.30→1.00°** · 0.32→1.27°)
  //     실패 경계(0.34)에서 두 칸 떨어져 있다. 더 낮추면 스윙 가속토크가 커지는데
  //     실기는 α 불확실성이 있어 토크 여유를 남기는 편이 낫다.
  double T_STEP=0.30, DS_FRAC=0.10, STEP_H=0.06, K_CAP=1.0, CAP_CLAMP=0.22;
  double SW_KP=800, SW_KD=60, K_RETURN=0.15, K_RET_LAT=0.0, K_LAT=0.5, SPREAD=1.0, GAP_MIN=0.14, GAP_MAX=0.34;
  double SS_NOMINAL=0.16, SS_MIN=0.10, SS_MAX=0.45, TRIG_Y=0.03, GVEC=9.81;
  double FLAT_KCAP=0.6;               // ★평발 전후 capture 게인(발목ZMP가 주 균형, 약한 보조)
  double FLAT_WANK=150;               // ★평발 보행 발목 flat 고정 가중(밑창 유지)
  double FLAT_WLAM=2;                 // ★평발 보행 MPC GRF 추종 가중(↓=WBIC 높이/CoM task 지배)
  double czwalk=0;                    // ★평발 보행 CoM 높이(0=reset값). 튜닝용
  double FLAT_WORI=5;                 // ★평발 보행 base pitch/roll 레벨링 가중
  double FLAT_WLEG=0.05;              // ★평발 정적 thigh/calf posture 가중(낮음=CoM 높이 조절 가능)
  double STANCE_KD=20, W_ORI=5, W_POST=1, W_ANKLE=20, MU_EFF=0.8*0.707, LAMZ_MIN=1;
  // ★★2026-08-21 **CoM 적분항 — 기본 꺼짐(STAND_KI=0).**
  //   왜 필요한가: WBIC 는 τ = h − Jᵀλ 인데 `h` 는 **모델의** 중력항이다. 모델이 실제보다
  //   가볍거나(질량 8% 확인됨) 토크 스케일 α<1 이면 힘이 모자라고, **되잡을 항이 없다**.
  //   지금 제어기는 CoM·자세 task 가 전부 PD 라 정상상태 오차가 **그대로 남는다**.
  //
  //   ⚠**기본을 꺼 두는 이유가 진단이다.** 지금 처짐은 모델 오차의 **유일한 관측창**이다 —
  //     α·질량·gear_k 중 무엇이 틀렸는지 그 처짐으로 가른다. 적분을 켜면 셋 다 조용히
  //     보상돼 증상이 사라지고, 원인을 영영 못 찾는다.
  //     ⇒ 원인 규명(저울·처짐 대조)이 끝난 **뒤에** 잔차 보상용으로 켤 것.
  //
  //   ⚠와인드업 방지가 필수다. 아래 세 경우에 **적분을 얼리거나 비운다**:
  //     ① QP 실패(중력보상 폴백) — 제어가 안 먹는데 쌓으면 복귀 순간 튄다
  //     ② 접촉 부족(K<4) — 발이 뜬 상태의 오차는 되잡을 대상이 아니다
  //     ③ 모드 진입 — 이전 세션의 적분을 물려받지 않는다
  //   ⚠적분 출력은 **가속도 단위**로 클램프한다(CoM task 가 가속도 공간이라).
  //   ★★★2026-08-24 **축 분리 — 지금은 z 만 켠다. IMU 가 살아나면 되돌릴 것.**
  //   ⚠왜: CoM 오차 `cerr` 를 쓰려면 **동체 자세**를 알아야 하는데, 이 로봇은 IMU 가 전부 0 이라
  //     biped_deploy 가 `d->qpos[3..6]` 에 **항등 쿼터니언을 상수로** 넣고 있다(:1054).
  //     ⇒ 모델은 "동체가 언제나 완벽히 수평" 이라고 믿는다. 그러면:
  //         z (높이)  ✅ 관측된다 — 다리가 접히면 엔코더가 보고, CoM 이 실제로 내려간다
  //         xy        ❌ **관측 안 된다** — 45° 기울어도 모델은 수평이라 cerr[xy] 가 근거 없는 값이다
  //     그걸 적분하면 **미해결인 "오른쪽으로 기움" 을 적분이 증폭**한다. 그래서 xy 를 잠근다.
  //   ★**되돌리는 조건**: emb/IMU_RECOVERY.md 대로 IMU 를 복구하고(Pi 앱 5줄),
  //     기동 배너가 "IMU 정상 — |acc|합 …(중력 감지)" 를 찍으면 `KI_AXIS` 를 {1,1,1} 로.
  //     그 전까지는 imu_ok=false 라 아래에서 **런타임으로도 강제 차단**된다(이중 안전).
  //     되돌릴 때 같이 볼 것: tilt E-stop 도 그때 처음 살아난다(지금 크레인이 유일한 안전장치).
  double dt_ctrl = 0.002;             // control(dt) 가 매 틱 갱신 — 적분에 쓴다
  double STAND_KI = 0.0;              // CoM 적분이득 [1/s³] — 0 = 꺼짐
  double STAND_I_CLAMP = 2.0;         // 적분 기여 상한 [m/s²] — 중력의 20% 수준
  Vector3d KI_AXIS{0,0,1};            // ★축별 on/off. 기본 z 만. IMU 복구 후 {1,1,1}
  bool     imu_ok = false;            // deploy 가 매 기동에 설정. false 면 xy 를 강제 차단
  Vector3d com_i = Vector3d::Zero();  // 적분 누적 [m·s]
  bool     ki_frozen = false;         // 직전 틱에 얼렸는지(상태 발행용)
  void reset_com_i(){ com_i.setZero(); ki_frozen=false; }
  // ★★stance 과제 게인 — env 조절 (2026-09-03, "stand-lite" 브링업용).
  //   왜: hold 자립 성공 vs stand 실패(8Hz 자려진동)의 갈림이 **지연 낀 위치의존
  //   피드백**(CoM kp120/200 · 레벨링 kp150, IMU 죽어 유령오차)으로 규명됐다.
  //   이 게인들을 0 으로 내리면 stand = "QP 가 발 힘을 분배하는 hold" 가 되어,
  //   hold(고정 50:50·toe) 와 **한 가지 차이만 남는 A/B** 가 성립한다.
  //   기본값 = 종전 하드코딩 그대로 (미지정 시 동작 불변).
  Vector3d COM_KP{120,120,200}, COM_KD{20,20,25};
  double ORI_KP=150, ORI_KD=20, POST_KP=60, POST_KD=5;
  double MPC_DT=0.02, W_LAM=10, head_lead=0.15;
  int MPC_N=14, mpc_decim=10;
  // ★2026-08-06: 하드코딩 폐기 → init() 이 MJCF 에서 읽는다. 감속비를 바꾸면 토크한계도
  //   따라가야 한다(종전 foot 96=12Nm×8 이 GEAR 8→8.4 를 안 따라가 11.43 이 돼 있었다).
  // ★2026-08-13 **개명 tau_peak8 → drv_peak8 + 출처 변경**. 이건 관절토크 한계가 아니라
  //   **드라이브(모터) 토크한계**다. 발목 액추에이터를 tendon 으로 옮기면서 둘이 갈라졌다:
  //     calf **관절**은 무릎·발목 두 드라이브를 합쳐 226.8 을 받지만(jnt_actfrcrange),
  //     무릎 **드라이브** 상한은 여전히 126 이다(actuator ctrlrange).
  //   ⇒ 출처를 jnt_actfrcrange → **actuator ctrlrange** 로 옮긴다. 이름을 같이 바꾸는 건
  //     의미가 달라졌기 때문이다 — 같은 이름으로 두면 다음 사람이 관절한계로 읽는다.
  double drv_peak8[8]={84,84,126,100.8,84,84,126,100.8};   // init() 이 MJCF 값으로 덮어씀
  // ★2026-08-12 새 CAD(몸통 placeholder→실측)로 재산출. 구값 (0.05,−0.2) 폐기.
  //   고관절 부착점이 26cm 이동해 구 자세는 HOME 에서 CoM 이 지지중심보다 6cm 앞(구 1.6cm)이었고,
  //   그 오차가 nominal_off 에 스폰 시점에 굳어 매 스텝 반복 → 전방 폭주로 1초 내 낙상했다.
  //   기준: nominal_off_x=+0.02 · 다리높이 0.4651(구와 동일). 상세는 biped_wbic.py Q_HOME 주석.
  //   검증: 15s × 8조건(정지·전진 0.05~0.20·후진·측방·선회) 8/8 무낙상, tilt 3.0~4.1°.
  //   ★T_STEP 은 배포값 0.38 그대로다 — 바뀐 것은 자세뿐이다.
  double Qhome8[8]={0,0.203054,-0.671148,0, 0,0.203054,-0.671148,0};
  int ankle_idx[2]={3,7};
  // ── 액추에이터 물리 — ★2026-08-05 실기 실측 (emb/pace/RESULTS.md) ──
  //   HL_hip·HR_hip 을 PACE 처프로 식별. 전 관절이 동일 모터+7:1 이고 관절별 추가
  //   감속단만 붙으므로 ROTOR_I(모터축 관성)는 **전 관절 공통 상수**다.
  //     ROTOR_I 7.652e-4(HL) / 7.121e-4(HR) → 7.4e-4 (양축 7% 일치).
  //             구 placeholder 1e-4 는 7.4배 과소였다.
  //     JDAMP   0.096~0.102(HL) / 0.071(HR) → 0.09. 등속스윕은 속도가 낮아 점성이
  //             신호에 안 잡히므로(HR 은 음수까지 나옴) **처프값**을 쓴다.
  //     JFRIC   처프 0.375(HL) / 0.382(HR) → 0.38. 저속 정지·유지는 0.50~0.52 인데
  //             Stribeck 때문이며, 보행은 동적 영역이라 처프값이 대표값이다.
  //   ⚠ 실측은 hip 2축·다리 미장착 상태. thigh/calf/foot 의 JDAMP/JFRIC 은 감속단이
  //     늘면 마찰도 늘어 달라진다(ROTOR_I 와 달리 공통 상수가 아님) → 장착 후 재측정.
  //   ⚠ GEAR foot 8 → 8.4 (총 감속비 8.4 = 7×1.2 추가단, 사용자 확인 2026-08-05).
  // ★★2026-08-14 **PACE 식별 최종값으로 전면 교체** (emb/pace/RESULTS.md §8).
  //   종전엔 `JDAMP 0.09 / JFRIC 0.38` **스칼라 하나를 8축 전부에** 썼다 — 근거가
  //   hip 2축·**다리 미장착** 실측이라 감속단이 다른 calf/foot 에 맞을 이유가 없었다.
  //   (그 사실이 바로 위 주석에 "장착 후 재측정" 으로 이미 적혀 있었다.)
  //   ROTOR_I 7.4e-4 → 7.327e-4 : foot τ_ff 7.327e-4 · calf 공통속도법 7.340e-4,
  //     두 축·두 방법이 **0.17%** 로 만났다(순환 없는 경로의 독립 검증).
  //   ⚠JFRIC 은 전 축이 종전보다 크다(0.66~2.2배) — **보행 거동이 바뀐다.**
  //   ⚠hip 의 JDAMP·JFRIC 은 **식별된 게 아니라 고정한 값**이다(자극이 비용의 4% 뿐).
  //   ⚠foot 의 dof_armature 는 0 이다(tendon 으로 이전).
  //   ⚠Python biped_wbic.py 와 **같은 값**이어야 한다. 한쪽만 고치면 파리티가 깨진다.
  //
  // ★2026-08-14 **fit_v2 → fit_v6** 로 갱신(emb/pace/RESULTS.md §1). 바뀐 것:
  //     JDAMP.calf  0.0092★ → **0** (확정)  · JDAMP.thigh 0.1696 → 0.022
  //     JFRIC.thigh 0.5064  → 0.592         · JFRIC.foot  0.2517★ → 0.241
  //   v6 은 **탐색범위 경고 0건** — 자유변수가 전부 범위 안에 앉은 첫 판이다.
  //   종전 ★표시(범위 끝에 붙은 값)가 둘 다 해소됐다:
  //     calf JDAMP 는 다섯 판 내내 바닥으로 밀렸다 ⇒ 0 으로 못박아도 성적이 안 나빠졌다
  //       (적합 0.4039 · 따로 뺀 구간 0.3933 — 다섯 판 중 **최고**). ⇒ 0 확정.
  //     foot JFRIC 은 자유였던 v5·v6 이 0.240/0.241 로 **±0.4%** 일치.
  //
  // ⚠★**thigh 의 두 값은 짝으로만 의미가 있다.** 다섯 판에서 b 가 0.022~0.180 으로
  //   **8.1배** 흔들리는 동안 `b·q̇ + τ_c` 는 ±4.9% 다(q̇=0.773 rad/s). 적합이 직선
  //   `b·q̇ + τ_c ≈ 0.64` 위를 미끄러질 뿐이다. 한쪽만 바꾸면 **총 손실이 깨진다** —
  //   도메인 랜덤화도 이 직선을 따라 해야 한다. 시험속도(q̇≈0.77) 밖에선 갈린다.
  double GEAR[4]={7,7,10.5,8.4}, ROTOR_I=7.327e-4;
  //                 hip     thigh    calf     foot
  // ★2026-08-14(2차) **최종 문서 반영** — fit_v6 → 새 수집 판(RESULTS.md §1).
  //     JDAMP.thigh 0.022 → **0** ★   JFRIC.thigh 0.592 → 0.603
  //     JDAMP.calf     0  →   0  ✓    JFRIC.calf  0.572 → **0.537**(−6%)
  //   ⚠`JFRIC.foot` 만 **옛 자료(fit_v6)의 0.241** 을 그대로 쓴다 — 새 수집은 foot 진폭을
  //     13°→7° 로 일부러 줄여(속도예산을 thigh 로 넘김) 그 축 정보가 적다. 문서 §1 ◆ 참조.
  //   ⚠**thigh·calf 의 마찰은 damping=0 과 짝이다.** 같은 판에서 와야 하므로 섞지 말 것.
  //
  // ★★`JDAMP` 가 0 으로 내려간 이유 — **지연과 같은 것을 본다**(RESULTS.md §1).
  //   `τ = kp(q_cmd(t−T_d) − q) − kd·q̇` 를 1차로 펴면 `−kp·T_d·q̇_cmd` 라는 감쇠항이 나온다.
  //   적합이 보는 건 `b + kp·T_d` 의 **합**뿐이라 b 단독은 식별 불가다.
  //     hip   kp·T_d 0.975 vs JDAMP 0.090 → **10.8배**
  //     thigh 0.488 / calf 0.548 → JDAMP 0 (∞)
  //   즉 진짜 감쇠는 지연이 만드는 감쇠의 10~30% 짜리 보정항이다. 지연 실측오차
  //   ±0.79ms 만으로 hip 은 ±0.079(그 축 감쇠의 ±88%)를 덮는다.
  //   ⇒ **지연을 실측 8.39ms 에 못박았기에** 이 b 값들이 의미를 갖는다. 지연을 자유로
  //     두면 b 는 아무 값이나 된다. LAT_COMP_MS 기본 8.4 가 그 실측값이다.
  //
  // ★★2026-08-19 **손실공간 변경 — foot 의 세 항 전부 tendon(모터축)** (RESULTS.md §1-b).
  //   종전엔 반사관성만 tendon 으로 옮기고 damping·frictionloss 는 관절에 남겼다.
  //   **물리 논거가 완전히 같은데** 오래 안 보였다 — 모터의 마찰·점성도 관절각이 아니라
  //   raw각(q_foot + q_calf)에서 작용한다. MuJoCo 로 직접 잰 반력:
  //       무릎만 회전 시   종전 dof: foot −0.037 · **calf 0** ← 틀림
  //                       tendon  : foot −0.109 · **calf −0.109** ✓
  //   무릎이 돌면 발목 모터 마찰이 calf 에도 반력을 줘야 하는데 0 이었다. 반대로 raw 가
  //   안 도는데 관절이 도는 경우엔 **없는 소산**을 넣고 있었다.
  //   ⚠각축(--solo) 측정에선 q̇_calf=0 이라 raw=관절각이어서 **차이가 안 난다.**
  //     그래서 여태 안 드러났다 — armature 와 똑같은 사연이다.
  //   통제실험(같은 코드, --loss-space 만 다름): 초기RMS −32% · 적합 −32.9% ·
  //     따로 뺀 구간 −32.2% · **게인 2배 검증 −25.5%** — 네 지표 전부 tendon 이 이긴다.
  //   ★진짜 근거는 **독립 실측과 맞아졌다**는 것이다(각축 등속스윕, PACE 와 무관한 시험):
  //       thigh −9.9% · calf −13.0% · foot −14.3%  ⇒ 산포 −10~−68% 가 −10~−14% 로 모였다.
  //     "JFRIC.calf 가 스윕 대비 −46%" 미해결이 이걸로 풀렸다 — 원인은 calf 가 아니라
  //     **foot 의 손실이 잘못된 좌표에 있던 것**이었다.
  //
  //   ⇒ 아래 배열의 **foot 칸은 관절이 아니라 tendon 으로 간다**(foot_rotor_to_tendon).
  double JDAMP[4]={0.0900, 0.0000, 0.0000, 0.1100};   // [Nm·s/rad] hip~calf=관절축 · foot=tendon
  double JFRIC[4]={0.8270, 0.6040, 0.8710, 0.6390};   // [Nm] 쿨롱마찰 (foot 0.639=tendon)
  // ── 상태 ──
  double vx_cmd=0, vy_cmd=0, wz_cmd=0, yaw_des=0, yaw_hold=0; bool yaw_hold_set=false;   // ★heading-hold latch
  Vector2d com0; Vector2d nominal_off[2]; double com_ref_z; Vector2d com_ref_xy;   // ★2점 정적 CoM xy 목표
  Vector4d foot_home_quat[2];         // ★평발 swing 발 수평 목표(home world quat, yaw=0)
  int stance=1, swing=0; double t_ss=0; long _k=0; bool walk_init=true; double walk_init_t=0;   // ★평발 보행개시 weight-shift
  // ── ZMP 프리뷰 보행(평발) ──
  ZmpPreview pv; long zkk=-1; double zanchor_x=0, zaf_y[2]={0,0}, z_sx=0;   // 발 앵커·스텝전진
  double cxr=0,vxr=0,cyr=0,vyr=0; int prev_ctr=0;                          // preview CoM ref
  double T_SS_Z=0.32; int PREV_DECIM=5; bool in_zmp_walk=false;            // 공칭 SS시간·preview 데시메이션
  long zlead=0;                                                            // ZMP 리드인 DS 잔여 틱
  // ── 일관 footstep-계획 ZMP 보행 (정통 휴머노이드: 고정 게이트 클럭이 발판·ZMP·CoM을 함께 구동) ──
  bool zw_started=false; double zw_t=0; long zw_i=0; int zw_sup=1;
  double ZW_TSS=0.34, ZW_TDS=0.08, ZW_LEAD=0.35, ZW_STEPH=0.05;
  double zw_x0=0, zw_yfoot[2]={0,0}; Vector2d zw_mid0=Vector2d::Zero();
  bool zw_sw_lifted=false; Vector3d zw_sw_from=Vector3d::Zero();
  bool in_zmp_walk2=false; int zw_pc=0;
  bool zw_shift=true; double zw_shift_t=0;   // ★보행개시 측방 체중이동(도착 대기)
  // ── 오프라인 1점/2점 전환(toe-pivot 굴림 궤적) ──
  bool trans_on=false; double trans_t=0, T_TRANS=1.4; int trans_to=0;      // 전환중·타이머·목표모드
  double q_from[8], q_to[8], q_live[8], cz_from=0, cz_to=0;                // 자세·높이 보간
  Matrix<double,2,3> lam; bool have_liftoff[2]={false,false}; Vector3d liftoff[2];
  Matrix<double,4,3> lam4=Matrix<double,4,3>::Zero();   // ★평발 MPC: [HL_heel,HL_toe,HR_heel,HR_toe] 점별 GRF(CoP)
  Matrix3d I_body; double mass;
  // ── WBIC QP 건강도 (접지 판정 지표) ──────────────────────────────────────
  //   qp_rate = **최근 200틱** 실패율. 접지 정상 ~0 · 발이 안 닿으면 ~1 로 붙는다.
  //   qp_K    = 그때 쓴 접촉점 수(평발 정상 4). 줄어들면 발이 뜬 것.
  //   qp_cerr = CoM 추종오차[m]. 실패 중에는 **줄지 않고 고정**된다 — 그게 특징이다.
  long qp_n=0, qp_fail=0, qp_w_n=0, qp_w_fail=0;
  double qp_rate=0.0, qp_cerr[3]={0,0,0};
  int qp_K=0;

  BipedControl(mjModel* m_, mjData* d_):m(m_),d(d_){
    nv=m->nv; nu=m->nu;
    // ★드라이브 토크한계를 MJCF **actuator ctrlrange** 에서 읽는다 (2026-08-13).
    //   종전엔 jnt_actfrcrange 를 hinge 순서로 훑었는데, 발목이 tendon 액추에이터가 된
    //   지금은 그 값이 드라이브 한계가 아니다(calf 관절 226.8 = 두 드라이브 합).
    //   ctrlrange 는 **액추에이터당 하나**라 인덱스가 제어벡터와 그대로 맞는다 —
    //   "hinge 를 순서대로 세면 액추에이터와 맞는다" 는 가정 자체가 사라져 더 안전하다.
    for(int i=0;i<nu && i<8;i++){
      double lim = m->actuator_ctrllimited[i] ? m->actuator_ctrlrange[i*2+1] : 0.0;
      drv_peak8[i] = (lim>0) ? lim : 1e8;
    }
    sph[0]=mj_name2id(m,mjOBJ_GEOM,"HL_sphere"); sph[1]=mj_name2id(m,mjOBJ_GEOM,"HR_sphere");
    fbody[0]=mj_name2id(m,mjOBJ_BODY,"HL_foot_contact_link"); fbody[1]=mj_name2id(m,mjOBJ_BODY,"HR_foot_contact_link");
    sph2[0]=mj_name2id(m,mjOBJ_GEOM,"HL_sphere2"); sph2[1]=mj_name2id(m,mjOBJ_GEOM,"HR_sphere2");
    has_heel=(sph2[0]>=0 && sph2[1]>=0);        // ★heel 구 보유=통합모델. 기본 평발(2점) 정적 rest.
    cmode = has_heel ? 1 : 0;
    if(getenv("FLAT_KCAP")) FLAT_KCAP=atof(getenv("FLAT_KCAP"));   // 튜닝용 env
    // ★스윙 게인·스텝시간 env — 반사관성(ROTOR_I)이 커지면 스윙 추종 대역이 부족해져
    //   착지가 틀어진다. 실측 armature 하에서 재튜닝하기 위한 노브.
    if(getenv("SW_KP")) SW_KP=atof(getenv("SW_KP"));
    if(getenv("SW_KD")) SW_KD=atof(getenv("SW_KD"));
    if(getenv("T_STEP")) T_STEP=atof(getenv("T_STEP"));
    if(getenv("FLAT_STEPH")) STEP_H=atof(getenv("FLAT_STEPH"));
    if(getenv("FLAT_WLAM")) FLAT_WLAM=atof(getenv("FLAT_WLAM"));
    if(getenv("FLAT_CZ")) czwalk=atof(getenv("FLAT_CZ"));
    if(getenv("FLAT_WORI")) FLAT_WORI=atof(getenv("FLAT_WORI"));
    if(getenv("T_TRANS")) T_TRANS=atof(getenv("T_TRANS"));
    // ★발디딤 게인 env — leg-odom 야코비안 편향(구중심 vs 접촉점)을 제거하면
    //   K_RETURN 이 보던 오차의 성격이 바뀐다. 편향 위에 얹혀 튜닝돼 있던 값이므로
    //   추정기 수정과 반드시 짝지어 재튜닝해야 한다.
    if(getenv("K_RETURN")) K_RETURN=atof(getenv("K_RETURN"));
    if(getenv("K_CAP"))    K_CAP   =atof(getenv("K_CAP"));
    if(getenv("FRIC_COMP")) FRIC_COMP=atof(getenv("FRIC_COMP"));   // ★마찰 전방보상 배율(0=끔)
    if(getenv("FRIC_V0"))   FRIC_V0  =atof(getenv("FRIC_V0"));
    if(getenv("FRIC_STANCE_ONLY")) FRIC_STANCE_ONLY=atoi(getenv("FRIC_STANCE_ONLY"));
    if(getenv("FLAT_CONTACT_ALL")) flat_contact_all=atoi(getenv("FLAT_CONTACT_ALL"));
    if(getenv("FRIC_ALL_MODES"))   FRIC_ALL_MODES  =atoi(getenv("FRIC_ALL_MODES"));
    if(getenv("SS_NOMINAL")) SS_NOMINAL=atof(getenv("SS_NOMINAL"));
    // ★CoM 적분항 — **기본 0(꺼짐)**. 원인 규명이 끝난 뒤에만 켤 것(위 선언부 주석 참조).
    if(getenv("STAND_KI"))      STAND_KI      = atof(getenv("STAND_KI"));
    if(getenv("STAND_I_CLAMP")) STAND_I_CLAMP = atof(getenv("STAND_I_CLAMP"));
    // ★STAND_KI_AXIS="1,1,1" 로 xy 를 열 수 있다 — **IMU 가 살아난 뒤에만.**
    //   imu_ok=false 면 여기서 켜도 런타임이 xy 를 다시 끈다(위 축 마스크).
    if(const char* a=getenv("STAND_KI_AXIS")){
      double v[3]={0,0,1}; if(sscanf(a,"%lf,%lf,%lf",&v[0],&v[1],&v[2])==3) KI_AXIS=Vector3d(v[0],v[1],v[2]); }
    // ★stance 과제 게인 env (stand-lite 브링업 — 위 선언부 주석 참조)
    { auto v3=[&](const char* k, Vector3d& dst){
        if(const char* e=getenv(k)){ double v[3];
          if(sscanf(e,"%lf,%lf,%lf",&v[0],&v[1],&v[2])==3) dst=Vector3d(v[0],v[1],v[2]); } };
      v3("STAND_COM_KP", COM_KP); v3("STAND_COM_KD", COM_KD);
      if(getenv("STAND_ORI_KP"))  ORI_KP =atof(getenv("STAND_ORI_KP"));
      if(getenv("STAND_ORI_KD"))  ORI_KD =atof(getenv("STAND_ORI_KD"));
      if(getenv("STAND_POST_KP")) POST_KP=atof(getenv("STAND_POST_KP"));
      if(getenv("STAND_POST_KD")) POST_KD=atof(getenv("STAND_POST_KD"));
      if(getenv("STAND_COM_KP")||getenv("STAND_ORI_KP")||getenv("STAND_POST_KP"))
        std::printf("[stance] ★과제게인 override — CoM kp(%.0f,%.0f,%.0f)/kd(%.0f,%.0f,%.0f) · "
                    "ori %.0f/%.0f · posture %.0f/%.0f\n",
                    COM_KP[0],COM_KP[1],COM_KP[2], COM_KD[0],COM_KD[1],COM_KD[2],
                    ORI_KP,ORI_KD, POST_KP,POST_KD); }
    // ★★2026-08-20 **좌우 발목 비대칭 주입**(진단 전용, 단위 = 도).
    //   실기 2점 stand 에서 두 밑창이 **반대로** 기울었다: HL −63.25 · HR −55.42
    //   (목표 −59.81). `Qflat8` 은 좌우 **완전 대칭**이라 시뮬에는 이 비대칭이 아예 없다
    //   — 그래서 sim 이 계속 green 인데 실기만 기운다. 실측값을 그대로 넣어
    //   **그 비대칭 하나만으로 기우는지**를 가른다. 기울면 원인이 기하로 좁혀지고,
    //   안 기울면 HR 구동 쪽(샤프트 컴플라이언스)이 남는다.
    //   ⚠운전자 관찰("HR 이 더 굽었다")이 위 기록과 **부호가 반대**다. 어느 쪽이
    //     맞는지 아직 미확정이므로 두 경우를 다 돌려볼 수 있게 좌우를 따로 받는다.
    //   ⚠진단용이다 — 지정하지 않으면 종전(대칭) 그대로다.
    if(const char* e=getenv("QFLAT_FOOT_L")) Qflat8[3]=atof(e)*M_PI/180.0;
    if(const char* e=getenv("QFLAT_FOOT_R")) Qflat8[7]=atof(e)*M_PI/180.0;
    pv.init(PREV_DECIM*0.002, 0.362);          // ★ZMP 프리뷰 게인(dt=preview간격, zc=평발 CoM높이)
    lam.setZero(); setup_gearbox();
  }
  void setup_gearbox(){
    // ★env 오버라이드(quad_mpc_wbic_17dof.py:259-261 규약과 동일) — 재빌드 없이 스윕/회귀비교용.
    //   미지정이면 위 실측 기본값을 쓴다.
    if(const char* e=getenv("ROTOR_I")) ROTOR_I=atof(e);
    //   ★kind 별 배열이 된 뒤로는 **전 축에 같은 값**을 덮는 뜻이다(스윕 편의).
    //     축 하나만 바꾸려면 JDAMP_CALF 처럼 kind 별 변수를 쓴다.
    if(const char* e=getenv("JDAMP")) for(int k=0;k<4;k++) JDAMP[k]=atof(e);
    if(const char* e=getenv("JFRIC")) for(int k=0;k<4;k++) JFRIC[k]=atof(e);
    { const char* kn[4]={"HIP","THIGH","CALF","FOOT"};
      for(int k=0;k<4;k++){
        char b[24];
        std::snprintf(b,sizeof b,"JDAMP_%s",kn[k]); if(const char* e=getenv(b)) JDAMP[k]=atof(e);
        std::snprintf(b,sizeof b,"JFRIC_%s",kn[k]); if(const char* e=getenv(b)) JFRIC[k]=atof(e); } }
    if(const char* e=getenv("GEAR_FOOT")) GEAR[3]=atof(e);
    for(int j=0;j<nu;j++){ double N=GEAR[j%4]; int dof=6+j;
      m->dof_armature[dof]=ROTOR_I*N*N; m->dof_damping[dof]=JDAMP[j%4]; m->dof_frictionloss[dof]=JFRIC[j%4]; }
    foot_rotor_to_tendon(); }

  // ★foot 로터 반사관성을 dof_armature 에서 **tendon 으로 옮긴다**(calf→foot 기구 커플링).
  //   foot 로터는 관절각이 아니라 raw 각으로 돈다(실기 coef=+1, biped_emb.yaml):
  //       raw_foot = q_foot + coef*q_calf
  //   ⇒ 로터 KE = ½·I_rot·N²·(q̇_foot + coef·q̇_calf)² 라 반사관성이 (calf,foot) **비대각**이다:
  //       M += a*[[coef², coef],[coef, 1]]
  //   ⚠dof_armature 는 M 의 **대각뿐**이라 표현 불가. fixed tendon 의 armature 가 위 형태를 만든다.
  //   ⚠**옮기는** 것이지 더하는 게 아니다 — dof_armature[foot] 을 0 으로 안 두면 이중 계상.
  //   ⚠축별 측정에선 이 항이 죽어 있었다(타축 고정). 전축 동시 가진에서만 살아난다.
  //   검증(2026-08-12 HOME): M[foot,foot] 불변 · M[calf,calf] +46% · M[calf,foot] 0.0045→0.0567
  void foot_rotor_to_tendon(){
    int t[2]={mj_name2id(m,mjOBJ_TENDON,"HL_foot_rotor"),
              mj_name2id(m,mjOBJ_TENDON,"HR_foot_rotor")};
    if(t[0]<0||t[1]<0){   // 구 MJCF(tendon 없음) 호환 — 커플링 누락 상태로 돈다
      fprintf(stderr,"  ⚠MJCF 에 *_foot_rotor tendon 이 없다 — calf↔foot 커플 반사관성 누락\n");
      return; }
    // ★반사관성뿐 아니라 **점성·마찰도** 옮긴다(2026-08-19, §1-b). 셋 다 모터축에서 작용한다.
    //   ⚠**옮기는** 것이지 더하는 게 아니다 — 관절 쪽을 0 으로 안 두면 이중 계상이다.
    for(int j=0;j<nu;j++) if(j%4==3){
      m->dof_armature[6+j]=0.0; m->dof_damping[6+j]=0.0; m->dof_frictionloss[6+j]=0.0; }
    for(int k=0;k<2;k++){
      m->tendon_armature[t[k]]     = ROTOR_I*GEAR[3]*GEAR[3];   // 0.0517
      m->tendon_damping[t[k]]      = JDAMP[3];                  // 0.110
      m->tendon_frictionloss[t[k]] = JFRIC[3];                  // 0.639
    } }

  double footz(int leg){ return d->geom_xpos[sph[leg]*3+2]; }
  Vector3d spos(int leg){ return Vector3d(d->geom_xpos[sph[leg]*3],d->geom_xpos[sph[leg]*3+1],d->geom_xpos[sph[leg]*3+2]); }

  MatrixXd foot_jac(int leg){ std::vector<double> jp(3*nv);
    double pt[3]={d->geom_xpos[sph[leg]*3],d->geom_xpos[sph[leg]*3+1],d->geom_xpos[sph[leg]*3+2]};
    mj_jac(m,d,jp.data(),nullptr,pt,fbody[leg]);
    MatrixXd J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jp[r*nv+c]; return J; }
  MatrixXd jac_com(){ std::vector<double> jc(3*nv); mj_jacSubtreeCom(m,d,jc.data(),0);
    MatrixXd J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jc[r*nv+c]; return J; }
  VectorXd qvel(){ return Map<VectorXd>(d->qvel,nv); }
  Vector3d com(){ return Vector3d(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]); }
  double base_yaw(){ double* q=&d->qpos[3];
    return std::atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3])); }

  // ── 평발(2점) 헬퍼 ──
  const double* Qcur(){ if(trans_on) return q_live; return (has_heel&&cmode==1)?Qflat8:Qhome8; }   // 전환중=보간자세
  Vector3d gpos(int geom){ return Vector3d(d->geom_xpos[geom*3],d->geom_xpos[geom*3+1],d->geom_xpos[geom*3+2]); }
  Vector3d foot_center(int leg){ if(cmode==1&&has_heel) return 0.5*(gpos(sph[leg])+gpos(sph2[leg])); return gpos(sph[leg]); }
  MatrixXd foot_jac_at(int geom,int body){ std::vector<double> jp(3*nv);
    double pt[3]={d->geom_xpos[geom*3],d->geom_xpos[geom*3+1],d->geom_xpos[geom*3+2]};
    mj_jac(m,d,jp.data(),nullptr,pt,body);
    MatrixXd J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jp[r*nv+c]; return J; }
  // 발 중심 자코비안(swing task) = 모드별 기준구(점발=tip / 평발=heel+toe) 평균
  MatrixXd foot_jac_center(int leg){ if(cmode==1&&has_heel)
      return 0.5*(foot_jac_at(sph[leg],(int)m->geom_bodyid[sph[leg]])+foot_jac_at(sph2[leg],(int)m->geom_bodyid[sph2[leg]]));
    return foot_jac(leg); }
  // swing 발 회전 자코비안(수평 유지용)
  MatrixXd foot_jacr(int leg){ std::vector<double> jr(3*nv);
    mj_jac(m,d,nullptr,jr.data(),&d->xpos[fbody[leg]*3],fbody[leg]);
    MatrixXd J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jr[r*nv+c]; return J; }
  // 접촉점(적응): 지면 근처 구만. (geom,body) 리스트.
  //
  // ★★2026-08-20 **2점 평발(cmode=1)에서는 높이로 고르지 않고 4점을 다 쓴다.**
  //   왜: 이 판정은 **모델 기하**로 하는데, 실기는 IMU 가 죽어 몸통 자세를 모른다.
  //   그러면 발 구의 높이가 통째로 틀린다. 실측(2026-08-20 stand):
  //       HL_sphere 18.0mm 뜸 · HL_sphere2 54.4mm 뜸 · HR_sphere 접촉 · HR_sphere2 35.0mm 뜸
  //       → 모델이 세는 K = **1** (평발 정상은 4). 모니터에도 K=2·com_err 85mm 로 찍혔다.
  //   지지면이 사각형이 아니라 점 하나가 되니 **균형을 잡을 수가 없다**(QP 는 0% 실패인데도).
  //   ⇒ 틀린 모델로 접촉을 세는 것보다 **아는 사실**을 쓴다 — 2점 평발 stand 는 운전자가
  //     양발을 바닥에 놓고 시작한다. 보행에서 게이트 위상을 접촉으로 쓰는 것과 같은 논리다
  //     ("실기엔 발 힘센서가 없다" — biped_deploy.cpp).
  //   ⚠발이 실제로 안 닿았는데 닿았다고 하면 QP 가 없는 지면반력을 요구한다. 그래서
  //     biped_deploy 의 **접지 가드**(실측 |τ| vs 매달림 예측 비 1.25)가 먼저 막는다.
  //   ⚠1점 점발 보행에는 적용하지 않는다 — 거기선 스윙발이 진짜로 떠 있다.
  //   FLAT_CONTACT_ALL=0 으로 종전(높이 판정) 동작.
  int flat_contact_all = 1;
  std::vector<std::pair<int,int>> contact_pts(std::vector<int> stance){
    std::vector<std::pair<int,int>> pts;
    const bool all = (cmode==1 && has_heel && flat_contact_all);
    for(int f:stance){ std::vector<int> in; int cand[2]={sph[f], has_heel?sph2[f]:-1};
      for(int g:cand){ if(g<0) continue;
        if(all || d->geom_xpos[g*3+2] < m->geom_size[g*3]+0.012) in.push_back(g); }
      if(in.empty()) in.push_back(sph[f]);
      for(int g:in) pts.push_back({g, m->geom_bodyid[g]}); }
    return pts; }

  // ── 2점 정적 양발지지 QP (Python wbic_stance 이식) ──
  void wbic_stance(){
    using namespace bipedwbic;
    auto cpts=contact_pts({0,1}); int K=(int)cpts.size(); int nz=nv+3*K;
    std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
    Map<Matrix<double,Dynamic,Dynamic,RowMajor>> M(Mb.data(),nv,nv);
    VectorXd h=Map<VectorXd>(d->qfrc_bias,nv), qv=qvel();
    std::vector<MatrixXd> Js; for(auto&cp:cpts) Js.push_back(foot_jac_at(cp.first,cp.second));
    MatrixXd Jc=jac_com(); Vector3d c=com();
    MatrixXd P=MatrixXd::Zero(nz,nz); VectorXd g=VectorXd::Zero(nz);
    // CoM task (xy+z) — 게인은 env 조절(STAND_COM_KP/KD · 선언부 주석)
    Vector3d kp=COM_KP, kd=COM_KD; Vector3d comref(com_ref_xy[0],com_ref_xy[1],com_ref_z);
    Vector3d cerr = comref - c;
    // ★적분항(기본 0). 얼리는 조건은 **적분하기 전에** 판정한다 — 이번 틱 오차를 쌓을지 말지다.
    Vector3d a_i = Vector3d::Zero();
    if(STAND_KI > 0.0){
      //   ⚠접촉이 부족하면(K<4) 발이 뜬 상태의 오차라 되잡을 대상이 아니다. QP 실패도 같다
      //     — 그때는 중력보상 폴백이라 제어가 안 먹는데 쌓으면 복귀 순간 튄다.
      const bool healthy = (qp_K >= 4) && (qp_rate < 0.5);
      ki_frozen = !healthy;
      // ★축 마스크 — imu_ok 가 아니면 xy 를 **강제로** 끈다(KI_AXIS 를 잘못 켜도 막힌다).
      Vector3d ax = KI_AXIS;
      if(!imu_ok){ ax[0] = 0.0; ax[1] = 0.0; }
      if(healthy){
        com_i += cerr.cwiseProduct(ax) * dt_ctrl;   // 꺼진 축은 **누적 자체를 안 한다**
        //   가속도 기여를 상한으로 잘라 되돌려 넣는다(적분 자체를 클램프해야 와인드업이 안 쌓인다).
        for(int j=0;j<3;j++){
          if(ax[j] == 0.0){ com_i[j] = 0.0; continue; }   // 꺼진 축은 비운다
          const double lim = STAND_I_CLAMP / std::max(1e-9, STAND_KI);
          com_i[j] = std::max(-lim, std::min(lim, com_i[j]));
        }
      }
      a_i = STAND_KI * com_i.cwiseProduct(ax);
    } else { com_i.setZero(); ki_frozen=false; }
    Vector3d a_com=kp.cwiseProduct(cerr)-kd.cwiseProduct(Jc*qv)+a_i;
    P.topLeftCorner(nv,nv)+=Jc.transpose()*Jc; g.head(nv)-=Jc.transpose()*a_com;
    // 자세 레벨링(현재 yaw 프레임)
    Vector4d qc; for(int i=0;i<4;i++) qc[i]=d->qpos[3+i];
    double yaw=std::atan2(2*(qc[0]*qc[3]+qc[1]*qc[2]),1-2*(qc[2]*qc[2]+qc[3]*qc[3]));
    double qlev[4]={std::cos(yaw/2),0,0,std::sin(yaw/2)}, ql_conj[4]={qlev[0],-qlev[1],-qlev[2],-qlev[3]};
    double dq[4]={ql_conj[0]*qc[0]-ql_conj[1]*qc[1]-ql_conj[2]*qc[2]-ql_conj[3]*qc[3],
                  ql_conj[0]*qc[1]+ql_conj[1]*qc[0]+ql_conj[2]*qc[3]-ql_conj[3]*qc[2],
                  ql_conj[0]*qc[2]-ql_conj[1]*qc[3]+ql_conj[2]*qc[0]+ql_conj[3]*qc[1],
                  ql_conj[0]*qc[3]+ql_conj[1]*qc[2]-ql_conj[2]*qc[1]+ql_conj[3]*qc[0]};
    Vector3d oerr; { double s=(dq[0]<0?-1:1); Vector3d v(dq[1],dq[2],dq[3]); double n=v.norm();
      oerr=(n<1e-12)?Vector3d(0,0,0):(2.0*std::atan2(n,std::abs(dq[0]))*s/n)*v; }
    for(int j=0;j<3;j++){ double a=ORI_KP*(-oerr[j])-ORI_KD*qv[3+j]; P(3+j,3+j)+=W_ORI; g[3+j]-=W_ORI*a; }
    // posture — ★thigh/calf는 약하게(CoM 높이 task가 다리 신전으로 높이 조절 가능하게), 발목/hip은 firm
    const double* Qh=Qcur();
    for(int j=0;j<nu;j++){ double a=POST_KP*(Qh[j]-d->qpos[7+j])-POST_KD*qv[6+j];
      int lj=j%4; double w=(lj==3)?W_ANKLE : (lj==1||lj==2)?FLAT_WLEG : W_POST;
      P(6+j,6+j)+=w; g[6+j]-=w*a; }
    P.topLeftCorner(nv,nv)+=1e-4*MatrixXd::Identity(nv,nv);
    for(int k=0;k<K;k++) P.block(nv+3*k,nv+3*k,3,3)+=1e-2*Matrix3d::Identity();   // ★λ 정칙화↑(rank-deficient 안정)
    // 등식: base6 (+ 접촉3K, soft 아니면). ★2026-09-22 접촉 가속 등식을 soft 고가중 목적항으로
    //   옮기는 옵션(WBIC_SOFT_CONTACT=1). 2점 강체발의 종속행(code4=rank축퇴)을 프루닝처럼 버리지
    //   않고 penalty 로 남긴다 → base6 만 등식(full-rank)이라 eiquadprog 항상 풀고, 접촉은 고가중
    //   강제라 프루닝처럼 다리가 처지지 않는다. λ→base 결합(6행)은 항상 등식 유지.
    static const bool   SOFT_CT = getenv("WBIC_SOFT_CONTACT") && atoi(getenv("WBIC_SOFT_CONTACT"));
    static const double W_CT    = getenv("WBIC_W_CONTACT") ? atof(getenv("WBIC_W_CONTACT")) : 5e3;
    int neq = SOFT_CT ? 6 : 6+3*K;
    MatrixXd A=MatrixXd::Zero(neq,nz); VectorXd bb=VectorXd::Zero(neq);
    A.block(0,0,6,nv)=M.topRows(6); bb.head(6)=-h.head(6);
    for(int k=0;k<K;k++){
      A.block(0,nv+3*k,6,3)=-Js[k].leftCols(6).transpose();       // λ→base 결합(항상 등식)
      Vector3d bct=-STANCE_KD*(Js[k]*qv);
      if(SOFT_CT){                                                // 접촉 가속 = soft penalty(고가중)
        P.topLeftCorner(nv,nv) += W_CT*(Js[k].transpose()*Js[k]);
        g.head(nv)             -= W_CT*(Js[k].transpose()*bct);
      } else {                                                    // 종전: 하드 등식(+ EQ_PRUNE 필요)
        A.block(6+3*k,0,3,nv)=Js[k]; bb.segment(6+3*k,3)=bct;
      }
    }
    // 부등식: 마찰추 + λz≥min (토크한계 없음, Python wbic_stance 동일)
    std::vector<VectorXd> Gr; std::vector<double> hv; int sgn[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    for(int k=0;k<K;k++){ int o=nv+3*k;
      for(int s=0;s<4;s++){ VectorXd r=VectorXd::Zero(nz); r[o]=sgn[s][0]; r[o+1]=sgn[s][1]; r[o+2]=-MU_EFF; Gr.push_back(r); hv.push_back(0.0);}
      VectorXd r=VectorXd::Zero(nz); r[o+2]=-1; Gr.push_back(r); hv.push_back(-LAMZ_MIN); }
    static const double WREG = getenv("WBIC_REG") ? atof(getenv("WBIC_REG")) : 1e-6;
    P=(0.5*(P+P.transpose())).eval()+WREG*MatrixXd::Identity(nz,nz);   // ★정칙화(1e-8→1e-6, eiquadprog 안정)
    MatrixXd CE=A; VectorXd ce0=-bb;
    // ★★등식 제약 프루닝 (2026-08-28 실기 확정) ───────────────────────────────
    //   실기 2점 평발 stand 에서 QP 실패 20~100%, 사유코드 **4 = REDUNDANT_EQUALITIES**.
    //   원인은 수치가 아니라 **구조**다: 한 발에 접촉점이 둘인데 발은 강체라, 두 점의
    //   가속도 구속 6행 중 **두 점을 잇는 선 방향 1행이 종속**이다(강체는 그 거리를 못 바꾼다).
    //   발이 둘이니 정확히 2행 남는다 → eiquadprog 는 통째로 포기하고 중력보상 폴백(τ=h)으로
    //   떨어진다. **두 해가 매 틱 번갈아 나가는 것이 실기에서 본 떨림**이었다.
    //   (sim(biped_mpc_wbic.py:243)은 이 경우 solver 를 proxqp 로 갈아타 우회한다)
    //   ⇒ rank-revealing QR 로 독립 행만 남긴다. 버리는 행은 남은 행들의 선형결합이라
    //     해집합이 안 바뀐다(구조적 종속이므로 일관성도 보장).  끄려면 WBIC_EQ_PRUNE=0.
    // ★기본 OFF 로 되돌림 (2026-08-28): sim 에선 QP 실패 18.5%→0% 로 좋아졌지만
    //   **실기에서는 균형이 오히려 나빠졌다**(사용자 관찰: "더 힘이 빠진다").
    //   실기 검증 전에 기본값을 바꾼 것이 잘못이었다 — 켜려면 WBIC_EQ_PRUNE=1.
    //   ⚠종전 동작(프루닝 없음)은 QP 실패 시 **중력보상 폴백**이라 안전측이다.
    static const bool EQ_PRUNE = getenv("WBIC_EQ_PRUNE") && atoi(getenv("WBIC_EQ_PRUNE"));
    if(EQ_PRUNE && CE.rows() > 0){
      Eigen::ColPivHouseholderQR<MatrixXd> qr(CE.transpose());
      qr.setThreshold(1e-9);
      const int rk = (int)qr.rank();
      { static int last_r=-1;
        if(rk != last_r){ last_r = rk;
          std::fprintf(stderr,"[wbic_stance] 등식 rank %d/%d %s\n", rk, neq,
                       rk<neq ? "→ **프루닝 발동**(축퇴 제거)" : "(축퇴 없음)"); } }
      if(rk < CE.rows()){
        const auto& perm = qr.colsPermutation().indices();
        std::vector<int> keep(perm.data(), perm.data()+rk);
        std::sort(keep.begin(), keep.end());
        MatrixXd CE2(rk, CE.cols()); VectorXd ce02(rk);
        for(int i=0;i<rk;i++){ CE2.row(i)=CE.row(keep[i]); ce02[i]=ce0[keep[i]]; }
        CE = CE2; ce0 = ce02; neq = rk;
      }
    }
    int nci=(int)Gr.size(); MatrixXd CI(nci,nz); VectorXd ci0(nci);
    for(int i=0;i<nci;i++){ CI.row(i)=-Gr[i]; ci0[i]=hv[i]; }
    VectorXd x(nz); eiquadprog::solvers::EiquadprogFast qp; qp.reset(nz,neq,nci);
    auto st=qp.solve_quadprog(P,g,CE,ce0,CI,ci0,x);
    // ★QP 건강도를 **항상** 집계해 밖으로 낸다 (2026-08-14). 종전엔 WBIC_DBG 일 때
    //   stderr 로만 나가서 실기에서는 사실상 못 봤다.
    //   ★이 지표가 **접지 여부를 그대로 가른다** — 매달림 시뮬 실측:
    //       접지   K=4 · 실패 0.05% · com_err 6 mm
    //       매달림 K=3 · 실패  95%  · com_err 127 mm 에서 **고정**(전혀 안 줄어듦)
    //     발이 덜 닿으면 WBIC 가 요구하는 λ 를 지면이 못 내줘 해가 안 나오고, 폴백인
    //     중력보상 홀드로 떨어진다. 그런데 겉보기엔 **안정돼 보인다** — 그게 위험하다.
    //     "stand 가 되는 것처럼 보이는데 실은 폐루프가 죽은" 상태를 이 숫자로만 가른다.
    //   ⚠누적률은 초기 과도에 희석된다 ⇒ **최근 창(200틱)** 비율을 따로 낸다.
    qp_n++; qp_w_n++;
    const bool qp_ok = (st==eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL);
    if(!qp_ok){ qp_fail++; qp_w_fail++;
      // ★실패 사유를 남긴다 (2026-08-28) — REDUNDANT_EQUALITIES 면 rank 축퇴(정규화로 해결),
      //   INFEASIBLE 이면 접촉/마찰콘 제약이 진짜로 안 맞는 것(접지·자세 문제).
      static long fs_cnt[8]={0}; const int si=(int)st & 7; fs_cnt[si]++;
      static long fs_tick=0;
      if((++fs_tick % 1500) == 0){                      // ~3s @500Hz
        std::fprintf(stderr,"[wbic] QP 실패 사유 누적: ");
        for(int q=0;q<8;q++) if(fs_cnt[q]) std::fprintf(stderr,"code%d=%ld ",q,fs_cnt[q]);
        std::fprintf(stderr,"(최근창 %.0f%%)\n", qp_rate*100.0); }
    }
    qp_K = K;
    qp_cerr[0]=com_ref_xy[0]-c[0]; qp_cerr[1]=com_ref_xy[1]-c[1]; qp_cerr[2]=com_ref_z-c[2];
    if(qp_w_n>=200){ qp_rate = (double)qp_w_fail/(double)qp_w_n; qp_w_n=0; qp_w_fail=0; }
    if(getenv("WBIC_DBG") && qp_n%200==0)
      std::fprintf(stderr,"[wbic_stance] K=%d QP실패 %ld/%ld(최근 %.0f%%) · com_err=(%.3f,%.3f,%.3f)\n",
                   K,qp_fail,qp_n,qp_rate*100.0,qp_cerr[0],qp_cerr[1],qp_cerr[2]);
    if(st==eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL){
      VectorXd qdd=x.head(nv); VectorXd tau=M.block(6,0,nu,nv)*qdd+h.segment(6,nu);
      for(int k=0;k<K;k++) tau-=Js[k].block(0,6,3,nu).transpose()*x.segment(nv+3*k,3);
      set_ctrl_from_tau(tau);
    } else { set_ctrl_from_tau(VectorXd(h.segment(6,nu))); }  // 실패=중력보상 홀드
  }

  // ★관절토크 → d->ctrl. MJCF 액추에이터가 **드라이브 좌표**라 전단 후 쓴다 (2026-08-13).
  //   발목 모터가 tendon(coef 1,1)에 물려 있어 ctrl[foot] 이 calf·foot 두 DOF 에 같이 걸린다.
  //   관절토크를 그대로 쓰면 무릎에 발목토크가 덤으로 실려 실기와 다른 로봇이 된다.
  //   클립도 여기서 드라이브 한계로 — 실기 한계는 모터에 걸리지 관절에 걸리지 않는다.
  // ★마찰 전방보상 (2026-08-14 신설). 기본 **꺼짐**(FRIC_COMP=1 로 켠다).
  //
  //   왜 필요한가 — **WBIC 는 관절 쿨롱마찰을 전혀 모른다.** JFRIC 은 여태 모델에
  //   (`dof_frictionloss`) 넣기만 하고 토크 계산엔 쓰이지 않았다. 마찰이 작을 땐
  //   무시해도 됐지만 실측값(hip 0.827)에선 안 된다:
  //       JFRIC 0.30 / 0.38  → 2점 stand 60s 무낙상
  //       JFRIC 축별 실측     → **20.7s 낙상**
  //   즉 "예전엔 됐다" 는 마찰을 **과소평가한 모델** 덕이었다. 모델이 정확해지자
  //   원래부터 없던 보상 부재가 드러난 것이다 — 되돌릴 회귀가 아니다.
  //
  //   보상식: τ += JFRIC·tanh(q̇/v0).  sign(q̇) 을 그대로 쓰면 q̇≈0 에서 부호가 튀어
  //   채터링이 난다. v0 는 그 전환폭이다(FRIC_V0, 기본 0.05 rad/s).
  //   ⚠**과보상은 자기가진**이 된다(마찰보다 크게 밀면 진동). 그래서 배율을 노출한다.
  //   ⚠Python biped_wbic.py 와 같은 식이어야 파리티가 유지된다.
  //
  // ★★**스탠스 다리에만 건다**(FRIC_STANCE_ONLY, 기본 1). 2026-08-14 실측 근거:
  //   전 관절에 걸었더니 stand 는 고쳐지는데(20.7s→60s) **보행이 5개 실패**했다
  //   (v0.15 낙상 · 전진거리 0.82→0.32). 이유가 분명하다 —
  //   스윙 중엔 q̇ 가 커서 `tanh→1` 이라 마찰 전량을 더해주는데, 스윙 다리는
  //   이미 마찰을 이기고 자유롭게 돌고 있다. 거기에 더하면 **과보상 = 에너지 주입**이다.
  //   반대로 문제가 난 자리는 스탠스다: 지지 다리는 CoM 을 미세하게 되잡아야 하는데
  //   그 보정토크가 마찰 밴드(hip 0.827Nm)에 먹혀 버린다.
  //   ⇒ 마찰이 **해가 되는 곳에만** 보상한다. 2점 평발(cmode=1)은 양발이 스탠스다.
  //
  // ★★**2점 평발(cmode=1)에서만 켠다.** 1점 점발 보행에서는 필요 없고 오히려 해롭다 —
  //   실측(2026-08-14):
  //     보상 OFF : 보행 3속도 + 선회·측방 + 배포경로 2속도 **전부 통과**
  //     보상 ON  : 배포경로 vx0.20 이 **2회 낙상**(x 2.77 → 0.00)
  //   이유가 물리적으로 분명하다. `τ += JFRIC·tanh(q̇/v0)` 는 **운동 방향으로 미는 항**,
  //   즉 **음의 감쇠**다. 안정여유를 깎으므로 지연 8.4ms 와 만나면 발산 쪽으로 간다.
  //   보행은 애초에 마찰이 문제가 아니었다(스윙은 이미 마찰을 이기고, 스탠스는 지지력이
  //   커서 마찰 밴드가 상대적으로 작다). 문제는 **정적 균형 조절**이다 —
  //   2점 stand 는 CoM 을 미세하게 되잡아야 하는데 그 보정토크가 마찰에 먹힌다.
  //   ⇒ 마찰이 해가 되는 **그 모드에서만** 보상한다.  `FRIC_ALL_MODES=1` 로 강제 가능.
  //   ★V0 0.05 → **0.20**: 0.05 는 전환이 급해 스탠스에서도 음의감쇠가 세다
  //     (배포 vx0.15 낙상). 0.20 이 stand tilt 0.1° 로 가장 좋았다.
  double FRIC_COMP=1.0, FRIC_V0=0.20; int FRIC_STANCE_ONLY=1, FRIC_ALL_MODES=0;
  void set_ctrl_from_tau(const VectorXd& tau){
    VectorXd t=tau;
    // ★★2026-08-19 보상도 **손실이 있는 좌표**에서 해야 한다(§1-b 로 손실공간이 바뀌었다).
    //   foot 마찰은 이제 관절이 아니라 tendon(raw각)에 있다. 그러니
    //     ① 판단 속도는 관절속도가 아니라 **raw 속도** L̇ = q̇_calf + q̇_foot 이고
    //     ② 결과 토크는 coefᵀ=(1,1) 로 **calf·foot 두 관절에 동시에** 걸린다.
    //   종전처럼 foot 관절에만 걸면 calf 쪽 몫이 통째로 빠지고, 무릎만 도는 구간에서는
    //   부호까지 틀린다(그게 §1-b 가 지적한 바로 그 오류다).
    if(FRIC_COMP>0 && (cmode==1 || FRIC_ALL_MODES)) for(int leg=0; leg<2; leg++){
      // cmode=1(2점 평발)은 양발 지지 · 1점 점발은 swing 다리만 제외한다
      if(FRIC_STANCE_ONLY && cmode!=1 && leg==swing) continue;
      const int b=4*leg;
      for(int k=0;k<3;k++)                                   // hip·thigh·calf — 관절축
        t[b+k] += FRIC_COMP*JFRIC[k]*std::tanh(d->qvel[6+b+k]/FRIC_V0);
      const double draw = d->qvel[6+b+2] + d->qvel[6+b+3];   // L̇ (coef=1,1)
      const double f = FRIC_COMP*JFRIC[3]*std::tanh(draw/FRIC_V0);
      t[b+2] += f; t[b+3] += f;                              // coefᵀ 로 두 관절에
    }
    VectorXd u=bipedwbic::tau_to_drive(t);
    for(int i=0;i<nu;i++) d->ctrl[i]=std::max(-drv_peak8[i],std::min(drv_peak8[i],u[i]));
  }

  void set_contact_mode(int cm){ if(!has_heel||cm==cmode) return; cmode=cm; reset(); }   // 초기용(스냅)

  // ★런타임 부드러운 전환 시작(toe-pivot 굴림): 발목·다리·높이를 목표자세로 서서히 굴림
  void transition_to(int cm){
    if(!has_heel || cm==cmode || trans_on) return;
    const double* qf=Qcur(); const double* qt=(cm==1)?Qflat8:Qhome8;
    for(int j=0;j<8;j++){ q_from[j]=qf[j]; q_to[j]=qt[j]; q_live[j]=qf[j]; }
    cz_from=com_ref_z; cz_to=(cm==1)?0.362:0.483;
    trans_on=true; trans_t=0; trans_to=cm;
  }
  // 전환 궤적 재생(양발/toe 적응접촉 wbic_stance로 추종)
  void do_transition(double dt){
    double a=trans_t/T_TRANS; a=a<0?0:(a>1?1:a); a=a*a*(3-2*a);   // smoothstep
    for(int j=0;j<8;j++) q_live[j]=q_from[j]*(1-a)+q_to[j]*a;
    com_ref_z=cz_from*(1-a)+cz_to*a;
    auto cpts=contact_pts({0,1}); Vector3d sc(0,0,0);             // 접지 구 중심(적응: 밑창→toe)
    for(auto&cp:cpts) sc+=gpos(cp.first); if(cpts.size()) sc/=(double)cpts.size();
    com_ref_xy<<sc[0],sc[1];
    wbic_stance();
    yaw_hold=base_yaw(); yaw_hold_set=true; yaw_des=base_yaw();
    trans_t+=dt;
    if(trans_t>=T_TRANS){ trans_on=false; cmode=trans_to; }        // 완료→목표모드 확정
  }

  void compute_Icom(){ mj_forward(m,d); mass=m->body_subtreemass[0];
    Vector3d c=com(); I_body.setZero();
    for(int b=1;b<m->nbody;b++){ double ms=m->body_mass[b]; if(ms<=0) continue;
      Vector3d r(d->xipos[b*3]-c[0],d->xipos[b*3+1]-c[1],d->xipos[b*3+2]-c[2]);
      Map<Matrix<double,3,3,RowMajor>> Rb(&d->ximat[b*9]);
      Vector3d bi(m->body_inertia[b*3],m->body_inertia[b*3+1],m->body_inertia[b*3+2]);
      Matrix3d Ib=Rb*bi.asDiagonal()*Rb.transpose();
      I_body+=Ib+ms*(r.dot(r)*Matrix3d::Identity()-r*r.transpose()); } }

  void reset(){
    for(int i=0;i<nq();i++) d->qpos[i]=0; d->qpos[3]=1;
    const double* Qh=Qcur();                          // ★모드별 home(점발 세움 / 평발 눕힘)
    for(int j=0;j<nu;j++) d->qpos[7+j]=Qh[j];
    d->qpos[2]=0.7; for(int i=0;i<nv;i++) d->qvel[i]=0; mj_forward(m,d);
    double zmin=1e9;
    for(int l=0;l<2;l++){ zmin=std::min(zmin,footz(l)-m->geom_size[sph[l]*3]);
      if(has_heel) zmin=std::min(zmin,d->geom_xpos[sph2[l]*3+2]-m->geom_size[sph2[l]*3]); }  // 평발=heel도 접지
    d->qpos[2]-=zmin; mj_forward(m,d);
    Vector3d c=com(); com0=c.head(2); com_ref_xy=c.head(2);
    for(int l=0;l<2;l++) nominal_off[l]=foot_center(l).head(2)-c.head(2);
    for(int l=0;l<2;l++) for(int i=0;i<4;i++) foot_home_quat[l][i]=d->xquat[fbody[l]*4+i];   // swing 수평 목표
    com_ref_z=c[2];
    stance=1; swing=0; t_ss=0; _k=0; yaw_des=0; yaw_hold_set=false; have_liftoff[0]=have_liftoff[1]=false;
    for(int i=0;i<nv;i++) d->qvel[i]=0;
    compute_Icom();
  }
  int nq(){ return m->nq; }

  // ── event-DCM 게이트 ──
  void step_gait(double dt,int&st,int&sw,double&s){
    Vector3d c=com(); VectorXd qv=qvel(); MatrixXd Jc=jac_com(); Vector2d vcom=(Jc*qv).head(2);
    double z=std::max(c[2]-std::min(footz(0),footz(1)),0.15), w=std::sqrt(GVEC/z);
    // ★측방 DCM 트리거를 body-frame으로(yaw 나도 올바른 측방=보행 강건). 발 중점 기준 DCM벡터를 body-y축에 투영.
    double ya=base_yaw(), cya=std::cos(ya), sya=std::sin(ya);
    Vector3d fc0=foot_center(0), fc1=foot_center(1);   // 발 중점(점발=tip / 평발=밑창중점)
    double midx=0.5*(fc0[0]+fc1[0]);
    double midy=0.5*(fc0[1]+fc1[1]);
    double dcmx=c[0]+vcom[0]/w-midx, dcmy=c[1]+vcom[1]/w-midy;   // world DCM(발중점 기준)
    double dcm_by=-sya*dcmx+cya*dcmy;                            // body-y 성분(직진 yaw=0시 =dcmy)
    double sy=(swing==0)?1.0:-1.0;
    s=std::min(std::max(t_ss/SS_NOMINAL,0.0),1.0);
    bool committed=sy*dcm_by>TRIG_Y;
    if(t_ss>SS_MIN&&(committed||t_ss>SS_MAX)){ std::swap(stance,swing); t_ss=0;
      liftoff[swing]=foot_center(swing); have_liftoff[swing]=true; st=stance; sw=swing; s=0; return; }
    t_ss+=dt; st=stance; sw=swing;
  }

  // ── base-frame 발배치 (dcm_target) ──
  Vector2d dcm_target(int sw,double s){
    Vector3d c=com(); MatrixXd Jc=jac_com(); VectorXd qv=qvel(); Vector2d vcom_w=(Jc*qv).head(2);
    double z=std::max(c[2]-std::min(footz(0),footz(1)),0.15), w=std::sqrt(GVEC/z);
    double yaw=yaw_des, cy=std::cos(yaw), sy=std::sin(yaw);
    auto to_b=[&](Vector2d v){ return Vector2d(cy*v[0]+sy*v[1],-sy*v[0]+cy*v[1]); };
    auto to_w=[&](Vector2d v){ return Vector2d(cy*v[0]-sy*v[1], sy*v[0]+cy*v[1]); };
    Vector2d v_b=to_b(vcom_w), err_b=to_b(c.head(2)-com0), off=nominal_off[sw];
    double lat=(off[1]>0)?1.0:-1.0;
    if(cmode==1 && has_heel){          // ★평발: 전후=capture+return(과속 브레이킹, 발목ZMP 보조)·측방=capture(밑창 좁음)
      Vector2d st_b=to_b(foot_center(1-sw).head(2)-c.head(2));
      double rel_fwd = off[0] + FLAT_KCAP*v_b[0]/w + K_RETURN*err_b[0];   // capture로 CoM 앞서기 방지
      rel_fwd = std::min(std::max(rel_fwd, off[0]-CAP_CLAMP), off[0]+CAP_CLAMP);
      double rel_lat_cap = SPREAD*off[1] + K_LAT*(v_b[1]/w);
      double gap=std::min(std::max(lat*(rel_lat_cap-st_b[1]),GAP_MIN),GAP_MAX);
      double rel_lat = st_b[1]+lat*gap;
      return c.head(2)+to_w(Vector2d(rel_fwd,rel_lat));
    }
    double rel_fwd=off[0]+K_CAP*v_b[0]/w+K_RETURN*err_b[0];
    rel_fwd=std::min(std::max(rel_fwd,off[0]-CAP_CLAMP),off[0]+CAP_CLAMP);
    double rel_lat=SPREAD*off[1]+K_LAT*(v_b[1]/w)+K_RET_LAT*err_b[1];
    Vector2d st_b=to_b(foot_center(1-sw).head(2)-c.head(2));
    double gap=std::min(std::max(lat*(rel_lat-st_b[1]),GAP_MIN),GAP_MAX);
    rel_lat=st_b[1]+lat*gap;
    return c.head(2)+to_w(Vector2d(rel_fwd,rel_lat));
  }
  // swing 궤적
  void swing_traj(int leg,double s,Vector3d&p,Vector3d&v){
    Vector3d p0=liftoff[leg]; Vector2d tgt=dcm_target(leg,s);
    double clr=m->geom_size[sph[leg]*3];    // sphere r
    double gz=std::min(footz(0),footz(1))+clr;
    Vector3d p1(tgt[0],tgt[1],gz);
    double ss=std::min(std::max(s,0.0),1.0);
    double sm=10*ss*ss*ss-15*ss*ss*ss*ss+6*ss*ss*ss*ss*ss;
    double dsm=(30*ss*ss-60*ss*ss*ss+30*ss*ss*ss*ss)/std::max(1e-6,(1-DS_FRAC)*T_STEP);
    p=p0+(p1-p0)*sm; double zl=4*STEP_H*ss*(1-ss);
    p[2]=p0[2]+(p1[2]-p0[2])*sm+zl;
    v=(p1-p0)*dsm; v[2]=(p1[2]-p0[2])*dsm+4*STEP_H*(1-2*ss)*dsm;
  }

  // ── MPC ──
  Matrix<double,2,3> mpc_grf(int stanceLeg){
    using namespace bipedmpc; MpcCfg c; c.N=MPC_N; c.DT=MPC_DT; c.TOTAL_MASS=mass; c.G_ACC=9.81;
    c.MU=MU_EFF; c.LAMZ_MIN=LAMZ_MIN; c.LAMZ_MAX=2.0*mass*9.81; c.I_BODY=I_body;
    double qd[13]={200,200,100,0,0,200,0,0,1,10,10,1,0}; for(int i=0;i<13;i++) c.Qdiag[i]=qd[i];
    c.Rdiag=Vector3d(1e-6,1e-6,1e-6);
    // body_x0
    double Rm[9]; mju_quat2Mat(Rm,&d->qpos[3]); Map<Matrix<double,3,3,RowMajor>> R(Rm);
    double pitch=std::asin(std::max(-1.0,std::min(1.0,-R(2,0))));
    double roll=std::atan2(R(2,1),R(2,2)), yaw=std::atan2(R(1,0),R(0,0));
    MatrixXd Jc=jac_com(); VectorXd qv=qvel(); Vector3d vcom=Jc*qv;
    Vector3d wb(d->qvel[3],d->qvel[4],d->qvel[5]), ow=R*wb; Vector3d cc=com();
    Matrix<double,13,1> x0; x0<<roll,pitch,yaw, cc[0],cc[1],cc[2], ow[0],ow[1],ow[2], vcom[0],vcom[1],vcom[2], -9.81;
    Vector3d frel[2]; for(int i=0;i<2;i++) frel[i]=foot_center(i)-cc;
    std::array<int,2> cur={stanceLeg==0?1:0, stanceLeg==1?1:0};
    std::vector<std::array<int,2>> cs(MPC_N,cur);
    std::array<Vector3d,2> fp0={frel[0],frel[1]}; std::vector<std::array<Vector3d,2>> fp(MPC_N,fp0);
    double ya=base_yaw(), cya=std::cos(ya),sya=std::sin(ya);   // ★속도명령=실제 base yaw(base-relative, 17-DOF yaw_m 방식)
    double vxw=cya*vx_cmd-sya*vy_cmd, vyw=sya*vx_cmd+cya*vy_cmd;
    Matrix<double,13,1> xr; xr<<0,0,yaw_des, cc[0],cc[1],com_ref_z, 0,0,wz_cmd, vxw,vyw,0, -9.81;  // 헤딩참조=yaw_des
    return mpc_qp_plan(c,x0,cs,fp,xr);
  }

  // ★평발 MPC: stance 발을 heel+toe 2점으로(총 4점 [HL_h,HL_t,HR_h,HR_t]) → heel/toe fz 분배=CoP=pitch 권한.
  //   f_z≥0(마찰추 내장)이 CoP를 발 안(heel~toe)으로 자동 제약 = ZMP 제약. 단일지지 pitch를 MPC가 계획.
  void mpc_grf_flat(int stanceLeg){
    using namespace bipedmpc; MpcCfg c; c.N=MPC_N; c.DT=MPC_DT; c.TOTAL_MASS=mass; c.G_ACC=9.81;
    c.MU=MU_EFF; c.LAMZ_MIN=LAMZ_MIN; c.LAMZ_MAX=2.0*mass*9.81; c.I_BODY=I_body;
    double qd[13]={200,200,100,0,0,200,0,0,1,10,10,1,0}; for(int i=0;i<13;i++) c.Qdiag[i]=qd[i];
    c.Rdiag=Vector3d(1e-6,1e-6,1e-6);
    double Rm[9]; mju_quat2Mat(Rm,&d->qpos[3]); Map<Matrix<double,3,3,RowMajor>> R(Rm);
    double pitch=std::asin(std::max(-1.0,std::min(1.0,-R(2,0))));
    double roll=std::atan2(R(2,1),R(2,2)), yaw=std::atan2(R(1,0),R(0,0));
    MatrixXd Jc=jac_com(); VectorXd qv=qvel(); Vector3d vcom=Jc*qv;
    Vector3d wb(d->qvel[3],d->qvel[4],d->qvel[5]), ow=R*wb; Vector3d cc=com();
    Matrix<double,13,1> x0; x0<<roll,pitch,yaw, cc[0],cc[1],cc[2], ow[0],ow[1],ow[2], vcom[0],vcom[1],vcom[2], -9.81;
    // 4 접촉점: [HL_heel(sph2[0]), HL_toe(sph[0]), HR_heel(sph2[1]), HR_toe(sph[1])], CoM 상대
    std::vector<Vector3d> pts={ gpos(sph2[0])-cc, gpos(sph[0])-cc, gpos(sph2[1])-cc, gpos(sph[1])-cc };
    std::vector<int> cur(4,0); cur[stanceLeg*2]=1; cur[stanceLeg*2+1]=1;   // stance 발 heel+toe만 접촉
    std::vector<std::vector<int>> cs(MPC_N,cur);
    std::vector<std::vector<Vector3d>> fp(MPC_N,pts);
    double ya=base_yaw(), cya=std::cos(ya),sya=std::sin(ya);
    double vxw=cya*vx_cmd-sya*vy_cmd, vyw=sya*vx_cmd+cya*vy_cmd;
    Matrix<double,13,1> xr; xr<<0,0,yaw_des, cc[0],cc[1],com_ref_z, 0,0,wz_cmd, vxw,vyw,0, -9.81;
    MatrixXd l=mpc_qp_plan_n(c,4,x0,cs,fp,xr);
    for(int i=0;i<4;i++) lam4.row(i)=l.row(i);
    // ★heel 메인·toe 보조(사용자): stance 발 수직력을 heel로 편중 → CoP 뒤(heel)=뒤로발라당 저항+전진push.
    static double HMAIN=getenv("HEEL_MAIN")?atof(getenv("HEEL_MAIN")):0.0;   // 0=균등·0.3=heel 80:20 등
    if(HMAIN>0){ int h=stanceLeg*2,t=stanceLeg*2+1; double tot=lam4(h,2)+lam4(t,2);
      double fh=tot*std::min(0.95,0.5+HMAIN), ft=std::max(0.0,tot-fh);
      lam4(h,2)=fh; lam4(t,2)=ft; }   // heel에 하중 몰고 toe는 보조(경하중)
    if(getenv("COP_DBG")){ int h=stanceLeg*2,t=stanceLeg*2+1;
      double fh=lam4(h,2),ft=lam4(t,2), cop=(fh+ft>1)?(fh*(gpos(sph2[stanceLeg])[0]-cc[0])+ft*(gpos(sph[stanceLeg])[0]-cc[0]))/(fh+ft):0;
      std::fprintf(stderr,"  [CoP] pitch%+.1f° heel_fz=%.0f toe_fz=%.0f CoP_x(CoM대비)=%+.3f\n",pitch*57.3,fh,ft,cop); }
  }

  // ── WBIC (MPC lam 추종) ──
  void wbic(int stanceLeg,int sw,const Vector3d&ptgt,const Vector3d&vtgt){
    using namespace bipedwbic; WbicIn in; in.nv=nv; in.nu=nu; in.Kc=1;
    std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
    in.M=Map<Matrix<double,Dynamic,Dynamic,RowMajor>>(Mb.data(),nv,nv);
    in.h=Map<VectorXd>(d->qfrc_bias,nv); in.qv=qvel();
    in.q=Map<VectorXd>(&d->qpos[7],nu); for(int i=0;i<4;i++) in.qc[i]=d->qpos[3+i];
    in.com=com(); in.zref=com_ref_z; in.Jc=jac_com();
    // ★stance 접촉: 점발=tip 1개 / 평발=stance 발 접지 구(heel+toe 다접촉)
    std::vector<std::pair<int,int>> scp;
    if(cmode==1&&has_heel) scp=contact_pts({stanceLeg}); else scp={{sph[stanceLeg],fbody[stanceLeg]}};
    in.Kc=(int)scp.size(); in.contacts.clear(); in.cjac.clear(); in.lam.clear();
    for(auto&cp:scp){ in.contacts.push_back(stanceLeg); in.cjac.push_back(foot_jac_at(cp.first,cp.second));
      if(in_zmp_walk2 && has_heel){   // ★평발 MPC=점별 CoP 힘(heel/toe 구분해 급전) → WBIC가 CoP realize
        int idx = (cp.first==sph2[stanceLeg]) ? stanceLeg*2 : stanceLeg*2+1;  // heel=sph2 / toe=sph
        in.lam.push_back(lam4.row(idx).transpose());
      } else in.lam.push_back(lam.row(stanceLeg).transpose()/(double)scp.size()); }   // (기타)발당→점 균등분배
    in.has_swing=true; in.swing_leg=sw;
    in.Jsw=(cmode==1&&has_heel)?foot_jac_center(sw):foot_jac(sw);
    in.sw_pos=(cmode==1&&has_heel)?foot_center(sw):spos(sw);
    in.sw_ptgt=ptgt; in.sw_vtgt=vtgt;
    if(cmode==1&&has_heel && !getenv("NO_SWORI")){   // ★평발 swing 발 수평 유지
      in.has_sw_ori=true; in.Jsw_rot=foot_jacr(sw);
      double ya=base_yaw(), qy[4]={std::cos(ya/2),0,0,std::sin(ya/2)}, ftgt[4];
      mju_mulQuat(ftgt,qy,foot_home_quat[sw].data());
      double fq[4]; for(int i=0;i<4;i++) fq[i]=d->xquat[fbody[sw]*4+i];
      double oe[3]; mju_subQuat(oe,fq,ftgt); for(int i=0;i<3;i++) in.sw_oerr[i]=oe[i];
    }
    in.Qhome=Map<const VectorXd>(Qcur(),nu); in.drv_peak=Map<VectorXd>(drv_peak8,nu);   // ★모드별 자세(평발=Qflat)
    in.ankle_idx={ankle_idx[0],ankle_idx[1]};
    if(in_zmp_walk2 && cmode==1 && has_heel && !getenv("NO_COP")){   // ★단일지지 발목ZMP: CoP를 CoM 밑에(후방토플 방지)
      in.cop_reg=true; in.cop_comx=com()[0]; in.cop_cx.clear();
      for(auto&cp:scp) in.cop_cx.push_back(gpos(cp.first)[0]);       // 각 접촉점 x
      if(getenv("FLAT_WCOP")) in.W_COP=atof(getenv("FLAT_WCOP"));
    }
    if(in_zmp_walk2 && !getenv("NO_COMX")){   // ★일관 ZMP: 전후(x)만 preview CoM 추종(16cm 밑창=CoP권한). NO_COMX=끄고 MPC+capture만
      in.com_x_track=true; in.com_x_ref=cxr; in.com_vx_ref=vxr;
    } else if(in_zmp_walk2){ /* NO_COMX: com_x 규제 없이 MPC+capture 자연 전진 */
    } else if(in_zmp_walk){       // (구)ZMP 프리뷰: 전후(x)만 CoM 추종(밑창 ZMP). 측방(y)은 capture 발배치.
      in.com_x_track=true; in.com_x_ref=cxr; in.com_vx_ref=vxr;
    } else if(cmode==1&&has_heel){ // (구)평발 보행: 전후 CoM을 com0에 규제
      in.com_x_track=true; in.com_x_ref=com0[0]; in.com_vx_ref=std::cos(base_yaw())*vx_cmd;
    }
    double wank=(cmode==1&&has_heel)?FLAT_WANK:W_ANKLE;   // ★평발=발목 강하게 flat 고정(밑창 유지, 안하면 발목 서서 토플)
    double wori=(cmode==1&&has_heel)?FLAT_WORI:W_ORI;     // ★평발=base pitch 레벨링↑(밑창 ZMP로 pitch 유지)
    in.SW_KP=SW_KP; in.SW_KD=SW_KD; in.W_ORI=wori; in.W_ANKLE=wank; in.W_POST=W_POST;
    static double FLAT_LEAN=getenv("FLAT_LEAN")?atof(getenv("FLAT_LEAN")):0.0;   // ★본체 forward lean(rad)
    in.lean=(in_zmp_walk2&&has_heel)?FLAT_LEAN:0.0;   // 평발 보행만 전방 기울임(뒤로 발라당 상쇄)
    in.W_LAM=(cmode==1&&has_heel)?FLAT_WLAM:W_LAM; in.STANCE_KD=STANCE_KD; in.MU_EFF=MU_EFF; in.LAMZ_MIN=LAMZ_MIN;   // 평발=MPC추종↓, WBIC task 지배
    static double ANK_KP=getenv("ANK_KP")?atof(getenv("ANK_KP")):60.0;   // ★점발 발목 posture PD(env)·기본 60/5(=종전). 점발 stand whip 억제(sim 최적 ~100/20, ζ≈1). deploy 는 16.25kg라 재튜닝.
    static double ANK_KD=getenv("ANK_KD")?atof(getenv("ANK_KD")):5.0;
    static bool _ankp=[&]{ std::printf("[deploy] 발목 posture PD kp=%.0f kd=%.0f%s\n",ANK_KP,ANK_KD,(ANK_KP!=60||ANK_KD!=5)?"  ★튜닝(env)":"  (기본)"); return true; }(); (void)_ankp;
    in.ANK_KP=ANK_KP; in.ANK_KD=ANK_KD;
    set_ctrl_from_tau(wbic_track(in));   // ★전단(관절토크→드라이브)은 한 곳에서만
  }

  // ZMP 레퍼런스(미래 tick fzkk): 초기 DS 리드인(중앙→첫지지발) 후 SS 계단
  void zmp_ref_at(long fzkk,int TICKS_SS,double&zx,double&zy){
    if(fzkk<TICKS_SS){                           // 리드인 DS: 중앙→첫 지지발(leg1=HR)
      double f=(double)fzkk/TICKS_SS, my=0.5*(zaf_y[0]+zaf_y[1]);
      zx=zanchor_x; zy=my*(1-f)+zaf_y[1]*f;
    } else { long fs=(fzkk-TICKS_SS)/TICKS_SS;    // SS: 지지발 위치
      zx=zanchor_x+fs*z_sx; zy=zaf_y[(fs%2==0)?1:0]; }
  }
  // ── ZMP 프리뷰 평발 보행 (clock 기반 footstep + 프리뷰 CoM 궤적) ──
  // ★event-DCM 측방 타이밍 + 프리뷰 전후. 고정clock 대신 측방 sway 동기(timing 충돌 해결).
  void zmp_walk(double dt){
    if(zkk<0){                                   // 보행 시작 초기화
      Vector3d f0=foot_center(0), f1=foot_center(1); Vector3d c=com();
      zanchor_x=f1[0]; zaf_y[0]=f0[1]; zaf_y[1]=f1[1];   // 앵커=현 지지발(HR) x
      pv.reset(c[0],c[1]); cxr=c[0]; vxr=0; zkk=0; prev_ctr=0;
      stance=1; swing=0; t_ss=0; zlead=(long)std::round(T_SS_Z/dt);   // 리드인 DS
      have_liftoff[0]=have_liftoff[1]=false;
    }
    z_sx=vx_cmd*T_SS_Z;
    Vector3d cc=com(); Vector2d vcm=(jac_com()*qvel()).head(2);
    double zz=std::max(cc[2]-std::min(footz(0),footz(1)),0.15), ww=std::sqrt(GVEC/zz);
    // ── 전후 프리뷰: ZMP staircase(현 지지발 x + 미래 공칭T_ss마다 z_sx) → CoM-x 궤적 ──
    if(prev_ctr==0){
      int Np=pv.N; std::vector<double> px(Np),py(Np,0.0);
      for(int j=0;j<Np;j++){ double ta=t_ss + (double)j*PREV_DECIM*dt;
        long sfut = (zlead>0)? 0 : (long)std::floor(ta/T_SS_Z);       // 리드인 중엔 현 지지발 유지
        px[j]=zanchor_x + sfut*z_sx; }
      double cyd,vyd; pv.step(px.data(),py.data(), cxr,vxr,cyd,vyd);
    }
    prev_ctr=(prev_ctr+1)%PREV_DECIM;
    // ── 리드인 DS: CoM을 첫 지지발로 이동(전후=프리뷰) ──
    if(zlead>0){ com_ref_xy<<cxr, 0.7*zaf_y[1]; wbic_stance();
      yaw_hold=base_yaw(); yaw_hold_set=true; yaw_des=base_yaw(); zlead--; zkk++; return; }
    // ── 측방 event-DCM 트리거(sway가 swing측 넘으면 착지) ──
    double midy=0.5*(foot_center(0)[1]+foot_center(1)[1]);
    double xi_y=cc[1]+vcm[1]/ww, sy=(swing==0)?1.0:-1.0;
    bool committed = sy*(xi_y-midy) > TRIG_Y;
    if(t_ss>SS_MIN && (committed || t_ss>SS_MAX)){
      std::swap(stance,swing); t_ss=0; zanchor_x+=z_sx;              // 전후 앵커 전진
      have_liftoff[swing]=false;
      pv.set_state(cc[0],vcm[0],cc[1],vcm[1]);                       // ★프리뷰 전후 실제CoM 재동기(발산방지)
    }
    int support=stance, sw=swing; double s=std::min(t_ss/T_SS_Z,1.0);
    // swing 발: 전후=다음 지지 x(앵커+z_sx)·측방=capture(밑창 좁음)
    double sw_tx=zanchor_x+z_sx;
    double lat=(sw==0)?1.0:-1.0;
    double sw_ty=cc[1]+lat*std::abs(zaf_y[sw])+K_LAT*vcm[1]/ww;
    { double stf=foot_center(support)[1]; double gap=std::min(std::max(lat*(sw_ty-stf),GAP_MIN),GAP_MAX); sw_ty=stf+lat*gap; }
    if(!have_liftoff[sw]){ liftoff[sw]=foot_center(sw); have_liftoff[sw]=true; }
    Vector3d p0=liftoff[sw];
    double gz=std::min(footz(0),footz(1))+m->geom_size[sph[sw]*3];
    double sm=10*s*s*s-15*s*s*s*s+6*s*s*s*s*s, dsm=(30*s*s-60*s*s*s+30*s*s*s*s)/std::max(1e-6,T_SS_Z);
    Vector3d p(p0[0]+(sw_tx-p0[0])*sm, p0[1]+(sw_ty-p0[1])*sm, p0[2]+(gz-p0[2])*sm+4*STEP_H*s*(1-s));
    Vector3d v((sw_tx-p0[0])*dsm,(sw_ty-p0[1])*dsm,(gz-p0[2])*dsm+4*STEP_H*(1-2*s)*dsm);
    if(getenv("ZMP_DBG")&&zkk%25==0) std::fprintf(stderr,
      "  z t%.2f sup%d com=(%.3f,%.3f,%.3f) cxref%.3f t_ss%.2f swT=(%.2f,%.2f)\n",
      zkk*dt,support,cc[0],cc[1],cc[2],cxr,t_ss,sw_tx,sw_ty);
    t_ss+=dt; zkk++;
    if(_k%mpc_decim==0) lam=mpc_grf(support); _k++;
    in_zmp_walk=true; wbic(support,sw,p,v); in_zmp_walk=false;
    yaw_hold=base_yaw(); yaw_hold_set=true; yaw_des=base_yaw();
  }

  // ── 일관 footstep-계획 ZMP 보행 (정통 휴머노이드 파이프라인) ──
  // 스텝 i의 지지발 목표(월드 xy). i=0=시작 지지발. 첫 스텝 반보 램프((i-0.5)).
  Vector2d zw_foothold(long i){
    double sl=vx_cmd*(ZW_TSS+ZW_TDS), sll=vy_cmd*(ZW_TSS+ZW_TDS);
    int leg=(int)((zw_sup+i)&1);
    double a=(i<=0)?0.0:((double)i-0.5);
    return Vector2d(zw_x0+a*sl, zw_yfoot[leg]+a*sll);
  }
  // 현재로부터 te초 뒤 ZMP 레퍼런스(월드 xy). 리드인=중앙→첫지지발 램프, 이후 SS=지지발·DS=다음발판 램프.
  Vector2d zw_zmp_future(double te){
    double tt=zw_t+te;
    if(tt<0){ double f=std::max(0.0,std::min(1.0,(tt+ZW_LEAD)/ZW_LEAD));
      return zw_mid0*(1-f)+zw_foothold(0)*f; }
    long i=zw_i; double T=ZW_TSS+ZW_TDS;
    while(tt>=T){ tt-=T; i++; }
    Vector2d Pi=zw_foothold(i);
    if(tt<=ZW_TSS) return Pi;
    double f=(tt-ZW_TSS)/ZW_TDS;
    return Pi*(1-f)+zw_foothold(i+1)*f;
  }
  // ★검증된 point-foot 측방(step_gait 타이밍 + dcm_target 발배치) 그대로 + 전후만 ZMP preview 주입.
  void zmp_walk2(double dt){
    double ya=base_yaw();
    if(!zw_started){                              // 개시: 상태 초기화 + 체중이동 준비
      Vector3d c=com(); pv.reset(c[0],c[1]); cxr=c[0]; vxr=0;
      zw_pc=0; zkk=0; zw_shift=true; zw_shift_t=0; zw_started=true;
      stance=1; swing=0; t_ss=0; have_liftoff[0]=have_liftoff[1]=false; com0=c.head(2);
    }
    com_ref_z=(czwalk>0)?czwalk:std::min(std::max(com_ref_z,0.36),0.42);
    if(zw_shift){                                 // 첫 지지발로 체중이동(도착 대기)=검증된 walk_init
      Vector3d c=com(); double tgt_y=0.75*foot_center(stance)[1];
      com_ref_xy<<foot_center(stance)[0],tgt_y; wbic_stance();
      yaw_hold=ya; yaw_hold_set=true; yaw_des=ya; zw_shift_t+=dt;
      if(std::abs(c[1]-tgt_y)<0.025 || zw_shift_t>1.0){
        zw_shift=false; t_ss=0; com0=c.head(2); have_liftoff[0]=have_liftoff[1]=false;
        pv.reset(c[0],c[1]); cxr=c[0]; vxr=0; }
      return;
    }
    // ── heading hold(직진) + 복귀목표 이동(측방 dcm_target return항용) ──
    if(std::abs(wz_cmd)>0.02){ yaw_des+=wz_cmd*dt;
      double lag=std::atan2(std::sin(yaw_des-ya),std::cos(yaw_des-ya));
      yaw_des=ya+std::min(std::max(lag,-head_lead),head_lead); yaw_hold_set=false;
    } else { if(!yaw_hold_set){ yaw_hold=ya; yaw_hold_set=true; }
      double err=std::atan2(std::sin(yaw_hold-ya),std::cos(yaw_hold-ya));
      yaw_des=ya+std::min(std::max(err,-head_lead),head_lead); }
    double cya=std::cos(ya),sya=std::sin(ya);
    com0[0]+=(cya*vx_cmd-sya*vy_cmd)*dt; com0[1]+=(sya*vx_cmd+cya*vy_cmd)*dt;
    // ── 전후 ZMP preview(★계획 전진위치 com0 앵커=드리프트 무관, 기준이 앞을 당김) → cxr,vxr ──
    if(zw_pc==0){ int Np=pv.N; std::vector<double> px(Np),py(Np);
      double vxw=cya*vx_cmd-sya*vy_cmd, vyw=sya*vx_cmd+cya*vy_cmd;
      for(int j=0;j<Np;j++){ double tf=(double)j*PREV_DECIM*dt;
        px[j]=com0[0]+vxw*tf; py[j]=com0[1]+vyw*tf; }
      pv.step(px.data(),py.data(), cxr,vxr,cyr,vyr); }
    zw_pc=(zw_pc+1)%PREV_DECIM;
    // ── 검증된 event-DCM 타이밍 + base-frame 발배치 + swing 궤적 (측방 그대로) ──
    int st,sw; double s;
    step_gait(dt,st,sw,s);                            // preview는 계획 전진궤적을 독립 추종(resync 없음=전진 복원 유지)
    if(_k%mpc_decim==0) mpc_grf_flat(st); _k++;       // ★평발 MPC(heel+toe 4점=CoP)=단일지지 pitch 계획
    if(!have_liftoff[sw]){ liftoff[sw]=foot_center(sw); have_liftoff[sw]=true; }
    Vector3d p,v; swing_traj(sw,s,p,v);
    // ★전후 착지 override: capture(드리프트 CoM 추종) 대신 계획 전진위치(com0 앵커)로 심어 backward 악순환 차단.
    { double vxw=cya*vx_cmd-sya*vy_cmd;
      static double FWD=getenv("FLAT_FWD")?atof(getenv("FLAT_FWD")):0.5;   // 착지 전진량(스텝의 배수)
      static double HREF=getenv("HEEL_REF")?atof(getenv("HEEL_REF")):0.0;  // ★heel 기준: 발을 앞으로 HREF만큼(heel이 계획위치=CoM 앞 마진)
      double sw_x=com0[0]+vxw*FWD*T_STEP+HREF;               // 계획 전진 착지 x + heel-ref 전방오프셋
      double ss=std::min(std::max(s,0.0),1.0);
      double sm=10*ss*ss*ss-15*ss*ss*ss*ss+6*ss*ss*ss*ss*ss;
      double dsm=(30*ss*ss-60*ss*ss*ss+30*ss*ss*ss*ss)/std::max(1e-6,T_STEP);
      double x0=liftoff[sw][0];
      p[0]=x0+(sw_x-x0)*sm; v[0]=(sw_x-x0)*dsm; }
    in_zmp_walk2=true; wbic(st,sw,p,v); in_zmp_walk2=false;   // wbic 전후 CoM=cxr(preview)
    if(getenv("ZMP_DBG")&&zkk%25==0){ Vector3d c=com();
      double heelx=gpos(sph2[st])[0], toex=gpos(sph[st])[0];   // st 지지발 heel/toe x
      double* q=&d->qpos[3]; double pitch=std::asin(std::max(-1.0,std::min(1.0,2*(q[0]*q[2]-q[3]*q[1]))))*57.3;
      double cop_frac=(c[0]-std::min(heelx,toex))/std::max(1e-6,std::abs(toex-heelx));  // CoM 폴리곤내 위치(0=heel,1=toe)
      std::fprintf(stderr,"  z2 t%.2f st%d com_x%.3f vx%+.2f pitch%+.1f° | CoM폴리곤위치%.2f(0heel~1toe)\n",
        zkk*dt,st,c[0],d->qvel[0],pitch,cop_frac); }
    zkk++;
  }

  void control(double dt){
    dt_ctrl = dt;                     // ★적분항이 쓴다(wbic_stance 는 dt 를 안 받는다)
    double ya=base_yaw();
    if(trans_on){ do_transition(dt); return; }   // ★1점/2점 전환 굴림 재생 중
    // ★2점 평발: 정지=정적 양발지지(밑창 ZMP). 이동명령=평발 동적 보행(아래 게이트, wbic 다접촉).
    if(has_heel && cmode==1){
      bool flat_walk_en = getenv("FLAT_WALK")!=nullptr;   // ★2점=서기 전용(기본). 보행은 adaptive-timing 프리뷰 완성 후 활성
      bool moving = flat_walk_en && (std::abs(vx_cmd)>0.02 || std::abs(vy_cmd)>0.02 || std::abs(wz_cmd)>0.02);
      if(!moving){                                   // 정지(또는 보행 미활성)=정적 양발지지
        com_ref_z=std::min(std::max(com_ref_z,0.36),0.42);   // ★평발 정적 높이 실현범위 클램프(발 flat 유지 기하제약)
        Vector3d fc=0.5*(foot_center(0)+foot_center(1)); com_ref_xy=fc.head(2);
        wbic_stance();
        t_ss=0; com0=com().head(2); have_liftoff[0]=have_liftoff[1]=false; yaw_hold=ya; yaw_hold_set=true;
        walk_init=true; zkk=-1; zw_started=false;    // 다음 보행개시 재무장(reactive weight-shift · ZMP 재초기화)
        return;
      }
      if(getenv("ZMP_WALK")){ if(getenv("ZMP_OLD")) zmp_walk(dt); else zmp_walk2(dt); return; }  // ★ZMP 프리뷰 평발 보행
      if(walk_init){                                 // ★보행개시: 첫 스텝 전 CoM을 첫 stance 발쪽 측방 이동
        Vector3d sf=foot_center(stance);             // stance=1(HR) 쪽으로 체중 이동(지지면끝까지 못가니 75%)
        double tgt_y=0.75*sf[1];
        com_ref_xy[0]=sf[0]; com_ref_xy[1]=tgt_y;
        wbic_stance(); walk_init_t+=dt;
        Vector3d c=com();
        if(std::abs(c[1]-tgt_y)<0.03 || walk_init_t>0.4){   // 지지발쪽 도달 or 시간초과→스텝 시작
          walk_init=false; walk_init_t=0; t_ss=0; com0=c.head(2);
          have_liftoff[0]=have_liftoff[1]=false; yaw_hold=ya; yaw_hold_set=true; }
        return;
      }
      if(czwalk>0) com_ref_z=czwalk;               // 평발 보행 CoM 높이(튜닝)
    }
    if(std::abs(wz_cmd)>0.02){                    // ★선회: 명령 적분 + 리드 클램프(폭주방지)
      yaw_des+=wz_cmd*dt;
      double lag=std::atan2(std::sin(yaw_des-ya),std::cos(yaw_des-ya));
      yaw_des=ya+std::min(std::max(lag,-head_lead),head_lead); yaw_hold_set=false;
    } else {                                      // ★선회 외 전부(정지/전후진/측방): heading latch(base0.50서 측방도 안정)
      if(!yaw_hold_set){ yaw_hold=ya; yaw_hold_set=true; }
      double err=std::atan2(std::sin(yaw_hold-ya),std::cos(yaw_hold-ya));
      yaw_des=ya+std::min(std::max(err,-head_lead),head_lead);
    }
    double cya=std::cos(ya),sya=std::sin(ya);   // ★복귀목표 이동=실제 base yaw 기준(base-relative)
    com0[0]+=(cya*vx_cmd-sya*vy_cmd)*dt; com0[1]+=(sya*vx_cmd+cya*vy_cmd)*dt;
    int st,sw; double s; step_gait(dt,st,sw,s);
    if(_k%mpc_decim==0) lam=mpc_grf(st);
    _k++;
    if(!have_liftoff[sw]){ liftoff[sw]=foot_center(sw); have_liftoff[sw]=true; }
    Vector3d p,v; swing_traj(sw,s,p,v);
    wbic(st,sw,p,v);
  }
};
