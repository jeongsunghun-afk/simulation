# 02_Leg 17-DOF C++ 제어기 — 파라미터 레퍼런스

`cpp/src/` 기준. env 오버라이드: **S**=trot_sim.cpp, **V**=trot_view.cpp, **A**=apply_env_gains(trot_controller.hpp).
실사용값은 멤버 기본값이며 env로 덮어쓸 수 있음. 뷰어(GUI)는 CMDFILE(JSON)로 일부를 런타임 갱신.

---

## ★ 자주 쓰는 튜닝 knob (요약)

| knob | 기본값 | env | 역할 |
|---|---|---|---|
| `REAR_ANKLE` | **−0.3** | REAR_ANKLE | 뒷발목 각(축별 τ·ω 최적. 발목 ω 병목 해소) |
| `FRONT_ANKLE` | −0.7 | FRONT_ANKLE | 앞발목 각 |
| `base_z0` / `body_h` | 0.5234 | BASE_Z0 / BODY_H | 서기 몸통 높이 |
| gait | trot | GAIT / GUI | trot·walk·run·gallop 프리셋 |
| `V` `VY` `WZ` | 0.3·0·0 | TROT_V·VY·WZ | 전진·측방·선회 명령 |
| `step_h` | 0.10 | TROT_STEPH / GUI | 스텝 발 높이(프리셋 위 override) |
| `raibert_k` | 0.5 | RAIBERT_K / GUI | 전방 reach(0.5=표준 중립) |
| `steer` | 0 | TROT_STEER / GUI | 자동차식 조향각 δ(허리 핸들) |
| GEARBOX | ON | GEARBOX=0 끔 | 반사관성·감쇠·마찰(발 flail 억제) |
| `perceptive` | ON | PERCEPTIVE=0 끔 | 지형인지(mj_ray 착지높이+몸통높이 적응, 계단/험지) |

## 지형인지 / perceptive

blind(평지 가정)→terrain-aware. mj_ray 하향캐스트로 지형표면 z 샘플. 평지=무영향(회귀 falls=0 동일).

| 파라미터 | 기본값 | env | 의미 |
|---|---|---|---|
| perceptive | true | PERCEPTIVE=0 끔 | ①스윙 착지 z=지형높이 ②몸통 목표높이=지형+com_h0 추종 |
| PCV_CLR | 0.04 | PCV_CLR | 상향 스텝 추가 스윙 클리어런스(라이저 헛디딤 방지) |
| ray_z0 | 0.40 | — | mj_ray 시작 z(벨리 아래·지형 위, 자기충돌 회피) |

효과: 종합코스(마찰→험지→계단) blind=계단서 낙상(x8.8) → perceptive=완주(x13.7·tilt2.9°·falls0).

---

## 기립자세 / posture

| 파라미터 | 기본값 | env | 의미 | 위치 |
|---|---|---|---|---|
| base_z0 | 0.52(17dof 런치 0.5234) | BASE_Z0 (A) | 서기 몸통 높이 | quad_control.hpp:27 |
| REAR_ANKLE | −0.3 | REAR_ANKLE (A) | 뒷발목 각. 스윕 최적(발목 ω 155%→100%·calf τ76%·falls0). REAR_ANKLE 지정 시 FRONT도 동일화 | quad_control.hpp:27 |
| FRONT_ANKLE | −0.7 | FRONT_ANKLE (A) | 앞발목 각 | quad_control.hpp:27 |
| body_h | 0.5234 | BODY_H (S) / GUI | 서기 높이 슬라이더(q_home 목표) | trot_controller.hpp:46 |

`crouch_home`(quad_control.hpp:114): z=base_z0서 발 XY 유지·무릎 굽힘 IK → q_home·com_ref·foot_hip_off·foot_gz0. 발목은 R/FRONT_ANKLE로 고정 후 hip/thigh/calf만 IK.

## 게이트 / gait — set_gait 프리셋 (trot_controller.hpp:96–99)

| 게이트 | gp_T | gp_SWF | off[HL,HR,FL,FR] | raibert_k | step_h |
|---|---|---|---|---|---|
| walk | 0.7 | 0.25 | 0.25,0.75,0.5,0 | 0.5 | 0.10 |
| trot(기본) | 0.5 | 0.5 | 0,0.5,0.5,0 | 0.5 | 0.10 |
| run(고속) | 0.40 | 0.5 | 0,0.5,0.5,0 | 0.5 | 0.08 |
| gallop | 0.35 | 0.55 | 0,0.05,0.55,0.5 | 0.8 | 0.10 |

override: `GAIT`(S)·`TROT_T`/`TROT_SWF`/`TROT_STEPH`/`RAIBERT_K`(S, 프리셋 뒤 적용). config 상수 TC_KCAP=0.16(Raibert 피드백)·TC_RAICLIP=0.25(발배치 클립)·TC_WARMUP=0.6·TC_ACC=0.6(명령 슬루).

## 발배치 · whip

| 파라미터 | 기본값 | env | 의미 |
|---|---|---|---|
| auto_whip | true | AUTO_WHIP=0 / GUI | 속도트리거 whip on/off |
| whip_v0 / whip_v1 | 0.8 / 1.6 | — | whip 선형증가 속도구간 |
| whip_hi | 2.0 | — | 저속(매끈) swing_w |
| whip_lo_f / whip_lo_r | 0.1 / 0.6 | SWING_W_F / SWING_W_R (S,V) | 앞/뒤 고속 whip 목표(작을수록 강함) |
| POS_HOLD | true | POS_HOLD=0 / GUI | 정지 위치홀드 |
| ALIP | false | ALIP=0 | 각운동량 발배치 보정(무효 확인) |

## WBIC 게인

| 파라미터 | 기본값 | env | 의미 |
|---|---|---|---|
| w_ori | 5.0(17dof 20) | W_ORI (A) | 자세(roll/pitch) task 가중 |
| w_yaw | 0.0 | W_YAW (A) | yaw 홀드(0=MPC 담당, 최적) |
| W_AM / KD_AM | 0.0 / 8.0 | W_AM / KD_AM (A) | 각운동량 보상 가중 / 감쇠 |
| STANCE_KD | 20.0 | STANCE_KD (S) | stance 접촉속도 감쇠(slip↓). ★하이브리드(기본) 추적 경로는 STANCE_KD_TRACK 사용 |
| WBIC 구성 | 2 | WBIC_MIT | ★2026-09-30 0=종전 가중QP · 1=MIT 전체(선택, 기본 MIT 게인 자세·몸통 100/10·스윙 500/10 — 드라이버 PD(PLANT) 있으면 실기조건 0낙상, 토크만이면 걷기 낙상. ※09-30 "8ms 붕괴"는 A 게인 조건, 10-06 정정) · **2=하이브리드(기본)**: 가중QP + 접촉·스윙 J̇q̇ + KinWBC 계획(q_des·q̇_des) |
| MIT 과제 게인 | 모드별 | MIT_ORI_KP/KD · MIT_COM_KD · MIT_Z_KP/KD · MIT_SW_KP/KD | ★2026-10-06 KinWBC/MIT 과제 게인. 기본 WBIC_MIT=1 → 100/10·10·100/10·500/10(MIT) · =2 → 150/20·25·200/25·2400/110(A). 종전엔 하드코딩(SW_KP/KD 는 HQP 전용) |
| STANCE_KD_TRACK | 0.0 | STANCE_KD_TRACK | 하이브리드 추적(wbic_track) 접촉 K_D. 0=폐기(J q̈ = −J̇q̇) |
| 드라이버 추종 | off | DRV_TRACK / TRK_KP_Q / TRK_KD_Q | τ_ff + kp(q_des−q) + kd(q̇_des−q̇). sim 권장 20 Nm/rad · 1 Nm·s/rad. ⚠실기 kp 단위(real_hal) 확인 전 off |
| MU | 0.6 | MU (A) | 마찰콘 μ(MPC 동시) |
| motor_curve | false | MOTOR_CURVE (A) | 토크-속도 곡선 |

하드코딩 task PD(참고): stance CoM (120,120,200)/(20,20,25)·자세 150/20(w5)·posture 60/5. swing 2400/110(w90)·z 200/25(w150).

## MPC (Di Carlo SRBD, quad_control.hpp:243–247)

N=14, dt=0.02. Q대각(roll,pitch,yaw/px,py,pz/ω/v/g)=[200,200,100, 0,0,200, 0,0,1, 10,10,1, 0]. R=1e-6(per-foot GRF). μ=MU·0.707. λz∈[1, 2·mg].

## 기어박스 · 모터물리

| 파라미터 | 기본값 | env | 의미 |
|---|---|---|---|
| gear[4] hip,thigh,calf,foot | 7.0, 7.0, 10.5, 14.0 | — | 감속비 N. ★허리(FB_waist)는 미매칭→gi=0=hip **7:1**(실제와 일치, 의도됨) |
| GEARBOX | ON | GEARBOX=0 끔 | 반사관성 armature=Irot·N² + 감쇠 + 마찰 |
| Irot | 1e-4 | ROTOR_I | 로터 관성(★실측 대기) |
| jdmp / jfrc | 0.1 / 0.5 | JDAMP / JFRIC | 기어 점성감쇠 / 마찰(★실측 대기) |
| gmul | 1.0 | GEAR_HIP/THIGH/CALF/FOOT | 재기어 배율. ★배포=GEAR_FOOT=0.5714(foot 14→8:1, peak168→96·ω→25.9) |
| w_limit | 207/(gear·gmul) | (파생) | 무부하 속도한계(MOTOR_CURVE용) |

모터 peak: hip/thigh 84, calf 126, **foot 96(8:1 재기어)** Nm. 상세=메모리 02leg-motor-spec.

## 조향 / steering

| 파라미터 | 기본값 | env | 의미 |
|---|---|---|---|
| steer(δ) | 0 | TROT_STEER (S,V) / GUI | 자동차식 조향각. Ackermann R=축거/tanδ |
| wheelbase | ≈0.61(런타임) | — | 축거 L(앞뒤 힙 x거리) |
| yaw-rate 캡 | ±0.9 | — | understeer(고속 tight turn 낙상방지) |

## 허리 / waist (17-DOF)

| 파라미터 | 기본값 | env | 의미 |
|---|---|---|---|
| WAIST_W | 80.0 | WAIST_W (A) | 허리 홀드 가중 |
| WAIST_KP / WAIST_KD | 150 / 20 | WAIST_KP / WAIST_KD (A) | 허리 강홀드 PD |
| waist_steer | 0.4 | WAIST_STEER (S,V) | 허리 lean 보조 게인(핸들만 구동) |
| waist_cap | 0.20 | — | 허리 lean 캡 ±0.20rad(≈11°) |

## 앉기 · 기립 / sit · getup

**개-앉기(haunch) 시드** (quad_control.hpp:28–29): HAUNCH_THIGH −1.0·CALF 1.2·FOOT −0.3·HOCK_Z 0.019·FRONT_REACH −0.22·FOOT_LAND −1.2 (모두 S,V env).
**스케줄** (trot_controller.hpp): HAUNCH_Z 0.30·FOLD_RATE 0.35(ease-out ~2.9s)·UNFOLD_Z 0.40·SIT_POSTURE_W 40·HAUNCH_PITCH 0.50·SIT_KP 90·SIT_SLEW 0.6.
**★sit 진입 방향 구분** (2026-07-10): `sit_from_below` 1회 latch — **위(서기, bz≥SIT_Z−0.02)** 진입=기존 0.32 crouch 하강 후 fold / **아래(눕기)** 진입=0.32 오버슈트 없이 현재 자세서 곧바로 haunch로 morph(base는 다리기하로 ~0.25만 자연상승). "일어섰다 다시앉기" 제거.
**★sit fold 속도 배수**: `SIT_BELOW_SPEED` 2.5(눕기→앉기, 엉덩방아 위험無→빠른 fold+얕은 tail)·`FRONT_PULL_SPEED` 2.2(서기→앉기 **앞다리만** 조기 끌어당김=haunch_fold×배수, 뒤 엉덩이 착지 pace는 유지). 둘 다 코드상수(env無).
**★눕기(ground/stand_down) 저자세 조각** (GUI 실시간 슬라이더+env): GROUND_LIE_Z 0.226·GROUND_REAR_FOOT −1.15·GROUND_FRONT_FOOT −0.5·GROUND_FRONT_THIGH −0.24·GROUND_FRONT_CALF −0.4 (기본값=뷰어 라이브튜닝 확정, base_z≈0.166 belly-lie). env=GROUND_LIE_Z/GROUND_REAR_FOOT/GROUND_FRONT_THIGH/GROUND_FRONT_CALF, GUI cmd=g_lie_z/g_rear_foot/g_front_thigh/g_front_calf. ★PD-fold 안정 바닥 z≈0.166(더 낮추면 뒤 붕괴, 동적궤적 필요=[[haunch-sit-posture]]).
**★앉기→서기 튕김 조절**: `GETUP_TRAJ_KP` 80(개-앉기 기립 궤적추종 강성=튕김 힘, **↓=부드럽게**)·`GETUP_TRAJ_KD` 6(↑=튕김 감쇠)·getup_dt 0.01. 크라우치-앉기 기립은 `SGU_KP` 120. 모두 env 조절 가능(S,V).
**스크립트 기립(SGU_*)**: KICK_T 0.5·FB_THIGH −0.55·FB_CALF 1.20·SLEW 1.5·KP 120·GATHER_Z 0.24·DONE_TILT 22·WALKOUT_V 0.6·HANDOFF_Z 0.34 (모두 S env).
**모드 상태**: GROUND_Z 0.18·GETUP_TRIG 0.32·GETUP_DONE 0.40·JOINT_SLEW 1.5·HRATE 0.3.
**★점프 — aligator OCP + WBIC 추종**(2026-07-13, `ocp/jump_solver.hpp`·`wbic_jump`): crouch 정착 시 별도 스레드(std::async)로 aligator SolverProxDDP solve(~331ms, 1kHz 루프 안 멈춤). JUMP_VX 0.6(전방 이륙속도)·JUMP_MAXIT 8(RTI)·3상 push22/flight~34/land40·dt0.01·하드 토크BoxConstraint(±Peak)·마찰추. wbic_jump 추종(push·land): CoM 가속ff kp_lin120/kd_lin22·자세 w_ori8·관절 kp_j160/kd_j12 w_j2·λ reg w_lam0.1, flight=관절PD. ★crocoddyl(soft)→aligator(하드제약)=B·C 통일. gen_jump.sh로 파일 fallback도 생성.
**★기립 — C++ 자립 gather**(`gen_getup`): 현재 sit qpos=q_sit→G(gather)→A1(HL)→A2(HR)→B(상승) 순수 MuJoCo IK(Python 파일 의존 제거). phaseA=관절PD(GETUP_TRAJ_KP80/KD6, WBIC 아님·준정적이라 정답)·phaseB=wbic_stance. ★wbic_jump는 기립엔 무익(lean과 싸움, 검증).

## 실행 / 진단 env

| env | 기본 | 의미 |
|---|---|---|
| MODE | move | move/stand_up/stand_down/sit/off (S) |
| MODE2 + SWITCH_T | — | t>SWITCH_T서 1회 모드전환(getup 검증, S) |
| STEPS | 3000 | 헤드리스 스텝 수 (S, argv2) |
| RATE / CMDFILE / STATE_PUB | 1.0 / — / /tmp/quad_state.json | 뷰어 배속·GUI 명령·상태발행 (V) |
| JSTAT | — | 관절별 τ·ω peak/RMS(정착후) 출력 (S) |
| QHDBG·SITDBG·SLIPLOG·GRFLOG·DUMP_QPOS | — | 진단 로깅 (S) |

---

## Python ↔ C++ 파리티

Python(`quad_mpc_wbic_17dof.py`)과 C++(`cpp/src/`)는 같은 알고리즘·같은 MJCF를 쓰며, 아래는 정합 상태.

**동일 (검증됨)** — 게인(C++가 17dof/허리모델 자동감지 → w_ori20·W_AM12·KD_AM24·FRONT_ANKLE−0.5·base_z0 0.5234를 Python 기본과 동일 적용) · REAR_ANKLE−0.3 · walk step_h 0.10 · MPC(N14·dt0.02·Q·R) · 기어박스(gear·ROTOR_I·JDAMP·JFRIC·허리 7:1) · 조향(waist_steer0.4·cap0.20·yaw캡0.9) · 허리(WAIST 80/150/20) · whip·raibert·KCAP·POS_HOLD · **맵**(Python도 `MJCF` env로 같은 지형 씬 로드) · **perceptive 발 착지높이**(gz+지형).
검증: 종합코스 완주 — Python x14.3·tilt3.2° / C++ x13.9·tilt3.7°, **둘 다 falls=0**. 평지도 둘 다 falls=0.

**남은 차이 (둘 다 정상 동작, 강제 동일화는 위험 대비 이득 작아 보류)**

| 항목 | Python | C++ | 비고 |
|---|---|---|---|
| perceptive 몸통높이 | 4-hip 평균 → MPC+WBIC z-task 양쪽 | base 1점+슬루 → MPC만(WBIC 미공급) | 코드베이스 디테일차. Python 3.2°/C++ 3.7° 둘 다 완주. C++서 4힙+WBIC공급 재현 시 6.1°로 악화(충실 이식 finicky) |
| STANCE_KD(터치다운 baumgarte) | trot 경로 없음(=0) | 종전 20 · 하이브리드 기본 0 | C++만. trot 접촉등식 b=−KD·cjac·q̇ (−J̇q̇) |
| ★하이브리드 WBC(2026-09-30) | 없음(가중QP 그대로) | 기본: J̇q̇ + KinWBC 계획(+DRV_TRACK 시 드라이버 PD) | C++만. sim 실기수준(EST+2+6ms+노이즈) 전 조건 0낙상. Python 미이식 |
| gallop 게이트 | 없음 | 있음(T0.35 등) | 프로젝트 방향상 미사용(leg-heavy 불가) |
| walk foot-lock | LOCK=0.35 late-commit | 없음(매틱 reactive) | Python만. walk 둘 다 falls=0 |

## 앉기→서기 튕김 조절
`GETUP_TRAJ_KP`(개-앉기 기립 궤적추종, 기본 120 — **↓=부드럽게**)·`GETUP_TRAJ_KD`(4, ↑=감쇠)·`SGU_KP`(크라우치-앉기, 120). C++ env(S,V)로 조절, 이번에 훅 추가.

## 테스트 지형 씬 (make_terrains.py 생성)

`<include>`로 로봇 재사용 + 지형만 얹음. 로봇은 원점서 +x 전진 → 지형 진입.

| 씬 | 내용 | walk v0.5 결과 |
|---|---|---|
| quad_terrain_stairs.mjcf | 오름6·랜딩·내림6 계단(rise 4cm·depth 28cm) | 계단 올라탐 z0.55·tilt3.0°·falls0 |
| quad_terrain_rough.mjcf | 불규칙 높이 블록필드(84개, h 1~10cm) | tilt1.4°·falls0 |
| quad_terrain_friction.mjcf | ice(μ0.2)·보통(μ1.3)·고마찰(μ2.5) 레인 | ice 통과·tilt1.5°·falls0 |

실행: `GEAR_FOOT=0.5714 ./cpp/build/trot_view quad_terrain_stairs.mjcf` (또는 GUI 연동 시 CMDFILE 추가).
지형 조정: make_terrains.py 편집 후 재생성.
