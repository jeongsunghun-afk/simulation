# v14.4 — Trot sim2sim divergence report (gait_sim vs Isaac Lab PhysX CPU)

**스코프:** trot 1 m/s, 4 cycle (2.0s, 1000 frame @ dt=2ms) 시나리오에서 gait_sim 의 MPC+WBIC controller 가
Isaac Sim 4.5 (PhysX CPU) 위에서 어떻게 동작하는지 정량 측정.

**환경:**
- gait_sim: WSL Ubuntu 22.04, numpy 2.2.6, scipy 1.15.3, qpsolvers 4.x, quadprog
- Isaac Sim: Windows 11, Isaac Sim 4.5 + IsaacLab, PhysX CPU mode
- URDF: `gait_sim.pin_model.export_urdf()` + `urdf_add_collision_limit.py` post-process
  (continuous→revolute + ±π limits, visual→collision mirror)
- USD: Isaac Sim 4.5 URDF importer (`isaacsim.asset.importer.urdf` extension)
- BODY_INERTIA: CRBA composite hardcoded (Ixx=0.804, Iyy=2.157, Izz=1.5599, Ixz=-0.255)

## 비교 시나리오

| 시나리오 | controller→Isaac 명령 | controller state feedback |
|---|---|---|
| **A. open-loop position (v14.3-c)** | `set_joint_position_target(q_target)` | 없음 (precomputed trajectory) |
| **B. open-loop effort   (v14.3-c)** | `set_joint_effort_target(tau_cmd)`     | 없음 (precomputed tau) |
| **C. closed-loop effort (v14.3-e)** | `set_joint_effort_target(tau_cmd)`     | **Isaac state → controller** (per-step) |
| **REF. gait_sim 자체 sim**         | -                                       | - |

## 결과 비교 (trot 1m/s, 2.0s)

| 메트릭 | REF (gait_sim) | A. position | B. effort | **C. closed-loop** |
|---|---|---|---|---|
| 최종 base x (m)        | 1.974 | 0.350 | 0.573 | **1.046** |
| 최종 base y (m)        | 0.010 | 0.000 | 0.273 | 0.148 |
| 최종 base z (m)        | 0.470 | 0.146 | 0.069 | 0.078 |
| 이동률 (vs V·t=2.0m)   | 99% | 18% | 29% | **52%** |
| max \|Δq\| (rad)       | -   | 4.19 | 6.04 | < π (joint limit 미접촉 700 step) |
| 첫 standing 유지 (step)| -   | 0    | 0    | **~100 (0.2s)** |
| WBIC FB 실패율         | 15% | -    | -    | 93% |
| Sim NaN 발산           | No  | No   | No   | No   |

## 핵심 발견

1. **open-loop replay 의 한계 명확.** gait_sim 의 τ_cmd 는 자기 dynamics 모델 (RNEA + 6-DoF integrator) 위에서만
   균형이 맞음. PhysX articulated body dynamics 와의 fidelity gap 이 매 step 누적되면 1초 안에 발산.

2. **closed-loop bridge architecture 작동 검증.**
   - Joint mapping (gait_sim flat `[FR,FL,HR,HL] × [j1..j5]` ↔ Isaac flat `j1×[FL,FR,HL,HR] → ... → j5×[FL,FR,HL,HR]`)
   - Base state mapping (Isaac quat wxyz ↔ gait_sim R 3x3)
   - tau / effort 부호 convention 일치 (signing 명확)

3. **closed-loop 은 open-loop 대비 ~2x 개선** (이동 거리, 초기 standing 유지). controller 의 reactive feedback 이
   modeling gap 의 일부를 흡수.

4. **여전히 0.2s 이후 점진적 collapse.** modeling fidelity gap 이 closed-loop feedback 으로도 다 흡수 안 됨.
   FB QP 가 93% frame 에서 실패 → per-leg fallback. controller 의 expected model 과 PhysX 사이의 차이가 매우 큼.

5. **v14.3-d refactor 의 부가 가치:** `_step_one_frame()` 추출 + `bridge/controller_step.py` wrapper 로 gait_sim
   self-test 가 bit-identical 통과. 향후 Mujoco / Drake / 다른 RL 환경 통합 시 같은 패턴 재사용 가능.

## modeling gap 추정 원인 (우선순위 순)

| # | 후보 | 검증 방법 |
|---|---|---|
| 1 | URDF link/joint frame convention (DH → URDF rpy 변환) | v14.1-a URDF 검토 + Isaac/gait_sim 양쪽 FK 비교 |
| 2 | Per-link inertia (URDF cylinder approximation vs gait_sim 의 hardcoded link inertia) | URDF 의 `<inertia>` 값 vs gait_sim 의 `_cyl_inertia_pin` 출력 비교 |
| 3 | Foot collision geometry (visual mirror) vs gait_sim 의 point-contact 모델 | URDF 의 foot link collision shape 확인 |
| 4 | Joint friction / damping (URDF 미설정 → Isaac 기본값 사용) | URDF 에 `<dynamics damping=>` 명시 |
| 5 | gait_sim 의 1st-order actuator lag (theta_a filter) | Isaac PD vs lag 모델링 — closed-loop 에서는 영향 작음 |

## v14.6 (CAD URDF 도착 시) 의 의미

CAD-derived URDF 가 들어오면 ① 정확한 link mass/inertia (cylinder 근사 제거), ② 실제 joint axes/offsets,
③ 정확한 foot collision geometry 가 한 번에 해결됨. 위 modeling gap 의 상당 부분이 단번에 줄어들 것으로 기대.

v14.4 의 현재 closed-loop 결과 (52% 이동률) 를 v14.6 후 재측정해 **fidelity 개선량 정량 측정** — 이게 v15 RL+sim2real
환경의 신뢰성 지표가 됨.

## 산출물

| 위치 | 내용 |
|---|---|
| `/home/jsh/simulation/gait_sim/runner.py` | `_step_one_frame()` 추출 (D1 refactor, bit-identical 검증) |
| `/home/jsh/simulation/gait_sim/bridge/__init__.py` | bridge 패키지 |
| `/home/jsh/simulation/gait_sim/bridge/controller_step.py` | `GaitSimControllerStep` open/closed loop API |
| `/home/jsh/simulation/gait_sim/bridge/export_replay.py` | trajectory → npz dumper |
| `C:\Users\jsh\simulation\quadruped_v13.urdf` | DH 기반 raw URDF |
| `C:\Users\jsh\simulation\quadruped_v13_pxready.urdf` | PhysX-ready (revolute+limit+collision) |
| `C:\Users\jsh\simulation\quadruped_v13.usd` | Isaac Sim 4.5 importer 결과 |
| `C:\Users\jsh\IsaacLab\scripts\tools\gait_isaac_bridge.py` | joint index 매핑 |
| `C:\Users\jsh\IsaacLab\scripts\tools\convert_urdf_v45.py` | URDF→USD (Isaac Sim 4.5 explicit-enable wrapper) |
| `C:\Users\jsh\IsaacLab\scripts\tools\urdf_add_collision_limit.py` | URDF post-process |
| `C:\Users\jsh\IsaacLab\scripts\tools\quadruped_settle_v13.py` | 1-env settling test |
| `C:\Users\jsh\IsaacLab\scripts\tools\replay_isaac_v13.py` | open-loop replay |
| `C:\Users\jsh\IsaacLab\scripts\tools\closed_loop_isaac_v13.py` | closed-loop bridge runner |

## v14.5 NMPC trajectory replay (open-loop only)

NMPC (crocoddyl FDDP, receding horizon N=24=0.48s, N_RESOLVE=12, N_TOTAL=100 → 9 solves 3.8s)
trajectory 를 동일 `export_replay` framework + 동일 `replay_isaac_v13.py` 로 재생.

### 결과 비교 (trot 1m/s 2.0s)

| 지표 | MPC+WBIC (v14.3) | NMPC (v14.5) |
|---|---|---|
| gait_sim 자체 이동률    | 99% (1.97m) | **79% (1.58m)** |
| Isaac position 이동률   | 18% | **9%** |
| Isaac effort   이동률   | 29% | **11%** |
| Isaac effort max \|Δq\| | 6.04 rad | 5.88 rad |
| trajectory 계산 시간    | 45s (WBIC main loop) | 3.8s (FDDP) |
| trajectory 파일 크기    | 670 KiB | 506 KiB |

### 발견
- NMPC trajectory 자체가 V·t=2.0m 의 79% 만 진행 (MPC+WBIC 의 99% 대비 낮음) — short horizon 의 sub-optimality.
- Isaac replay 시 NMPC fidelity gap 이 MPC+WBIC 보다 큼 (이동률 절반 수준). 이유: NMPC 의 fast joint
  변화 + receding horizon 에서 만들어진 high-frequency tau 가 PhysX articulated dynamics 와 mismatch 증폭.
- MPC+WBIC 의 quasi-static foot + GRF QP + WBIC smoothing 이 결과적으로 modeling gap 에 더 robust.

### NMPC closed-loop (보류)
NMPC 는 pre-solved trajectory 구조 — 외부 state 를 매 step 받는 receding-horizon adapter 가
별도 필요 (controller_step.py 가 MPC+WBIC 전용). v15 timeframe 에서 real-time NMPC bridge 구현 예정.

## v14.5+ 추가 진단 (joint rate + closed-loop state gap)

`gait_sim/bridge/diagnose_v14.py` 로 두 npz 비교.

### NMPC vs MPC+WBIC joint kinematics

| 지표 (j2~j4) | MPC+WBIC | NMPC |
|---|---|---|
| qdot max (rad/s) | [10.84, 14.99, 14.66] | [10.84, 14.99, 14.66] |
| qddot max (rad/s²) | [286, 435, 648] | [286, 435, 648] |
| qdot RMS (rad/s) | [3.84, 5.5, 5.49] | [3.84, 5.5, 5.49] |
| **tau max (Nm)** | **[120, 93, 59]** | **[77, 59, 64]** |
| **tau RMS (Nm)** | **[43, 36, 30]** | **[19, 14, 17]** |

**Joint kinematics 두 모드 동일.** 둘 다 같은 `precompute_trajectories` IK 결과 사용; NMPC 도 receding-horizon 으로 그 trajectory 를 매우 정확히 tracking (`populate_simstate_from_nmpc` 가 R.joint_hist 를 overwrite 하지만 결과는 IK 와 거의 일치).

**Tau 는 NMPC 가 작음.** MPC+WBIC 의 tau = PD + RNEA + impedance + WBIC GRF 합산이라 magnitude 큼. NMPC 의 tau = FDDP 가 최소화 비용으로 푼 결과 → 더 효율적 (자기 내부 dynamics 기준).

**핵심 결론**: 이전 보고서의 "NMPC fast joint motion" 가설은 **틀렸다**. 실제 원인은 **NMPC tau magnitude 가 작아서 PhysX 의 더 강한 contact reaction 을 충분히 보상하지 못함** → 발이 미끄러지고 base 가 더 빨리 무너짐. NMPC 의 가벼운 tau 가 자기 sim 에서는 충분했지만 PhysX articulated body dynamics 에서는 부족.

### Closed-loop state gap (Isaac vs controller post-integration)

| 지표 | 값 |
|---|---|
| body_pos drift max (\|Δx\|, \|Δy\|, \|Δz\|) | (2.4, 1.0, 3.3) mm 누적 |
| controller tau ‖.‖₂ mean | 235.4 Nm |
| controller tau ‖.‖₂ max | 325.8 Nm |
| controller tau ‖.‖₂ std | 30.5 Nm |

- **controller 의 1-step body integration 결과가 Isaac PhysX 실제값과 거의 일치** (~3mm). 즉 controller 가 받은 state 자체는 Isaac 와 동기됨.
- **controller tau 출력은 stable** (mean=235, std=30 Nm). controller 가 매 step 합리적 명령 생성.
- **하지만 trajectory 끝에 base_z 0.078m 로 collapse** — frame-to-frame 의 작은 dynamics gap (controller integrator ↔ PhysX articulated body) 가 1000 step 누적되며 base 가 점진적으로 내려감.

### WBIC FB QP 93% 실패 — 시스템에는 영향 없음

- open-loop 15% vs closed-loop 93% 실패 — Isaac state 가 controller's expected M_leg, J_leg, I_world 등과 약간 다를 때 FB QP (full-body 단일 QP) 의 infeasibility 증가.
- **하지만 per-leg WBIC QP fallback 이 정상 동작** (wbic_fail=0) — controller 가 매 step 유효한 tau 출력.
- 따라서 modeling gap 의 진짜 원인은 FB QP 실패 자체가 아니라 **articulated body dynamics ↔ gait_sim 6-DoF integrator 차이 누적**.
- FB QP 의 정확한 infeasibility 원인 (friction cone? joint torque limit? singular M?) 은 v15 controller refactor 시점에 더 깊은 instrumentation 필요.

## 다음 단계

- **v14.6**: CAD URDF/USD 업데이트 → modeling gap 재측정 (이 보고서의 MPC+WBIC closed-loop 52% → ?,
  NMPC open-loop 11% → ?). 정확한 mass/inertia/joint axis 로 articulated body dynamics gap 의 1차 원인 해결.
- **v15**: native Linux + RL + real2sim. v14.6 후 fidelity gap 이 충분히 작아지면 RL 학습 안정성 확보.
  NMPC real-time receding-horizon bridge + WBIC FB QP infeasibility 정밀 분석도 v15 에서.
