# v13 파일 분할 Plan (Phase 0)

`gait_sim_v12.py` (4032 line, 50 def, 2 class) → 다중 모듈 package `gait_sim/`.

**원본 보존**: `gait_sim_v12.py` 는 frozen (참고/회귀 검증용).

---

## 디렉토리 구조

```
gait_sim/
├── __init__.py          # 패키지 진입점
├── __main__.py          # `python -m gait_sim` 실행
├── config.py            # GaitConfig dataclass + alias
├── model.py             # DH, Q_HOME, LEG_HIP_OFFSETS, mass, inertia, joint limits
├── kinematics.py        # FK, analytical/opt IK, Jacobian
├── dynamics.py          # RNEA, M, h, gravity torque
├── body_dyn.py          # float-base 적분 (Lie group)
├── gait.py              # GaitScheduler, swing/stance foot trajectory
├── controllers/
│   ├── __init__.py
│   ├── mpc.py           # linear MPC + QP GRF distribute
│   ├── wbic.py          # WBIC per-leg + FB QP
│   └── nmpc.py          # crocoddyl FDDP (one-shot + receding + populate)
└── viz/
    ├── __init__.py
    ├── fig_main.py      # fig1 (3D animation)
    ├── fig_legs.py      # fig2 (per-leg)
    ├── fig_wbc.py       # fig3 (GRF + friction cone)
    ├── fig_tau.py       # fig4 (tau decompose)
    └── fig_diag.py      # fig5-8 (body state, foot, gait diagram, diagnostic)
```

---

## 함수 → 모듈 매핑 (line 번호는 v12.py 기준)

### `config.py` (~280 line)
- `class GaitConfig` (110)
- `CFG = GaitConfig()` instantiation
- GAIT_PRESETS application logic
- module-level alias 변수들 (GAIT_TYPE, DT, N_CYCLES, V, T, D, STEP_HEIGHT 등)

### `model.py` (~120 line)
- DH_FRONT, DH_HIND, _A2_F~_D2_F
- Q_HOME_FRONT/HIND_DEG, Q_HOME_FRONT/HIND, Q_SWING_*
- PHI_FRONT/HIND, THETA5_FRONT/HIND
- LEG_HIP_OFFSETS, _HIP_Y_BIAS, LEG_NAMES, LEG_COLORS, LEG_DH
- LINK_MASS, LINK_MASS_PER_LEG, BODY_MASS, TOTAL_MASS, LINK_RADIUS
- BODY_INERTIA + **CRBA composite upgrade**
- JOINT_VEL_LIMIT_RAD_S, JOINT_TORQUE_LIMIT
- KP_PD, KD_PD, KP_IMP, KD_IMP
- MU_FRICTION, MU_DAMP, TAU_LAG, INIT_ERR_RAD, G_ACC

### `kinematics.py` (~250 line)
- `_dh_matrix` (388)
- `forward_kinematics` (398)
- `_dh_to_sim`, `_sim_to_dh` (409, 416)
- `analytical_ik_front`, `analytical_ik_hind` (434, 455)
- `opt_ik_front`, `opt_ik_hind` (483, 544)
- `compute_jacobian_sim` (594)

### `dynamics.py` (~180 line)
- `compute_gravity_torque_sim` (613)
- `_skew` (638)
- `_rod_inertia_local` (647)
- `rnea` (656)
- `compute_mh_leg` (743)

### `body_dyn.py` (~90 line)
- `_skew` (821, **dynamics.py 와 중복** — 통합 필요)
- `_exp_so3` (828)
- `_R_to_euler_xyz` (845)
- `integrate_body_state` (862)

### `gait.py` (~140 line)
- `class GaitScheduler` (1489)
- `_swing_z_coeffs` (1514)
- `swing_foot_pos` (1539)
- `stance_foot_pos` (1568)
- `_quintic_s`, `_smootherstep` (1579, 1584)
- `foot_pos_at_phase` (1593)
- GAIT_PRESETS, PHASE_OFFSETS (또는 config.py 와 분담)

### `controllers/mpc.py` (~250 line)
- `_euler_to_R` (1297)
- `_euler_rate_T` (1308)
- `_build_Ac_at`, `_build_Bc_at` (1324, 1335)
- `_build_Ac_d`, `_build_Bc` (1346, 1359)
- `mpc_qp_plan` (1364)
- `qp_grf_distribute` (1228) — fallback
- BODY_REF_STEP, _Q_DIAG, MPC_Q, MPC_R 상수

### `controllers/wbic.py` (~470 line)
- `wbic_qp_leg` (760)
- `wbic_qp_full_pin` (931)
- `wbic_qp_full` (1041)

### `controllers/nmpc.py` (~600 line)
- `_solve_nmpc_trot` (2110)
- `_solve_nmpc_trot_receding` (2319)
- `_populate_arrays_from_nmpc` (2600)

### `viz/fig_main.py` (~300 line)
- fig1 (3D animation)
- `_style_ax`, `_body_T_at`, `_body_T_apply` (3229, 3277, 3289)
- `init_anim`, `animate` (3426, 3438)

### `viz/fig_legs.py` (~80 line)
- fig2 setup
- `_style_ax2`, `_leg_subplots` (3540, 3550)

### `viz/fig_wbc.py` (~80 line)
- fig3 setup
- `_style_ax3` (3614)

### `viz/fig_tau.py` (~50 line)
- fig4 setup
- `_style_ax4` (3676)

### `viz/fig_diag.py` (~400 line)
- fig5 (body state), fig6 (foot traj), fig7 (gait diagram), fig8 (diagnostic)
- `_style_ax5`, `_style_ax6`, `_style_ax8` (3713, 3791, 3921)

### `__main__.py` (~500 line)
- Array allocations (joint_hist, body_pos_hist, ...)
- Main loop:
  - Pre-loop (foot trajectory IK)
  - WBC + MPC + WBIC main loop
  - Body integration
- NMPC dispatch
- foot_actual/target population
- 진단 출력

---

## 의존성 그래프

```
config ──────────────┐
   │                  │
   ↓                  │
 model ───────┐       │
   │           │       │
   ↓           ↓       ↓
kinematics  dynamics  gait
   │           │       │
   ↓           ↓       │
body_dyn ←────┘       │
   │                   │
   ↓                   ↓
controllers/{mpc, wbic, nmpc}
   │                   │
   ↓                   ↓
       __main__
         │
         ↓
        viz/
```

규칙:
1. **상위→하위 import 만** (cycle 없음)
2. `config.py` 가 최상단 (모든 모듈이 import)
3. `__main__.py` 가 오케스트레이터
4. `viz/` 가 가장 하위 (모든 array 접근)

---

## 상태 공유 전략

### Module-level constants (CFG 파생)
- `from config import CFG, GAIT_TYPE, DT, V, T, ...` 
- 각 module 이 필요한 것만 import
- runtime 변경 불가 (CFG 자체 재생성하면 alias 갱신 안 됨)

### Simulation state arrays (joint_hist, body_pos_hist, ...)
- `__main__.py` 에서 생성
- function 호출 시 인자로 전달 OR module-level shared state
- **선택**: 단순화를 위해 `__main__.py` 에서 module-level 변수 + 다른 module 이 import 하는 방식 (현재 monolithic 과 호환)

### 옵션 A: 명시적 인자 전달 (정통)
```python
def populate_arrays_from_nmpc(xs, us, forces, joint_hist, body_pos_hist, ...):
    ...
```
장점: 순수 함수, 테스트 쉬움  
단점: 인자 많음, 호출 복잡

### 옵션 B: 공유 state 객체 (실용)
```python
# state.py
@dataclass
class SimState:
    joint_hist: np.ndarray
    body_pos_hist: np.ndarray
    ...
```
호출: `populate(state, xs, us, ...)`

### 옵션 C: __main__ module-level globals + import
```python
# __main__.py
joint_hist = np.zeros(...)
# nmpc.py
from gait_sim.__main__ import joint_hist  # 위험: circular
```
→ **비추**

**채택 권장: 옵션 A or B (B 가 인자 수 줄임)**

---

## Phase 실행 순서

### Phase 1 — config.py 추출 (v13.1)
- 가장 안전 (이미 dataclass)
- 작업: GaitConfig + CFG + alias → config.py
- `gait_sim_v12.py` 에서 `from gait_sim.config import *` 로 import (transition)
- 검증: trot/walk 시뮬 정상 작동

### Phase 2 — model.py 추출 (v13.2)
- DH/Q_HOME/mass/inertia/joint limits
- BODY_INERTIA CRBA 도 model 안으로

### Phase 3 — kinematics + dynamics + body_dyn (v13.3)
- 순수 함수 위주 (state 의존성 적음)
- _skew 중복 해결 (dynamics.py 로 통합, body_dyn은 import)

### Phase 4 — gait.py (v13.4)
- GaitScheduler, swing/stance foot trajectory
- GAIT_PRESETS 위치 결정 (config 또는 gait)

### Phase 5 — controllers/ (v13.5)
- mpc.py 먼저 (단순)
- wbic.py 다음
- nmpc.py 마지막 (crocoddyl 의존)

### Phase 6 — viz/ (v13.6)
- fig_main.py 부터 (animation)
- 나머지 fig 모듈 병렬 가능

### Phase 7 — __main__.py 완성 (v13.7)
- 모든 모듈 import + orchestrate
- 기존 gait_sim_v12.py 와 메트릭 비교 (회귀 검증)

---

## 검증 체크리스트 (각 Phase 후)

1. ✅ Syntax pass (`python3 -m py_compile gait_sim/*.py`)
2. ✅ Import 가능 (`python3 -c "import gait_sim"`)
3. ✅ trot 시뮬 실행 (`USE_NMPC=True`)
4. ✅ 핵심 metric 일치 (gait_sim_v12.py 대비):
   - body x final
   - vx mean
   - τ peak
   - HL Fz peak
5. ✅ walk 시뮬 실행
6. ✅ v11 standalone (USE_NMPC=False) 실행

---

## Risk & Mitigation

### Risk 1: Circular import
- **Mitigation**: 의존성 단방향 (config → model → ... → __main__)
- 함수 안에서 import (lazy)

### Risk 2: Module-level state mismatch
- **Mitigation**: state 명시적 전달 또는 SimState dataclass

### Risk 3: 함수 동작 변경 (refactor 중)
- **Mitigation**: 함수 내용 *건드리지 않음*. 단지 파일만 옮김.

### Risk 4: 시각화 의존성 (matplotlib + array 다수)
- **Mitigation**: viz 가 가장 마지막. 모든 array 가 안정되면 분리.

---

## v13 출시 후 v12.py 처분

- **유지** (참고용): `gait_sim_v12.py` 그대로
- **deprecate 표시**: docstring 에 "frozen at v12.8, use gait_sim/ for v13+" 명시
- 향후 v14, v15 등은 `gait_sim/` 만 갱신

---

## 다음 단계

Phase 1 (config.py 추출) 시작 시 user 확인 후 진행.
