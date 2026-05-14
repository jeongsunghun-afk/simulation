"""
gait_sim_v13.py  —  v12 + DH d2 부호 미러링 (좌우 mirror 정합)

v12 대비 핵심 변경:
    · DH d2 부호 좌우 분리: 좌측 다리 (FL, HL) 는 d2 = -0.0075 (우측 +0.0075 의 거울상)
        - 같은 Q_HOME 으로 *수치적 거울 대칭* foot 위치 산출 (검증 완료)
        - foot leg-local sim: 우측 (-0.0568, -0.0075, -0.4650), 좌측 (-0.0568, +0.0075, -0.4650)
    · hip_y_bias = 0  (v12 의 "부분 보정" kludge 제거 — 더 이상 불필요)
    · LEG_HIP_OFFSETS = ±BODY_LAT  (깔끔한 좌우 대칭 hip translation)
    · LEG_DH = [DH_FRONT_R, DH_FRONT_L, DH_HIND_R, DH_HIND_L]  (4벌)
    · analytical_ik_front / opt_ik_front / opt_ik_hind 모두 `dh` 파라미터 받도록 refactor
        - 기존 module-level _D2_F / DH_FRONT / DH_HIND 사용을 명시적 dh 인자로 전환
    · v12 smoothness 패치 (DT_MPC=DT*5, wbic_w_dtau, CubicSpline q̇/q̈) 그대로 이식
      → v13.1 에서 정량 측정 결과 net negative (NMPC τ peak +4988%, jerk peak +1454%
        ; v11 jerk 소폭 ↑) → 모두 OFF default 로 비활성화.
        Toggle: GaitConfig.wbic_w_dtau (=0.0), GaitConfig.use_spline_diff (=False),
                DT_MPC = DT*10 (원래 값)

기대 효과:
    · foot world 위치 좌우 대칭 (±0.0575) → 정적 roll moment ≈ 0
    · 다리 mass distribution 좌우 거울 대칭 → dynamic drift 차단
    · body y drift 근본 해결 (v12 의 ~수십 mm 잔존 drift → ≈ 0)

URDF / Pinocchio 미러링은 별도 작업 (v13 의 USE_PINOCCHIO=False 일 때 정합).

──────────────────────────────────────────────────────────────────
이하 v12 변경 사항 기록 (참고용)
──────────────────────────────────────────────────────────────────

gait_sim_v12.py  —  v11 + crocoddyl NMPC 통합 (USE_NMPC 토글)

v11 대비 추가:
    · USE_NMPC 토글 (True 시 v11의 MPC + WBIC 우회, crocoddyl FDDP로 trajectory 풀이)
    · NMPC가 직접 푸는 변수: τ_cmd, q, body 6-DoF 모두
    · v11의 figure 1 시각화 그대로 활용 (joint_hist, body_pos_hist 등 NMPC 결과로 채움)
    · 1-2 cycle one-shot 안정 (multi-cycle은 추후 receding horizon 작업)

동작:
    USE_NMPC = False : v11 동작 (MPC + WBIC + body 적분)
    USE_NMPC = True  : 시작 시 N_FRAMES 한 번에 NMPC 풀이 → trajectory 사용

gait_sim_v11.py  —  4족 보행 Gait 시뮬레이터 + WBIC QP + Floating-Base

v10 대비 추가 (Phase 7 Tier 1):
    · Floating-base 통합 동역학 (USE_BODY_DYNAMICS):
        v10: body 위치 = V·t (kinematic only, GRF 무관하게 강제 직진)
        v11: body 6-DoF (CoM pos, R, v, ω) 동적 적분
            M·v̇ = Σλ + M·g_world          (linear, world frame)
            I_body·ω̇ + ω×(I_body·ω) = Σ(r_i × λ_i)   (angular, about CoM)
            R_{k+1} = exp(Δt·skew(ω)) · R_k          (Lie group rotation)
            symplectic Euler 적분
        효과: 실제 pitch/roll 응답, ZMP, CoM 진동 측정 가능.
              MPC가 제대로 동작하면 body가 ref 추종, 실패하면 발산.

    · WBIC 부유 베이스 통합 QP (USE_WBIC_FB):
        v10: per-leg 4 separate QPs, 다리 간 결합 없음
        v11: 단일 QP, [Δv̇_fb (6); Δq̈ (4·5); Δτ (4·5); Δλ (4·3)] = 58 vars
        등식:
            body 6-DoF: M_fb·(v̇_des + Δv̇_fb) = F_des + ΔF(Δλ)
                        ΔF_lin = Σ Δλ_i,  ΔF_ang = Σ r_i × Δλ_i
            per-leg dyn: M_i·(q̈_i_des + Δq̈_i) + h_i = (τ_ff_i + Δτ_i) + Jᵀ_i·(λ_des_i + Δλ_i)
                        → M_i·Δq̈_i − Δτ_i − Jᵀ_i·Δλ_i = r_i  (per-leg residual)
        부등: τ 한계, 마찰 추, λ_z ≥ lamz_min (per-leg), swing Δλ = −λ_des
        효과: GRF 재배분이 body force/torque 평형 동시 만족하도록 강제.
              per-leg에서는 ΔΛ가 독립적이라 body 평형 깨질 수 있음.

    · 진단 배열: body_pos/R/v/omega_hist (4 array, N_FRAMES 길이)
    · 신규 figure: body trajectory + orientation + CoM 진동 시계열

v10 누적 (참고):
    · figure 4 tau_grf line 숨김 (저장은 유지)
v9 누적 (참고):
    · Phase 5 WBIC QP (per-leg baseline):
        변수  x_leg = [Δq̈ (nj), Δτ (nj), Δλ (3)]
        비용  α·||Δq̈||² + β·||Δτ||² + γ·||Δλ||²
        등식  M(q)·(q̈+Δq̈) + h(q,q̇) = (τ_ff+Δτ) + Jᵀ·(λ_des+Δλ)
        부등  τ_min ≤ τ_ff+Δτ ≤ τ_max
              stance: |λ_x|,|λ_y| ≤ μ·λ_z,  λ_z ≥ λ_z_min
              swing : λ = 0 (Δλ = −λ_des)

v8 누적 (참고):
    · Phase 1 MPC QP, Phase 2 QP GRF, Phase 3 RNEA, Phase 4 Opt-IK (앞/뒷다리)
"""

import math
import os
import sys
import pickle
import time
from dataclasses import dataclass, field
import numpy as np
import qpsolvers

# v11 Phase 7: Pinocchio (선택적 — USE_PINOCCHIO=True일 때 사용)
try:
    import pin_helpers as _pin_helpers
    _PIN_AVAILABLE = True
except ImportError:
    _PIN_AVAILABLE = False

# v12: crocoddyl (선택적 — USE_NMPC=True일 때 사용)
try:
    import crocoddyl as _crocoddyl
    import build_pin_model as _bm
    _CROCODDYL_AVAILABLE = True
except ImportError:
    _CROCODDYL_AVAILABLE = False

# 비교 모드: HIND_VARIANT={'orig','ext'}, COMPARE_MODE=1이면 figure 생략 후 metrics 덤프
_HIND_VARIANT = os.environ.get('HIND_VARIANT', 'ext')  # 'orig' (원본) vs 'ext' (뒷발 -50mm 확장)
_COMPARE_MODE = os.environ.get('COMPARE_MODE', '0') == '1'
assert _HIND_VARIANT in ('orig', 'ext'), f"HIND_VARIANT={_HIND_VARIANT} (expected 'orig' or 'ext')"
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize as _sp_minimize
from scipy.interpolate import CubicSpline as _CubicSpline

for key in mpl.rcParams:
    if key.startswith("keymap."):
        mpl.rcParams[key] = []
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'NanumGothic', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════
# 0. 파라미터 — GaitConfig dataclass (모든 user-tunable 파라미터 통합)
# ══════════════════════════════════════════════════════════════
# 사용:
#   - 기본 동작: 아래 필드 default 값을 직접 수정하고 재실행
#   - 임시 실험: 파일 끝에서 CFG = GaitConfig(use_nmpc=False, ...) 로 재생성
#     (단, 모듈 globals(GAIT_TYPE 등)는 아래 alias 라인에서 한 번만 평가되므로
#      runtime 변경 시 alias 도 다시 할당 필요)
#   - 카테고리 별 설명은 README 또는 v12.4 commit message 참조
# ──────────────────────────────────────────────────────────────
@dataclass
class GaitConfig:
    """v12 NMPC + WBIC + MPC + IK 통합 설정. 섹션별로 묶음."""
    # ─────────── Gait pattern ───────────
    gait_type: str = 'trot'       # 'walk' / 'amble' / 'pace' / 'trot' / 'canter' / 'gallop'
    dt: float = 0.002             # 시뮬 타임스텝 [s] (WBC 제어 주기)
    n_cycles: int = 4             # 사이클 수
    tau_land: float = 1.0         # swing phase 내 착지 비율 (0~1)
    # ─────────── Robot geometry ───────────
    body_fwd_f: float = 0.250     # 앞다리 hip x [m]
    body_fwd_h: float = -0.250    # 뒷다리 hip x [m]
    body_lat: float = 0.050       # 좌우 hip y [m]
    body_z_h: float = -0.050      # 뒷다리 hip z 오프셋 [m]
    hip_y_bias: float = 0.0       # v13: DH d2 미러링으로 근본 해결. v12 의 부분 보정 제거.
    # ─────────── Robot mass / inertia ───────────
    body_mass: float = 15.0       # body 질량 [kg]
    g_acc: float = 9.81           # [m/s²]
    link_mass: np.ndarray = field(default_factory=lambda: np.array([3., 2., 1., 0.2, 0.1]))
    link_radius: float = 0.015    # 링크 단면 반경 [m] (RNEA 원통 관성용)
    # v13.14: CRBA composite (다리 포함) hardcode — base-link only 면 MPC body 6-DoF
    #   under-damped → 3+ cycle 발산 (Fz spike ~4500N). pinocchio CRBA(home pose) 측정값.
    body_inertia: np.ndarray = field(default_factory=lambda: np.array([
        [ 0.8044,  0.0,    -0.2547],
        [ 0.0,     2.1571,  0.0   ],
        [-0.2547,  0.0,     1.5599],
    ]))
    # ─────────── Joint motor limits ───────────
    joint_vel_limit_rad_s: np.ndarray = field(
        default_factory=lambda: np.array([14.66, 15.91, 15.91, 14.66, 14.66]))
    joint_torque_limit: np.ndarray = field(
        default_factory=lambda: np.array([60., 120., 120., 60., 60.]))
    vel_limit_margin: float = 999
    # ─────────── Joint PD / impedance ───────────
    kp_pd: np.ndarray = field(default_factory=lambda: np.array([30., 80., 80., 60., 20.]))
    kd_pd: np.ndarray = field(default_factory=lambda: np.array([3., 8., 8., 6., 2.]))
    kp_imp: np.ndarray = field(default_factory=lambda: np.array([400., 400., 400.]))
    kd_imp: np.ndarray = field(default_factory=lambda: np.array([20., 20., 20.]))
    # ─────────── Friction / actuator dynamics ───────────
    mu_friction: float = 0.6      # 마찰 계수 (friction cone)
    mu_damp: float = 1e-3         # joint damping
    tau_lag: float = 0.03         # actuator 1st-order lag [s]
    init_err_rad: float = math.radians(1.0)
    grf_ramp_ratio: float = 0.10  # stance 시작/끝 GRF ramp 구간 비율
    stance_delta: float = 0.005
    # ─────────── Optimization-based IK ───────────
    lambda_q_opt: float = 1.0     # smoothness weight
    lambda_tau_opt: float = 0.01  # torque minimize weight
    opt_ik_maxiter: int = 100
    opt_ik_use_vel_limit: bool = True
    opt_ik_use_tau_limit: bool = True
    use_swing_qref_blend: bool = True
    max_traj_opt_iters: int = 6
    # ─────────── Linear MPC (body trajectory) ───────────
    use_mpc: bool = True
    use_mpc_closed_loop: bool = True
    n_mpc: int = 10               # horizon length
    # ─────────── WBIC ───────────
    use_wbic: bool = True
    use_wbic_fb: bool = True
    use_body_dynamics: bool = True
    use_stance_foot_constraint: bool = False
    wbic_w_ddq: float = 1.0
    wbic_w_tau: float = 0.01
    wbic_w_lam: float = 0.001
    wbic_lamz_min: float = 1.0    # stance 발 최소 법선력 [N]
    wbic_w_fb: float = 0.1        # body fb weight
    wbic_w_dtau: float = 0.0      # τ smoothness: w_dtau·‖τ−τ_prev‖²  (v13.1: 정량 측정 결과 net negative → OFF default)
    use_spline_diff: bool = False # q̇/q̈ 미분: True=CubicSpline, False=np.gradient (v13.1: 정량 측정 결과 net negative → OFF default)
    # ─────────── NMPC: solver ───────────
    use_nmpc: bool = True
    use_nmpc_receding: bool = True
    nmpc_maxiter: int = 200
    nmpc_init_reg: float = 1.0
    nmpc_rh_n_horizon: int = 24   # 0.5s @ DT_NMPC=0.02 (v13.1: DT_MPC 원복으로 환원)
    nmpc_rh_n_resolve: int = 12   # half cycle re-solve
    nmpc_rh_maxiter: int = 50
    # ─────────── NMPC: cost weights ───────────
    nmpc_w_track_xy: float = 100.0
    nmpc_w_track_z: float = 10000.0
    nmpc_baumgarte_kp: float = 0.0
    nmpc_baumgarte_kd: float = 20.0
    nmpc_w_state_reg: float = 1.0
    nmpc_w_ctrl_reg: float = 1e-3
    nmpc_w_terminal: float = 1e2
    nmpc_w_friction: float = 1.0
    nmpc_fric_nf: int = 4         # pyramid sides (4 / 8)
    nmpc_fric_fz_max: float = 1e4 # 최대 normal force [N]
    nmpc_w_force_reg: float = 1e-4
    nmpc_w_force_xy: float = 5.0
    nmpc_w_force_z: float = 1.0
    nmpc_w_tau_lim: float = 1e-2
    nmpc_w_touchdown_v: float = 1e1
    nmpc_touchdown_last_n: int = 2
    nmpc_w_stance_pos_xy: float = 1e2
    nmpc_w_stance_pos_z: float = 2.0
    # ─────────── Perturbation test ───────────
    use_perturbation: bool = False
    perturb_time: float = 1.0     # [s]
    perturb_vel_lin: np.ndarray = field(default_factory=lambda: np.array([0., 0.5, 0.]))
    perturb_vel_ang: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
    # ─────────── Pinocchio backend ───────────
    use_pinocchio: bool = False
    use_pinocchio_full_m: bool = False
    # ─────────── Visualization ───────────
    viz_body_mode: str = 'world'  # 'static' / 'world' / 'body_follow'

CFG = GaitConfig()

# ── 모듈 globals: CFG 필드 alias (기존 v11/v12 코드 호환) ────────
GAIT_TYPE   = CFG.gait_type
DT          = CFG.dt
N_CYCLES    = CFG.n_cycles

# ── Gait 프리셋 (V/T/D/STEP_HEIGHT/offsets 통합) ────────────────
# offsets: phase=0이 swing 시작인 컨벤션, 순서 [FR, FL, HR, HL]
GAIT_PRESETS = {
    'walk':   dict(V=0.4, T=1.0, D=0.25, STEP_HEIGHT=0.05,
                   offsets=[0.00, 0.50, 0.75, 0.25]),
    'amble':  dict(V=0.7, T=0.7, D=0.40, STEP_HEIGHT=0.06,
                   offsets=[0.00, 0.50, 0.75, 0.25]),
    'pace':   dict(V=1.0, T=0.5, D=0.50, STEP_HEIGHT=0.07,
                   offsets=[0.00, 0.50, 0.00, 0.50]),
    'trot':   dict(V=1.0, T=0.5, D=0.50, STEP_HEIGHT=0.08,
                   offsets=[0.00, 0.50, 0.50, 0.00]),
    'canter': dict(V=1.3, T=0.45, D=0.50, STEP_HEIGHT=0.09,
                   offsets=[0.00, 0.33, 0.33, 0.67]),
    'gallop': dict(V=1.3, T=0.40, D=0.55, STEP_HEIGHT=0.10,
                   offsets=[0.00, 0.05, 0.55, 0.50]),
}
_preset = GAIT_PRESETS[GAIT_TYPE]
V           = _preset['V']            # m/s (전진 속도)
T           = _preset['T']            # s (사이클 주기)
D           = _preset['D']            # swing 비율 (T_SW/T)
STEP_HEIGHT = _preset['STEP_HEIGHT']  # m (발 들리는 높이)
TAU_LAND    = CFG.tau_land

T_SW = T * D
T_ST = T * (1.0 - D)

STRIDE_D_MIN = 2.0 * V * T_SW
STRIDE_D     = V * T + 2.0 * V * T_SW
assert STRIDE_D >= STRIDE_D_MIN, f"STRIDE_D({STRIDE_D:.3f}m) < MIN({STRIDE_D_MIN:.3f}m)"

STEP_LENGTH = STRIDE_D / 2.0 - V * T_SW
STANCE_DELTA = CFG.stance_delta

BODY_FWD_F = CFG.body_fwd_f
BODY_FWD_H = CFG.body_fwd_h
BODY_LAT   = CFG.body_lat
BODY_Z_H   = CFG.body_z_h

# ── DH 파라미터 (v13: 좌우 d2 부호 분리 — 거울 대칭 다리) ─────────
# Right legs (FR, HR): d2 = +0.0075   (외측으로 +y abduction)
# Left  legs (FL, HL): d2 = -0.0075   (외측으로 -y abduction → body frame +y 발 위치)
# 같은 Q_HOME 으로 FK 시 leg-local sim foot y 부호만 반전 → 좌우 거울 대칭.
DH_FRONT_R = [
    (+math.pi/2, 0.0,   0.0,    ),
    (0.0,        0.21,  +0.0075,),    # ← d2 = +0.0075
    (0.0,        0.235, 0.0,    ),
    (0.0,        0.1,   0.0,    ),
    (0.0,        0.045, 0.0,    ),
]
DH_FRONT_L = [
    (+math.pi/2, 0.0,   0.0,    ),
    (0.0,        0.21,  -0.0075,),    # ← d2 = -0.0075  (좌측 미러)
    (0.0,        0.235, 0.0,    ),
    (0.0,        0.1,   0.0,    ),
    (0.0,        0.045, 0.0,    ),
]
DH_HIND_R = [
    (-math.pi/2, 0.0,   0.0,    ),
    (0.0,        0.21,  +0.0075,),
    (0.0,        0.21,  0.0,    ),
    (0.0,        0.148, 0.0,    ),
    (0.0,        0.045, 0.0,    ),
]
DH_HIND_L = [
    (-math.pi/2, 0.0,   0.0,    ),
    (0.0,        0.21,  -0.0075,),    # ← d2 = -0.0075  (좌측 미러)
    (0.0,        0.21,  0.0,    ),
    (0.0,        0.148, 0.0,    ),
    (0.0,        0.045, 0.0,    ),
]

# 호환용 alias — v12 코드에서 DH_FRONT/DH_HIND 단일 참조하던 곳은 우측 (R) 로 매핑.
# 좌측 IK 호출 시 LEG_DH[leg] 로 적절한 dh 전달.
DH_FRONT = DH_FRONT_R
DH_HIND  = DH_HIND_R

_A2_F = 0.21; _A3_F = 0.235; _A4_F = 0.1; _A5_F = 0.045; _D2_F = 0.0075

#Q_HOME_FRONT_DEG = [0.0, 157.5, 22.5, 30.6583, 59.3417] # BODY_Z_H=-0.1
Q_HOME_FRONT_DEG = [0.0, 133.2973, 46.7027, 30.6583, 59.3417]    # 원본 (BODY_Z_H=-0.05)
# HIND variant 분기 (비교용; HIND_VARIANT 환경변수)
if _HIND_VARIANT == 'ext':
    Q_HOME_HIND_DEG  = [0.0, -154.8138, -92.8840, 88.6091, 60.0000]  # 뒷발 -50mm 확장 (간격 451mm)
else:  # 'orig'
    Q_HOME_HIND_DEG  = [0.0, -150.0, -90.0, 90.0, 60.0]              # 원본 (간격 401mm)
Q_HOME_FRONT = [math.radians(a) for a in Q_HOME_FRONT_DEG]
Q_HOME_HIND  = [math.radians(a) for a in Q_HOME_HIND_DEG]

# swing 중 opt_ik 비용함수 참조 자세 (th4 음수 유도, 나머지는 home과 동일)
Q_SWING_FRONT_DEG = [0.0, 118.2973, 96.7027, -25.0, 59.3417]  # th4 -25° (vel-safe boundary)
Q_SWING_FRONT = [math.radians(a) for a in Q_SWING_FRONT_DEG]
# 뒷다리 swing 참조 자세 (placeholder = home; 발 들기 효과 원하면 튜닝 필요)
Q_SWING_HIND_DEG = list(Q_HOME_HIND_DEG)
Q_SWING_HIND = [math.radians(a) for a in Q_SWING_HIND_DEG]

PHI_FRONT    = Q_HOME_FRONT[1] + Q_HOME_FRONT[2] + Q_HOME_FRONT[3]
PHI_HIND     = Q_HOME_HIND[1]  + Q_HOME_HIND[2]  + Q_HOME_HIND[3]
THETA5_FRONT = PHI_FRONT + Q_HOME_FRONT[4]
THETA5_HIND  = PHI_HIND  + Q_HOME_HIND[4]

Q_HOME_PER_LEG      = [Q_HOME_FRONT, Q_HOME_FRONT, Q_HOME_HIND, Q_HOME_HIND]
PHI_PER_LEG         = [PHI_FRONT, PHI_FRONT, PHI_HIND, PHI_HIND]
TRAJ_PT_IDX_PER_LEG = [4, 4, 4, 4]

LEG_NAMES        = ['FR', 'FL', 'HR', 'HL']
LEG_COLORS       = ['#00d4ff', '#ff6b35', '#00ff99', '#c264ff']
LEG_DH           = [DH_FRONT_R, DH_FRONT_L, DH_HIND_R, DH_HIND_L]   # v13: 좌우 d2 미러
N_JOINTS_PER_LEG = [5, 5, 5, 5]
N_JOINTS_MAX     = 5

# DH의 D2 오프셋(=0.0075)으로 인한 좌우 비대칭 보정:
# foot_local_y = -0.0075 (모든 다리 동일) → hip_y에 +0.0075 적용 시
# foot_world_y = ±BODY_LAT 정확히 대칭 (CoM 기준 roll moment ≈ 0)
_HIP_Y_BIAS = CFG.hip_y_bias   # ≡ DH dh[1][2] (= D2_F = D2_H)
LEG_HIP_OFFSETS = np.array([
    [+BODY_FWD_F, -BODY_LAT + _HIP_Y_BIAS, 0.0     ],
    [+BODY_FWD_F, +BODY_LAT + _HIP_Y_BIAS, 0.0     ],
    [+BODY_FWD_H, -BODY_LAT + _HIP_Y_BIAS, BODY_Z_H],
    [+BODY_FWD_H, +BODY_LAT + _HIP_Y_BIAS, BODY_Z_H],
])

PHASE_OFFSETS = {name: cfg['offsets'] for name, cfg in GAIT_PRESETS.items()}

# ── WBC 파라미터 ─────────────────────────────────────────────
BODY_MASS = CFG.body_mass
G_ACC     = CFG.g_acc

#LINK_MASS         = np.array([3.34, 0.8, 0.2, 0.2, 0.05])  # link1~5 질량 [kg]
#LINK_MASS         = np.array([4.125, 1.795, 0.78, 0.78, 0.05])  # link1~5 질량 [kg] 
LINK_MASS         = CFG.link_mass
# 80형번 0.915kg, 90형번 1.605kg
LINK_MASS_PER_LEG = [LINK_MASS] * 4
TOTAL_MASS        = BODY_MASS + float(np.sum(LINK_MASS)) * 4.0
LINK_RADIUS       = CFG.link_radius

KP_PD  = CFG.kp_pd
KD_PD  = CFG.kd_pd
KP_IMP = CFG.kp_imp
KD_IMP = CFG.kd_imp

MU_DAMP      = CFG.mu_damp
TAU_LAG      = CFG.tau_lag
INIT_ERR_RAD = CFG.init_err_rad

# ── MPC / QP GRF 파라미터 ────────────────────────────────────
MU_FRICTION  = CFG.mu_friction

BODY_INERTIA = CFG.body_inertia

# v12.7: pinocchio CRBA 로 *다리 포함* composite body inertia 계산 (home pose 근사).
# 기존 diag([0.07, 0.26, 0.26]) 는 body link 만 → 다리 무게 무시 → v11 standalone trot 발산.
# CRBA M[3:6, 3:6] = body frame angular inertia (legs at home).
if _CROCODDYL_AVAILABLE:
    try:
        import pinocchio as _pin_init
        _m_init = _bm.build_model()
        _d_init = _m_init.createData()
        _q0_init = _pin_init.neutral(_m_init)
        _Q_HOME_LEGS = {'FR': Q_HOME_FRONT, 'FL': Q_HOME_FRONT,
                        'HR': Q_HOME_HIND,  'HL': Q_HOME_HIND}
        for _leg_name, _qh in _Q_HOME_LEGS.items():
            for _i, _qi in enumerate(_qh):
                _q0_init[_m_init.idx_qs[_m_init.getJointId(f'leg_{_leg_name}_j{_i+1}')]] = _qi
        _M_full = _pin_init.crba(_m_init, _d_init, _q0_init)
        BODY_INERTIA = np.array(_M_full[3:6, 3:6])
        print(f"  [v12.7] BODY_INERTIA upgraded (CRBA composite, home pose): "
              f"diag=[{BODY_INERTIA[0,0]:.3f}, {BODY_INERTIA[1,1]:.3f}, {BODY_INERTIA[2,2]:.3f}] "
              f"(vs base-link only [0.07, 0.26, 0.26])")
        # v13: CRBA off-diagonal 좌우 대칭 검증용 출력
        print(f"  [v13]  BODY_INERTIA off-diag: Ixy={BODY_INERTIA[0,1]:+.5f}, "
              f"Ixz={BODY_INERTIA[0,2]:+.5f}, Iyz={BODY_INERTIA[1,2]:+.5f}  "
              f"(좌우 대칭이면 Ixy≈Iyz≈0)")
        del _m_init, _d_init, _q0_init, _M_full
    except Exception as _e:
        print(f"  [v12.7] BODY_INERTIA CRBA upgrade failed: {_e}")

N_MPC  = CFG.n_mpc
DT_MPC = DT * 10         # MPC 샘플링 주기 [s]  (= 0.02s)  v13.1: v12 smoothness 패치 원복

# MPC 상태 가중치: x=[roll,pitch,yaw, px,py,pz, ωx,ωy,ωz, vx,vy,vz, g]
# v13: py/vy 가중치 활성화 — MPC closed-loop 으로 y drift 보정.
# (v12 에서는 _HIP_Y_BIAS 부분 보정 + 다리 비대칭 의 우연한 상쇄로 drift 작았으나,
#  v13 DH 미러 완벽 → 잔여 y momentum 을 MPC 가 직접 잡아야 함.)
_Q_DIAG = np.array([
    200, 200, 100,   # roll, pitch, yaw
      0, 100, 200,   # px, py, pz  (v13: py 활성화)
      0,   0,   0,   # ωx, ωy, ωz
     10,  10,   0,   # vx, vy, vz  (v13: vy 활성화)
      0,           # g (상수, 추종 불필요)
], dtype=float)
MPC_Q = np.diag(_Q_DIAG)
MPC_R = 1e-6 * np.eye(3)   # GRF 가중치 (per foot, 3×3)

USE_MPC = CFG.use_mpc

# Contact GRF ramp: stance 시작/끝 N% 구간에서 λ_des를 smoothstep으로 ramp.
# swing↔stance 경계의 cmd torque step jump 완화 (논문 임피던스 흡수 효과).
GRF_RAMP_RATIO = CFG.grf_ramp_ratio

# ══════════════════════════════════════════════════════════════
# 1. 기구학
# ══════════════════════════════════════════════════════════════

def _dh_matrix(alpha, a, d, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1],
    ], dtype=float)

def forward_kinematics(thetas, dh=None):
    if dh is None:
        dh = DH_FRONT
    T = np.eye(4)
    pts = [np.zeros(3)]
    for i, (alpha, a, d) in enumerate(dh):
        T = T @ _dh_matrix(alpha, a, d, thetas[i])
        pts.append(T[:3, 3].copy())
    return pts


def _dh_to_sim(vec, front_leg=False):
    sim = np.array([vec[2], -vec[1], vec[0]], dtype=float)
    if front_leg:
        sim[:2] *= -1.0
    return sim


def _sim_to_dh(vec, front_leg=False):
    sim = np.array(vec, dtype=float)
    if front_leg:
        sim[:2] *= -1.0
    return np.array([sim[2], -sim[1], sim[0]], dtype=float)


_fk_front_home      = forward_kinematics(Q_HOME_FRONT, DH_FRONT)
_FRONT_J4_TO_J5_DH  = np.array(_fk_front_home[5]) - np.array(_fk_front_home[4])
_FRONT_J4_TO_J5_SIM = _dh_to_sim(_FRONT_J4_TO_J5_DH, front_leg=True)

_fk_hind_home      = forward_kinematics(Q_HOME_HIND, DH_HIND)
_HIND_J4_TO_J5_DH  = np.array(_fk_hind_home[5]) - np.array(_fk_hind_home[4])
_HIND_J4_TO_J5_SIM = _dh_to_sim(_HIND_J4_TO_J5_DH, front_leg=False)

J4_TO_J5_SIM_PER_LEG = [_FRONT_J4_TO_J5_SIM, _FRONT_J4_TO_J5_SIM,
                          _HIND_J4_TO_J5_SIM,  _HIND_J4_TO_J5_SIM]

def analytical_ik_front(Px, Py, Pz, phi, theta5_target, dh=None):
    """v13: dh 인자로 좌우 (DH_FRONT_R / DH_FRONT_L) 모두 처리.
       dh=None 이면 후방호환 — _D2_F (=DH_FRONT_R[1][2]) 사용.
    """
    if dh is None:
        d2 = _D2_F; a2 = _A2_F; a3 = _A3_F; a4 = _A4_F; a5 = _A5_F
    else:
        d2 = dh[1][2]; a2 = dh[1][1]; a3 = dh[2][1]; a4 = dh[3][1]; a5 = dh[4][1]
    D2 = Px**2 + Py**2 - d2**2
    if D2 < 0:
        return None
    R = math.sqrt(D2)
    theta1 = math.atan2(d2, -R) - math.atan2(-Py, Px)
    c1, s1 = math.cos(theta1), math.sin(theta1)
    x_s = c1 * Px + s1 * Py
    x3 = x_s - a4 * math.cos(phi) - a5 * math.cos(theta5_target)
    z3 = Pz   - a4 * math.sin(phi) - a5 * math.sin(theta5_target)
    cos_th3 = (x3**2 + z3**2 - a2**2 - a3**2) / (2.0 * a2 * a3)
    cos_th3 = max(-1.0, min(1.0, cos_th3))
    theta3  = math.acos(cos_th3)
    theta2 = (math.atan2(z3, x3)
              - math.atan2(a3 * math.sin(theta3), a2 + a3 * math.cos(theta3)))
    theta4 = phi - theta2 - theta3
    theta5 = theta5_target - (theta2 + theta3 + theta4)
    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi
    return [wrap(theta1), wrap(theta2), wrap(theta3), wrap(theta4), wrap(theta5)]

def analytical_ik_hind(Px, Py, Pz, phi, dh, theta5_target=None):
    a2 = dh[1][1]; a3 = dh[2][1]; a4 = dh[3][1]; d2 = dh[1][2]
    D2 = Px**2 + Py**2 - d2**2
    if D2 < 0:
        return None
    R = math.sqrt(D2)
    theta1 = math.atan2(-Px, Py) - math.atan2(R, d2)
    c1, s1 = math.cos(theta1), math.sin(theta1)
    x_s = c1 * Px + s1 * Py
    Z   = -Pz
    if theta5_target is not None:
        a5 = dh[4][1]
        x2 = x_s - a4 * math.cos(phi) - a5 * math.cos(theta5_target)
        z2 = Z   - a4 * math.sin(phi) - a5 * math.sin(theta5_target)
    else:
        x2 = x_s - a4 * math.cos(phi)
        z2 = Z   - a4 * math.sin(phi)
    cos_th3 = (x2**2 + z2**2 - a2**2 - a3**2) / (2.0 * a2 * a3)
    cos_th3 = max(-1.0, min(1.0, cos_th3))
    theta3  = -math.acos(cos_th3)
    theta2  = (math.atan2(z2, x2)
               - math.atan2(a3 * math.sin(theta3), a2 + a3 * math.cos(theta3)))
    theta4  = phi - theta2 - theta3
    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi
    return [wrap(theta1), wrap(theta2), wrap(theta3), wrap(theta4)]


def opt_ik_front(p_target_dh, q_init, q_ref=None, dh=None):
    """SLSQP 최적화 IK — 앞다리. v13: dh 인자로 좌우 (DH_FRONT_R/L) 분리.

    등식 제약 : FK_tip(q) = p_target          (위치 정확도 보장)
                q[4] = Q_HOME_FRONT[4]        (toe 고정 — IK에서 제외)
    부등식 제약: |τ_grav(q)| ≤ τ_limit        (OPT_IK_USE_TAU_LIMIT, 중력 토크 근사)
    bounds     : FRONT_Q_LIM ∩ 각속도 한계    (OPT_IK_USE_VEL_LIMIT, 정확)
    비용       : LAMBDA_Q_OPT·||q - q_ref||² + LAMBDA_TAU_OPT·||τ_grav(q)||²
                 - q_ref가 주어지면 그 자세 추종 (swing1/swing2 quintic blend)
                 - q_ref=None이면 q_init 사용 (smoothness only)
    """
    if dh is None:
        dh = DH_FRONT_R   # v13 후방호환 default (= 우측)
    p_t   = np.asarray(p_target_dh, dtype=float)
    q0    = np.asarray(q_init, dtype=float)
    q_tgt = q0 if q_ref is None else np.asarray(q_ref, dtype=float)

    # ── 각속도 제약: FRONT_Q_LIM을 |Δq| ≤ vel_limit*DT 범위로 동적 수축 ──
    if OPT_IK_USE_VEL_LIMIT:
        vel_dt = JOINT_VEL_LIMIT_RAD_S * DT
        lo = np.maximum([b[0] for b in FRONT_Q_LIM], q0 - vel_dt)
        hi = np.minimum([b[1] for b in FRONT_Q_LIM], q0 + vel_dt)
        hi = np.maximum(lo, hi)          # lo > hi 방지 (위치 한계 경계부)
        active_bounds = list(zip(lo, hi))
    else:
        active_bounds = FRONT_Q_LIM

    # ── 등식 제약: 발끝 위치 ──────────────────────────────────────────────
    constraints = [{'type': 'eq',
                    'fun': lambda q: np.array(forward_kinematics(q, dh)[-1]) - p_t}]

    # ── 등식 제약: q5(toe) 고정 — IK에서 제외 (analytical과 동일하게 home 값 유지) ─
    # th5는 가벼운 toe link이라 IK 자유도로 활용하지 않음.
    # q5 변동이 dq 폭증 원인이 됐으므로 고정하여 4 DoF redundancy(q1~q4)만 사용.
    constraints.append({'type': 'eq',
                        'fun': lambda q: q[4] - Q_HOME_FRONT[4]})

    # ── 부등식 제약: 중력 토크 한계 (τ_full 근사, 속도·GRF 항 미포함) ─────
    _lm_front = LINK_MASS_PER_LEG[0]
    if OPT_IK_USE_TAU_LIMIT:
        def _torque_ineq(q):
            tau_g = compute_gravity_torque_sim(q, dh, _lm_front, front_leg=True)
            return JOINT_TORQUE_LIMIT[:len(tau_g)] - np.abs(tau_g)  # ≥ 0 이어야 통과
        constraints.append({'type': 'ineq', 'fun': _torque_ineq})

    def cost(q):
        # 참조 자세 추종 (q_ref=None이면 q_init = warm-start smoothness)
        c_qref = LAMBDA_Q_OPT * np.dot(q - q_tgt, q - q_tgt)
        # τ_grav minimize: redundancy를 토크 작은 자세로 자동 사용
        tau_g  = compute_gravity_torque_sim(q, dh, _lm_front, front_leg=True)
        c_tau  = LAMBDA_TAU_OPT * np.dot(tau_g, tau_g)
        return float(c_qref + c_tau)

    res = _sp_minimize(cost, q0, method='SLSQP', bounds=active_bounds,
                       constraints=constraints,
                       options={'ftol': 1e-8, 'maxiter': OPT_IK_MAXITER})
    tip_final = np.array(forward_kinematics(res.x, dh)[-1])
    pos_err_sq = float(np.dot(tip_final - p_t, tip_final - p_t))
    if pos_err_sq < 1e-6:
        return list(res.x), res.nit, pos_err_sq
    return None, res.nit, pos_err_sq


def opt_ik_hind(p_target_dh, q_init, q_ref=None, dh=None):
    """SLSQP 최적화 IK — 뒷다리. v13: dh 인자로 좌우 (DH_HIND_R/L) 분리.

    등식 제약 : FK_tip(q) = p_target          (위치 정확도 보장)
                q[4] = Q_HOME_HIND[4]         (toe 고정)
    부등식 제약: |τ_grav(q)| ≤ τ_limit        (OPT_IK_USE_TAU_LIMIT)
    bounds     : HIND_Q_LIM ∩ 각속도 한계      (OPT_IK_USE_VEL_LIMIT)
    비용       : LAMBDA_Q_OPT·||q - q_ref||² + LAMBDA_TAU_OPT·||τ_grav(q)||²
    """
    if dh is None:
        dh = DH_HIND_R   # v13 후방호환 default (= 우측)
    p_t   = np.asarray(p_target_dh, dtype=float)
    q0    = np.asarray(q_init, dtype=float)
    q_tgt = q0 if q_ref is None else np.asarray(q_ref, dtype=float)

    if OPT_IK_USE_VEL_LIMIT:
        vel_dt = JOINT_VEL_LIMIT_RAD_S * DT
        lo = np.maximum([b[0] for b in HIND_Q_LIM], q0 - vel_dt)
        hi = np.minimum([b[1] for b in HIND_Q_LIM], q0 + vel_dt)
        hi = np.maximum(lo, hi)
        active_bounds = list(zip(lo, hi))
    else:
        active_bounds = HIND_Q_LIM

    constraints = [{'type': 'eq',
                    'fun': lambda q: np.array(forward_kinematics(q, dh)[-1]) - p_t}]
    constraints.append({'type': 'eq',
                        'fun': lambda q: q[4] - Q_HOME_HIND[4]})

    _lm_hind = LINK_MASS_PER_LEG[2]
    if OPT_IK_USE_TAU_LIMIT:
        def _torque_ineq(q):
            tau_g = compute_gravity_torque_sim(q, dh, _lm_hind, front_leg=False)
            return JOINT_TORQUE_LIMIT[:len(tau_g)] - np.abs(tau_g)
        constraints.append({'type': 'ineq', 'fun': _torque_ineq})

    def cost(q):
        c_qref = LAMBDA_Q_OPT * np.dot(q - q_tgt, q - q_tgt)
        tau_g  = compute_gravity_torque_sim(q, dh, _lm_hind, front_leg=False)
        c_tau  = LAMBDA_TAU_OPT * np.dot(tau_g, tau_g)
        return float(c_qref + c_tau)

    res = _sp_minimize(cost, q0, method='SLSQP', bounds=active_bounds,
                       constraints=constraints,
                       options={'ftol': 1e-8, 'maxiter': OPT_IK_MAXITER})
    tip_final = np.array(forward_kinematics(res.x, dh)[-1])
    pos_err_sq = float(np.dot(tip_final - p_t, tip_final - p_t))
    if pos_err_sq < 1e-6:
        return list(res.x), res.nit, pos_err_sq
    return None, res.nit, pos_err_sq


def compute_jacobian_sim(thetas, dh, front_leg):
    n = len(thetas)
    T = np.eye(4)
    origins_dh = [np.zeros(3)]
    z_axes_dh  = [np.array([0.0, 0.0, 1.0])]
    for i in range(n):
        alpha, a, d = dh[i]
        T = T @ _dh_matrix(alpha, a, d, thetas[i])
        origins_dh.append(T[:3, 3].copy())
        z_axes_dh.append(T[:3, 2].copy())
    origins_sim = [_dh_to_sim(p, front_leg) for p in origins_dh]
    z_axes_sim  = [_dh_to_sim(z, front_leg) for z in z_axes_dh]
    pe = origins_sim[-1]
    J  = np.zeros((3, n))
    for i in range(n):
        J[:, i] = np.cross(z_axes_sim[i], pe - origins_sim[i])
    return J


def compute_gravity_torque_sim(thetas, dh, link_mass, front_leg):
    G_VEC_SIM = np.array([0.0, 0.0, -G_ACC])
    n = len(thetas)
    T = np.eye(4)
    origins_dh = [np.zeros(3)]
    z_axes_dh  = [np.array([0.0, 0.0, 1.0])]
    for i in range(n):
        alpha, a, d = dh[i]
        T = T @ _dh_matrix(alpha, a, d, thetas[i])
        origins_dh.append(T[:3, 3].copy())
        z_axes_dh.append(T[:3, 2].copy())
    origins_sim = [_dh_to_sim(p, front_leg) for p in origins_dh]
    z_axes_sim  = [_dh_to_sim(z, front_leg) for z in z_axes_dh]
    tau_g = np.zeros(n)
    for k in range(n):
        p_com  = (origins_sim[k] + origins_sim[k+1]) / 2.0
        f_grav = link_mass[k] * G_VEC_SIM
        for j in range(k + 1):
            tau_g[j] += np.dot(np.cross(z_axes_sim[j], p_com - origins_sim[j]), f_grav)
    return tau_g

# ══════════════════════════════════════════════════════════════
# 1.5  QP GRF / MPC QP
# ══════════════════════════════════════════════════════════════

def _skew(v):
    """3D 벡터 → 반대칭(skew-symmetric) 행렬"""
    return np.array([[ 0.0,  -v[2],  v[1]],
                     [ v[2],  0.0,  -v[0]],
                     [-v[1],  v[0],  0.0 ]], dtype=float)


# ── Phase 3: RNEA ────────────────────────────────────────────

def _rod_inertia_local(mass, length, radius=LINK_RADIUS):
    """원통 막대 관성 텐서 (로컬 프레임, x축 = 막대 축) [kg·m²]
    Ixx(축 방향): m·r²/2   Iyy=Izz(수직): m(3r²+L²)/12
    """
    Ixx = 0.5 * mass * radius ** 2
    Iyy = mass * (3.0 * radius**2 + length**2) / 12.0
    return np.diag([Ixx, Iyy, Iyy])


def rnea(q, dq, ddq, dh, link_mass):
    """Phase 3 — RNEA:  tau = M(q)·q̈ + C(q,q̇)·q̇ + g(q)

    DH 세계 프레임에서 순·역방향 재귀 계산.
    중력 처리: sim [0,0,-g] → DH [-g,0,0]  →  기저 등가 가속도 a0=[g,0,0]

    q, dq, ddq : (n,) 관절 각도·속도·가속도
    Returns    : tau (n,) — 완전 강체 동역학 관절 토크 [N·m]
    """
    n  = len(q)
    a0 = np.array([G_ACC, 0.0, 0.0])   # 기저 등가 가속도 (중력 포함)

    # FK: 관절 원점·z축·회전행렬 (DH 세계 프레임)
    T       = np.eye(4)
    R_list  = [np.eye(3)]               # R_list[i] = frame{i} 회전 in world
    origins = [np.zeros(3)]
    z_axes  = [np.array([0., 0., 1.])]
    for i in range(n):
        alpha, a, d = dh[i]
        T = T @ _dh_matrix(alpha, a, d, q[i])
        R_list.append(T[:3, :3].copy())
        origins.append(T[:3, 3].copy())
        z_axes.append(T[:3, 2].copy())

    # COM 위치, 링크 관성 텐서 (세계 프레임)
    p_com   = [(origins[i] + origins[i+1]) * 0.5 for i in range(n)]
    I_world = []
    for i in range(n):
        a_len = dh[i][1]; d_off = dh[i][2]
        L     = math.sqrt(a_len**2 + d_off**2)
        Iloc  = _rod_inertia_local(link_mass[i], L)
        R     = R_list[i]
        I_world.append(R @ Iloc @ R.T)

    # ── Forward Pass ──────────────────────────────────────────
    omega  = np.zeros(3)
    alpha_ = np.zeros(3)
    a_orig = a0.copy()
    omega_list = []; alpha_list = []; a_c_list = []

    for i in range(n):
        zi    = z_axes[i]
        w_new = omega  + dq[i]  * zi
        a_new = alpha_ + ddq[i] * zi + np.cross(omega, dq[i] * zi)

        r_jnt   = origins[i+1] - origins[i]
        a_o_new = (a_orig
                   + np.cross(a_new, r_jnt)
                   + np.cross(w_new, np.cross(w_new, r_jnt)))

        r_com = p_com[i] - origins[i]
        a_c   = (a_orig
                 + np.cross(a_new, r_com)
                 + np.cross(w_new, np.cross(w_new, r_com)))

        omega_list.append(w_new.copy())
        alpha_list.append(a_new.copy())
        a_c_list.append(a_c.copy())
        omega = w_new; alpha_ = a_new; a_orig = a_o_new

    # ── Backward Pass ─────────────────────────────────────────
    f_out = np.zeros(3)
    n_out = np.zeros(3)
    tau   = np.zeros(n)

    for i in range(n - 1, -1, -1):
        mi     = link_mass[i]
        Ii     = I_world[i]
        r_com  = p_com[i]     - origins[i]
        r_next = origins[i+1] - origins[i]

        f_i = mi * a_c_list[i] + f_out
        n_i = (Ii @ alpha_list[i]
               + np.cross(omega_list[i], Ii @ omega_list[i])
               + np.cross(r_com,  mi * a_c_list[i])
               + np.cross(r_next, f_out)
               + n_out)

        tau[i] = np.dot(z_axes[i], n_i)
        f_out  = f_i
        n_out  = n_i

    return tau


# ── Phase 5: WBIC QP (per-leg baseline) ──────────────────────

def compute_mh_leg(q, dq, dh, lm):
    """Mass matrix M(q) (n×n)과 h(q,q̇)=C·q̇+g(q) (n,)를 RNEA로 추출.
    g(q) = RNEA(q, 0, 0)
    h    = RNEA(q, q̇, 0)
    M[:,j] = RNEA(q, 0, e_j) - g(q)   (composite rigid body 단위가속도 trick)
    """
    n = len(q)
    zero = np.zeros(n)
    g_vec = rnea(q, zero, zero, dh, lm)
    h_vec = rnea(q, dq,   zero, dh, lm)
    M = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n); ej[j] = 1.0
        M[:, j] = rnea(q, zero, ej, dh, lm) - g_vec
    return M, h_vec


def wbic_qp_leg(M, h, ddq_des, tau_ff, lam_des, J, contact, nj,
                w_ddq, w_tau, w_lam, lamz_min, mu,
                tau_prev=None, w_dtau=0.0):
    """Per-leg WBIC QP correction.

    변수 : x = [Δq̈ (nj); Δτ (nj); Δλ (3)]
    비용 : w_ddq‖Δq̈‖² + w_tau‖Δτ‖² + w_lam‖Δλ‖²
           [+ w_dtau‖(tau_ff+Δτ) − tau_prev‖² if w_dtau>0 and tau_prev given]
    등식 : M·Δq̈ - Δτ - Jᵀ·Δλ = r,  r = tau_ff + Jᵀ·λ_des - M·ddq_des - h
    부등 : τ_min ≤ tau_ff+Δτ ≤ τ_max
           stance: λ_z+Δλ_z ≥ lamz_min,  |λ_x,y+Δλ_x,y| ≤ μ(λ_z+Δλ_z)
           swing : Δλ = -λ_des  (λ=0 고정)

    Returns (dq̈, dτ, dλ, success, residual_pre)
    """
    n_v = nj + nj + 3
    P = np.diag([w_ddq]*nj + [w_tau]*nj + [w_lam]*3).astype(float)
    qv = np.zeros(n_v)

    # τ smoothness: ‖(tau_ff + Δτ) − tau_prev‖²
    #   = ‖Δτ + c‖²  where c = tau_ff − tau_prev
    #   → P[Δτ] += w_dtau·I,  qv[Δτ] += w_dtau·c
    if (tau_prev is not None) and (w_dtau > 0.0):
        c = tau_ff - tau_prev
        sl_dt = slice(nj, 2*nj)
        P[sl_dt, sl_dt] += w_dtau * np.eye(nj)
        qv[sl_dt]       += w_dtau * c

    # 등식
    A_eq = np.hstack([M, -np.eye(nj), -J.T])
    r    = tau_ff + J.T @ lam_des - M @ ddq_des - h
    b_eq = r.copy()
    residual_pre = float(np.linalg.norm(r))

    # bounds
    lb = np.full(n_v, -1e8)
    ub = np.full(n_v,  1e8)
    lb[nj:2*nj] = -JOINT_TORQUE_LIMIT[:nj] - tau_ff
    ub[nj:2*nj] =  JOINT_TORQUE_LIMIT[:nj] - tau_ff

    G = None; h_ineq = None
    if contact:
        # λ_z + Δλ_z ≥ lamz_min  →  Δλ_z ≥ lamz_min - λ_z (bound 사용)
        lb[2*nj + 2] = max(lb[2*nj + 2], lamz_min - lam_des[2])
        # 마찰 추 4개 부등식
        rows = []; rhs = []
        # +Δλ_x - μ·Δλ_z ≤ μ·λ_z - λ_x
        rows.append([(2*nj + 0,  1.0), (2*nj + 2, -mu)]); rhs.append(mu*lam_des[2] - lam_des[0])
        rows.append([(2*nj + 0, -1.0), (2*nj + 2, -mu)]); rhs.append(mu*lam_des[2] + lam_des[0])
        rows.append([(2*nj + 1,  1.0), (2*nj + 2, -mu)]); rhs.append(mu*lam_des[2] - lam_des[1])
        rows.append([(2*nj + 1, -1.0), (2*nj + 2, -mu)]); rhs.append(mu*lam_des[2] + lam_des[1])
        G = np.zeros((4, n_v))
        for i, row in enumerate(rows):
            for idx, val in row:
                G[i, idx] = val
        h_ineq = np.array(rhs, dtype=float)
    else:
        # swing: Δλ = -λ_des (Δλ를 정확한 값으로 고정)
        lb[2*nj:2*nj+3] = -lam_des
        ub[2*nj:2*nj+3] = -lam_des

    try:
        sol = qpsolvers.solve_qp(P, qv, G, h_ineq, A_eq, b_eq, lb, ub, solver='quadprog')
    except Exception:
        sol = None
    if sol is None:
        return None, None, None, False, residual_pre
    return sol[:nj], sol[nj:2*nj], sol[2*nj:], True, residual_pre


# ── v11 Phase 7: Floating-Base 동역학 + WBIC FB ───────────────

def _skew(v):
    """3-vector → 3×3 skew-symmetric (cross product matrix)."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]], dtype=float)


def _exp_so3(omega_dt):
    """Lie group exponential: skew-vec → SO(3) rotation matrix.
    Rodrigues' formula. ω·dt 입력, ‖ω·dt‖ < ε이면 I 반환.
    NaN/inf/극대값 입력은 발산 발생 시 안전하게 I 반환 (math domain error 방지).
    """
    if not np.all(np.isfinite(omega_dt)):
        return np.eye(3)
    angle = float(np.linalg.norm(omega_dt))
    if angle > 1e2:   # ω·dt > 100 rad는 비물리적 → 발산 가드
        return np.eye(3)
    if angle < 1e-12:
        return np.eye(3)
    axis = omega_dt / angle
    K = _skew(axis)
    return np.eye(3) + math.sin(angle)*K + (1.0 - math.cos(angle))*(K @ K)


def _R_to_euler_xyz(R):
    """Rotation matrix → (roll, pitch, yaw) [rad].
    XYZ 고정축 회전 순서 가정 (small-angle MPC 모델과 일관).
    pitch 특이점(±π/2) 부근에서는 atan2 fallback.
    """
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(R[2, 0]) < 0.99999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw  = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw  = 0.0
    return roll, pitch, yaw


def integrate_body_state(body_state, lam_used_all, foot_world_all,
                         M_total, I_body, dt):
    """
    Floating-base 6-DoF 동역학 적분 (symplectic Euler).

    Args:
        body_state: dict with keys 'pos','R','v','omega'
        lam_used_all: (4, 3) — 각 발의 GRF (world frame)
        foot_world_all: (4, 3) — 각 발의 world position (CoM 기준 r_i 계산용)
        M_total: 전체 질량 [kg]
        I_body: body inertia tensor [kg·m²] (body frame)
        dt: 시간 간격 [s]

    Returns:
        업데이트된 body_state (in-place 수정 + 반환)
    """
    pos   = body_state['pos']
    R     = body_state['R']
    v     = body_state['v']
    omega = body_state['omega']

    # Linear: M·v̇ = Σλ + M·g_world
    F_grf  = np.sum(lam_used_all, axis=0)
    F_grav = np.array([0.0, 0.0, -M_total * G_ACC])
    a_lin  = (F_grf + F_grav) / M_total

    # Angular about CoM: I_world·ω̇ + ω×(I_world·ω) = Σ r_i × λ_i
    I_world = R @ I_body @ R.T
    tau_com = np.zeros(3)
    for i in range(4):
        r_i = foot_world_all[i] - pos
        tau_com += np.cross(r_i, lam_used_all[i])
    rhs = tau_com - np.cross(omega, I_world @ omega)
    a_ang = np.linalg.solve(I_world, rhs)

    # Symplectic Euler: 속도 먼저 업데이트, 위치/회전은 새 속도로 적분
    v_new     = v     + dt * a_lin
    omega_new = omega + dt * a_ang

    # 발산 가드: 비물리적 영역 도달 시 clamp (open-loop에서 body 발산 빈번)
    OMEGA_LIMIT = 50.0   # rad/s (실 robot은 거의 도달 불가)
    V_LIMIT     = 50.0   # m/s
    omega_norm = float(np.linalg.norm(omega_new))
    if omega_norm > OMEGA_LIMIT:
        omega_new = omega_new * (OMEGA_LIMIT / omega_norm)
        body_state['_diverged'] = True
    v_norm = float(np.linalg.norm(v_new))
    if v_norm > V_LIMIT:
        v_new = v_new * (V_LIMIT / v_norm)
        body_state['_diverged'] = True
    if not np.all(np.isfinite(omega_new)):
        omega_new = np.zeros(3)
        body_state['_diverged'] = True
    if not np.all(np.isfinite(v_new)):
        v_new = np.zeros(3)
        body_state['_diverged'] = True

    pos_new   = pos   + dt * v_new
    R_new     = _exp_so3(omega_new * dt) @ R

    body_state['pos']   = pos_new
    body_state['R']     = R_new
    body_state['v']     = v_new
    body_state['omega'] = omega_new
    body_state['a_lin'] = a_lin
    body_state['a_ang'] = a_ang
    return body_state


def wbic_qp_full_pin(M_full, h_full, J_full,
                      ddq_des_legs, tau_ff_legs, lam_des_all,
                      contact_mask, nj_per_leg,
                      v_dot_des_fb,
                      w_ddq, w_tau, w_lam, w_fb, lamz_min, mu):
    """
    v11 Phase 7 (FULL M version) — pinocchio가 제공한 26×26 M, 26 h, 12×26 J 사용.

    변수: x = [Δv̇_fb (6); Δq̈_legs (Σ nj); Δτ_legs (Σ nj); Δλ (12)]

    등식 (26개): M_full·Δq̈_full − [0(6); Δτ_legs] − J_full^T · Δλ = r_full
        Δq̈_full = [Δv̇_fb; Δq̈_legs]   (26-dim)
        r_full   = τ_ff_full + J_full^T·λ_des − M_full·ddq_des_full − h_full
        τ_ff_full = [0(6); τ_ff_legs] (base는 actuator 없음)
        ddq_des_full = [v_dot_des_fb; ddq_des_legs]

    이 형태는 floating base + leg coupling을 행렬에 자동 반영.
    block-diagonal 근사 wbic_qp_full 보다 정확.

    Note: pinocchio LOCAL convention (FB v는 body frame). 호출자가 body 상태를
          적절히 변환해서 넘기되, R=I 근사면 world와 동일.
    """
    n_legs = 4
    nj_total = sum(nj_per_leg[:n_legs])
    n_fb  = 6
    n_v   = n_fb + nj_total + nj_total + 12
    n_eq  = n_fb + nj_total   # 26 = base 6 + legs 20

    sl_fb  = slice(0, n_fb)
    sl_ddq = slice(n_fb, n_fb + nj_total)
    sl_tau = slice(n_fb + nj_total, n_fb + 2*nj_total)
    sl_lam = slice(n_fb + 2*nj_total, n_v)
    leg_off = [sum(nj_per_leg[:i]) for i in range(n_legs)]

    # 비용
    P = np.zeros((n_v, n_v), dtype=float)
    P[sl_fb,  sl_fb]  = w_fb  * np.eye(n_fb)
    P[sl_ddq, sl_ddq] = w_ddq * np.eye(nj_total)
    P[sl_tau, sl_tau] = w_tau * np.eye(nj_total)
    P[sl_lam, sl_lam] = w_lam * np.eye(12)
    qv = np.zeros(n_v, dtype=float)

    # 등식: M_full·Δq̈_full − [0(6); Δτ_legs] − J^T · Δλ = r_full
    # 변수 layout: [Δv̇_fb(6) | Δq̈_legs(20) | Δτ_legs(20) | Δλ(12)]
    # M_full @ Δq̈_full = M_full[:, 0:6] @ Δv̇_fb + M_full[:, 6:26] @ Δq̈_legs
    A_eq = np.zeros((n_eq, n_v), dtype=float)
    A_eq[:, sl_fb]  = M_full[:, 0:6]
    A_eq[:, sl_ddq] = M_full[:, 6:26]
    # τ 부분: Δτ_full = [0(6); Δτ_legs] → -A_eq[:, sl_tau] = [0(6); -I(20)]
    A_eq[6:26, sl_tau] = -np.eye(nj_total)
    # J_full^T · Δλ
    A_eq[:, sl_lam] = -J_full.T

    # ddq_des_full, τ_ff_full
    ddq_des_full = np.concatenate([v_dot_des_fb, *ddq_des_legs])
    tau_ff_full  = np.concatenate([np.zeros(n_fb), *tau_ff_legs])
    lam_des_flat = lam_des_all.reshape(-1)
    # residual r_full = τ_ff + J^T·λ_des − M·ddq_des − h
    r_full = tau_ff_full + J_full.T @ lam_des_flat - M_full @ ddq_des_full - h_full
    b_eq = r_full

    # bounds
    lb = np.full(n_v, -1e8)
    ub = np.full(n_v,  1e8)
    for i in range(n_legs):
        nj = nj_per_leg[i]
        tau_lim = JOINT_TORQUE_LIMIT[:nj]
        s_off = sl_tau.start + leg_off[i]
        lb[s_off:s_off+nj] = -tau_lim - tau_ff_legs[i]
        ub[s_off:s_off+nj] =  tau_lim - tau_ff_legs[i]

    # 부등식: 마찰 추 (stance) + bounds로 swing Δλ=−λ_des 고정
    G_list = []; h_list = []
    for i in range(n_legs):
        l_off = sl_lam.start + 3*i
        if contact_mask[i]:
            lb[l_off + 2] = max(lb[l_off + 2], lamz_min - lam_des_all[i, 2])
            for sgn_x, sgn_y in [(+1, 0), (-1, 0), (0, +1), (0, -1)]:
                row = np.zeros(n_v)
                row[l_off + 0] = sgn_x
                row[l_off + 1] = sgn_y
                row[l_off + 2] = -mu
                rhs = mu * lam_des_all[i, 2] - sgn_x * lam_des_all[i, 0] - sgn_y * lam_des_all[i, 1]
                G_list.append(row); h_list.append(rhs)
        else:
            for k in range(3):
                lb[l_off + k] = -lam_des_all[i, k]
                ub[l_off + k] = -lam_des_all[i, k]

    G_ineq = np.vstack(G_list) if G_list else None
    h_ineq = np.array(h_list) if h_list else None

    try:
        sol = qpsolvers.solve_qp(P, qv, G_ineq, h_ineq, A_eq, b_eq, lb, ub, solver='quadprog')
    except Exception:
        sol = None

    if sol is None:
        return None
    return {
        'd_v_fb': sol[sl_fb],
        'd_ddq_legs': [sol[sl_ddq.start + leg_off[i] : sl_ddq.start + leg_off[i] + nj_per_leg[i]]
                       for i in range(n_legs)],
        'd_tau_legs': [sol[sl_tau.start + leg_off[i] : sl_tau.start + leg_off[i] + nj_per_leg[i]]
                       for i in range(n_legs)],
        'd_lam_legs': [sol[sl_lam.start + 3*i : sl_lam.start + 3*i + 3] for i in range(n_legs)],
        'residual_full': float(np.linalg.norm(r_full)),
    }


def wbic_qp_full(M_legs, h_legs, ddq_des_legs, tau_ff_legs, lam_des_all,
                 J_legs, contact_mask, nj_per_leg, foot_world_all, body_pos,
                 v_dot_des_fb,
                 M_total, I_body, omega_world,
                 w_ddq, w_tau, w_lam, w_fb, lamz_min, mu,
                 stance_foot_J_v11=None,
                 tau_prev_legs=None, w_dtau=0.0):
    """
    v11 Phase 7 — WBIC 부유 베이스 통합 단일 QP.

    변수 (총 nv = 6 + 4·nj·2 + 4·3):
        x = [ Δv̇_fb (6); Δq̈ (sum_nj); Δτ (sum_nj); Δλ (12) ]

    비용:
        w_fb·||Δv̇_fb||² + w_ddq·||Δq̈||² + w_tau·||Δτ||² + w_lam·||Δλ||²

    등식:
        body 6-DoF (6):
            M_fb·(v̇_fb_des + Δv̇_fb) = F_des(λ_des) + ΔF(Δλ)
            → M_fb·Δv̇_fb − [I_3⊗4 ; r̂_i⊗4]·Δλ = F_des − M_fb·v̇_fb_des
            (M_fb = block_diag(M·I3, I_world);  r̂_i = skew(foot_i - CoM))
        per-leg dyn (Σ nj):
            M_i·Δq̈_i − Δτ_i − Jᵀ_i·Δλ_i = r_i
            r_i = tau_ff_i + Jᵀ_i·λ_des_i − M_i·ddq_des_i − h_i

    부등:
        per-leg torque limits, friction cone, λ_z ≥ lamz_min (stance only)
        swing: Δλ = −λ_des (bound으로 고정)

    body 6-DoF residual (좌변 검증용; 작을수록 MPC와 정합)도 반환.
    """
    # 인덱스 헬퍼
    n_legs = 4
    nj_total = sum(nj_per_leg[:n_legs])
    n_fb  = 6
    n_ddq = nj_total
    n_tau = nj_total
    n_lam = 12
    n_v   = n_fb + n_ddq + n_tau + n_lam

    # 슬라이스
    sl_fb  = slice(0, n_fb)
    sl_ddq = slice(n_fb, n_fb + n_ddq)
    sl_tau = slice(n_fb + n_ddq, n_fb + n_ddq + n_tau)
    sl_lam = slice(n_fb + n_ddq + n_tau, n_v)
    leg_off_ddq = [sum(nj_per_leg[:i]) for i in range(n_legs)]   # leg i의 ddq 시작 (within sl_ddq)
    leg_off_tau = leg_off_ddq                                     # 동일 사이즈
    leg_off_lam = [3*i for i in range(n_legs)]

    # 비용
    P = np.zeros((n_v, n_v), dtype=float)
    P[sl_fb,  sl_fb]  = w_fb  * np.eye(n_fb)
    P[sl_ddq, sl_ddq] = w_ddq * np.eye(n_ddq)
    P[sl_tau, sl_tau] = w_tau * np.eye(n_tau)
    P[sl_lam, sl_lam] = w_lam * np.eye(n_lam)
    qv = np.zeros(n_v, dtype=float)

    # τ smoothness penalty: w_dtau·‖(tau_ff + Δτ) − tau_prev‖²  per leg.
    # Expansion: ‖Δτ + c‖² with c = tau_ff − tau_prev  →  P[Δτ]+=w·I, qv[Δτ]+=w·c
    if (tau_prev_legs is not None) and (w_dtau > 0.0):
        for i in range(n_legs):
            nj = nj_per_leg[i]
            tau_prev_i = tau_prev_legs[i]
            if tau_prev_i is None:
                continue
            c_i = tau_ff_legs[i] - tau_prev_i
            s_off = sl_tau.start + leg_off_tau[i]
            sl_i  = slice(s_off, s_off + nj)
            P[sl_i, sl_i] += w_dtau * np.eye(nj)
            qv[sl_i]      += w_dtau * c_i

    # ── 등식 1: body 6-DoF ─────────────────────────────────────
    # M_fb·v̇_fb_des = F_des  (이 등식이 정확히 만족되면 RHS=0)
    # M_fb = [M·I3, 0; 0, I_world]
    I_world = np.zeros((3,3))
    # body_R는 호출자가 omega와 함께 넘겨야 정확. 여기선 I_body 자체 사용 (world ≈ body 가정).
    # → 호출자가 R·I_body·R.T를 미리 넣어 호출하도록 (인터페이스 단순화 위해 I_body 그대로 사용)
    I_world = I_body.copy()
    M_fb_lin = M_total * np.eye(3)
    M_fb     = np.zeros((6, 6))
    M_fb[:3, :3] = M_fb_lin
    M_fb[3:, 3:] = I_world

    # F_des: linear = Σ λ_des + M·g, angular = Σ r_i × λ_des_i − ω×(I·ω)
    F_lin_des = np.sum(lam_des_all, axis=0) + np.array([0.0, 0.0, -M_total*G_ACC])
    F_ang_des = -np.cross(omega_world, I_world @ omega_world)
    for i in range(n_legs):
        r_i = foot_world_all[i] - body_pos
        F_ang_des += np.cross(r_i, lam_des_all[i])
    F_des = np.concatenate([F_lin_des, F_ang_des])

    # 등식 RHS: M_fb·v̇_fb_des − F_des  (= 0이어야 일관)
    rhs_fb = M_fb @ v_dot_des_fb - F_des

    A_eq_fb = np.zeros((6, n_v))
    A_eq_fb[:, sl_fb] = M_fb
    # ΔF coupling: -[I_3 -r̂_i ; ...] · Δλ_i (각 다리)
    for i in range(n_legs):
        r_i = foot_world_all[i] - body_pos
        col = sl_lam.start + leg_off_lam[i]
        A_eq_fb[:3, col:col+3] += -np.eye(3)
        A_eq_fb[3:, col:col+3] += -_skew(r_i)
    b_eq_fb = -rhs_fb   # M_fb·Δv̇_fb − ΔF = -rhs_fb  (즉, F_des + ΔF = M_fb·(v̇_des+Δv̇))

    # ── 등식 2: per-leg dynamics ─────────────────────────────────
    A_eq_legs_list = []
    b_eq_legs_list = []
    residual_legs = np.zeros(n_legs)
    for i in range(n_legs):
        nj = nj_per_leg[i]
        Mi  = M_legs[i]
        hi  = h_legs[i]
        Ji  = J_legs[i]
        ddq_des_i = ddq_des_legs[i]
        tau_ff_i  = tau_ff_legs[i]
        lam_des_i = lam_des_all[i]
        r_i_resid = tau_ff_i + Ji.T @ lam_des_i - Mi @ ddq_des_i - hi

        Ai = np.zeros((nj, n_v))
        Ai[:, sl_ddq.start + leg_off_ddq[i] : sl_ddq.start + leg_off_ddq[i] + nj] = Mi
        Ai[:, sl_tau.start + leg_off_tau[i] : sl_tau.start + leg_off_tau[i] + nj] = -np.eye(nj)
        Ai[:, sl_lam.start + leg_off_lam[i] : sl_lam.start + leg_off_lam[i] + 3]  = -Ji.T
        A_eq_legs_list.append(Ai)
        b_eq_legs_list.append(r_i_resid)
        residual_legs[i] = float(np.linalg.norm(r_i_resid))

    A_eq = np.vstack([A_eq_fb] + A_eq_legs_list)
    b_eq = np.concatenate([b_eq_fb] + b_eq_legs_list)

    # v11 ANYmal-style: stance foot acceleration = 0 (SOFT constraint via cost)
    # Cost 추가: w_sf · ||J_stance · Δq̈_full||²
    # → Hessian P에 w_sf·J^T·J 가 더해지고, friction cone과 충돌 시 자동 trade-off
    if stance_foot_J_v11 is not None:
        W_STANCE = 1.0   # 가중치 (1.0 = 보통, 100 = 매우 strict, 0.1 = 약함)
        for i in range(n_legs):
            if contact_mask[i]:
                Ji = stance_foot_J_v11[i]   # 3×26 (FB 6 + 20 legs v11 order)
                # Δq̈_full = [Δv̇_fb (6) | Δq̈_legs (sl_ddq, FR/FL/HR/HL)]
                J_aug = np.zeros((3, n_v))
                J_aug[:, :6]      = Ji[:, :6]
                J_aug[:, sl_ddq]  = Ji[:, 6:]
                # cost: ||J_aug · x||² → P += J_aug^T·J_aug, no linear term
                P += W_STANCE * (J_aug.T @ J_aug)

    # ── bounds ────────────────────────────────────────────────
    lb = np.full(n_v, -1e8)
    ub = np.full(n_v,  1e8)
    for i in range(n_legs):
        nj = nj_per_leg[i]
        tau_lim = JOINT_TORQUE_LIMIT[:nj]
        s_off = sl_tau.start + leg_off_tau[i]
        lb[s_off : s_off + nj] = -tau_lim - tau_ff_legs[i]
        ub[s_off : s_off + nj] =  tau_lim - tau_ff_legs[i]

    # ── 부등식: 마찰 추 (stance) ────────────────────────────────
    G_ineq_list = []
    h_ineq_list = []
    for i in range(n_legs):
        l_off = sl_lam.start + leg_off_lam[i]
        if contact_mask[i]:
            # λ_z + Δλ_z ≥ lamz_min → bound으로 적용
            lb[l_off + 2] = max(lb[l_off + 2], lamz_min - lam_des_all[i, 2])
            # 마찰: ±Δλ_x − μ·Δλ_z ≤ μ·λ_z ∓ λ_x   (4 행)
            mu_l = mu
            for sgn_x, sgn_y in [(+1, 0), (-1, 0), (0, +1), (0, -1)]:
                row = np.zeros(n_v)
                row[l_off + 0] = sgn_x
                row[l_off + 1] = sgn_y
                row[l_off + 2] = -mu_l
                rhs = mu_l * lam_des_all[i, 2] - sgn_x * lam_des_all[i, 0] - sgn_y * lam_des_all[i, 1]
                G_ineq_list.append(row)
                h_ineq_list.append(rhs)
        else:
            # swing: Δλ = -λ_des (bound으로 정확 고정)
            for k in range(3):
                lb[l_off + k] = -lam_des_all[i, k]
                ub[l_off + k] = -lam_des_all[i, k]

    G_ineq = np.vstack(G_ineq_list) if G_ineq_list else None
    h_ineq = np.array(h_ineq_list) if h_ineq_list else None

    try:
        sol = qpsolvers.solve_qp(P, qv, G_ineq, h_ineq, A_eq, b_eq, lb, ub, solver='quadprog')
    except Exception:
        sol = None

    if sol is None:
        return None
    out = {
        'd_v_fb': sol[sl_fb],
        'd_ddq_legs': [sol[sl_ddq.start + leg_off_ddq[i] : sl_ddq.start + leg_off_ddq[i] + nj_per_leg[i]]
                       for i in range(n_legs)],
        'd_tau_legs': [sol[sl_tau.start + leg_off_tau[i] : sl_tau.start + leg_off_tau[i] + nj_per_leg[i]]
                       for i in range(n_legs)],
        'd_lam_legs': [sol[sl_lam.start + leg_off_lam[i] : sl_lam.start + leg_off_lam[i] + 3]
                       for i in range(n_legs)],
        'residual_legs': residual_legs,
        'residual_fb':   float(np.linalg.norm(rhs_fb)),
    }
    return out


def qp_grf_distribute(contact_mask, foot_pos_world):
    """
    Phase 2 — 단일 스텝 QP GRF 배분 (MPC fallback)
    힘 평형 + 모멘트 평형 + 마찰 추 구속 하에서 ||λ||² 최소화

    contact_mask    : (4,) bool  — True = stance foot
    foot_pos_world  : (4, 3)    — world frame 발 위치
    Returns         : lam_des (4, 3)  — 각 발 [Fx, Fy, Fz]
    """
    stance = np.where(contact_mask)[0]
    n_s = len(stance)
    if n_s == 0:
        return np.zeros((4, 3))

    n_var = 3 * n_s
    P = np.eye(n_var, dtype=float)
    q = np.zeros(n_var, dtype=float)

    # 등식: 힘 평형(3) + x/y 모멘트(2, n_s≥2일 때만)
    # n_s==1: 모멘트 불균형 → 힘 평형만 (λ=[0,0,Mg] 유일해)
    if n_s >= 2:
        A_eq = np.zeros((5, n_var), dtype=float)
        b_eq = np.zeros(5, dtype=float)
        b_eq[2] = TOTAL_MASS * G_ACC
        for idx, leg in enumerate(stance):
            col = idx * 3
            r   = foot_pos_world[leg]
            rx, ry, rz = r[0], r[1], r[2]
            A_eq[0:3, col:col+3] = np.eye(3)
            A_eq[3, col:col+3]   = [0.0,  -rz,  ry]   # Mx: ry·Fz − rz·Fy
            A_eq[4, col:col+3]   = [ rz,  0.0, -rx]   # My: rz·Fx − rx·Fz
    else:
        A_eq = np.zeros((3, n_var), dtype=float)
        b_eq = np.array([0.0, 0.0, TOTAL_MASS * G_ACC])
        A_eq[0:3, 0:3] = np.eye(3)

    # 부등식: 마찰 추 (5행/발)
    mu = MU_FRICTION
    G  = np.zeros((5 * n_s, n_var), dtype=float)
    h  = np.zeros(5 * n_s, dtype=float)
    for idx in range(n_s):
        col = idx * 3; row = idx * 5
        G[row,   col+2] = -1.0
        G[row+1, col]   =  1.0;  G[row+1, col+2] = -mu
        G[row+2, col]   = -1.0;  G[row+2, col+2] = -mu
        G[row+3, col+1] =  1.0;  G[row+3, col+2] = -mu
        G[row+4, col+1] = -1.0;  G[row+4, col+2] = -mu

    try:
        x_opt = qpsolvers.solve_qp(P, q, G, h, A_eq, b_eq, solver='quadprog')
    except Exception:
        x_opt = None

    lam_des = np.zeros((4, 3))
    if x_opt is not None:
        for idx, leg in enumerate(stance):
            lam_des[leg] = x_opt[idx*3:(idx+1)*3]
    else:
        fz = TOTAL_MASS * G_ACC / n_s
        for leg in stance:
            lam_des[leg] = [0.0, 0.0, fz]
    return lam_des


# MPC 관련 캐시 (프레임 간 재사용)
_I_inv = np.linalg.inv(BODY_INERTIA)
_I_BODY = BODY_INERTIA.copy()


def _euler_to_R(roll, pitch, yaw):
    """XYZ Euler (intrinsic) → R = Rx(r)·Ry(p)·Rz(y)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _euler_rate_T(roll, pitch):
    """world ω → Euler-rate transform (XYZ extrinsic).
    [ṙ, ṗ, ẏ] = T(r,p)·ω_world.  pitch=±π/2에서 특이점 → cos(p) clamp.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    if abs(cp) < 1e-3:
        cp = 1e-3 * (1.0 if cp >= 0 else -1.0)
    tp = sp / cp
    return np.array([
        [1.0,    sr * tp,    cr * tp],
        [0.0,    cr,         -sr    ],
        [0.0,    sr / cp,    cr / cp],
    ])


def _build_Ac_at(roll, pitch):
    """v11 LTV-MPC: 현재 자세에서의 연속 시간 A 행렬 (13×13).
    Euler rate가 작은 각도 가정(I)이 아닌 T(r,p)·ω로 정확화.
    """
    Ac = np.zeros((13, 13), dtype=float)
    Ac[0:3, 6:9]  = _euler_rate_T(roll, pitch)   # dΘ/dt = T·ω
    Ac[3:6, 9:12] = np.eye(3)                    # dp/dt = v
    Ac[9:12, 12]  = [0.0, 0.0, 1.0]              # dv/dt += g·ẑ
    return Ac


def _build_Bc_at(contact_mask_k, foot_pos_k, I_world_inv):
    """v11 LTV-MPC: B 행렬에 현재 자세의 I_world_inv 반영 (13×12)."""
    Bc = np.zeros((13, 12), dtype=float)
    for i in range(4):
        if contact_mask_k[i]:
            r = foot_pos_k[i]
            Bc[6:9,  i*3:(i+1)*3] = I_world_inv @ _skew(r)   # angular (world)
            Bc[9:12, i*3:(i+1)*3] = np.eye(3) / TOTAL_MASS    # linear
    return Bc


def _build_Ac_d():
    """[deprecated, hover-at-x0 호환용] 시불변 Ac_d — small-angle 가정 (T=I)."""
    Ac = _build_Ac_at(0.0, 0.0)
    return np.eye(13) + DT_MPC * Ac


_Ac_d = _build_Ac_d()
# Ac_d 거듭제곱 사전 계산 (0 ~ N_MPC) — closed_loop=False 경로용
_Ad_powers = [np.eye(13, dtype=float)]
for _k in range(N_MPC):
    _Ad_powers.append(_Ac_d @ _Ad_powers[-1])


def _build_Bc(contact_mask_k, foot_pos_k):
    """[deprecated, hover-at-x0 호환용] 시불변 I_inv 사용."""
    return DT_MPC * _build_Bc_at(contact_mask_k, foot_pos_k, _I_inv)


def mpc_qp_plan(x0, contact_schedule, foot_positions, x_ref_step=None, ltv=False):
    """
    Phase 1 — Convex MPC QP  (Di Carlo et al., IROS 2018 simplified)

    x0              : (13,) 현재 body 상태
                      [roll,pitch,yaw, px,py,pz, ωx,ωy,ωz, vx,vy,vz, g]
    contact_schedule: (N_MPC, 4) bool  — horizon 내 접촉 패턴
    foot_positions  : (N_MPC, 4, 3)   — horizon 내 발 위치 (world frame)
    x_ref_step      : (13,) or None — horizon 내 모든 스텝의 목표 상태 (steady ref).
                      None이면 hover-at-x0 (open-loop 동작, v10 호환).
                      ≠None이면 closed-loop: MPC가 x0을 x_ref_step으로 끌어옴.
    ltv             : True면 v11 Linear Time-Varying — 매 호출마다 현재 자세 (x0의 r,p,y)로
                      A_d, I_world_inv 재계산 (큰 각도 정확). 단, horizon 내 R은 고정.
                      False면 small-angle 시불변 (캐시된 _Ad_powers 사용, 빠름).
    Returns         : lam_des (4, 3)  — 첫 번째 스텝 최적 GRF
    """
    nx = 13
    nu = 12   # 4 feet × 3

    # ── A_d, B_c 빌드 ─────────────────────────────────
    if ltv:
        # 현재 자세에서 LTV 모델 (horizon 내 R 고정 가정)
        roll0, pitch0, yaw0 = x0[0], x0[1], x0[2]
        R_now      = _euler_to_R(roll0, pitch0, yaw0)
        I_world    = R_now @ _I_BODY @ R_now.T
        I_world_inv = np.linalg.inv(I_world)
        Ac_now     = _build_Ac_at(roll0, pitch0)
        Ad_now     = np.eye(nx) + DT_MPC * Ac_now
        # horizon 내 거듭제곱 (재계산)
        Ad_powers_now = [np.eye(nx, dtype=float)]
        for _k in range(N_MPC):
            Ad_powers_now.append(Ad_now @ Ad_powers_now[-1])
        Bc_list = [DT_MPC * _build_Bc_at(contact_schedule[k], foot_positions[k], I_world_inv)
                   for k in range(N_MPC)]
        Ad_powers_use = Ad_powers_now
    else:
        # 시불변 (캐시 사용) — 작은 각도 가정
        Bc_list = [_build_Bc(contact_schedule[k], foot_positions[k]) for k in range(N_MPC)]
        Ad_powers_use = _Ad_powers

    # 응축 행렬 Aq (N*nx × nx), Bq (N*nx × N*nu)
    N   = N_MPC
    Aq  = np.zeros((N*nx, nx),   dtype=float)
    Bq  = np.zeros((N*nx, N*nu), dtype=float)
    for i in range(N):
        Aq[i*nx:(i+1)*nx, :] = Ad_powers_use[i+1]
        for j in range(i+1):
            Bq[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = Ad_powers_use[i-j] @ Bc_list[j]

    # 목표 상태: x_ref_step 우선, 없으면 hover-at-x0
    if x_ref_step is None:
        X_ref = np.tile(x0, N)
    else:
        X_ref = np.tile(x_ref_step, N)

    # 비용 H, f  — block-diagonal Q_bar 직접 누산으로 메모리 절약
    # H = 2*(Bq^T Q_bar Bq + R_bar),  f = 2*Bq^T Q_bar (Aq x0 − X_ref)
    err0  = Aq @ x0 - X_ref  # (N*nx,)
    QBq   = np.zeros_like(Bq)
    Qerr  = np.zeros(N*nx, dtype=float)
    for i in range(N):
        sl = slice(i*nx, (i+1)*nx)
        QBq[sl, :]  = MPC_Q @ Bq[sl, :]
        Qerr[sl]    = MPC_Q @ err0[sl]

    R_bar_diag = np.tile(np.diag(np.kron(np.eye(4), MPC_R)), N)
    H = 2.0 * (Bq.T @ QBq + np.diag(R_bar_diag))
    f = 2.0 * (Bq.T @ Qerr)
    H = (H + H.T) * 0.5   # 수치 대칭 보정

    # 부등식: stance foot 마찰 추 (5행/발)
    mu = MU_FRICTION
    G_list = []
    h_list = []
    for k in range(N):
        for i in range(4):
            if contact_schedule[k, i]:
                col = k*nu + i*3
                g_blk = np.zeros((5, N*nu), dtype=float)
                g_blk[0, col+2] = -1.0
                g_blk[1, col]   =  1.0;  g_blk[1, col+2] = -mu
                g_blk[2, col]   = -1.0;  g_blk[2, col+2] = -mu
                g_blk[3, col+1] =  1.0;  g_blk[3, col+2] = -mu
                g_blk[4, col+1] = -1.0;  g_blk[4, col+2] = -mu
                G_list.append(g_blk)
                h_list.append(np.zeros(5))

    G_mpc = np.vstack(G_list) if G_list else np.zeros((1, N*nu))
    h_mpc = np.concatenate(h_list) if h_list else np.zeros(1)

    # 등식: swing foot 힘 = 0  (3행/발)
    A_list = []
    b_list = []
    for k in range(N):
        for i in range(4):
            if not contact_schedule[k, i]:
                col = k*nu + i*3
                for d in range(3):
                    row = np.zeros(N*nu)
                    row[col + d] = 1.0
                    A_list.append(row)
                    b_list.append(0.0)
    A_mpc = np.array(A_list, dtype=float) if A_list else None
    b_mpc = np.array(b_list, dtype=float) if b_list else None

    try:
        u_opt = qpsolvers.solve_qp(
            P=H, q=f, G=G_mpc, h=h_mpc, A=A_mpc, b=b_mpc, solver='quadprog'
        )
    except Exception:
        u_opt = None

    lam_des = np.zeros((4, 3))
    if u_opt is not None:
        for i in range(4):
            lam_des[i] = u_opt[i*3:(i+1)*3]
    else:
        # MPC 실패 → QP GRF fallback
        lam_des = qp_grf_distribute(contact_schedule[0], foot_positions[0])
    return lam_des

# ══════════════════════════════════════════════════════════════
# 2. Gait Scheduler & Foot Trajectory
# ══════════════════════════════════════════════════════════════

class GaitScheduler:
    def __init__(self, gait=GAIT_TYPE, period=T, swing_ratio=D):
        self.period      = period
        self.swing_ratio = swing_ratio
        self.offsets     = PHASE_OFFSETS[gait]

    def phase(self, leg, t):
        return (t / self.period + self.offsets[leg]) % 1.0

    def is_swing(self, leg, t):
        return self.phase(leg, t) < self.swing_ratio

    def swing_t(self, leg, t):
        p = self.phase(leg, t)
        return p / self.swing_ratio if p < self.swing_ratio else 0.0

    def stance_t(self, leg, t):
        p = self.phase(leg, t)
        if p >= self.swing_ratio:
            return (p - self.swing_ratio) / (1.0 - self.swing_ratio)
        return 0.0


_swing_z_cache = {}

def _swing_z_coeffs(step_h, T_half, V_z_boundary):
    """Swing Z 6차 짝수 다항식 계수 [c2, c4, c6] (Zeng 2019 Eq 23-25 + jerk 연속).
    Z(u) = step_h + c2·u² + c4·u⁴ + c6·u⁶,  u ∈ [-T_half, +T_half]
    조건:
      Z(±T_half) = 0                       (양 끝 ground level)
      Z'(-T_half) = +V_z_boundary          (stance 끝 vel과 C1 연속)
      Z''(±T_half) = 0                     (stance acc=0 과 C2 연속)
    홀수 미분(Z', Z''', Z⁽⁵⁾)은 u=0에서 자동 0 → peak에서 jerk=0 (논문 6차 spline 동치)
    """
    key = (round(step_h, 9), round(T_half, 9), round(V_z_boundary, 9))
    c = _swing_z_cache.get(key)
    if c is not None:
        return c
    T = T_half
    A = np.array([
        [T**2,   T**4,    T**6   ],
        [2.0,    12.0*T**2, 30.0*T**4],
        [2.0*T,  4.0*T**3,  6.0*T**5],
    ])
    b = np.array([-step_h, 0.0, -V_z_boundary])
    c2, c4, c6 = np.linalg.solve(A, b)
    _swing_z_cache[key] = (c2, c4, c6)
    return (c2, c4, c6)


def swing_foot_pos(sw_t, p_start, p_end, body_vel,
                   step_height=STEP_HEIGHT, tau_land=TAU_LAND):
    """Swing 궤적 (Zeng 2019 Scheme I, Spline 기반).
      X: 5차 spline (양 끝 vel = -body_vel·x, acc = 0; stance 등속과 C2 연속)
      Y: 5차 smoothstep (좌우 보간)
      Z: 6차 대칭 다항식 (peak=step_height, stance Z의 vel/acc와 C2 연속, jerk 연속)
    """
    if sw_t >= tau_land:
        return p_end.copy()
    tau = sw_t / tau_land
    s5  = 10*tau**3 - 15*tau**4 + 6*tau**5     # 5차 smoothstep (vel/acc 양끝 0)
    T_local = tau_land * T_SW

    # X: stance 등속(-body_vel·x)과 vel/acc 연속.  ΔX = (X_e - X_s) + V·T_local
    DX_x = (p_end[0] - p_start[0]) + body_vel[0] * T_local
    pos = np.empty(3)
    pos[0] = p_start[0] - body_vel[0] * tau * T_local + DX_x * s5
    pos[1] = (1.0 - s5) * p_start[1] + s5 * p_end[1]

    # Z: 6차 짝수 다항식 (u = t - T_half)
    T_half = T_local / 2.0
    u = tau * T_local - T_half
    V_z_boundary = STANCE_DELTA * math.pi / T_ST   # stance Z'(end) = +Δπ/T_ST
    c2, c4, c6 = _swing_z_coeffs(step_height, T_half, V_z_boundary)
    z_offset = step_height + c2*u*u + c4*u**4 + c6*u**6   # 0 at u=±T_half, peak at u=0
    pos[2] = p_start[2] + z_offset
    return pos


def stance_foot_pos(st_t, p_contact, body_vel, stance_dur):
    """Stance 궤적 (Zeng 2019 Eq 11-13).
      X: 등속도 V·t  (지면 미끄러짐 0)
      Z: -Δ·sin(π·t/T_st)  (가상 침투, 임피던스 흡수용; swing 끝/시작과 C2 연속)
    """
    pos = p_contact - body_vel * stance_dur * st_t
    pos = pos.copy()
    pos[2] = p_contact[2] - STANCE_DELTA * math.sin(math.pi * st_t)
    return pos


def _quintic_s(tau):
    """5차 다항식: s(0)=0,s(1)=1, s'=s''=0 at endpoints → C2 보장"""
    return 10*tau**3 - 15*tau**4 + 6*tau**5


def _smootherstep(tau):
    """7차 smootherstep: τ∈[0,1]에서 0→1
    s7(τ) = -20τ⁷ + 70τ⁶ - 84τ⁵ + 35τ⁴
    boundary(τ=0,1)에서 s7=0/1, s7'=s7''=s7'''=0 → C3 연속 (jerk까지 0).
    swing1/swing2 split에 사용해 sw_t=0, 0.5, 1 모든 경계에서 jerk 매끄러움.
    """
    return -20*tau**7 + 70*tau**6 - 84*tau**5 + 35*tau**4


def foot_pos_at_phase(phase, p_start, p_contact, p_end, body_vel,
                      swing_ratio=D, step_height=STEP_HEIGHT, tau_land=TAU_LAND, stance_dur=T_ST):
    """
    Swing/Stance 통합 궤적 (Zeng 2019 Scheme I).
      Swing X: 5차 spline (양 끝 vel = -V, acc = 0; stance 등속과 C2 연속)
      Swing Z: 6차 대칭 다항식 (peak=step_height, vel/acc/jerk 모두 stance와 매칭)
      Stance X: 등속도 V·t (지면 미끄러짐 0)
      Stance Z: -Δ·sin(π·st_t) (가상 침투; swing Z의 양 끝과 C2 연속)
    """
    if phase < swing_ratio:
        sw_t = phase / swing_ratio
        return swing_foot_pos(sw_t, p_start, p_end, body_vel,
                              step_height=step_height, tau_land=tau_land)
    else:
        st_t = (phase - swing_ratio) / (1.0 - swing_ratio)
        return stance_foot_pos(st_t, p_contact, body_vel, stance_dur)

# ══════════════════════════════════════════════════════════════
# 3. 궤적 사전 계산
# ══════════════════════════════════════════════════════════════
N_FRAMES   = int(N_CYCLES * T / DT)
sched      = GaitScheduler()
stance_dur = T_ST
body_vel   = np.array([V, 0.0, 0.0])

home_foot_per_leg = [
    _dh_to_sim(
        forward_kinematics(Q_HOME_PER_LEG[leg], dh=LEG_DH[leg])[TRAJ_PT_IDX_PER_LEG[leg]],
        front_leg=(leg < 2)
    )
    for leg in range(4)
]
home_foot = home_foot_per_leg[0]

joint_hist = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
foot_hist  = np.zeros((N_FRAMES, 4, 3))
phase_hist = np.zeros((N_FRAMES, 4))
swing_flag = np.zeros((N_FRAMES, 4), dtype=bool)
foot_target_world_hist = np.full((N_FRAMES, 4, 3), np.nan)   # NMPC swing target (world)
foot_actual_world_hist = np.zeros((N_FRAMES, 4, 3))            # actual foot world pos
frame_calc_time = np.zeros(N_FRAMES, dtype=float)

JOINT_VEL_LIMIT_RAD_S   = CFG.joint_vel_limit_rad_s.astype(float)
JOINT_TORQUE_LIMIT      = CFG.joint_torque_limit
VEL_LIMIT_MARGIN  = CFG.vel_limit_margin

# ── Phase 4: Optimization-based IK 파라미터 ──────────────────
LAMBDA_Q_OPT   = CFG.lambda_q_opt
LAMBDA_TAU_OPT = CFG.lambda_tau_opt
OPT_IK_MAXITER = CFG.opt_ik_maxiter

# 제약 ON/OFF — True/False 한 줄로 켜고 끔
OPT_IK_USE_VEL_LIMIT = CFG.opt_ik_use_vel_limit
OPT_IK_USE_TAU_LIMIT = CFG.opt_ik_use_tau_limit

# ── Phase 5: WBIC QP 파라미터 ────────────────────────────────
USE_WBIC      = CFG.use_wbic
WBIC_W_DDQ    = CFG.wbic_w_ddq
WBIC_W_TAU    = CFG.wbic_w_tau
WBIC_W_LAM    = CFG.wbic_w_lam
WBIC_LAMZ_MIN = CFG.wbic_lamz_min

# ── v11 Phase 7: Floating-base 동역학 + WBIC FB 파라미터 ─────
USE_BODY_DYNAMICS = CFG.use_body_dynamics
# v11 Phase 7: Pinocchio dynamics 백엔드 (URDF 우회, DH→pinocchio.Model 직접 빌드)
# True : pin_helpers.py의 rnea/compute_mh_leg/compute_jacobian_sim 사용 (검증된 정확)
# False: v11 native 함수 사용 (이전 동작)
USE_PINOCCHIO = CFG.use_pinocchio
# 실험적: USE_PINOCCHIO + USE_WBIC_FB일 때 Full M(q) (26×26 floating base 결합) 사용
# 현재 90% solver fail (per-leg τ_ff와 full M 모델 inconsistency).
# False면 pinocchio per-leg M(5×5) 사용 (block-diagonal, 안정).
USE_PINOCCHIO_FULL_M = CFG.use_pinocchio_full_m
# v11 ANYmal-style: stance foot acceleration = 0 제약 (soft cost term)
# 코드는 wbic_qp_full에 통합되어 있지만 효과 검증 결과 본질 해결 X.
# 원인: body 발산은 MPC linearization 한계라서 WBIC 제약 추가로는 못 잡음.
# 실험용 토글 (default False).
USE_STANCE_FOOT_CONSTRAINT = CFG.use_stance_foot_constraint
# v11 토글 조합:
#   (CL=False, FB=False) ← 기본. v10 호환 동작 + body state 진단 추적
#                          body는 발산 가능 (open-loop). 시각화는 VIZ='static' 권장
#   (CL=True,  FB=True)  — closed-loop 추적 (pitch<1°, z<6mm). 단 trot의 경우 roll은
#                          여전히 80°+ 까지 발산 (선형 MPC 한계). VIZ='world' 가능
#   (CL=True,  FB=False) — 위험: 선형 MPC가 큰 보정 시도→발산. 사용 권장X
#   (CL=False, FB=True)  — body 평형 강제, MPC는 idealized
USE_WBIC_FB         = CFG.use_wbic_fb
USE_MPC_CLOSED_LOOP = CFG.use_mpc_closed_loop
WBIC_W_FB           = CFG.wbic_w_fb
WBIC_W_DTAU         = CFG.wbic_w_dtau    # τ smoothness penalty weight (v13.1: 0.0 OFF default)
USE_SPLINE_DIFF     = CFG.use_spline_diff  # q̇/q̈ 미분: True=CubicSpline, False=np.gradient (v13.1: OFF default)

# v12: crocoddyl NMPC 토글
# True : 시뮬 시작 시 한번 NMPC 풀이 → trajectory 사용 (MPC+WBIC 우회)
# False: v11 동작 유지 (MPC + WBIC)
# 권장 N_CYCLES ≤ 2 (one-shot FDDP 한계, multi-cycle은 receding horizon 필요)
USE_NMPC          = CFG.use_nmpc
USE_NMPC_RECEDING = CFG.use_nmpc_receding
NMPC_W_TRACK_XY   = CFG.nmpc_w_track_xy
NMPC_W_TRACK_Z    = CFG.nmpc_w_track_z
NMPC_BAUMGARTE_KP = CFG.nmpc_baumgarte_kp
NMPC_BAUMGARTE_KD = CFG.nmpc_baumgarte_kd
NMPC_W_STATE_REG  = CFG.nmpc_w_state_reg
NMPC_W_CTRL_REG   = CFG.nmpc_w_ctrl_reg
NMPC_W_TERMINAL   = CFG.nmpc_w_terminal
NMPC_MAXITER      = CFG.nmpc_maxiter
NMPC_INIT_REG     = CFG.nmpc_init_reg
NMPC_RH_N_HORIZON = CFG.nmpc_rh_n_horizon
NMPC_RH_N_RESOLVE = CFG.nmpc_rh_n_resolve
NMPC_RH_MAXITER   = CFG.nmpc_rh_maxiter
NMPC_W_FRICTION   = CFG.nmpc_w_friction
NMPC_FRIC_NF      = CFG.nmpc_fric_nf
NMPC_FRIC_FZ_MAX  = CFG.nmpc_fric_fz_max
NMPC_W_FORCE_REG  = CFG.nmpc_w_force_reg
NMPC_W_FORCE_XY   = CFG.nmpc_w_force_xy
NMPC_W_FORCE_Z    = CFG.nmpc_w_force_z
NMPC_W_TAU_LIM    = CFG.nmpc_w_tau_lim
NMPC_W_TOUCHDOWN_V    = CFG.nmpc_w_touchdown_v
NMPC_TOUCHDOWN_LAST_N = CFG.nmpc_touchdown_last_n
NMPC_W_STANCE_POS_XY  = CFG.nmpc_w_stance_pos_xy
NMPC_W_STANCE_POS_Z   = CFG.nmpc_w_stance_pos_z
USE_PERTURBATION  = CFG.use_perturbation
PERTURB_TIME      = CFG.perturb_time
PERTURB_VEL_LIN   = CFG.perturb_vel_lin
PERTURB_VEL_ANG   = CFG.perturb_vel_ang
# 단순 body 궤적 (steady ref): upright + V 직진 + z=0
# 형식: [roll,pitch,yaw, px,py,pz, ωx,ωy,ωz, vx,vy,vz, g]
# px/py weight=0이라 자유, pz=0 추종, vx=V 추종, 자세는 0(평탄) 추종
BODY_REF_STEP = np.array([
    0.0, 0.0, 0.0,   # roll, pitch, yaw (평탄 자세)
    0.0, 0.0, 0.0,   # px, py, pz (z=0; px/py는 weight 0)
    0.0, 0.0, 0.0,   # ω
    0.0, 0.0, 0.0,   # v (vx는 main loop에서 V로 채움)
    0.0,             # g (constant)
])

# Figure 1 시각화 모드 (USE_BODY_DYNAMICS=True일 때만 효과)
#   'static'      : v10 동작 — robot 항상 원점 고정 (body state 무시) ← 기본
#   'world'       : 카메라 원점 고정, robot이 body_pos만큼 이동 + R 회전 (drift 그대로)
#   'body_follow' : 카메라가 body_pos 따라감, R 회전만 적용 (자세 진단용)
# ※ trot+open-loop+FB=False 조합에선 body roll 179°까지 발산 → 'world' 모드에서
#   다리가 거의 수평으로 누워보임. body_dyn 보고 싶으면 closed-loop+FB 둘 다 ON 권장.
VIZ_BODY_MODE = CFG.viz_body_mode
USE_SWING_QREF_BLEND = CFG.use_swing_qref_blend
# ↑ 권장: trot/pace = True (발 들기 효과), walk/amble = False (jerk 폭주 방지) or 속도한계완화

# 앞다리 관절 위치 한계 [rad]  — home: [0, 157.5, 22.5, 30.66, 59.34] deg
FRONT_Q_LIM = [
    (-math.radians(45),  math.radians(45)),   # th1: 어깨 벌림
    ( math.radians(45),  math.radians(210)),  # th2: 어깨 굴곡 (swing 고각 여유)
    (-math.radians(45),  math.radians(135)),  # th3: 팔꿈치
    (-math.radians(120), math.radians(60)),  # th4: 손목 (home +30.66 대비 마진 ~30°)
    (-math.radians(90),  math.radians(120)),  # th5: 발끝
]
# 뒷다리 관절 위치 한계 [rad]  — home: [0, -150, -90, 90, 60] deg
HIND_Q_LIM = [
    (-math.radians(45),  math.radians(45)),   # th1: 고관절 벌림
    (-math.radians(180), -math.radians(60)),  # th2: 고관절 굴곡
    (-math.radians(120),  math.radians(30)),  # th3: 무릎
    (-math.radians(30),   math.radians(150)), # th4: 발목fallback=0/500, nit_avg≈3.7 (매우 빠른 수렴)

    (-math.radians(90),   math.radians(120)), # th5: 발끝
]
MAX_TRAJ_OPT_ITERS = CFG.max_traj_opt_iters

print("─" * 55)
print(f"궤적 계산 중...  [{GAIT_TYPE}]  {N_CYCLES}사이클  {N_FRAMES}프레임")
print(f"  V={V}m/s  T={T}s  D={D}  T_SW={T_SW:.3f}s  T_ST={T_ST:.3f}s  "
      f"STEP_HEIGHT={STEP_HEIGHT*1e3:.0f}mm  STEP_LENGTH={STEP_LENGTH*1e3:.0f}mm")

traj_scale   = 1.0
height_scale = 1.0
opt_iter_used = 0
opt_ik_nit_hist      = np.zeros((N_FRAMES, 4), dtype=int)    # [FR, FL, HR, HL] 수렴 반복 횟수
opt_ik_fallback_hist = np.zeros((N_FRAMES, 4), dtype=bool)   # True = analytical fallback 사용
opt_ik_pos_err_hist  = np.full((N_FRAMES, 4), np.nan)        # opt IK 위치 오차² [m²]

for opt_iter in range(1, MAX_TRAJ_OPT_ITERS + 1):
    opt_iter_used = opt_iter
    joint_hist.fill(0.0); foot_hist.fill(0.0)
    phase_hist.fill(0.0); swing_flag.fill(False)
    frame_calc_time.fill(0.0)
    opt_ik_nit_hist.fill(0); opt_ik_fallback_hist.fill(False); opt_ik_pos_err_hist.fill(np.nan)

    _step_vec = np.array([STEP_LENGTH * traj_scale, 0.0, 0.0])
    foot_contact    = [
        home_foot_per_leg[leg].copy() + (np.zeros(3) if sched.is_swing(leg, 0) else _step_vec)
        for leg in range(4)
    ]
    foot_sw_start   = [home_foot_per_leg[leg].copy() for leg in range(4)]
    foot_local_prev = [foot_contact[leg].copy() for leg in range(4)]
    prev_swing      = [sched.is_swing(leg, 0) for leg in range(4)]

    # warm-start 초기화: 실제 fi=0 foot 위치(phase 반영) 기준으로 analytical IK
    #   → opt_ik(vel_limit OFF)로 정제 → home-near branch 수렴
    # ※ walk처럼 offsets=[0,0.5,0.75,0.25]면 fi=0에서 stance 다리마다 st_t가 다르므로
    #   foot_contact(touchdown point)가 아닌 phase별 실제 위치로 warm-start해야
    #   fi=0~수십 프레임 vel_limit hit 트랜션트 회피.
    _saved_vel_limit = OPT_IK_USE_VEL_LIMIT
    OPT_IK_USE_VEL_LIMIT = False
    prev_q_per_leg = []
    for leg in range(4):
        front_l = leg < 2
        # fi=0 실제 foot 위치 (phase 반영, swing이면 sw_t=0=p_start, stance면 st_t에 따라)
        _phase0 = sched.phase(leg, 0.0)
        _p_end0 = home_foot_per_leg[leg] + _step_vec
        _foot_loc0 = foot_pos_at_phase(
            _phase0, foot_sw_start[leg], foot_contact[leg], _p_end0,
            body_vel * traj_scale,
            swing_ratio=sched.swing_ratio,
            step_height=STEP_HEIGHT * height_scale,
            stance_dur=stance_dur,
        )
        _dh_leg = LEG_DH[leg]   # v13: 좌우 분리 dh
        if front_l:
            _foot_dh0 = _sim_to_dh(_foot_loc0 + _FRONT_J4_TO_J5_SIM, front_leg=True)
            _q_a = analytical_ik_front(_foot_dh0[0], _foot_dh0[1], _foot_dh0[2],
                                       PHI_FRONT, THETA5_FRONT, dh=_dh_leg)
            _q_init0 = list(_q_a) if _q_a is not None else list(Q_HOME_FRONT)
            _q_opt, _, _ = opt_ik_front(_foot_dh0, _q_init0, q_ref=list(Q_HOME_FRONT), dh=_dh_leg)
            prev_q_per_leg.append(_q_opt if _q_opt is not None else _q_init0)
        else:
            _foot_dh0 = _sim_to_dh(_foot_loc0 + _HIND_J4_TO_J5_SIM, front_leg=False)
            _q_h = analytical_ik_hind(_foot_dh0[0], _foot_dh0[1], _foot_dh0[2],
                                      PHI_HIND, dh=_dh_leg, theta5_target=THETA5_HIND)
            _q_init0 = list(_q_h) + [Q_HOME_HIND[4]] if _q_h is not None else list(Q_HOME_HIND)
            _q_opt, _, _ = opt_ik_hind(_foot_dh0, _q_init0, q_ref=list(Q_HOME_HIND), dh=_dh_leg)
            prev_q_per_leg.append(_q_opt if _q_opt is not None else _q_init0)
    OPT_IK_USE_VEL_LIMIT = _saved_vel_limit

    calc_start = time.perf_counter()
    for fi in range(N_FRAMES):
        frame_start = time.perf_counter()
        t = fi * DT
        for leg in range(4):
            is_sw = sched.is_swing(leg, t)
            phase_hist[fi, leg] = sched.phase(leg, t)
            swing_flag[fi, leg] = is_sw

            phase = sched.phase(leg, t)
            p_end = home_foot_per_leg[leg] + np.array([STEP_LENGTH * traj_scale, 0, 0])
            _bv   = body_vel * traj_scale

            if is_sw and not prev_swing[leg]:
                # stance→swing 전환: 해석적 끝점 계산 (이산화 오차 제거)
                # stance st_t=1.0 정확한 위치: XY = p_contact - bv*stance_dur, Z = p_contact[2]
                foot_sw_start[leg] = np.array([
                    foot_contact[leg][0] - _bv[0] * stance_dur,
                    foot_contact[leg][1] - _bv[1] * stance_dur,
                    foot_contact[leg][2],
                ])
            if not is_sw and prev_swing[leg]:
                # swing→stance 전환: swing2 해석적 끝점 = p_end (이산화 오차 제거)
                foot_contact[leg] = p_end.copy()
            foot_loc = foot_pos_at_phase(
                phase,
                foot_sw_start[leg],
                foot_contact[leg],
                p_end,
                body_vel * traj_scale,
                swing_ratio=sched.swing_ratio,
                step_height=STEP_HEIGHT * height_scale,
                stance_dur=stance_dur
            )

            foot_local_prev[leg] = foot_loc.copy()
            prev_swing[leg]      = is_sw
            foot_hist[fi, leg]   = LEG_HIP_OFFSETS[leg] + foot_loc

            if leg < 2:
                foot_ik_sim = foot_loc + _FRONT_J4_TO_J5_SIM
                foot_dh = _sim_to_dh(foot_ik_sim, front_leg=True)
                col = leg  # 0=FR, 1=FL

                # Phase 4: swing/stance 모두 optimization IK (warm-start = 이전 프레임 q)
                # 참조 자세 추종:
                #   swing(blend ON): home → Q_SWING_FRONT → home (quintic 블렌드)
                #   stance/swing(blend OFF): 현재 foot 위치 기반 analytical IK
                #     ← q_ref와 FK constraint 정합 → vel_limit hit 방지
                # q_ref 정책:
                #   swing(blend ON): home + α × (Q_SWING - Q_HOME) (quintic blend, 발 들기)
                #   stance / swing(blend OFF): home (Q_HOME_FRONT)
                # 시작 시 stance 다리는 fi=0~12에서 vel_limit hit 발생 (1회 transient, 무시 가능)
                # 매 swing 사이클의 부드러움 우선
                if is_sw and USE_SWING_QREF_BLEND:
                    _sw_t = sched.swing_t(leg, t)
                    if _sw_t <= 0.5:
                        _alpha = _quintic_s(_sw_t / 0.5)             # 0→1
                    else:
                        _alpha = 1.0 - _quintic_s((_sw_t - 0.5) / 0.5)  # 1→0
                    _q_ref = [h + _alpha * (sw - h)
                              for h, sw in zip(Q_HOME_FRONT, Q_SWING_FRONT)]
                else:
                    _q_ref = list(Q_HOME_FRONT)
                q_opt, nit, pos_err_sq = opt_ik_front(foot_dh, prev_q_per_leg[leg][:5],
                                                      q_ref=_q_ref, dh=LEG_DH[leg])   # v13
                opt_ik_nit_hist[fi, col]     = nit
                opt_ik_pos_err_hist[fi, col] = pos_err_sq
                if q_opt is not None:
                    q = q_opt
                else:
                    # bounds 내 해 없음 → analytical fallback
                    opt_ik_fallback_hist[fi, col] = True
                    q_ana = analytical_ik_front(foot_dh[0], foot_dh[1], foot_dh[2],
                                                PHI_FRONT, THETA5_FRONT, dh=LEG_DH[leg])  # v13
                    q = q_ana if q_ana is not None else list(Q_HOME_FRONT)

                pq = prev_q_per_leg[leg]
                for j in range(len(q)):
                    best = q[j]
                    for off in (-2.0*math.pi, 2.0*math.pi):
                        cand = q[j] + off
                        if abs(cand - pq[j]) < abs(best - pq[j]):
                            best = cand
                    q[j] = best
            else:
                foot_ik_sim = foot_loc + _HIND_J4_TO_J5_SIM
                foot_dh = _sim_to_dh(foot_ik_sim, front_leg=False)
                col = leg  # 2=HR, 3=HL

                # 뒷다리: USE_SWING_QREF_BLEND 무시하고 항상 home 추종
                _q_ref_h = list(Q_HOME_HIND)

                q_opt, nit, pos_err_sq = opt_ik_hind(foot_dh, prev_q_per_leg[leg][:5],
                                                     q_ref=_q_ref_h, dh=LEG_DH[leg])   # v13
                opt_ik_nit_hist[fi, col]     = nit
                opt_ik_pos_err_hist[fi, col] = pos_err_sq
                if q_opt is not None:
                    q = q_opt
                else:
                    opt_ik_fallback_hist[fi, col] = True
                    q_h = analytical_ik_hind(foot_dh[0], foot_dh[1], foot_dh[2],
                                             PHI_HIND, dh=LEG_DH[leg], theta5_target=THETA5_HIND)  # v13
                    if q_h is None:
                        q = list(Q_HOME_HIND)
                    else:
                        q = list(q_h) + [Q_HOME_HIND[4]]
                pq = prev_q_per_leg[leg]
                for j in range(len(q)):
                    best = q[j]
                    for off in (-2.0*math.pi, 2.0*math.pi):
                        cand = q[j] + off
                        if abs(cand - pq[j]) < abs(best - pq[j]):
                            best = cand
                    q[j] = best

            nj = N_JOINTS_PER_LEG[leg]
            # 각속도 클리핑: |q - q_prev| / DT ≤ JOINT_VEL_LIMIT (모든 다리 공통)
            _vel_dt  = JOINT_VEL_LIMIT_RAD_S[:nj] * DT
            _q_arr   = np.array(q[:nj])
            _q_prev  = np.array(prev_q_per_leg[leg][:nj])
            _q_arr   = _q_prev + np.clip(_q_arr - _q_prev, -_vel_dt, _vel_dt)
            q[:nj]   = list(_q_arr)
            joint_hist[fi, leg, :nj] = q[:nj]
            prev_q_per_leg[leg][:nj] = q[:nj]
        frame_calc_time[fi] = time.perf_counter() - frame_start

    calc_total = time.perf_counter() - calc_start

    joint_hist_unwrapped = np.unwrap(joint_hist, axis=0)
    # v13.1: USE_SPLINE_DIFF 토글로 CubicSpline (smoothness 패치) vs np.gradient (원래) 선택.
    # 정량 측정 결과 spline 이 τ peak / jerk peak 폭주 유발 → default False.
    if USE_SPLINE_DIFF:
        _t_grid_jh = np.arange(joint_hist_unwrapped.shape[0]) * DT
        _jh_spline = _CubicSpline(_t_grid_jh, joint_hist_unwrapped, axis=0, bc_type='natural')
        joint_vel_hist = _jh_spline(_t_grid_jh, 1)
    else:
        # np.gradient: boundary는 forward/backward 차분 → 첫 frame spike 제거
        joint_vel_hist = np.gradient(joint_hist_unwrapped, DT, axis=0)

    peak_per_joint  = np.max(np.abs(joint_vel_hist), axis=(0, 1))
    ratio_per_joint = peak_per_joint / JOINT_VEL_LIMIT_RAD_S
    worst_ratio     = float(np.max(ratio_per_joint))

    if worst_ratio <= VEL_LIMIT_MARGIN:
        break
    scale_decay  = max(0.60, min(0.98 / worst_ratio, 0.98))
    traj_scale  *= scale_decay
    height_scale *= scale_decay

print(f"궤적 완료. iter={opt_iter_used}  scale={traj_scale:.4f}")
_fb_per_leg = [int(np.sum(opt_ik_fallback_hist[:, c])) for c in range(4)]
_sw_per_leg = [int(np.sum(swing_flag[:, c])) for c in range(4)]
_q_home_f = np.array(Q_HOME_FRONT)
_q_home_h = np.array(Q_HOME_HIND)
_front_lim_lo = np.array([b[0] for b in FRONT_Q_LIM])
_front_lim_hi = np.array([b[1] for b in FRONT_Q_LIM])
_hind_lim_lo  = np.array([b[0] for b in HIND_Q_LIM])
_hind_lim_hi  = np.array([b[1] for b in HIND_Q_LIM])

# Opt-IK 진단 항목 설명
print("  [INFO] nit_avg     : SLSQP 수렴 반복 횟수 평균 (낮을수록 빠른 수렴, maxiter 도달 시 fallback)")
print("  [INFO] fallback    : N/M = swing M프레임 중 N프레임이 SLSQP 실패 → analytical IK 강등")
print("  [INFO] pos_err     : FK(q_opt)와 목표 발 위치의 잔차 (mm, 0이면 등식 제약 정확히 만족)")
print("  [INFO] bound_viol  : opt IK 성공 프레임 중 관절 한계(FRONT/HIND_Q_LIM) 벗어난 프레임 수")
print("  [INFO] home_dev_avg: opt IK 성공 프레임 평균 ‖q − q_home‖ [rad] (자세가 home에서 멀수록 큼)")

for _col, _leg, _sw_mask, _fb, _q_home, _lim_lo, _lim_hi, _leg_name, _lim_name in [
    (0, 0, swing_flag[:, 0], _fb_per_leg[0], _q_home_f, _front_lim_lo, _front_lim_hi, 'FR', 'FRONT_Q_LIM'),
    (1, 1, swing_flag[:, 1], _fb_per_leg[1], _q_home_f, _front_lim_lo, _front_lim_hi, 'FL', 'FRONT_Q_LIM'),
    (2, 2, swing_flag[:, 2], _fb_per_leg[2], _q_home_h, _hind_lim_lo,  _hind_lim_hi,  'HR', 'HIND_Q_LIM'),
    (3, 3, swing_flag[:, 3], _fb_per_leg[3], _q_home_h, _hind_lim_lo,  _hind_lim_hi,  'HL', 'HIND_Q_LIM'),
]:
    _sw_n   = int(np.sum(_sw_mask))
    _nit    = opt_ik_nit_hist[_sw_mask, _col]
    _perr   = opt_ik_pos_err_hist[_sw_mask, _col]           # [m²], nan if fallback
    _ok     = ~np.isnan(_perr)
    _perr_mm_max  = float(np.sqrt(np.nanmax(_perr))) * 1e3 if _ok.any() else 0.0
    _perr_mm_mean = float(np.sqrt(np.nanmean(_perr))) * 1e3 if _ok.any() else 0.0

    # 관절 한계 위반: opt IK 성공 프레임 중 bounds 벗어난 것
    _q_sw   = joint_hist[_sw_mask, _leg, :5]               # (sw_n, 5)
    _ok_idx = np.where(_ok)[0]
    _q_ok   = _q_sw[_ok_idx]
    _viol   = int(np.sum(np.any((_q_ok < _lim_lo) | (_q_ok > _lim_hi), axis=1)))

    # home 이탈: opt IK 성공 프레임 평균 ||q - q_home||
    _home_dev = float(np.mean(np.linalg.norm(_q_ok - _q_home, axis=1))) if len(_q_ok) else 0.0

    print(f"  Opt-IK {_leg_name}: nit_avg={_nit.mean():.1f}  fallback={_fb}/{_sw_n}  "
          f"pos_err(max={_perr_mm_max:.3f}mm mean={_perr_mm_mean:.3f}mm)  "
          f"bound_viol={_viol}프레임  home_dev_avg={_home_dev:.4f}rad")

    # ── 경고 ──────────────────────────────────────────────────
    if _fb > 0 and _sw_n > 0:
        _fb_rate = _fb / _sw_n * 100
        print(f"  [WARNING] {_leg_name} Opt-IK fallback {_fb_rate:.1f}% — "
              f"V({V}m/s)·STEP_HEIGHT({STEP_HEIGHT}m) 조합이 {_lim_name} 초과 가능성. "
              f"속도↓ or STEP_HEIGHT↓ or {_lim_name} 완화 권장.")
    if _perr_mm_max > 0.1:
        print(f"  [WARNING] {_leg_name} 최대 위치 오차 {_perr_mm_max:.3f}mm > 0.1mm — "
              f"SLSQP 수렴 불충분. OPT_IK_MAXITER({OPT_IK_MAXITER}) 증가 권장.")
    if _viol > 0:
        print(f"  [WARNING] {_leg_name} 관절 한계 위반 {_viol}프레임 — "
              f"SLSQP 수치 오차로 bounds 미세 초과. {_lim_name} 여유 ±0.01rad 추가 권장.")
    if _home_dev > 1.0:
        print(f"  [INFO]    {_leg_name} home 이탈 평균 {_home_dev:.4f}rad > 1.0rad — "
              f"swing 궤적이 home 자세와 크게 다름. T({T}s)↑ or V({V}m/s)↓ 시 완화됨.")

# v13.1: USE_SPLINE_DIFF 토글로 미분 방식 선택 (CubicSpline vs np.gradient).
_t_grid_final = np.arange(N_FRAMES) * DT
_jh_unwrap_final = np.unwrap(joint_hist, axis=0)
if USE_SPLINE_DIFF:
    _jh_spline_final = _CubicSpline(_t_grid_final, _jh_unwrap_final, axis=0, bc_type='natural')
    joint_vel_hist  = _jh_spline_final(_t_grid_final, 1)
    joint_acc_hist  = _jh_spline_final(_t_grid_final, 2)
    _joint_jrk_hist = _jh_spline_final(_t_grid_final, 3)
else:
    joint_vel_hist = np.gradient(_jh_unwrap_final, DT, axis=0)
    joint_acc_hist = np.gradient(joint_vel_hist, DT, axis=0)
    _joint_jrk_hist = np.gradient(joint_acc_hist, DT, axis=0)

joint_vel_FR = joint_vel_hist[:, 0, :]
joint_acc_FR = joint_acc_hist[:, 0, :]
joint_jrk_FR = _joint_jrk_hist[:, 0, :]

joint_vel_HR = joint_vel_hist[:, 2, :]
joint_acc_HR = joint_acc_hist[:, 2, :]
joint_jrk_HR = _joint_jrk_hist[:, 2, :]

joint_vel_HL = joint_vel_hist[:, 3, :]
joint_acc_HL = joint_acc_hist[:, 3, :]
joint_jrk_HL = _joint_jrk_hist[:, 3, :]

foot_local = foot_hist - LEG_HIP_OFFSETS[np.newaxis, :, :]
foot_vel_t = np.gradient(foot_local, DT, axis=0)
foot_acc_t = np.gradient(foot_vel_t,  DT, axis=0)

# ══════════════════════════════════════════════════════════════
# 3.5  WBC + MPC QP GRF
# ══════════════════════════════════════════════════════════════
print(f"WBC + {'MPC QP' if USE_MPC else 'QP GRF'} 계산 중...")
wbc_t0 = time.perf_counter()

# 1차 지연 추종 오차
theta_a_hist  = np.zeros_like(joint_hist)
dtheta_a_hist = np.zeros_like(joint_hist)
for leg in range(4):
    nj = N_JOINTS_PER_LEG[leg]
    theta_a_hist[0, leg, :nj] = joint_hist[0, leg, :nj] + INIT_ERR_RAD
    for fi in range(1, N_FRAMES):
        prev   = theta_a_hist[fi-1, leg, :nj]
        target = joint_hist[fi-1, leg, :nj]
        theta_a_hist[fi, leg, :nj] = prev + (DT / TAU_LAG) * (target - prev)
    # v13.1: USE_SPLINE_DIFF 토글
    if USE_SPLINE_DIFF:
        _cs_ta = _CubicSpline(_t_grid_final, theta_a_hist[:, leg, :nj],
                              axis=0, bc_type='natural')
        dtheta_a_hist[:, leg, :nj] = _cs_ta(_t_grid_final, 1)
    else:
        dtheta_a_hist[:, leg, :nj] = np.gradient(theta_a_hist[:, leg, :nj], DT, axis=0)

wbc_tau_ff   = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
wbc_tau_dyn  = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))  # M·q̈+C·q̇+g(q) (RNEA, 중력 포함)
wbc_tau_pd   = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
wbc_tau_imp  = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
wbc_tau_cmd  = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
wbc_tau_grf  = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))  # -Jᵀ·λ_des (저장만, plot 안 함)
wbc_lam_des  = np.zeros((N_FRAMES, 4, 3))   # [Fx, Fy, Fz]
wbc_lam_calc = np.zeros((N_FRAMES, 4, 3))

# Phase 5: WBIC 진단 배열
wbic_dtau_hist     = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
wbic_dlam_hist     = np.zeros((N_FRAMES, 4, 3))
wbic_residual_hist = np.zeros((N_FRAMES, 4))    # eq 잔차 norm (사실상 0이어야 함)
wbic_status_hist   = np.zeros((N_FRAMES, 4), dtype=bool)  # solver 성공 여부
wbic_lam_used      = np.zeros((N_FRAMES, 4, 3))   # 실제 사용된 λ = λ_des + Δλ

# v12 smoothness 패치: WBIC τ 의 frame-간 차분 페널티용 이전 frame τ 상태.
# tau_ff_corrected[leg] = leg_data[leg]['tau_ff'] after WBIC d_tau 적용. None = first frame.
tau_ff_corrected_prev = [None] * 4

# ── v11: Floating-base 동역학 진단 배열 ────────────────────
body_pos_hist   = np.zeros((N_FRAMES, 3))           # CoM world position
body_R_hist     = np.zeros((N_FRAMES, 3, 3))        # rotation matrix
body_v_hist     = np.zeros((N_FRAMES, 3))           # CoM linear velocity
body_omega_hist = np.zeros((N_FRAMES, 3))           # body angular velocity
body_alin_hist  = np.zeros((N_FRAMES, 3))           # CoM linear accel
body_aang_hist  = np.zeros((N_FRAMES, 3))           # body angular accel
# Reference (kinematic V·t) for deviation 계산
body_pos_ref_hist = np.zeros((N_FRAMES, 3))
body_v_ref_hist   = np.zeros((N_FRAMES, 3))

# 초기 body 상태 (steady-state stance posture)
# foot 가 ground (z=0) 에 닿도록 body z 조정 (NMPC convention 과 일치)
_foot_z_home = float(foot_hist[0, 0, 2])    # FR foot z @ home, body-local frame (≈ -0.465m)
body_state = {
    'pos':   np.array([0.0, 0.0, -_foot_z_home]),  # body z = +0.465 → foot world z = 0
    'R':     np.eye(3),
    'v':     np.array([V, 0.0, 0.0]),              # 정상 보행 속도로 시작
    'omega': np.zeros(3),
    'a_lin': np.zeros(3),
    'a_ang': np.zeros(3),
    '_diverged': False,                            # 발산 시 integrate_body_state가 set
}

# WBIC FB 진단
wbic_fb_residual_hist = np.zeros(N_FRAMES)          # body 6-DoF residual norm
wbic_fb_status_hist   = np.zeros(N_FRAMES, dtype=bool)
wbic_fb_dvfb_hist     = np.zeros((N_FRAMES, 6))     # Δv̇_fb

mpc_fail_count = 0
wbic_fail_count = 0
wbic_fb_fail_count = 0


# ══════════════════════════════════════════════════════════════
# v12: NMPC (crocoddyl) 통합
# ══════════════════════════════════════════════════════════════
def _solve_nmpc_trot():
    """crocoddyl FDDP로 trot trajectory 풀이.
    Returns: xs (state, N+1×nx), us (controls, N×nu), success (bool).
    """
    import math as _m
    import pinocchio as pin
    if not _CROCODDYL_AVAILABLE:
        print("  ⚠ crocoddyl 미설치 — USE_NMPC 비활성")
        return None, None, None, False, None, None

    # pinocchio Model 빌드 (build_pin_model의 quadruped)
    pin_model = _bm.build_model()
    pin_data  = pin_model.createData()
    cstate     = _crocoddyl.StateMultibody(pin_model)
    cactuation = _crocoddyl.ActuationModelFloatingBase(cstate)

    # 초기 상태: home pose, base z 조정으로 발이 ground (z=0) 닿음
    Q_HOME_PER_LEG_PIN = {'FR': Q_HOME_FRONT, 'FL': Q_HOME_FRONT,
                           'HR': Q_HOME_HIND,  'HL': Q_HOME_HIND}
    q0 = pin.neutral(pin_model)
    for leg, qh in Q_HOME_PER_LEG_PIN.items():
        for i, qi in enumerate(qh):
            jid = pin_model.getJointId(f'leg_{leg}_j{i+1}')
            q0[pin_model.idx_qs[jid]] = qi
    pin.forwardKinematics(pin_model, pin_data, q0)
    pin.updateFramePlacements(pin_model, pin_data)
    foot_z_native = pin_data.oMf[pin_model.getFrameId('leg_FR_foot')].translation[2]
    q0[2] = -foot_z_native
    v0 = np.zeros(pin_model.nv)
    x0 = np.concatenate([q0, v0])

    foot_frames_pin = {leg: pin_model.getFrameId(f'leg_{leg}_foot')
                        for leg in ['FR','FL','HR','HL']}
    pin.forwardKinematics(pin_model, pin_data, q0)
    pin.updateFramePlacements(pin_model, pin_data)
    foot_home_pin = {leg: pin_data.oMf[foot_frames_pin[leg]].translation.copy()
                     for leg in ['FR','FL','HR','HL']}

    # Joint torque limit array (cactuation.nu 차원, idx_vs 매핑 반영)
    tau_lim_full = np.zeros(cactuation.nu)
    for leg in ['FR','FL','HR','HL']:
        for i in range(5):
            u_idx = pin_model.idx_vs[pin_model.getJointId(f'leg_{leg}_j{i+1}')] - 6
            tau_lim_full[u_idx] = JOINT_TORQUE_LIMIT[i]
    tau_lim_lb = -tau_lim_full
    tau_lim_ub = +tau_lim_full

    # Swing target: smoothstep horizontal + bell-curve vertical (v=0 at touchdown)
    # 기존 cycloid (h*4*t*(1-t)) 는 sw_t=1 에서 dz/dt=-h*4 = -1.28 m/s 로 충격 유발 →
    # x/y 는 smoothstep s(t)=t²(3-2t) (v=0 at endpoints), z 는 16t²(1-t)² (v=0 at endpoints)
    def _cycloid(p_start, p_end, sw_t, h):
        s_xy = sw_t * sw_t * (3.0 - 2.0 * sw_t)
        p = p_start + s_xy * (p_end - p_start)
        p[2] = p_start[2] + h * 16.0 * sw_t * sw_t * (1.0 - sw_t) * (1.0 - sw_t)
        return p

    # v12.6: 모든 gait 일반 지원 — sched 시간 기반 dispatch
    N_PER_CYCLE = max(2, round(T / DT_MPC))
    DT_NMPC = T / N_PER_CYCLE
    N_TOTAL = N_PER_CYCLE * N_CYCLES

    _LEGS_OS = ['FR', 'FL', 'HR', 'HL']
    actions = []
    for k in range(N_TOTAL):
        t = k * DT_NMPC
        stance, swing = [], []
        swing_info = {}
        for leg_idx, leg in enumerate(_LEGS_OS):
            ph = sched.phase(leg_idx, t)
            if ph < sched.swing_ratio:
                sw_t_i = ph / sched.swing_ratio
                t_sw_start = t - ph * T
                t_sw_end   = t_sw_start + T_SW
                swing.append(leg)
                swing_info[leg] = (sw_t_i, t_sw_start, t_sw_end)
            else:
                stance.append(leg)

        # Build action model
        cm_contact = _crocoddyl.ContactModelMultiple(cstate, cactuation.nu)
        for leg in stance:
            c = _crocoddyl.ContactModel3D(
                cstate, foot_frames_pin[leg], np.zeros(3),
                pin.LOCAL_WORLD_ALIGNED, cactuation.nu,
                np.array([NMPC_BAUMGARTE_KP, NMPC_BAUMGARTE_KD]))
            cm_contact.addContact(f'c_{leg}', c)
        cost = _crocoddyl.CostModelSum(cstate, cactuation.nu)
        cost.addCost('stateReg',
            _crocoddyl.CostModelResidual(cstate,
                _crocoddyl.ResidualModelState(cstate, x0, cactuation.nu)),
            NMPC_W_STATE_REG)
        cost.addCost('ctrlReg',
            _crocoddyl.CostModelResidual(cstate,
                _crocoddyl.ResidualModelControl(cstate, cactuation.nu)),
            NMPC_W_CTRL_REG)
        # Friction cone soft barrier — stance leg only (contact 추가 후)
        for leg in stance:
            fc = _crocoddyl.FrictionCone(np.eye(3), MU_FRICTION, NMPC_FRIC_NF,
                                         True, 0.0, NMPC_FRIC_FZ_MAX)
            fc_act = _crocoddyl.ActivationModelQuadraticBarrier(
                _crocoddyl.ActivationBounds(fc.lb, fc.ub))
            fc_res = _crocoddyl.ResidualModelContactFrictionCone(
                cstate, foot_frames_pin[leg], fc, cactuation.nu, True)
            cost.addCost(f'fric_{leg}',
                _crocoddyl.CostModelResidual(cstate, fc_act, fc_res),
                NMPC_W_FRICTION)
        # Contact force regularization (||F||² → 0, xy 더 강하게)
        for leg in stance:
            fref = pin.Force(np.zeros(6))
            f_act = _crocoddyl.ActivationModelWeightedQuad(
                np.array([NMPC_W_FORCE_XY, NMPC_W_FORCE_XY, NMPC_W_FORCE_Z]))
            f_res = _crocoddyl.ResidualModelContactForce(
                cstate, foot_frames_pin[leg], fref, 3, cactuation.nu, True)
            cost.addCost(f'freg_{leg}',
                _crocoddyl.CostModelResidual(cstate, f_act, f_res),
                NMPC_W_FORCE_REG)
        # Joint torque limit barrier
        tau_act = _crocoddyl.ActivationModelQuadraticBarrier(
            _crocoddyl.ActivationBounds(tau_lim_lb, tau_lim_ub))
        tau_res = _crocoddyl.ResidualModelControl(cstate, cactuation.nu)
        cost.addCost('tau_lim',
            _crocoddyl.CostModelResidual(cstate, tau_act, tau_res),
            NMPC_W_TAU_LIM)
        for leg in swing:
            sw_t, t_sw_start, t_sw_end = swing_info[leg]
            ps = foot_home_pin[leg].copy()
            ps[0] += V * t_sw_start - STEP_LENGTH/2
            pe = foot_home_pin[leg].copy()
            pe[0] += V * t_sw_end + STEP_LENGTH/2
            tgt = _cycloid(ps, pe, sw_t, STEP_HEIGHT)
            res = _crocoddyl.ResidualModelFrameTranslation(
                cstate, foot_frames_pin[leg], tgt, cactuation.nu)
            act = _crocoddyl.ActivationModelWeightedQuad(
                np.array([NMPC_W_TRACK_XY, NMPC_W_TRACK_XY, NMPC_W_TRACK_Z]))
            cost.addCost(f'foot_{leg}',
                _crocoddyl.CostModelResidual(cstate, act, res), 1.0)
            # Pre-touchdown velocity penalty: swing 마지막 N_step × DT 시간 동안 v → 0
            if sw_t >= max(0.0, 1.0 - NMPC_TOUCHDOWN_LAST_N * DT_NMPC / max(T_SW, 1e-6)):
                v_ref = pin.Motion(np.zeros(6))
                v_res = _crocoddyl.ResidualModelFrameVelocity(
                    cstate, foot_frames_pin[leg], v_ref,
                    pin.LOCAL_WORLD_ALIGNED, cactuation.nu)
                v_act = _crocoddyl.ActivationModelWeightedQuad(
                    np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))   # linear only
                cost.addCost(f'tdv_{leg}',
                    _crocoddyl.CostModelResidual(cstate, v_act, v_res),
                    NMPC_W_TOUCHDOWN_V)
        # Stance leg foot world-frame velocity → 0 cost (slip 방지)
        # 위치 target 은 NMPC가 전역으로 모르므로 (cycle 0 swing 불완전 등),
        # velocity 직접 0 으로 강제 — 본질적으로 contact constraint 보강
        for leg in stance:
            v_ref = pin.Motion(np.zeros(6))
            v_res = _crocoddyl.ResidualModelFrameVelocity(
                cstate, foot_frames_pin[leg], v_ref,
                pin.LOCAL_WORLD_ALIGNED, cactuation.nu)
            v_act = _crocoddyl.ActivationModelWeightedQuad(
                np.array([NMPC_W_STANCE_POS_XY, NMPC_W_STANCE_POS_XY,
                          NMPC_W_STANCE_POS_Z, 0.0, 0.0, 0.0]))   # linear only
            cost.addCost(f'svel_{leg}',
                _crocoddyl.CostModelResidual(cstate, v_act, v_res), 1.0)
        diff = _crocoddyl.DifferentialActionModelContactFwdDynamics(
            cstate, cactuation, cm_contact, cost, 0.0, True)
        actions.append(_crocoddyl.IntegratedActionModelEuler(diff, DT_NMPC))

    # Terminal model
    cm_T = _crocoddyl.ContactModelMultiple(cstate, cactuation.nu)
    for leg in ['FR','FL','HR','HL']:
        c = _crocoddyl.ContactModel3D(
            cstate, foot_frames_pin[leg], np.zeros(3),
            pin.LOCAL_WORLD_ALIGNED, cactuation.nu,
            np.array([NMPC_BAUMGARTE_KP, NMPC_BAUMGARTE_KD]))
        cm_T.addContact(f'c_{leg}', c)
    x_ref_T = x0.copy(); x_ref_T[0] = q0[0] + V * T * N_CYCLES
    cost_T = _crocoddyl.CostModelSum(cstate, cactuation.nu)
    cost_T.addCost('stateReg_T',
        _crocoddyl.CostModelResidual(cstate,
            _crocoddyl.ResidualModelState(cstate, x_ref_T, cactuation.nu)),
        NMPC_W_TERMINAL)
    diff_T = _crocoddyl.DifferentialActionModelContactFwdDynamics(
        cstate, cactuation, cm_T, cost_T, 0.0, True)
    terminal = _crocoddyl.IntegratedActionModelEuler(diff_T, 0.0)

    problem = _crocoddyl.ShootingProblem(x0, actions, terminal)
    solver = _crocoddyl.SolverFDDP(problem)
    xs_init = [x0] * (N_TOTAL + 1)
    us_init = [np.zeros(cactuation.nu)] * N_TOTAL
    print(f"  NMPC FDDP 풀이 중 ({N_TOTAL} steps × DT={DT_NMPC*1e3:.1f}ms = {N_TOTAL*DT_NMPC:.2f}s)...")
    t_solve = time.time()
    done = solver.solve(xs_init, us_init, maxiter=NMPC_MAXITER,
                        is_feasible=False, init_reg=NMPC_INIT_REG)
    elapsed = time.time() - t_solve
    print(f"  NMPC: done={done}, iters={solver.iter}, time={elapsed*1e3:.0f}ms, cost={solver.cost:.2e}")
    # Contact forces 추출 (각 step의 GRF, world frame)
    forces = []
    for k in range(N_TOTAL):
        f_step = {leg: np.zeros(3) for leg in ['FR','FL','HR','HL']}
        try:
            ad_k = solver.problem.runningDatas[k]
            contacts = ad_k.differential.multibody.contacts.contacts.todict()
            for name, cdata in contacts.items():
                leg = name.replace('c_', '')
                if leg in f_step:
                    f_step[leg] = np.array(cdata.f.linear).copy()
        except Exception:
            pass
        forces.append(f_step)
    return np.array(solver.xs), np.array(solver.us), forces, done, pin_model, pin_data


def _solve_nmpc_trot_receding():
    """Receding horizon NMPC.
    매 NMPC_RH_N_RESOLVE step마다 NMPC_RH_N_HORIZON 길이의 NMPC 풀이 (warm-start).
    one-shot 4+ cycle 발산 회피 — 짧은 horizon × 다회 풀이.
    """
    import math as _m
    import pinocchio as pin
    if not _CROCODDYL_AVAILABLE:
        return None, None, None, False, None, None

    pin_model = _bm.build_model()
    pin_data  = pin_model.createData()
    cstate     = _crocoddyl.StateMultibody(pin_model)
    cactuation = _crocoddyl.ActuationModelFloatingBase(cstate)

    Q_HOME_PER_LEG_PIN = {'FR': Q_HOME_FRONT, 'FL': Q_HOME_FRONT,
                           'HR': Q_HOME_HIND,  'HL': Q_HOME_HIND}
    q0 = pin.neutral(pin_model)
    for leg, qh in Q_HOME_PER_LEG_PIN.items():
        for i, qi in enumerate(qh):
            jid = pin_model.getJointId(f'leg_{leg}_j{i+1}')
            q0[pin_model.idx_qs[jid]] = qi
    pin.forwardKinematics(pin_model, pin_data, q0)
    pin.updateFramePlacements(pin_model, pin_data)
    foot_z_native = pin_data.oMf[pin_model.getFrameId('leg_FR_foot')].translation[2]
    q0[2] = -foot_z_native
    v0 = np.zeros(pin_model.nv)
    x0 = np.concatenate([q0, v0])

    foot_frames_pin = {leg: pin_model.getFrameId(f'leg_{leg}_foot')
                        for leg in ['FR','FL','HR','HL']}
    pin.forwardKinematics(pin_model, pin_data, q0)
    pin.updateFramePlacements(pin_model, pin_data)
    foot_home_pin = {leg: pin_data.oMf[foot_frames_pin[leg]].translation.copy()
                     for leg in ['FR','FL','HR','HL']}

    # Joint torque limit array (cactuation.nu 차원, idx_vs 매핑 반영)
    tau_lim_full = np.zeros(cactuation.nu)
    for leg in ['FR','FL','HR','HL']:
        for i in range(5):
            u_idx = pin_model.idx_vs[pin_model.getJointId(f'leg_{leg}_j{i+1}')] - 6
            tau_lim_full[u_idx] = JOINT_TORQUE_LIMIT[i]
    tau_lim_lb = -tau_lim_full
    tau_lim_ub = +tau_lim_full

    # Swing target: smoothstep horizontal + bell-curve vertical (v=0 at touchdown)
    def _cycloid(p_start, p_end, sw_t, h):
        s_xy = sw_t * sw_t * (3.0 - 2.0 * sw_t)
        p = p_start + s_xy * (p_end - p_start)
        p[2] = p_start[2] + h * 16.0 * sw_t * sw_t * (1.0 - sw_t) * (1.0 - sw_t)
        return p

    # v12.6: 모든 gait 일반 지원 (walk, amble, pace, trot, canter, gallop)
    # 각 leg phase 는 GaitScheduler 가 결정 — 시간 기반 dispatching
    N_PER_CYCLE = max(2, round(T / DT_MPC))
    DT_NMPC = T / N_PER_CYCLE
    N_TOTAL = N_PER_CYCLE * N_CYCLES

    _LEGS = ['FR', 'FL', 'HR', 'HL']

    def _gait_legs_at(t):
        """At time t: return (stance_list, swing_list, swing_info_dict).
        swing_info[leg] = (sw_t, t_sw_start, t_sw_end). 각 leg 독립 swing window.
        """
        stance, swing = [], []
        swing_info = {}
        for leg_idx, leg in enumerate(_LEGS):
            ph = sched.phase(leg_idx, t)
            if ph < sched.swing_ratio:
                sw_t = ph / sched.swing_ratio
                t_sw_start = t - ph * T
                t_sw_end   = t_sw_start + T_SW
                swing.append(leg)
                swing_info[leg] = (sw_t, t_sw_start, t_sw_end)
            else:
                stance.append(leg)
        return stance, swing, swing_info

    def _build_action_at_step(global_k):
        """전역 step index → action model. 시간 기반 gait dispatch (any gait 지원)."""
        t = global_k * DT_NMPC
        stance, swing, swing_info = _gait_legs_at(t)

        cm_contact = _crocoddyl.ContactModelMultiple(cstate, cactuation.nu)
        for leg in stance:
            c = _crocoddyl.ContactModel3D(
                cstate, foot_frames_pin[leg], np.zeros(3),
                pin.LOCAL_WORLD_ALIGNED, cactuation.nu,
                np.array([NMPC_BAUMGARTE_KP, NMPC_BAUMGARTE_KD]))
            cm_contact.addContact(f'c_{leg}', c)
        cost = _crocoddyl.CostModelSum(cstate, cactuation.nu)
        cost.addCost('stateReg',
            _crocoddyl.CostModelResidual(cstate,
                _crocoddyl.ResidualModelState(cstate, x0, cactuation.nu)),
            NMPC_W_STATE_REG)
        cost.addCost('ctrlReg',
            _crocoddyl.CostModelResidual(cstate,
                _crocoddyl.ResidualModelControl(cstate, cactuation.nu)),
            NMPC_W_CTRL_REG)
        # Friction cone soft barrier — stance leg only (contact 추가 후)
        for leg in stance:
            fc = _crocoddyl.FrictionCone(np.eye(3), MU_FRICTION, NMPC_FRIC_NF,
                                         True, 0.0, NMPC_FRIC_FZ_MAX)
            fc_act = _crocoddyl.ActivationModelQuadraticBarrier(
                _crocoddyl.ActivationBounds(fc.lb, fc.ub))
            fc_res = _crocoddyl.ResidualModelContactFrictionCone(
                cstate, foot_frames_pin[leg], fc, cactuation.nu, True)
            cost.addCost(f'fric_{leg}',
                _crocoddyl.CostModelResidual(cstate, fc_act, fc_res),
                NMPC_W_FRICTION)
        # Contact force regularization (||F||² → 0, xy 더 강하게)
        for leg in stance:
            fref = pin.Force(np.zeros(6))
            f_act = _crocoddyl.ActivationModelWeightedQuad(
                np.array([NMPC_W_FORCE_XY, NMPC_W_FORCE_XY, NMPC_W_FORCE_Z]))
            f_res = _crocoddyl.ResidualModelContactForce(
                cstate, foot_frames_pin[leg], fref, 3, cactuation.nu, True)
            cost.addCost(f'freg_{leg}',
                _crocoddyl.CostModelResidual(cstate, f_act, f_res),
                NMPC_W_FORCE_REG)
        # Joint torque limit barrier
        tau_act = _crocoddyl.ActivationModelQuadraticBarrier(
            _crocoddyl.ActivationBounds(tau_lim_lb, tau_lim_ub))
        tau_res = _crocoddyl.ResidualModelControl(cstate, cactuation.nu)
        cost.addCost('tau_lim',
            _crocoddyl.CostModelResidual(cstate, tau_act, tau_res),
            NMPC_W_TAU_LIM)
        for leg in swing:
            sw_t, t_sw_start, t_sw_end = swing_info[leg]
            ps = foot_home_pin[leg].copy()
            ps[0] += V * t_sw_start - STEP_LENGTH/2
            pe = foot_home_pin[leg].copy()
            pe[0] += V * t_sw_end + STEP_LENGTH/2
            tgt = _cycloid(ps, pe, sw_t, STEP_HEIGHT)
            res = _crocoddyl.ResidualModelFrameTranslation(
                cstate, foot_frames_pin[leg], tgt, cactuation.nu)
            act = _crocoddyl.ActivationModelWeightedQuad(
                np.array([NMPC_W_TRACK_XY, NMPC_W_TRACK_XY, NMPC_W_TRACK_Z]))
            cost.addCost(f'foot_{leg}',
                _crocoddyl.CostModelResidual(cstate, act, res), 1.0)
            # Pre-touchdown velocity penalty: swing 마지막 N_step × DT 시간 동안 v → 0
            if sw_t >= max(0.0, 1.0 - NMPC_TOUCHDOWN_LAST_N * DT_NMPC / max(T_SW, 1e-6)):
                v_ref = pin.Motion(np.zeros(6))
                v_res = _crocoddyl.ResidualModelFrameVelocity(
                    cstate, foot_frames_pin[leg], v_ref,
                    pin.LOCAL_WORLD_ALIGNED, cactuation.nu)
                v_act = _crocoddyl.ActivationModelWeightedQuad(
                    np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))   # linear only
                cost.addCost(f'tdv_{leg}',
                    _crocoddyl.CostModelResidual(cstate, v_act, v_res),
                    NMPC_W_TOUCHDOWN_V)
        # Stance leg foot world-frame velocity → 0 cost (slip 방지)
        # 위치 target 은 NMPC가 전역으로 모르므로 (cycle 0 swing 불완전 등),
        # velocity 직접 0 으로 강제 — 본질적으로 contact constraint 보강
        for leg in stance:
            v_ref = pin.Motion(np.zeros(6))
            v_res = _crocoddyl.ResidualModelFrameVelocity(
                cstate, foot_frames_pin[leg], v_ref,
                pin.LOCAL_WORLD_ALIGNED, cactuation.nu)
            v_act = _crocoddyl.ActivationModelWeightedQuad(
                np.array([NMPC_W_STANCE_POS_XY, NMPC_W_STANCE_POS_XY,
                          NMPC_W_STANCE_POS_Z, 0.0, 0.0, 0.0]))   # linear only
            cost.addCost(f'svel_{leg}',
                _crocoddyl.CostModelResidual(cstate, v_act, v_res), 1.0)
        diff = _crocoddyl.DifferentialActionModelContactFwdDynamics(
            cstate, cactuation, cm_contact, cost, 0.0, True)
        return _crocoddyl.IntegratedActionModelEuler(diff, DT_NMPC)

    def _build_terminal(target_x):
        cm_T = _crocoddyl.ContactModelMultiple(cstate, cactuation.nu)
        for leg in ['FR','FL','HR','HL']:
            c = _crocoddyl.ContactModel3D(
                cstate, foot_frames_pin[leg], np.zeros(3),
                pin.LOCAL_WORLD_ALIGNED, cactuation.nu,
                np.array([NMPC_BAUMGARTE_KP, NMPC_BAUMGARTE_KD]))
            cm_T.addContact(f'c_{leg}', c)
        cost_T = _crocoddyl.CostModelSum(cstate, cactuation.nu)
        cost_T.addCost('stateReg_T',
            _crocoddyl.CostModelResidual(cstate,
                _crocoddyl.ResidualModelState(cstate, target_x, cactuation.nu)),
            NMPC_W_TERMINAL)
        diff_T = _crocoddyl.DifferentialActionModelContactFwdDynamics(
            cstate, cactuation, cm_T, cost_T, 0.0, True)
        return _crocoddyl.IntegratedActionModelEuler(diff_T, 0.0)

    # Receding horizon main loop
    N_HORIZON = NMPC_RH_N_HORIZON
    N_RESOLVE = NMPC_RH_N_RESOLVE
    print(f"  Receding horizon NMPC: N_HORIZON={N_HORIZON} ({N_HORIZON*DT_NMPC:.2f}s), "
          f"N_RESOLVE={N_RESOLVE}, N_TOTAL={N_TOTAL}")

    xs_full = [x0.copy()]   # 모든 step state
    us_full = []             # 모든 step control
    forces_full = []         # 각 step의 GRF dict (per-leg world-frame Fxyz)
    x_current = x0.copy()
    xs_warm = None           # 이전 풀이의 xs (warm-start용)
    us_warm = None

    t_solve_total = time.time()
    n_solves = 0
    n_failures = 0
    iter_total = 0
    fi_nmpc = 0
    perturb_done = False
    while fi_nmpc < N_TOTAL:
        # Perturbation 주입: PERTURB_TIME 도달 시 x_current 의 body velocity 에 impulse 추가
        t_now = fi_nmpc * DT_NMPC
        if USE_PERTURBATION and (not perturb_done) and t_now >= PERTURB_TIME:
            # x_current layout: [q (nq), v (nv)]; v[0:3] = body linear vel, v[3:6] = ang vel
            x_current[pin_model.nq    :pin_model.nq + 3] += PERTURB_VEL_LIN
            x_current[pin_model.nq + 3:pin_model.nq + 6] += PERTURB_VEL_ANG
            print(f"  [PERTURB] t={t_now:.2f}s: body v += {PERTURB_VEL_LIN}, ω += {PERTURB_VEL_ANG}")
            perturb_done = True
        # 남은 step에 맞춰 horizon 길이 조정
        rem = N_TOTAL - fi_nmpc
        h_eff = min(N_HORIZON, rem)
        actions_h = [_build_action_at_step(fi_nmpc + k) for k in range(h_eff)]
        # Terminal: 끝에 도달하면 final ref, 아니면 N_HORIZON 끝의 ref
        target_x_T = x0.copy()
        target_x_T[0] = q0[0] + V * (fi_nmpc + h_eff) * DT_NMPC
        terminal = _build_terminal(target_x_T)

        problem = _crocoddyl.ShootingProblem(x_current, actions_h, terminal)
        solver = _crocoddyl.SolverFDDP(problem)

        # Cold start each iteration (warm-start 로직 부정확하면 발산 위험)
        # 매 iteration 짧은 horizon (~24 steps)이라 cold start로도 100~200 iter 안에 수렴
        xs_init = [x_current] * (h_eff + 1)
        us_init = [np.zeros(cactuation.nu)] * h_eff
        # Warm-start (선택적): 이전 풀이의 us 패턴 활용
        if us_warm is not None:
            for k in range(min(len(us_warm) - N_RESOLVE, h_eff)):
                us_init[k] = np.array(us_warm[N_RESOLVE + k])

        done = solver.solve(xs_init, us_init,
                             maxiter=NMPC_MAXITER,
                             is_feasible=False,
                             init_reg=NMPC_INIT_REG)
        n_solves += 1
        iter_total += solver.iter
        if not done:
            n_failures += 1

        # Apply 첫 N_RESOLVE step (또는 horizon 끝까지)
        n_apply = min(N_RESOLVE, h_eff)
        for k in range(n_apply):
            xs_full.append(np.array(solver.xs[k+1]))
            us_full.append(np.array(solver.us[k]))
            # Contact forces 추출 (각 발의 GRF, world frame)
            f_step = {leg: np.zeros(3) for leg in ['FR','FL','HR','HL']}
            try:
                ad_k = solver.problem.runningDatas[k]
                contacts = ad_k.differential.multibody.contacts.contacts.todict()
                for name, cdata in contacts.items():
                    # name = 'c_FR', 'c_FL' etc.
                    leg = name.replace('c_', '')
                    if leg in f_step:
                        # ContactModel3D: f is pin.Force (linear + angular)
                        f_step[leg] = np.array(cdata.f.linear).copy()
            except Exception:
                pass   # contacts 추출 실패해도 진행 (0으로)
            forces_full.append(f_step)
        x_current = np.array(solver.xs[n_apply])
        xs_warm = np.array(solver.xs)
        us_warm = np.array(solver.us)
        fi_nmpc += n_apply

    elapsed = time.time() - t_solve_total
    print(f"  Receding horizon 완료: {n_solves} solves, total {iter_total} iters, "
          f"{n_failures} fails, {elapsed*1e3:.0f}ms")

    # FDDP의 partial solution도 동역학 만족하므로 사용
    # 단, 모두 실패하면 fallback (1번이라도 성공해야 의미 있음)
    success = (n_solves - n_failures) >= 1
    if n_failures > 0:
        print(f"    [INFO] {n_failures}/{n_solves} solves failed line search but partial "
              f"trajectories used (FDDP enforces dynamics).")
    return np.array(xs_full), np.array(us_full), forces_full, success, pin_model, pin_data



def _populate_arrays_from_nmpc(xs, us, forces, pin_model, pin_data):
    """NMPC xs/us → v11 arrays 채움.
    DT_NMPC × N_NMPC = 시뮬 총 시간이지만 v11 N_FRAMES은 DT 기반.
    NMPC trajectory를 v11 frame rate에 맞게 보간해서 채움.
    """
    import pinocchio as pin
    N_NMPC = len(us)   # NMPC step 수
    DT_NMPC = T / 2 / int((T/2)/DT_MPC)
    # NMPC time axis: 0, DT_NMPC, 2·DT_NMPC, ..., N_NMPC·DT_NMPC
    t_nmpc = np.arange(N_NMPC + 1) * DT_NMPC

    # v11 leg ordering: FR(0), FL(1), HR(2), HL(3)
    # pinocchio idx_q/idx_v 매핑
    leg_q_idx_pin = {leg: [pin_model.idx_qs[pin_model.getJointId(f'leg_{leg}_j{i+1}')]
                            for i in range(5)]
                     for leg in ['FR','FL','HR','HL']}
    leg_v_idx_pin = {leg: [pin_model.idx_vs[pin_model.getJointId(f'leg_{leg}_j{i+1}')]
                            for i in range(5)]
                     for leg in ['FR','FL','HR','HL']}

    # Linear interp from t_nmpc → t_v11
    for fi in range(N_FRAMES):
        t_v11 = fi * DT
        # Clamp to NMPC range
        if t_v11 >= t_nmpc[-1]:
            k = N_NMPC
            alpha = 0.0
        else:
            k = int(t_v11 / DT_NMPC)
            alpha = (t_v11 - k * DT_NMPC) / DT_NMPC
            k = min(k, N_NMPC - 1)
        # State interp (xs[k] and xs[k+1])
        x = xs[k] * (1 - alpha) + xs[k+1] * alpha if k < N_NMPC else xs[-1]
        # us[k] (zero-order hold)
        u = us[min(k, N_NMPC - 1)]

        # Joint state per leg (v11 ordering)
        for leg_idx, leg in enumerate(['FR', 'FL', 'HR', 'HL']):
            nj = N_JOINTS_PER_LEG[leg_idx]
            for j in range(5):
                joint_hist[fi, leg_idx, j] = x[leg_q_idx_pin[leg][j]]
        # Body state (in pinocchio convention: q[0:3] pos, q[3:7] quat, v[0:3] lin, v[3:6] ang)
        body_pos_hist[fi]   = x[0:3]
        # Quaternion → R
        qx, qy, qz, qw = x[3], x[4], x[5], x[6]
        body_R_hist[fi]     = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        body_v_hist[fi]     = x[pin_model.nq:pin_model.nq+3]
        body_omega_hist[fi] = x[pin_model.nq+3:pin_model.nq+6]
        body_pos_ref_hist[fi] = np.array([V * t_v11, 0.0, -_foot_z_home])
        body_v_ref_hist[fi]   = np.array([V, 0.0, 0.0])

        # Control (per-leg, 5 dim each)
        for leg_idx, leg in enumerate(['FR', 'FL', 'HR', 'HL']):
            for j in range(5):
                # u indexed by actuation.nu (= 20)
                # Actuation idx: pin_model.idx_vs[joint_id] - 6 (base 6 unactuated)
                u_idx = leg_v_idx_pin[leg][j] - 6
                wbc_tau_cmd[fi, leg_idx, j] = u[u_idx]

        # GRF per leg (zero-order hold, NMPC step k = us index)
        if forces is not None and len(forces) > 0:
            k_force = min(int(t_v11 / DT_NMPC), len(forces) - 1)
            f_dict = forces[k_force]
            for leg_idx, leg in enumerate(['FR', 'FL', 'HR', 'HL']):
                wbc_lam_des[fi, leg_idx]   = f_dict[leg]
                wbic_lam_used[fi, leg_idx] = f_dict[leg]

    # foot_hist 재계산 (joint_hist 기반 forward kinematics)
    for fi in range(N_FRAMES):
        for leg_idx in range(4):
            nj = N_JOINTS_PER_LEG[leg_idx]
            q_leg = joint_hist[fi, leg_idx, :nj]
            pts_dh = forward_kinematics(q_leg, dh=LEG_DH[leg_idx])
            foot_local_sim = _dh_to_sim(pts_dh[-1], front_leg=(leg_idx < 2))
            foot_hist[fi, leg_idx] = LEG_HIP_OFFSETS[leg_idx] + foot_local_sim

    # foot world pos (actual) + swing target (cmd) — fig6 용
    leg_to_idx = {'FR':0, 'FL':1, 'HR':2, 'HL':3}
    for fi in range(N_FRAMES):
        body_p = body_pos_hist[fi]
        R_b    = body_R_hist[fi]
        for leg_idx in range(4):
            foot_actual_world_hist[fi, leg_idx] = body_p + R_b @ foot_hist[fi, leg_idx]
    # swing target (smoothstep + bell, world frame)
    for fi in range(N_FRAMES):
        t = fi * DT
        in_cycle_t = t % T
        in_phase_A = in_cycle_t < (T / 2)
        if in_phase_A:
            swing_legs = ['FR','HL']; t_in_phase = in_cycle_t
            phase_start_t = (t // T) * T
        else:
            swing_legs = ['FL','HR']; t_in_phase = in_cycle_t - T/2
            phase_start_t = (t // T) * T + T/2
        sw_t = max(0.0, min(1.0, t_in_phase / (T/2)))
        for leg in swing_legs:
            li = leg_to_idx[leg]
            ps = foot_actual_world_hist[0, li].copy()
            ps[0] += V * phase_start_t - STEP_LENGTH/2
            pe = foot_actual_world_hist[0, li].copy()
            pe[0] += V * (phase_start_t + T/2) + STEP_LENGTH/2
            s_xy = sw_t * sw_t * (3.0 - 2.0 * sw_t)
            tgt  = ps + s_xy * (pe - ps)
            tgt[2] = ps[2] + STEP_HEIGHT * 16.0 * sw_t * sw_t * (1-sw_t) * (1-sw_t)
            foot_target_world_hist[fi, li] = tgt

    # tau_cmd 후처리 분해 (fig3/fig4 시각화용) — 제어 루프는 NMPC 단독, 값 변경 없음.
    # tau_grf = -Jᵀ × R_body^T × λ_world         (body-local GRF feedforward)
    # tau_dyn = full RNEA(q, q̇, q̈) = M·q̈ + C·q̇ + g  (MPC+WBIC의 tau_dyn과 동일 의미)
    # tau_pd  = 0  (NMPC는 명시적 PD 피드백 항이 없음 — whole-body OC가 통합 처리)
    # tau_imp = tau_cmd - tau_dyn - tau_grf       (잔차: per-leg 분해 ↔ whole-body NMPC 불일치)
    #           작을수록 분해가 NMPC 해와 정합. 크면 모델 mismatch / λ_des≠실제 접촉력.
    _dq_hist  = np.gradient(joint_hist, DT, axis=0)
    _ddq_hist = np.gradient(_dq_hist,   DT, axis=0)
    for fi in range(N_FRAMES):
        R_b = body_R_hist[fi]
        for leg_idx in range(4):
            nj   = N_JOINTS_PER_LEG[leg_idx]
            q_leg   = joint_hist[fi, leg_idx, :nj]
            dq_leg  = _dq_hist[fi, leg_idx, :nj]
            ddq_leg = _ddq_hist[fi, leg_idx, :nj]
            front = (leg_idx < 2)
            dh    = LEG_DH[leg_idx]
            lm    = LINK_MASS_PER_LEG[leg_idx]
            J     = compute_jacobian_sim(q_leg, dh, front)   # 3×nj
            lam_world  = wbc_lam_des[fi, leg_idx]
            lam_local  = R_b.T @ lam_world
            wbc_tau_grf[fi, leg_idx, :nj] = -(J.T @ lam_local)
            # full RNEA (중력만이 아닌 M·q̈ + C·q̇ + g 전체)
            if USE_PINOCCHIO and _PIN_AVAILABLE:
                wbc_tau_dyn[fi, leg_idx, :nj] = _pin_helpers.rnea_pin_per_leg(
                    list(q_leg), list(dq_leg), list(ddq_leg), LEG_NAMES[leg_idx])
            else:
                wbc_tau_dyn[fi, leg_idx, :nj] = rnea(q_leg, dq_leg, ddq_leg, dh, lm)
            # PD = 0 (NMPC는 명시적 PD 항 없음), imp = remaining 잔차
            wbc_tau_pd[fi, leg_idx, :nj]  = 0.0
            wbc_tau_imp[fi, leg_idx, :nj] = (wbc_tau_cmd[fi, leg_idx, :nj]
                                              - wbc_tau_dyn[fi, leg_idx, :nj]
                                              - wbc_tau_grf[fi, leg_idx, :nj])
            # lam_calc = lam_des (NMPC는 QP 분리 안 됨)
            wbc_lam_calc[fi, leg_idx] = wbc_lam_des[fi, leg_idx]


_USE_NMPC_ACTIVE = False
if USE_NMPC and _CROCODDYL_AVAILABLE:
    print("─" * 55)
    if USE_NMPC_RECEDING:
        print(f"v12: NMPC (receding horizon FDDP) 풀이 시작...")
        _xs_nmpc, _us_nmpc, _forces_nmpc, _done_nmpc, _pin_model_nmpc, _pin_data_nmpc = _solve_nmpc_trot_receding()
    else:
        print(f"v12: NMPC (one-shot FDDP) 풀이 시작...")
        _xs_nmpc, _us_nmpc, _forces_nmpc, _done_nmpc, _pin_model_nmpc, _pin_data_nmpc = _solve_nmpc_trot()
    if _done_nmpc:
        _populate_arrays_from_nmpc(_xs_nmpc, _us_nmpc, _forces_nmpc, _pin_model_nmpc, _pin_data_nmpc)
        _USE_NMPC_ACTIVE = True
        print(f"  v11 arrays NMPC 결과로 채움. WBC + MPC 메인 루프 SKIP.")
    else:
        print(f"  ⚠ NMPC 수렴 실패 — v11 동작 (MPC+WBIC) fallback")

# v11 main loop: NMPC 활성 시 frame body 전체 skip (continue)
for fi in range(N_FRAMES):
    if _USE_NMPC_ACTIVE:
        continue   # NMPC가 이미 모든 array 채워둠
    t_cur = fi * DT

    # ── GRF 목표 계산 (MPC QP 또는 QP GRF) ──────────────────
    contact_mask = ~swing_flag[fi]   # (4,) bool

    if USE_MPC:
        # horizon 내 contact schedule 예측
        cs = np.zeros((N_MPC, 4), dtype=bool)
        fp = np.zeros((N_MPC, 4, 3))
        for k in range(N_MPC):
            t_k = t_cur + k * DT_MPC
            for leg in range(4):
                cs[k, leg] = not sched.is_swing(leg, t_k)
                fp[k, leg] = foot_hist[fi, leg]   # 발 위치 현재값 유지 (quasi-static)

        # body 상태 x0
        if USE_MPC_CLOSED_LOOP and USE_BODY_DYNAMICS:
            # 실 body state 피드백 (closed-loop)
            roll, pitch, yaw = _R_to_euler_xyz(body_state['R'])
            x0_mpc = np.array([
                roll, pitch, yaw,
                body_state['pos'][0], body_state['pos'][1], body_state['pos'][2],
                body_state['omega'][0], body_state['omega'][1], body_state['omega'][2],
                body_state['v'][0], body_state['v'][1], body_state['v'][2],
                -G_ACC,
            ])
            # x_ref: 단순 steady 궤적 (upright + vx=V + z=0)
            x_ref_step = BODY_REF_STEP.copy()
            x_ref_step[5]  = -_foot_z_home   # body z 목표 = home (≈ +0.465m, foot at ground)
            x_ref_step[9]  = V       # vx 추종
            x_ref_step[12] = -G_ACC
        else:
            # open-loop (이상적 hover-at-origin, v10 동작)
            x0_mpc = np.array([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                V,   0.0, 0.0,
                -G_ACC
            ])
            x_ref_step = None  # mpc_qp_plan이 hover-at-x0 사용
        # closed-loop이면 LTV (현재 자세 기반 정확한 A,B), 아니면 small-angle 시불변
        lam_des_all = mpc_qp_plan(x0_mpc, cs, fp, x_ref_step=x_ref_step,
                                   ltv=USE_MPC_CLOSED_LOOP and USE_BODY_DYNAMICS)
        # swing foot 강제 0
        for leg in range(4):
            if swing_flag[fi, leg]:
                lam_des_all[leg] = 0.0
    else:
        lam_des_all = qp_grf_distribute(contact_mask, foot_hist[fi])

    # Contact ramp: stance 시작/끝 GRF_RAMP_RATIO 구간에서 λ_des를 smoothstep으로 보간
    # → swing↔stance 전환 시 step jump 제거 (cmd torque 부드러운 전환)
    for leg in range(4):
        if not swing_flag[fi, leg]:
            st_t = sched.stance_t(leg, t_cur)
            if st_t < GRF_RAMP_RATIO:
                tau_r = st_t / GRF_RAMP_RATIO
                ramp  = 10*tau_r**3 - 15*tau_r**4 + 6*tau_r**5
                lam_des_all[leg] = lam_des_all[leg] * ramp
            elif st_t > 1.0 - GRF_RAMP_RATIO:
                tau_r = (1.0 - st_t) / GRF_RAMP_RATIO
                ramp  = 10*tau_r**3 - 15*tau_r**4 + 6*tau_r**5
                lam_des_all[leg] = lam_des_all[leg] * ramp

    wbc_lam_des[fi] = lam_des_all

    # ── Pass 1: per-leg kinematics + RNEA + M, h ─────────────
    leg_data = [None]*4
    foot_world_all = np.zeros((4, 3))
    for leg in range(4):
        nj    = N_JOINTS_PER_LEG[leg]
        front = leg < 2
        dh    = LEG_DH[leg]
        lm    = LINK_MASS_PER_LEG[leg]

        q_t   = joint_hist[fi, leg, :nj]
        q_a   = theta_a_hist[fi, leg, :nj]
        dq_t  = joint_vel_hist[fi, leg, :nj]
        dq_a  = dtheta_a_hist[fi, leg, :nj]
        ddq_t = joint_acc_hist[fi, leg, :nj]

        if USE_PINOCCHIO and _PIN_AVAILABLE:
            _leg_name_pin = LEG_NAMES[leg]
            J   = _pin_helpers.compute_jacobian_sim_pin(list(q_t), _leg_name_pin)
            J_a = _pin_helpers.compute_jacobian_sim_pin(list(q_a), _leg_name_pin)
        else:
            J   = compute_jacobian_sim(q_t, dh, front)
            J_a = compute_jacobian_sim(q_a, dh, front)
        tau_g = compute_gravity_torque_sim(q_t, dh, lm, front)

        lam_des_leg = lam_des_all[leg]

        if USE_PINOCCHIO and _PIN_AVAILABLE:
            tau_dyn_leg = _pin_helpers.rnea_pin_per_leg(list(q_t), list(dq_t), list(ddq_t),
                                                         LEG_NAMES[leg])
        else:
            tau_dyn_leg  = rnea(q_t, dq_t, ddq_t, dh, lm)
        tau_grf_leg  = -J.T @ lam_des_leg
        tau_ff_leg   = tau_dyn_leg + tau_grf_leg

        # foot world position (body integration용 r_i 계산)
        foot_world_all[leg] = body_state['pos'] + body_state['R'] @ foot_hist[fi, leg]

        if (USE_WBIC or USE_WBIC_FB):
            if USE_PINOCCHIO and _PIN_AVAILABLE:
                M_leg, h_leg = _pin_helpers.compute_mh_leg_pin(list(q_t), list(dq_t), LEG_NAMES[leg])
            else:
                M_leg, h_leg = compute_mh_leg(q_t, dq_t, dh, lm)
        else:
            M_leg, h_leg = None, None

        foot_t_j5 = foot_local[fi, leg] + J4_TO_J5_SIM_PER_LEG[leg]
        pts_a     = forward_kinematics(q_a, dh=dh)
        foot_a_j5 = _dh_to_sim(pts_a[-1], front_leg=front)
        vel_t     = foot_vel_t[fi, leg]
        vel_a     = J_a @ dq_a
        f_imp       = KP_IMP * (foot_t_j5 - foot_a_j5) + KD_IMP * (vel_t - vel_a)
        tau_imp_leg = J.T @ f_imp
        tau_pd_leg  = KP_PD[:nj] * (q_t - q_a) + KD_PD[:nj] * (dq_t - dq_a)

        leg_data[leg] = dict(nj=nj, J=J, ddq_t=ddq_t,
                             tau_g=tau_g, tau_dyn=tau_dyn_leg, tau_grf=tau_grf_leg,
                             tau_ff=tau_ff_leg, tau_pd=tau_pd_leg, tau_imp=tau_imp_leg,
                             M_leg=M_leg, h_leg=h_leg,
                             lam_des=lam_des_leg, lam_used=lam_des_leg.copy())

    # ── Pass 2: WBIC (FB single QP OR per-leg) ──────────────
    used_fb = False
    if USE_WBIC_FB:
        v_dot_des_fb = np.zeros(6)

        if USE_PINOCCHIO and USE_PINOCCHIO_FULL_M and _PIN_AVAILABLE:
            # FULL M(q) WBIC FB — pinocchio CRBA + Jacobian (floating base coupling 정확)
            # ⚠️ 실험적: 현재 90% solver fail (per-leg τ_ff와 full M 모델 inconsistency).
            # τ_ff와 ddq_des를 full pinocchio 모델로 일관되게 계산하지 않으면 발산.
            leg_q_dict  = {LEG_NAMES[i]: list(joint_hist[fi, i, :N_JOINTS_PER_LEG[i]])
                           for i in range(4)}
            leg_dq_dict = {LEG_NAMES[i]: list(joint_vel_hist[fi, i, :N_JOINTS_PER_LEG[i]])
                           for i in range(4)}
            M_full_pin, h_full_pin, q_full_pin, _ = _pin_helpers.compute_full_M_h(
                leg_q_dict, leg_dq_dict)
            # Pinocchio는 다리를 알파벳순 (FL/FR/HL/HR) 정렬 → v11 순서 (FR/FL/HR/HL)로 permute
            # perm[i] = pinocchio v-idx that 매핑되는 v11 v-idx i
            _pin_v_idx = _pin_helpers._LEG_V_IDX   # {FR, FL, HR, HL: list of 5 pin v-idx}
            perm = list(range(6))   # base (0..5) 그대로
            for v11_leg_idx in range(4):
                leg_name = LEG_NAMES[v11_leg_idx]   # FR, FL, HR, HL
                perm.extend(_pin_v_idx[leg_name])
            perm = np.array(perm)
            # M, h를 v11 순서로 재배열
            M_full = M_full_pin[np.ix_(perm, perm)]
            h_full = h_full_pin[perm]
            # 12×26 contact Jacobian (4 feet × 3 linear), pin 결과를 v11 순서 col로
            import pinocchio as _pin
            _pin.computeJointJacobians(_pin_helpers._MODEL, _pin_helpers._DATA, q_full_pin)
            _pin.updateFramePlacements(_pin_helpers._MODEL, _pin_helpers._DATA)
            J_full = np.zeros((12, _pin_helpers._MODEL.nv))
            for fi_idx, leg in enumerate(LEG_NAMES):
                fid = _pin_helpers._LEG_FOOT_FID[leg]
                J6 = _pin.getFrameJacobian(_pin_helpers._MODEL, _pin_helpers._DATA, fid,
                                            _pin.LOCAL_WORLD_ALIGNED)
                J_full[fi_idx*3:(fi_idx+1)*3, :] = J6[:3, :]
            # J columns도 v11 순서로 permute
            J_full = J_full[:, perm]

            # ddq_des=0 (quasi-static target) — numerical noise 회피.
            # τ_ff도 ddq=0 가정으로 다시 계산 (pinocchio gravity + Coriolis만).
            # 이렇게 하면 model consistency 확보.
            v_full_pin = np.zeros(_pin_helpers._MODEL.nv)
            for v11_leg_idx in range(4):
                leg_name = LEG_NAMES[v11_leg_idx]
                for j, dqj in enumerate(joint_vel_hist[fi, v11_leg_idx, :N_JOINTS_PER_LEG[v11_leg_idx]]):
                    v_full_pin[_pin_helpers._LEG_V_IDX[leg_name][j]] = dqj
            import pinocchio as _pin
            # tau_full_q = pin.rnea(q, v, 0) = M·0 + C·v + g = h(q,v)
            tau_full_q_pin = _pin.rnea(_pin_helpers._MODEL, _pin_helpers._DATA,
                                        q_full_pin, v_full_pin, np.zeros(_pin_helpers._MODEL.nv))
            tau_full_q = tau_full_q_pin[perm]   # v11 순서로 재배열
            # ⇒ tau_ff_legs = tau_full_q[6:] in v11 leg order (5 per leg)
            tau_ff_legs_pin = []
            for v11_leg_idx in range(4):
                start = 6 + v11_leg_idx * 5
                tau_ff_legs_pin.append(tau_full_q[start:start+5])

            fb_out = wbic_qp_full_pin(
                M_full=M_full, h_full=h_full, J_full=J_full,
                ddq_des_legs=[np.zeros(N_JOINTS_PER_LEG[i]) for i in range(4)],
                tau_ff_legs=tau_ff_legs_pin,
                lam_des_all=lam_des_all,
                contact_mask=contact_mask, nj_per_leg=N_JOINTS_PER_LEG,
                v_dot_des_fb=v_dot_des_fb,
                w_ddq=WBIC_W_DDQ, w_tau=WBIC_W_TAU, w_lam=WBIC_W_LAM, w_fb=WBIC_W_FB,
                lamz_min=WBIC_LAMZ_MIN, mu=MU_FRICTION,
            )
            if fb_out is not None:
                used_fb = True
                wbic_fb_residual_hist[fi] = fb_out['residual_full']
                wbic_fb_status_hist[fi]   = True
                wbic_fb_dvfb_hist[fi]     = fb_out['d_v_fb']
                for leg in range(4):
                    nj = leg_data[leg]['nj']
                    d_tau = fb_out['d_tau_legs'][leg]
                    d_lam = fb_out['d_lam_legs'][leg]
                    leg_data[leg]['tau_ff']  = leg_data[leg]['tau_ff'] + d_tau
                    leg_data[leg]['lam_used'] = leg_data[leg]['lam_des'] + d_lam
                    wbic_dtau_hist[fi, leg, :nj] = d_tau
                    wbic_dlam_hist[fi, leg]      = d_lam
                    wbic_status_hist[fi, leg]    = True
        else:
            # 기존 block-diagonal WBIC FB (v11 native)
            R_world = body_state['R']
            I_world = R_world @ BODY_INERTIA @ R_world.T

            # ANYmal-style stance foot 제약: J·Δq̈_full = 0 (pinocchio J_full 사용)
            stance_foot_J = None
            if USE_STANCE_FOOT_CONSTRAINT and USE_PINOCCHIO and _PIN_AVAILABLE:
                leg_q_dict_sf = {LEG_NAMES[i]: list(joint_hist[fi, i, :N_JOINTS_PER_LEG[i]])
                                  for i in range(4)}
                q_full_sf = _pin_helpers._build_full_q(leg_q_dict_sf)
                import pinocchio as _pin_sf
                _pin_sf.computeJointJacobians(_pin_helpers._MODEL,
                                               _pin_helpers._DATA, q_full_sf)
                _pin_sf.updateFramePlacements(_pin_helpers._MODEL, _pin_helpers._DATA)
                # v11 순서로 col permute
                _perm_sf = list(range(6))
                for v11_idx in range(4):
                    _perm_sf.extend(_pin_helpers._LEG_V_IDX[LEG_NAMES[v11_idx]])
                _perm_sf = np.array(_perm_sf)
                stance_foot_J = []
                for v11_idx in range(4):
                    fid = _pin_helpers._LEG_FOOT_FID[LEG_NAMES[v11_idx]]
                    J6 = _pin_sf.getFrameJacobian(_pin_helpers._MODEL,
                                                   _pin_helpers._DATA, fid,
                                                   _pin_sf.LOCAL_WORLD_ALIGNED)
                    stance_foot_J.append(J6[:3, _perm_sf])  # 3×26 in v11 order

            fb_out = wbic_qp_full(
                M_legs=[leg_data[i]['M_leg'] for i in range(4)],
                h_legs=[leg_data[i]['h_leg'] for i in range(4)],
                ddq_des_legs=[leg_data[i]['ddq_t'] for i in range(4)],
                tau_ff_legs=[leg_data[i]['tau_ff'] for i in range(4)],
                lam_des_all=lam_des_all,
                J_legs=[leg_data[i]['J'] for i in range(4)],
                contact_mask=contact_mask, nj_per_leg=N_JOINTS_PER_LEG,
                foot_world_all=foot_world_all, body_pos=body_state['pos'],
                v_dot_des_fb=v_dot_des_fb,
                M_total=TOTAL_MASS, I_body=I_world, omega_world=body_state['omega'],
                w_ddq=WBIC_W_DDQ, w_tau=WBIC_W_TAU, w_lam=WBIC_W_LAM, w_fb=WBIC_W_FB,
                lamz_min=WBIC_LAMZ_MIN, mu=MU_FRICTION,
                stance_foot_J_v11=stance_foot_J,
                tau_prev_legs=tau_ff_corrected_prev, w_dtau=WBIC_W_DTAU,
            )
            if fb_out is not None:
                used_fb = True
                wbic_fb_residual_hist[fi] = fb_out['residual_fb']
                wbic_fb_status_hist[fi]   = True
                wbic_fb_dvfb_hist[fi]     = fb_out['d_v_fb']
                for leg in range(4):
                    nj = leg_data[leg]['nj']
                    d_tau = fb_out['d_tau_legs'][leg]
                    d_lam = fb_out['d_lam_legs'][leg]
                    leg_data[leg]['tau_ff']  = leg_data[leg]['tau_ff'] + d_tau
                    leg_data[leg]['lam_used'] = leg_data[leg]['lam_des'] + d_lam
                    wbic_dtau_hist[fi, leg, :nj] = d_tau
                    wbic_dlam_hist[fi, leg]      = d_lam
                    wbic_residual_hist[fi, leg]  = fb_out['residual_legs'][leg]
                    wbic_status_hist[fi, leg]    = True

        if not used_fb:
            wbic_fb_fail_count += 1   # FB 시도 모두 실패 → fallback per-leg

    if (not used_fb) and USE_WBIC:
        for leg in range(4):
            d = leg_data[leg]
            nj = d['nj']
            d_ddq, d_tau, d_lam, ok, res = wbic_qp_leg(
                d['M_leg'], d['h_leg'], d['ddq_t'], d['tau_ff'], d['lam_des'], d['J'],
                contact=contact_mask[leg], nj=nj,
                w_ddq=WBIC_W_DDQ, w_tau=WBIC_W_TAU, w_lam=WBIC_W_LAM,
                lamz_min=WBIC_LAMZ_MIN, mu=MU_FRICTION,
                tau_prev=tau_ff_corrected_prev[leg], w_dtau=WBIC_W_DTAU,
            )
            wbic_residual_hist[fi, leg] = res
            wbic_status_hist[fi, leg]   = ok
            if ok:
                d['tau_ff']  = d['tau_ff'] + d_tau
                d['lam_used'] = d['lam_des'] + d_lam
                wbic_dtau_hist[fi, leg, :nj] = d_tau
                wbic_dlam_hist[fi, leg]      = d_lam
            else:
                wbic_fail_count += 1

    # ── Pass 3: τ_cmd 계산 + 히스토리 저장 ──────────────────
    for leg in range(4):
        d = leg_data[leg]
        nj = d['nj']
        tau_cmd_leg = d['tau_pd'] + d['tau_ff'] + d['tau_imp']
        tau_cmd_leg = np.clip(tau_cmd_leg, -JOINT_TORQUE_LIMIT[:nj], JOINT_TORQUE_LIMIT[:nj])
        wbic_lam_used[fi, leg] = d['lam_used']

        JJT          = d['J'] @ d['J'].T + MU_DAMP * np.eye(3)
        lam_calc_leg = np.linalg.solve(JJT, d['J'] @ (d['tau_g'] - tau_cmd_leg))

        wbc_tau_ff  [fi, leg, :nj] = d['tau_ff']
        wbc_tau_dyn [fi, leg, :nj] = d['tau_dyn']
        wbc_tau_pd  [fi, leg, :nj] = d['tau_pd']
        wbc_tau_imp [fi, leg, :nj] = d['tau_imp']
        wbc_tau_grf [fi, leg, :nj] = d['tau_grf']
        wbc_tau_cmd [fi, leg, :nj] = tau_cmd_leg
        wbc_lam_calc[fi, leg]      = lam_calc_leg
        # v12 smoothness: 다음 frame WBIC 의 w_dtau 항을 위해 *post-correction* tau_ff 보관.
        tau_ff_corrected_prev[leg] = d['tau_ff'].copy()

    # ── Pass 4: Floating-base body 6-DoF 적분 ──────────────
    if USE_BODY_DYNAMICS:
        integrate_body_state(body_state, wbic_lam_used[fi], foot_world_all,
                             TOTAL_MASS, BODY_INERTIA, DT)
    # 항상 히스토리 저장 (USE_BODY_DYNAMICS=False면 정상 보행 ref)
    body_pos_hist[fi]   = body_state['pos']
    body_R_hist[fi]     = body_state['R']
    body_v_hist[fi]     = body_state['v']
    body_omega_hist[fi] = body_state['omega']
    body_alin_hist[fi]  = body_state.get('a_lin', np.zeros(3))
    body_aang_hist[fi]  = body_state.get('a_ang', np.zeros(3))
    # Reference (kinematic V·t)
    body_pos_ref_hist[fi] = np.array([V * t_cur, 0.0, -_foot_z_home])
    body_v_ref_hist[fi]   = np.array([V, 0.0, 0.0])

# fig6 용 foot_actual / foot_target — NMPC populate 외에서도 채움 (v11 standalone 포함)
# sched 기반 일반 gait (walk/pace/trot/...) 지원
_LEGS_FIG6 = ['FR', 'FL', 'HR', 'HL']
_FOOT_HOME_WORLD = np.zeros((4, 3))
for _li in range(4):
    foot_actual_world_hist[0, _li] = body_pos_hist[0] + body_R_hist[0] @ foot_hist[0, _li]
    _FOOT_HOME_WORLD[_li] = foot_actual_world_hist[0, _li]
for _fi in range(N_FRAMES):
    body_p = body_pos_hist[_fi]; R_b = body_R_hist[_fi]
    for _li in range(4):
        foot_actual_world_hist[_fi, _li] = body_p + R_b @ foot_hist[_fi, _li]
    t_now = _fi * DT
    for _li, _leg_name in enumerate(_LEGS_FIG6):
        ph = sched.phase(_li, t_now)
        if ph < sched.swing_ratio:
            sw_t = ph / sched.swing_ratio
            t_sw_start = t_now - ph * T
            t_sw_end   = t_sw_start + T_SW
            ps = _FOOT_HOME_WORLD[_li].copy()
            ps[0] += V * t_sw_start - STEP_LENGTH/2
            pe = _FOOT_HOME_WORLD[_li].copy()
            pe[0] += V * t_sw_end + STEP_LENGTH/2
            s_xy = sw_t * sw_t * (3.0 - 2.0 * sw_t)
            tgt  = ps + s_xy * (pe - ps)
            tgt[2] = ps[2] + STEP_HEIGHT * 16.0 * sw_t * sw_t * (1-sw_t) * (1-sw_t)
            foot_target_world_hist[_fi, _li] = tgt
        else:
            foot_target_world_hist[_fi, _li] = np.nan   # swing 외 frame = NaN

wbc_dur = time.perf_counter() - wbc_t0
mode_str = f"MPC(N={N_MPC},dt={DT_MPC*1e3:.0f}ms)" if USE_MPC else "QP GRF"
wbic_str = "WBIC ON" if USE_WBIC else "WBIC OFF"
if _USE_NMPC_ACTIVE:
    print(f"v12 NMPC 활성 — v11 main loop SKIP, joint/body arrays NMPC trajectory로 채움")
    # NMPC 결과 진단 출력
    import math as _m
    _roll_max = _m.degrees(np.max(np.abs(np.arctan2(body_R_hist[:,2,1], body_R_hist[:,2,2]))))
    _pitch_max = _m.degrees(np.max(np.abs(np.arcsin(np.clip(-body_R_hist[:,2,0],-1,1)))))
    print(f"  body roll_max={_roll_max:.2f}°, pitch_max={_pitch_max:.2f}°")
    print(f"  body z range: [{body_pos_hist[:,2].min()*1e3:.2f}, {body_pos_hist[:,2].max()*1e3:.2f}] mm")
    print(f"  body x final: {body_pos_hist[-1,0]:.3f}m  (target {V*T*N_CYCLES:.3f})")
    print(f"  body vx mean: {body_v_hist[:,0].mean():.3f} m/s  (target {V})")
    print(f"  |τ|max: {np.abs(wbc_tau_cmd).max():.2f} Nm")
    fz_sum_des = np.zeros(N_FRAMES)   # placeholder (NMPC는 GRF 직접 풀이 안 함)
    fz_sum_used = np.zeros(N_FRAMES)
else:
    print(f"WBC 완료 [{mode_str}, {wbic_str}].  {wbc_dur*1e3:.1f}ms 총  ({wbc_dur/N_FRAMES*1e6:.1f}μs/frame)")

# ══════════════════════════════════════════════════════════════
# v13 좌우 mirror 진단 — 잔여 y drift 원인 추적용
# 6가지 후보를 자동 측정. 진짜 mirror 정합 시 모든 값 ≈ 0 또는 평균 ≈ 0.
# ══════════════════════════════════════════════════════════════
if not _USE_NMPC_ACTIVE:
    print("─" * 60)
    print("[v13 좌우 mirror 진단]")

    # 1) opt-IK 좌우 q 미러 일치: q_FR 과 q_FL 이 정확히 같아야 함 (Q_HOME 동일)
    q_diff_front = joint_hist[:, 0, :5] - joint_hist[:, 1, :5]   # FR - FL
    q_diff_hind  = joint_hist[:, 2, :5] - joint_hist[:, 3, :5]   # HR - HL
    print(f"  (1) opt-IK 좌우 q 차이 RMS [rad]:")
    print(f"      FR-FL: q1={np.sqrt(np.mean(q_diff_front[:,0]**2)):.2e}  "
          f"q2={np.sqrt(np.mean(q_diff_front[:,1]**2)):.2e}  "
          f"q3={np.sqrt(np.mean(q_diff_front[:,2]**2)):.2e}  "
          f"q4={np.sqrt(np.mean(q_diff_front[:,3]**2)):.2e}")
    print(f"      HR-HL: q1={np.sqrt(np.mean(q_diff_hind[:,0]**2)):.2e}  "
          f"q2={np.sqrt(np.mean(q_diff_hind[:,1]**2)):.2e}  "
          f"q3={np.sqrt(np.mean(q_diff_hind[:,2]**2)):.2e}  "
          f"q4={np.sqrt(np.mean(q_diff_hind[:,3]**2)):.2e}")

    # 2) MPC GRF y 시간 평균 — 대칭 시 0
    grf_y_mean_per_leg = wbc_lam_des[:, :, 1].mean(axis=0)
    print(f"  (2) MPC GRF λ_y 시간평균 [N]:")
    print(f"      FR={grf_y_mean_per_leg[0]:+.3f}  FL={grf_y_mean_per_leg[1]:+.3f}  "
          f"HR={grf_y_mean_per_leg[2]:+.3f}  HL={grf_y_mean_per_leg[3]:+.3f}")
    print(f"      4-leg sum (lateral net force) = {grf_y_mean_per_leg.sum():+.4f} N  "
          f"(대칭이면 0)")

    # 3) GRF 좌우 짝 합 (FR+HR=우측, FL+HL=좌측)
    grf_y_right = wbc_lam_des[:, [0,2], 1].sum(axis=1).mean()    # FR + HR
    grf_y_left  = wbc_lam_des[:, [1,3], 1].sum(axis=1).mean()    # FL + HL
    print(f"  (3) GRF y 좌우 합:  우측(FR+HR) avg={grf_y_right:+.4f}N  "
          f"좌측(FL+HL) avg={grf_y_left:+.4f}N  "
          f"합={grf_y_right+grf_y_left:+.4f}N")

    # 4) body 좌표축별 정상상태 offset (settle 후 후반 25% 평균)
    _settle_idx = slice(int(0.75*N_FRAMES), N_FRAMES)
    bp_steady = body_pos_hist[_settle_idx].mean(axis=0)
    bv_steady = body_v_hist[_settle_idx].mean(axis=0)
    print(f"  (4) body 정상상태 (후반 25% 평균):")
    print(f"      pos: x={bp_steady[0]*1e3:+.1f}mm  y={bp_steady[1]*1e3:+.2f}mm  "
          f"z={bp_steady[2]*1e3:+.1f}mm")
    print(f"      vel: vx={bv_steady[0]:+.4f}m/s  vy={bv_steady[1]:+.4f}m/s  "
          f"vz={bv_steady[2]:+.4f}m/s")

    # 5) WBIC fb residual y 성분 평균 (body 6-DoF QP 잔차)
    if USE_WBIC_FB and wbic_fb_status_hist.any():
        fb_res_mean = wbic_fb_residual_hist[wbic_fb_status_hist].mean()
        dvfb_mean = wbic_fb_dvfb_hist[wbic_fb_status_hist].mean(axis=0)
        print(f"  (5) WBIC FB residual avg={fb_res_mean:.2e}  "
              f"Δv̇_fb mean=({dvfb_mean[0]:+.3e},{dvfb_mean[1]:+.3e},{dvfb_mean[2]:+.3e}) "
              f"(lin) ({dvfb_mean[3]:+.3e},{dvfb_mean[4]:+.3e},{dvfb_mean[5]:+.3e}) (ang)")

    # 6) WBIC dtau / dlam 좌우 비교
    if USE_WBIC:
        dtau_y_rms_fr = np.sqrt(np.mean(wbic_dtau_hist[:, 0, :]**2))
        dtau_y_rms_fl = np.sqrt(np.mean(wbic_dtau_hist[:, 1, :]**2))
        dtau_y_rms_hr = np.sqrt(np.mean(wbic_dtau_hist[:, 2, :]**2))
        dtau_y_rms_hl = np.sqrt(np.mean(wbic_dtau_hist[:, 3, :]**2))
        print(f"  (6) WBIC Δτ RMS [Nm] per leg: FR={dtau_y_rms_fr:.4f}  FL={dtau_y_rms_fl:.4f}  "
              f"HR={dtau_y_rms_hr:.4f}  HL={dtau_y_rms_hl:.4f}  (대칭이면 FR≈FL, HR≈HL)")
    print("─" * 60)

# GRF 합산 검증 (λ_des = MPC/QP 출력, λ_used = WBIC 보정 후)
if not _USE_NMPC_ACTIVE:
    fz_sum_des  = np.sum(wbc_lam_des[:, :, 2], axis=1)
    fz_sum_used = np.sum(wbic_lam_used[:, :, 2], axis=1)
    print(f"  Σλz_des  평균={fz_sum_des.mean():.2f}N  (Mg={TOTAL_MASS*G_ACC:.2f}N)  "
          f"오차={abs(fz_sum_des.mean()-TOTAL_MASS*G_ACC):.2f}N")
    if USE_WBIC:
        print(f"  Σλz_used 평균={fz_sum_used.mean():.2f}N  "
              f"오차={abs(fz_sum_used.mean()-TOTAL_MASS*G_ACC):.2f}N")
        # WBIC 보정 통계
        dtau_max  = float(np.max(np.abs(wbic_dtau_hist)))
        dtau_mean = float(np.mean(np.abs(wbic_dtau_hist)))
        dlam_max  = float(np.max(np.abs(wbic_dlam_hist)))
        dlam_mean = float(np.mean(np.abs(wbic_dlam_hist)))
        res_max   = float(np.max(wbic_residual_hist))
        res_mean  = float(np.mean(wbic_residual_hist))
        n_fail    = int(np.sum(~wbic_status_hist))
        print(f"  WBIC: |Δτ|max={dtau_max:.3f}Nm mean={dtau_mean:.4f}Nm  "
              f"|Δλ|max={dlam_max:.3f}N mean={dlam_mean:.4f}N")
        print(f"  WBIC: residual max={res_max:.2e} mean={res_mean:.2e}  "
              f"solver fail={n_fail}/{4*N_FRAMES}")
        if dtau_max < 1e-3 and dlam_max < 1e-3:
            print(f"  [INFO] WBIC 보정량 0 — 모든 제약이 비활성. WBIC OFF와 동일 결과.")
        if n_fail > 0:
            print(f"  [WARNING] WBIC solver {n_fail}회 실패. 제약 충돌 가능성.")

    # v11: WBIC FB 진단
    if USE_WBIC_FB:
        fb_res_max  = float(np.max(wbic_fb_residual_hist))
        fb_res_mean = float(np.mean(wbic_fb_residual_hist))
        fb_dvfb_max = float(np.max(np.abs(wbic_fb_dvfb_hist)))
        print(f"  WBIC FB: residual_fb max={fb_res_max:.2e} mean={fb_res_mean:.2e}  "
              f"|Δv̇_fb|max={fb_dvfb_max:.4f}  fail={wbic_fb_fail_count}/{N_FRAMES}")

# v11/v12: Floating-base body 동역학 진단
if USE_BODY_DYNAMICS and not _USE_NMPC_ACTIVE:
    body_pos_dev   = body_pos_hist - body_pos_ref_hist
    body_v_dev     = body_v_hist   - body_v_ref_hist
    pos_dev_norm   = float(np.max(np.linalg.norm(body_pos_dev, axis=1)))
    v_dev_norm     = float(np.max(np.linalg.norm(body_v_dev, axis=1)))
    pitch_max      = float(np.max(np.abs(np.arcsin(np.clip(-body_R_hist[:, 2, 0], -1, 1)))))
    roll_max       = float(np.max(np.abs(np.arctan2(body_R_hist[:, 2, 1], body_R_hist[:, 2, 2]))))
    omega_max      = float(np.max(np.abs(body_omega_hist)))
    z_dev_max      = float(np.max(np.abs(body_pos_hist[:, 2])))   # CoM 수직 진동
    print(f"  Body dyn: |pos_dev|max={pos_dev_norm*1e3:.2f}mm  |v_dev|max={v_dev_norm*1e3:.2f}mm/s  "
          f"|z|max={z_dev_max*1e3:.2f}mm")
    print(f"  Body dyn: pitch_max={math.degrees(pitch_max):.3f}° roll_max={math.degrees(roll_max):.3f}° "
          f"|ω|max={omega_max:.4f}rad/s")
    if body_state.get('_diverged', False):
        print(f"  [WARNING] body 발산 감지 (|ω|>{50}rad/s 또는 |v|>{50}m/s 발생) — clamp 적용됨.")
        print(f"            open-loop MPC + 작은 Ixx + trot 한계로 인한 정상적 발산. "
              f"USE_MPC_CLOSED_LOOP=True + USE_WBIC_FB=True 권장.")

_wbc_tau_cmd_no_grf = wbc_tau_cmd - wbc_tau_grf   # GRF feedforward 제외 (실 액추에이터 부담)
for leg in [0, 3]:
    nj = N_JOINTS_PER_LEG[leg]
    peaks_cmd = "  ".join(f"th{j+1}:{np.max(np.abs(wbc_tau_cmd[:, leg, j])):6.2f}"
                      for j in range(nj))
    peaks_dq  = "  ".join(f"th{j+1}:{np.max(np.abs(joint_vel_hist[:, leg, j])):6.2f}"
                      for j in range(nj))
    fx_peak = np.max(np.abs(wbc_lam_des[:, leg, 0]))
    fy_peak = np.max(np.abs(wbc_lam_des[:, leg, 1]))
    fz_peak = np.max(np.abs(wbc_lam_des[:, leg, 2]))
    print(f"  {LEG_NAMES[leg]} τ_cmd peak [N·m]: {peaks_cmd}")
    print(f"  {LEG_NAMES[leg]} dq    peak [rad/s]: {peaks_dq}")
    print(f"  {LEG_NAMES[leg]} λ (GRF) peak [N]:   Fx={fx_peak:6.2f}, Fy={fy_peak:6.2f}, Fz={fz_peak:6.2f}")
print("─" * 55)

# 비교 모드: metrics를 pickle로 덤프하고 figure/animation 건너뜀
if _COMPARE_MODE:
    _metrics = {
        'variant': _HIND_VARIANT,
        'gait': GAIT_TYPE, 'V': V, 'T': T, 'D': D, 'DT': DT, 'N_FRAMES': N_FRAMES,
        'TOTAL_MASS': TOTAL_MASS, 'G_ACC': G_ACC,
        'Q_HOME_HIND_DEG': Q_HOME_HIND_DEG, 'Q_HOME_FRONT_DEG': Q_HOME_FRONT_DEG,
        'joint_hist': joint_hist, 'joint_vel_hist': joint_vel_hist,
        'joint_acc_hist': joint_acc_hist, 'foot_hist': foot_hist,
        'swing_flag': swing_flag, 'phase_hist': phase_hist,
        'wbc_tau_cmd': wbc_tau_cmd, 'wbc_tau_grf': wbc_tau_grf,
        'wbc_tau_ff': wbc_tau_ff, 'wbc_tau_dyn': wbc_tau_dyn,
        'wbc_tau_pd': wbc_tau_pd, 'wbc_tau_imp': wbc_tau_imp,
        'wbc_lam_des': wbc_lam_des, 'wbc_lam_calc': wbc_lam_calc,
        'wbic_lam_used': wbic_lam_used,
        'wbic_dtau_hist': wbic_dtau_hist, 'wbic_dlam_hist': wbic_dlam_hist,
        'wbic_residual_hist': wbic_residual_hist, 'wbic_status_hist': wbic_status_hist,
        'opt_ik_nit_hist': opt_ik_nit_hist, 'opt_ik_fallback_hist': opt_ik_fallback_hist,
        'fz_sum_des': fz_sum_des, 'fz_sum_used': fz_sum_used,
    }
    _out_path = f'/tmp/v10_metrics_{_HIND_VARIANT}.pkl'
    with open(_out_path, 'wb') as _f:
        pickle.dump(_metrics, _f)
    print(f"[COMPARE_MODE] metrics dumped → {_out_path}")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════
# 4. Figure 1: 3D 애니메이션 + Gait 분석
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 11))
fig.patch.set_facecolor('#1a1a2e')
gs = gridspec.GridSpec(4, 3, figure=fig, wspace=0.38, hspace=0.72,
                       left=0.03, right=0.98, top=0.93, bottom=0.06)
_dark = '#16213e'
_gray = 'gray'

def _style_ax(ax, title, xlabel='Frame', ylabel=''):
    ax.set_facecolor(_dark)
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors=_gray)
    ax.grid(True, alpha=0.25, color=_gray)
    for sp in ax.spines.values():
        sp.set_edgecolor(_gray)

ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax3d.set_facecolor(_dark)
reach = 0.65
ax3d.set_xlim(-reach, reach); ax3d.set_ylim(-0.5, 0.5); ax3d.set_zlim(-0.65, 0.15)
ax3d.set_xlabel('X (m)', color='white', labelpad=4)
ax3d.set_ylabel('Y (m)', color='white', labelpad=4)
ax3d.set_zlabel('Z (m)', color='white', labelpad=4)
ax3d.tick_params(colors=_gray)
ax3d.set_title(
    f'Gait Sim v12  [{GAIT_TYPE.upper()}]  v={V}m/s  T={T}s  D={D}  '
    f'step_h={STEP_HEIGHT}m  step_l={STEP_LENGTH:.3f}m  total_mass={TOTAL_MASS:.2f}kg',
    color='white', fontsize=9)
ax3d.view_init(elev=20, azim=-55)
ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False

# Body chassis 박스 + hip 마커 — animate에서 매 프레임 갱신 (VIZ_BODY_MODE='world'/'body_follow')
_BC_BODY = np.array([
    LEG_HIP_OFFSETS[0], LEG_HIP_OFFSETS[2],
    LEG_HIP_OFFSETS[3], LEG_HIP_OFFSETS[1],
    LEG_HIP_OFFSETS[0],
])
body_chassis_line, = ax3d.plot([], [], [], '-', color='white', lw=2.5, alpha=0.7)
hip_markers = [ax3d.plot([], [], [], 'o', color=LEG_COLORS[leg],
                          markersize=7, alpha=0.8)[0] for leg in range(4)]
# CoM 마커 (USE_BODY_DYNAMICS=True에서 body_pos 시각화)
body_com_marker, = ax3d.plot([], [], [], 'X', color='yellow',
                              markersize=10, alpha=0.9, markeredgecolor='black')
# hip 라벨은 정적 — body 따라 움직이지 않음 (matplotlib 3D text 동적 갱신 비용 큼)
for leg in range(4):
    h = LEG_HIP_OFFSETS[leg]
    ax3d.text(h[0], h[1], h[2]+0.02, LEG_NAMES[leg], color=LEG_COLORS[leg], fontsize=7)

gnd_z = home_foot[2]
xx, yy = np.meshgrid([-reach, reach], [-0.5, 0.5])
ax3d.plot_surface(xx, yy, np.full_like(xx, gnd_z), alpha=0.12, color='#888888')

_AX_COLORS = ['#ff4444', '#44ff44', '#4444ff']

def _body_T_at(fi):
    """프레임 fi의 시각화 변환 (translation, R) — VIZ_BODY_MODE 기반.
    'static' or USE_BODY_DYNAMICS=False : I (변환 없음)
    'world'                              : (body_pos, body_R)  ← drift 그대로
    'body_follow'                        : (0, body_R)         ← 회전만, 카메라가 body 따라감
    """
    if (not USE_BODY_DYNAMICS) or VIZ_BODY_MODE == 'static':
        return np.zeros(3), np.eye(3)
    if VIZ_BODY_MODE == 'body_follow':
        return np.zeros(3), body_R_hist[fi]
    return body_pos_hist[fi], body_R_hist[fi]   # 'world' 기본

def _body_T_apply(p_body, fi):
    """body-frame 점 → 시각화 좌표."""
    pos, R = _body_T_at(fi)
    return pos + R @ p_body

# VIZ_BODY_MODE='world'에서 body가 +X 방향으로 이동하니 ax 범위 확장
if USE_BODY_DYNAMICS and VIZ_BODY_MODE == 'world':
    _x_max = float(np.max(body_pos_hist[:, 0]))
    _x_min = float(np.min(body_pos_hist[:, 0]))
    _z_min_b = float(np.min(body_pos_hist[:, 2]))
    _z_max_b = float(np.max(body_pos_hist[:, 2]))
    ax3d.set_xlim(min(-reach, _x_min - 0.3), max(reach, _x_max + 0.3))
    ax3d.set_zlim(min(-0.65, _z_min_b - 0.3), max(0.15, _z_max_b + 0.3))

_BASE_FRAME_LEN = 0.12
for ax_i, lbl in enumerate(['X (fwd)', 'Y (lat)', 'Z (up)']):
    dv = np.zeros(3); dv[ax_i] = _BASE_FRAME_LEN
    ax3d.quiver(0, 0, 0, dv[0], dv[1], dv[2],
                color=_AX_COLORS[ax_i], linewidth=2.5, arrow_length_ratio=0.25)
    ax3d.text(dv[0]*1.15, dv[1]*1.15, dv[2]*1.15,
              lbl, color=_AX_COLORS[ax_i], fontsize=8, fontweight='bold')
ax3d.plot([0], [0], [0], 'w+', markersize=12, markeredgewidth=2.5, zorder=10)

leg_links = []
for leg in range(4):
    nj = N_JOINTS_PER_LEG[leg]
    lns = [ax3d.plot([], [], [], '-o', color=LEG_COLORS[leg],
                     lw=2.5, markersize=5)[0] for _ in range(nj)]
    leg_links.append(lns)

TRACE_LEN  = int(T / DT)
leg_traces = [ax3d.plot([], [], [], '-', color=LEG_COLORS[leg],
                        lw=1.2, alpha=0.6)[0] for leg in range(4)]
trace_buf  = [[[], [], []] for _ in range(4)]
swing_dots = [ax3d.plot([], [], [], 'o', color=LEG_COLORS[leg],
                        markersize=9, alpha=0.9)[0] for leg in range(4)]

# 링크별 질량중심 마커 (joint origin 중점, 마커 크기는 √m 비례)
link_com_markers = []
for leg in range(4):
    nj = N_JOINTS_PER_LEG[leg]
    lm = LINK_MASS_PER_LEG[leg]
    mks = [ax3d.plot([], [], [], '*', color='#ffd700',
                     markersize=4.0 + 3.5*math.sqrt(float(lm[k])),
                     markeredgecolor='black', markeredgewidth=0.5,
                     alpha=0.9, zorder=15)[0] for k in range(nj)]
    link_com_markers.append(mks)

FRAME_LEN   = 0.035
_jf_quivers = [
    [[None, None, None] for _ in range(N_JOINTS_PER_LEG[leg] + 1)]
    for leg in range(4)
]
info_text = ax3d.text2D(0.02, 0.98, "", transform=ax3d.transAxes,
                         color='white', fontfamily='monospace', fontsize=7.5, va='top')

_fr = np.arange(N_FRAMES)
axis_colors = ['#ff6b6b', '#ffd166', '#06d6a0', '#4cc9f0', '#f72585']

ax_phase = fig.add_subplot(gs[0, 1:])
_style_ax(ax_phase, f'Gait Phase  [{GAIT_TYPE}]  (Bright=Swing)', ylabel='Leg')
ax_phase.set_xlim(0, N_FRAMES); ax_phase.set_ylim(-0.5, 3.5)
ax_phase.set_yticks([0, 1, 2, 3])
ax_phase.set_yticklabels(LEG_NAMES[::-1], color='white')
for leg in range(4):
    row = 3 - leg
    in_sw = False; sw_start = 0
    for fi in range(N_FRAMES):
        if swing_flag[fi, leg] and not in_sw:
            sw_start = fi; in_sw = True
        elif not swing_flag[fi, leg] and in_sw:
            ax_phase.barh(row, fi-sw_start, left=sw_start, height=0.7,
                          color=LEG_COLORS[leg], alpha=0.85)
            in_sw = False
    if in_sw:
        ax_phase.barh(row, N_FRAMES-sw_start, left=sw_start, height=0.7,
                      color=LEG_COLORS[leg], alpha=0.85)
phase_cursor = ax_phase.axvline(x=0, color='white', lw=1.5, ls='--')

ax_z = fig.add_subplot(gs[1, 1])
_style_ax(ax_z, 'Step Height  Z [m]', ylabel='Z [m]')
ax_z.set_xlim(0, N_FRAMES)
for leg in range(4):
    ax_z.plot(_fr, foot_local[:, leg, 2], lw=1.6, color=LEG_COLORS[leg], label=LEG_NAMES[leg])
ax_z.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=_gray, ncol=4)
z_cursor = ax_z.axvline(x=0, color='white', lw=1.5, ls='--')

ax_x = fig.add_subplot(gs[1, 2])
_style_ax(ax_x, 'Step Length  X [m]', ylabel='X [m]')
ax_x.set_xlim(0, N_FRAMES)
for leg in range(4):
    ax_x.plot(_fr, foot_local[:, leg, 0], lw=1.6, color=LEG_COLORS[leg], label=LEG_NAMES[leg])
ax_x.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=_gray, ncol=4)
x_cursor = ax_x.axvline(x=0, color='white', lw=1.5, ls='--')

ax_zv = fig.add_subplot(gs[2, 1])
_style_ax(ax_zv, 'Step Height Velocity  dZ/dt [m/s]', ylabel='[m/s]')
ax_zv.set_xlim(0, N_FRAMES)
for leg in range(4):
    ax_zv.plot(_fr, foot_vel_t[:, leg, 2], lw=1.6, color=LEG_COLORS[leg],
               ls='--' if leg >= 2 else '-', label=LEG_NAMES[leg])
ax_zv.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=_gray, ncol=4)
zv_cursor = ax_zv.axvline(x=0, color='white', lw=1.5, ls='--')

ax_xv = fig.add_subplot(gs[2, 2])
_style_ax(ax_xv, 'Step Length Velocity  dX/dt [m/s]', ylabel='[m/s]')
ax_xv.set_xlim(0, N_FRAMES)
for leg in range(4):
    ax_xv.plot(_fr, foot_vel_t[:, leg, 0], lw=1.6, color=LEG_COLORS[leg],
               ls='--' if leg >= 2 else '-', label=LEG_NAMES[leg])
ax_xv.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=_gray, ncol=4)
xv_cursor = ax_xv.axvline(x=0, color='white', lw=1.5, ls='--')

ax_za = fig.add_subplot(gs[3, 1])
_style_ax(ax_za, 'Step Height Acceleration  d²Z/dt² [m/s²]', ylabel='[m/s²]')
ax_za.set_xlim(0, N_FRAMES)
for leg in range(4):
    ax_za.plot(_fr, foot_acc_t[:, leg, 2], lw=1.6, color=LEG_COLORS[leg],
               ls='--' if leg >= 2 else '-', label=LEG_NAMES[leg])
ax_za.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=_gray, ncol=4)
za_cursor = ax_za.axvline(x=0, color='white', lw=1.5, ls='--')

ax_xa = fig.add_subplot(gs[3, 2])
_style_ax(ax_xa, 'Step Length Acceleration  d²X/dt² [m/s²]', ylabel='[m/s²]')
ax_xa.set_xlim(0, N_FRAMES)
for leg in range(4):
    ax_xa.plot(_fr, foot_acc_t[:, leg, 0], lw=1.6, color=LEG_COLORS[leg],
               ls='--' if leg >= 2 else '-', label=LEG_NAMES[leg])
ax_xa.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=_gray, ncol=4)
xa_cursor = ax_xa.axvline(x=0, color='white', lw=1.5, ls='--')

all_cursors = [phase_cursor, z_cursor, x_cursor,
               zv_cursor, xv_cursor, za_cursor, xa_cursor]

# ══════════════════════════════════════════════════════════════
# 5. 애니메이션
# ══════════════════════════════════════════════════════════════
def init_anim():
    for leg in range(4):
        for ln in leg_links[leg]:
            ln.set_data([], []); ln.set_3d_properties([])
        leg_traces[leg].set_data([], []); leg_traces[leg].set_3d_properties([])
        swing_dots[leg].set_data([], []); swing_dots[leg].set_3d_properties([])
        for mk in link_com_markers[leg]:
            mk.set_data([], []); mk.set_3d_properties([])
    info_text.set_text('')
    return []


def animate(fi):
    t = fi * DT
    body_pos_v, body_R_v = _body_T_at(fi)

    # Body chassis 박스 갱신 (4개 hip을 잇는 사각형)
    bc_world = np.array([body_pos_v + body_R_v @ p for p in _BC_BODY])
    body_chassis_line.set_data(bc_world[:, 0], bc_world[:, 1])
    body_chassis_line.set_3d_properties(bc_world[:, 2])
    # CoM 마커 (body_pos)
    body_com_marker.set_data([body_pos_v[0]], [body_pos_v[1]])
    body_com_marker.set_3d_properties([body_pos_v[2]])

    for leg in range(4):
        nj     = N_JOINTS_PER_LEG[leg]
        q      = joint_hist[fi, leg, :nj]
        pts_dh = forward_kinematics(q, dh=LEG_DH[leg])
        pts    = [_dh_to_sim(p, front_leg=(leg < 2)) for p in pts_dh]
        hip_b  = LEG_HIP_OFFSETS[leg]   # body-frame
        hip_w  = body_pos_v + body_R_v @ hip_b
        # hip 마커 갱신
        hip_markers[leg].set_data([hip_w[0]], [hip_w[1]])
        hip_markers[leg].set_3d_properties([hip_w[2]])
        for k in range(nj):
            A_b = hip_b + pts[k];   B_b = hip_b + pts[k+1]
            A   = body_pos_v + body_R_v @ A_b
            B   = body_pos_v + body_R_v @ B_b
            leg_links[leg][k].set_data([A[0], B[0]], [A[1], B[1]])
            leg_links[leg][k].set_3d_properties([A[2], B[2]])
            mid = 0.5 * (A + B)
            link_com_markers[leg][k].set_data([mid[0]], [mid[1]])
            link_com_markers[leg][k].set_3d_properties([mid[2]])
        pe_b = foot_hist[fi, leg]   # body-frame
        pe   = body_pos_v + body_R_v @ pe_b
        if swing_flag[fi, leg]:
            swing_dots[leg].set_data([pe[0]], [pe[1]])
            swing_dots[leg].set_3d_properties([pe[2]])
        else:
            swing_dots[leg].set_data([], []); swing_dots[leg].set_3d_properties([])
        trace_buf[leg][0].append(pe[0])
        trace_buf[leg][1].append(pe[1])
        trace_buf[leg][2].append(pe[2])
        leg_traces[leg].set_data(trace_buf[leg][0][-TRACE_LEN:], trace_buf[leg][1][-TRACE_LEN:])
        leg_traces[leg].set_3d_properties(trace_buf[leg][2][-TRACE_LEN:])
        T_dh = np.eye(4)
        for j in range(nj + 1):
            orig_sim_b = _dh_to_sim(T_dh[:3, 3], front_leg=(leg < 2))
            pos_b      = hip_b + orig_sim_b
            pos        = body_pos_v + body_R_v @ pos_b
            for ax_i in range(3):
                dv_b = _dh_to_sim(T_dh[:3, ax_i], front_leg=(leg < 2))
                dv   = body_R_v @ dv_b
                if _jf_quivers[leg][j][ax_i] is not None:
                    _jf_quivers[leg][j][ax_i].remove()
                _jf_quivers[leg][j][ax_i] = ax3d.quiver(
                    pos[0], pos[1], pos[2],
                    dv[0]*FRAME_LEN, dv[1]*FRAME_LEN, dv[2]*FRAME_LEN,
                    color=_AX_COLORS[ax_i], linewidth=1.0, arrow_length_ratio=0.3)
            if j < nj:
                T_dh = T_dh @ _dh_matrix(
                    LEG_DH[leg][j][0], LEG_DH[leg][j][1],
                    LEG_DH[leg][j][2], float(q[j]))
    for cur in all_cursors:
        cur.set_xdata([fi, fi])
    sw_str = "  ".join(
        f"{LEG_NAMES[l]}:{'SW' if swing_flag[fi, l] else 'ST'}" for l in range(4))
    deg   = np.degrees(joint_hist[fi])
    jnt_lines = []
    tau_lines = []
    grf_lines = []
    for leg in range(4):
        d  = deg[leg]
        tc = wbc_tau_cmd[fi, leg]   # 실제 motor τ_cmd (NMPC 출력 또는 v11 WBIC 출력)
        lm = wbc_lam_des[fi, leg]
        jnt_lines.append(f"{LEG_NAMES[leg]} "
                         f"th1={d[0]:+5.1f}d th2={d[1]:+6.1f}d th3={d[2]:+6.1f}d "
                         f"th4={d[3]:+5.1f}d th5={d[4]:+5.1f}d")
        tau_lines.append(f"{LEG_NAMES[leg]} "
                         f"tau_cmd=[{tc[0]:+5.1f} {tc[1]:+5.1f} {tc[2]:+5.1f} {tc[3]:+5.1f} {tc[4]:+5.1f}]Nm")
        grf_lines.append(f"{LEG_NAMES[leg]} "
                         f"lam=[{lm[0]:+5.1f} {lm[1]:+5.1f} {lm[2]:+5.1f}]N")
    info_text.set_text(
        f"t={t:.3f}s\n{sw_str}\n\n"
        + "\n".join(jnt_lines)
        + "\n\n"
        + "\n".join(tau_lines)
        + "\n\n"
        + "\n".join(grf_lines)
    )
    return []


ani = FuncAnimation(fig, animate, frames=N_FRAMES,
                    init_func=init_anim, interval=DT*1000, blit=False, repeat=True)

# ══════════════════════════════════════════════════════════════
# 6. Figure 2: FR / HR 조인트 분석 (4×2)
# ══════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(12, 13))
fig2.patch.set_facecolor('#1a1a2e')
gs2 = gridspec.GridSpec(5, 2, figure=fig2, wspace=0.35, hspace=0.55,
                        left=0.07, right=0.97, top=0.94, bottom=0.04)

def _style_ax2(ax, title, xlabel='Frame', ylabel=''):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=10)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors='gray')
    ax.grid(True, alpha=0.25, color='gray')
    for sp in ax.spines.values():
        sp.set_edgecolor('gray')

def _leg_subplots(gs_pos, title, data, ylabel):
    ax = fig2.add_subplot(gs_pos)
    _style_ax2(ax, title, ylabel=ylabel)
    ax.set_xlim(0, N_FRAMES)
    nj = data.shape[1]
    for j in range(nj):
        ax.plot(_fr, data[:, j], lw=1.6, color=axis_colors[j % len(axis_colors)], label=f'th{j+1}')
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray', ncol=5)
    return ax.axvline(x=0, color='white', lw=1.5, ls='--')

_leg_subplots(gs2[0, 0], 'FR Joint Pos [deg]',
              np.degrees(joint_hist[:, 0, :5]), '[deg]')
_leg_subplots(gs2[0, 1], 'HL Joint Pos [deg]',
              np.degrees(joint_hist[:, 3, :5]), '[deg]')
_leg_subplots(gs2[1, 0], 'FR Joint Angular Velocity [rad/s]', joint_vel_FR[:, :5], '[rad/s]')
_leg_subplots(gs2[1, 1], 'HL Joint Angular Velocity [rad/s]', joint_vel_HL[:, :5], '[rad/s]')
_leg_subplots(gs2[2, 0], 'FR Joint Angular Acceleration [rad/s²]', joint_acc_FR[:, :5], '[rad/s²]')
_leg_subplots(gs2[2, 1], 'HL Joint Angular Acceleration [rad/s²]', joint_acc_HL[:, :5], '[rad/s²]')
_leg_subplots(gs2[3, 0], 'FR Joint Jerk [rad/s³]', joint_jrk_FR[:, :5], '[rad/s³]')
_leg_subplots(gs2[3, 1], 'HL Joint Jerk [rad/s³]', joint_jrk_HL[:, :5], '[rad/s³]')

# Phase 4 진단: IK 수렴 반복 횟수 + 위치 오차 (4다리 overlay)
_ax_nit = fig2.add_subplot(gs2[4, 0])
_style_ax2(_ax_nit, 'Opt-IK Iterations  (★=fallback frame)', ylabel='nit')
_ax_nit.set_xlim(0, N_FRAMES)
_ax_nit.axhline(OPT_IK_MAXITER, color='red', lw=0.8, ls='--', alpha=0.6,
                label=f'maxiter={OPT_IK_MAXITER}')
for _c, _name, _color in [(0, 'FR', LEG_COLORS[0]), (1, 'FL', LEG_COLORS[1]),
                          (2, 'HR', LEG_COLORS[2]), (3, 'HL', LEG_COLORS[3])]:
    _ax_nit.plot(_fr, opt_ik_nit_hist[:, _c], lw=1.2, color=_color, alpha=0.85, label=_name)
    _fb_idx = np.where(opt_ik_fallback_hist[:, _c])[0]
    if len(_fb_idx) > 0:
        _ax_nit.plot(_fb_idx, opt_ik_nit_hist[_fb_idx, _c], '*',
                     color=_color, markersize=8, markeredgecolor='red',
                     markeredgewidth=0.7, alpha=0.95)
_ax_nit.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
               edgecolor='gray', ncol=5)

_ax_perr = fig2.add_subplot(gs2[4, 1])
_style_ax2(_ax_perr, 'Opt-IK Position Error  (NaN=fallback)', ylabel='[mm]')
_ax_perr.set_xlim(0, N_FRAMES)
_ax_perr.set_yscale('log')
for _c, _name, _color in [(0, 'FR', LEG_COLORS[0]), (1, 'FL', LEG_COLORS[1]),
                          (2, 'HR', LEG_COLORS[2]), (3, 'HL', LEG_COLORS[3])]:
    _perr_mm = np.sqrt(opt_ik_pos_err_hist[:, _c]) * 1e3   # nan stays nan
    _ax_perr.plot(_fr, _perr_mm, lw=1.2, color=_color, alpha=0.85, label=_name)
_ax_perr.axhline(0.1, color='red', lw=0.8, ls='--', alpha=0.6, label='0.1mm')
_ax_perr.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                edgecolor='gray', ncol=5)

fig2.suptitle(
    f'FR / HL Joint Analysis  |  {GAIT_TYPE.upper()}  |  '
    f'v={V}m/s  T={T}s  D={D}  step_h={STEP_HEIGHT*1e3:.0f}mm  step_l={STEP_LENGTH*1e3:.0f}mm',
    color='white', fontsize=11)

# ══════════════════════════════════════════════════════════════
# 7. Figure 3: WBC 분석 (3×2) — FR/HL
#    row0: tau_cmd   row1: GRF lam_z   row2: GRF lam_x/lam_y + 마찰 추
# ══════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(12, 10))
fig3.patch.set_facecolor('#1a1a2e')
gs3 = gridspec.GridSpec(3, 2, figure=fig3, wspace=0.38, hspace=0.60,
                        left=0.07, right=0.97, top=0.93, bottom=0.06)

def _style_ax3(ax, title, xlabel='Frame', ylabel=''):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors='gray')
    ax.grid(True, alpha=0.25, color='gray')
    for sp in ax.spines.values():
        sp.set_edgecolor('gray')

_ax5col = ['#ff6b6b', '#ffd166', '#06d6a0', '#4cc9f0', '#f72585']

for col, leg in enumerate([0, 3]):   # FR=0, HL=3
    nj = N_JOINTS_PER_LEG[leg]

    # row 0: tau_cmd − tau_grf  (GRF feedforward 성분 제외 — 실제 액추에이터 부담만 표시)
    ax_tc = fig3.add_subplot(gs3[0, col])
    _style_ax3(ax_tc, f'{LEG_NAMES[leg]} tau_cmd − tau_grf [N·m]', ylabel='[N·m]')
    ax_tc.set_xlim(0, N_FRAMES)
    _tau_disp = wbc_tau_cmd[:, leg, :] - wbc_tau_grf[:, leg, :]
    for j in range(nj):
        ax_tc.plot(_fr, _tau_disp[:, j], lw=1.4, color=_ax5col[j], label=f'th{j+1}')
    ax_tc.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
    ax_tc.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray', ncol=5)

    # row 1: GRF lam_z (lam_des vs lam_calc)
    ax_fz = fig3.add_subplot(gs3[1, col])
    _style_ax3(ax_fz, f'{LEG_NAMES[leg]} GRF lam_z [N]', ylabel='[N]')
    ax_fz.set_xlim(0, N_FRAMES)
    ax_fz.plot(_fr, wbc_lam_des [:, leg, 2], lw=1.8, color='#00d4ff', label='lam_z des')
    ax_fz.plot(_fr, wbc_lam_calc[:, leg, 2], lw=1.4, color='magenta', ls='--', label='lam_z calc')
    ax_fz.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
    ax_fz.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray')

    # row 2: GRF lam_x/lam_y + 마찰 추 한계 (mu * lam_z_des)
    ax_fxy = fig3.add_subplot(gs3[2, col])
    _style_ax3(ax_fxy, f'{LEG_NAMES[leg]} GRF lam_x/lam_y + Friction Cone [N]', ylabel='[N]')
    ax_fxy.set_xlim(0, N_FRAMES)
    fric_limit = MU_FRICTION * np.abs(wbc_lam_des[:, leg, 2])
    ax_fxy.plot(_fr, wbc_lam_des [:, leg, 0], lw=1.4, color='#ff6b6b', label='lam_x des')
    ax_fxy.plot(_fr, wbc_lam_des [:, leg, 1], lw=1.4, color='#ffd166', label='lam_y des')
    ax_fxy.plot(_fr, wbc_lam_calc[:, leg, 0], lw=1.2, color='#ff6b6b', ls='--', label='lam_x calc')
    ax_fxy.plot(_fr, wbc_lam_calc[:, leg, 1], lw=1.2, color='#ffd166', ls='--', label='lam_y calc')
    ax_fxy.fill_between(_fr,  fric_limit, -fric_limit,
                        color='white', alpha=0.07, label=f'mu*lam_z (mu={MU_FRICTION})')
    ax_fxy.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
    ax_fxy.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray', ncol=3)

fig3.suptitle(
    f'WBC Analysis  FR/HL  |  {GAIT_TYPE.upper()}  |  v={V}m/s  T={T}s  D={D}  '
    f'{"MPC QP (N=" + str(N_MPC) + ")" if USE_MPC else "QP GRF"}  '
    f'mu={MU_FRICTION}  total_mass={TOTAL_MASS:.2f}kg',
    color='white', fontsize=10)

# ══════════════════════════════════════════════════════════════
# 8. Figure 4: tau decompose th1~th4 (4×2) — FR/HL
# ══════════════════════════════════════════════════════════════
fig4 = plt.figure(figsize=(12, 13))
fig4.patch.set_facecolor('#1a1a2e')
gs4 = gridspec.GridSpec(4, 2, figure=fig4, wspace=0.38, hspace=0.58,
                        left=0.07, right=0.97, top=0.93, bottom=0.05)

def _style_ax4(ax, title, xlabel='Frame', ylabel=''):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors='gray')
    ax.grid(True, alpha=0.25, color='gray')
    for sp in ax.spines.values():
        sp.set_edgecolor('gray')

for col, leg in enumerate([0, 3]):   # FR=0, HL=3
    for row, ji in enumerate([0, 1, 2, 3]):   # th1, th2, th3, th4
        ax_td = fig4.add_subplot(gs4[row, col])
        _style_ax4(ax_td, f'{LEG_NAMES[leg]} tau decompose th{ji+1} [N·m]', ylabel='[N·m]')
        ax_td.set_xlim(0, N_FRAMES)
        ax_td.plot(_fr, wbc_tau_dyn [:, leg, ji], lw=1.4, color='#00d4ff',           label='tau_dyn')
        ax_td.plot(_fr, wbc_tau_pd  [:, leg, ji], lw=1.4, color='#ff6b35',           label='tau_pd')
        ax_td.plot(_fr, wbc_tau_imp [:, leg, ji], lw=1.4, color='#00ff99',           label='tau_imp')
        ax_td.plot(_fr, wbc_tau_grf [:, leg, ji], lw=1.4, color='#ffd166',           label='tau_grf')
        ax_td.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
        ax_td.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray', ncol=2)

fig4.suptitle(
    f'FR / HL tau decompose th1~th4  |  {GAIT_TYPE.upper()}  |  '
    f'v={V}m/s  T={T}s  D={D}  '
    f'{"MPC QP (N=" + str(N_MPC) + ")" if USE_MPC else "QP GRF"}',
    color='white', fontsize=10)

# ══════════════════════════════════════════════════════════════
# 9. Figure 5: body state vs cmd (reference) tracking
#    pos x/y/z, vel x/y/z, orientation (roll/pitch/yaw), angular vel
# ══════════════════════════════════════════════════════════════
fig5 = plt.figure(figsize=(12, 13))
fig5.patch.set_facecolor('#1a1a2e')
gs5 = gridspec.GridSpec(4, 2, figure=fig5, wspace=0.30, hspace=0.60,
                        left=0.07, right=0.97, top=0.93, bottom=0.05)

def _style_ax5(ax, title, xlabel='Frame', ylabel=''):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors='gray')
    ax.grid(True, alpha=0.25, color='gray')
    for sp in ax.spines.values():
        sp.set_edgecolor('gray')

_roll_deg  = np.degrees(np.arctan2(body_R_hist[:, 2, 1], body_R_hist[:, 2, 2]))
_pitch_deg = np.degrees(np.arcsin(np.clip(-body_R_hist[:, 2, 0], -1, 1)))
_yaw_deg   = np.degrees(np.arctan2(body_R_hist[:, 1, 0], body_R_hist[:, 0, 0]))

_axis_names = ['x', 'y', 'z']
for ri in range(3):
    err_p = body_pos_hist[:, ri] - body_pos_ref_hist[:, ri]
    rms_p = float(np.sqrt(np.mean(err_p**2)))
    max_p = float(np.max(np.abs(err_p)))
    ax_p = fig5.add_subplot(gs5[ri, 0])
    _style_ax5(ax_p, f'body pos {_axis_names[ri]} [m]   '
                     f'(err: rms={rms_p*1e3:.1f}mm, max={max_p*1e3:.1f}mm)', ylabel='[m]')
    ax_p.set_xlim(0, N_FRAMES)
    ax_p.plot(_fr, body_pos_ref_hist[:, ri], lw=1.4, color='#ff6b6b', ls='--', label='cmd')
    ax_p.plot(_fr, body_pos_hist[:, ri],     lw=1.6, color='#00d4ff',          label='actual')
    ax_p.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray')

    err_v = body_v_hist[:, ri] - body_v_ref_hist[:, ri]
    rms_v = float(np.sqrt(np.mean(err_v**2)))
    max_v = float(np.max(np.abs(err_v)))
    ax_v = fig5.add_subplot(gs5[ri, 1])
    _style_ax5(ax_v, f'body vel {_axis_names[ri]} [m/s]   '
                     f'(err: rms={rms_v:.3f}, max={max_v:.3f})', ylabel='[m/s]')
    ax_v.set_xlim(0, N_FRAMES)
    ax_v.plot(_fr, body_v_ref_hist[:, ri], lw=1.4, color='#ff6b6b', ls='--', label='cmd')
    ax_v.plot(_fr, body_v_hist[:, ri],     lw=1.6, color='#00d4ff',          label='actual')
    ax_v.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray')

ax_or = fig5.add_subplot(gs5[3, 0])
_or_max = max(np.max(np.abs(_roll_deg)), np.max(np.abs(_pitch_deg)), np.max(np.abs(_yaw_deg)))
_style_ax5(ax_or, f'body orientation [deg]  cmd=0  '
                  f'(|err|max: roll={np.max(np.abs(_roll_deg)):.2f}°, '
                  f'pitch={np.max(np.abs(_pitch_deg)):.2f}°, yaw={np.max(np.abs(_yaw_deg)):.2f}°)',
           ylabel='[deg]')
ax_or.set_xlim(0, N_FRAMES)
ax_or.plot(_fr, _roll_deg,  lw=1.4, color='#ff6b6b', label='roll')
ax_or.plot(_fr, _pitch_deg, lw=1.4, color='#ffd166', label='pitch')
ax_or.plot(_fr, _yaw_deg,   lw=1.4, color='#06d6a0', label='yaw')
ax_or.axhline(0, color='white', lw=0.6, ls='--', alpha=0.4)
ax_or.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray', ncol=3)

ax_om = fig5.add_subplot(gs5[3, 1])
_om_max = float(np.max(np.abs(body_omega_hist)))
_style_ax5(ax_om, f'body angular vel [rad/s]  cmd=0  (|err|max={_om_max:.2f})',
           ylabel='[rad/s]')
ax_om.set_xlim(0, N_FRAMES)
ax_om.plot(_fr, body_omega_hist[:, 0], lw=1.4, color='#ff6b6b', label='omega_x')
ax_om.plot(_fr, body_omega_hist[:, 1], lw=1.4, color='#ffd166', label='omega_y')
ax_om.plot(_fr, body_omega_hist[:, 2], lw=1.4, color='#06d6a0', label='omega_z')
ax_om.axhline(0, color='white', lw=0.6, ls='--', alpha=0.4)
ax_om.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray', ncol=3)

fig5.suptitle(
    f'Body State vs Cmd  |  {GAIT_TYPE.upper()}  |  '
    f'v={V}m/s  T={T}s  D={D}  '
    f'{"NMPC (FDDP)" if _USE_NMPC_ACTIVE else "MPC + WBIC"}',
    color='white', fontsize=10)

# ══════════════════════════════════════════════════════════════
# 10. Figure 6: foot trajectory cmd vs actual (world frame)
#    3 rows (x, y, z) × 4 cols (FR, FL, HR, HL)
#    cmd shown only during swing (NaN during stance), actual continuous
# ══════════════════════════════════════════════════════════════
fig6 = plt.figure(figsize=(15, 9))
fig6.patch.set_facecolor('#1a1a2e')
gs6 = gridspec.GridSpec(3, 4, figure=fig6, wspace=0.32, hspace=0.55,
                        left=0.05, right=0.98, top=0.92, bottom=0.06)

def _style_ax6(ax, title, xlabel='Frame', ylabel=''):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=8)
    ax.set_xlabel(xlabel, color='white', fontsize=7)
    ax.set_ylabel(ylabel, color='white', fontsize=7)
    ax.tick_params(colors='gray', labelsize=7)
    ax.grid(True, alpha=0.25, color='gray')
    for sp in ax.spines.values():
        sp.set_edgecolor('gray')

_axis_lbl = ['x [m]', 'y [m]', 'z [m]']
_leg_color = ['#ff6b6b', '#ffd166', '#06d6a0', '#4cc9f0']
for li in range(4):
    for ai in range(3):
        ax_f = fig6.add_subplot(gs6[ai, li])
        _style_ax6(ax_f, f'{LEG_NAMES[li]}  foot {_axis_lbl[ai].split()[0]}',
                   ylabel=_axis_lbl[ai])
        ax_f.set_xlim(0, N_FRAMES)
        actual = foot_actual_world_hist[:, li, ai]
        target = foot_target_world_hist[:, li, ai]
        ax_f.plot(_fr, target, lw=1.6, color='#ff6b6b', ls='--', label='cmd (swing)')
        ax_f.plot(_fr, actual, lw=1.4, color='#00d4ff',          label='actual')
        # swing-end error (excluding NaN)
        err = actual - target
        valid = ~np.isnan(target)
        if valid.sum() > 0:
            err_max = float(np.max(np.abs(err[valid])))
            err_rms = float(np.sqrt(np.mean(err[valid]**2)))
            ax_f.set_title(
                f'{LEG_NAMES[li]} {_axis_lbl[ai].split()[0]}  '
                f'(swing err: rms={err_rms*1e3:.1f}mm max={err_max*1e3:.1f}mm)',
                color='white', fontsize=8)
        if ai == 0 and li == 0:
            ax_f.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                        edgecolor='gray', loc='upper left')

fig6.suptitle(
    f'Foot Trajectory Cmd vs Actual (world frame)  |  {GAIT_TYPE.upper()}  |  '
    f'v={V}m/s  T={T}s  step_h={STEP_HEIGHT*1e3:.0f}mm  step_l={STEP_LENGTH*1e3:.0f}mm  '
    f'{"NMPC" if _USE_NMPC_ACTIVE else "MPC + WBIC"}',
    color='white', fontsize=10)

# ══════════════════════════════════════════════════════════════
# 11. Figure 7: Gait diagram (Hildebrand-style stance chart)
#    4 horizontal bars (legs) × time. 색칠 = stance, 비움 = swing.
#    Fz 색상 진하기로 contact force 강도 시각화.
# ══════════════════════════════════════════════════════════════
fig7 = plt.figure(figsize=(13, 5))
fig7.patch.set_facecolor('#1a1a2e')
gs7 = gridspec.GridSpec(2, 1, figure=fig7, hspace=0.42,
                        left=0.07, right=0.98, top=0.86, bottom=0.10,
                        height_ratios=[3, 2])

# Stance/swing 행렬 (fi×4)
_stance_mat = np.zeros((N_FRAMES, 4), dtype=bool)
for fi in range(N_FRAMES):
    t = fi * DT
    in_cycle_t = t % T
    in_phase_A = in_cycle_t < (T / 2)
    if in_phase_A:
        _stance_mat[fi, 1] = True   # FL stance
        _stance_mat[fi, 2] = True   # HR stance
    else:
        _stance_mat[fi, 0] = True   # FR stance
        _stance_mat[fi, 3] = True   # HL stance

# Top: gait diagram with Fz heatmap
ax_g = fig7.add_subplot(gs7[0, 0])
ax_g.set_facecolor('#16213e')
ax_g.set_title(f'Gait Diagram  |  {GAIT_TYPE.upper()}  |  T={T}s  D={D}  '
               f'(filled = stance, color depth = Fz [N])',
               color='white', fontsize=9)
ax_g.set_xlim(0, N_FRAMES)
ax_g.set_ylim(-0.5, 3.5)
ax_g.set_yticks([0, 1, 2, 3])
ax_g.set_yticklabels(['FR', 'FL', 'HR', 'HL'], color='white')
ax_g.set_xlabel('Frame', color='white', fontsize=8)
ax_g.tick_params(colors='gray')
ax_g.grid(True, alpha=0.2, axis='x', color='gray')
for sp in ax_g.spines.values():
    sp.set_edgecolor('gray')

_fz_max = max(1.0, float(np.max(wbc_lam_des[:, :, 2])))
for li in range(4):
    fz = wbc_lam_des[:, li, 2]
    # 인접 stance 구간 묶어서 그리기
    in_stance = False
    seg_start = 0
    for fi in range(N_FRAMES + 1):
        is_stance = _stance_mat[fi, li] if fi < N_FRAMES else False
        if is_stance and not in_stance:
            seg_start = fi
            in_stance = True
        elif not is_stance and in_stance:
            seg_fz_avg = float(np.mean(fz[seg_start:fi]))
            alpha = 0.25 + 0.7 * min(1.0, seg_fz_avg / _fz_max)
            ax_g.fill_between([seg_start, fi], li - 0.35, li + 0.35,
                              color='#06d6a0', alpha=alpha, edgecolor='none')
            in_stance = False

# Bottom: contact force Fz over time per leg (작은 line plot)
ax_fz_all = fig7.add_subplot(gs7[1, 0])
ax_fz_all.set_facecolor('#16213e')
ax_fz_all.set_title('Contact Force Fz [N] per leg',
                    color='white', fontsize=9)
ax_fz_all.set_xlim(0, N_FRAMES)
ax_fz_all.set_xlabel('Frame', color='white', fontsize=8)
ax_fz_all.set_ylabel('Fz [N]', color='white', fontsize=8)
ax_fz_all.tick_params(colors='gray')
ax_fz_all.grid(True, alpha=0.25, color='gray')
for sp in ax_fz_all.spines.values():
    sp.set_edgecolor('gray')
for li, (lname, lc) in enumerate(zip(LEG_NAMES, _leg_color)):
    ax_fz_all.plot(_fr, wbc_lam_des[:, li, 2], lw=1.2, color=lc, label=lname)
ax_fz_all.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                 edgecolor='gray', ncol=4)

fig7.suptitle(
    f'Gait Diagram  |  {GAIT_TYPE.upper()}  |  '
    f'{"NMPC (FDDP)" if _USE_NMPC_ACTIVE else "MPC + WBIC"}',
    color='white', fontsize=10)

# ══════════════════════════════════════════════════════════════
# 12. Figure 8: Diagnostic — friction cone 사용률 + CoT + slip + τ margin
# ══════════════════════════════════════════════════════════════
fig8 = plt.figure(figsize=(13, 9))
fig8.patch.set_facecolor('#1a1a2e')
gs8 = gridspec.GridSpec(2, 2, figure=fig8, wspace=0.32, hspace=0.50,
                        left=0.07, right=0.97, top=0.92, bottom=0.07)

def _style_ax8(ax, title, xlabel='Frame', ylabel=''):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors='gray')
    ax.grid(True, alpha=0.25, color='gray')
    for sp in ax.spines.values():
        sp.set_edgecolor('gray')

# (1) Friction cone usage ratio: |F_xy| / (μ·F_z) per leg per frame (stance only)
ax_fc = fig8.add_subplot(gs8[0, 0])
fc_ratio = np.zeros((N_FRAMES, 4))
for li in range(4):
    fxy = np.linalg.norm(wbc_lam_des[:, li, :2], axis=1)
    fz  = wbc_lam_des[:, li, 2]
    safe = fz > 1.0
    fc_ratio[safe, li] = fxy[safe] / (MU_FRICTION * fz[safe])
    fc_ratio[~safe, li] = np.nan
fc_max_per_leg = [float(np.nanmax(fc_ratio[:, li])) for li in range(4)]
fc_p95_per_leg = [float(np.nanpercentile(fc_ratio[:, li], 95)) for li in range(4)]
_style_ax8(ax_fc,
    f'Friction Cone Usage  |F_xy|/(μ·F_z)  μ={MU_FRICTION}  '
    f'(>1 = slip)\n'
    f'peak: FR={fc_max_per_leg[0]:.2f} FL={fc_max_per_leg[1]:.2f} '
    f'HR={fc_max_per_leg[2]:.2f} HL={fc_max_per_leg[3]:.2f}',
    ylabel='ratio')
ax_fc.set_xlim(0, N_FRAMES)
for li, (lname, lc) in enumerate(zip(LEG_NAMES, _leg_color)):
    ax_fc.plot(_fr, fc_ratio[:, li], lw=1.2, color=lc, label=lname)
ax_fc.axhline(1.0, color='red', lw=1.0, ls='--', alpha=0.6, label='slip threshold')
ax_fc.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
             edgecolor='gray', ncol=5)

# (2) Mechanical power + Cost-of-Transport
ax_pw = fig8.add_subplot(gs8[0, 1])
# dq from numerical gradient of joint_hist, then power = Σ |τ × dq|
dq_num = np.gradient(joint_hist, axis=0) / DT   # (N_FRAMES, 4, N_JOINTS_MAX)
power_per_leg = np.sum(np.abs(wbc_tau_cmd * dq_num), axis=2)   # (N_FRAMES, 4)
power_total   = np.sum(power_per_leg, axis=1)                  # (N_FRAMES,)
v_actual = np.linalg.norm(body_v_hist, axis=1)                 # speed
cot_inst = power_total / np.maximum(TOTAL_MASS * G_ACC * np.maximum(v_actual, 0.05), 1e-3)
power_avg = float(np.mean(power_total))
cot_avg   = float(np.mean(cot_inst))
_style_ax8(ax_pw,
    f'Mechanical Power [W]  +  CoT (right axis)\n'
    f'P_avg={power_avg:.1f}W  P_peak={float(np.max(power_total)):.1f}W  '
    f'CoT_avg={cot_avg:.2f}  (lower = more efficient)',
    ylabel='Power [W]')
ax_pw.set_xlim(0, N_FRAMES)
ax_pw.plot(_fr, power_total, lw=1.4, color='#06d6a0', label='Σ |τ·dq|')
ax_pw.axhline(power_avg, color='#06d6a0', lw=0.8, ls='--', alpha=0.5)
ax_pw_r = ax_pw.twinx()
ax_pw_r.plot(_fr, cot_inst, lw=1.0, color='#ffd166', alpha=0.7, label='CoT')
ax_pw_r.set_ylabel('CoT (-)', color='#ffd166', fontsize=8)
ax_pw_r.tick_params(axis='y', colors='#ffd166')
ax_pw.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
             edgecolor='gray', loc='upper left')
ax_pw_r.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
               edgecolor='gray', loc='upper right')

# (3) Stance foot slip velocity (foot world-frame ‖v‖, stance only)
ax_sl = fig8.add_subplot(gs8[1, 0])
v_foot_world = np.gradient(foot_actual_world_hist, axis=0) / DT   # (N_FRAMES, 4, 3)
slip_speed = np.linalg.norm(v_foot_world, axis=2)                  # (N_FRAMES, 4)
slip_stance = np.where(_stance_mat, slip_speed, np.nan)
slip_max_per_leg = [float(np.nanmax(slip_stance[:, li])) for li in range(4)]
_style_ax8(ax_sl,
    f'Stance Foot Slip Velocity ‖v_foot‖ [m/s]  (ideal: 0)\n'
    f'peak: FR={slip_max_per_leg[0]:.2f} FL={slip_max_per_leg[1]:.2f} '
    f'HR={slip_max_per_leg[2]:.2f} HL={slip_max_per_leg[3]:.2f}',
    ylabel='|v| [m/s]')
ax_sl.set_xlim(0, N_FRAMES)
for li, (lname, lc) in enumerate(zip(LEG_NAMES, _leg_color)):
    ax_sl.plot(_fr, slip_stance[:, li], lw=1.2, color=lc, label=lname)
ax_sl.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
             edgecolor='gray', ncol=4)

# (4) Torque saturation margin: |τ| / τ_limit (per joint per leg, max across joints)
ax_tm = fig8.add_subplot(gs8[1, 1])
# 가장 위험한 joint = 가장 큰 |τ|/limit ratio per frame
tau_ratio = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
for j in range(N_JOINTS_MAX):
    if j < len(JOINT_TORQUE_LIMIT) and JOINT_TORQUE_LIMIT[j] > 0:
        tau_ratio[:, :, j] = np.abs(wbc_tau_cmd[:, :, j]) / JOINT_TORQUE_LIMIT[j]
tau_ratio_worst = np.max(tau_ratio, axis=2)   # (N_FRAMES, 4)
tau_max_per_leg = [float(np.max(tau_ratio_worst[:, li])) for li in range(4)]
_style_ax8(ax_tm,
    f'Torque Saturation: max_j |τ_j|/τ_limit_j per leg  (>1 = saturate)\n'
    f'peak: FR={tau_max_per_leg[0]:.2f} FL={tau_max_per_leg[1]:.2f} '
    f'HR={tau_max_per_leg[2]:.2f} HL={tau_max_per_leg[3]:.2f}',
    ylabel='|τ|/τ_max')
ax_tm.set_xlim(0, N_FRAMES)
for li, (lname, lc) in enumerate(zip(LEG_NAMES, _leg_color)):
    ax_tm.plot(_fr, tau_ratio_worst[:, li], lw=1.2, color=lc, label=lname)
ax_tm.axhline(1.0, color='red', lw=1.0, ls='--', alpha=0.6, label='saturation')
ax_tm.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
             edgecolor='gray', ncol=5)

fig8.suptitle(
    f'Diagnostic  |  {GAIT_TYPE.upper()}  |  '
    f'v={V}m/s  T={T}s  μ={MU_FRICTION}  '
    f'{"NMPC (FDDP)" if _USE_NMPC_ACTIVE else "MPC + WBIC"}',
    color='white', fontsize=10)

plt.figure(fig.number)
plt.suptitle(
    f'Gait Sim v12  |  {GAIT_TYPE.upper()}  |  '
    f'v={V}m/s  T={T}s  D={D}  T_sw={T_SW:.2f}s  '
    f'step_h={STEP_HEIGHT*1e3:.0f}mm  step_l={STEP_LENGTH*1e3:.0f}mm',
    color='white', fontsize=9)
plt.show()
