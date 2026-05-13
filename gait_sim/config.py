"""gait_sim.config — 전체 시뮬레이션 파라미터 (GaitConfig dataclass + alias).

v13.0 Phase 1: gait_sim_v12.py 의 GaitConfig 블록 (line 100~250) 을 모듈로 추출.
다른 모듈들은 이 파일에서 import 해서 사용:

    from gait_sim.config import CFG, GAIT_TYPE, V, T, D, STEP_HEIGHT, ...

CFG 필드를 *runtime 변경* 하려면 새 CFG 객체 생성 후 alias 도 재할당 필요.
일반 사용은 dataclass 의 default 값을 직접 편집.
"""
import math
from dataclasses import dataclass, field

import numpy as np


# ══════════════════════════════════════════════════════════════
# GaitConfig — 모든 user-tunable 파라미터 통합 dataclass
# ══════════════════════════════════════════════════════════════
@dataclass
class GaitConfig:
    """gait_sim NMPC + WBIC + MPC + IK 통합 설정. 섹션별로 묶음."""
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
    body_inertia: np.ndarray = field(default_factory=lambda: np.diag([0.07, 0.26, 0.26]))
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
    wbic_w_dtau: float = 0.0      # τ smoothness: w_dtau·‖τ−τ_prev‖²  (v13.1: net negative → OFF default)
    use_spline_diff: bool = False # q̇/q̈ 미분: True=CubicSpline, False=np.gradient (v13.1: net negative → OFF default)
    qp_solver: str = 'quadprog'   # v13.0a: 'quadprog' / 'osqp' / 'proxqp' — MPC + WBIC QP solver 선택
    # ─────────── Actuator model (v13.x Phase 7 Tier 1) ───────────
    use_actuator_model: bool = False           # OFF default — opt-in
    actuator_dq_idle_ratio: float = 0.3        # T-N curve: peak τ 유지 비율 of dq_max
    actuator_stiction_ratio: float = 0.02      # stiction = τ_limit × ratio (2% default)
    actuator_viscous_b: np.ndarray = field(
        default_factory=lambda: np.zeros(5))   # viscous damping per joint [N·m·s/rad]
    actuator_eps_v: float = 0.01               # static/kinetic 전환 속도 [rad/s]
    # ─────────── NMPC: solver ───────────
    use_nmpc: bool = False
    use_nmpc_receding: bool = True
    nmpc_maxiter: int = 200
    nmpc_init_reg: float = 1.0
    nmpc_rh_n_horizon: int = 24   # 0.5s @ DT_MPC=0.02
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


# ══════════════════════════════════════════════════════════════
# Gait 프리셋 (V / T / D / STEP_HEIGHT / offsets 통합)
# ══════════════════════════════════════════════════════════════
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
PHASE_OFFSETS = {name: cfg['offsets'] for name, cfg in GAIT_PRESETS.items()}


# ══════════════════════════════════════════════════════════════
# CFG instantiation + module-level alias 들 (다른 모듈 호환)
# ══════════════════════════════════════════════════════════════
CFG = GaitConfig()

GAIT_TYPE = CFG.gait_type
DT        = CFG.dt
N_CYCLES  = CFG.n_cycles
TAU_LAND  = CFG.tau_land

# gait preset 에서 V/T/D/STEP_HEIGHT 추출
_preset     = GAIT_PRESETS[GAIT_TYPE]
V           = _preset['V']            # m/s (전진 속도)
T           = _preset['T']            # s (사이클 주기)
D           = _preset['D']            # swing 비율 (T_SW/T)
STEP_HEIGHT = _preset['STEP_HEIGHT']  # m (발 들리는 높이)

T_SW = T * D
T_ST = T * (1.0 - D)
STRIDE_D_MIN = 2.0 * V * T_SW
STRIDE_D     = V * T + 2.0 * V * T_SW
assert STRIDE_D >= STRIDE_D_MIN, f"STRIDE_D({STRIDE_D:.3f}m) < MIN({STRIDE_D_MIN:.3f}m)"
STEP_LENGTH  = STRIDE_D / 2.0 - V * T_SW
STANCE_DELTA = CFG.stance_delta

# Robot geometry alias (자주 사용됨)
BODY_FWD_F = CFG.body_fwd_f
BODY_FWD_H = CFG.body_fwd_h
BODY_LAT   = CFG.body_lat
BODY_Z_H   = CFG.body_z_h

# Frame count (derived)
N_FRAMES = int(N_CYCLES * T / DT)

# Physical constants
G_ACC = CFG.g_acc
