"""
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
import numpy as np
import qpsolvers

# v11 Phase 7: Pinocchio (선택적 — USE_PINOCCHIO=True일 때 사용)
try:
    import pin_helpers as _pin_helpers
    _PIN_AVAILABLE = True
except ImportError:
    _PIN_AVAILABLE = False

# 비교 모드: HIND_VARIANT={'orig','ext'}, COMPARE_MODE=1이면 figure 생략 후 metrics 덤프
_HIND_VARIANT = os.environ.get('HIND_VARIANT', 'ext')
_COMPARE_MODE = os.environ.get('COMPARE_MODE', '0') == '1'
assert _HIND_VARIANT in ('orig', 'ext'), f"HIND_VARIANT={_HIND_VARIANT} (expected 'orig' or 'ext')"
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize as _sp_minimize

for key in mpl.rcParams:
    if key.startswith("keymap."):
        mpl.rcParams[key] = []
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'NanumGothic', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════
# 0. 파라미터
# ══════════════════════════════════════════════════════════════
GAIT_TYPE   = 'trot'   # 'walk', 'amble', 'pace', 'trot', 'canter', 'gallop'
DT          = 0.002 # s (시뮬레이터 타임스텝, WBC 제어 주기) 0.002s 이상이어야 함 (QP GRF fallback 고려)
N_CYCLES    = 4 # 사이클 수 (1사이클 = 1주기 = T초 동안의 발 움직임 패턴)

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
TAU_LAND    = 1.0 # (swing phase 내 착지까지의 비율, 0~1) 1.0이면 선형 보간

T_SW = T * D
T_ST = T * (1.0 - D)

STRIDE_D_MIN = 2.0 * V * T_SW
STRIDE_D     = V * T + 2.0 * V * T_SW
assert STRIDE_D >= STRIDE_D_MIN, f"STRIDE_D({STRIDE_D:.3f}m) < MIN({STRIDE_D_MIN:.3f}m)"

STEP_LENGTH = STRIDE_D / 2.0 - V * T_SW
STANCE_DELTA = 0.005

BODY_FWD_F =  0.250
BODY_FWD_H = -0.250
BODY_LAT   =  0.050
BODY_Z_H   = -0.050  # 앞다리보다 아래쪽으로 뒷다리 힙의 수직(Z) 오프셋 [m]

# ── DH 파라미터 ──────────────────────────────────────────────
DH_FRONT = [
    (+math.pi/2, 0.0,   0.0,   ),
    (0.0,        0.21,  0.0075,),
    (0.0,        0.235, 0.0,   ),
    (0.0,        0.1,   0.0,   ),
    (0.0,        0.045, 0.0,   ),
]
DH_HIND = [
    (-math.pi/2, 0.0,   0.0,   ),
    (0.0,        0.21,  0.0075,),
    (0.0,        0.21,  0.0,   ),
    (0.0,        0.148, 0.0,   ),
    (0.0,        0.045, 0.0,   ),
]

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
LEG_DH           = [DH_FRONT, DH_FRONT, DH_HIND, DH_HIND]
N_JOINTS_PER_LEG = [5, 5, 5, 5]
N_JOINTS_MAX     = 5

# DH의 D2 오프셋(=0.0075)으로 인한 좌우 비대칭 보정:
# foot_local_y = -0.0075 (모든 다리 동일) → hip_y에 +0.0075 적용 시
# foot_world_y = ±BODY_LAT 정확히 대칭 (CoM 기준 roll moment ≈ 0)
_HIP_Y_BIAS = 0.0075   # ≡ DH dh[1][2] (= D2_F = D2_H)
LEG_HIP_OFFSETS = np.array([
    [+BODY_FWD_F, -BODY_LAT + _HIP_Y_BIAS, 0.0     ],
    [+BODY_FWD_F, +BODY_LAT + _HIP_Y_BIAS, 0.0     ],
    [+BODY_FWD_H, -BODY_LAT + _HIP_Y_BIAS, BODY_Z_H],
    [+BODY_FWD_H, +BODY_LAT + _HIP_Y_BIAS, BODY_Z_H],
])

PHASE_OFFSETS = {name: cfg['offsets'] for name, cfg in GAIT_PRESETS.items()}

# ── WBC 파라미터 ─────────────────────────────────────────────
BODY_MASS = 15.0 # kg (몸무게)
G_ACC     = 9.81

#LINK_MASS         = np.array([3.34, 0.8, 0.2, 0.2, 0.05])  # link1~5 질량 [kg]
#LINK_MASS         = np.array([4.125, 1.795, 0.78, 0.78, 0.05])  # link1~5 질량 [kg] 
LINK_MASS         = np.array([3, 2, 1, 0.2, 0.1])  # link1~5 질량 [kg] 
# 80형번 0.915kg, 90형번 1.605kg
LINK_MASS_PER_LEG = [LINK_MASS] * 4
TOTAL_MASS        = BODY_MASS + float(np.sum(LINK_MASS)) * 4.0
LINK_RADIUS       = 0.015   # [m] 링크 단면 반경 근사 (RNEA 원통 관성 텐서용)

KP_PD = np.array([30.0, 80.0, 80.0, 60.0, 20.0])
KD_PD = np.array([ 3.0,  8.0,  8.0,  6.0,  2.0])
KP_IMP = np.array([400.0, 400.0, 400.0])
KD_IMP = np.array([ 20.0,  20.0,  20.0])

MU_DAMP      = 1e-3
TAU_LAG      = 0.03
INIT_ERR_RAD = math.radians(1.0)

# ── MPC / QP GRF 파라미터 ────────────────────────────────────
MU_FRICTION  = 0.6       # 마찰 계수

# 본체 관성 텐서 (직육면체 근사, 0.5×0.1×0.1m, 15kg) [kg·m²]
BODY_INERTIA = np.diag([0.07, 0.26, 0.26])   # Ixx, Iyy, Izz

N_MPC  = 10              # MPC 예측 구간 [스텝]
DT_MPC = DT * 10         # MPC 샘플링 주기 [s]  (= 0.02s)

# MPC 상태 가중치: x=[roll,pitch,yaw, px,py,pz, ωx,ωy,ωz, vx,vy,vz, g]
_Q_DIAG = np.array([
    200, 200, 100,   # roll, pitch, yaw
      0,   0, 200,   # px, py, pz (높이 추종)
      0,   0,   0,   # ωx, ωy, ωz
     10,   0,   0,   # vx (전진속도 추종), vy, vz
      0,           # g (상수, 추종 불필요)
], dtype=float)
MPC_Q = np.diag(_Q_DIAG)
MPC_R = 1e-6 * np.eye(3)   # GRF 가중치 (per foot, 3×3)

USE_MPC = True   # False → QP GRF (단일 스텝 fallback) 사용

# Contact GRF ramp: stance 시작/끝 N% 구간에서 λ_des를 smoothstep으로 ramp.
# swing↔stance 경계의 cmd torque step jump 완화 (논문 임피던스 흡수 효과).
GRF_RAMP_RATIO = 0.10

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

def analytical_ik_front(Px, Py, Pz, phi, theta5_target):
    D2 = Px**2 + Py**2 - _D2_F**2
    if D2 < 0:
        return None
    R = math.sqrt(D2)
    theta1 = math.atan2(_D2_F, -R) - math.atan2(-Py, Px)
    c1, s1 = math.cos(theta1), math.sin(theta1)
    x_s = c1 * Px + s1 * Py
    x3 = x_s - _A4_F * math.cos(phi) - _A5_F * math.cos(theta5_target)
    z3 = Pz   - _A4_F * math.sin(phi) - _A5_F * math.sin(theta5_target)
    cos_th3 = (x3**2 + z3**2 - _A2_F**2 - _A3_F**2) / (2.0 * _A2_F * _A3_F)
    cos_th3 = max(-1.0, min(1.0, cos_th3))
    theta3  = math.acos(cos_th3)
    theta2 = (math.atan2(z3, x3)
              - math.atan2(_A3_F * math.sin(theta3), _A2_F + _A3_F * math.cos(theta3)))
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


def opt_ik_front(p_target_dh, q_init, q_ref=None):
    """SLSQP 최적화 IK — 앞다리.

    등식 제약 : FK_tip(q) = p_target          (위치 정확도 보장)
                q[4] = Q_HOME_FRONT[4]        (toe 고정 — IK에서 제외)
    부등식 제약: |τ_grav(q)| ≤ τ_limit        (OPT_IK_USE_TAU_LIMIT, 중력 토크 근사)
    bounds     : FRONT_Q_LIM ∩ 각속도 한계    (OPT_IK_USE_VEL_LIMIT, 정확)
    비용       : LAMBDA_Q_OPT·||q - q_ref||² + LAMBDA_TAU_OPT·||τ_grav(q)||²
                 - q_ref가 주어지면 그 자세 추종 (swing1/swing2 quintic blend)
                 - q_ref=None이면 q_init 사용 (smoothness only)
    """
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
                    'fun': lambda q: np.array(forward_kinematics(q, DH_FRONT)[-1]) - p_t}]

    # ── 등식 제약: q5(toe) 고정 — IK에서 제외 (analytical과 동일하게 home 값 유지) ─
    # th5는 가벼운 toe link이라 IK 자유도로 활용하지 않음.
    # q5 변동이 dq 폭증 원인이 됐으므로 고정하여 4 DoF redundancy(q1~q4)만 사용.
    constraints.append({'type': 'eq',
                        'fun': lambda q: q[4] - Q_HOME_FRONT[4]})

    # ── 부등식 제약: 중력 토크 한계 (τ_full 근사, 속도·GRF 항 미포함) ─────
    _lm_front = LINK_MASS_PER_LEG[0]
    if OPT_IK_USE_TAU_LIMIT:
        def _torque_ineq(q):
            tau_g = compute_gravity_torque_sim(q, DH_FRONT, _lm_front, front_leg=True)
            return JOINT_TORQUE_LIMIT[:len(tau_g)] - np.abs(tau_g)  # ≥ 0 이어야 통과
        constraints.append({'type': 'ineq', 'fun': _torque_ineq})

    def cost(q):
        # 참조 자세 추종 (q_ref=None이면 q_init = warm-start smoothness)
        c_qref = LAMBDA_Q_OPT * np.dot(q - q_tgt, q - q_tgt)
        # τ_grav minimize: redundancy를 토크 작은 자세로 자동 사용
        tau_g  = compute_gravity_torque_sim(q, DH_FRONT, _lm_front, front_leg=True)
        c_tau  = LAMBDA_TAU_OPT * np.dot(tau_g, tau_g)
        return float(c_qref + c_tau)

    res = _sp_minimize(cost, q0, method='SLSQP', bounds=active_bounds,
                       constraints=constraints,
                       options={'ftol': 1e-8, 'maxiter': OPT_IK_MAXITER})
    tip_final = np.array(forward_kinematics(res.x, DH_FRONT)[-1])
    pos_err_sq = float(np.dot(tip_final - p_t, tip_final - p_t))
    if pos_err_sq < 1e-6:
        return list(res.x), res.nit, pos_err_sq
    return None, res.nit, pos_err_sq


def opt_ik_hind(p_target_dh, q_init, q_ref=None):
    """SLSQP 최적화 IK — 뒷다리. opt_ik_front와 구조 동일.

    등식 제약 : FK_tip(q) = p_target          (위치 정확도 보장)
                q[4] = Q_HOME_HIND[4]         (toe 고정)
    부등식 제약: |τ_grav(q)| ≤ τ_limit        (OPT_IK_USE_TAU_LIMIT)
    bounds     : HIND_Q_LIM ∩ 각속도 한계      (OPT_IK_USE_VEL_LIMIT)
    비용       : LAMBDA_Q_OPT·||q - q_ref||² + LAMBDA_TAU_OPT·||τ_grav(q)||²
    """
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
                    'fun': lambda q: np.array(forward_kinematics(q, DH_HIND)[-1]) - p_t}]
    constraints.append({'type': 'eq',
                        'fun': lambda q: q[4] - Q_HOME_HIND[4]})

    _lm_hind = LINK_MASS_PER_LEG[2]
    if OPT_IK_USE_TAU_LIMIT:
        def _torque_ineq(q):
            tau_g = compute_gravity_torque_sim(q, DH_HIND, _lm_hind, front_leg=False)
            return JOINT_TORQUE_LIMIT[:len(tau_g)] - np.abs(tau_g)
        constraints.append({'type': 'ineq', 'fun': _torque_ineq})

    def cost(q):
        c_qref = LAMBDA_Q_OPT * np.dot(q - q_tgt, q - q_tgt)
        tau_g  = compute_gravity_torque_sim(q, DH_HIND, _lm_hind, front_leg=False)
        c_tau  = LAMBDA_TAU_OPT * np.dot(tau_g, tau_g)
        return float(c_qref + c_tau)

    res = _sp_minimize(cost, q0, method='SLSQP', bounds=active_bounds,
                       constraints=constraints,
                       options={'ftol': 1e-8, 'maxiter': OPT_IK_MAXITER})
    tip_final = np.array(forward_kinematics(res.x, DH_HIND)[-1])
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
                w_ddq, w_tau, w_lam, lamz_min, mu):
    """Per-leg WBIC QP correction.

    변수 : x = [Δq̈ (nj); Δτ (nj); Δλ (3)]
    비용 : w_ddq‖Δq̈‖² + w_tau‖Δτ‖² + w_lam‖Δλ‖²
    등식 : M·Δq̈ - Δτ - Jᵀ·Δλ = r,  r = tau_ff + Jᵀ·λ_des - M·ddq_des - h
    부등 : τ_min ≤ tau_ff+Δτ ≤ τ_max
           stance: λ_z+Δλ_z ≥ lamz_min,  |λ_x,y+Δλ_x,y| ≤ μ(λ_z+Δλ_z)
           swing : Δλ = -λ_des  (λ=0 고정)

    Returns (dq̈, dτ, dλ, success, residual_pre)
    """
    n_v = nj + nj + 3
    P = np.diag([w_ddq]*nj + [w_tau]*nj + [w_lam]*3)
    qv = np.zeros(n_v)

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
                 w_ddq, w_tau, w_lam, w_fb, lamz_min, mu):
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
frame_calc_time = np.zeros(N_FRAMES, dtype=float)

JOINT_VEL_LIMIT_RAD_S   = np.array([14.66, 15.91, 15.91, 14.66, 14.66], dtype=float)
JOINT_TORQUE_LIMIT      = np.array([60.0, 120.0, 120.0, 60.0, 60.0])   # [N·m]
VEL_LIMIT_MARGIN  = 999

# ── Phase 4: Optimization-based IK 파라미터 ──────────────────
LAMBDA_Q_OPT   = 1.0    # smoothness 가중치 (||q - q_init||², warm-start 변화량 최소화)
LAMBDA_TAU_OPT = 0.01   # 토크 minimize 가중치 (||τ_grav(q)||², redundancy를 토크 작은 자세로 자동 활용)
OPT_IK_MAXITER = 100

# 제약 ON/OFF — True/False 한 줄로 켜고 끔
OPT_IK_USE_VEL_LIMIT = True   # 각속도 제약: |Δq/DT| ≤ JOINT_VEL_LIMIT_RAD_S (bounds 동적 수축)
OPT_IK_USE_TAU_LIMIT = True   # 토크 제약: |τ_grav(q)| ≤ JOINT_TORQUE_LIMIT  (근사, 속도 영향)

# ── Phase 5: WBIC QP 파라미터 ────────────────────────────────
USE_WBIC      = True     # False면 v8 동작 (RNEA τ_ff 직접 사용 + clip)
WBIC_W_DDQ    = 1.0      # ‖Δq̈‖² 가중치 (가속도 추종)
WBIC_W_TAU    = 0.01     # ‖Δτ‖² 가중치 (τ_ff 변경 최소화)
WBIC_W_LAM    = 0.001    # ‖Δλ‖² 가중치 (λ_des 변경 최소화)
WBIC_LAMZ_MIN = 1.0      # stance 발 최소 법선력 [N]

# ── v11 Phase 7: Floating-base 동역학 + WBIC FB 파라미터 ─────
USE_BODY_DYNAMICS = True  # True: GRF 기반 body 6-DoF 적분 (v11, 진단 목적), False: V·t kinematic
# v11 Phase 7: Pinocchio dynamics 백엔드 (URDF 우회, DH→pinocchio.Model 직접 빌드)
# True : pin_helpers.py의 rnea/compute_mh_leg/compute_jacobian_sim 사용 (검증된 정확)
# False: v11 native 함수 사용 (이전 동작)
USE_PINOCCHIO = False
# 실험적: USE_PINOCCHIO + USE_WBIC_FB일 때 Full M(q) (26×26 floating base 결합) 사용
# 현재 90% solver fail (per-leg τ_ff와 full M 모델 inconsistency).
# False면 pinocchio per-leg M(5×5) 사용 (block-diagonal, 안정).
USE_PINOCCHIO_FULL_M = False
# v11 토글 조합:
#   (CL=False, FB=False) ← 기본. v10 호환 동작 + body state 진단 추적
#                          body는 발산 가능 (open-loop). 시각화는 VIZ='static' 권장
#   (CL=True,  FB=True)  — closed-loop 추적 (pitch<1°, z<6mm). 단 trot의 경우 roll은
#                          여전히 80°+ 까지 발산 (선형 MPC 한계). VIZ='world' 가능
#   (CL=True,  FB=False) — 위험: 선형 MPC가 큰 보정 시도→발산. 사용 권장X
#   (CL=False, FB=True)  — body 평형 강제, MPC는 idealized
USE_WBIC_FB         = True
USE_MPC_CLOSED_LOOP = True
WBIC_W_FB           = 0.1   # ‖Δv̇_fb‖² 가중치 (body acc 보정 최소화)
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
VIZ_BODY_MODE = 'world' 
USE_SWING_QREF_BLEND = True   # True: swing1/swing2 → Q_SWING_FRONT blend / False: home 고정
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
MAX_TRAJ_OPT_ITERS = 6

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
        if front_l:
            _foot_dh0 = _sim_to_dh(_foot_loc0 + _FRONT_J4_TO_J5_SIM, front_leg=True)
            _q_a = analytical_ik_front(_foot_dh0[0], _foot_dh0[1], _foot_dh0[2],
                                       PHI_FRONT, THETA5_FRONT)
            _q_init0 = list(_q_a) if _q_a is not None else list(Q_HOME_FRONT)
            _q_opt, _, _ = opt_ik_front(_foot_dh0, _q_init0, q_ref=list(Q_HOME_FRONT))
            prev_q_per_leg.append(_q_opt if _q_opt is not None else _q_init0)
        else:
            _foot_dh0 = _sim_to_dh(_foot_loc0 + _HIND_J4_TO_J5_SIM, front_leg=False)
            _q_h = analytical_ik_hind(_foot_dh0[0], _foot_dh0[1], _foot_dh0[2],
                                      PHI_HIND, dh=DH_HIND, theta5_target=THETA5_HIND)
            _q_init0 = list(_q_h) + [Q_HOME_HIND[4]] if _q_h is not None else list(Q_HOME_HIND)
            _q_opt, _, _ = opt_ik_hind(_foot_dh0, _q_init0, q_ref=list(Q_HOME_HIND))
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
                                                      q_ref=_q_ref)
                opt_ik_nit_hist[fi, col]     = nit
                opt_ik_pos_err_hist[fi, col] = pos_err_sq
                if q_opt is not None:
                    q = q_opt
                else:
                    # bounds 내 해 없음 → analytical fallback
                    opt_ik_fallback_hist[fi, col] = True
                    q_ana = analytical_ik_front(foot_dh[0], foot_dh[1], foot_dh[2],
                                                PHI_FRONT, THETA5_FRONT)
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
                                                     q_ref=_q_ref_h)
                opt_ik_nit_hist[fi, col]     = nit
                opt_ik_pos_err_hist[fi, col] = pos_err_sq
                if q_opt is not None:
                    q = q_opt
                else:
                    opt_ik_fallback_hist[fi, col] = True
                    q_h = analytical_ik_hind(foot_dh[0], foot_dh[1], foot_dh[2],
                                             PHI_HIND, dh=DH_HIND, theta5_target=THETA5_HIND)
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

joint_vel_FR = joint_vel_hist[:, 0, :]
joint_acc_FR = np.gradient(joint_vel_FR, DT, axis=0)
joint_jrk_FR = np.gradient(joint_acc_FR, DT, axis=0)

joint_vel_HR = joint_vel_hist[:, 2, :]
joint_acc_HR = np.gradient(joint_vel_HR, DT, axis=0)
joint_jrk_HR = np.gradient(joint_acc_HR, DT, axis=0)

joint_vel_HL = joint_vel_hist[:, 3, :]
joint_acc_HL = np.gradient(joint_vel_HL, DT, axis=0)
joint_jrk_HL = np.gradient(joint_acc_HL, DT, axis=0)

# 전 관절 가속도 (RNEA용, 4 legs × N_JOINTS_MAX)
joint_acc_hist = np.gradient(joint_vel_hist, DT, axis=0)

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
body_state = {
    'pos':   np.array([0.0, 0.0, 0.0]),     # initial CoM at origin
    'R':     np.eye(3),
    'v':     np.array([V, 0.0, 0.0]),       # 정상 보행 속도로 시작
    'omega': np.zeros(3),
    'a_lin': np.zeros(3),
    'a_ang': np.zeros(3),
    '_diverged': False,                     # 발산 시 integrate_body_state가 set
}

# WBIC FB 진단
wbic_fb_residual_hist = np.zeros(N_FRAMES)          # body 6-DoF residual norm
wbic_fb_status_hist   = np.zeros(N_FRAMES, dtype=bool)
wbic_fb_dvfb_hist     = np.zeros((N_FRAMES, 6))     # Δv̇_fb

mpc_fail_count = 0
wbic_fail_count = 0
wbic_fb_fail_count = 0

for fi in range(N_FRAMES):
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
    body_pos_ref_hist[fi] = np.array([V * t_cur, 0.0, 0.0])
    body_v_ref_hist[fi]   = np.array([V, 0.0, 0.0])

wbc_dur = time.perf_counter() - wbc_t0
mode_str = f"MPC(N={N_MPC},dt={DT_MPC*1e3:.0f}ms)" if USE_MPC else "QP GRF"
wbic_str = "WBIC ON" if USE_WBIC else "WBIC OFF"
print(f"WBC 완료 [{mode_str}, {wbic_str}].  {wbc_dur*1e3:.1f}ms 총  ({wbc_dur/N_FRAMES*1e6:.1f}μs/frame)")

# GRF 합산 검증 (λ_des = MPC/QP 출력, λ_used = WBIC 보정 후)
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

# v11: Floating-base body 동역학 진단
if USE_BODY_DYNAMICS:
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
    peaks_cmd = "  ".join(f"th{j+1}:{np.max(np.abs(_wbc_tau_cmd_no_grf[:, leg, j])):6.2f}"
                      for j in range(nj))
    peaks_dq  = "  ".join(f"th{j+1}:{np.max(np.abs(joint_vel_hist[:, leg, j])):6.2f}"
                      for j in range(nj))
    fx_peak = np.max(np.abs(wbc_lam_des[:, leg, 0]))
    fy_peak = np.max(np.abs(wbc_lam_des[:, leg, 1]))
    fz_peak = np.max(np.abs(wbc_lam_des[:, leg, 2]))
    print(f"  {LEG_NAMES[leg]} τ_cmd−τ_grf peak [N·m]: {peaks_cmd}")
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
    f'Gait Sim v11  [{GAIT_TYPE.upper()}]  v={V}m/s  T={T}s  D={D}  '
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
        tc = wbc_tau_cmd[fi, leg] - wbc_tau_grf[fi, leg]   # GRF 성분 제외
        lm = wbc_lam_des[fi, leg]
        jnt_lines.append(f"{LEG_NAMES[leg]} "
                         f"th1={d[0]:+5.1f}d th2={d[1]:+6.1f}d th3={d[2]:+6.1f}d "
                         f"th4={d[3]:+5.1f}d th5={d[4]:+5.1f}d")
        tau_lines.append(f"{LEG_NAMES[leg]} "
                         f"tau_cmd−grf=[{tc[0]:+5.1f} {tc[1]:+5.1f} {tc[2]:+5.1f} {tc[3]:+5.1f} {tc[4]:+5.1f}]Nm")
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
        ax_td.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
        ax_td.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray', ncol=2)

fig4.suptitle(
    f'FR / HL tau decompose th1~th4  |  {GAIT_TYPE.upper()}  |  '
    f'v={V}m/s  T={T}s  D={D}  '
    f'{"MPC QP (N=" + str(N_MPC) + ")" if USE_MPC else "QP GRF"}',
    color='white', fontsize=10)

plt.figure(fig.number)
plt.suptitle(
    f'Gait Sim v11  |  {GAIT_TYPE.upper()}  |  '
    f'v={V}m/s  T={T}s  D={D}  T_sw={T_SW:.2f}s  '
    f'step_h={STEP_HEIGHT*1e3:.0f}mm  step_l={STEP_LENGTH*1e3:.0f}mm',
    color='white', fontsize=9)
plt.show()
