# ┌─────────────────────────────────────────────────────────────────────┐
# │  Opt_IK 클리핑 위치 및 주석 처리 대상                                 │
# │                                                                     │
# │  종류      위치                    주석 처리 대상 (비활성화 시)       │
# │  ──────    ──────────────────────  ──────────────────────────────── │
# │  각속도    궤적 루프 (line ~910)    _vel_dt ~ q[:nj]=list(_q_arr) 4줄│
# │  토크      WBC 루프  (line ~1103)  tau_cmd_leg = np.clip(...) 1줄    │
# └─────────────────────────────────────────────────────────────────────┘

"""
gait_sim_v8.py  —  4족 보행 Gait 시뮬레이터 + WBC + MPC QP GRF
v8: v7 대비 변경사항
    · Phase 2 — QP GRF: 단일 스텝 힘 평형 + 마찰 추 QP (fallback)
    · Phase 1 — MPC QP: N스텝 horizon 선형화 부유 베이스 MPC
        - 상태: x=[roll,pitch,yaw, px,py,pz, ω, v, g] (13dim)
        - 제어: u=[λ_FR, λ_FL, λ_HR, λ_HL] (12dim, swing=0)
        - 비용: ||x-x_ref||²_Q + ||u||²_R
        - 구속: 마찰 추 |λ_x|,|λ_y| ≤ μ·λ_z, λ_z≥0
    · Figure 3: 4×2로 확장 — GRF Fx/Fy 마찰 추 시각화 추가
    · Phase 3 — RNEA: M(q)·q̈ + C(q,q̇)·q̇ + g(q) 완전 강체 동역학
        - tau_ff = RNEA(q,q̇,q̈) − Jᵀ·λ_des  (기존 quasi-static g(q) → 완전 동역학)

[MOD] v8.1 수정사항:
    · LINK_MASS 값 수정: [3.34, 0.8, 0.2, 0.2, 0.05] (기존: [0.5, 0.8, 0.2, 0.2, 0.05])
    · GRF 계산에 TOTAL_MASS 적용: body_mass + 4×link_mass 기반 힘 평형
    · tau_GRF 추가: GRF로부터 유도되는 조인트 토크 계산 및 로깅
    · Figure 3 제목 수정: Body_MASS → total_mass 표기
"""

import math
import time
import numpy as np
import qpsolvers
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
mpl.rcParams['font.sans-serif'] = ['NanumGothic', 'DejaVu Sans', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════
# 0. 파라미터
# ══════════════════════════════════════════════════════════════
GAIT_TYPE   = 'trot'
DT          = 0.002 # s (시뮬레이터 타임스텝, WBC 제어 주기) 0.002s 이상이어야 함 (QP GRF fallback 고려)
N_CYCLES    = 4 # 사이클 수 (1사이클 = 1주기 = T초 동안의 발 움직임 패턴)

V           = 1 # m/s (전진 속도)
T           = 0.5 # s (사이클 주기)
D           = 0.5 # (swing 비율, duty factor) 0.5 이상이어야 함 (최소 2발 접지)
STEP_HEIGHT = 0.06 # m (발 들리는 높이, 지면과의 간격)
TAU_LAND    = 1.0 # (swing phase 내 착지까지의 비율, 0~1) 1.0이면 선형 보간, 0.5이면 50% 지점에서 착지 시작

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
BODY_X_H   = -0.100

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

Q_HOME_FRONT_DEG = [0.0, 157.5, 22.5, 30.6583, 59.3417]
Q_HOME_HIND_DEG  = [0.0, -150.0, -90.0, 90.0, 60.0]
Q_HOME_FRONT = [math.radians(a) for a in Q_HOME_FRONT_DEG]
Q_HOME_HIND  = [math.radians(a) for a in Q_HOME_HIND_DEG]

# swing 중 opt_ik 비용함수 참조 자세 (th4 음수 유도, 나머지는 home과 동일)
Q_SWING_FRONT_DEG = [0.0, 157.5, 22.5, -100.0, 59.3417]   # th4: 30.66° → -30°
Q_SWING_FRONT = [math.radians(a) for a in Q_SWING_FRONT_DEG]

PHI_FRONT    = Q_HOME_FRONT[1] + Q_HOME_FRONT[2] + Q_HOME_FRONT[3]
PHI_HIND     = Q_HOME_HIND[1]  + Q_HOME_HIND[2]  + Q_HOME_HIND[3]
THETA5_FRONT = PHI_FRONT + Q_HOME_FRONT[4]
THETA5_HIND  = PHI_HIND  + Q_HOME_HIND[4]

Q_HOME_PER_LEG      = [Q_HOME_FRONT, Q_HOME_FRONT, Q_HOME_HIND, Q_HOME_HIND]
PHI_PER_LEG         = [PHI_FRONT, PHI_FRONT, PHI_HIND, PHI_HIND]
TRAJ_PT_IDX_PER_LEG = [4, 4, 4, 4]

LEG_NAMES        = ['FR', 'FL', 'HR', 'HL']
LEG_COLORS       = ['#00d4ff', '#ff6b35', '#00ff99', '#ffcc00']
LEG_DH           = [DH_FRONT, DH_FRONT, DH_HIND, DH_HIND]
N_JOINTS_PER_LEG = [5, 5, 5, 5]
N_JOINTS_MAX     = 5

LEG_HIP_OFFSETS = np.array([
    [+BODY_FWD_F, -BODY_LAT, 0.0     ],
    [+BODY_FWD_F, +BODY_LAT, 0.0     ],
    [+BODY_FWD_H, -BODY_LAT, BODY_X_H],
    [+BODY_FWD_H, +BODY_LAT, BODY_X_H],
])

PHASE_OFFSETS = {
    'trot': [0.0, 0.5, 0.5, 0.0],
    'walk': [0.0, 0.5, 0.75, 0.25],
}

# ── WBC 파라미터 ─────────────────────────────────────────────
BODY_MASS = 15.0 # kg (몸무게)
G_ACC     = 9.81

#LINK_MASS         = np.array([3.34, 0.8, 0.2, 0.2, 0.05])  # link1~5 질량 [kg]
LINK_MASS         = np.array([4.125, 1.215, 0.2, 0.2, 0.05])  # link1~5 질량 [kg] 
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
    """SLSQP 최적화 IK — 앞다리 swing 전용.

    등식 제약 : FK_tip(q) = p_target          (위치 정확도 보장)
    부등식 제약: |τ_grav(q)| ≤ τ_limit        (OPT_IK_USE_TAU_LIMIT, 중력 토크 근사)
    bounds     : FRONT_Q_LIM ∩ 각속도 한계    (OPT_IK_USE_VEL_LIMIT, 정확)
    비용       : LAMBDA_Q_OPT·||q - q_ref||²  (q_ref=None 시 Q_HOME_FRONT 사용)
    """
    p_t = np.asarray(p_target_dh, dtype=float)
    q0  = np.asarray(q_init,      dtype=float)
    q_h = np.asarray(Q_HOME_FRONT if q_ref is None else q_ref, dtype=float)

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

    # ── 부등식 제약: 중력 토크 한계 (τ_full 근사, 속도·GRF 항 미포함) ─────
    if OPT_IK_USE_TAU_LIMIT:
        _lm_front = LINK_MASS_PER_LEG[0]
        def _torque_ineq(q):
            tau_g = compute_gravity_torque_sim(q, DH_FRONT, _lm_front, front_leg=True)
            return JOINT_TORQUE_LIMIT[:len(tau_g)] - np.abs(tau_g)  # ≥ 0 이어야 통과
        constraints.append({'type': 'ineq', 'fun': _torque_ineq})

    def cost(q):
        return float(LAMBDA_Q_OPT * np.dot(q - q_h, q - q_h))

    res = _sp_minimize(cost, q0, method='SLSQP', bounds=active_bounds,
                       constraints=constraints,
                       options={'ftol': 1e-8, 'maxiter': OPT_IK_MAXITER})
    tip_final = np.array(forward_kinematics(res.x, DH_FRONT)[-1])
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

def _build_Ac_d():
    """시불변 Ac_d (13×13) — contact 무관"""
    Ac = np.zeros((13, 13), dtype=float)
    Ac[0:3, 6:9]  = np.eye(3)    # dΘ/dt = ω
    Ac[3:6, 9:12] = np.eye(3)    # dp/dt = v
    Ac[9:12, 12]  = [0.0, 0.0, 1.0]   # dv/dt += g·ẑ
    return np.eye(13) + DT_MPC * Ac

_Ac_d = _build_Ac_d()

# Ac_d 거듭제곱 사전 계산 (0 ~ N_MPC)
_Ad_powers = [np.eye(13, dtype=float)]
for _k in range(N_MPC):
    _Ad_powers.append(_Ac_d @ _Ad_powers[-1])


def _build_Bc(contact_mask_k, foot_pos_k):
    """접촉 패턴별 Bc_d (13×12)"""
    Bc = np.zeros((13, 12), dtype=float)
    for i in range(4):
        if contact_mask_k[i]:
            r = foot_pos_k[i]
            Bc[6:9,  i*3:(i+1)*3] = _I_inv @ _skew(r)   # angular
            Bc[9:12, i*3:(i+1)*3] = np.eye(3) / TOTAL_MASS  # linear
    return DT_MPC * Bc


def mpc_qp_plan(x0, contact_schedule, foot_positions):
    """
    Phase 1 — Convex MPC QP  (Di Carlo et al., IROS 2018 simplified)

    x0              : (13,) 현재 body 상태
                      [roll,pitch,yaw, px,py,pz, ωx,ωy,ωz, vx,vy,vz, g]
    contact_schedule: (N_MPC, 4) bool  — horizon 내 접촉 패턴
    foot_positions  : (N_MPC, 4, 3)   — horizon 내 발 위치 (world frame)
    Returns         : lam_des (4, 3)  — 첫 번째 스텝 최적 GRF
    """
    nx = 13
    nu = 12   # 4 feet × 3

    # Bc_k 리스트 (horizon별)
    Bc_list = [_build_Bc(contact_schedule[k], foot_positions[k]) for k in range(N_MPC)]

    # 응축 행렬 Aq (N*nx × nx), Bq (N*nx × N*nu)
    N   = N_MPC
    Aq  = np.zeros((N*nx, nx),   dtype=float)
    Bq  = np.zeros((N*nx, N*nu), dtype=float)
    for i in range(N):
        Aq[i*nx:(i+1)*nx, :] = _Ad_powers[i+1]
        for j in range(i+1):
            Bq[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = _Ad_powers[i-j] @ Bc_list[j]

    # 목표 상태: 현재 상태 유지 (hover at x0)
    X_ref = np.tile(x0, N)   # (N*nx,)

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


def swing_foot_pos(sw_t, p_start, p_end, step_height=STEP_HEIGHT, tau_land=TAU_LAND):
    """Swing 궤적: 원래 설계대로 (p_end에서 착지)"""
    if sw_t >= tau_land:
        return p_end.copy()
    tau = sw_t / tau_land
    s   = 10*tau**3 - 15*tau**4 + 6*tau**5
    pos = (1.0 - s) * p_start + s * p_end
    pos = pos.copy()
    # Z: p_start에서 p_end로 상승-하강
    pos[2] = p_start[2] + step_height * (4 * tau * (1 - tau))**3
    return pos


def stance_foot_pos(st_t, p_contact, body_vel, stance_dur):
    """Stance 궤적: 사인 함수 (착지 연속성)"""
    s = st_t**3 * (10.0 - 15.0*st_t + 6.0*st_t**2)
    pos = p_contact - body_vel * stance_dur * s
    pos = pos.copy()
    # Z: 사인 함수 (C2 연속)
    # st_t=0: sin(0)=0      → p_contact[2] (swing과 연속)
    # st_t=0.5: sin(π/2)=1  → p_contact[2] - STANCE_DELTA (최대 침투)
    # st_t=1: sin(π)=0      → p_contact[2] (다음 swing과 연속)
    pos[2] = p_contact[2] - STANCE_DELTA * math.sin(math.pi * st_t)
    return pos


def foot_pos_at_phase(phase, p_start, p_contact, p_end, body_vel,
                      swing_ratio=D, step_height=STEP_HEIGHT, tau_land=TAU_LAND, stance_dur=T_ST):
    """
    단일 함수로 Swing/Stance 통합 (C2 연속성 보장)
    phase: [0, 1] - 전체 사이클 정규화 시간
    """
    pos = np.zeros(3)

    if phase < swing_ratio:
        # ━━━━━━━━━━━━ SWING PHASE ━━━━━━━━━━━━
        sw_t = phase / swing_ratio  # [0, 1]

        if sw_t >= tau_land:
            pos = p_end.copy()
        else:
            tau = sw_t / tau_land
            s = 10*tau**3 - 15*tau**4 + 6*tau**5

            pos[:2] = (1.0 - s) * p_start[:2] + s * p_end[:2]
            pos[2] = p_start[2] + step_height * (4 * tau * (1 - tau))**3
    else:
        # ━━━━━━━━━━━━ STANCE PHASE ━━━━━━━━━━━━
        st_t = (phase - swing_ratio) / (1.0 - swing_ratio)  # [0, 1]

        s = st_t**3 * (10.0 - 15.0*st_t + 6.0*st_t**2)
        pos[:2] = p_contact[:2] - body_vel[:2] * stance_dur * s

        # Z: 사인 함수 (착지 연속성)
        pos[2] = p_contact[2] - STANCE_DELTA * math.sin(math.pi * st_t)

    return pos

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
LAMBDA_Q_OPT   = 0.01   # 관절 home 위치 정규화 가중치
OPT_IK_MAXITER = 100

# 제약 ON/OFF — True/False 한 줄로 켜고 끔
OPT_IK_USE_VEL_LIMIT = True   # 각속도 제약: |Δq/DT| ≤ JOINT_VEL_LIMIT_RAD_S (bounds 동적 수축)
OPT_IK_USE_TAU_LIMIT = True   # 토크 제약: |τ_grav(q)| ≤ JOINT_TORQUE_LIMIT  (근사, 속도 영향)

# 앞다리 관절 위치 한계 [rad]  — home: [0, 157.5, 22.5, 30.66, 59.34] deg
FRONT_Q_LIM = [
    (-math.radians(45),  math.radians(45)),   # th1: 어깨 벌림
    ( math.radians(45),  math.radians(210)),  # th2: 어깨 굴곡 (swing 고각 여유)
    (-math.radians(45),  math.radians(135)),  # th3: 팔꿈치
    (-math.radians(120), math.radians(31)),  # th4: 손목
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
print(f"  V={V}m/s  T={T}s  D={D}  T_SW={T_SW:.3f}s  T_ST={T_ST:.3f}s")
print(f"  STRIDE_D={STRIDE_D*1e3:.1f}mm  STEP_LENGTH={STEP_LENGTH*1e3:.1f}mm")

traj_scale   = 1.0
height_scale = 1.0
opt_iter_used = 0
opt_ik_nit_hist      = np.zeros((N_FRAMES, 2), dtype=int)    # [FR, FL] 수렴 반복 횟수
opt_ik_fallback_hist = np.zeros((N_FRAMES, 2), dtype=bool)   # True = analytical fallback 사용
opt_ik_pos_err_hist  = np.full((N_FRAMES, 2), np.nan)        # opt IK 위치 오차² [m²]

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
    prev_q_per_leg  = [list(Q_HOME_FRONT), list(Q_HOME_FRONT),
                       list(Q_HOME_HIND),  list(Q_HOME_HIND)]

    calc_start = time.perf_counter()
    for fi in range(N_FRAMES):
        frame_start = time.perf_counter()
        t = fi * DT
        for leg in range(4):
            is_sw = sched.is_swing(leg, t)
            phase_hist[fi, leg] = sched.phase(leg, t)
            swing_flag[fi, leg] = is_sw

            if is_sw and not prev_swing[leg]:
                foot_sw_start[leg] = foot_local_prev[leg].copy()
            if not is_sw and prev_swing[leg]:
                foot_contact[leg] = foot_local_prev[leg].copy()

            # 통합 함수 사용: Swing/Stance를 하나의 연속 함수로 계산
            phase = sched.phase(leg, t)
            p_end = home_foot_per_leg[leg] + np.array([STEP_LENGTH * traj_scale, 0, 0])
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

                if is_sw:
                    # Phase 4: swing → optimization IK (warm-start = 이전 프레임 q)
                    # q_ref: swing 중간에서만 Q_SWING_FRONT 쪽으로 유도 (sine 벨커브)
                    _sw_t  = sched.swing_t(leg, t)
                    _alpha = math.sin(math.pi * _sw_t)   # 0→1→0
                    _q_ref = [h + _alpha * (s - h)
                              for h, s in zip(Q_HOME_FRONT, Q_SWING_FRONT)]
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
                else:
                    # stance → analytical IK 유지
                    q = analytical_ik_front(foot_dh[0], foot_dh[1], foot_dh[2],
                                            PHI_FRONT, THETA5_FRONT)
                    if q is None:
                        q = list(Q_HOME_FRONT)

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
    joint_vel_hist = np.zeros_like(joint_hist)
    joint_vel_hist[1:] = (joint_hist_unwrapped[1:] - joint_hist_unwrapped[:-1]) / DT

    peak_per_joint  = np.max(np.abs(joint_vel_hist), axis=(0, 1))
    ratio_per_joint = peak_per_joint / JOINT_VEL_LIMIT_RAD_S
    worst_ratio     = float(np.max(ratio_per_joint))

    if worst_ratio <= VEL_LIMIT_MARGIN:
        break
    scale_decay  = max(0.60, min(0.98 / worst_ratio, 0.98))
    traj_scale  *= scale_decay
    height_scale *= scale_decay

print(f"궤적 완료. iter={opt_iter_used}  scale={traj_scale:.4f}")
_fb_fr = int(np.sum(opt_ik_fallback_hist[:, 0]))
_fb_fl = int(np.sum(opt_ik_fallback_hist[:, 1]))
_sw_fr = int(np.sum(swing_flag[:, 0]))
_sw_fl = int(np.sum(swing_flag[:, 1]))
_q_home_f = np.array(Q_HOME_FRONT)
_front_lim_lo = np.array([b[0] for b in FRONT_Q_LIM])
_front_lim_hi = np.array([b[1] for b in FRONT_Q_LIM])

for _col, _leg, _sw_mask, _fb in [
    (0, 0, swing_flag[:, 0], _fb_fr),
    (1, 1, swing_flag[:, 1], _fb_fl),
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
    _viol   = int(np.sum(np.any((_q_ok < _front_lim_lo) | (_q_ok > _front_lim_hi), axis=1)))

    # home 이탈: opt IK 성공 프레임 평균 ||q - q_home||
    _home_dev = float(np.mean(np.linalg.norm(_q_ok - _q_home_f, axis=1))) if len(_q_ok) else 0.0

    _leg_name = ['FR', 'FL'][_col]
    print(f"  Opt-IK {_leg_name}: nit_avg={_nit.mean():.1f}  fallback={_fb}/{_sw_n}  "
          f"pos_err(max={_perr_mm_max:.3f}mm mean={_perr_mm_mean:.3f}mm)  "
          f"bound_viol={_viol}프레임  home_dev_avg={_home_dev:.4f}rad")

    # ── 경고 ──────────────────────────────────────────────────
    if _fb > 0:
        _fb_rate = _fb / _sw_n * 100
        print(f"  [WARNING] {_leg_name} Opt-IK fallback {_fb_rate:.1f}% — "
              f"V({V}m/s)·STEP_HEIGHT({STEP_HEIGHT}m) 조합이 FRONT_Q_LIM 초과 가능성. "
              f"속도↓ or STEP_HEIGHT↓ or FRONT_Q_LIM 완화 권장.")
    if _perr_mm_max > 0.1:
        print(f"  [WARNING] {_leg_name} 최대 위치 오차 {_perr_mm_max:.3f}mm > 0.1mm — "
              f"SLSQP 수렴 불충분. OPT_IK_MAXITER({OPT_IK_MAXITER}) 증가 권장.")
    if _viol > 0:
        print(f"  [WARNING] {_leg_name} 관절 한계 위반 {_viol}프레임 — "
              f"SLSQP 수치 오차로 bounds 미세 초과. FRONT_Q_LIM 여유 ±0.01rad 추가 권장.")
    if _home_dev > 1.0:
        print(f"  [INFO]    {_leg_name} home 이탈 평균 {_home_dev:.4f}rad > 1.0rad — "
              f"swing 궤적이 home 자세와 크게 다름. T({T}s)↑ or V({V}m/s)↓ 시 완화됨.")

joint_vel_FR = joint_vel_hist[:, 0, :]
joint_acc_FR = np.zeros_like(joint_vel_FR)
joint_acc_FR[1:] = (joint_vel_FR[1:] - joint_vel_FR[:-1]) / DT
joint_jrk_FR = np.zeros_like(joint_acc_FR)
joint_jrk_FR[1:] = (joint_acc_FR[1:] - joint_acc_FR[:-1]) / DT

joint_vel_HR = joint_vel_hist[:, 2, :]
joint_acc_HR = np.zeros_like(joint_vel_HR)
joint_acc_HR[1:] = (joint_vel_HR[1:] - joint_vel_HR[:-1]) / DT
joint_jrk_HR = np.zeros_like(joint_acc_HR)
joint_jrk_HR[1:] = (joint_acc_HR[1:] - joint_acc_HR[:-1]) / DT

joint_vel_HL = joint_vel_hist[:, 3, :]
joint_acc_HL = np.zeros_like(joint_vel_HL)
joint_acc_HL[1:] = (joint_vel_HL[1:] - joint_vel_HL[:-1]) / DT
joint_jrk_HL = np.zeros_like(joint_acc_HL)
joint_jrk_HL[1:] = (joint_acc_HL[1:] - joint_acc_HL[:-1]) / DT

# 전 관절 가속도 (RNEA용, 4 legs × N_JOINTS_MAX)
joint_acc_hist = np.zeros_like(joint_vel_hist)
joint_acc_hist[1:] = (joint_vel_hist[1:] - joint_vel_hist[:-1]) / DT

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
    dtheta_a_hist[1:, leg, :nj] = np.diff(theta_a_hist[:, leg, :nj], axis=0) / DT

wbc_tau_grav = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
wbc_tau_ff   = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
wbc_tau_pd   = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
wbc_tau_imp  = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
wbc_tau_cmd  = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
wbc_tau_grf  = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))  # GRF로부터 유도되는 토크
wbc_lam_des  = np.zeros((N_FRAMES, 4, 3))   # [Fx, Fy, Fz]
wbc_lam_calc = np.zeros((N_FRAMES, 4, 3))

mpc_fail_count = 0

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

        # body 상태 x0 (이상적: 항상 upright, vx=V)
        x0_mpc = np.array([
            0.0, 0.0, 0.0,      # roll, pitch, yaw
            0.0, 0.0, 0.0,      # px, py, pz
            0.0, 0.0, 0.0,      # ωx, ωy, ωz
            V,   0.0, 0.0,      # vx, vy, vz
            -G_ACC              # g
        ])
        lam_des_all = mpc_qp_plan(x0_mpc, cs, fp)
        # swing foot 강제 0
        for leg in range(4):
            if swing_flag[fi, leg]:
                lam_des_all[leg] = 0.0
    else:
        lam_des_all = qp_grf_distribute(contact_mask, foot_hist[fi])

    wbc_lam_des[fi] = lam_des_all

    # ── leg별 WBC 계산 ────────────────────────────────────────
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

        J   = compute_jacobian_sim(q_t, dh, front)
        J_a = compute_jacobian_sim(q_a, dh, front)
        tau_g = compute_gravity_torque_sim(q_t, dh, lm, front)   # 플롯용 g(q) 보존

        lam_des_leg = lam_des_all[leg]   # [Fx, Fy, Fz]

        # Phase 3: RNEA — M(q)q̈ + C(q,q̇)q̇ + g(q) 완전 동역학
        tau_ff_leg = rnea(q_t, dq_t, ddq_t, dh, lm) - J.T @ lam_des_leg

        foot_t_j5 = foot_local[fi, leg] + J4_TO_J5_SIM_PER_LEG[leg]
        pts_a     = forward_kinematics(q_a, dh=dh)
        foot_a_j5 = _dh_to_sim(pts_a[-1], front_leg=front)
        vel_t     = foot_vel_t[fi, leg]
        vel_a     = J_a @ dq_a

        f_imp       = KP_IMP * (foot_t_j5 - foot_a_j5) + KD_IMP * (vel_t - vel_a)
        tau_imp_leg = J.T @ f_imp
        tau_pd_leg  = KP_PD[:nj] * (q_t - q_a) + KD_PD[:nj] * (dq_t - dq_a)
        tau_grf_leg = J.T @ lam_des_leg  # [MOD] GRF로부터 유도되는 토크
        tau_cmd_leg = tau_pd_leg + tau_ff_leg + tau_imp_leg
#        tau_cmd_leg = np.clip(tau_cmd_leg, -JOINT_TORQUE_LIMIT[:nj], JOINT_TORQUE_LIMIT[:nj])  # 토크 클리핑

        JJT          = J @ J.T + MU_DAMP * np.eye(3)
        lam_calc_leg = np.linalg.solve(JJT, J @ (tau_g - tau_cmd_leg))

        wbc_tau_grav[fi, leg, :nj] = tau_g
        wbc_tau_ff  [fi, leg, :nj] = tau_ff_leg
        wbc_tau_pd  [fi, leg, :nj] = tau_pd_leg
        wbc_tau_imp [fi, leg, :nj] = tau_imp_leg
        wbc_tau_grf [fi, leg, :nj] = tau_grf_leg  # [MOD] 저장
        wbc_tau_cmd [fi, leg, :nj] = tau_cmd_leg
        wbc_lam_calc[fi, leg]      = lam_calc_leg

wbc_dur = time.perf_counter() - wbc_t0
mode_str = f"MPC(N={N_MPC},dt={DT_MPC*1e3:.0f}ms)" if USE_MPC else "QP GRF"
print(f"WBC 완료 [{mode_str}].  {wbc_dur*1e3:.1f}ms 총  ({wbc_dur/N_FRAMES*1e6:.1f}μs/frame)")

# GRF 합산 검증
fz_sum = np.sum(wbc_lam_des[:, :, 2], axis=1)   # (N_FRAMES,)
print(f"  Σλz 평균={fz_sum.mean():.2f}N  (Mg={TOTAL_MASS*G_ACC:.2f}N)  "
      f"오차={abs(fz_sum.mean()-TOTAL_MASS*G_ACC):.2f}N")

for leg in [0, 2]:
    nj = N_JOINTS_PER_LEG[leg]
    peaks_cmd = "  ".join(f"th{j+1}:{np.max(np.abs(wbc_tau_cmd[:, leg, j])):6.2f}"
                      for j in range(nj))
    fx_peak = np.max(np.abs(wbc_lam_des[:, leg, 0]))
    fy_peak = np.max(np.abs(wbc_lam_des[:, leg, 1]))
    fz_peak = np.max(np.abs(wbc_lam_des[:, leg, 2]))
    print(f"  {LEG_NAMES[leg]} τ_cmd peak [N·m]: {peaks_cmd}")
    print(f"  {LEG_NAMES[leg]} λ (GRF) peak [N]: Fx={fx_peak:6.2f}, Fy={fy_peak:6.2f}, Fz={fz_peak:6.2f}")
print("─" * 55)

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
    f'Gait Sim v8  [{GAIT_TYPE.upper()}]  v={V}m/s  T={T}s  D={D}  '
    f'step_h={STEP_HEIGHT}m  step_l={STEP_LENGTH:.3f}m',
    color='white', fontsize=9)
ax3d.view_init(elev=20, azim=-55)
ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False

_bc = np.array([
    LEG_HIP_OFFSETS[0], LEG_HIP_OFFSETS[2],
    LEG_HIP_OFFSETS[3], LEG_HIP_OFFSETS[1],
    LEG_HIP_OFFSETS[0],
])
ax3d.plot(_bc[:,0], _bc[:,1], _bc[:,2], '-', color='white', lw=2.5, alpha=0.7)

gnd_z = home_foot[2]
xx, yy = np.meshgrid([-reach, reach], [-0.5, 0.5])
ax3d.plot_surface(xx, yy, np.full_like(xx, gnd_z), alpha=0.12, color='#888888')

_AX_COLORS = ['#ff4444', '#44ff44', '#4444ff']
for leg in range(4):
    h = LEG_HIP_OFFSETS[leg]
    ax3d.plot([h[0]], [h[1]], [h[2]], 'o', color=LEG_COLORS[leg], markersize=7, alpha=0.8)
    ax3d.text(h[0], h[1], h[2]+0.02, LEG_NAMES[leg], color=LEG_COLORS[leg], fontsize=7)

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
    ax_zv.plot(_fr, foot_vel_t[:, leg, 2], lw=1.6, color=LEG_COLORS[leg], label=LEG_NAMES[leg])
ax_zv.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=_gray, ncol=4)
zv_cursor = ax_zv.axvline(x=0, color='white', lw=1.5, ls='--')

ax_xv = fig.add_subplot(gs[2, 2])
_style_ax(ax_xv, 'Step Length Velocity  dX/dt [m/s]', ylabel='[m/s]')
ax_xv.set_xlim(0, N_FRAMES)
for leg in range(4):
    ax_xv.plot(_fr, foot_vel_t[:, leg, 0], lw=1.6, color=LEG_COLORS[leg], label=LEG_NAMES[leg])
ax_xv.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=_gray, ncol=4)
xv_cursor = ax_xv.axvline(x=0, color='white', lw=1.5, ls='--')

ax_za = fig.add_subplot(gs[3, 1])
_style_ax(ax_za, 'Step Height Acceleration  d²Z/dt² [m/s²]', ylabel='[m/s²]')
ax_za.set_xlim(0, N_FRAMES)
for leg in range(4):
    ax_za.plot(_fr, foot_acc_t[:, leg, 2], lw=1.6, color=LEG_COLORS[leg], label=LEG_NAMES[leg])
ax_za.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', edgecolor=_gray, ncol=4)
za_cursor = ax_za.axvline(x=0, color='white', lw=1.5, ls='--')

ax_xa = fig.add_subplot(gs[3, 2])
_style_ax(ax_xa, 'Step Length Acceleration  d²X/dt² [m/s²]', ylabel='[m/s²]')
ax_xa.set_xlim(0, N_FRAMES)
for leg in range(4):
    ax_xa.plot(_fr, foot_acc_t[:, leg, 0], lw=1.6, color=LEG_COLORS[leg], label=LEG_NAMES[leg])
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
    info_text.set_text('')
    return []


def animate(fi):
    t = fi * DT
    for leg in range(4):
        nj     = N_JOINTS_PER_LEG[leg]
        q      = joint_hist[fi, leg, :nj]
        pts_dh = forward_kinematics(q, dh=LEG_DH[leg])
        pts    = [_dh_to_sim(p, front_leg=(leg < 2)) for p in pts_dh]
        hip    = LEG_HIP_OFFSETS[leg]
        for k in range(nj):
            A = hip + pts[k]; B = hip + pts[k+1]
            leg_links[leg][k].set_data([A[0], B[0]], [A[1], B[1]])
            leg_links[leg][k].set_3d_properties([A[2], B[2]])
        pe = foot_hist[fi, leg]
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
            orig_sim = _dh_to_sim(T_dh[:3, 3], front_leg=(leg < 2))
            pos = hip + orig_sim
            for ax_i in range(3):
                dv = _dh_to_sim(T_dh[:3, ax_i], front_leg=(leg < 2))
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
        tc = wbc_tau_cmd[fi, leg]
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
gs2 = gridspec.GridSpec(4, 2, figure=fig2, wspace=0.35, hspace=0.55,
                        left=0.07, right=0.97, top=0.93, bottom=0.05)

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

    # row 0: tau_cmd
    ax_tc = fig3.add_subplot(gs3[0, col])
    _style_ax3(ax_tc, f'{LEG_NAMES[leg]} tau_cmd [N·m]', ylabel='[N·m]')
    ax_tc.set_xlim(0, N_FRAMES)
    for j in range(nj):
        ax_tc.plot(_fr, wbc_tau_cmd[:, leg, j], lw=1.4, color=_ax5col[j], label=f'th{j+1}')
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
    _style_ax3(ax_fxy, f'{LEG_NAMES[leg]} GRF lam_x/lam_y + 마찰 추 [N]', ylabel='[N]')
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
        ax_td.plot(_fr, wbc_tau_grav[:, leg, ji], lw=1.4, color='#ffcc00', ls='--', label='tau_grav')
        ax_td.plot(_fr, wbc_tau_ff  [:, leg, ji], lw=1.4, color='#00d4ff',           label='tau_ff')
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
    f'Gait Sim v8  |  {GAIT_TYPE.upper()}  |  '
    f'v={V}m/s  T={T}s  D={D}  T_sw={T_SW:.2f}s  '
    f'step_h={STEP_HEIGHT*1e3:.0f}mm  step_l={STEP_LENGTH*1e3:.0f}mm',
    color='white', fontsize=9)
plt.show()
