"""gait_sim.dynamics — Leg rigid-body dynamics (RNEA / M·h decomposition).

v13.0 Phase 3-b: gait_sim_v13.py 의 dynamics 영역 (line 709~828) 추출.

함수:
  · _skew(v)                            3D skew-symmetric matrix
  · _rod_inertia_local(mass, length)    원통 link 관성 텐서 (로컬)
  · rnea(q, dq, ddq, dh, link_mass)     τ = M(q)q̈ + C(q,q̇)q̇ + g(q)  (per-leg)
  · compute_mh_leg(q, dq, dh, lm)       (M, h)  composite-rigid-body trick
"""
import math

import numpy as np

from gait_sim.config import G_ACC
from gait_sim.model import LINK_RADIUS
from gait_sim.kinematics import _dh_matrix


# ══════════════════════════════════════════════════════════════
# Skew-symmetric matrix
# ══════════════════════════════════════════════════════════════
def _skew(v):
    """3D 벡터 → 반대칭(skew-symmetric) 행렬"""
    return np.array([[ 0.0,  -v[2],  v[1]],
                     [ v[2],  0.0,  -v[0]],
                     [-v[1],  v[0],  0.0 ]], dtype=float)


# ══════════════════════════════════════════════════════════════
# Cylindrical rod inertia (link 단위)
# ══════════════════════════════════════════════════════════════
def _rod_inertia_local(mass, length, radius=LINK_RADIUS):
    """원통 막대 관성 텐서 (로컬 프레임, x축 = 막대 축) [kg·m²]
    Ixx(축 방향): m·r²/2   Iyy=Izz(수직): m(3r²+L²)/12
    """
    Ixx = 0.5 * mass * radius ** 2
    Iyy = mass * (3.0 * radius**2 + length**2) / 12.0
    return np.diag([Ixx, Iyy, Iyy])


# ══════════════════════════════════════════════════════════════
# Recursive Newton-Euler Algorithm — per-leg
# ══════════════════════════════════════════════════════════════
def rnea(q, dq, ddq, dh, link_mass):
    """RNEA:  tau = M(q)·q̈ + C(q,q̇)·q̇ + g(q)

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


# ══════════════════════════════════════════════════════════════
# Mass-matrix + bias-force decomposition (composite-rigid-body trick)
# ══════════════════════════════════════════════════════════════
def compute_mh_leg(q, dq, dh, lm):
    """Mass matrix M(q) (n×n) 과 h(q,q̇)=C·q̇+g(q) (n,) 를 RNEA로 추출.
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
