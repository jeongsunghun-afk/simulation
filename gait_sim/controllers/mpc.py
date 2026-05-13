"""gait_sim.controllers.mpc — Linear Convex MPC body trajectory planner.

v13.0 Phase 4-b: gait_sim_v13.py 의 MPC + QP GRF distribute 영역
                 (line 1325~1580 + module-level _Q_DIAG/MPC_Q/MPC_R) 추출.

함수 / 상수:
  · DT_MPC, N_MPC, MPC_Q, MPC_R          MPC config (DT*10 = 0.02s)
  · qp_grf_distribute(contact, foot_pos) 단일-스텝 QP GRF fallback
  · _euler_to_R / _euler_rate_T          회전 / Euler-rate transform
  · _build_Ac_at(roll, pitch)            연속시간 A (13×13, large-angle)
  · _build_Bc_at(contact, foot, I⁻¹)     연속시간 B (13×12)
  · _build_Ac_d / _build_Bc              [legacy] 시불변 small-angle
  · mpc_qp_plan(x0, contact_schedule,    Di Carlo 2018 LMPC + LTV 확장
                foot_positions,
                x_ref_step, ltv)
"""
import math

import numpy as np

import qpsolvers

from gait_sim.config import CFG, DT, G_ACC
from gait_sim.model import BODY_INERTIA, TOTAL_MASS, MU_FRICTION
from gait_sim.dynamics import _skew


# ══════════════════════════════════════════════════════════════
# MPC config — DT*10 = 0.02s, horizon N_MPC, weights _Q_DIAG
# v13: py/vy 가중치 활성화 (MPC closed-loop y drift 보정)
# ══════════════════════════════════════════════════════════════
DT_MPC = DT * 10         # MPC 샘플링 주기 [s]  (= 0.02s)  v13.1: v12 smoothness 패치 원복
N_MPC  = CFG.n_mpc

# MPC 상태 가중치: x=[roll,pitch,yaw, px,py,pz, ωx,ωy,ωz, vx,vy,vz, g]
_Q_DIAG = np.array([
    200, 200, 100,   # roll, pitch, yaw
      0, 100, 200,   # px, py, pz  (v13: py 활성화)
      0,   0,   0,   # ωx, ωy, ωz
     10,  10,   0,   # vx, vy, vz  (v13: vy 활성화)
      0,             # g (상수, 추종 불필요)
], dtype=float)
MPC_Q = np.diag(_Q_DIAG)
MPC_R = 1e-6 * np.eye(3)   # GRF 가중치 (per foot, 3×3)


# ══════════════════════════════════════════════════════════════
# QP GRF distribute — 단일 스텝 fallback (MPC 실패 시)
# ══════════════════════════════════════════════════════════════
def qp_grf_distribute(contact_mask, foot_pos_world):
    """단일 스텝 QP GRF 배분 (MPC fallback).
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
        x_opt = qpsolvers.solve_qp(P, q, G, h, A_eq, b_eq, solver=CFG.qp_solver)
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


# ══════════════════════════════════════════════════════════════
# MPC 관련 캐시
# ══════════════════════════════════════════════════════════════
_I_inv  = np.linalg.inv(BODY_INERTIA)
_I_BODY = BODY_INERTIA.copy()


# ══════════════════════════════════════════════════════════════
# 회전 / Euler-rate transform
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# LTV-MPC A/B 빌드
# ══════════════════════════════════════════════════════════════
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
    """[legacy, hover-at-x0 호환용] 시불변 Ac_d — small-angle 가정 (T=I)."""
    Ac = _build_Ac_at(0.0, 0.0)
    return np.eye(13) + DT_MPC * Ac


_Ac_d = _build_Ac_d()
# Ac_d 거듭제곱 사전 계산 (0 ~ N_MPC) — closed_loop=False 경로용
_Ad_powers = [np.eye(13, dtype=float)]
for _k in range(N_MPC):
    _Ad_powers.append(_Ac_d @ _Ad_powers[-1])


def _build_Bc(contact_mask_k, foot_pos_k):
    """[legacy, hover-at-x0 호환용] 시불변 I_inv 사용."""
    return DT_MPC * _build_Bc_at(contact_mask_k, foot_pos_k, _I_inv)


# ══════════════════════════════════════════════════════════════
# Convex MPC QP (Di Carlo 2018 + v11 LTV 확장)
# ══════════════════════════════════════════════════════════════
def mpc_qp_plan(x0, contact_schedule, foot_positions, x_ref_step=None, ltv=False):
    """Convex MPC QP (Di Carlo et al., IROS 2018 simplified).

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
        roll0, pitch0, yaw0 = x0[0], x0[1], x0[2]
        R_now      = _euler_to_R(roll0, pitch0, yaw0)
        I_world    = R_now @ _I_BODY @ R_now.T
        I_world_inv = np.linalg.inv(I_world)
        Ac_now     = _build_Ac_at(roll0, pitch0)
        Ad_now     = np.eye(nx) + DT_MPC * Ac_now
        Ad_powers_now = [np.eye(nx, dtype=float)]
        for _k in range(N_MPC):
            Ad_powers_now.append(Ad_now @ Ad_powers_now[-1])
        Bc_list = [DT_MPC * _build_Bc_at(contact_schedule[k], foot_positions[k], I_world_inv)
                   for k in range(N_MPC)]
        Ad_powers_use = Ad_powers_now
    else:
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

    # 비용 H, f
    err0  = Aq @ x0 - X_ref  # (N*nx,)
    QBq   = np.zeros_like(Bq)
    Qerr  = np.zeros(N*nx, dtype=float)
    for i in range(N):
        sl = slice(i*nx, (i+1)*nx)
        QBq[sl, :] = MPC_Q @ Bq[sl, :]
        Qerr[sl]   = MPC_Q @ err0[sl]

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
            P=H, q=f, G=G_mpc, h=h_mpc, A=A_mpc, b=b_mpc, solver=CFG.qp_solver
        )
    except Exception:
        u_opt = None

    lam_des = np.zeros((4, 3))
    if u_opt is not None:
        for i in range(4):
            lam_des[i] = u_opt[i*3:(i+1)*3]
    else:
        lam_des = qp_grf_distribute(contact_schedule[0], foot_positions[0])
    return lam_des
