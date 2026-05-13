"""gait_sim.controllers.wbic — WBIC QP correction (per-leg + full body 6-DoF).

v13.0 Phase 4-c: gait_sim_v13.py 의 WBIC QP 영역
                 (line 831~898 wbic_qp_leg
                  + line 1013~1121 wbic_qp_full_pin
                  + line 1123~1322 wbic_qp_full) 추출.

함수:
  · wbic_qp_leg(...)         per-leg WBIC QP (block-diagonal 근사)
  · wbic_qp_full(...)        body 6-DoF + per-leg 통합 단일 QP (no pinocchio)
  · wbic_qp_full_pin(...)    pinocchio full M_full (26×26) + J_full 사용

τ smoothness penalty: tau_prev / w_dtau 인자, w_dtau=0 이면 OFF (v13.1 default).
"""
import numpy as np

import qpsolvers

from gait_sim.config import CFG, G_ACC
from gait_sim.model import JOINT_TORQUE_LIMIT
from gait_sim.dynamics import _skew


# ══════════════════════════════════════════════════════════════
# Per-leg WBIC QP correction (block-diagonal approximation)
# ══════════════════════════════════════════════════════════════
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

    # τ smoothness: ‖(tau_ff + Δτ) − tau_prev‖²  →  P[Δτ] += w·I, qv[Δτ] += w·c
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
        sol = qpsolvers.solve_qp(P, qv, G, h_ineq, A_eq, b_eq, lb, ub, solver=CFG.qp_solver)
    except Exception:
        sol = None
    if sol is None:
        return None, None, None, False, residual_pre
    return sol[:nj], sol[nj:2*nj], sol[2*nj:], True, residual_pre


# ══════════════════════════════════════════════════════════════
# Full body 6-DoF + per-leg WBIC QP (no pinocchio)
# ══════════════════════════════════════════════════════════════
def wbic_qp_full(M_legs, h_legs, ddq_des_legs, tau_ff_legs, lam_des_all,
                 J_legs, contact_mask, nj_per_leg, foot_world_all, body_pos,
                 v_dot_des_fb,
                 M_total, I_body, omega_world,
                 w_ddq, w_tau, w_lam, w_fb, lamz_min, mu,
                 stance_foot_J_v11=None,
                 tau_prev_legs=None, w_dtau=0.0):
    """v11 Phase 7 — WBIC 부유 베이스 통합 단일 QP.

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
    n_legs = 4
    nj_total = sum(nj_per_leg[:n_legs])
    n_fb  = 6
    n_ddq = nj_total
    n_tau = nj_total
    n_lam = 12
    n_v   = n_fb + n_ddq + n_tau + n_lam

    sl_fb  = slice(0, n_fb)
    sl_ddq = slice(n_fb, n_fb + n_ddq)
    sl_tau = slice(n_fb + n_ddq, n_fb + n_ddq + n_tau)
    sl_lam = slice(n_fb + n_ddq + n_tau, n_v)
    leg_off_ddq = [sum(nj_per_leg[:i]) for i in range(n_legs)]
    leg_off_tau = leg_off_ddq
    leg_off_lam = [3*i for i in range(n_legs)]

    # 비용
    P = np.zeros((n_v, n_v), dtype=float)
    P[sl_fb,  sl_fb]  = w_fb  * np.eye(n_fb)
    P[sl_ddq, sl_ddq] = w_ddq * np.eye(n_ddq)
    P[sl_tau, sl_tau] = w_tau * np.eye(n_tau)
    P[sl_lam, sl_lam] = w_lam * np.eye(n_lam)
    qv = np.zeros(n_v, dtype=float)

    # τ smoothness penalty: w_dtau·‖(tau_ff + Δτ) − tau_prev‖²  per leg.
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

    rhs_fb = M_fb @ v_dot_des_fb - F_des

    A_eq_fb = np.zeros((6, n_v))
    A_eq_fb[:, sl_fb] = M_fb
    for i in range(n_legs):
        r_i = foot_world_all[i] - body_pos
        col = sl_lam.start + leg_off_lam[i]
        A_eq_fb[:3, col:col+3] += -np.eye(3)
        A_eq_fb[3:, col:col+3] += -_skew(r_i)
    b_eq_fb = -rhs_fb

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
    if stance_foot_J_v11 is not None:
        W_STANCE = 1.0
        for i in range(n_legs):
            if contact_mask[i]:
                Ji = stance_foot_J_v11[i]
                J_aug = np.zeros((3, n_v))
                J_aug[:, :6]      = Ji[:, :6]
                J_aug[:, sl_ddq]  = Ji[:, 6:]
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
            lb[l_off + 2] = max(lb[l_off + 2], lamz_min - lam_des_all[i, 2])
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
            for k in range(3):
                lb[l_off + k] = -lam_des_all[i, k]
                ub[l_off + k] = -lam_des_all[i, k]

    G_ineq = np.vstack(G_ineq_list) if G_ineq_list else None
    h_ineq = np.array(h_ineq_list) if h_ineq_list else None

    try:
        sol = qpsolvers.solve_qp(P, qv, G_ineq, h_ineq, A_eq, b_eq, lb, ub, solver=CFG.qp_solver)
    except Exception:
        sol = None

    if sol is None:
        return None
    return {
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


# ══════════════════════════════════════════════════════════════
# Full body 6-DoF + per-leg WBIC QP (pinocchio M_full version)
# ══════════════════════════════════════════════════════════════
def wbic_qp_full_pin(M_full, h_full, J_full,
                     ddq_des_legs, tau_ff_legs, lam_des_all,
                     contact_mask, nj_per_leg,
                     v_dot_des_fb,
                     w_ddq, w_tau, w_lam, w_fb, lamz_min, mu):
    """v11 Phase 7 (FULL M version) — pinocchio가 제공한 26×26 M, 26 h, 12×26 J 사용.

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
    n_eq  = n_fb + nj_total

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
    A_eq = np.zeros((n_eq, n_v), dtype=float)
    A_eq[:, sl_fb]  = M_full[:, 0:6]
    A_eq[:, sl_ddq] = M_full[:, 6:26]
    A_eq[6:26, sl_tau] = -np.eye(nj_total)
    A_eq[:, sl_lam] = -J_full.T

    ddq_des_full = np.concatenate([v_dot_des_fb, *ddq_des_legs])
    tau_ff_full  = np.concatenate([np.zeros(n_fb), *tau_ff_legs])
    lam_des_flat = lam_des_all.reshape(-1)
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
        sol = qpsolvers.solve_qp(P, qv, G_ineq, h_ineq, A_eq, b_eq, lb, ub, solver=CFG.qp_solver)
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
