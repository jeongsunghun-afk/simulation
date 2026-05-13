"""gait_sim.kinematics — FK / IK / Jacobian / gravity torque.

v13.0 Phase 3-a: gait_sim_v12.py 의 kinematics 영역 (line 385~632) 추출.

함수:
  · _dh_matrix(alpha, a, d, theta)            DH transform
  · forward_kinematics(thetas, dh)            FK chain → joint positions
  · _dh_to_sim(vec, front_leg) / _sim_to_dh   DH↔sim frame 변환
  · analytical_ik_front / _hind               해석적 IK
  · opt_ik_front / _hind                      SLSQP 최적화 IK
  · compute_jacobian_sim(thetas, dh, front)   3×nj velocity Jacobian
  · compute_gravity_torque_sim(thetas, dh, link_mass, front)  중력 토크
"""
import math

import numpy as np
from scipy.optimize import minimize as _sp_minimize

from gait_sim.config import (
    DT, G_ACC,
)
from gait_sim.model import (
    DH_FRONT, DH_HIND, DH_FRONT_R, DH_FRONT_L, DH_HIND_R, DH_HIND_L,
    _A2_F, _A3_F, _A4_F, _A5_F, _D2_F,
    Q_HOME_FRONT, Q_HOME_HIND,
    FRONT_Q_LIM, HIND_Q_LIM,
    JOINT_VEL_LIMIT_RAD_S, JOINT_TORQUE_LIMIT,
    LINK_MASS_PER_LEG,
)


# ══════════════════════════════════════════════════════════════
# DH transform + FK
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


# ══════════════════════════════════════════════════════════════
# DH ↔ sim frame 변환
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# Home pose 기반 J4→J5 offset (시각화 / IK 보정용)
# ══════════════════════════════════════════════════════════════
_fk_front_home      = forward_kinematics(Q_HOME_FRONT, DH_FRONT)
_FRONT_J4_TO_J5_DH  = np.array(_fk_front_home[5]) - np.array(_fk_front_home[4])
_FRONT_J4_TO_J5_SIM = _dh_to_sim(_FRONT_J4_TO_J5_DH, front_leg=True)

_fk_hind_home      = forward_kinematics(Q_HOME_HIND, DH_HIND)
_HIND_J4_TO_J5_DH  = np.array(_fk_hind_home[5]) - np.array(_fk_hind_home[4])
_HIND_J4_TO_J5_SIM = _dh_to_sim(_HIND_J4_TO_J5_DH, front_leg=False)

J4_TO_J5_SIM_PER_LEG = [_FRONT_J4_TO_J5_SIM, _FRONT_J4_TO_J5_SIM,
                        _HIND_J4_TO_J5_SIM,  _HIND_J4_TO_J5_SIM]


# ══════════════════════════════════════════════════════════════
# Analytical IK (앞다리 / 뒷다리)
# ══════════════════════════════════════════════════════════════
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
    theta2  = (math.atan2(z3, x3)
               - math.atan2(a3 * math.sin(theta3), a2 + a3 * math.cos(theta3)))
    theta4  = phi - theta2 - theta3
    theta5  = theta5_target - (theta2 + theta3 + theta4)
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


# ══════════════════════════════════════════════════════════════
# Jacobian (linear part, sim frame)
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# Gravity torque (각 joint 의 중력 부담)
# ══════════════════════════════════════════════════════════════
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
# Optimization-based IK (SLSQP)
# ══════════════════════════════════════════════════════════════
def opt_ik_front(p_target_dh, q_init, q_ref=None, dh=None,
                 lambda_q=1.0, lambda_tau=0.01, maxiter=100,
                 use_vel_limit=True, use_tau_limit=True):
    """SLSQP 최적화 IK — 앞다리. v13: dh 인자로 좌우 (DH_FRONT_R/L) 분리.

    등식 제약 : FK_tip(q) = p_target, q[4] = Q_HOME_FRONT[4]
    부등식    : |τ_grav(q)| ≤ τ_limit (use_tau_limit)
    bounds    : FRONT_Q_LIM ∩ 각속도 한계 (use_vel_limit)
    비용      : λ_q·||q - q_ref||² + λ_tau·||τ_grav(q)||²
    """
    if dh is None:
        dh = DH_FRONT_R   # v13 후방호환 default (= 우측)
    p_t   = np.asarray(p_target_dh, dtype=float)
    q0    = np.asarray(q_init, dtype=float)
    q_tgt = q0 if q_ref is None else np.asarray(q_ref, dtype=float)

    if use_vel_limit:
        vel_dt = JOINT_VEL_LIMIT_RAD_S * DT
        lo = np.maximum([b[0] for b in FRONT_Q_LIM], q0 - vel_dt)
        hi = np.minimum([b[1] for b in FRONT_Q_LIM], q0 + vel_dt)
        hi = np.maximum(lo, hi)
        active_bounds = list(zip(lo, hi))
    else:
        active_bounds = FRONT_Q_LIM

    constraints = [{'type': 'eq',
                    'fun': lambda q: np.array(forward_kinematics(q, dh)[-1]) - p_t}]
    constraints.append({'type': 'eq',
                        'fun': lambda q: q[4] - Q_HOME_FRONT[4]})

    _lm_front = LINK_MASS_PER_LEG[0]
    if use_tau_limit:
        def _torque_ineq(q):
            tau_g = compute_gravity_torque_sim(q, dh, _lm_front, front_leg=True)
            return JOINT_TORQUE_LIMIT[:len(tau_g)] - np.abs(tau_g)
        constraints.append({'type': 'ineq', 'fun': _torque_ineq})

    def cost(q):
        c_qref = lambda_q * np.dot(q - q_tgt, q - q_tgt)
        tau_g  = compute_gravity_torque_sim(q, dh, _lm_front, front_leg=True)
        c_tau  = lambda_tau * np.dot(tau_g, tau_g)
        return float(c_qref + c_tau)

    res = _sp_minimize(cost, q0, method='SLSQP', bounds=active_bounds,
                       constraints=constraints,
                       options={'ftol': 1e-8, 'maxiter': maxiter})
    tip_final  = np.array(forward_kinematics(res.x, dh)[-1])
    pos_err_sq = float(np.dot(tip_final - p_t, tip_final - p_t))
    if pos_err_sq < 1e-6:
        return list(res.x), res.nit, pos_err_sq
    return None, res.nit, pos_err_sq


def opt_ik_hind(p_target_dh, q_init, q_ref=None, dh=None,
                lambda_q=1.0, lambda_tau=0.01, maxiter=100,
                use_vel_limit=True, use_tau_limit=True):
    """SLSQP 최적화 IK — 뒷다리. v13: dh 인자로 좌우 (DH_HIND_R/L) 분리."""
    if dh is None:
        dh = DH_HIND_R   # v13 후방호환 default (= 우측)
    p_t   = np.asarray(p_target_dh, dtype=float)
    q0    = np.asarray(q_init, dtype=float)
    q_tgt = q0 if q_ref is None else np.asarray(q_ref, dtype=float)

    if use_vel_limit:
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
    if use_tau_limit:
        def _torque_ineq(q):
            tau_g = compute_gravity_torque_sim(q, dh, _lm_hind, front_leg=False)
            return JOINT_TORQUE_LIMIT[:len(tau_g)] - np.abs(tau_g)
        constraints.append({'type': 'ineq', 'fun': _torque_ineq})

    def cost(q):
        c_qref = lambda_q * np.dot(q - q_tgt, q - q_tgt)
        tau_g  = compute_gravity_torque_sim(q, dh, _lm_hind, front_leg=False)
        c_tau  = lambda_tau * np.dot(tau_g, tau_g)
        return float(c_qref + c_tau)

    res = _sp_minimize(cost, q0, method='SLSQP', bounds=active_bounds,
                       constraints=constraints,
                       options={'ftol': 1e-8, 'maxiter': maxiter})
    tip_final  = np.array(forward_kinematics(res.x, dh)[-1])
    pos_err_sq = float(np.dot(tip_final - p_t, tip_final - p_t))
    if pos_err_sq < 1e-6:
        return list(res.x), res.nit, pos_err_sq
    return None, res.nit, pos_err_sq
