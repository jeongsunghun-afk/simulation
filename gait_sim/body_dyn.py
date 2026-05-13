"""gait_sim.body_dyn — Floating-base body 6-DoF dynamics integration.

v13.0 Phase 3-c: gait_sim_v13.py 의 body dynamics 영역 (line 903~1010) 추출.

함수:
  · _exp_so3(omega_dt)                   SO(3) Lie 지수 (Rodrigues)
  · _R_to_euler_xyz(R)                   R → (roll, pitch, yaw) [rad]
  · integrate_body_state(body_state,     symplectic Euler 6-DoF body 적분
        lam_used_all, foot_world_all,    GRF + r×λ 모멘트 누적
        M_total, I_body, dt)
"""
import math

import numpy as np

from gait_sim.config import G_ACC
from gait_sim.dynamics import _skew


# ══════════════════════════════════════════════════════════════
# SO(3) exponential — Rodrigues' formula
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# Rotation matrix → Euler XYZ
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# Floating-base body 6-DoF integration (symplectic Euler)
# ══════════════════════════════════════════════════════════════
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

    pos_new = pos + dt * v_new
    R_new   = _exp_so3(omega_new * dt) @ R

    body_state['pos']   = pos_new
    body_state['R']     = R_new
    body_state['v']     = v_new
    body_state['omega'] = omega_new
    body_state['a_lin'] = a_lin
    body_state['a_ang'] = a_ang
    return body_state
