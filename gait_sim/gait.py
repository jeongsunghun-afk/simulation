"""gait_sim.gait — GaitScheduler + foot trajectory generation.

v13.0 Phase 4-a: gait_sim_v13.py 의 gait 영역 (line 1582~1707) 추출.

포함:
  · GaitScheduler                       per-leg phase / swing-stance 판정
  · _swing_z_coeffs(step_h, T_half, V_z_boundary)
                                        swing Z 6차 다항식 계수 (Zeng 2019)
  · swing_foot_pos(sw_t, p_start, p_end, body_vel, ...)
                                        swing 궤적 (X 5차, Y 5차 smoothstep, Z 6차)
  · stance_foot_pos(st_t, p_contact, body_vel, stance_dur)
                                        stance 등속도 + 가상 침투
  · _quintic_s(tau) / _smootherstep(tau)  C2/C3 보간
  · foot_pos_at_phase(phase, ...)        swing/stance 통합 dispatcher
"""
import math

import numpy as np

from gait_sim.config import (
    GAIT_TYPE, T, D, T_SW, T_ST, STEP_HEIGHT, TAU_LAND,
    STANCE_DELTA, PHASE_OFFSETS,
)


# ══════════════════════════════════════════════════════════════
# GaitScheduler — per-leg phase / swing-stance 판정
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


# ══════════════════════════════════════════════════════════════
# Swing Z 6차 다항식 계수 (cached)
# ══════════════════════════════════════════════════════════════
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
    T_ = T_half
    A = np.array([
        [T_**2,   T_**4,    T_**6   ],
        [2.0,    12.0*T_**2, 30.0*T_**4],
        [2.0*T_,  4.0*T_**3,  6.0*T_**5],
    ])
    b = np.array([-step_h, 0.0, -V_z_boundary])
    c2, c4, c6 = np.linalg.solve(A, b)
    _swing_z_cache[key] = (c2, c4, c6)
    return (c2, c4, c6)


# ══════════════════════════════════════════════════════════════
# Swing foot trajectory (Zeng 2019 Scheme I)
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# Stance foot trajectory (Zeng 2019)
# ══════════════════════════════════════════════════════════════
def stance_foot_pos(st_t, p_contact, body_vel, stance_dur):
    """Stance 궤적 (Zeng 2019 Eq 11-13).
      X: 등속도 V·t  (지면 미끄러짐 0)
      Z: -Δ·sin(π·t/T_st)  (가상 침투, 임피던스 흡수용; swing 끝/시작과 C2 연속)
    """
    pos = p_contact - body_vel * stance_dur * st_t
    pos = pos.copy()
    pos[2] = p_contact[2] - STANCE_DELTA * math.sin(math.pi * st_t)
    return pos


# ══════════════════════════════════════════════════════════════
# 보간 helper (C2 / C3)
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# 통합 dispatcher (phase 기반)
# ══════════════════════════════════════════════════════════════
def foot_pos_at_phase(phase, p_start, p_contact, p_end, body_vel,
                      swing_ratio=D, step_height=STEP_HEIGHT,
                      tau_land=TAU_LAND, stance_dur=T_ST):
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
