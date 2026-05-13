"""gait_sim.actuator — Actuator dynamics model (T-N curve + stiction).

v13.x Phase 7 Tier 1: sim2real gap 축소 위한 actuator 비선형 모델.

함수:
  · apply_torque_speed_curve(tau_cmd, dq, cfg) → tau_available_clipped
        Motor T-N (torque-speed) curve 적용. 고속 시 τ 감소.
  · apply_stiction(tau_cmd, dq, cfg) → tau_with_stiction
        Coulomb + viscous 정마찰 모델.
  · apply_actuator_dynamics(tau_cmd, dq, cfg) → tau_real
        통합 actuator pipeline (T-N + stiction).

기존 TAU_LAG (1st-order lag) 는 main loop 에서 별도 처리 (theta_a_hist).
이 모듈은 *순간* torque transformation 만 담당.

수식 (Motor T-N curve, linear drop-off):
    tau_available(dq) = tau_peak * (1 - max(0, |dq| - dq_idle) / (dq_max - dq_idle))
        dq_idle ≤ |dq| ≤ dq_max 범위에서 linear 감소
        dq < dq_idle 이면 tau_peak 그대로
        dq > dq_max 면 0 (over-speed)

수식 (Stiction):
    tau_friction(dq) = tau_static * sign(dq) (if |dq| < eps_v else 0)
                     + b_v * dq                  (viscous damping)
"""
import numpy as np


def apply_torque_speed_curve(tau_cmd: np.ndarray, dq: np.ndarray,
                              tau_peak: np.ndarray, dq_max: np.ndarray,
                              dq_idle_ratio: float = 0.3) -> np.ndarray:
    """Motor T-N curve linear drop-off.

    Args:
        tau_cmd:  (n,) commanded torque [N·m]
        dq:       (n,) joint velocity   [rad/s]
        tau_peak: (n,) stall torque (low-speed peak) [N·m]
        dq_max:   (n,) no-load max speed [rad/s]
        dq_idle_ratio: peak τ 유지 비율 (0~1). 기본 0.3 = dq_max 의 30% 까지 peak.

    Returns: (n,) actual achievable torque (clipped to T-N envelope)
    """
    abs_dq = np.abs(dq)
    dq_idle = dq_max * dq_idle_ratio
    # Available torque envelope (positive value)
    above_idle = np.maximum(0.0, abs_dq - dq_idle)
    range_drop = np.maximum(dq_max - dq_idle, 1e-9)
    tau_avail = tau_peak * np.maximum(0.0, 1.0 - above_idle / range_drop)
    # Clip cmd torque to [-tau_avail, +tau_avail]
    return np.clip(tau_cmd, -tau_avail, +tau_avail)


def apply_stiction(tau_cmd: np.ndarray, dq: np.ndarray,
                    tau_static: np.ndarray, b_viscous: np.ndarray = None,
                    eps_v: float = 0.01) -> np.ndarray:
    """Coulomb stiction + viscous friction.

    정지/저속 (|dq| < eps_v) 시 static friction 으로 인해 실제 토크 감소.
    고속 시 viscous damping 추가.

    Args:
        tau_cmd:    (n,) commanded torque
        dq:         (n,) joint velocity
        tau_static: (n,) static friction torque (sign-dependent on dq direction)
        b_viscous:  (n,) viscous damping coefficient [N·m·s/rad]. None → 0.
        eps_v:      static/kinetic 전환 속도 threshold

    Returns: (n,) τ after friction subtraction
    """
    abs_dq = np.abs(dq)
    static_mask = abs_dq < eps_v

    # Static regime: tau_cmd 가 tau_static 보다 작으면 motion 없음 (output=0)
    # tau_cmd 가 |tau_static| 초과 시 그 차이만 net torque
    static_friction = np.where(static_mask,
                                np.clip(tau_cmd, -tau_static, +tau_static),
                                tau_static * np.sign(dq))
    tau_after = tau_cmd - static_friction

    # Viscous damping
    if b_viscous is not None:
        tau_after = tau_after - b_viscous * dq

    return tau_after


def apply_actuator_dynamics(tau_cmd: np.ndarray, dq: np.ndarray,
                              tau_peak: np.ndarray, dq_max: np.ndarray,
                              tau_static: np.ndarray = None,
                              b_viscous: np.ndarray = None,
                              dq_idle_ratio: float = 0.3,
                              eps_v: float = 0.01) -> np.ndarray:
    """통합 actuator pipeline: T-N curve → stiction → final τ.

    Args:
        tau_cmd, dq:       (n,) command + measurement
        tau_peak, dq_max:  (n,) motor characteristics
        tau_static:        (n,) static friction (None → skip)
        b_viscous:         (n,) viscous damping (None → skip)
        dq_idle_ratio:     T-N curve idle range
        eps_v:             stiction threshold

    Returns: (n,) actual joint torque applied
    """
    tau_clipped = apply_torque_speed_curve(tau_cmd, dq, tau_peak, dq_max, dq_idle_ratio)
    if tau_static is not None or b_viscous is not None:
        ts = tau_static if tau_static is not None else np.zeros_like(tau_cmd)
        tau_clipped = apply_stiction(tau_clipped, dq, ts, b_viscous, eps_v)
    return tau_clipped
