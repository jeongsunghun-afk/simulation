"""gait_sim.sim_state — SimState dataclass (모든 시뮬레이션 출력 array 통합).

v13.2 Phase 5-a: viz / nmpc / __main__ 간 데이터 전달용 일관 컨테이너.

설계 원칙:
  · main loop 가 SimState 객체 1개 생성 → 각 frame 시점에 array slot 에 write
  · viz 함수는 SimState 1개 인자로 받아 plot
  · 모듈 간 globals 의존 제거 (refactor 의 핵심 단계)

사용:
    from gait_sim.sim_state import SimState
    R = SimState.alloc(n_frames=N_FRAMES)
    R.joint_hist[fi, leg] = q
    ...
    plot_fig_main(R)
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np


N_JOINTS_MAX = 5
N_LEGS       = 4


@dataclass
class SimState:
    """v13 simulation output container.

    필드 이름은 v13.py 의 module-level array 와 1:1 매칭 (검색·치환 호환).
    """
    # ─────────── meta ───────────
    n_frames: int
    dt: float

    # ─────────── Joint state (N, 4, 5) ───────────
    joint_hist:       np.ndarray             # 명령 관절각 [rad]
    joint_vel_hist:   np.ndarray             # q̇  (CubicSpline or gradient)
    joint_acc_hist:   np.ndarray             # q̈
    joint_jrk_hist:   np.ndarray             # q⃛ (viz 전용)
    theta_a_hist:     np.ndarray             # actuator 1st-order lag 적용 θ
    dtheta_a_hist:    np.ndarray             # dθ_a/dt

    # ─────────── Foot trajectory (N, 4, 3) + meta ───────────
    foot_hist:                 np.ndarray    # target world position
    foot_target_world_hist:    np.ndarray    # NMPC swing target (world)
    foot_actual_world_hist:    np.ndarray    # 실제 world foot 위치
    foot_local:                np.ndarray    # body-frame foot
    foot_vel_t:                np.ndarray    # leg-local foot velocity
    foot_acc_t:                np.ndarray    # leg-local foot accel
    phase_hist:                np.ndarray    # (N, 4) per-leg phase [0, 1)

    # ─────────── Body 6-DoF state (N, 3 or 3×3) ───────────
    body_pos_hist:    np.ndarray             # CoM world position
    body_R_hist:      np.ndarray             # (N, 3, 3) rotation
    body_v_hist:      np.ndarray             # CoM linear velocity
    body_omega_hist:  np.ndarray             # angular velocity
    body_alin_hist:   np.ndarray             # linear accel (integrate output)
    body_aang_hist:   np.ndarray             # angular accel
    body_pos_ref_hist: np.ndarray            # MPC reference position
    body_v_ref_hist:   np.ndarray            # MPC reference velocity

    # ─────────── WBC commands (N, 4, 5) + GRF (N, 4, 3) ───────────
    wbc_tau_ff:    np.ndarray                # τ feed-forward (RNEA + Jᵀλ)
    wbc_tau_dyn:   np.ndarray                # M·q̈ + C·q̇ + g(q)
    wbc_tau_pd:    np.ndarray                # joint PD (q tracking)
    wbc_tau_imp:   np.ndarray                # impedance (foot tracking)
    wbc_tau_cmd:   np.ndarray                # final 명령 τ
    wbc_tau_grf:   np.ndarray                # -Jᵀ·λ_des (저장용)
    wbc_lam_des:   np.ndarray                # (N, 4, 3) MPC 출력 GRF
    wbc_lam_calc:  np.ndarray                # (N, 4, 3) per-leg QP GRF calc

    # ─────────── WBIC correction terms ───────────
    wbic_dtau_hist:        np.ndarray        # (N, 4, 5)  Δτ per leg
    wbic_dlam_hist:        np.ndarray        # (N, 4, 3)  Δλ per leg
    wbic_residual_hist:    np.ndarray        # (N, 4)     eq residual norm
    wbic_status_hist:      np.ndarray        # (N, 4) bool 솔버 성공 여부
    wbic_lam_used:         np.ndarray        # (N, 4, 3)  실제 사용된 λ
    wbic_fb_residual_hist: np.ndarray        # (N,) body 6-DoF residual
    wbic_fb_status_hist:   np.ndarray        # (N,) bool
    wbic_fb_dvfb_hist:     np.ndarray        # (N, 6) Δv̇_fb

    # ─────────── opt-IK diagnostics (N, 4) ───────────
    opt_ik_nit_hist:      np.ndarray         # SLSQP iter 수
    opt_ik_fallback_hist: np.ndarray         # True = analytical fallback
    opt_ik_pos_err_hist:  np.ndarray         # 위치 오차² [m²]

    # ─────────── Misc (N,) ───────────
    frame_calc_time: np.ndarray              # per-frame wall time [s]

    # ─────────── NMPC mode 전용 ───────────
    fz_sum_des:  np.ndarray = field(default_factory=lambda: np.zeros(0))
    fz_sum_used: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # ─────────── 단일 state (배열 아님) ───────────
    tau_ff_corrected_prev: List[Optional[np.ndarray]] = field(default_factory=lambda: [None] * N_LEGS)
    body_state: Dict[str, Any] = field(default_factory=dict)

    # ─────────── meta tags (run 정보) ───────────
    mode_str:   str = ""    # 'NMPC'/'v11 hybrid'/'v11 standalone'
    wbic_str:   str = ""    # 'WBIC FB'/'WBIC per-leg'/'no WBIC'
    diverged:   bool = False


    # ══════════════════════════════════════════════════════════════
    # Factory
    # ══════════════════════════════════════════════════════════════
    @staticmethod
    def alloc(n_frames: int, dt: float,
              n_legs: int = N_LEGS, n_joints_max: int = N_JOINTS_MAX) -> 'SimState':
        """모든 array 를 적절한 shape 으로 0-init 한 SimState 생성."""
        N = n_frames
        shape_joint = (N, n_legs, n_joints_max)
        shape_3leg  = (N, n_legs, 3)
        shape_leg   = (N, n_legs)
        shape_3     = (N, 3)
        shape_33    = (N, 3, 3)
        shape_6     = (N, 6)

        return SimState(
            n_frames=N, dt=dt,

            joint_hist=     np.zeros(shape_joint),
            joint_vel_hist= np.zeros(shape_joint),
            joint_acc_hist= np.zeros(shape_joint),
            joint_jrk_hist= np.zeros(shape_joint),
            theta_a_hist=   np.zeros(shape_joint),
            dtheta_a_hist=  np.zeros(shape_joint),

            foot_hist=              np.zeros(shape_3leg),
            foot_target_world_hist= np.full(shape_3leg, np.nan),
            foot_actual_world_hist= np.zeros(shape_3leg),
            foot_local=             np.zeros(shape_3leg),
            foot_vel_t=             np.zeros(shape_3leg),
            foot_acc_t=             np.zeros(shape_3leg),
            phase_hist=             np.zeros(shape_leg),

            body_pos_hist=    np.zeros(shape_3),
            body_R_hist=      np.zeros(shape_33),
            body_v_hist=      np.zeros(shape_3),
            body_omega_hist=  np.zeros(shape_3),
            body_alin_hist=   np.zeros(shape_3),
            body_aang_hist=   np.zeros(shape_3),
            body_pos_ref_hist=np.zeros(shape_3),
            body_v_ref_hist=  np.zeros(shape_3),

            wbc_tau_ff=   np.zeros(shape_joint),
            wbc_tau_dyn=  np.zeros(shape_joint),
            wbc_tau_pd=   np.zeros(shape_joint),
            wbc_tau_imp=  np.zeros(shape_joint),
            wbc_tau_cmd=  np.zeros(shape_joint),
            wbc_tau_grf=  np.zeros(shape_joint),
            wbc_lam_des=  np.zeros(shape_3leg),
            wbc_lam_calc= np.zeros(shape_3leg),

            wbic_dtau_hist=        np.zeros(shape_joint),
            wbic_dlam_hist=        np.zeros(shape_3leg),
            wbic_residual_hist=    np.zeros(shape_leg),
            wbic_status_hist=      np.zeros(shape_leg, dtype=bool),
            wbic_lam_used=         np.zeros(shape_3leg),
            wbic_fb_residual_hist= np.zeros(N),
            wbic_fb_status_hist=   np.zeros(N, dtype=bool),
            wbic_fb_dvfb_hist=     np.zeros(shape_6),

            opt_ik_nit_hist=      np.zeros(shape_leg, dtype=int),
            opt_ik_fallback_hist= np.zeros(shape_leg, dtype=bool),
            opt_ik_pos_err_hist=  np.full(shape_leg, np.nan),

            frame_calc_time= np.zeros(N),

            fz_sum_des=  np.zeros(N),
            fz_sum_used= np.zeros(N),

            tau_ff_corrected_prev=[None] * n_legs,
            body_state={},
        )

    # ══════════════════════════════════════════════════════════════
    # Convenience derived views
    # ══════════════════════════════════════════════════════════════
    def per_leg_joint_vel(self, leg: int) -> np.ndarray:
        return self.joint_vel_hist[:, leg, :]

    def per_leg_joint_acc(self, leg: int) -> np.ndarray:
        return self.joint_acc_hist[:, leg, :]

    def per_leg_joint_jrk(self, leg: int) -> np.ndarray:
        return self.joint_jrk_hist[:, leg, :]
