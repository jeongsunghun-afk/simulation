"""v14.3-d4: Thin wrapper around gait_sim.runner._step_one_frame for external simulators.

Usage (Isaac Lab closed-loop):

    from gait_sim.bridge.controller_step import GaitSimControllerStep

    ctrl = GaitSimControllerStep(gait_type='trot')
    for fi in range(ctrl.n_frames):
        # 1. read Isaac state
        q_meas    = ...  # (4, 5) joint angles
        qdot_meas = ...  # (4, 5) joint velocities
        body_pos  = ...  # (3,) world
        body_R    = ...  # (3, 3) rotation
        body_v    = ...  # (3,) world linear velocity
        body_w    = ...  # (3,) world angular velocity

        # 2. step controller with external state
        ctrl.step_closed_loop(fi, q_meas, qdot_meas, body_pos, body_R, body_v, body_w)

        # 3. read controller outputs
        tau_cmd  = ctrl.get_tau_cmd(fi)   # (4, 5)
        q_target = ctrl.get_q_target(fi)  # (4, 5)

        # 4. apply to Isaac (effort or position+effort)
        ...
"""
from __future__ import annotations
import numpy as np

from gait_sim.runner import (
    _step_one_frame, precompute_trajectories, compute_derivatives, init_wbic_state,
)
from gait_sim.gait import GaitScheduler
from gait_sim.sim_state import SimState
from gait_sim.config import N_FRAMES, DT, CFG
from gait_sim.model import N_JOINTS_PER_LEG


class GaitSimControllerStep:
    """Per-frame MPC + WBIC controller, callable from external simulator."""

    def __init__(self, gait_type: str | None = None):
        """Pre-compute trajectories and gait scheduler.

        Args:
            gait_type: 'trot', 'walk', etc. — overrides CFG.gait_type if given.
        """
        if gait_type is not None:
            CFG.gait_type = gait_type

        self.R = SimState.alloc(n_frames=N_FRAMES, dt=DT)
        self.sched = GaitScheduler()
        precompute_trajectories(self.R, self.sched)
        compute_derivatives(self.R)

        self.body_state, self.foot_z_home = init_wbic_state(self.R)
        self.swing_flag = (self.R.phase_hist < self.sched.swing_ratio)
        self.n_frames = N_FRAMES
        self.dt = DT
        self._fail = dict(mpc=0, wbic=0, wbic_fb=0)

    # ── controller invocation ─────────────────────────────────
    def step_open_loop(self, fi: int) -> None:
        """One frame using gait_sim's internal body integration (identical to run_wbic_loop)."""
        m, w, fb = _step_one_frame(self.R, self.sched, self.body_state,
                                     fi, self.foot_z_home, self.swing_flag)
        self._fail['mpc']    += m
        self._fail['wbic']   += w
        self._fail['wbic_fb'] += fb

    def step_closed_loop(self,
                          fi: int,
                          q_meas: np.ndarray,      # (4, 5)
                          qdot_meas: np.ndarray,   # (4, 5)
                          body_pos: np.ndarray,    # (3,)
                          body_R: np.ndarray,      # (3, 3)
                          body_vel: np.ndarray,    # (3,) world linear
                          body_omega: np.ndarray,  # (3,) world angular
                          ) -> None:
        """Override controller's state with external measurements, then step.

        Body integration inside _step_one_frame still runs (writes to body_state),
        but at next step_closed_loop call the body_state is overwritten again.
        Joint state (theta_a, dtheta_a) is fed from external — controller's
        feed-forward (RNEA, WBIC) uses Isaac's actual state.
        """
        for leg in range(4):
            nj = N_JOINTS_PER_LEG[leg]
            self.R.theta_a_hist[fi,  leg, :nj] = q_meas[leg, :nj]
            self.R.dtheta_a_hist[fi, leg, :nj] = qdot_meas[leg, :nj]

        self.body_state['pos']   = np.asarray(body_pos,   dtype=float).copy()
        self.body_state['R']     = np.asarray(body_R,     dtype=float).copy()
        self.body_state['v']     = np.asarray(body_vel,   dtype=float).copy()
        self.body_state['omega'] = np.asarray(body_omega, dtype=float).copy()

        m, w, fb = _step_one_frame(self.R, self.sched, self.body_state,
                                     fi, self.foot_z_home, self.swing_flag)
        self._fail['mpc']    += m
        self._fail['wbic']   += w
        self._fail['wbic_fb'] += fb

    # ── readouts ──────────────────────────────────────────────
    def get_tau_cmd(self, fi: int) -> np.ndarray:
        """(4, 5) final commanded joint torque (= tau_pd + tau_ff + tau_imp, clipped)."""
        return self.R.wbc_tau_cmd[fi].copy()

    def get_q_target(self, fi: int) -> np.ndarray:
        """(4, 5) target joint angles from precomputed trajectory."""
        return self.R.joint_hist[fi].copy()

    def get_qdot_target(self, fi: int) -> np.ndarray:
        """(4, 5) target joint velocities."""
        return self.R.joint_vel_hist[fi].copy()

    def get_lam_used(self, fi: int) -> np.ndarray:
        """(4, 3) per-leg WBIC-corrected ground reaction force."""
        return self.R.wbic_lam_used[fi].copy()

    @property
    def fail_counts(self) -> dict:
        return dict(self._fail)


__all__ = ["GaitSimControllerStep"]
