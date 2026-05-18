"""gait_sim.runner — Top-level simulation orchestrator.

v13.2 Phase 6: gait_sim_v13.py 의 main loop (line 1707~3414) 를 함수 분할 추출.

함수:
  · precompute_trajectories(R, sched)         opt-IK 루프 → joint_hist/foot_hist/phase 채움
  · compute_derivatives(R)                     joint_vel/acc/jrk + foot_local/vel/acc (gradient or spline)
  · init_wbic_state(R)                         theta_a/dtheta_a 초기화 + body_state dict
  · run_wbic_loop(R, sched, body_state, ...)   MPC + WBIC + body integration main loop (Phase 6-c TODO)
  · run_nmpc_branch(R, sched)                  NMPC + populate_simstate_from_nmpc (Phase 6-d TODO)
  · postprocess_foot_world(R, sched)           foot_actual/target world frame (Phase 6-d TODO)
  · run_simulation(cfg=None)                   전체 통합 entry point (Phase 6-e TODO)

설계:
  · SimState 1개 객체에 모든 sim array write
  · GaitScheduler 인자 받음 (config 변경 가능성 대비)
  · NMPC / WBIC 분기는 cfg.use_nmpc 기준
"""
import math
import time

import numpy as np
from scipy.interpolate import CubicSpline as _CubicSpline

from gait_sim.config import (
    CFG, DT, T, T_SW, T_ST, D, V, N_CYCLES, STEP_LENGTH, STEP_HEIGHT,
    STANCE_DELTA, GAIT_TYPE,
)
from gait_sim.config import G_ACC
from gait_sim.model import (
    LEG_DH, LEG_HIP_OFFSETS, LEG_NAMES, N_JOINTS_PER_LEG, N_JOINTS_MAX,
    Q_HOME_FRONT, Q_HOME_HIND, Q_SWING_FRONT, Q_SWING_HIND,
    Q_HOME_PER_LEG, PHI_PER_LEG, TRAJ_PT_IDX_PER_LEG,
    PHI_FRONT, PHI_HIND, THETA5_FRONT, THETA5_HIND,
    FRONT_Q_LIM, HIND_Q_LIM,
    JOINT_VEL_LIMIT_RAD_S, JOINT_TORQUE_LIMIT, VEL_LIMIT_MARGIN,
    INIT_ERR_RAD, TAU_LAG,
    KP_PD, KD_PD, KP_IMP, KD_IMP,
    BODY_MASS, TOTAL_MASS, BODY_INERTIA, MU_FRICTION, MU_DAMP,
    LINK_MASS_PER_LEG,
)
from gait_sim.kinematics import (
    forward_kinematics, _dh_to_sim, _sim_to_dh,
    analytical_ik_front, analytical_ik_hind,
    opt_ik_front, opt_ik_hind,
    compute_jacobian_sim, compute_gravity_torque_sim,
    J4_TO_J5_SIM_PER_LEG,
)
from gait_sim.dynamics import rnea, compute_mh_leg
from gait_sim.body_dyn import integrate_body_state, _R_to_euler_xyz
from gait_sim.gait import (
    GaitScheduler, foot_pos_at_phase, _quintic_s,
)
from gait_sim.controllers.mpc import (
    mpc_qp_plan, qp_grf_distribute, DT_MPC, N_MPC,
)
from gait_sim.controllers.wbic import (
    wbic_qp_leg, wbic_qp_full,
)
from gait_sim.actuator import apply_actuator_dynamics
from gait_sim.sim_state import SimState


# ══════════════════════════════════════════════════════════════
# 1) Trajectory pre-compute — opt-IK loop (v13.py L1707~2125)
# ══════════════════════════════════════════════════════════════
def precompute_trajectories(R: SimState, sched: GaitScheduler,
                             max_iters: int = None) -> int:
    """opt-IK 기반 foot trajectory pre-compute.

    fills R.joint_hist, R.foot_hist, R.phase_hist + R.opt_ik_*_hist diagnostics.

    Args:
        R:         SimState (alloc 된 빈 객체)
        sched:     GaitScheduler instance
        max_iters: 궤적 재시도 횟수 (None 이면 CFG.max_traj_opt_iters)

    Returns: 사용된 opt_iter 수 (1 ~ max_iters)
    """
    if max_iters is None:
        max_iters = CFG.max_traj_opt_iters

    N_FRAMES = R.n_frames
    body_vel = np.array([V, 0.0, 0.0])
    home_foot_per_leg = [
        _dh_to_sim(
            forward_kinematics(Q_HOME_PER_LEG[leg], dh=LEG_DH[leg])[TRAJ_PT_IDX_PER_LEG[leg]],
            front_leg=(leg < 2)
        )
        for leg in range(4)
    ]

    # FRONT/HIND swing q_ref (opt-IK 시 사용)
    Q_SW_FRONT = list(Q_SWING_FRONT)
    Q_SW_HIND  = list(Q_SWING_HIND)

    use_swing_qref_blend = CFG.use_swing_qref_blend
    opt_ik_use_vel_limit = CFG.opt_ik_use_vel_limit
    opt_ik_use_tau_limit = CFG.opt_ik_use_tau_limit
    lambda_q   = CFG.lambda_q_opt
    lambda_tau = CFG.lambda_tau_opt
    maxiter    = CFG.opt_ik_maxiter

    # FRONT/HIND J4→J5 offset cache (from kinematics module)
    _FRONT_J4_TO_J5_SIM = J4_TO_J5_SIM_PER_LEG[0]
    _HIND_J4_TO_J5_SIM  = J4_TO_J5_SIM_PER_LEG[2]

    traj_scale   = 1.0
    height_scale = 1.0
    opt_iter_used = 0
    stance_dur = T_ST

    print("─" * 55)
    print(f"궤적 계산 중...  [{GAIT_TYPE}]  {N_CYCLES}사이클  {N_FRAMES}프레임")
    print(f"  V={V}m/s  T={T}s  D={D}  T_SW={T_SW:.3f}s  T_ST={T_ST:.3f}s  "
          f"STEP_HEIGHT={STEP_HEIGHT*1e3:.0f}mm  STEP_LENGTH={STEP_LENGTH*1e3:.0f}mm")

    for opt_iter in range(1, max_iters + 1):
        opt_iter_used = opt_iter
        # Reset arrays
        R.joint_hist.fill(0.0)
        R.foot_hist.fill(0.0)
        R.phase_hist.fill(0.0)
        R.frame_calc_time.fill(0.0)
        R.opt_ik_nit_hist.fill(0)
        R.opt_ik_fallback_hist.fill(False)
        R.opt_ik_pos_err_hist.fill(np.nan)

        _step_vec = np.array([STEP_LENGTH * traj_scale, 0.0, 0.0])
        foot_contact    = [
            home_foot_per_leg[leg].copy() + (np.zeros(3) if sched.is_swing(leg, 0) else _step_vec)
            for leg in range(4)
        ]
        foot_sw_start   = [home_foot_per_leg[leg].copy() for leg in range(4)]
        foot_local_prev = [foot_contact[leg].copy() for leg in range(4)]
        prev_swing      = [sched.is_swing(leg, 0) for leg in range(4)]

        # warm-start: phase 0 정확 위치 기반 analytical → opt_ik (vel_limit OFF)
        _saved_vel_limit = opt_ik_use_vel_limit
        opt_ik_use_vel_limit_t = False
        prev_q_per_leg = []
        for leg in range(4):
            front_l = leg < 2
            _phase0 = sched.phase(leg, 0.0)
            _p_end0 = home_foot_per_leg[leg] + _step_vec
            _foot_loc0 = foot_pos_at_phase(
                _phase0, foot_sw_start[leg], foot_contact[leg], _p_end0,
                body_vel * traj_scale,
                swing_ratio=sched.swing_ratio,
                step_height=STEP_HEIGHT * height_scale,
                stance_dur=stance_dur,
            )
            _dh_leg = LEG_DH[leg]
            if front_l:
                _foot_dh0 = _sim_to_dh(_foot_loc0 + _FRONT_J4_TO_J5_SIM, front_leg=True)
                _q_a = analytical_ik_front(_foot_dh0[0], _foot_dh0[1], _foot_dh0[2],
                                            PHI_FRONT, THETA5_FRONT, dh=_dh_leg)
                _q_init0 = list(_q_a) if _q_a is not None else list(Q_HOME_FRONT)
                _q_opt, _, _ = opt_ik_front(_foot_dh0, _q_init0, q_ref=list(Q_HOME_FRONT),
                                             dh=_dh_leg, lambda_q=lambda_q,
                                             lambda_tau=lambda_tau, maxiter=maxiter,
                                             use_vel_limit=opt_ik_use_vel_limit_t,
                                             use_tau_limit=opt_ik_use_tau_limit)
                prev_q_per_leg.append(_q_opt if _q_opt is not None else _q_init0)
            else:
                _foot_dh0 = _sim_to_dh(_foot_loc0 + _HIND_J4_TO_J5_SIM, front_leg=False)
                _q_h = analytical_ik_hind(_foot_dh0[0], _foot_dh0[1], _foot_dh0[2],
                                           PHI_HIND, dh=_dh_leg, theta5_target=THETA5_HIND)
                _q_init0 = (list(_q_h) + [Q_HOME_HIND[4]]) if _q_h is not None else list(Q_HOME_HIND)
                _q_opt, _, _ = opt_ik_hind(_foot_dh0, _q_init0, q_ref=list(Q_HOME_HIND),
                                            dh=_dh_leg, lambda_q=lambda_q,
                                            lambda_tau=lambda_tau, maxiter=maxiter,
                                            use_vel_limit=opt_ik_use_vel_limit_t,
                                            use_tau_limit=opt_ik_use_tau_limit)
                prev_q_per_leg.append(_q_opt if _q_opt is not None else _q_init0)

        calc_start = time.perf_counter()
        for fi in range(N_FRAMES):
            frame_start = time.perf_counter()
            t = fi * DT
            for leg in range(4):
                is_sw = sched.is_swing(leg, t)
                R.phase_hist[fi, leg] = sched.phase(leg, t)

                phase = sched.phase(leg, t)
                p_end = home_foot_per_leg[leg] + np.array([STEP_LENGTH * traj_scale, 0, 0])
                _bv   = body_vel * traj_scale

                if is_sw and not prev_swing[leg]:
                    # stance→swing 전환: 해석적 끝점 (이산화 오차 제거)
                    foot_sw_start[leg] = np.array([
                        foot_contact[leg][0] - _bv[0] * stance_dur,
                        foot_contact[leg][1] - _bv[1] * stance_dur,
                        foot_contact[leg][2],
                    ])
                if not is_sw and prev_swing[leg]:
                    # swing→stance 전환: 해석적 끝점
                    foot_contact[leg] = p_end.copy()
                foot_loc = foot_pos_at_phase(
                    phase, foot_sw_start[leg], foot_contact[leg], p_end,
                    body_vel * traj_scale,
                    swing_ratio=sched.swing_ratio,
                    step_height=STEP_HEIGHT * height_scale,
                    stance_dur=stance_dur)

                foot_local_prev[leg] = foot_loc.copy()
                prev_swing[leg]      = is_sw
                R.foot_hist[fi, leg]   = LEG_HIP_OFFSETS[leg] + foot_loc

                if leg < 2:
                    foot_ik_sim = foot_loc + _FRONT_J4_TO_J5_SIM
                    foot_dh = _sim_to_dh(foot_ik_sim, front_leg=True)
                    col = leg

                    if is_sw and use_swing_qref_blend:
                        _sw_t = sched.swing_t(leg, t)
                        if _sw_t <= 0.5:
                            _alpha = _quintic_s(_sw_t / 0.5)
                        else:
                            _alpha = 1.0 - _quintic_s((_sw_t - 0.5) / 0.5)
                        _q_ref = [h + _alpha * (sw - h)
                                  for h, sw in zip(Q_HOME_FRONT, Q_SW_FRONT)]
                    else:
                        _q_ref = list(Q_HOME_FRONT)
                    q_opt, nit, pos_err_sq = opt_ik_front(
                        foot_dh, prev_q_per_leg[leg][:5], q_ref=_q_ref,
                        dh=LEG_DH[leg], lambda_q=lambda_q, lambda_tau=lambda_tau,
                        maxiter=maxiter,
                        use_vel_limit=opt_ik_use_vel_limit,
                        use_tau_limit=opt_ik_use_tau_limit)
                    R.opt_ik_nit_hist[fi, col]     = nit
                    R.opt_ik_pos_err_hist[fi, col] = pos_err_sq
                    if q_opt is not None:
                        q = q_opt
                    else:
                        R.opt_ik_fallback_hist[fi, col] = True
                        q_ana = analytical_ik_front(foot_dh[0], foot_dh[1], foot_dh[2],
                                                     PHI_FRONT, THETA5_FRONT, dh=LEG_DH[leg])
                        q = q_ana if q_ana is not None else list(Q_HOME_FRONT)

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
                    col = leg

                    _q_ref_h = list(Q_HOME_HIND)
                    q_opt, nit, pos_err_sq = opt_ik_hind(
                        foot_dh, prev_q_per_leg[leg][:5], q_ref=_q_ref_h,
                        dh=LEG_DH[leg], lambda_q=lambda_q, lambda_tau=lambda_tau,
                        maxiter=maxiter,
                        use_vel_limit=opt_ik_use_vel_limit,
                        use_tau_limit=opt_ik_use_tau_limit)
                    R.opt_ik_nit_hist[fi, col]     = nit
                    R.opt_ik_pos_err_hist[fi, col] = pos_err_sq
                    if q_opt is not None:
                        q = q_opt
                    else:
                        R.opt_ik_fallback_hist[fi, col] = True
                        q_h = analytical_ik_hind(foot_dh[0], foot_dh[1], foot_dh[2],
                                                  PHI_HIND, dh=LEG_DH[leg],
                                                  theta5_target=THETA5_HIND)
                        q = (list(q_h) + [Q_HOME_HIND[4]]) if q_h is not None else list(Q_HOME_HIND)
                    pq = prev_q_per_leg[leg]
                    for j in range(len(q)):
                        best = q[j]
                        for off in (-2.0*math.pi, 2.0*math.pi):
                            cand = q[j] + off
                            if abs(cand - pq[j]) < abs(best - pq[j]):
                                best = cand
                        q[j] = best

                nj = N_JOINTS_PER_LEG[leg]
                _vel_dt = JOINT_VEL_LIMIT_RAD_S[:nj] * DT
                _q_arr  = np.array(q[:nj])
                _q_prev = np.array(prev_q_per_leg[leg][:nj])
                _q_arr  = _q_prev + np.clip(_q_arr - _q_prev, -_vel_dt, _vel_dt)
                q[:nj] = list(_q_arr)
                R.joint_hist[fi, leg, :nj] = q[:nj]
                prev_q_per_leg[leg][:nj] = q[:nj]
            R.frame_calc_time[fi] = time.perf_counter() - frame_start

        calc_total = time.perf_counter() - calc_start

        # Velocity check (re-traject if violates limit)
        joint_hist_unwrapped = np.unwrap(R.joint_hist, axis=0)
        if CFG.use_spline_diff:
            _t_grid_jh = np.arange(joint_hist_unwrapped.shape[0]) * DT
            _jh_spline = _CubicSpline(_t_grid_jh, joint_hist_unwrapped, axis=0, bc_type='natural')
            joint_vel_check = _jh_spline(_t_grid_jh, 1)
        else:
            joint_vel_check = np.gradient(joint_hist_unwrapped, DT, axis=0)

        peak_per_joint  = np.max(np.abs(joint_vel_check), axis=(0, 1))
        ratio_per_joint = peak_per_joint / JOINT_VEL_LIMIT_RAD_S
        worst_ratio     = float(np.max(ratio_per_joint))

        if worst_ratio <= VEL_LIMIT_MARGIN:
            break
        scale_decay  = max(0.60, min(0.98 / worst_ratio, 0.98))
        traj_scale  *= scale_decay
        height_scale *= scale_decay

    print(f"궤적 완료. iter={opt_iter_used}  scale={traj_scale:.4f}  calc={calc_total*1e3:.0f}ms")
    return opt_iter_used


# ══════════════════════════════════════════════════════════════
# 2) Derivatives — joint_vel/acc/jrk + foot_local/vel/acc
# ══════════════════════════════════════════════════════════════
def compute_derivatives(R: SimState):
    """joint_hist → joint_vel/acc/jrk + foot_local/vel/acc 채움.

    CFG.use_spline_diff = True 면 CubicSpline 해석미분,
    False 면 np.gradient (default).
    """
    N_FRAMES = R.n_frames
    DT_R = R.dt
    t_grid = np.arange(N_FRAMES) * DT_R
    jh_unwrap = np.unwrap(R.joint_hist, axis=0)

    if CFG.use_spline_diff:
        jh_spline = _CubicSpline(t_grid, jh_unwrap, axis=0, bc_type='natural')
        R.joint_vel_hist[:] = jh_spline(t_grid, 1)
        R.joint_acc_hist[:] = jh_spline(t_grid, 2)
        R.joint_jrk_hist[:] = jh_spline(t_grid, 3)
    else:
        R.joint_vel_hist[:] = np.gradient(jh_unwrap, DT_R, axis=0)
        R.joint_acc_hist[:] = np.gradient(R.joint_vel_hist, DT_R, axis=0)
        R.joint_jrk_hist[:] = np.gradient(R.joint_acc_hist, DT_R, axis=0)

    R.foot_local[:] = R.foot_hist - LEG_HIP_OFFSETS[np.newaxis, :, :]
    R.foot_vel_t[:] = np.gradient(R.foot_local, DT_R, axis=0)
    R.foot_acc_t[:] = np.gradient(R.foot_vel_t,   DT_R, axis=0)


# ══════════════════════════════════════════════════════════════
# 3) WBIC state init — theta_a / dtheta_a + body_state dict
# ══════════════════════════════════════════════════════════════
def init_wbic_state(R: SimState) -> tuple:
    """WBIC main loop 진입 직전 상태 초기화.

    fills R.theta_a_hist, R.dtheta_a_hist (1st-order lag of joint_hist)

    Returns:
        body_state: dict — floating-base 6-DoF 상태 (pos, R, v, omega, ...)
        foot_z_home: float — FR foot z @ home (body z 0-ref 용)
    """
    N_FRAMES = R.n_frames
    DT_R = R.dt
    t_grid = np.arange(N_FRAMES) * DT_R

    # theta_a_hist: 1st-order lag (initial offset = INIT_ERR_RAD)
    for leg in range(4):
        nj = N_JOINTS_PER_LEG[leg]
        R.theta_a_hist[0, leg, :nj] = R.joint_hist[0, leg, :nj] + INIT_ERR_RAD
        for fi in range(1, N_FRAMES):
            prev   = R.theta_a_hist[fi-1, leg, :nj]
            target = R.joint_hist[fi-1, leg, :nj]
            R.theta_a_hist[fi, leg, :nj] = prev + (DT_R / TAU_LAG) * (target - prev)
        # dtheta_a (CubicSpline or gradient)
        if CFG.use_spline_diff:
            _cs = _CubicSpline(t_grid, R.theta_a_hist[:, leg, :nj], axis=0, bc_type='natural')
            R.dtheta_a_hist[:, leg, :nj] = _cs(t_grid, 1)
        else:
            R.dtheta_a_hist[:, leg, :nj] = np.gradient(R.theta_a_hist[:, leg, :nj], DT_R, axis=0)

    # body_state: FR foot @ home → body z = -foot_z_home so foot world z = 0
    foot_z_home = float(R.foot_hist[0, 0, 2])   # FR foot z (body-local, ≈ -0.465m)
    body_state = {
        'pos':   np.array([0.0, 0.0, -foot_z_home]),
        'R':     np.eye(3),
        'v':     np.array([V, 0.0, 0.0]),
        'omega': np.zeros(3),
        'a_lin': np.zeros(3),
        'a_ang': np.zeros(3),
        '_diverged': False,
    }
    return body_state, foot_z_home


# ══════════════════════════════════════════════════════════════
# 4) Main WBIC control loop — MPC + WBIC + body integration
#    v13.py L2876~3206 추출
# ══════════════════════════════════════════════════════════════
# Body reference (steady-state) — used by MPC closed-loop
_BODY_REF_STEP_TEMPLATE = np.array([
    0.0, 0.0, 0.0,   # roll, pitch, yaw
    0.0, 0.0, 0.0,   # px, py, pz
    0.0, 0.0, 0.0,   # ω
    0.0, 0.0, 0.0,   # v
    0.0,             # g (constant)
])


def _step_one_frame(R: SimState, sched: GaitScheduler,
                     body_state: dict, fi: int, foot_z_home: float,
                     swing_flag: np.ndarray) -> tuple:
    """v14.3-d1 Per-frame WBIC + MPC + body integration.

    Mutates R (writes fi-indexed slots) and body_state (in-place integrate).
    Pre-condition: R.theta_a_hist[fi], R.dtheta_a_hist[fi], R.joint_hist[fi],
                    R.joint_vel_hist[fi], R.joint_acc_hist[fi], R.foot_hist[fi],
                    R.foot_local[fi], R.foot_vel_t[fi], R.phase_hist[fi] all populated.
                    body_state holds current body 6-DoF state.
                    swing_flag (N,4) precomputed = R.phase_hist < sched.swing_ratio.
    Post-condition: R.wbc_*[fi], R.wbic_*[fi], R.body_*_hist[fi] filled;
                     body_state updated by integrate_body_state;
                     R.tau_ff_corrected_prev mutated (state carry to next frame).

    Returns: (mpc_fail_inc, wbic_fail_inc, wbic_fb_fail_inc)
    """
    DT_R = R.dt
    USE_MPC             = CFG.use_mpc
    USE_MPC_CLOSED_LOOP = CFG.use_mpc_closed_loop
    USE_WBIC            = CFG.use_wbic
    USE_WBIC_FB         = CFG.use_wbic_fb
    USE_BODY_DYNAMICS   = CFG.use_body_dynamics
    GRF_RAMP_RATIO      = CFG.grf_ramp_ratio
    WBIC_W_DDQ          = CFG.wbic_w_ddq
    WBIC_W_TAU          = CFG.wbic_w_tau
    WBIC_W_LAM          = CFG.wbic_w_lam
    WBIC_W_FB           = CFG.wbic_w_fb
    WBIC_LAMZ_MIN       = CFG.wbic_lamz_min
    WBIC_W_DTAU         = CFG.wbic_w_dtau

    mpc_fail_inc = 0
    wbic_fail_inc = 0
    wbic_fb_fail_inc = 0

    t_cur = fi * DT_R
    contact_mask = ~swing_flag[fi]   # (4,) bool

    # ── GRF 목표 (MPC QP or QP GRF) ─────────────────────
    if USE_MPC:
        cs = np.zeros((N_MPC, 4), dtype=bool)
        fp = np.zeros((N_MPC, 4, 3))
        for k in range(N_MPC):
            t_k = t_cur + k * DT_MPC
            for leg in range(4):
                cs[k, leg] = not sched.is_swing(leg, t_k)
                fp[k, leg] = R.foot_hist[fi, leg]   # quasi-static foot

        if USE_MPC_CLOSED_LOOP and USE_BODY_DYNAMICS:
            roll, pitch, yaw = _R_to_euler_xyz(body_state['R'])
            x0_mpc = np.array([
                roll, pitch, yaw,
                body_state['pos'][0], body_state['pos'][1], body_state['pos'][2],
                body_state['omega'][0], body_state['omega'][1], body_state['omega'][2],
                body_state['v'][0], body_state['v'][1], body_state['v'][2],
                -G_ACC,
            ])
            x_ref_step = _BODY_REF_STEP_TEMPLATE.copy()
            x_ref_step[5]  = -foot_z_home
            x_ref_step[9]  = V
            x_ref_step[12] = -G_ACC
        else:
            x0_mpc = np.array([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                V,   0.0, 0.0,
                -G_ACC,
            ])
            x_ref_step = None

        lam_des_all = mpc_qp_plan(x0_mpc, cs, fp, x_ref_step=x_ref_step,
                                   ltv=USE_MPC_CLOSED_LOOP and USE_BODY_DYNAMICS)
        for leg in range(4):
            if swing_flag[fi, leg]:
                lam_des_all[leg] = 0.0
    else:
        lam_des_all = qp_grf_distribute(contact_mask, R.foot_hist[fi])

    # Contact ramp (stance 시작/끝 GRF smoothstep)
    for leg in range(4):
        if not swing_flag[fi, leg]:
            st_t = sched.stance_t(leg, t_cur)
            if st_t < GRF_RAMP_RATIO:
                tau_r = st_t / GRF_RAMP_RATIO
                ramp = 10*tau_r**3 - 15*tau_r**4 + 6*tau_r**5
                lam_des_all[leg] = lam_des_all[leg] * ramp
            elif st_t > 1.0 - GRF_RAMP_RATIO:
                tau_r = (1.0 - st_t) / GRF_RAMP_RATIO
                ramp = 10*tau_r**3 - 15*tau_r**4 + 6*tau_r**5
                lam_des_all[leg] = lam_des_all[leg] * ramp

    R.wbc_lam_des[fi] = lam_des_all

    # ── Pass 1: per-leg kinematics + RNEA + M, h ─────────
    leg_data = [None] * 4
    foot_world_all = np.zeros((4, 3))
    for leg in range(4):
        nj    = N_JOINTS_PER_LEG[leg]
        front = leg < 2
        dh    = LEG_DH[leg]
        lm    = LINK_MASS_PER_LEG[leg]

        q_t   = R.joint_hist[fi, leg, :nj]
        q_a   = R.theta_a_hist[fi, leg, :nj]
        dq_t  = R.joint_vel_hist[fi, leg, :nj]
        dq_a  = R.dtheta_a_hist[fi, leg, :nj]
        ddq_t = R.joint_acc_hist[fi, leg, :nj]

        J     = compute_jacobian_sim(q_t, dh, front)
        J_a   = compute_jacobian_sim(q_a, dh, front)
        tau_g = compute_gravity_torque_sim(q_t, dh, lm, front)

        lam_des_leg = lam_des_all[leg]
        tau_dyn_leg = rnea(q_t, dq_t, ddq_t, dh, lm)
        tau_grf_leg = -J.T @ lam_des_leg
        tau_ff_leg  = tau_dyn_leg + tau_grf_leg

        # foot world position (body integration용)
        foot_world_all[leg] = body_state['pos'] + body_state['R'] @ R.foot_hist[fi, leg]

        if USE_WBIC or USE_WBIC_FB:
            M_leg, h_leg = compute_mh_leg(q_t, dq_t, dh, lm)
        else:
            M_leg, h_leg = None, None

        foot_t_j5 = R.foot_local[fi, leg] + J4_TO_J5_SIM_PER_LEG[leg]
        pts_a     = forward_kinematics(q_a, dh=dh)
        foot_a_j5 = _dh_to_sim(pts_a[-1], front_leg=front)
        vel_t     = R.foot_vel_t[fi, leg]
        vel_a     = J_a @ dq_a
        f_imp       = KP_IMP * (foot_t_j5 - foot_a_j5) + KD_IMP * (vel_t - vel_a)
        tau_imp_leg = J.T @ f_imp
        tau_pd_leg  = KP_PD[:nj] * (q_t - q_a) + KD_PD[:nj] * (dq_t - dq_a)

        leg_data[leg] = dict(nj=nj, J=J, ddq_t=ddq_t,
                              tau_g=tau_g, tau_dyn=tau_dyn_leg, tau_grf=tau_grf_leg,
                              tau_ff=tau_ff_leg, tau_pd=tau_pd_leg, tau_imp=tau_imp_leg,
                              M_leg=M_leg, h_leg=h_leg,
                              lam_des=lam_des_leg, lam_used=lam_des_leg.copy())

    # ── Pass 2: WBIC (FB single QP OR per-leg) ──────────
    used_fb = False
    if USE_WBIC_FB:
        v_dot_des_fb = np.zeros(6)
        R_world = body_state['R']
        I_world = R_world @ BODY_INERTIA @ R_world.T

        fb_out = wbic_qp_full(
            M_legs=[leg_data[i]['M_leg'] for i in range(4)],
            h_legs=[leg_data[i]['h_leg'] for i in range(4)],
            ddq_des_legs=[leg_data[i]['ddq_t'] for i in range(4)],
            tau_ff_legs=[leg_data[i]['tau_ff'] for i in range(4)],
            lam_des_all=lam_des_all,
            J_legs=[leg_data[i]['J'] for i in range(4)],
            contact_mask=contact_mask, nj_per_leg=N_JOINTS_PER_LEG,
            foot_world_all=foot_world_all, body_pos=body_state['pos'],
            v_dot_des_fb=v_dot_des_fb,
            M_total=TOTAL_MASS, I_body=I_world, omega_world=body_state['omega'],
            w_ddq=WBIC_W_DDQ, w_tau=WBIC_W_TAU, w_lam=WBIC_W_LAM, w_fb=WBIC_W_FB,
            lamz_min=WBIC_LAMZ_MIN, mu=MU_FRICTION,
            stance_foot_J_v11=None,
            tau_prev_legs=R.tau_ff_corrected_prev, w_dtau=WBIC_W_DTAU,
        )
        if fb_out is not None:
            used_fb = True
            R.wbic_fb_residual_hist[fi] = fb_out['residual_fb']
            R.wbic_fb_status_hist[fi]   = True
            R.wbic_fb_dvfb_hist[fi]     = fb_out['d_v_fb']
            for leg in range(4):
                nj = leg_data[leg]['nj']
                d_tau = fb_out['d_tau_legs'][leg]
                d_lam = fb_out['d_lam_legs'][leg]
                leg_data[leg]['tau_ff']   = leg_data[leg]['tau_ff'] + d_tau
                leg_data[leg]['lam_used'] = leg_data[leg]['lam_des'] + d_lam
                R.wbic_dtau_hist[fi, leg, :nj] = d_tau
                R.wbic_dlam_hist[fi, leg]      = d_lam
                R.wbic_residual_hist[fi, leg]  = fb_out['residual_legs'][leg]
                R.wbic_status_hist[fi, leg]    = True

        if not used_fb:
            wbic_fb_fail_inc += 1

    if (not used_fb) and USE_WBIC:
        for leg in range(4):
            d  = leg_data[leg]
            nj = d['nj']
            d_ddq, d_tau, d_lam, ok, res = wbic_qp_leg(
                d['M_leg'], d['h_leg'], d['ddq_t'], d['tau_ff'], d['lam_des'], d['J'],
                contact=contact_mask[leg], nj=nj,
                w_ddq=WBIC_W_DDQ, w_tau=WBIC_W_TAU, w_lam=WBIC_W_LAM,
                lamz_min=WBIC_LAMZ_MIN, mu=MU_FRICTION,
                tau_prev=R.tau_ff_corrected_prev[leg], w_dtau=WBIC_W_DTAU,
            )
            R.wbic_residual_hist[fi, leg] = res
            R.wbic_status_hist[fi, leg]   = ok
            if ok:
                d['tau_ff']  = d['tau_ff'] + d_tau
                d['lam_used'] = d['lam_des'] + d_lam
                R.wbic_dtau_hist[fi, leg, :nj] = d_tau
                R.wbic_dlam_hist[fi, leg]      = d_lam
            else:
                wbic_fail_inc += 1

    # ── Pass 3: τ_cmd + 히스토리 저장 ────────────────────
    USE_ACT = CFG.use_actuator_model
    for leg in range(4):
        d  = leg_data[leg]
        nj = d['nj']
        tau_cmd_leg = d['tau_pd'] + d['tau_ff'] + d['tau_imp']
        tau_cmd_leg = np.clip(tau_cmd_leg, -JOINT_TORQUE_LIMIT[:nj], JOINT_TORQUE_LIMIT[:nj])
        # Actuator dynamics (T-N curve + stiction + viscous) — opt-in
        if USE_ACT:
            dq_leg = R.joint_vel_hist[fi, leg, :nj]
            tau_cmd_leg = apply_actuator_dynamics(
                tau_cmd_leg, dq_leg,
                tau_peak=JOINT_TORQUE_LIMIT[:nj],
                dq_max=JOINT_VEL_LIMIT_RAD_S[:nj],
                tau_static=JOINT_TORQUE_LIMIT[:nj] * CFG.actuator_stiction_ratio,
                b_viscous=CFG.actuator_viscous_b[:nj],
                dq_idle_ratio=CFG.actuator_dq_idle_ratio,
                eps_v=CFG.actuator_eps_v,
            )
        R.wbic_lam_used[fi, leg] = d['lam_used']

        JJT          = d['J'] @ d['J'].T + MU_DAMP * np.eye(3)
        lam_calc_leg = np.linalg.solve(JJT, d['J'] @ (d['tau_g'] - tau_cmd_leg))

        R.wbc_tau_ff  [fi, leg, :nj] = d['tau_ff']
        R.wbc_tau_dyn [fi, leg, :nj] = d['tau_dyn']
        R.wbc_tau_pd  [fi, leg, :nj] = d['tau_pd']
        R.wbc_tau_imp [fi, leg, :nj] = d['tau_imp']
        R.wbc_tau_grf [fi, leg, :nj] = d['tau_grf']
        R.wbc_tau_cmd [fi, leg, :nj] = tau_cmd_leg
        R.wbc_lam_calc[fi, leg]      = lam_calc_leg
        R.tau_ff_corrected_prev[leg] = d['tau_ff'].copy()

    # ── Pass 4: Floating-base body integration ──────────
    if USE_BODY_DYNAMICS:
        integrate_body_state(body_state, R.wbic_lam_used[fi], foot_world_all,
                              TOTAL_MASS, BODY_INERTIA, DT_R)
    # Always save history
    R.body_pos_hist[fi]   = body_state['pos']
    R.body_R_hist[fi]     = body_state['R']
    R.body_v_hist[fi]     = body_state['v']
    R.body_omega_hist[fi] = body_state['omega']
    R.body_alin_hist[fi]  = body_state.get('a_lin', np.zeros(3))
    R.body_aang_hist[fi]  = body_state.get('a_ang', np.zeros(3))
    # Reference (kinematic V·t)
    R.body_pos_ref_hist[fi] = np.array([V * t_cur, 0.0, -foot_z_home])
    R.body_v_ref_hist[fi]   = np.array([V, 0.0, 0.0])

    return mpc_fail_inc, wbic_fail_inc, wbic_fb_fail_inc


def run_wbic_loop(R: SimState, sched: GaitScheduler,
                   body_state: dict, foot_z_home: float) -> dict:
    """WBIC + MPC main loop (v13.py L2876~3206).

    v14.3-d1: inner per-frame body extracted to _step_one_frame() — same behavior,
    enables external (Isaac Lab / Mujoco / RL) per-step controller invocation.

    Returns: dict — diagnostic counters {mpc_fail, wbic_fail, wbic_fb_fail, duration_ms}
    """
    N_FRAMES = R.n_frames
    t0 = time.perf_counter()
    mpc_fail = wbic_fail = wbic_fb_fail = 0

    # swing_flag precomputed once for full horizon
    swing_flag = (R.phase_hist < sched.swing_ratio)

    for fi in range(N_FRAMES):
        m, w, fb = _step_one_frame(R, sched, body_state, fi, foot_z_home, swing_flag)
        mpc_fail     += m
        wbic_fail    += w
        wbic_fb_fail += fb

    duration = time.perf_counter() - t0
    diag = dict(mpc_fail=mpc_fail, wbic_fail=wbic_fail, wbic_fb_fail=wbic_fb_fail,
                duration_ms=duration*1e3)
    return diag


# ══════════════════════════════════════════════════════════════
# 5) Foot world post-process (fig6 용)
# ══════════════════════════════════════════════════════════════
def postprocess_foot_world(R: SimState, sched: GaitScheduler):
    """foot_actual_world_hist (body + R @ foot_local) + foot_target_world_hist (swing target).

    NMPC 모드는 nmpc.populate_simstate_from_nmpc 가 직접 채움 → 호출 불필요.
    """
    N_FRAMES = R.n_frames
    DT_R = R.dt

    # Initial home world per-leg (used as swing trajectory start anchor)
    foot_home_world = np.zeros((4, 3))
    # (4,3) @ (3,3).T = (4,3): row i = R_b @ foot_hist[0, i]
    R.foot_actual_world_hist[0] = R.body_pos_hist[0] + R.foot_hist[0] @ R.body_R_hist[0].T
    for li in range(4):
        foot_home_world[li] = R.foot_actual_world_hist[0, li]

    for fi in range(N_FRAMES):
        body_p = R.body_pos_hist[fi]
        R_b    = R.body_R_hist[fi]
        for li in range(4):
            R.foot_actual_world_hist[fi, li] = body_p + R_b @ R.foot_hist[fi, li]

        t_now = fi * DT_R
        for li in range(4):
            ph = sched.phase(li, t_now)
            if ph < sched.swing_ratio:
                sw_t = ph / sched.swing_ratio
                t_sw_start = t_now - ph * T
                t_sw_end   = t_sw_start + T_SW
                ps = foot_home_world[li].copy()
                ps[0] += V * t_sw_start - STEP_LENGTH / 2
                pe = foot_home_world[li].copy()
                pe[0] += V * t_sw_end + STEP_LENGTH / 2
                s_xy = sw_t * sw_t * (3.0 - 2.0 * sw_t)
                tgt  = ps + s_xy * (pe - ps)
                tgt[2] = ps[2] + STEP_HEIGHT * 16.0 * sw_t * sw_t * (1 - sw_t) * (1 - sw_t)
                R.foot_target_world_hist[fi, li] = tgt
            else:
                R.foot_target_world_hist[fi, li] = np.nan


# ══════════════════════════════════════════════════════════════
# 6) Top-level orchestrator
# ══════════════════════════════════════════════════════════════
def run_simulation(*, gait_type: str = None,
                    use_nmpc: bool = None) -> tuple:
    """전체 시뮬레이션 실행 (Phase 6-e).

    Args:
        gait_type: optional override CFG.gait_type
        use_nmpc:  optional override CFG.use_nmpc

    Returns:
        (R, meta) — SimState + viz meta dict
    """
    from gait_sim.config import N_FRAMES
    if gait_type:
        CFG.gait_type = gait_type
    if use_nmpc is not None:
        CFG.use_nmpc = use_nmpc

    R = SimState.alloc(n_frames=N_FRAMES, dt=DT)
    sched = GaitScheduler()

    # 1) Trajectory precompute + derivatives
    precompute_trajectories(R, sched)
    compute_derivatives(R)

    nmpc_active = False
    if CFG.use_nmpc:
        from gait_sim.controllers.nmpc import (
            _CROCODDYL_AVAILABLE, solve_nmpc_one_shot, solve_nmpc_receding,
            populate_simstate_from_nmpc,
        )
        if _CROCODDYL_AVAILABLE:
            print("─" * 55)
            print("v12: NMPC ({}horizon FDDP) 풀이 시작...".format(
                'receding ' if CFG.use_nmpc_receding else 'one-shot '))
            if CFG.use_nmpc_receding:
                xs, us, forces, done, pin_m, pin_d = solve_nmpc_receding(sched)
            else:
                xs, us, forces, done, pin_m, pin_d = solve_nmpc_one_shot(sched)
            if done:
                foot_z_home = float(R.foot_hist[0, 0, 2])
                populate_simstate_from_nmpc(R, xs, us, forces, pin_m, pin_d,
                                              foot_z_home=-foot_z_home)
                nmpc_active = True
                R.mode_str = 'NMPC (FDDP)'
                print("  v11 arrays NMPC 결과로 채움. WBC + MPC 메인 루프 SKIP.")
            else:
                print("  ⚠ NMPC 수렴 실패 — v11 동작 (MPC+WBIC) fallback")
        else:
            print("  ⚠ crocoddyl 미설치 — v11 동작 fallback")

    # 2) WBIC main loop (NMPC 활성 아니면)
    if not nmpc_active:
        body_state, foot_z_home = init_wbic_state(R)
        diag = run_wbic_loop(R, sched, body_state, foot_z_home)
        mode = f"MPC(N={N_MPC},dt={DT_MPC*1e3:.0f}ms)" if CFG.use_mpc else "QP GRF"
        wbic = "WBIC ON" if CFG.use_wbic else "WBIC OFF"
        print(f"WBC 완료 [{mode}, {wbic}].  {diag['duration_ms']:.1f}ms 총  "
              f"({diag['duration_ms']*1e3/R.n_frames:.1f}μs/frame)")
        R.mode_str = mode
        R.wbic_str = wbic
        # foot world post-process
        postprocess_foot_world(R, sched)

    R.diverged = body_state['_diverged'] if not nmpc_active else False

    meta = {
        'gait_type': CFG.gait_type, 'V': V, 'T': T, 'D': D,
        'step_height': STEP_HEIGHT, 'step_length': STEP_LENGTH,
        'use_nmpc': nmpc_active, 'use_mpc': CFG.use_mpc, 'n_mpc': CFG.n_mpc,
        'mu_friction': CFG.mu_friction, 'opt_ik_maxiter': CFG.opt_ik_maxiter,
        'joint_torque_limit': JOINT_TORQUE_LIMIT, 'total_mass': TOTAL_MASS,
        'g_acc': G_ACC,
    }
    return R, meta
