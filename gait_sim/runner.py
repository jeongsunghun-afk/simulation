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
from gait_sim.model import (
    LEG_DH, LEG_HIP_OFFSETS, LEG_NAMES, N_JOINTS_PER_LEG, N_JOINTS_MAX,
    Q_HOME_FRONT, Q_HOME_HIND, Q_SWING_FRONT, Q_SWING_HIND,
    Q_HOME_PER_LEG, PHI_PER_LEG, TRAJ_PT_IDX_PER_LEG,
    PHI_FRONT, PHI_HIND, THETA5_FRONT, THETA5_HIND,
    FRONT_Q_LIM, HIND_Q_LIM,
    JOINT_VEL_LIMIT_RAD_S, JOINT_TORQUE_LIMIT, VEL_LIMIT_MARGIN,
    INIT_ERR_RAD, TAU_LAG,
)
from gait_sim.kinematics import (
    forward_kinematics, _dh_to_sim, _sim_to_dh,
    analytical_ik_front, analytical_ik_hind,
    opt_ik_front, opt_ik_hind,
    J4_TO_J5_SIM_PER_LEG,
)
from gait_sim.gait import (
    GaitScheduler, foot_pos_at_phase, _quintic_s,
)
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
# 3) WBIC main loop + NMPC branch + foot world (Phase 6-c/d) — TODO
# ══════════════════════════════════════════════════════════════
def run_simulation(*, gait_type: str = None) -> tuple:
    """전체 시뮬레이션 실행 (Phase 6-e). 현재는 trajectory precompute + derivatives 만.

    Returns:
        (R, meta) — SimState + viz meta dict
    """
    from gait_sim.config import N_FRAMES   # alias 갱신용
    if gait_type:
        CFG.gait_type = gait_type   # caller override (alias 재할당 필요)

    R = SimState.alloc(n_frames=N_FRAMES, dt=DT)
    sched = GaitScheduler()

    precompute_trajectories(R, sched)
    compute_derivatives(R)

    # TODO Phase 6-c/d: WBIC main loop or NMPC branch
    # TODO Phase 6-d:   postprocess_foot_world

    meta = {
        'gait_type': CFG.gait_type, 'V': V, 'T': T, 'D': D,
        'step_height': STEP_HEIGHT, 'step_length': STEP_LENGTH,
        'use_nmpc': CFG.use_nmpc, 'use_mpc': CFG.use_mpc, 'n_mpc': CFG.n_mpc,
        'mu_friction': CFG.mu_friction, 'opt_ik_maxiter': CFG.opt_ik_maxiter,
        'joint_torque_limit': JOINT_TORQUE_LIMIT,
    }
    return R, meta
