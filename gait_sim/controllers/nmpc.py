"""gait_sim.controllers.nmpc — crocoddyl FDDP NMPC trajectory generator.

v13.2 Phase 4-d: gait_sim_v13.py 의 NMPC 영역 (line 2237~2858) 추출.

함수:
  · solve_nmpc_one_shot(sched)              one-shot full-horizon FDDP solve
  · solve_nmpc_receding(sched)              receding horizon FDDP (warm-start)
  · populate_simstate_from_nmpc(R, ...)     NMPC trajectory → SimState slot fill

설계 원칙:
  · crocoddyl/pinocchio/build_pin_model 는 module top 에서 try-import (없으면 _CROCODDYL_AVAILABLE=False)
  · 모든 NMPC weight / DT_NMPC / horizon → gait_sim.config.CFG 에서 직접 읽음
  · sched (GaitScheduler), perturbation 같은 run-time 상태는 함수 인자
  · 결과: xs (state traj), us (control traj), forces (per-step GRF dict), done flag, pin_model/data

Returns from solve_nmpc_*:
  (xs, us, forces, success, pin_model, pin_data) 튜플 — crocoddyl/pinocchio 미설치 시 (None,..,False,..)
"""
import time
from typing import Optional

import numpy as np

# ── External dependencies (optional) ─────────────────────────
try:
    import crocoddyl as _crocoddyl
    import build_pin_model as _bm
    _CROCODDYL_AVAILABLE = True
except ImportError:
    _crocoddyl = None
    _bm = None
    _CROCODDYL_AVAILABLE = False

# ── gait_sim modules ─────────────────────────────────────────
from gait_sim.config import (
    CFG, DT, T, T_SW, V, N_CYCLES, STEP_LENGTH, STEP_HEIGHT, G_ACC,
)
from gait_sim.model import (
    Q_HOME_FRONT, Q_HOME_HIND, JOINT_TORQUE_LIMIT, MU_FRICTION,
    LEG_DH, LEG_HIP_OFFSETS, LINK_MASS_PER_LEG, N_JOINTS_PER_LEG, LEG_NAMES,
)
from gait_sim.kinematics import (
    forward_kinematics, _dh_to_sim,
    compute_jacobian_sim, compute_gravity_torque_sim,
)
from gait_sim.controllers.mpc import DT_MPC
from gait_sim.sim_state import SimState


# ══════════════════════════════════════════════════════════════
# Cycloid swing target (smoothstep XY + bell Z)
# ══════════════════════════════════════════════════════════════
def _cycloid(p_start, p_end, sw_t, h):
    """Swing target trajectory:
        XY: smoothstep s(t) = t²(3-2t)   (v=0 at t=0,1)
        Z : 16·t²·(1-t)²                  (v=0 at t=0,1, peak h at t=0.5)
    """
    s_xy = sw_t * sw_t * (3.0 - 2.0 * sw_t)
    p = p_start + s_xy * (p_end - p_start)
    p[2] = p_start[2] + h * 16.0 * sw_t * sw_t * (1.0 - sw_t) * (1.0 - sw_t)
    return p


# ══════════════════════════════════════════════════════════════
# Internal helper — pin_model + initial state setup (공통)
# ══════════════════════════════════════════════════════════════
def _build_pin_state():
    """build_pin_model + initial home-pose foot-on-ground state."""
    import pinocchio as pin
    pin_model = _bm.build_model()
    pin_data  = pin_model.createData()
    cstate     = _crocoddyl.StateMultibody(pin_model)
    cactuation = _crocoddyl.ActuationModelFloatingBase(cstate)

    q_home_per_leg = {'FR': Q_HOME_FRONT, 'FL': Q_HOME_FRONT,
                      'HR': Q_HOME_HIND,  'HL': Q_HOME_HIND}
    q0 = pin.neutral(pin_model)
    for leg, qh in q_home_per_leg.items():
        for i, qi in enumerate(qh):
            jid = pin_model.getJointId(f'leg_{leg}_j{i+1}')
            q0[pin_model.idx_qs[jid]] = qi
    pin.forwardKinematics(pin_model, pin_data, q0)
    pin.updateFramePlacements(pin_model, pin_data)
    foot_z_native = pin_data.oMf[pin_model.getFrameId('leg_FR_foot')].translation[2]
    q0[2] = -foot_z_native
    v0 = np.zeros(pin_model.nv)
    x0 = np.concatenate([q0, v0])

    foot_frames_pin = {leg: pin_model.getFrameId(f'leg_{leg}_foot')
                        for leg in ['FR', 'FL', 'HR', 'HL']}
    pin.forwardKinematics(pin_model, pin_data, q0)
    pin.updateFramePlacements(pin_model, pin_data)
    foot_home_pin = {leg: pin_data.oMf[foot_frames_pin[leg]].translation.copy()
                     for leg in ['FR', 'FL', 'HR', 'HL']}

    # Joint torque limit array (cactuation.nu 차원, idx_vs 매핑 반영)
    tau_lim_full = np.zeros(cactuation.nu)
    for leg in ['FR', 'FL', 'HR', 'HL']:
        for i in range(5):
            u_idx = pin_model.idx_vs[pin_model.getJointId(f'leg_{leg}_j{i+1}')] - 6
            tau_lim_full[u_idx] = JOINT_TORQUE_LIMIT[i]

    return (pin_model, pin_data, cstate, cactuation,
            q0, v0, x0, foot_frames_pin, foot_home_pin,
            tau_lim_full)


def _build_action(cstate, cactuation, pin_model, foot_frames_pin, foot_home_pin,
                  x0, tau_lim_full, sched, t, dt_nmpc):
    """단일 시점 t 의 IntegratedActionModel 빌드 (one-shot / receding 공통)."""
    import pinocchio as pin
    _LEGS = ['FR', 'FL', 'HR', 'HL']
    stance, swing = [], []
    swing_info = {}
    for leg_idx, leg in enumerate(_LEGS):
        ph = sched.phase(leg_idx, t)
        if ph < sched.swing_ratio:
            sw_t = ph / sched.swing_ratio
            t_sw_start = t - ph * T
            t_sw_end   = t_sw_start + T_SW
            swing.append(leg)
            swing_info[leg] = (sw_t, t_sw_start, t_sw_end)
        else:
            stance.append(leg)

    cm_contact = _crocoddyl.ContactModelMultiple(cstate, cactuation.nu)
    for leg in stance:
        c = _crocoddyl.ContactModel3D(
            cstate, foot_frames_pin[leg], np.zeros(3),
            pin.LOCAL_WORLD_ALIGNED, cactuation.nu,
            np.array([CFG.nmpc_baumgarte_kp, CFG.nmpc_baumgarte_kd]))
        cm_contact.addContact(f'c_{leg}', c)
    cost = _crocoddyl.CostModelSum(cstate, cactuation.nu)
    cost.addCost('stateReg',
        _crocoddyl.CostModelResidual(cstate,
            _crocoddyl.ResidualModelState(cstate, x0, cactuation.nu)),
        CFG.nmpc_w_state_reg)
    cost.addCost('ctrlReg',
        _crocoddyl.CostModelResidual(cstate,
            _crocoddyl.ResidualModelControl(cstate, cactuation.nu)),
        CFG.nmpc_w_ctrl_reg)
    # Friction cone soft barrier (stance only)
    for leg in stance:
        fc = _crocoddyl.FrictionCone(np.eye(3), MU_FRICTION, CFG.nmpc_fric_nf,
                                     True, 0.0, CFG.nmpc_fric_fz_max)
        fc_act = _crocoddyl.ActivationModelQuadraticBarrier(
            _crocoddyl.ActivationBounds(fc.lb, fc.ub))
        fc_res = _crocoddyl.ResidualModelContactFrictionCone(
            cstate, foot_frames_pin[leg], fc, cactuation.nu, True)
        cost.addCost(f'fric_{leg}',
            _crocoddyl.CostModelResidual(cstate, fc_act, fc_res),
            CFG.nmpc_w_friction)
    # Contact force regularization (xy weighted higher)
    for leg in stance:
        fref = pin.Force(np.zeros(6))
        f_act = _crocoddyl.ActivationModelWeightedQuad(
            np.array([CFG.nmpc_w_force_xy, CFG.nmpc_w_force_xy, CFG.nmpc_w_force_z]))
        f_res = _crocoddyl.ResidualModelContactForce(
            cstate, foot_frames_pin[leg], fref, 3, cactuation.nu, True)
        cost.addCost(f'freg_{leg}',
            _crocoddyl.CostModelResidual(cstate, f_act, f_res),
            CFG.nmpc_w_force_reg)
    # Joint torque limit barrier
    tau_act = _crocoddyl.ActivationModelQuadraticBarrier(
        _crocoddyl.ActivationBounds(-tau_lim_full, +tau_lim_full))
    tau_res = _crocoddyl.ResidualModelControl(cstate, cactuation.nu)
    cost.addCost('tau_lim',
        _crocoddyl.CostModelResidual(cstate, tau_act, tau_res),
        CFG.nmpc_w_tau_lim)
    # Swing foot tracking
    for leg in swing:
        sw_t, t_sw_start, t_sw_end = swing_info[leg]
        ps = foot_home_pin[leg].copy()
        ps[0] += V * t_sw_start - STEP_LENGTH / 2
        pe = foot_home_pin[leg].copy()
        pe[0] += V * t_sw_end + STEP_LENGTH / 2
        tgt = _cycloid(ps, pe, sw_t, STEP_HEIGHT)
        res = _crocoddyl.ResidualModelFrameTranslation(
            cstate, foot_frames_pin[leg], tgt, cactuation.nu)
        act = _crocoddyl.ActivationModelWeightedQuad(
            np.array([CFG.nmpc_w_track_xy, CFG.nmpc_w_track_xy, CFG.nmpc_w_track_z]))
        cost.addCost(f'foot_{leg}',
            _crocoddyl.CostModelResidual(cstate, act, res), 1.0)
        # Pre-touchdown velocity penalty
        if sw_t >= max(0.0, 1.0 - CFG.nmpc_touchdown_last_n * dt_nmpc / max(T_SW, 1e-6)):
            v_ref = pin.Motion(np.zeros(6))
            v_res = _crocoddyl.ResidualModelFrameVelocity(
                cstate, foot_frames_pin[leg], v_ref,
                pin.LOCAL_WORLD_ALIGNED, cactuation.nu)
            v_act = _crocoddyl.ActivationModelWeightedQuad(
                np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
            cost.addCost(f'tdv_{leg}',
                _crocoddyl.CostModelResidual(cstate, v_act, v_res),
                CFG.nmpc_w_touchdown_v)
    # Stance leg foot world-frame velocity → 0 (slip 방지)
    for leg in stance:
        v_ref = pin.Motion(np.zeros(6))
        v_res = _crocoddyl.ResidualModelFrameVelocity(
            cstate, foot_frames_pin[leg], v_ref,
            pin.LOCAL_WORLD_ALIGNED, cactuation.nu)
        v_act = _crocoddyl.ActivationModelWeightedQuad(
            np.array([CFG.nmpc_w_stance_pos_xy, CFG.nmpc_w_stance_pos_xy,
                      CFG.nmpc_w_stance_pos_z, 0.0, 0.0, 0.0]))
        cost.addCost(f'svel_{leg}',
            _crocoddyl.CostModelResidual(cstate, v_act, v_res), 1.0)
    diff = _crocoddyl.DifferentialActionModelContactFwdDynamics(
        cstate, cactuation, cm_contact, cost, 0.0, True)
    return _crocoddyl.IntegratedActionModelEuler(diff, dt_nmpc)


def _build_terminal(cstate, cactuation, foot_frames_pin, target_x):
    """Terminal action (all feet stance + state ref)."""
    import pinocchio as pin
    cm_T = _crocoddyl.ContactModelMultiple(cstate, cactuation.nu)
    for leg in ['FR', 'FL', 'HR', 'HL']:
        c = _crocoddyl.ContactModel3D(
            cstate, foot_frames_pin[leg], np.zeros(3),
            pin.LOCAL_WORLD_ALIGNED, cactuation.nu,
            np.array([CFG.nmpc_baumgarte_kp, CFG.nmpc_baumgarte_kd]))
        cm_T.addContact(f'c_{leg}', c)
    cost_T = _crocoddyl.CostModelSum(cstate, cactuation.nu)
    cost_T.addCost('stateReg_T',
        _crocoddyl.CostModelResidual(cstate,
            _crocoddyl.ResidualModelState(cstate, target_x, cactuation.nu)),
        CFG.nmpc_w_terminal)
    diff_T = _crocoddyl.DifferentialActionModelContactFwdDynamics(
        cstate, cactuation, cm_T, cost_T, 0.0, True)
    return _crocoddyl.IntegratedActionModelEuler(diff_T, 0.0)


def _extract_forces(solver, n_steps):
    """크로코딜 솔버의 contact GRF 추출 (per-step per-leg world-frame Fxyz)."""
    forces = []
    for k in range(n_steps):
        f_step = {leg: np.zeros(3) for leg in ['FR', 'FL', 'HR', 'HL']}
        try:
            ad_k = solver.problem.runningDatas[k]
            contacts = ad_k.differential.multibody.contacts.contacts.todict()
            for name, cdata in contacts.items():
                leg = name.replace('c_', '')
                if leg in f_step:
                    f_step[leg] = np.array(cdata.f.linear).copy()
        except Exception:
            pass
        forces.append(f_step)
    return forces


# ══════════════════════════════════════════════════════════════
# One-shot full-horizon NMPC
# ══════════════════════════════════════════════════════════════
def solve_nmpc_one_shot(sched):
    """crocoddyl FDDP 로 trot trajectory 풀이 (one-shot 전체 horizon).

    Returns: (xs, us, forces, done, pin_model, pin_data)
             — _CROCODDYL_AVAILABLE 가 False 면 (None, None, None, False, None, None)
    """
    if not _CROCODDYL_AVAILABLE:
        print("  ⚠ crocoddyl 미설치 — USE_NMPC 비활성")
        return None, None, None, False, None, None

    (pin_model, pin_data, cstate, cactuation,
     q0, v0, x0, foot_frames_pin, foot_home_pin,
     tau_lim_full) = _build_pin_state()

    N_PER_CYCLE = max(2, round(T / DT_MPC))
    dt_nmpc = T / N_PER_CYCLE
    N_TOTAL = N_PER_CYCLE * N_CYCLES

    actions = []
    for k in range(N_TOTAL):
        t = k * dt_nmpc
        actions.append(_build_action(
            cstate, cactuation, pin_model, foot_frames_pin, foot_home_pin,
            x0, tau_lim_full, sched, t, dt_nmpc))

    x_ref_T = x0.copy()
    x_ref_T[0] = q0[0] + V * T * N_CYCLES
    terminal = _build_terminal(cstate, cactuation, foot_frames_pin, x_ref_T)

    problem = _crocoddyl.ShootingProblem(x0, actions, terminal)
    solver = _crocoddyl.SolverFDDP(problem)
    xs_init = [x0] * (N_TOTAL + 1)
    us_init = [np.zeros(cactuation.nu)] * N_TOTAL
    print(f"  NMPC FDDP 풀이 중 ({N_TOTAL} steps × DT={dt_nmpc*1e3:.1f}ms = "
          f"{N_TOTAL*dt_nmpc:.2f}s)...")
    t_solve = time.time()
    done = solver.solve(xs_init, us_init, maxiter=CFG.nmpc_maxiter,
                        is_feasible=False, init_reg=CFG.nmpc_init_reg)
    elapsed = time.time() - t_solve
    print(f"  NMPC: done={done}, iters={solver.iter}, "
          f"time={elapsed*1e3:.0f}ms, cost={solver.cost:.2e}")

    forces = _extract_forces(solver, N_TOTAL)
    return np.array(solver.xs), np.array(solver.us), forces, done, pin_model, pin_data


# ══════════════════════════════════════════════════════════════
# Receding horizon NMPC (warm-start)
# ══════════════════════════════════════════════════════════════
def solve_nmpc_receding(sched, perturb: Optional[dict] = None):
    """Receding horizon NMPC.
    매 CFG.nmpc_rh_n_resolve step 마다 CFG.nmpc_rh_n_horizon 길이 NMPC 풀이 (warm-start).
    one-shot 4+ cycle 발산 회피 — 짧은 horizon × 다회 풀이.

    Args:
        sched:   GaitScheduler
        perturb: dict (optional) — {'time': float, 'v_lin': np.array(3), 'v_ang': np.array(3)}
                  주어지면 t >= time 도달 시 body velocity 에 impulse 추가.

    Returns: (xs_full, us_full, forces_full, success, pin_model, pin_data)
    """
    if not _CROCODDYL_AVAILABLE:
        return None, None, None, False, None, None

    (pin_model, pin_data, cstate, cactuation,
     q0, v0, x0, foot_frames_pin, foot_home_pin,
     tau_lim_full) = _build_pin_state()

    N_PER_CYCLE = max(2, round(T / DT_MPC))
    dt_nmpc = T / N_PER_CYCLE
    N_TOTAL = N_PER_CYCLE * N_CYCLES

    N_HORIZON = CFG.nmpc_rh_n_horizon
    N_RESOLVE = CFG.nmpc_rh_n_resolve
    print(f"  Receding horizon NMPC: N_HORIZON={N_HORIZON} "
          f"({N_HORIZON*dt_nmpc:.2f}s), N_RESOLVE={N_RESOLVE}, N_TOTAL={N_TOTAL}")

    xs_full = [x0.copy()]
    us_full = []
    forces_full = []
    x_current = x0.copy()
    xs_warm = None
    us_warm = None

    t_solve_total = time.time()
    n_solves = 0
    n_failures = 0
    iter_total = 0
    fi_nmpc = 0
    perturb_done = False
    while fi_nmpc < N_TOTAL:
        t_now = fi_nmpc * dt_nmpc
        if perturb and (not perturb_done) and t_now >= perturb['time']:
            x_current[pin_model.nq    : pin_model.nq + 3] += perturb['v_lin']
            x_current[pin_model.nq + 3: pin_model.nq + 6] += perturb['v_ang']
            print(f"  [PERTURB] t={t_now:.2f}s: body v += {perturb['v_lin']}, "
                  f"ω += {perturb['v_ang']}")
            perturb_done = True

        rem = N_TOTAL - fi_nmpc
        h_eff = min(N_HORIZON, rem)
        actions_h = [
            _build_action(cstate, cactuation, pin_model, foot_frames_pin,
                          foot_home_pin, x0, tau_lim_full, sched,
                          (fi_nmpc + k) * dt_nmpc, dt_nmpc)
            for k in range(h_eff)
        ]
        target_x_T = x0.copy()
        target_x_T[0] = q0[0] + V * (fi_nmpc + h_eff) * dt_nmpc
        terminal = _build_terminal(cstate, cactuation, foot_frames_pin, target_x_T)

        problem = _crocoddyl.ShootingProblem(x_current, actions_h, terminal)
        solver = _crocoddyl.SolverFDDP(problem)

        xs_init = [x_current] * (h_eff + 1)
        us_init = [np.zeros(cactuation.nu)] * h_eff
        if us_warm is not None:
            for k in range(min(len(us_warm) - N_RESOLVE, h_eff)):
                us_init[k] = np.array(us_warm[N_RESOLVE + k])

        done = solver.solve(xs_init, us_init,
                            maxiter=CFG.nmpc_maxiter,
                            is_feasible=False,
                            init_reg=CFG.nmpc_init_reg)
        n_solves += 1
        iter_total += solver.iter
        if not done:
            n_failures += 1

        n_apply = min(N_RESOLVE, h_eff)
        forces_step = _extract_forces(solver, n_apply)
        for k in range(n_apply):
            xs_full.append(np.array(solver.xs[k + 1]))
            us_full.append(np.array(solver.us[k]))
            forces_full.append(forces_step[k])
        x_current = np.array(solver.xs[n_apply])
        xs_warm = np.array(solver.xs)
        us_warm = np.array(solver.us)
        fi_nmpc += n_apply

    elapsed = time.time() - t_solve_total
    print(f"  Receding horizon 완료: {n_solves} solves, total {iter_total} iters, "
          f"{n_failures} fails, {elapsed*1e3:.0f}ms")

    success = (n_solves - n_failures) >= 1
    if n_failures > 0:
        print(f"    [INFO] {n_failures}/{n_solves} solves failed line search but partial "
              f"trajectories used (FDDP enforces dynamics).")
    return np.array(xs_full), np.array(us_full), forces_full, success, pin_model, pin_data


# ══════════════════════════════════════════════════════════════
# NMPC trajectory → SimState slots
# ══════════════════════════════════════════════════════════════
def populate_simstate_from_nmpc(R: SimState, xs, us, forces, pin_model, pin_data,
                                 foot_z_home: float):
    """NMPC xs/us/forces → SimState 의 v11 frame-rate array 채움 (linear interp).

    Args:
        R:           SimState (allocated)
        xs, us:      NMPC trajectory (state, control)
        forces:      list[dict[leg → Fxyz]] (각 step)
        pin_model:   pinocchio Model (NMPC solve 시 빌드된 것)
        pin_data:    pinocchio Data
        foot_z_home: body z 0-ref (= home pose 시의 발 z 절댓값) — body_pos_ref_hist[:, 2] 용
    """
    import pinocchio as pin
    N_FRAMES = R.n_frames
    DT_R = R.dt

    N_NMPC = len(us)
    # v13.py: DT_NMPC = T / 2 / int((T/2)/DT_MPC) — robust 형태로 환산
    dt_nmpc = T / 2.0 / int((T / 2.0) / DT_MPC)
    t_nmpc = np.arange(N_NMPC + 1) * dt_nmpc

    leg_q_idx_pin = {leg: [pin_model.idx_qs[pin_model.getJointId(f'leg_{leg}_j{i+1}')]
                            for i in range(5)]
                     for leg in ['FR', 'FL', 'HR', 'HL']}
    leg_v_idx_pin = {leg: [pin_model.idx_vs[pin_model.getJointId(f'leg_{leg}_j{i+1}')]
                            for i in range(5)]
                     for leg in ['FR', 'FL', 'HR', 'HL']}

    # Linear interp from t_nmpc → t_v11 (R frame rate)
    for fi in range(N_FRAMES):
        t_v11 = fi * DT_R
        if t_v11 >= t_nmpc[-1]:
            k = N_NMPC - 1
            alpha = 1.0
        else:
            k = int(t_v11 / dt_nmpc)
            alpha = (t_v11 - k * dt_nmpc) / dt_nmpc
            k = min(k, N_NMPC - 1)
        # State interp
        if k < N_NMPC:
            x = xs[k] * (1 - alpha) + xs[k + 1] * alpha
        else:
            x = xs[-1]
        u = us[min(k, N_NMPC - 1)]

        # Joint state per leg (v11 ordering)
        for leg_idx, leg in enumerate(['FR', 'FL', 'HR', 'HL']):
            for j in range(5):
                R.joint_hist[fi, leg_idx, j] = x[leg_q_idx_pin[leg][j]]

        # Body state
        R.body_pos_hist[fi]   = x[0:3]
        qx, qy, qz, qw = x[3], x[4], x[5], x[6]
        R.body_R_hist[fi]     = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        R.body_v_hist[fi]     = x[pin_model.nq : pin_model.nq + 3]
        R.body_omega_hist[fi] = x[pin_model.nq + 3 : pin_model.nq + 6]
        R.body_pos_ref_hist[fi] = np.array([V * t_v11, 0.0, -foot_z_home])
        R.body_v_ref_hist[fi]   = np.array([V, 0.0, 0.0])

        # Control (per-leg)
        for leg_idx, leg in enumerate(['FR', 'FL', 'HR', 'HL']):
            for j in range(5):
                u_idx = leg_v_idx_pin[leg][j] - 6
                R.wbc_tau_cmd[fi, leg_idx, j] = u[u_idx]

        # GRF per leg
        if forces is not None and len(forces) > 0:
            k_force = min(int(t_v11 / dt_nmpc), len(forces) - 1)
            f_dict = forces[k_force]
            for leg_idx, leg in enumerate(['FR', 'FL', 'HR', 'HL']):
                R.wbc_lam_des[fi, leg_idx]    = f_dict[leg]
                R.wbic_lam_used[fi, leg_idx]  = f_dict[leg]

    # foot_hist (body-frame) 재계산 (joint_hist 기반 FK)
    for fi in range(N_FRAMES):
        for leg_idx in range(4):
            nj = N_JOINTS_PER_LEG[leg_idx]
            q_leg = R.joint_hist[fi, leg_idx, :nj]
            pts_dh = forward_kinematics(q_leg, dh=LEG_DH[leg_idx])
            foot_local_sim = _dh_to_sim(pts_dh[-1], front_leg=(leg_idx < 2))
            R.foot_hist[fi, leg_idx] = LEG_HIP_OFFSETS[leg_idx] + foot_local_sim

    # foot world (actual) + swing target (cmd)
    leg_to_idx = {'FR': 0, 'FL': 1, 'HR': 2, 'HL': 3}
    for fi in range(N_FRAMES):
        body_p = R.body_pos_hist[fi]
        R_b    = R.body_R_hist[fi]
        for leg_idx in range(4):
            R.foot_actual_world_hist[fi, leg_idx] = body_p + R_b @ R.foot_hist[fi, leg_idx]

    for fi in range(N_FRAMES):
        t = fi * DT_R
        in_cycle_t = t % T
        in_phase_A = in_cycle_t < (T / 2)
        if in_phase_A:
            swing_legs = ['FR', 'HL']
            t_in_phase = in_cycle_t
            phase_start_t = (t // T) * T
        else:
            swing_legs = ['FL', 'HR']
            t_in_phase = in_cycle_t - T / 2
            phase_start_t = (t // T) * T + T / 2
        sw_t = max(0.0, min(1.0, t_in_phase / (T / 2)))
        for leg in swing_legs:
            li = leg_to_idx[leg]
            ps = R.foot_actual_world_hist[0, li].copy()
            ps[0] += V * phase_start_t - STEP_LENGTH / 2
            pe = R.foot_actual_world_hist[0, li].copy()
            pe[0] += V * (phase_start_t + T / 2) + STEP_LENGTH / 2
            s_xy = sw_t * sw_t * (3.0 - 2.0 * sw_t)
            tgt  = ps + s_xy * (pe - ps)
            tgt[2] = ps[2] + STEP_HEIGHT * 16.0 * sw_t * sw_t * (1 - sw_t) * (1 - sw_t)
            R.foot_target_world_hist[fi, li] = tgt

    # τ decomposition 후처리 (fig3/fig4 시각화용)
    # tau_grf = -Jᵀ × R_body^T × λ_world  (body-local force)
    # tau_dyn = compute_gravity_torque_sim (정적 중력 보상 — NMPC는 PD 분리 안 됨)
    # tau_pd  = 0, tau_imp = tau_cmd − tau_dyn − tau_grf
    for fi in range(N_FRAMES):
        R_b = R.body_R_hist[fi]
        for leg_idx in range(4):
            nj    = N_JOINTS_PER_LEG[leg_idx]
            q_leg = R.joint_hist[fi, leg_idx, :nj]
            front = (leg_idx < 2)
            dh    = LEG_DH[leg_idx]
            lm    = LINK_MASS_PER_LEG[leg_idx]
            J     = compute_jacobian_sim(q_leg, dh, front)
            lam_world  = R.wbc_lam_des[fi, leg_idx]
            lam_local  = R_b.T @ lam_world
            R.wbc_tau_grf[fi, leg_idx, :nj] = -(J.T @ lam_local)
            R.wbc_tau_dyn[fi, leg_idx, :nj] = compute_gravity_torque_sim(q_leg, dh, lm, front)
            R.wbc_tau_pd[fi, leg_idx, :nj]  = 0.0
            R.wbc_tau_imp[fi, leg_idx, :nj] = (R.wbc_tau_cmd[fi, leg_idx, :nj]
                                                - R.wbc_tau_dyn[fi, leg_idx, :nj]
                                                - R.wbc_tau_grf[fi, leg_idx, :nj])
            R.wbc_lam_calc[fi, leg_idx] = R.wbc_lam_des[fi, leg_idx]
