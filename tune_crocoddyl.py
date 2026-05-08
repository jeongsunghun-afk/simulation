"""
crocoddyl trot cost 튜닝 grid search.

여러 조합을 빠르게 시뮬해서 best 찾기:
    · contact baumgarte (kp, kd)
    · foot tracking weight (전체 + 개별 z 가중치)
    · state regularization weight

기준: 1-cycle trot, 평가 항목:
    · 발산 여부 (FDDP done)
    · stance foot z drift (max 침투)
    · swing foot z peak (target 0.08)
    · body roll/pitch
    · 수렴 시간
"""
import math
import time
import numpy as np
import pinocchio as pin
import crocoddyl

import build_pin_model as bm

V_TARGET   = 1.0
T_PERIOD   = 0.5
DT         = 0.02
N_PER_PHASE = 12
N_CYCLES   = 1
STEP_HEIGHT = 0.08
STEP_LENGTH = 0.25

model = bm.build_model()
data = model.createData()
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFloatingBase(state)

Q_HOME_FRONT = [0.0, math.radians(133.2973), math.radians(46.7027),
                math.radians(30.6583), math.radians(59.3417)]
Q_HOME_HIND  = [0.0, math.radians(-150.0), math.radians(-90.0),
                math.radians(90.0), math.radians(60.0)]
q0 = pin.neutral(model)
for leg, q_home in [('FR', Q_HOME_FRONT), ('FL', Q_HOME_FRONT),
                     ('HR', Q_HOME_HIND),  ('HL', Q_HOME_HIND)]:
    for i, qi in enumerate(q_home):
        jid = model.getJointId(f'leg_{leg}_j{i+1}')
        q0[model.idx_qs[jid]] = qi
pin.forwardKinematics(model, data, q0)
pin.updateFramePlacements(model, data)
foot_z = data.oMf[model.getFrameId('leg_FR_foot')].translation[2]
q0[2] = -foot_z
v0 = np.zeros(model.nv); v0[0] = V_TARGET
x0 = np.concatenate([q0, v0])

foot_frames = {leg: model.getFrameId(f'leg_{leg}_foot') for leg in ['FR','FL','HR','HL']}
pin.forwardKinematics(model, data, q0)
pin.updateFramePlacements(model, data)
foot_home = {leg: data.oMf[foot_frames[leg]].translation.copy() for leg in ['FR','FL','HR','HL']}


def cycloid(p_start, p_end, sw_t, h):
    pos = p_start + sw_t * (p_end - p_start)
    pos[2] = p_start[2] + h * 4.0 * sw_t * (1.0 - sw_t)
    return pos


def solve_trot(W_track, W_track_z, baumgarte_kp, baumgarte_kd, N_CYCLES=1):
    """Trot N_CYCLES 풀이 후 metrics 반환."""
    T_SW = N_PER_PHASE * DT
    actions = []
    N_TOTAL = N_PER_PHASE * 2 * N_CYCLES

    for k in range(N_TOTAL):
        cycle_idx = k // (N_PER_PHASE * 2)
        in_cycle_idx = k % (N_PER_PHASE * 2)
        in_phase_A = in_cycle_idx < N_PER_PHASE
        if in_phase_A:
            stance, swing = ['FL','HR'], ['FR','HL']
            k_in_phase = in_cycle_idx
            phase_start_k = cycle_idx * N_PER_PHASE * 2
        else:
            stance, swing = ['FR','HL'], ['FL','HR']
            k_in_phase = in_cycle_idx - N_PER_PHASE
            phase_start_k = cycle_idx * N_PER_PHASE * 2 + N_PER_PHASE

        sw_t = (k_in_phase + 0.5) / N_PER_PHASE
        t_phase_start = phase_start_k * DT

        # Build action
        contact_model = crocoddyl.ContactModelMultiple(state, actuation.nu)
        for leg in stance:
            contact = crocoddyl.ContactModel3D(
                state, foot_frames[leg], np.zeros(3),
                pin.LOCAL_WORLD_ALIGNED, actuation.nu,
                np.array([baumgarte_kp, baumgarte_kd]))
            contact_model.addContact(f'contact_{leg}', contact)

        cost_model = crocoddyl.CostModelSum(state, actuation.nu)
        cost_model.addCost('stateReg',
                           crocoddyl.CostModelResidual(state,
                               crocoddyl.ResidualModelState(state, x0, actuation.nu)),
                           1e0)
        cost_model.addCost('ctrlReg',
                           crocoddyl.CostModelResidual(state,
                               crocoddyl.ResidualModelControl(state, actuation.nu)),
                           1e-3)
        for leg in swing:
            p_start = foot_home[leg].copy()
            p_start[0] += V_TARGET * t_phase_start - STEP_LENGTH/2
            p_end   = foot_home[leg].copy()
            p_end[0]   += V_TARGET * (t_phase_start + T_SW) + STEP_LENGTH/2
            tgt = cycloid(p_start, p_end, sw_t, STEP_HEIGHT)
            # ResidualModelFrameTranslation은 모든 축 동일 가중치
            # 개별 축 가중치는 ActivationModelWeightedQuad 사용
            residual = crocoddyl.ResidualModelFrameTranslation(
                state, foot_frames[leg], tgt, actuation.nu)
            activation = crocoddyl.ActivationModelWeightedQuad(
                np.array([W_track, W_track, W_track_z]))
            cost = crocoddyl.CostModelResidual(state, activation, residual)
            cost_model.addCost(f'foot_{leg}', cost, 1.0)   # weight in activation

        diff = crocoddyl.DifferentialActionModelContactFwdDynamics(
            state, actuation, contact_model, cost_model, 0.0, True)
        actions.append(crocoddyl.IntegratedActionModelEuler(diff, DT))

    # Terminal
    cm_T = crocoddyl.ContactModelMultiple(state, actuation.nu)
    for leg in ['FR','FL','HR','HL']:
        c = crocoddyl.ContactModel3D(
            state, foot_frames[leg], np.zeros(3),
            pin.LOCAL_WORLD_ALIGNED, actuation.nu,
            np.array([baumgarte_kp, baumgarte_kd]))
        cm_T.addContact(f'contact_{leg}', c)
    x_ref_T = x0.copy(); x_ref_T[0] = q0[0] + V_TARGET * T_PERIOD * N_CYCLES
    cost_T = crocoddyl.CostModelSum(state, actuation.nu)
    cost_T.addCost('stateReg_T',
                   crocoddyl.CostModelResidual(state,
                       crocoddyl.ResidualModelState(state, x_ref_T, actuation.nu)),
                   1e2)
    diff_T = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, cm_T, cost_T, 0.0, True)
    terminal = crocoddyl.IntegratedActionModelEuler(diff_T, 0.0)

    problem = crocoddyl.ShootingProblem(x0, actions, terminal)
    solver = crocoddyl.SolverFDDP(problem)
    xs_init = [x0] * (N_TOTAL + 1)
    us_init = [np.zeros(actuation.nu)] * N_TOTAL

    t0 = time.time()
    done = solver.solve(xs_init, us_init, maxiter=200, is_feasible=False, init_reg=1.0)
    elapsed = time.time() - t0

    if not done:
        return None

    xs_arr = np.array(solver.xs)
    # Foot trajectory analysis
    foot_z_min = {leg: 1e9 for leg in ['FR','FL','HR','HL']}
    foot_z_max = {leg: -1e9 for leg in ['FR','FL','HR','HL']}
    for fi in range(N_TOTAL + 1):
        q = xs_arr[fi, :model.nq]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for leg in ['FR','FL','HR','HL']:
            z = data.oMf[foot_frames[leg]].translation[2]
            foot_z_min[leg] = min(foot_z_min[leg], z)
            foot_z_max[leg] = max(foot_z_max[leg], z)

    # Body angles
    quats = xs_arr[:, 3:7]
    roll_max, pitch_max = 0.0, 0.0
    for q in quats:
        R = pin.Quaternion(q[3], q[0], q[1], q[2]).toRotationMatrix()
        pitch = math.asin(max(-1, min(1, -R[2,0])))
        roll  = math.atan2(R[2,1], R[2,2])
        roll_max  = max(roll_max,  abs(roll))
        pitch_max = max(pitch_max, abs(pitch))

    body_z = xs_arr[:, 2]
    return {
        'iters': solver.iter,
        'time_ms': elapsed * 1e3,
        'final_cost': float(solver.cost),
        'foot_z_penetration': abs(min(foot_z_min.values())),  # 침투 (음수의 절대값)
        'foot_z_peak':         max(foot_z_max.values()),       # swing 피크
        'roll_max_deg':  math.degrees(roll_max),
        'pitch_max_deg': math.degrees(pitch_max),
        'body_z_range_mm': (body_z.max() - body_z.min()) * 1e3,
    }


# Grid search
print(f'{"W_track":>10}{"W_z":>8}{"kp":>5}{"kd":>5}|'
      f'{"iter":>5}{"t(ms)":>7}{"cost":>9}|'
      f'{"penet":>8}{"peak":>7}|{"roll":>6}{"pitch":>6}|{"z_rng":>6}')
print('─' * 95)

configs = [
    # (W_track_xy, W_track_z, kp, kd)
    (1e2, 1e4,    0,  50),  # round 2 best
    (1e2, 1e4,    0,  20),  # 1e4 + kd 약화
    (1e2, 3e4,    0,  20),  # z 더 강화 + kd 약화
    (1e2, 5e3,    0,  20),  # z 5000 + kd 20
    (1e2, 1e4,    0,  10),  # kd 매우 약화
    (5e1, 1e4,    0,  20),  # xy 약화
    (1e2, 1e4,    1,  20),  # 매우 약한 kp 추가
    (2e2, 1e4,    0,  20),  # xy 강화
]

for W_track, W_z, kp, kd in configs:
    r = solve_trot(W_track, W_z, kp, kd, N_CYCLES=1)
    if r is None:
        print(f'{W_track:>10.0f}{W_z:>8.0f}{kp:>5}{kd:>5}|  DIVERGED')
        continue
    print(f'{W_track:>10.0f}{W_z:>8.0f}{kp:>5}{kd:>5}|'
          f'{r["iters"]:>5}{r["time_ms"]:>7.0f}{r["final_cost"]:>9.2e}|'
          f'{r["foot_z_penetration"]*1e3:>7.1f}mm{r["foot_z_peak"]*1e3:>6.1f}mm|'
          f'{r["roll_max_deg"]:>+5.2f}°{r["pitch_max_deg"]:>+5.2f}°|'
          f'{r["body_z_range_mm"]:>5.1f}mm')
