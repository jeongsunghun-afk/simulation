"""
crocoddyl trot NMPC 테스트.

Trot 패턴 (T=0.5s, swing_ratio=0.5):
    Phase A (t∈[0, T/2]): FL + HR stance, FR + HL swing
    Phase B (t∈[T/2, T]): FR + HL stance, FL + HR swing

각 phase마다 다른 ContactModel을 가진 IntegratedActionModel 생성.
ShootingProblem이 이를 horizon에 따라 alternate.

cost:
    · Body z height 유지 (pz = pz_home)
    · Body forward velocity vx = V
    · Body upright (roll=pitch=0)
    · Swing foot tracking (cycloid 궤적)
    · Control regularization

검증: trot 1 cycle, body 안정 유지 (roll/pitch 작음, vx≈V).
"""
import math
import numpy as np
import pinocchio as pin
import crocoddyl

import build_pin_model as bm

# ── 게이트 파라미터 ──────────────────────────────────
V_TARGET   = 1.0    # m/s (전진 속도)
T_PERIOD   = 0.5    # s
T_SWING    = 0.25   # s (per-foot swing 시간)
DT         = 0.02   # s (NMPC sampling)
N_PER_PHASE = int((T_PERIOD/2) / DT)   # = 12.5 → 12 (1 phase 길이)
STEP_HEIGHT = 0.08
STEP_LENGTH = 0.25

# ── 모델 ─────────────────────────────────────────
model = bm.build_model()
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# ── x0 ───────────────────────────────────────────
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

# 발이 ground level z=0 닿도록 base z 조정
data_tmp = model.createData()
pin.forwardKinematics(model, data_tmp, q0)
pin.updateFramePlacements(model, data_tmp)
foot_z_native = data_tmp.oMf[model.getFrameId('leg_FR_foot')].translation[2]
q0[2] = -foot_z_native
v0 = np.zeros(model.nv)
v0[0] = V_TARGET   # initial vx = V (forward 직진)
x0 = np.concatenate([q0, v0])

foot_frames = {leg: model.getFrameId(f'leg_{leg}_foot') for leg in ['FR','FL','HR','HL']}

# 발 home positions (world)
pin.forwardKinematics(model, data_tmp, q0)
pin.updateFramePlacements(model, data_tmp)
foot_home = {leg: data_tmp.oMf[foot_frames[leg]].translation.copy()
             for leg in ['FR','FL','HR','HL']}
print(f'Foot home positions:')
for leg, p in foot_home.items():
    print(f'  {leg}: {p.round(4)}')


def build_action(stance_legs, swing_legs, swing_targets, swing_t_in_phase):
    """1 step ActionModel.
    stance_legs: list of leg names that are in contact
    swing_legs: list of leg names that are swinging
    swing_targets: dict {leg: 3d target position} (for foot tracking cost)
    swing_t_in_phase: 0~1 progress within swing phase
    """
    # Contact model
    contact_model = crocoddyl.ContactModelMultiple(state, actuation.nu)
    for leg in stance_legs:
        contact = crocoddyl.ContactModel3D(
            state, foot_frames[leg], np.zeros(3),
            pin.LOCAL_WORLD_ALIGNED, actuation.nu,
            np.array([0.0, 50.0]),
        )
        contact_model.addContact(f'contact_{leg}', contact)

    # Cost model
    cost_model = crocoddyl.CostModelSum(state, actuation.nu)

    # 1) State regularization (q,v stays near home, vx=V)
    x_ref = np.concatenate([q0, v0])
    state_reg = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ResidualModelState(state, x_ref, actuation.nu),
    )
    cost_model.addCost('stateReg', state_reg, 1e0)

    # 2) Control regularization
    ctrl_reg = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ResidualModelControl(state, actuation.nu),
    )
    cost_model.addCost('ctrlReg', ctrl_reg, 1e-3)

    # 3) Swing foot tracking
    for leg in swing_legs:
        if leg in swing_targets:
            track = crocoddyl.CostModelResidual(
                state,
                crocoddyl.ResidualModelFrameTranslation(
                    state, foot_frames[leg], swing_targets[leg], actuation.nu),
            )
            cost_model.addCost(f'foot_{leg}', track, 1e2)

    diff_model = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contact_model, cost_model, 0.0, True
    )
    return crocoddyl.IntegratedActionModelEuler(diff_model, DT)


def cycloid_swing(p_start, p_end, sw_t, height=STEP_HEIGHT):
    """간단한 cycloid swing trajectory (sw_t ∈ [0,1])."""
    # X: linear lerp
    pos = p_start + sw_t * (p_end - p_start)
    # Z: parabola (peak at sw_t=0.5)
    pos_z = p_start[2] + height * 4.0 * sw_t * (1.0 - sw_t)
    return np.array([pos[0], pos[1], pos_z])


# Trot pattern: 12 steps phase A + 12 steps phase B
# Phase A: FL+HR stance, FR+HL swing
# Phase B: FR+HL stance, FL+HR swing
N_TOTAL = N_PER_PHASE * 2
print(f'\nN_TOTAL = {N_TOTAL} steps × DT={DT}s = {N_TOTAL*DT}s = 1 trot cycle')

actions = []
for k in range(N_TOTAL):
    in_phase_A = k < N_PER_PHASE
    if in_phase_A:
        stance = ['FL', 'HR']
        swing  = ['FR', 'HL']
        sw_t   = (k + 0.5) / N_PER_PHASE   # 0~1
    else:
        stance = ['FR', 'HL']
        swing  = ['FL', 'HR']
        sw_t   = ((k - N_PER_PHASE) + 0.5) / N_PER_PHASE

    # Swing 발의 target position (cycloid)
    swing_targets = {}
    body_advance = V_TARGET * (k * DT)   # body가 V·t만큼 전진했다고 가정
    for leg in swing:
        # Swing 시작 = home_x - STEP_LENGTH/2 + body_advance, end = home_x + STEP_LENGTH/2 + body_advance
        p_start = foot_home[leg].copy()
        p_start[0] += body_advance - STEP_LENGTH/2
        p_end = foot_home[leg].copy()
        p_end[0] += body_advance + STEP_LENGTH/2
        swing_targets[leg] = cycloid_swing(p_start, p_end, sw_t)

    actions.append(build_action(stance, swing, swing_targets, sw_t))

# Terminal model (4발 contact, body home 자세)
contact_model_T = crocoddyl.ContactModelMultiple(state, actuation.nu)
for leg in ['FR','FL','HR','HL']:
    contact = crocoddyl.ContactModel3D(
        state, foot_frames[leg], np.zeros(3),
        pin.LOCAL_WORLD_ALIGNED, actuation.nu, np.array([0.0, 50.0]))
    contact_model_T.addContact(f'contact_{leg}', contact)

# Terminal body shifted forward by V·T
x_ref_T = x0.copy()
x_ref_T[0] = q0[0] + V_TARGET * T_PERIOD   # base x advanced
cost_T = crocoddyl.CostModelSum(state, actuation.nu)
cost_T.addCost('stateReg_T',
               crocoddyl.CostModelResidual(state,
                   crocoddyl.ResidualModelState(state, x_ref_T, actuation.nu)),
               1e2)
diff_T = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contact_model_T, cost_T, 0.0, True)
terminal_action = crocoddyl.IntegratedActionModelEuler(diff_T, 0.0)

problem = crocoddyl.ShootingProblem(x0, actions, terminal_action)
print(f'ShootingProblem built: {len(actions)} action models + 1 terminal')

# ── FDDP ─────────────────────────────────────────
solver = crocoddyl.SolverFDDP(problem)
xs_init = [x0] * (N_TOTAL + 1)
us_init = [np.zeros(actuation.nu)] * N_TOTAL

import time
t0 = time.time()
done = solver.solve(xs_init, us_init, maxiter=200, is_feasible=False, init_reg=1.0)
elapsed = time.time() - t0

print(f'\n━━━ Solver result ━━━')
print(f'done = {done}, iters = {solver.iter}, time = {elapsed*1e3:.1f}ms')
print(f'final cost = {solver.cost:.4e}')

# 결과 분석
xs_arr = np.array(solver.xs)   # (N+1, nx)
us_arr = np.array(solver.us)   # (N, nu)
print(f'\nState trajectory shape: {xs_arr.shape}')

# Body z 변화
body_z = xs_arr[:, 2]
print(f'Body z range: [{body_z.min()*1e3:.2f}, {body_z.max()*1e3:.2f}] mm  (start={body_z[0]*1e3:.2f}mm)')

# Body vx
body_vx = xs_arr[:, model.nq]   # v starts at index nq
print(f'Body vx range: [{body_vx.min():.3f}, {body_vx.max():.3f}] m/s  (target={V_TARGET})')

# Roll/pitch
quats = xs_arr[:, 3:7]
rolls = []
pitches = []
for q in quats:
    R = pin.Quaternion(q[3], q[0], q[1], q[2]).toRotationMatrix()
    pitch = math.asin(max(-1, min(1, -R[2,0])))
    roll  = math.atan2(R[2,1], R[2,2])
    rolls.append(roll)
    pitches.append(pitch)
rolls   = np.degrees(rolls)
pitches = np.degrees(pitches)
print(f'Roll  range: [{rolls.min():+.3f}, {rolls.max():+.3f}] °')
print(f'Pitch range: [{pitches.min():+.3f}, {pitches.max():+.3f}] °')

# Joint torque max
print(f'|τ| max:  {np.max(np.abs(us_arr)):.2f} Nm')
