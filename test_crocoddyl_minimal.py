"""
crocoddyl minimal 테스트: 4발 지지 자세에서 N step stand-still NMPC.

목표:
    · build_pin_model의 quadruped pinocchio model을 crocoddyl이 사용 가능 검증
    · 모든 발 contact + body home pose 추종 cost
    · FDDP로 N step 풀이
    · 결과: 거의 0 control (균형 잡힌 가만히 서있기)
"""
import math
import numpy as np
import pinocchio as pin
import crocoddyl

import build_pin_model as bm

# ── 모델 ─────────────────────────────────────────
model = bm.build_model()
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFloatingBase(state)
print(f'State: nx={state.nx}, ndx={state.ndx}')
print(f'Actuation: nu={actuation.nu}')

# ── x0: home 자세 (4발 지지) ─────────────────────
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
v0 = np.zeros(model.nv)
x0 = np.concatenate([q0, v0])
print(f'x0 shape: {x0.shape}')

# 발이 정확히 ground level(z=-0.465)에 닿도록 base z 조정
data_tmp = model.createData()
pin.forwardKinematics(model, data_tmp, q0)
pin.updateFramePlacements(model, data_tmp)
foot_z_world = data_tmp.oMf[model.getFrameId('leg_FR_foot')].translation[2]
print(f'foot_z @ neutral base: {foot_z_world*1e3:.2f}mm')
# base를 위로 올려 발이 z=0에 닿게 (ground = z=0 for crocoddyl simplicity)
q0[2] = -foot_z_world
x0 = np.concatenate([q0, v0])
pin.forwardKinematics(model, data_tmp, q0)
pin.updateFramePlacements(model, data_tmp)
foot_z_after = data_tmp.oMf[model.getFrameId('leg_FR_foot')].translation[2]
print(f'foot_z @ adjusted base (z={q0[2]*1e3:.2f}mm): {foot_z_after*1e3:.4f}mm')

# ── ContactModel: 4발 모두 ground contact ────────
contact_model = crocoddyl.ContactModelMultiple(state, actuation.nu)
foot_frames = {leg: model.getFrameId(f'leg_{leg}_foot') for leg in ['FR','FL','HR','HL']}
for leg, fid in foot_frames.items():
    contact = crocoddyl.ContactModel3D(
        state, fid, np.zeros(3),  # foot velocity reference = 0
        pin.LOCAL_WORLD_ALIGNED, actuation.nu,
        np.array([0.0, 50.0])  # baumgarte gains (kp=0, kd=50)
    )
    contact_model.addContact(f'contact_{leg}', contact)

# ── CostModel: body home + control reg ──────────
cost_model = crocoddyl.CostModelSum(state, actuation.nu)

# State regularization (q,v를 x0 근처로)
state_reg = crocoddyl.CostModelResidual(
    state,
    crocoddyl.ResidualModelState(state, x0, actuation.nu)
)
cost_model.addCost('stateReg', state_reg, 1e1)

# Control regularization (torque 작게)
ctrl_reg = crocoddyl.CostModelResidual(
    state,
    crocoddyl.ResidualModelControl(state, actuation.nu)
)
cost_model.addCost('ctrlReg', ctrl_reg, 1e-3)

# ── DifferentialActionModel: 1 step dynamics + cost ──
diff_action = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contact_model, cost_model,
    0.0, True   # JMinvJt damping, enable_force
)

DT = 0.02
action_model = crocoddyl.IntegratedActionModelEuler(diff_action, DT)
terminal_cost = crocoddyl.CostModelSum(state, actuation.nu)
terminal_cost.addCost('stateReg_T',
                       crocoddyl.CostModelResidual(state,
                           crocoddyl.ResidualModelState(state, x0, actuation.nu)),
                       1e2)   # 종단에서 home 자세
diff_terminal = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contact_model, terminal_cost, 0.0, True)
terminal_action = crocoddyl.IntegratedActionModelEuler(diff_terminal, 0.0)

# ── ShootingProblem ──────────────────────────────
N = 10
problem = crocoddyl.ShootingProblem(x0, [action_model] * N, terminal_action)
print(f'\nShooting problem: N={N} steps, DT={DT}s')

# ── FDDP 솔버 ────────────────────────────────────
solver = crocoddyl.SolverFDDP(problem)
solver.setCallbacks([crocoddyl.CallbackVerbose()])

us_init = [np.zeros(actuation.nu)] * N
xs_init = [x0] * (N + 1)

print('\n━━━━━━━━━━ Solving ━━━━━━━━━━')
import time
t0 = time.time()
done = solver.solve(xs_init, us_init, maxiter=100, is_feasible=False, init_reg=0.1)
elapsed = time.time() - t0
print(f'\nSolved: done={done}, iters={solver.iter}, time={elapsed*1e3:.1f}ms')
print(f'Final cost: {solver.cost:.6e}')

# 결과 분석
print(f'\n첫 control |u_0|max: {np.max(np.abs(solver.us[0])):.4f}')
print(f'첫 control u_0:')
print(f'  joint torques (20 dim): {solver.us[0].round(3)}')
