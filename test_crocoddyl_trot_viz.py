"""
crocoddyl trot 2-cycle 시뮬레이션 + 3D 시각화.

test_crocoddyl_trot.py를 확장:
    · 2 cycle (T=1.0s, 48 steps) 풀이
    · 풀이 결과 trajectory를 matplotlib 3D animation으로 재생
    · v11 figure 1 스타일 (chassis + 4 legs + foot traces)
"""
import math
import numpy as np
import pinocchio as pin
import crocoddyl
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import build_pin_model as bm

# ── Style ─────────────────────────────────────
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ── 게이트 파라미터 ──────────────────────────
V_TARGET   = 1.0
T_PERIOD   = 0.5
DT         = 0.02
N_PER_PHASE = int((T_PERIOD/2) / DT)   # 12
N_CYCLES   = 2
STEP_HEIGHT = 0.08
STEP_LENGTH = 0.25

# ── 모델 + 초기 상태 ────────────────────────
model = bm.build_model()
data  = model.createData()
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
foot_z_native = data.oMf[model.getFrameId('leg_FR_foot')].translation[2]
q0[2] = -foot_z_native    # base z 조정 → 발이 z=0 ground
v0 = np.zeros(model.nv); v0[0] = V_TARGET
x0 = np.concatenate([q0, v0])

foot_frames = {leg: model.getFrameId(f'leg_{leg}_foot') for leg in ['FR','FL','HR','HL']}
pin.forwardKinematics(model, data, q0)
pin.updateFramePlacements(model, data)
foot_home = {leg: data.oMf[foot_frames[leg]].translation.copy()
             for leg in ['FR','FL','HR','HL']}

# ── ActionModel 빌드 helper ────────────────────
def build_action(stance_legs, swing_legs, swing_targets, dt):
    # 튜닝된 가중치 (tune_crocoddyl.py 결과):
    #   contact baumgarte: kp=0, kd=20 (drift 최소화)
    #   foot tracking: xy 100, z 10000 (z 강화로 swing peak 정확)
    contact_model = crocoddyl.ContactModelMultiple(state, actuation.nu)
    for leg in stance_legs:
        contact = crocoddyl.ContactModel3D(
            state, foot_frames[leg], np.zeros(3),
            pin.LOCAL_WORLD_ALIGNED, actuation.nu, np.array([0.0, 20.0]))
        contact_model.addContact(f'contact_{leg}', contact)

    cost_model = crocoddyl.CostModelSum(state, actuation.nu)
    state_reg = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelState(state, x0, actuation.nu))
    cost_model.addCost('stateReg', state_reg, 1e0)
    ctrl_reg = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu))
    cost_model.addCost('ctrlReg', ctrl_reg, 1e-3)
    for leg, tgt in swing_targets.items():
        residual = crocoddyl.ResidualModelFrameTranslation(
            state, foot_frames[leg], tgt, actuation.nu)
        # 개별 축 가중치 (xy=100, z=10000) for swing peak 정확 추종
        activation = crocoddyl.ActivationModelWeightedQuad(
            np.array([100.0, 100.0, 10000.0]))
        cost = crocoddyl.CostModelResidual(state, activation, residual)
        cost_model.addCost(f'foot_{leg}', cost, 1.0)

    diff_model = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contact_model, cost_model, 0.0, True)
    return crocoddyl.IntegratedActionModelEuler(diff_model, dt)


def cycloid_swing(p_start, p_end, sw_t, height=STEP_HEIGHT):
    pos = p_start + sw_t * (p_end - p_start)
    pos_z = p_start[2] + height * 4.0 * sw_t * (1.0 - sw_t)
    return np.array([pos[0], pos[1], pos_z])


# ── 2-cycle ShootingProblem ────────────────────
N_TOTAL = N_PER_PHASE * 2 * N_CYCLES   # 48 steps
print(f'N_TOTAL = {N_TOTAL} steps × DT={DT}s = {N_TOTAL*DT}s = {N_CYCLES} cycles')

# 각 phase 시작 시점에 swing 이벤트의 p_start/p_end를 한 번만 계산
# 이벤트 진행 동안 (k = phase_start_k .. phase_start_k+N_PER_PHASE-1) 두 점은 고정.
# p_start/p_end (world frame) for each swing event:
#   p_start = home_world + V·t_phase_start - STEP/2
#   p_end   = home_world + V·t_phase_end   + STEP/2
#   (t_phase_end = t_phase_start + T_SW)
T_SW_PHASE = N_PER_PHASE * DT   # = T_PERIOD/2 = 0.25s
STEP = STEP_LENGTH

actions = []
for k in range(N_TOTAL):
    cycle_idx = k // (N_PER_PHASE * 2)
    in_cycle_idx = k % (N_PER_PHASE * 2)
    in_phase_A = in_cycle_idx < N_PER_PHASE
    if in_phase_A:
        stance, swing = ['FL', 'HR'], ['FR', 'HL']
        k_in_phase = in_cycle_idx
        phase_start_k = cycle_idx * N_PER_PHASE * 2
    else:
        stance, swing = ['FR', 'HL'], ['FL', 'HR']
        k_in_phase = in_cycle_idx - N_PER_PHASE
        phase_start_k = cycle_idx * N_PER_PHASE * 2 + N_PER_PHASE

    # sw_t를 0~1 정확히 cover: k=0 → sw_t=0 (발 시작), k=N-1 → sw_t≈1 (발 도착)
    sw_t = k_in_phase / max(N_PER_PHASE - 1, 1)
    t_phase_start = phase_start_k * DT
    t_phase_end   = t_phase_start + T_SW_PHASE

    swing_targets = {}
    for leg in swing:
        p_start = foot_home[leg].copy()
        p_start[0] += V_TARGET * t_phase_start - STEP/2
        p_end   = foot_home[leg].copy()
        p_end[0]   += V_TARGET * t_phase_end   + STEP/2
        swing_targets[leg] = cycloid_swing(p_start, p_end, sw_t)

    actions.append(build_action(stance, swing, swing_targets, DT))

# Terminal model (4발 contact + body home advanced)
contact_model_T = crocoddyl.ContactModelMultiple(state, actuation.nu)
for leg in ['FR','FL','HR','HL']:
    c = crocoddyl.ContactModel3D(
        state, foot_frames[leg], np.zeros(3),
        pin.LOCAL_WORLD_ALIGNED, actuation.nu, np.array([0.0, 50.0]))
    contact_model_T.addContact(f'contact_{leg}', c)
x_ref_T = x0.copy(); x_ref_T[0] = q0[0] + V_TARGET * T_PERIOD * N_CYCLES
cost_T = crocoddyl.CostModelSum(state, actuation.nu)
cost_T.addCost('stateReg_T',
               crocoddyl.CostModelResidual(state,
                   crocoddyl.ResidualModelState(state, x_ref_T, actuation.nu)),
               1e2)
diff_T = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contact_model_T, cost_T, 0.0, True)
terminal_action = crocoddyl.IntegratedActionModelEuler(diff_T, 0.0)

problem = crocoddyl.ShootingProblem(x0, actions, terminal_action)
print(f'ShootingProblem: {len(actions)} actions + 1 terminal')

# ── Solve ──────────────────────────────────────
solver = crocoddyl.SolverFDDP(problem)
xs_init = [x0] * (N_TOTAL + 1)
us_init = [np.zeros(actuation.nu)] * N_TOTAL

t0 = time.time()
done = solver.solve(xs_init, us_init, maxiter=300, is_feasible=False, init_reg=1.0)
elapsed = time.time() - t0
print(f'\nFDDP: done={done}, iters={solver.iter}, time={elapsed*1e3:.1f}ms')

xs_arr = np.array(solver.xs)   # (N+1, 53)
us_arr = np.array(solver.us)   # (N, 20)

# ── 진단 출력 ──────────────────────────────────
body_z  = xs_arr[:, 2]
body_vx = xs_arr[:, model.nq]
quats   = xs_arr[:, 3:7]
rolls, pitches = [], []
for q in quats:
    R = pin.Quaternion(q[3], q[0], q[1], q[2]).toRotationMatrix()
    pitch = math.asin(max(-1, min(1, -R[2,0])))
    roll  = math.atan2(R[2,1], R[2,2])
    rolls.append(roll); pitches.append(pitch)
rolls   = np.degrees(rolls)
pitches = np.degrees(pitches)

print(f'\n━━━ 2-cycle 결과 ━━━')
print(f'Body z: {body_z.min()*1e3:.2f} ~ {body_z.max()*1e3:.2f} mm  (start={body_z[0]*1e3:.2f})')
print(f'Body vx: {body_vx.min():.3f} ~ {body_vx.max():.3f} m/s (target={V_TARGET})')
print(f'Roll:    {rolls.min():+.2f} ~ {rolls.max():+.2f}°')
print(f'Pitch:   {pitches.min():+.2f} ~ {pitches.max():+.2f}°')
print(f'|τ| max: {np.max(np.abs(us_arr)):.2f} Nm')

# ── 3D 시각화 ──────────────────────────────────
print('\n━━━ 시각화 시작 ━━━')

# 매 step의 leg link 위치 미리 계산
N_FRAMES = N_TOTAL + 1
leg_link_hist = np.zeros((N_FRAMES, 4, 6, 3))   # [frame, leg, joint(0-5), xyz]
foot_hist = np.zeros((N_FRAMES, 4, 3))
LEG_NAMES = ['FR', 'FL', 'HR', 'HL']

# leg_base frame은 model에 없으니 base_link + hip_offset으로 계산
base_link_id = model.getFrameId('base_link')
hip_offsets_local = np.array(bm.LEG_HIP_OFFSETS)   # (4, 3)

for fi in range(N_FRAMES):
    q = xs_arr[fi, :model.nq]
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    base_T = data.oMf[base_link_id]
    for leg_idx, leg in enumerate(LEG_NAMES):
        # hip position in world = base * hip_offset
        hip_w = base_T.translation + base_T.rotation @ hip_offsets_local[leg_idx]
        leg_link_hist[fi, leg_idx, 0] = hip_w
        for j in range(5):
            link_id = model.getFrameId(f'leg_{leg}_l{j+1}')
            leg_link_hist[fi, leg_idx, j+1] = data.oMf[link_id].translation
        foot_id = model.getFrameId(f'leg_{leg}_foot')
        foot_hist[fi, leg_idx] = data.oMf[foot_id].translation

# Figure: 3D 애니메이션(좌) + 발 높이 시계열(우)
fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor('#1a1a2e')
ax = fig.add_subplot(121, projection='3d')
ax.set_facecolor('#16213e')
ax.tick_params(colors='gray')

# Body-following view: 카메라가 body 따라가도록 — body local frame 기준
# axis ranges: body 주변 ±0.6m 만 보이도록 (좁게)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.4, 0.4)
ax.set_zlim(-0.6, 0.4)
ax.set_xlabel('X rel body (m)', color='white')
ax.set_ylabel('Y (m)', color='white')
ax.set_zlabel('Z (m)', color='white')
ax.view_init(elev=20, azim=-55)

# Ground plane (body 따라가는 view에서도 ground는 보이도록)
xx, yy = np.meshgrid([-0.6, 0.6], [-0.4, 0.4])
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.15, color='#888888')

LEG_COLORS = ['#00d4ff', '#ff6b35', '#00ff99', '#c264ff']

# Body chassis
chassis_line, = ax.plot([], [], [], '-', color='white', lw=2.5)
# Leg links (5 segments per leg)
leg_lines = []
for c in LEG_COLORS:
    lns = [ax.plot([], [], [], '-o', color=c, lw=2, ms=4)[0] for _ in range(5)]
    leg_lines.append(lns)
# Foot traces
foot_traces = [ax.plot([], [], [], '-', color=c, lw=1, alpha=0.5)[0] for c in LEG_COLORS]
trace_buf = [[[], [], []] for _ in range(4)]
TRACE_LEN = 50
# Body CoM
com_marker, = ax.plot([], [], [], 'X', color='yellow', ms=10)
# Time text
time_text = ax.text2D(0.02, 0.97, '', transform=ax.transAxes, color='white', fontsize=11)


def update(fi):
    # body x를 빼서 body-following 시점 (body가 화면 중앙에 머무름)
    body_x = xs_arr[fi, 0]
    # Chassis: 4 hips
    bases = leg_link_hist[fi, :, 0, :].copy()   # (4, 3)
    bases[:, 0] -= body_x
    chassis = np.array([bases[0], bases[2], bases[3], bases[1], bases[0]])
    chassis_line.set_data(chassis[:,0], chassis[:,1])
    chassis_line.set_3d_properties(chassis[:,2])
    # Body CoM (relative to body x = 0)
    com = xs_arr[fi, :3].copy(); com[0] = 0.0
    com_marker.set_data([com[0]], [com[1]])
    com_marker.set_3d_properties([com[2]])
    # Leg links
    for leg_idx in range(4):
        for j in range(5):
            A = leg_link_hist[fi, leg_idx, j].copy(); A[0] -= body_x
            B = leg_link_hist[fi, leg_idx, j+1].copy(); B[0] -= body_x
            leg_lines[leg_idx][j].set_data([A[0], B[0]], [A[1], B[1]])
            leg_lines[leg_idx][j].set_3d_properties([A[2], B[2]])
        # Foot trace (body-relative)
        fp = foot_hist[fi, leg_idx].copy(); fp[0] -= body_x
        trace_buf[leg_idx][0].append(fp[0])
        trace_buf[leg_idx][1].append(fp[1])
        trace_buf[leg_idx][2].append(fp[2])
        foot_traces[leg_idx].set_data(
            trace_buf[leg_idx][0][-TRACE_LEN:], trace_buf[leg_idx][1][-TRACE_LEN:])
        foot_traces[leg_idx].set_3d_properties(trace_buf[leg_idx][2][-TRACE_LEN:])
    time_text.set_text(f't = {fi*DT:.3f}s  ({fi}/{N_FRAMES-1})\n'
                        f'body x = {body_x:+.3f}m (world)\n'
                        f'roll={rolls[fi]:+.2f}° pitch={pitches[fi]:+.2f}°\n'
                        f'vx={body_vx[fi]:.3f} m/s')
    return [chassis_line, com_marker, time_text] + sum(leg_lines, []) + foot_traces


ax.set_title(f'crocoddyl NMPC: trot {N_CYCLES} cycles ({N_TOTAL*DT:.1f}s) — '
              f'roll<{rolls.max():.1f}° pitch<{pitches.max():.1f}°',
              color='white', fontsize=11)

# 발 높이 + x 위치 시계열 (오른쪽 plot)
ax_z = fig.add_subplot(222)
ax_z.set_facecolor('#16213e')
ax_z.tick_params(colors='gray')
ax_z.set_title('Foot Z (world) — swing peaks visible', color='white', fontsize=10)
ax_z.set_ylabel('Z (m)', color='white')
ax_z.grid(alpha=0.3, color='gray')
t_axis = np.arange(N_FRAMES) * DT
for leg_idx, c in enumerate(LEG_COLORS):
    ax_z.plot(t_axis, foot_hist[:, leg_idx, 2], '-', color=c,
              label=LEG_NAMES[leg_idx], lw=1.5)
ax_z.axhline(0, color='white', ls='--', lw=0.5, alpha=0.5)
ax_z.legend(fontsize=8, ncol=4, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray')

ax_x = fig.add_subplot(224)
ax_x.set_facecolor('#16213e')
ax_x.tick_params(colors='gray')
ax_x.set_title('Foot X (world) — stance flat / swing forward', color='white', fontsize=10)
ax_x.set_xlabel('t (s)', color='white')
ax_x.set_ylabel('X (m)', color='white')
ax_x.grid(alpha=0.3, color='gray')
for leg_idx, c in enumerate(LEG_COLORS):
    ax_x.plot(t_axis, foot_hist[:, leg_idx, 0], '-', color=c,
              label=LEG_NAMES[leg_idx], lw=1.5)
ax_x.legend(fontsize=8, ncol=4, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray')

ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=50, blit=False, repeat=True)

# Save GIF (Pillow writer, ffmpeg 불필요)
gif_path = '/tmp/crocoddyl_trot_2cycle.gif'
try:
    ani.save(gif_path, fps=20, writer='pillow')
    print(f'animation saved → {gif_path}')
except Exception as e:
    print(f'(GIF save failed: {e})')

# 추가: 키 프레임 8장을 PNG로
fig2, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={'projection': '3d'})
fig2.patch.set_facecolor('#1a1a2e')
key_frames = np.linspace(0, N_FRAMES-1, 8, dtype=int)
for ax_i, fi in enumerate(key_frames):
    ax_k = axes[ax_i // 4, ax_i % 4]
    ax_k.set_facecolor('#16213e')
    ax_k.tick_params(colors='gray', labelsize=6)
    ax_k.set_xlim(x_range); ax_k.set_ylim(-0.4, 0.4); ax_k.set_zlim(-0.6, 0.4)
    ax_k.view_init(elev=20, azim=-55)
    # ground
    ax_k.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.15, color='#888888')
    # chassis
    bases = leg_link_hist[fi, :, 0, :]
    chassis = np.array([bases[0], bases[2], bases[3], bases[1], bases[0]])
    ax_k.plot(chassis[:,0], chassis[:,1], chassis[:,2], '-', color='white', lw=2.5)
    # legs
    for leg_idx, c in enumerate(LEG_COLORS):
        for j in range(5):
            A = leg_link_hist[fi, leg_idx, j]; B = leg_link_hist[fi, leg_idx, j+1]
            ax_k.plot([A[0],B[0]], [A[1],B[1]], [A[2],B[2]], '-o', color=c, lw=1.5, ms=3)
    # com
    ax_k.plot([xs_arr[fi,0]], [xs_arr[fi,1]], [xs_arr[fi,2]],
              'X', color='yellow', ms=8)
    ax_k.set_title(f't={fi*DT:.2f}s  roll={rolls[fi]:+.1f}°', color='white', fontsize=9)

png_path = '/tmp/crocoddyl_trot_2cycle.png'
fig2.tight_layout()
fig2.savefig(png_path, facecolor='#1a1a2e', dpi=110)
print(f'키 프레임 PNG saved → {png_path}')

plt.tight_layout()
plt.show()
