"""
Open-loop trot 시뮬: v10 식 발 궤적 + pinocchio dynamics + PD 토크.

목적: 단순 trot 다리 동작을 하면서 body는 자유 적분되어
     점차 발산/넘어지는 모습을 시뮬로 확인.

구성:
    1. v10 식 cycloid swing 발 궤적 (body-frame)
    2. body-frame 발 위치 → home pose 기반 q_target(t)
    3. PD 토크: τ = KP·(q_target - q) + KD·(v_target - v)
    4. crocoddyl의 IntegratedActionModelContactFwdDynamics를 forward simulator로 사용
    5. 매 DT마다 body 6-DoF 자동 적분 (gravity + 접촉력)
"""
import math
import time
import numpy as np
import pinocchio as pin
import crocoddyl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import build_pin_model as bm

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ── Sim params ─────────────────────────────────
V_TARGET   = 1.0
T_PERIOD   = 0.5
DT_SIM     = 0.005
N_PER_PHASE = int((T_PERIOD/2) / DT_SIM)
T_TOTAL    = 1.5     # 3 cycles
N_FRAMES   = int(T_TOTAL / DT_SIM)
STEP_HEIGHT = 0.08
STEP_LENGTH = 0.25

KP_JOINT = 200.0
KD_JOINT = 10.0

# ── Model + state ──────────────────────────────
model = bm.build_model()
data  = model.createData()
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFloatingBase(state)

Q_HOME_FRONT = [0.0, math.radians(133.2973), math.radians(46.7027),
                math.radians(30.6583), math.radians(59.3417)]
Q_HOME_HIND  = [0.0, math.radians(-150.0), math.radians(-90.0),
                math.radians(90.0), math.radians(60.0)]
LEG_NAMES = ['FR', 'FL', 'HR', 'HL']

q_init = pin.neutral(model)
for leg, qh in [('FR', Q_HOME_FRONT), ('FL', Q_HOME_FRONT),
                 ('HR', Q_HOME_HIND), ('HL', Q_HOME_HIND)]:
    for i, qi in enumerate(qh):
        q_init[model.idx_qs[model.getJointId(f'leg_{leg}_j{i+1}')]] = qi
pin.forwardKinematics(model, data, q_init)
pin.updateFramePlacements(model, data)
foot_z_native = data.oMf[model.getFrameId('leg_FR_foot')].translation[2]
q_init[2] = -foot_z_native
v_init = np.zeros(model.nv)   # 정지 상태로 시작 (초기 V 주면 contact constraint와 충돌)
x_init = np.concatenate([q_init, v_init])

leg_v_idx = {leg: [model.idx_vs[model.getJointId(f'leg_{leg}_j{i+1}')] for i in range(5)]
             for leg in LEG_NAMES}
leg_q_idx = {leg: [model.idx_qs[model.getJointId(f'leg_{leg}_j{i+1}')] for i in range(5)]
             for leg in LEG_NAMES}
foot_frames = {leg: model.getFrameId(f'leg_{leg}_foot') for leg in LEG_NAMES}
Q_HOME_PER_LEG = {'FR': Q_HOME_FRONT, 'FL': Q_HOME_FRONT,
                   'HR': Q_HOME_HIND, 'HL': Q_HOME_HIND}


def get_phase(t):
    cycle_t = t % T_PERIOD
    if cycle_t < T_PERIOD / 2:
        sw_t = cycle_t / (T_PERIOD / 2)
        return ['FL', 'HR'], ['FR', 'HL'], sw_t
    else:
        sw_t = (cycle_t - T_PERIOD/2) / (T_PERIOD / 2)
        return ['FR', 'HL'], ['FL', 'HR'], sw_t


def compute_q_target(t):
    """간단화 q_target: home pose + swing leg에 cycloid lift."""
    stance, swing, sw_t = get_phase(t)
    q_target_per_leg = {}
    for leg in LEG_NAMES:
        q_t = list(Q_HOME_PER_LEG[leg])
        if leg in swing:
            lift = 4.0 * sw_t * (1.0 - sw_t)   # 0 → 1 → 0
            if leg in ['FR', 'FL']:
                q_t[1] -= 0.25 * lift
                q_t[2] += 0.5  * lift
                q_t[3] -= 0.25 * lift
            else:
                q_t[1] += 0.25 * lift
                q_t[2] -= 0.5  * lift
                q_t[3] += 0.25 * lift
        q_target_per_leg[leg] = q_t
    return q_target_per_leg


def compute_torque_full(x, t):
    """Returns 26-dim torque for pinocchio.constraintDynamics.
    Floating base 6 = 0 (unactuated), legs = PD + RNEA gravity comp.
    """
    q = x[:model.nq]; v = x[model.nq:]
    # RNEA feedforward (gravity + Coriolis)
    tau_full = pin.rnea(model, data, q, v, np.zeros(model.nv))
    # 그러나 floating base는 unactuated → 0으로 설정
    tau_full_actuated = tau_full.copy()
    tau_full_actuated[:6] = 0.0   # base는 actuator 없음
    # Legs: PD on top of RNEA gravity comp
    q_target_dict = compute_q_target(t)
    for leg in LEG_NAMES:
        for i, qi_tgt in enumerate(q_target_dict[leg]):
            v_idx = leg_v_idx[leg][i]
            q_idx = leg_q_idx[leg][i]
            err  = qi_tgt - q[q_idx]
            err_v = -v[v_idx]
            # tau_full_actuated[v_idx]는 이미 RNEA gravity comp (base 빼고)
            tau_full_actuated[v_idx] += KP_JOINT * err + KD_JOINT * err_v
    return tau_full_actuated


# ── Direct pinocchio constraint dynamics (crocoddyl wrap 우회) ───
def build_contacts(stance_legs):
    """Contacts list for given stance legs."""
    contacts = []
    for leg in stance_legs:
        fr = model.frames[foot_frames[leg]]
        cm = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_3D,
            model, fr.parentJoint, fr.placement,
            pin.LOCAL_WORLD_ALIGNED)
        # baumgarte gains (kp=0, kd=10)
        cm.corrector.Kp[:] = 0.0
        cm.corrector.Kd[:] = 10.0
        contacts.append(cm)
    cdatas = [c.createData() for c in contacts]
    return contacts, cdatas


# 두 phase의 contact 미리 생성
contacts_A, cdatas_A = build_contacts(['FL', 'HR'])
contacts_B, cdatas_B = build_contacts(['FR', 'HL'])

# ── Simulate ───────────────────────────────────
print(f'Open-loop trot 시뮬 ({T_TOTAL}s, {N_FRAMES} frames, DT={DT_SIM}s)')
print(f'Total mass = {pin.computeTotalMass(model):.2f}kg')

q_hist = np.zeros((N_FRAMES + 1, model.nq))
v_hist = np.zeros((N_FRAMES + 1, model.nv))
x = x_init.copy()
q_hist[0] = x[:model.nq]
v_hist[0] = x[model.nq:]

t0 = time.time()
diverged_at = None
for fi in range(N_FRAMES):
    t = fi * DT_SIM
    stance, swing, _ = get_phase(t)
    tau = compute_torque_full(x, t)
    contacts = contacts_A if 'FL' in stance else contacts_B
    cdatas = cdatas_A if 'FL' in stance else cdatas_B

    try:
        q = x[:model.nq]; v = x[model.nq:]
        pin.initConstraintDynamics(model, data, contacts)
        a = pin.constraintDynamics(model, data, q, v, tau, contacts, cdatas)
        # Semi-implicit Euler
        v_next = v + DT_SIM * a
        q_next = pin.integrate(model, q, DT_SIM * v_next)
        x = np.concatenate([q_next, v_next])
    except Exception as e:
        if diverged_at is None:
            diverged_at = fi
            print(f'  ⚠ Dynamics 실패 at fi={fi} (t={t:.3f}s): {e}')
        x[model.nq:] = np.clip(x[model.nq:], -50, 50)

    if not np.all(np.isfinite(x)) or np.linalg.norm(x[model.nq:]) > 100:
        if diverged_at is None:
            diverged_at = fi
            print(f'  ⚠ 발산 감지 at fi={fi} (t={t:.3f}s)')
        x[model.nq:] = np.clip(x[model.nq:], -50, 50)
        if not np.all(np.isfinite(x)):
            x = x_init.copy()
            x[0] += V_TARGET * t

    q_hist[fi+1] = x[:model.nq]
    v_hist[fi+1] = x[model.nq:]

elapsed = time.time() - t0
print(f'시뮬 완료 ({elapsed*1e3:.0f}ms)')
print(f'발산 시작: {"frame "+str(diverged_at) if diverged_at is not None else "없음"}')

# ── 진단 ───────────────────────────────────────
body_x = q_hist[:, 0]; body_z = q_hist[:, 2]
body_vx = v_hist[:, 0]
quats = q_hist[:, 3:7]
rolls, pitches = [], []
for q_q in quats:
    R = pin.Quaternion(q_q[3], q_q[0], q_q[1], q_q[2]).toRotationMatrix()
    pitch = math.asin(max(-1, min(1, -R[2,0])))
    roll  = math.atan2(R[2,1], R[2,2])
    rolls.append(roll); pitches.append(pitch)
rolls = np.degrees(rolls); pitches = np.degrees(pitches)

print(f'\n━━ 진단 ━━')
print(f'Body x  : {body_x[0]*1e3:.0f} → {body_x[-1]*1e3:.0f} mm  (이동 {(body_x[-1]-body_x[0])*1e3:.0f}mm)')
print(f'Body z  : {body_z.min()*1e3:+.0f} ~ {body_z.max()*1e3:+.0f} mm  (start {body_z[0]*1e3:.0f})')
print(f'Body vx : {body_vx.min():+.2f} ~ {body_vx.max():+.2f} m/s')
print(f'Roll    : {rolls.min():+.1f} ~ {rolls.max():+.1f}°')
print(f'Pitch   : {pitches.min():+.1f} ~ {pitches.max():+.1f}°')

# ── 시각화 ─────────────────────────────────────
fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor('#1a1a2e')
ax = fig.add_subplot(121, projection='3d')
ax.set_facecolor('#16213e')
ax.tick_params(colors='gray')
ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.4, 0.4)
ax.set_zlim(-1.0, 0.6)
ax.set_xlabel('X rel body (m)', color='white')
ax.set_ylabel('Y (m)', color='white')
ax.set_zlabel('Z (m)', color='white')
ax.view_init(elev=20, azim=-55)
xx, yy = np.meshgrid([-0.6, 0.6], [-0.4, 0.4])
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.15, color='#888888')

LEG_COLORS = ['#00d4ff', '#ff6b35', '#00ff99', '#c264ff']
chassis_line, = ax.plot([], [], [], '-', color='white', lw=2.5)
leg_lines = []
for c in LEG_COLORS:
    lns = [ax.plot([], [], [], '-o', color=c, lw=2, ms=4)[0] for _ in range(5)]
    leg_lines.append(lns)
foot_traces = [ax.plot([], [], [], '-', color=c, lw=1, alpha=0.5)[0] for c in LEG_COLORS]
trace_buf = [[[], [], []] for _ in range(4)]
TRACE_LEN = 100
com_marker, = ax.plot([], [], [], 'X', color='yellow', ms=10)
time_text = ax.text2D(0.02, 0.97, '', transform=ax.transAxes, color='white', fontsize=11)

# Pre-compute leg link positions
hip_offsets = np.array(bm.LEG_HIP_OFFSETS)
base_link_id = model.getFrameId('base_link')
leg_link_hist = np.zeros((N_FRAMES+1, 4, 6, 3))
foot_hist = np.zeros((N_FRAMES+1, 4, 3))
for fi in range(N_FRAMES+1):
    pin.forwardKinematics(model, data, q_hist[fi])
    pin.updateFramePlacements(model, data)
    base_T = data.oMf[base_link_id]
    for li, leg in enumerate(LEG_NAMES):
        hip_w = base_T.translation + base_T.rotation @ hip_offsets[li]
        leg_link_hist[fi, li, 0] = hip_w
        for j in range(5):
            link_id = model.getFrameId(f'leg_{leg}_l{j+1}')
            leg_link_hist[fi, li, j+1] = data.oMf[link_id].translation
        foot_hist[fi, li] = data.oMf[foot_frames[leg]].translation


def update(fi):
    bx = q_hist[fi, 0]
    bases = leg_link_hist[fi, :, 0, :].copy(); bases[:, 0] -= bx
    chassis = np.array([bases[0], bases[2], bases[3], bases[1], bases[0]])
    chassis_line.set_data(chassis[:,0], chassis[:,1])
    chassis_line.set_3d_properties(chassis[:,2])
    com = q_hist[fi, :3].copy(); com[0] = 0.0
    com_marker.set_data([com[0]], [com[1]])
    com_marker.set_3d_properties([com[2]])
    for li in range(4):
        for j in range(5):
            A = leg_link_hist[fi, li, j].copy(); A[0] -= bx
            B = leg_link_hist[fi, li, j+1].copy(); B[0] -= bx
            leg_lines[li][j].set_data([A[0], B[0]], [A[1], B[1]])
            leg_lines[li][j].set_3d_properties([A[2], B[2]])
        fp = foot_hist[fi, li].copy(); fp[0] -= bx
        trace_buf[li][0].append(fp[0])
        trace_buf[li][1].append(fp[1])
        trace_buf[li][2].append(fp[2])
        foot_traces[li].set_data(
            trace_buf[li][0][-TRACE_LEN:], trace_buf[li][1][-TRACE_LEN:])
        foot_traces[li].set_3d_properties(trace_buf[li][2][-TRACE_LEN:])
    time_text.set_text(
        f't = {fi*DT_SIM:.3f}s ({fi}/{N_FRAMES})\n'
        f'body x = {bx:+.3f}m\n'
        f'roll = {rolls[fi]:+.1f}°  pitch = {pitches[fi]:+.1f}°\n'
        f'vx = {body_vx[fi]:.2f} m/s')
    return [chassis_line, com_marker, time_text] + sum(leg_lines, []) + foot_traces


ax.set_title(f'Open-loop trot (PD only) — '
              f'body falls/diverges as no NMPC stabilization',
              color='white', fontsize=11)

# Side plots
ax_z = fig.add_subplot(222)
ax_z.set_facecolor('#16213e'); ax_z.tick_params(colors='gray')
ax_z.set_title('Foot Z + body z over time', color='white', fontsize=10)
ax_z.set_ylabel('Z (m)', color='white')
ax_z.grid(alpha=0.3, color='gray')
t_axis = np.arange(N_FRAMES+1) * DT_SIM
for li, c in enumerate(LEG_COLORS):
    ax_z.plot(t_axis, foot_hist[:, li, 2], '-', color=c, label=LEG_NAMES[li], lw=1.2)
ax_z.plot(t_axis, body_z, '--', color='white', label='body z', lw=2)
ax_z.axhline(0, color='gray', ls=':', lw=0.5)
ax_z.legend(fontsize=8, ncol=5, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray')

ax_b = fig.add_subplot(224)
ax_b.set_facecolor('#16213e'); ax_b.tick_params(colors='gray')
ax_b.set_title('Body roll/pitch + vx', color='white', fontsize=10)
ax_b.set_xlabel('t (s)', color='white')
ax_b.grid(alpha=0.3, color='gray')
ax_b.plot(t_axis, rolls, '-', color='#ff6b35', label='roll (°)', lw=1.5)
ax_b.plot(t_axis, pitches, '-', color='#c264ff', label='pitch (°)', lw=1.5)
ax_b2 = ax_b.twinx(); ax_b2.tick_params(colors='gray')
ax_b2.plot(t_axis, body_vx, '-', color='#00d4ff', label='vx (m/s)', lw=1)
ax_b.legend(fontsize=8, loc='upper left', facecolor='#1a1a2e',
             labelcolor='white', edgecolor='gray')
ax_b2.legend(fontsize=8, loc='upper right', facecolor='#1a1a2e',
              labelcolor='white', edgecolor='gray')

ani = FuncAnimation(fig, update, frames=N_FRAMES+1, interval=40,
                     blit=False, repeat=True)
gif_path = '/tmp/openloop_trot_fall.gif'
ani.save(gif_path, fps=25, writer='pillow')
print(f'\n시각화 저장 → {gif_path}')

png_path = '/tmp/openloop_trot_fall.png'
fig.savefig(png_path, facecolor='#1a1a2e', dpi=110)
print(f'정적 plot → {png_path}')
plt.show()
