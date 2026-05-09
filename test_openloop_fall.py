"""
Open-loop trot 시뮬 + Hunt-Crossley compliant ground contact.

v10/v11과 동일한 아이디어로 단순한 시뮬:
    1. 발 궤적 cycloid swing (body-frame)
    2. q_target(t) PD 토크 + 중력 보상
    3. **Hunt-Crossley ground contact** (실제 floor 추가):
        F_z = K·δ·(1 + B·δ̇)  (n=1, δ = penetration)
        F_xy = -μ·F_z·v_xy/|v_xy|  (Coulomb 마찰, 마찰 추 한계)
    4. pin.aba (free dynamics) + ground forces via J^T
    5. 시뮬 진행 → robot이 trot하다 점차 발산하면 floor에 부딪혀 settle

목적: NMPC 없이도 robot이 ground에 정착하는 모습 → 자연스러운 fall+rest 시각화.
"""
import math
import time
import numpy as np
import pinocchio as pin
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import build_pin_model as bm

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ── 시뮬 파라미터 ──────────────────────────────
V_TARGET    = 1.0
T_PERIOD    = 0.5
DT_SIM      = 0.0005   # 0.5ms (free-fall + body-leg coupling용 작은 DT)
T_TOTAL     = 1.0
N_FRAMES    = int(T_TOTAL / DT_SIM)
STEP_HEIGHT = 0.08
STEP_LENGTH = 0.25

# Joint PD gains (강한 KP로 leg를 home에 stiff하게 hold = quasi-rigid body 효과)
# trot pattern 줄이려면 STIFF_JOINT_HOLD=True로 (leg 동작 없이 ground physics만 보기)
STIFF_JOINT_HOLD = True   # True: leg 강하게 home에 hold, gait 없음
KP_JOINT = 1000.0 if STIFF_JOINT_HOLD else 50.0
KD_JOINT = 50.0   if STIFF_JOINT_HOLD else 5.0

# Ground (z=0) parameters: 단순 linear spring-damper + Coulomb 마찰
GROUND_K   = 1e4    # 강성 [N/m] — robot 40kg, 한 발 ~10kg → δ_eq ≈ 1cm
GROUND_B   = 200.0  # damping [N·s/m]
GROUND_MU  = 0.7    # 마찰 계수
GROUND_VEL_TOL = 0.1   # 마찰 sliding 임계 [m/s] (regularization)

# ── 모델 + 초기 상태 ──────────────────────────
model = bm.build_model()
data  = model.createData()
LEG_NAMES = ['FR', 'FL', 'HR', 'HL']

Q_HOME_FRONT = [0.0, math.radians(133.2973), math.radians(46.7027),
                math.radians(30.6583), math.radians(59.3417)]
Q_HOME_HIND  = [0.0, math.radians(-150.0), math.radians(-90.0),
                math.radians(90.0), math.radians(60.0)]
Q_HOME_PER_LEG = {'FR': Q_HOME_FRONT, 'FL': Q_HOME_FRONT,
                   'HR': Q_HOME_HIND,  'HL': Q_HOME_HIND}

q_init = pin.neutral(model)
for leg, qh in [('FR', Q_HOME_FRONT), ('FL', Q_HOME_FRONT),
                 ('HR', Q_HOME_HIND), ('HL', Q_HOME_HIND)]:
    for i, qi in enumerate(qh):
        q_init[model.idx_qs[model.getJointId(f'leg_{leg}_j{i+1}')]] = qi
pin.forwardKinematics(model, data, q_init)
pin.updateFramePlacements(model, data)
foot_z = data.oMf[model.getFrameId('leg_FR_foot')].translation[2]
q_init[2] = -foot_z + 0.05    # 5cm 위에서 시작 (떨어져서 ground 충격 + 정착 보기)
v_init = np.zeros(model.nv)

leg_v_idx = {leg: [model.idx_vs[model.getJointId(f'leg_{leg}_j{i+1}')] for i in range(5)]
             for leg in LEG_NAMES}
leg_q_idx = {leg: [model.idx_qs[model.getJointId(f'leg_{leg}_j{i+1}')] for i in range(5)]
             for leg in LEG_NAMES}
foot_frames = {leg: model.getFrameId(f'leg_{leg}_foot') for leg in LEG_NAMES}


# ── Gait scheduler ────────────────────────────
def get_phase(t):
    cycle_t = t % T_PERIOD
    if cycle_t < T_PERIOD / 2:
        sw_t = cycle_t / (T_PERIOD / 2)
        return ['FL', 'HR'], ['FR', 'HL'], sw_t
    else:
        sw_t = (cycle_t - T_PERIOD/2) / (T_PERIOD / 2)
        return ['FR', 'HL'], ['FL', 'HR'], sw_t


def compute_q_target(t):
    """q_target(t): home + swing leg에 lift 패턴 (STIFF_JOINT_HOLD=True면 항상 home)."""
    if STIFF_JOINT_HOLD:
        return {leg: list(Q_HOME_PER_LEG[leg]) for leg in LEG_NAMES}
    stance, swing, sw_t = get_phase(t)
    q_target = {}
    for leg in LEG_NAMES:
        q_t = list(Q_HOME_PER_LEG[leg])
        if leg in swing:
            lift = 4.0 * sw_t * (1.0 - sw_t)
            if leg in ['FR', 'FL']:
                q_t[1] -= 0.25 * lift
                q_t[2] += 0.5  * lift
                q_t[3] -= 0.25 * lift
            else:
                q_t[1] += 0.25 * lift
                q_t[2] -= 0.5  * lift
                q_t[3] += 0.25 * lift
        q_target[leg] = q_t
    return q_target


# ── Hunt-Crossley ground contact ─────────────
def ground_contact_forces(q, v):
    """각 발의 ground 접촉력 (penetration 기반).
    Returns: dict {leg: F_world (3,)}, 발이 ground 위면 0.
    """
    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q)

    forces = {}
    for leg in LEG_NAMES:
        fid = foot_frames[leg]
        foot_pos = data.oMf[fid].translation
        # foot velocity in world (linear part)
        J6 = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
        J = J6[:3]  # 3×nv
        v_foot = J @ v   # world frame foot velocity

        z = foot_pos[2]
        if z >= 0:
            forces[leg] = (np.zeros(3), J)   # 공중 → 0 force
            continue

        delta = -z                         # penetration depth ≥ 0
        delta_dot = -v_foot[2]              # penetration rate (positive = moving down)

        # 단순 linear spring + damper (compression damping only)
        F_z = GROUND_K * delta + GROUND_B * max(delta_dot, 0.0)
        F_z = max(0.0, F_z)             # ground는 push만 (no pull)

        # Coulomb friction (tangential)
        v_tan = v_foot[:2]
        v_tan_norm = np.linalg.norm(v_tan)
        if v_tan_norm > GROUND_VEL_TOL:
            F_tan = -GROUND_MU * F_z * v_tan / v_tan_norm
        else:
            # 정지 마찰 (regularized): velocity 작으면 stiff
            F_tan = -GROUND_MU * F_z * v_tan / GROUND_VEL_TOL
        F = np.array([F_tan[0], F_tan[1], F_z])
        forces[leg] = (F, J)
    return forces


def compute_torque_full(q, v, t):
    """Leg PD + RNEA gravity comp (base unactuated)."""
    tau_full = pin.rnea(model, data, q, v, np.zeros(model.nv))
    tau_full[:6] = 0.0   # base 6 unactuated

    q_target_dict = compute_q_target(t)
    for leg in LEG_NAMES:
        for i, qi_tgt in enumerate(q_target_dict[leg]):
            v_idx = leg_v_idx[leg][i]
            q_idx = leg_q_idx[leg][i]
            err  = qi_tgt - q[q_idx]
            err_v = -v[v_idx]
            tau_full[v_idx] += KP_JOINT * err + KD_JOINT * err_v
    return tau_full


# ── Main sim loop ──────────────────────────────
print(f'Open-loop trot + Hunt-Crossley ground ({T_TOTAL}s, {N_FRAMES} frames)')
print(f'Total mass = {pin.computeTotalMass(model):.2f}kg')
print(f'Ground: K={GROUND_K:.0f} N/m, B={GROUND_B:.0f}, μ={GROUND_MU}')

q_hist = np.zeros((N_FRAMES + 1, model.nq))
v_hist = np.zeros((N_FRAMES + 1, model.nv))
foot_z_hist = np.zeros((N_FRAMES + 1, 4))
foot_F_hist = np.zeros((N_FRAMES + 1, 4, 3))

q = q_init.copy()
v = v_init.copy()
q_hist[0] = q
v_hist[0] = v

t0 = time.time()
diverged_at = None
for fi in range(N_FRAMES):
    t = fi * DT_SIM

    # 1. 토크
    tau = compute_torque_full(q, v, t)

    # 2. Ground contact forces (각 발 별)
    forces = ground_contact_forces(q, v)
    # τ_total = τ_actuated + Σ J_i^T · F_i_ground
    f_ext_full = np.zeros(model.nv)
    for leg, (F, J) in forces.items():
        f_ext_full += J.T @ F
    tau_total = tau + f_ext_full

    # 3. Forward dynamics (free + ground)
    a = pin.aba(model, data, q, v, tau_total)

    # 4. Integrate (semi-implicit)
    v = v + DT_SIM * a
    q = pin.integrate(model, q, DT_SIM * v)

    # 5. 발산 가드
    if not np.all(np.isfinite(q)) or np.linalg.norm(v) > 100:
        if diverged_at is None:
            diverged_at = fi
            print(f'  ⚠ 발산 at fi={fi} (t={t:.3f}s)')
        v = np.clip(v, -50, 50)
        if not np.all(np.isfinite(q)):
            q = q_init.copy()

    q_hist[fi+1] = q
    v_hist[fi+1] = v
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    for li, leg in enumerate(LEG_NAMES):
        foot_z_hist[fi+1, li] = data.oMf[foot_frames[leg]].translation[2]
        if leg in forces:
            foot_F_hist[fi+1, li] = forces[leg][0]

elapsed = time.time() - t0
print(f'시뮬 완료 ({elapsed*1e3:.0f}ms)')
print(f'발산: {"frame "+str(diverged_at) if diverged_at is not None else "안 함"}')

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
print(f'Body x  : {body_x[0]*1e3:.0f} → {body_x[-1]*1e3:.0f} mm')
print(f'Body z  : {body_z.min()*1e3:+.0f} ~ {body_z.max()*1e3:+.0f} mm  (start {body_z[0]*1e3:.0f})')
print(f'Body vx : {body_vx.min():+.2f} ~ {body_vx.max():+.2f} m/s')
print(f'Roll    : {rolls.min():+.1f} ~ {rolls.max():+.1f}°')
print(f'Pitch   : {pitches.min():+.1f} ~ {pitches.max():+.1f}°')
print(f'F_z max : {foot_F_hist[:,:,2].max():.0f} N (per foot peak)')

# ── 시각화 ─────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#1a1a2e')

ax = fig.add_subplot(121, projection='3d')
ax.set_facecolor('#16213e')
ax.tick_params(colors='gray')
x_min, x_max = body_x.min() - 0.3, body_x.max() + 0.3
ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.4, 0.4)
ax.set_zlim(-0.05, 0.6)
ax.set_xlabel('X (m)', color='white')
ax.set_ylabel('Y (m)', color='white')
ax.set_zlabel('Z (m)', color='white')
ax.view_init(elev=15, azim=-65)

# Floor (z=0) — 진짜 ground
xx, yy = np.meshgrid([x_min, x_max], [-0.4, 0.4])
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.3, color='#666666', edgecolor='gray')

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
foot_pos_hist = np.zeros((N_FRAMES+1, 4, 3))
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
        foot_pos_hist[fi, li] = data.oMf[foot_frames[leg]].translation


# 표시는 매 5 frame (DT=2ms*5=10ms)
DISPLAY_STRIDE = 5

def update(fi_idx):
    fi = fi_idx * DISPLAY_STRIDE
    if fi > N_FRAMES:
        fi = N_FRAMES
    bases = leg_link_hist[fi, :, 0, :].copy()
    chassis = np.array([bases[0], bases[2], bases[3], bases[1], bases[0]])
    chassis_line.set_data(chassis[:,0], chassis[:,1])
    chassis_line.set_3d_properties(chassis[:,2])
    com = q_hist[fi, :3]
    com_marker.set_data([com[0]], [com[1]])
    com_marker.set_3d_properties([com[2]])
    for li in range(4):
        for j in range(5):
            A = leg_link_hist[fi, li, j]
            B = leg_link_hist[fi, li, j+1]
            leg_lines[li][j].set_data([A[0], B[0]], [A[1], B[1]])
            leg_lines[li][j].set_3d_properties([A[2], B[2]])
        fp = foot_pos_hist[fi, li]
        trace_buf[li][0].append(fp[0])
        trace_buf[li][1].append(fp[1])
        trace_buf[li][2].append(fp[2])
        foot_traces[li].set_data(
            trace_buf[li][0][-TRACE_LEN:], trace_buf[li][1][-TRACE_LEN:])
        foot_traces[li].set_3d_properties(trace_buf[li][2][-TRACE_LEN:])
    time_text.set_text(
        f't = {fi*DT_SIM:.3f}s ({fi}/{N_FRAMES})\n'
        f'body = ({com[0]:+.2f}, {com[1]:+.2f}, {com[2]:+.2f})m\n'
        f'roll = {rolls[fi]:+.1f}°  pitch = {pitches[fi]:+.1f}°\n'
        f'vx = {body_vx[fi]:.2f} m/s')
    return [chassis_line, com_marker, time_text] + sum(leg_lines, []) + foot_traces


ax.set_title(f'Open-loop trot + Hunt-Crossley ground (K={GROUND_K:.0e}N/m) — '
              f'real floor at z=0',
              color='white', fontsize=11)

# Side plots
ax_z = fig.add_subplot(222)
ax_z.set_facecolor('#16213e'); ax_z.tick_params(colors='gray')
ax_z.set_title('Foot Z + body z (real ground at z=0)', color='white', fontsize=10)
ax_z.set_ylabel('Z (m)', color='white')
ax_z.grid(alpha=0.3, color='gray')
t_axis = np.arange(N_FRAMES+1) * DT_SIM
for li, c in enumerate(LEG_COLORS):
    ax_z.plot(t_axis, foot_z_hist[:, li], '-', color=c, label=LEG_NAMES[li], lw=1)
ax_z.plot(t_axis, body_z, '--', color='white', label='body z', lw=2)
ax_z.axhline(0, color='red', ls='-', lw=0.8, alpha=0.7, label='floor')
ax_z.legend(fontsize=8, ncol=6, facecolor='#1a1a2e', labelcolor='white', edgecolor='gray')

ax_b = fig.add_subplot(224)
ax_b.set_facecolor('#16213e'); ax_b.tick_params(colors='gray')
ax_b.set_title('Body roll/pitch + ground F_z (per foot)', color='white', fontsize=10)
ax_b.set_xlabel('t (s)', color='white')
ax_b.grid(alpha=0.3, color='gray')
ax_b.plot(t_axis, rolls, '-', color='#ff6b35', label='roll (°)', lw=1.2)
ax_b.plot(t_axis, pitches, '-', color='#c264ff', label='pitch (°)', lw=1.2)
ax_b2 = ax_b.twinx(); ax_b2.tick_params(colors='gray')
for li, c in enumerate(LEG_COLORS):
    ax_b2.plot(t_axis, foot_F_hist[:, li, 2], '-', color=c, lw=0.8, alpha=0.7)
ax_b.legend(fontsize=8, loc='upper left', facecolor='#1a1a2e',
             labelcolor='white', edgecolor='gray')
ax_b2.set_ylabel('F_z (N)', color='white')

n_disp = N_FRAMES // DISPLAY_STRIDE + 1
ani = FuncAnimation(fig, update, frames=n_disp, interval=50,
                     blit=False, repeat=True)
gif_path = '/tmp/openloop_trot_with_ground.gif'
ani.save(gif_path, fps=20, writer='pillow')
print(f'\n시각화 저장 → {gif_path}')

png_path = '/tmp/openloop_trot_with_ground.png'
fig.savefig(png_path, facecolor='#1a1a2e', dpi=110)
print(f'정적 plot → {png_path}')
plt.show()
