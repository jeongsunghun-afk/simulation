"""
leg_statics_v4.py
v1 (동역학: 피드포워드 토크 + PD 제어 + GRF 역산)
v2 (3D 해석적 IK 궤적)
v3: Impedance Control 추가 (Cartesian 공간에서 발끝 위치/속도 오차에 비례하는 보정 토크)
v4: 힘 피드백 보정 방향 수정 (λ_calc - λ_des → λ_des - λ_calc)

[제어]
  tau_cmd = Kp(θt-θa) + Kd(θ̇t-θ̇a) + τ_ff
  tau_ff  = J^T · λ_des               [N·m]  (목표 GRF → 관절 토크)
  λ_calc = (J·J^T + μI)^{-1} · J · τ_cmd  [N]  (τ_cmd → GRF 역산)

[자코비안]
  회전 관절: J[:,i] = z_i × (p_e - p_i)
  DH 기반 3D 자코비안 (3×5)

[좌표 기준]
  힙 = 원점.  다리가 -X 방향으로 신전.
  GRF 주성분: +X 방향 (지면 법선 = 힙 방향)
"""

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ── 한글 폰트
for key in mpl.rcParams:
    if key.startswith("keymap."):
        mpl.rcParams[key] = []
mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════
# 0. 파라미터
# ══════════════════════════════════════════════════════════════
TRAJ_MODE = 'jump'    # 'jump' | 'gait'
USE_IMP   = True      # True: Impedance Control 활성화 / False: 비활성화 (비교용)
DT        = 0.005     # 시간 간격 [s] — 하드웨어 제어 주기에 맞춰 조정
N_STEPS   = 120       # waypoint 총 개수 (총 시간 = N_STEPS × DT)

DH_PARAMS = [
    (-math.pi/2, 0.0,    0.0    ),   # Joint 1 : Hip Abduction
    (0.0,        0.21,   0.0075 ),   # Joint 2 : Hip Pitch
    (0.0,        0.21,   0.0    ),   # Joint 3 : Knee
    (0.0,        0.148,  0.0    ),   # Joint 4 : Ankle
    (0.0,        0.0,    0.0    ),   # Joint 5 : Toe
]

Q_INIT = [math.radians(a) for a in [0,  -90,   0,  0,  0]]
Q_HOME = [math.radians(a) for a in [0, -150, -90, 90, 60]]

# IK 링크 파라미터 (DH_PARAMS 동기화)
_A2 = DH_PARAMS[1][1]   # 0.21 m
_A3 = DH_PARAMS[2][1]   # 0.21 m
_A4 = DH_PARAMS[3][1]   # 0.148 m
_D2 = DH_PARAMS[1][2]   # 0.0075 m

# PD 제어 게인 (관절 1~5)
Kp = np.array([30.0, 80.0, 80.0, 60.0, 20.0])   # [N·m/rad]
Kd = np.array([ 3.0,  8.0,  8.0,  6.0,  2.0])   # [N·m·s/rad]

# 추종 오차 모델 (1차 지연)
TAU_LAG  = 0.03                                  # 지연 상수 [s]
INIT_ERR = np.deg2rad([1.0, -2.0, 2.0, -1.5, 0.5])  # 초기 각도 오차

# 링크 질량 [kg]  (J1:Hip Abduction, J2:Thigh, J3:Shin, J4:Foot, J5:Toe)
LINK_MASS = np.array([3.34, 0.8, 0.2, 0.2, 0.05]) #link1, link2, link3, link4, link5 질량
# 80형번 0.5kg, 90형번 1.42kg
#LINK_MASS         = np.array([4.125, 1.215, 0.2, 0.2, 0.05])  # link1~5 질량 [kg] 
# 80형번 0.915kg, 90형번 1.605kg
G         = 9.81   # 중력 가속도 [m/s²]
G_VEC     = np.array([-G, 0.0, 0.0])  # 중력 방향 (월드 기준, -X = 아래)

# Impedance Control 게인 (Cartesian 공간)
Kp_imp = np.array([800.0, 800.0, 800.0])   # [N/m]
Kd_imp = np.array([ 40.0,  40.0,  40.0])   # [N·s/m]

# GRF 설정
F_PEAK   = 400.0    # 최대 접촉력 [N]
MU_DAMP  = 1e-3    # 자코비안 댐핑 계수 (특이점 방지)

# 힘 피드백 게인
Kf = np.array([0.1, 0.1, 0.1])   # λ_err → λ_des 보정 게인(0.1~0.5 정도가 적당)


# ══════════════════════════════════════════════════════════════
# 1. FK / 해석적 IK / 자코비안
# ══════════════════════════════════════════════════════════════

def get_dh_matrix(alpha, a, d, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ], dtype=float)

def forward_kinematics(thetas):
    T = np.eye(4)
    pts = [np.zeros(3)]
    for i, (alpha, a, d) in enumerate(DH_PARAMS):
        T = T @ get_dh_matrix(alpha, a, d, thetas[i])
        pts.append(T[:3, 3].copy())
    return pts

def analytical_ik(Px, Py, Pz, phi, theta5_target, elbow_up=True):
    """해석적 역기구학 (leg_IK3.py 참조)"""
    D2 = Px**2 + Py**2 - _D2**2
    if D2 < 0:
        return None
    R = math.sqrt(D2)
    theta1 = math.atan2(-Px, Py) - math.atan2(R, _D2)
    c1, s1 = math.cos(theta1), math.sin(theta1)
    x_s = c1 * Px + s1 * Py
    Z   = -Pz
    x3 = x_s - _A4 * math.cos(phi)
    z3 = Z   - _A4 * math.sin(phi)
    cos_th3 = (x3**2 + z3**2 - _A2**2 - _A3**2) / (2.0 * _A2 * _A3)
    cos_th3 = max(-1.0, min(1.0, cos_th3))
    theta3  = -math.acos(cos_th3) if elbow_up else math.acos(cos_th3)
    theta2 = (math.atan2(z3, x3)
              - math.atan2(_A3 * math.sin(theta3),
                           _A2 + _A3 * math.cos(theta3)))
    theta4 = phi - theta2 - theta3
    theta5 = theta5_target - (theta2 + theta3 + theta4)
    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi
    return [wrap(theta1), wrap(theta2), wrap(theta3), wrap(theta4), wrap(theta5)]

def _get_origins_zaxes(thetas):
    """DH 순기구학으로 관절 원점·z축 배열 반환 (공통 유틸)"""
    n = len(thetas)
    T = np.eye(4)
    origins = [np.zeros(3)]
    z_axes  = [np.array([0.0, 0.0, 1.0])]
    for i in range(n):
        alpha, a, d = DH_PARAMS[i]
        T = T @ get_dh_matrix(alpha, a, d, thetas[i])
        origins.append(T[:3, 3].copy())
        z_axes.append(T[:3, 2].copy())
    return origins, z_axes

def compute_jacobian(thetas):
    """
    3D 위치 자코비안 (3×5)
    J[:,i] = z_i × (p_e - p_i)
    """
    origins, z_axes = _get_origins_zaxes(thetas)
    pe = origins[-1]
    n  = len(thetas)
    J  = np.zeros((3, n))
    for i in range(n):
        J[:, i] = np.cross(z_axes[i], pe - origins[i])
    return J

def compute_gravity_torque(thetas):
    """
    링크 무게에 의한 중력 보상 토크 (5×1) [N·m]
    τ_grav[j] = Σ_{k≥j}  (z_j × (p_com_k − p_j)) · (m_k · G_VEC)

    p_com_k : 링크 k의 무게중심 = 관절 k와 k+1 원점의 중간점
    """
    origins, z_axes = _get_origins_zaxes(thetas)
    n      = len(thetas)
    tau_g  = np.zeros(n)
    for k in range(n):                          # 링크 k (관절 k → k+1 사이)
        p_com   = (origins[k] + origins[k+1]) / 2.0
        f_grav  = LINK_MASS[k] * G_VEC         # 중력 하중 [N]
        for j in range(k + 1):                  # 관절 j (j ≤ k 인 관절이 링크 k에 영향)
            tau_g[j] += np.dot(np.cross(z_axes[j], p_com - origins[j]), f_grav)
    return tau_g

# ══════════════════════════════════════════════════════════════
# 2. 궤적 생성 (v2 동일)
# ══════════════════════════════════════════════════════════════

def solve_quintic_spline(t0, tf, p0, v0, a0, pf, vf, af):
    T_mat = np.array([
        [1, t0, t0**2,    t0**3,    t0**4,    t0**5],
        [0,  1, 2*t0,  3*t0**2,  4*t0**3,  5*t0**4],
        [0,  0,    2,    6*t0, 12*t0**2, 20*t0**3 ],
        [1, tf, tf**2,    tf**3,    tf**4,    tf**5],
        [0,  1, 2*tf,  3*tf**2,  4*tf**3,  5*tf**4],
        [0,  0,    2,    6*tf, 12*tf**2, 20*tf**3 ],
    ])
    return np.linalg.solve(T_mat, np.array([p0, v0, a0, pf, vf, af]))

def eval_quintic(c, t):
    pos = c[0]+c[1]*t    +c[2]*t**2    +c[3]*t**3    +c[4]*t**4    +c[5]*t**5
    vel = c[1]+2*c[2]*t  +3*c[3]*t**2  +4*c[4]*t**3  +5*c[5]*t**4
    acc = 2*c[2]+6*c[3]*t+12*c[4]*t**2+20*c[5]*t**3
    return pos, vel, acc

def make_jump_trajectory(start, h_crouch=0.04, h_jump=0.06, n=120):
    n1 = n // 3;  n2 = n // 6;  n3 = n - n1 - n2
    x0 = start[0]
    c1 = solve_quintic_spline(0, n1*DT, x0,          0, 0, x0+h_crouch, 0, 0)
    c2 = solve_quintic_spline(0, n2*DT, x0+h_crouch, 0, 0, x0-h_jump,   0, 0)
    c3 = solve_quintic_spline(0, n3*DT, x0-h_jump,   0, 0, x0,          0, 0)
    pts = []
    for j in range(n1):
        x, _, _ = eval_quintic(c1, j*DT);  pts.append([x, start[1], start[2]])
    for j in range(n2):
        x, _, _ = eval_quintic(c2, j*DT);  pts.append([x, start[1], start[2]])
    for j in range(n3):
        x, _, _ = eval_quintic(c3, j*DT);  pts.append([x, start[1], start[2]])
    return np.array(pts), (n1, n1 + n2)

def make_gait_trajectory(start, step_x=0.06, lift=0.04, n=120):
    n_stance = n // 2;  n_swing = n - n_stance
    t1 = np.linspace(0, 1, n_stance)
    sx = start[0] + step_x * t1
    t2 = np.linspace(0, np.pi, n_swing)
    wx = (start[0] + step_x) - step_x * (1 - np.cos(t2)) / 2
    wz = start[2] + lift * np.sin(t2)
    wx[-1] = start[0];  wz[-1] = start[2]
    pts_stance = np.column_stack([sx, np.full(n_stance, start[1]), np.full(n_stance, start[2])])
    pts_swing  = np.column_stack([wx, np.full(n_swing, start[1]),  wz])
    return np.vstack([pts_stance, pts_swing]), None


# ══════════════════════════════════════════════════════════════
# 3. 사전 계산 — IK
# ══════════════════════════════════════════════════════════════
print("─" * 55)
print("궤적 IK 계산 중...")
toe_start = np.array(forward_kinematics(Q_HOME)[4])
print(f"  발끝 시작점:  X={toe_start[0]*1e3:.1f}mm  "
      f"Y={toe_start[1]*1e3:.1f}mm  Z={toe_start[2]*1e3:.1f}mm")

if TRAJ_MODE == 'jump':
    trajectory, phase_idx = make_jump_trajectory(toe_start, h_crouch=0.02, h_jump=0.03, n=N_STEPS)
    mode_label = '수직 점프 (준비→도약→착지)'
else:
    trajectory, phase_idx = make_gait_trajectory(toe_start, step_x=0.06, lift=0.04, n=N_STEPS)
    mode_label = '보행 (Stance→Swing)'

q_a = [Q_HOME[:]]
current    = Q_HOME[:]
for target in trajectory:
    phi           = current[1] + current[2] + current[3]
    theta5_target = current[1] + current[2] + current[3] + current[4]
    result = analytical_ik(target[0], target[1], target[2], phi, theta5_target)
    if result is not None:
        current = result
    q_a.append(current[:])

q_a = np.array(q_a)   # (N, 5)
n_frames   = len(q_a)
x_a   = np.array([forward_kinematics(th)[4] for th in q_a])
print(f"IK 완료: {n_frames} 프레임  |  {mode_label}")


# ══════════════════════════════════════════════════════════════
# 4. 사전 계산 — 동역학
# ══════════════════════════════════════════════════════════════
print("동역학 계산 중...")

# ── GRF 프로파일 (N×3) : 주성분 = Fx (+X, 지면 법선)
lam_des = np.zeros((n_frames, 3))
if TRAJ_MODE == 'jump':
    n1_p, n2_p = phase_idx
    n3_p = n_frames - n2_p
    # Phase 1 (준비-crouch): 0 → F_PEAK×0.4
    lam_des[:n1_p, 0]    = F_PEAK * 0.4 * np.sin(np.linspace(0, np.pi/2, n1_p))
    # Phase 2 (도약-push): sin 최대
    lam_des[n1_p:n2_p, 0] = F_PEAK * np.sin(np.linspace(0, np.pi, n2_p - n1_p))
    # Phase 3 (착지): 충격 흡수
    lam_des[n2_p:, 0]    = F_PEAK * 0.5 * np.sin(np.linspace(0, np.pi/2, n3_p))
else:
    n_st = n_frames // 2
    lam_des[:n_st, 0] = F_PEAK * np.sin(np.linspace(0, np.pi, n_st))

# ── 목표각 & 속도
theta_t  = q_a.copy()
dtheta_t = np.zeros_like(theta_t)
dtheta_t[1:] = np.diff(theta_t, axis=0) / DT

# ── Cartesian 참조 궤적 & 속도 (직교좌표 기준)
x_t     = x_a.copy()                           # (n_frames, 3)  발끝 위치 참조
x_t_dot = np.zeros_like(x_t)
x_t_dot[1:] = np.diff(x_t, axis=0) / DT      # 발끝 속도 참조

# ── 실제각 (1차 지연 추종)
theta_a  = np.zeros_like(theta_t)
dtheta_a = np.zeros_like(theta_t)
theta_a[0] = theta_t[0] + INIT_ERR
for i in range(1, n_frames):
    theta_a[i]  = theta_a[i-1] + (DT / TAU_LAG) * (theta_t[i-1] - theta_a[i-1])
    dtheta_a[i] = (theta_a[i] - theta_a[i-1]) / DT

# ── 토크 & GRF 역산
tau_ff     = np.zeros((n_frames, 5))
tau_pd     = np.zeros((n_frames, 5))
tau_imp    = np.zeros((n_frames, 5))
tau_offset = np.zeros((n_frames, 5))
tau_cmd    = np.zeros((n_frames, 5))
tau_grav   = np.zeros((n_frames, 5))
lam_calc   = np.zeros((n_frames, 3))
lam_fb     = lam_des.copy()           # 힘 피드백으로 보정되는 실시간 λ

for i in range(n_frames):
    J   = compute_jacobian(theta_t[i])
    J_a = compute_jacobian(theta_a[i])

    # WBIC (G - J^T · λ_fb)  ← 힘 피드백 보정값 사용
    tau_grav[i] = compute_gravity_torque(theta_t[i])
    tau_ff[i]   = tau_grav[i] - J.T @ lam_fb[i]

    # Impedance Control
    x_t_i  = x_t[i]
    x_a_i  = forward_kinematics(theta_a[i])[-1]
    dx_t_i = x_t_dot[i]
    dx_a_i = J_a @ dtheta_a[i]
    f_imp        = Kp_imp * (x_t_i - x_a_i) + Kd_imp * (dx_t_i - dx_a_i)
    tau_imp[i]   = J.T @ f_imp if USE_IMP else np.zeros(5)
    tau_offset[i] = tau_imp[i] + tau_ff[i]

    # CSP
    tau_pd[i]  = Kp * (theta_t[i] - theta_a[i]) + Kd * (dtheta_t[i] - dtheta_a[i])
    tau_cmd[i] = tau_pd[i] + tau_offset[i]

    # GRF 역산
    JJT = J @ J.T + MU_DAMP * np.eye(3)
    lam_calc[i] = np.linalg.solve(JJT, J @ (tau_grav[i] - tau_cmd[i]))

    # 힘 피드백: λ_fb[i+1] = λ_fb[i] + Kf · (λ_des[i] - λ_calc[i])
    # lam_calc ≈ lam_fb 이므로 오차 = lam_des - lam_fb → (1-Kf) 수렴 조건
    if i + 1 < n_frames:
        lam_fb[i+1] = lam_fb[i] + Kf * (lam_des[i] - lam_calc[i])

print(f"Dynamcis_calc  [Impedance: {'ON' if USE_IMP else 'OFF'}]")
i_pk = int(np.argmax(lam_des[:, 0]))
print(f"\n=== peak frame (i={i_pk}, GRF Fx 최대) ===")
print(f"link_mass  = {np.round(LINK_MASS, 3)} kg")
print(f"tau_cmd  (th1~5)  = {np.round(tau_cmd[i_pk], 3)} N·m\n")
print(f"tau_pd   (th1~5)  = {np.round(tau_pd[i_pk], 3)} N·m")
print(f"tau_offset (th1~5) = {np.round(tau_offset[i_pk], 3)} N·m\n")
print(f"tau_imp  (th1~5)  = {np.round(tau_imp[i_pk], 3)} N·m")
print(f"tau_ff   (th1~5)  = {np.round(tau_ff[i_pk], 3)} N·m")
tau_grav_pk = compute_gravity_torque(theta_t[i_pk])
print(f"tau_grav (th1~5)  = {np.round(tau_grav_pk, 3)} N·m\n")
print(f"lam_des  (Fx)    = {lam_des[i_pk,0]:+.2f} N")
print(f"lam_calc (Fx)    = {lam_calc[i_pk,0]:+.2f} N")
max_speed = np.max(np.abs(dtheta_t), axis=0)
print(f"\n=== max joint speed ===")
print(f"  [rad/s] = {np.round(max_speed, 3)}")
print(f"  [RPS]   = {np.round(max_speed / (2 * np.pi), 3)}")

if TRAJ_MODE == 'jump':
    M_body       = float(np.sum(LINK_MASS))                        # 총 링크 질량 [kg]
    n1_p, n2_p   = phase_idx
    impulse      = np.sum(lam_calc[n1_p:n2_p, 0]) * DT            # push-off 충격량 [N·s]
    grav_impulse = M_body * G * (n2_p - n1_p) * DT                # 중력 충격량 [N·s]
    net_impulse  = impulse - grav_impulse                          # 순 충격량 [N·s]
    v_takeoff    = net_impulse / M_body                            # 이륙 속도 [m/s]
    h_jump       = v_takeoff**2 / (2 * G) if v_takeoff > 0 else 0.0
    print(f"\n=== jump height estimate ===")
    print(f"  M_body        = {M_body:.3f} kg  (link mass total)")
    print(f"  push-off T    = {(n2_p-n1_p)*DT:.3f} s  (frames {n1_p}~{n2_p})")
    print(f"  GRF impulse   = {impulse:+.3f} N·s")
    print(f"  grav impulse  = {grav_impulse:+.3f} N·s")
    print(f"  net impulse   = {net_impulse:+.3f} N·s")
    print(f"  v_takeoff     = {v_takeoff:+.3f} m/s")
    print(f"  h_jump        = {h_jump:.4f} m")
max_fh_mm  = x_a[:, 0].max() * 1e3
min_fh_mm  = x_a[:, 0].min() * 1e3
home_x_mm  = toe_start[0] * 1e3
print(f"\n=== max foot height ===")
print(f"  foot X max = {max_fh_mm:.1f} mm  (Δ home: {max_fh_mm - home_x_mm:+.1f} mm)")
print(f"  foot X min = {min_fh_mm:.1f} mm  (Δ home: {min_fh_mm - home_x_mm:+.1f} mm)")
print("─" * 55)

# ══════════════════════════════════════════════════════════════
# 5. 시각화 설정
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#1a1a2e')
gs  = gridspec.GridSpec(4, 2, figure=fig, wspace=0.38, hspace=0.60,
                        left=0.04, right=0.97, top=0.93, bottom=0.06)

_dark = '#16213e'
_gray = 'gray'

def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(_dark)
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors=_gray)
    ax.grid(True, alpha=0.25, color=_gray)
    for sp in ax.spines.values():
        sp.set_edgecolor(_gray)

# ── 3D 다리 (왼쪽 전체, 4행 모두)
ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax3d.set_facecolor(_dark)
reach = 0.55
ax3d.set_xlim(-reach, reach); ax3d.set_ylim(-reach, reach); ax3d.set_zlim(-reach, reach)
ax3d.set_xlabel('X (m)', color='white', labelpad=5)
ax3d.set_ylabel('Y (m)', color='white', labelpad=5)
ax3d.set_zlabel('Z (m)', color='white', labelpad=5)
ax3d.tick_params(colors=_gray)
ax3d.set_title(f'{mode_label}  [Imp: {"ON" if USE_IMP else "OFF"}]', color='white', fontsize=10)
# ax3d.view_init(elev=90, azim=180)    # Z축 방향에서 바라봄, X수평 Y수직
ax3d.view_init(elev=0, azim=90)    # Z축 방향에서 바라봄, X수평 Y수직
ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False

# ── 관절각 그래프 (우상) — 주석처리
# ax_ang = fig.add_subplot(gs[0, 1])
# _style_ax(ax_ang, 'pos_cmd(th1~th4)[deg]', '프레임', '[deg]')
# ang_min = np.degrees(q_a[:, :4].min()) - 10
# ang_max = np.degrees(q_a[:, :4].max()) + 10
# ax_ang.set_xlim(0, n_frames); ax_ang.set_ylim(ang_min, ang_max)

# ── 발끝 높이 그래프 (우상)
ax_fh = fig.add_subplot(gs[0, 1])
_fh_vals = x_a[:, 0] * 1e3   # mm
_style_ax(ax_fh, '발끝 높이 foot X [mm]', '프레임', '[mm]')
ax_fh.set_xlim(0, n_frames); ax_fh.set_ylim(_fh_vals.min() - 5, _fh_vals.max() + 5)

# ── 관절별 τ_cmd 그래프 (우중)
ax_tau = fig.add_subplot(gs[1, 1])
tau_range = np.abs(tau_cmd).max() * 1.3 + 0.01
_style_ax(ax_tau, 'tau_cmd(th1~th5)[N·m]', '프레임', '[N·m]')
ax_tau.set_xlim(0, n_frames); ax_tau.set_ylim(-tau_range * 0.6, tau_range)
ax_tau.axhline(0, color='white', lw=0.6, ls='--', alpha=0.5)

# ── τ_ff vs τ_pd 분해 그래프 (우하)
ax_fftau = fig.add_subplot(gs[2, 1])
fftau_range = max(np.abs(tau_offset).max(), np.abs(tau_pd).max()) * 1.3 + 0.01
_style_ax(ax_fftau, 'tau_offset vs tau_pd(th2~th4)[N·m]', '프레임', '[N·m]')
ax_fftau.set_xlim(0, n_frames); ax_fftau.set_ylim(-fftau_range * 0.6, fftau_range)
ax_fftau.axhline(0, color='white', lw=0.6, ls='--', alpha=0.5)

# ── GRF 그래프 (최하단)
ax_grf = fig.add_subplot(gs[3, 1])
lam_max = np.abs(lam_des).max() * 1.2 + 1.0
_style_ax(ax_grf, 'GRF lam_des vs lam_calc[N]', '프레임', '[N]')
ax_grf.set_xlim(0, n_frames); ax_grf.set_ylim(-lam_max * 3, lam_max * 3)
ax_grf.axhline(0, color='white', lw=0.6, ls='--', alpha=0.5)

# ── 링크 컬러
_LINK_COL = ['#00d4ff', '#00ff99', '#ff6b35', '#ffcc00', '#cc88ff']
_ANG_COL  = ['#00d4ff', '#00ff99', '#ff6b35', '#ffcc00']
_TAU_COL  = ['#00d4ff', '#ff6b35', '#ffcc00', '#cc88ff', '#ff88aa']
_AX_COL   = ['#ff4444', '#44ff44', '#4444ff']
AXIS_LEN  = 0.06

# 3D 링크
link_lines = [ax3d.plot([], [], [], '-o', color=_LINK_COL[i], lw=3, markersize=8)[0]
              for i in range(5)]
trace_line, = ax3d.plot([], [], [], '-', color='#ff88aa', lw=1.5, alpha=0.8)

# GRF 화살표 (발끝에서 +X 방향)
_grf_quiver = [None]

info_text  = ax3d.text2D(0.02, 0.97, "", transform=ax3d.transAxes,
                          color='white', fontfamily='monospace', fontsize=7.5, va='top')
phase_text = ax3d.text2D(0.02, 1, "", transform=ax3d.transAxes,
                          color='yellow', fontsize=9, fontweight='bold', va='top')

# 관절각 선 (θ1~θ4) — 주석처리
_fr = np.arange(n_frames)
# ang_lines = [ax_ang.plot(_fr, np.degrees(theta_t[:, k]), lw=1.8, color=_ANG_COL[k], label=f'th{k+1}')[0]
#              for k in range(4)]
# ax_ang.legend(loc='upper right', fontsize=7.5, facecolor='#1a1a2e',
#               labelcolor='white', edgecolor=_gray)

# 발끝 높이 선 (foot X 위치)
line_fh, = ax_fh.plot(_fr, x_a[:, 0] * 1e3, lw=2.0, color='#00ff99', label='foot X')
ax_fh.axhline(x_a[0, 0] * 1e3, color='white', lw=0.8, ls='--', alpha=0.5)
ax_fh.legend(loc='upper right', fontsize=7.5, facecolor='#1a1a2e',
             labelcolor='white', edgecolor=_gray)

# τ_cmd 선 (θ1~θ5 전체)
_TAU5_COL = ['#00d4ff', '#00ff99', '#ff6b35', '#ffcc00', '#cc88ff']
lines_cmd = [ax_tau.plot(_fr, tau_cmd[:, k], lw=2.0, color=_TAU5_COL[k], label=f'tau_cmd{k+1}')[0]
             for k in range(5)]
ax_tau.legend(fontsize=7.5, facecolor='#1a1a2e',
              labelcolor='white', edgecolor=_gray)

# τ_pd / τ_imp / τ_ff 분해 (θ2~θ4) — 색상은 _TAU5_COL[j]로 tau_cmd 그래프와 통일
_tau_joints = [1, 2, 3]   # θ2, θ3, θ4
lines_pd  = [ax_fftau.plot(_fr, tau_pd[:,  j], lw=1.8, color=_TAU5_COL[j],
                            label=f'tau_pd{j+1}')[0]  for j in _tau_joints]
lines_imp = [ax_fftau.plot(_fr, tau_imp[:, j], lw=1.8, color=_TAU5_COL[j], ls='--',
                            label=f'tau_imp{j+1}')[0] for j in _tau_joints]
lines_ff  = [ax_fftau.plot(_fr, tau_ff[:,  j], lw=1.8, color=_TAU5_COL[j], ls=':',
                            label=f'tau_ff{j+1}')[0]  for j in _tau_joints]
ax_fftau.legend(fontsize=7.0, ncol=3, facecolor='#1a1a2e',
                labelcolor='white', edgecolor=_gray,
                handles=[*lines_pd, *lines_imp, *lines_ff])

# GRF 선
line_ld, = ax_grf.plot(_fr, lam_des[:, 0], lw=2.2, color='#00d4ff', label='lam_des Fx')
line_lc, = ax_grf.plot(_fr, lam_calc[:, 0], lw=1.8, ls='--', color='magenta', label='lam_calc Fx')
ax_grf.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', edgecolor=_gray)

# 수직 커서선 (현재 프레임 위치 표시)
_cursors = [ax.axvline(x=0, color='white', lw=1.2, ls='-', alpha=0.75)
            for ax in [ax_fh, ax_tau, ax_fftau, ax_grf]]

# 위상 구분선
if phase_idx:
    for p_i, p_c in zip(phase_idx, ['orangered', 'royalblue']):
        for ax in [ax_fh, ax_tau, ax_fftau, ax_grf]:
            ax.axvline(x=p_i + 1, color=p_c, lw=1.2, ls=':', alpha=0.7)

# 좌표축 quiver (월드 + 발끝 프레임)
_frame_quivers = [
    ax3d.quiver(0, 0, 0, 1, 0, 0, length=AXIS_LEN, color=c, linewidth=1.5)
    for _ in range(2) for c in _AX_COL
]

def _draw_frame(T, quivers_xyz):
    orig = T[:3, 3]
    for j, q in enumerate(quivers_xyz):
        axis = T[:3, j]
        q.remove()
        quivers_xyz[j] = ax3d.quiver(
            orig[0], orig[1], orig[2],
            axis[0], axis[1], axis[2],
            length=AXIS_LEN, color=_AX_COL[j],
            linewidth=1.5, arrow_length_ratio=0.3
        )
    return quivers_xyz

GRF_SCALE = 0.001   # [m/N]  (시각화 스케일)


# ══════════════════════════════════════════════════════════════
# 6. 애니메이션
# ══════════════════════════════════════════════════════════════
trace_x, trace_y, trace_z = [], [], []

def init_anim():
    for ln in link_lines:
        ln.set_data([], []); ln.set_3d_properties([])
    trace_line.set_data([], []); trace_line.set_3d_properties([])
    for cur in _cursors:
        cur.set_xdata([0, 0])
    info_text.set_text(''); phase_text.set_text('')
    return []

def animate(i):
    global _frame_quivers, _grf_quiver

    thetas = q_a[i]
    pts    = forward_kinematics(thetas)
    Pe     = pts[4]

    # 3D 링크
    for k in range(5):
        A, B = pts[k], pts[k+1]
        link_lines[k].set_data([A[0], B[0]], [A[1], B[1]])
        link_lines[k].set_3d_properties([A[2], B[2]])

    # 발끝 궤적
    trace_x.append(Pe[0]); trace_y.append(Pe[1]); trace_z.append(Pe[2])
    trace_line.set_data(trace_x, trace_y)
    trace_line.set_3d_properties(trace_z)

    # GRF 화살표 (Fx 성분)
    if _grf_quiver[0] is not None:
        _grf_quiver[0].remove()
    fx = lam_des[i, 0]
    if fx > 1.0:
        _grf_quiver[0] = ax3d.quiver(
            Pe[0], Pe[1], Pe[2],
            fx * GRF_SCALE, 0, 0,
            color='magenta', linewidth=2.5, arrow_length_ratio=0.2
        )
    else:
        _grf_quiver[0] = None

    # 좌표축
    T = np.eye(4)
    for k, (alpha, a, d) in enumerate(DH_PARAMS):
        T = T @ get_dh_matrix(alpha, a, d, thetas[k])
    _frame_quivers[:3] = _draw_frame(np.eye(4), _frame_quivers[:3])
    _frame_quivers[3:] = _draw_frame(T, _frame_quivers[3:])

    # 관절각 그래프
    deg = np.degrees(thetas)

    # 수직 커서선 이동
    for cur in _cursors:
        cur.set_xdata([i, i])

    # 위상 텍스트
    if phase_idx:
        n1_p, n2_p = phase_idx
        if i <= n1_p:      p_str = "Phase 1: 준비 (Crouch)"
        elif i <= n2_p:    p_str = "Phase 2: 도약 (Push-off)"
        else:              p_str = "Phase 3: 착지 복귀"
    else:
        p_str = "Stance" if i < n_frames // 2 else "Swing"
    phase_text.set_text(p_str)

    # 정보 텍스트
    tff  = tau_ff[i];  tcmd = tau_cmd[i];  tpd = tau_pd[i];  tgrav = tau_grav[i]
    msg = (f"joint pos\n"
           f"th1:{deg[0]:+6.1f}°  th2:{deg[1]:+6.1f}°\n"
           f"th3:{deg[2]:+6.1f}°  th4:{deg[3]:+6.1f}°\n\n"

           f"foot pos(mm)\n"
           f"X:{Pe[0]*1e3:+7.1f}\n"
           f"Y:{Pe[1]*1e3:+7.1f}\n"
           f"Z:{Pe[2]*1e3:+7.1f}\n\n"

           f"tau_cmd [N·m]\n"
           f"tau1={tcmd[0]:+5.2f}  tau2={tcmd[1]:+5.2f}  tau3={tcmd[2]:+5.2f}  tau4={tcmd[3]:+5.2f}\n"
           f"tau_pd1={tpd[0]:+5.2f} tau_pd2={tpd[1]:+5.2f} tau_pd3={tpd[2]:+5.2f} tau_pd4={tpd[3]:+5.2f}\n\n"

           f"tau_ff1={tff[0]:+5.2f} tau_ff2={tff[1]:+5.2f} tau_ff3={tff[2]:+5.2f} tau_ff4={tff[3]:+5.2f}\n"
           f"tau_grav1={tgrav[0]:+5.2f} tau_grav2={tgrav[1]:+5.2f} tau_grav3={tgrav[2]:+5.2f} tau_grav4={tgrav[3]:+5.2f}\n"
           f"lam_des (Fx)= {lam_des[i,0]:+.2f}N lam_calc(Fx)= {lam_calc[i,0]:+.2f}N\n\n"

           f"link mass {LINK_MASS.sum():.2f}[kg]\n"
           f"L1:{LINK_MASS[0]:.2f} L2:{LINK_MASS[1]:.2f} L3:{LINK_MASS[2]:.2f}\n"
           f"L4:{LINK_MASS[3]:.2f} L5:{LINK_MASS[4]:.2f}\n")
    info_text.set_text(msg)
    return []


ani = FuncAnimation(
    fig, animate, frames=n_frames,
    init_func=init_anim,
    interval=DT * 1000,
    blit=False, repeat=True
)

# fig.suptitle(
#     f'leg_statics_v3  |  3D 해석적 IK + 동역학  |  '
#     f'τ_cmd = Kp·Δθ + Kd·Δθ̇ + Jᵀ·λ_des   '
#     f'λ_calc = (JJᵀ+μI)⁻¹·J·τ_cmd',
#     color='white', fontsize=10
# )
plt.show()
