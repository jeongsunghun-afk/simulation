"""gait_sim.viz.fig_main — Main visualization figures.

v13.2 Phase 5-b/h/7: fig5 (body state) + fig1 static portion + plot_anim 추출.
모든 figure 함수는 SimState 객체를 받아 plot.

함수:
  · plot_body_state(R, meta)    — Figure 5: body pos/vel x/y/z + orient + omega vs cmd
  · plot_main_static(R, meta)   — Figure 1 static portion:
                                    gait phase chart + foot Z/X/dZ/dX/d²Z/d²X (7 panels)
  · plot_anim(R, meta)          — Figure 1 3D animation (FuncAnimation):
                                    leg links + COM markers + foot traces + info text
                                    Returns (fig, anim) — anim 반드시 retain 필요 (GC 방지)
"""
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gait_sim.sim_state import SimState


# ══════════════════════════════════════════════════════════════
# Dark theme helper
# ══════════════════════════════════════════════════════════════
def _style_ax(ax, title, xlabel='Frame', ylabel=''):
    """v13 dark theme axes 적용."""
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors='gray')
    ax.grid(True, alpha=0.25, color='gray')
    for sp in ax.spines.values():
        sp.set_edgecolor('gray')


# ══════════════════════════════════════════════════════════════
# Figure 5 — Body state (pos/vel/orientation/omega) vs cmd
# ══════════════════════════════════════════════════════════════
def plot_body_state(R: SimState, meta: Optional[dict] = None) -> plt.Figure:
    """body 6-DoF state vs MPC reference (v13.py Figure 5 와 동일).

    Args:
        R:    SimState — body_pos_hist / body_R_hist / body_v_hist / body_omega_hist
                          + body_pos_ref_hist / body_v_ref_hist 사용
        meta: dict (optional) — title suffix 용
              {gait_type, V, T, D, use_nmpc}

    Returns: plt.Figure
    """
    meta = meta or {}
    N = R.n_frames
    fr = np.arange(N)

    fig = plt.figure(figsize=(12, 13))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.30, hspace=0.60,
                           left=0.07, right=0.97, top=0.93, bottom=0.05)

    # R-matrix → roll/pitch/yaw 추출
    bR = R.body_R_hist
    roll_deg  = np.degrees(np.arctan2(bR[:, 2, 1], bR[:, 2, 2]))
    pitch_deg = np.degrees(np.arcsin(np.clip(-bR[:, 2, 0], -1, 1)))
    yaw_deg   = np.degrees(np.arctan2(bR[:, 1, 0], bR[:, 0, 0]))

    axis_names = ['x', 'y', 'z']
    for ri in range(3):
        # pos
        err_p = R.body_pos_hist[:, ri] - R.body_pos_ref_hist[:, ri]
        rms_p = float(np.sqrt(np.mean(err_p**2)))
        max_p = float(np.max(np.abs(err_p)))
        ax_p = fig.add_subplot(gs[ri, 0])
        _style_ax(ax_p, f'body pos {axis_names[ri]} [m]   '
                        f'(err: rms={rms_p*1e3:.1f}mm, max={max_p*1e3:.1f}mm)',
                  ylabel='[m]')
        ax_p.set_xlim(0, N)
        ax_p.plot(fr, R.body_pos_ref_hist[:, ri], lw=1.4, color='#ff6b6b',
                  ls='--', label='cmd')
        ax_p.plot(fr, R.body_pos_hist[:, ri],     lw=1.6, color='#00d4ff',
                  label='actual')
        ax_p.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                    edgecolor='gray')

        # vel
        err_v = R.body_v_hist[:, ri] - R.body_v_ref_hist[:, ri]
        rms_v = float(np.sqrt(np.mean(err_v**2)))
        max_v = float(np.max(np.abs(err_v)))
        ax_v = fig.add_subplot(gs[ri, 1])
        _style_ax(ax_v, f'body vel {axis_names[ri]} [m/s]   '
                        f'(err: rms={rms_v:.3f}, max={max_v:.3f})',
                  ylabel='[m/s]')
        ax_v.set_xlim(0, N)
        ax_v.plot(fr, R.body_v_ref_hist[:, ri], lw=1.4, color='#ff6b6b',
                  ls='--', label='cmd')
        ax_v.plot(fr, R.body_v_hist[:, ri],     lw=1.6, color='#00d4ff',
                  label='actual')
        ax_v.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                    edgecolor='gray')

    # orientation
    ax_or = fig.add_subplot(gs[3, 0])
    _style_ax(ax_or,
        f'body orientation [deg]  cmd=0  '
        f'(|err|max: roll={np.max(np.abs(roll_deg)):.2f}°, '
        f'pitch={np.max(np.abs(pitch_deg)):.2f}°, '
        f'yaw={np.max(np.abs(yaw_deg)):.2f}°)',
        ylabel='[deg]')
    ax_or.set_xlim(0, N)
    ax_or.plot(fr, roll_deg,  lw=1.4, color='#ff6b6b', label='roll')
    ax_or.plot(fr, pitch_deg, lw=1.4, color='#ffd166', label='pitch')
    ax_or.plot(fr, yaw_deg,   lw=1.4, color='#06d6a0', label='yaw')
    ax_or.axhline(0, color='white', lw=0.6, ls='--', alpha=0.4)
    ax_or.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                 edgecolor='gray', ncol=3)

    # angular vel
    ax_om = fig.add_subplot(gs[3, 1])
    om_max = float(np.max(np.abs(R.body_omega_hist)))
    _style_ax(ax_om,
        f'body angular vel [rad/s]  cmd=0  (|err|max={om_max:.2f})',
        ylabel='[rad/s]')
    ax_om.set_xlim(0, N)
    ax_om.plot(fr, R.body_omega_hist[:, 0], lw=1.4, color='#ff6b6b', label='omega_x')
    ax_om.plot(fr, R.body_omega_hist[:, 1], lw=1.4, color='#ffd166', label='omega_y')
    ax_om.plot(fr, R.body_omega_hist[:, 2], lw=1.4, color='#06d6a0', label='omega_z')
    ax_om.axhline(0, color='white', lw=0.6, ls='--', alpha=0.4)
    ax_om.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                 edgecolor='gray', ncol=3)

    # title
    gait = meta.get('gait_type', '')
    V    = meta.get('V', '?')
    T    = meta.get('T', '?')
    D    = meta.get('D', '?')
    mode = 'NMPC (FDDP)' if meta.get('use_nmpc') else 'MPC + WBIC'
    fig.suptitle(
        f'Body State vs Cmd  |  {gait.upper()}  |  '
        f'v={V}m/s  T={T}s  D={D}  {mode}',
        color='white', fontsize=10)

    return fig


# ══════════════════════════════════════════════════════════════
# Figure 1 — Main static portion (gait phase + foot kinematic profiles)
# 3D animation 은 별도 plot_anim() 으로 분리 예정.
# ══════════════════════════════════════════════════════════════
LEG_NAMES   = ['FR', 'FL', 'HR', 'HL']
LEG_COLORS  = ['#00d4ff', '#ff6b35', '#00ff99', '#c264ff']


def plot_main_static(R: SimState, meta: Optional[dict] = None) -> plt.Figure:
    """v13.py Figure 1 의 static 부분 (3D anim 제외) — 7 panels:
        · Gait Phase chart (FR/FL/HR/HL, bright=swing)
        · Step Z [m]          + Step X [m]
        · Step dZ/dt [m/s]    + Step dX/dt [m/s]
        · Step d²Z/dt² [m/s²] + Step d²X/dt² [m/s²]

    Args:
        R:    SimState — foot_local + foot_vel_t + foot_acc_t + phase_hist 사용
        meta: dict — {gait_type, V, T, D, step_height, step_length, total_mass}

    Returns: plt.Figure
    """
    meta = meta or {}
    N  = R.n_frames
    fr = np.arange(N)
    D_swing = meta.get('D', 0.5)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.30, hspace=0.65,
                           left=0.07, right=0.98, top=0.93, bottom=0.05)

    # row 0: Gait Phase chart (full width)
    ax_phase = fig.add_subplot(gs[0, :])
    gait = meta.get('gait_type', '')
    _style_ax(ax_phase, f'Gait Phase  [{gait}]  (Bright=Swing)', ylabel='Leg')
    ax_phase.set_xlim(0, N)
    ax_phase.set_ylim(-0.5, 3.5)
    ax_phase.set_yticks([0, 1, 2, 3])
    ax_phase.set_yticklabels(LEG_NAMES[::-1], color='white')
    swing_flag = (R.phase_hist < D_swing)
    for leg in range(4):
        row = 3 - leg
        in_sw = False
        sw_start = 0
        for fi in range(N):
            if swing_flag[fi, leg] and not in_sw:
                sw_start = fi
                in_sw = True
            elif not swing_flag[fi, leg] and in_sw:
                ax_phase.barh(row, fi - sw_start, left=sw_start, height=0.7,
                              color=LEG_COLORS[leg], alpha=0.85)
                in_sw = False
        if in_sw:
            ax_phase.barh(row, N - sw_start, left=sw_start, height=0.7,
                          color=LEG_COLORS[leg], alpha=0.85)

    # row 1: Step Height Z + Step Length X
    ax_z = fig.add_subplot(gs[1, 0])
    _style_ax(ax_z, 'Step Height  Z [m]', ylabel='Z [m]')
    ax_z.set_xlim(0, N)
    for leg in range(4):
        ax_z.plot(fr, R.foot_local[:, leg, 2], lw=1.6,
                  color=LEG_COLORS[leg], label=LEG_NAMES[leg])
    ax_z.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                edgecolor='gray', ncol=4)

    ax_x = fig.add_subplot(gs[1, 1])
    _style_ax(ax_x, 'Step Length  X [m]', ylabel='X [m]')
    ax_x.set_xlim(0, N)
    for leg in range(4):
        ax_x.plot(fr, R.foot_local[:, leg, 0], lw=1.6,
                  color=LEG_COLORS[leg], label=LEG_NAMES[leg])
    ax_x.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                edgecolor='gray', ncol=4)

    # row 2: Z velocity + X velocity
    ax_zv = fig.add_subplot(gs[2, 0])
    _style_ax(ax_zv, 'Step Height Velocity  dZ/dt [m/s]', ylabel='[m/s]')
    ax_zv.set_xlim(0, N)
    for leg in range(4):
        ax_zv.plot(fr, R.foot_vel_t[:, leg, 2], lw=1.6,
                   color=LEG_COLORS[leg],
                   ls='--' if leg >= 2 else '-', label=LEG_NAMES[leg])
    ax_zv.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                 edgecolor='gray', ncol=4)

    ax_xv = fig.add_subplot(gs[2, 1])
    _style_ax(ax_xv, 'Step Length Velocity  dX/dt [m/s]', ylabel='[m/s]')
    ax_xv.set_xlim(0, N)
    for leg in range(4):
        ax_xv.plot(fr, R.foot_vel_t[:, leg, 0], lw=1.6,
                   color=LEG_COLORS[leg],
                   ls='--' if leg >= 2 else '-', label=LEG_NAMES[leg])
    ax_xv.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                 edgecolor='gray', ncol=4)

    # row 3: Z acceleration + X acceleration
    ax_za = fig.add_subplot(gs[3, 0])
    _style_ax(ax_za, 'Step Height Acceleration  d²Z/dt² [m/s²]', ylabel='[m/s²]')
    ax_za.set_xlim(0, N)
    for leg in range(4):
        ax_za.plot(fr, R.foot_acc_t[:, leg, 2], lw=1.6,
                   color=LEG_COLORS[leg],
                   ls='--' if leg >= 2 else '-', label=LEG_NAMES[leg])
    ax_za.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                 edgecolor='gray', ncol=4)

    ax_xa = fig.add_subplot(gs[3, 1])
    _style_ax(ax_xa, 'Step Length Acceleration  d²X/dt² [m/s²]', ylabel='[m/s²]')
    ax_xa.set_xlim(0, N)
    for leg in range(4):
        ax_xa.plot(fr, R.foot_acc_t[:, leg, 0], lw=1.6,
                   color=LEG_COLORS[leg],
                   ls='--' if leg >= 2 else '-', label=LEG_NAMES[leg])
    ax_xa.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                 edgecolor='gray', ncol=4)

    V        = meta.get('V', '?')
    T        = meta.get('T', '?')
    step_h   = meta.get('step_height', 0.0) * 1e3
    step_l   = meta.get('step_length', 0.0) * 1e3
    total_m  = meta.get('total_mass', 0.0)
    fig.suptitle(
        f'Gait Main Static  |  {gait.upper()}  |  '
        f'v={V}m/s  T={T}s  D={D_swing}  '
        f'step_h={step_h:.0f}mm  step_l={step_l:.0f}mm  total_mass={total_m:.2f}kg',
        color='white', fontsize=11)

    return fig


# ══════════════════════════════════════════════════════════════
# Figure 1 — 3D Animation (FuncAnimation)
# ══════════════════════════════════════════════════════════════
import math
from matplotlib.animation import FuncAnimation
# mpl_toolkits.mplot3d 는 import 만 해도 projection='3d' 등록됨
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from gait_sim.kinematics import (
    forward_kinematics as _fwk, _dh_to_sim as _d2s, _dh_matrix as _dhM,
)
from gait_sim.model import (
    LEG_DH as _LEG_DH, N_JOINTS_PER_LEG as _NJ, LINK_MASS_PER_LEG as _LM,
    LEG_HIP_OFFSETS,
)

_AX_COLORS = ['#ff4444', '#44ff44', '#4444ff']


def plot_anim(R: SimState, meta: dict = None,
              viz_body_mode: str = 'world',
              interval_ms: int = None,
              trace_cycle: bool = True) -> tuple:
    """3D animation of quadruped over time (v13.py Figure 1 anim portion).

    Args:
        R:    SimState — joint_hist, foot_hist, body_pos/R_hist, wbc_tau_cmd, wbc_lam_des
        meta: dict — {gait_type, V, T, D, step_height, step_length, total_mass}
        viz_body_mode: 'world' / 'body_follow' / 'static'
            - 'world': camera 고정, robot 이동 + 회전 (default)
            - 'body_follow': camera body 따라감, R 회전만
            - 'static': body_pos/R 무시 (robot 원점 고정)
        interval_ms: FuncAnimation interval (None 이면 R.dt*1000)
        trace_cycle: 발 trace 길이 = T/DT (1 cycle) 이면 True

    Returns:
        (fig, anim) — anim 객체는 caller 가 retain 해야 GC 안 됨.
    """
    meta = meta or {}
    N = R.n_frames
    DT_R = R.dt
    T_meta = meta.get('T', 0.5)
    D_meta = meta.get('D', 0.5)
    swing_flag = (R.phase_hist < D_meta)

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.set_facecolor('#16213e')
    # 3D ax 가 기본적으로 figure 가운데 좁게 자리 잡는 문제 — 명시적으로 폭 확장
    ax3d.set_position([0.02, 0.04, 0.96, 0.88])

    # Viewport
    reach = 0.65
    ax3d.set_xlim(-reach, reach)
    ax3d.set_ylim(-0.5, 0.5)
    ax3d.set_zlim(-0.65, 0.15)
    ax3d.set_xlabel('X (m)', color='white', labelpad=4)
    ax3d.set_ylabel('Y (m)', color='white', labelpad=4)
    ax3d.set_zlabel('Z (m)', color='white', labelpad=4)
    ax3d.tick_params(colors='gray')
    gait = meta.get('gait_type', '')
    V    = meta.get('V', '?')
    Tt   = meta.get('T', '?')
    step_h = meta.get('step_height', 0.0) * 1e3
    step_l = meta.get('step_length', 0.0) * 1e3
    total_m = meta.get('total_mass', 0.0)
    ax3d.set_title(
        f'Gait Anim  [{gait.upper()}]  v={V}m/s  T={Tt}s  D={D_meta}  '
        f'step_h={step_h:.0f}mm  step_l={step_l:.0f}mm  total_mass={total_m:.2f}kg',
        color='white', fontsize=9)
    ax3d.view_init(elev=20, azim=-55)
    try:
        ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False
    except AttributeError:
        pass

    # World 모드에서 body x 이동 범위 반영
    if viz_body_mode == 'world':
        x_max = float(np.max(R.body_pos_hist[:, 0]))
        x_min = float(np.min(R.body_pos_hist[:, 0]))
        ax3d.set_xlim(min(-reach, x_min - 0.3), max(reach, x_max + 0.3))

    # 3D box aspect: 데이터 range 비례로 자동 설정 — 기본 cubical 이 z 를 stretch 시키는 문제 해결
    # robot 비율이 실제 m 단위 그대로 보임 (v13 fig1 와 유사한 가로형 view)
    _xr = ax3d.get_xlim()[1] - ax3d.get_xlim()[0]
    _yr = ax3d.get_ylim()[1] - ax3d.get_ylim()[0]
    _zr = ax3d.get_zlim()[1] - ax3d.get_zlim()[0]
    ax3d.set_box_aspect([_xr, _yr, _zr])

    # Body chassis 박스 라인 (4 hip 잇는 사각형)
    BC_BODY = np.array([
        LEG_HIP_OFFSETS[0], LEG_HIP_OFFSETS[2],
        LEG_HIP_OFFSETS[3], LEG_HIP_OFFSETS[1],
        LEG_HIP_OFFSETS[0],
    ])
    body_chassis_line, = ax3d.plot([], [], [], '-', color='white', lw=2.5, alpha=0.7)
    hip_markers = [ax3d.plot([], [], [], 'o', color=LEG_COLORS[leg],
                              markersize=7, alpha=0.8)[0] for leg in range(4)]
    body_com_marker, = ax3d.plot([], [], [], 'X', color='yellow',
                                  markersize=10, alpha=0.9, markeredgecolor='black')
    for leg in range(4):
        h = LEG_HIP_OFFSETS[leg]
        ax3d.text(h[0], h[1], h[2] + 0.02, LEG_NAMES[leg], color=LEG_COLORS[leg], fontsize=7)

    # Ground plane (foot z @ home)
    gnd_z = float(R.foot_hist[0, 0, 2])
    xx, yy = np.meshgrid([ax3d.get_xlim()[0], ax3d.get_xlim()[1]], [-0.5, 0.5])
    try:
        ax3d.plot_surface(xx, yy, np.full_like(xx, gnd_z), alpha=0.12, color='#888888')
    except Exception:
        pass

    # Leg links / swing dots / leg traces / link COM markers
    leg_links = []
    for leg in range(4):
        nj = _NJ[leg]
        lns = [ax3d.plot([], [], [], '-o', color=LEG_COLORS[leg],
                          lw=2.5, markersize=5)[0] for _ in range(nj)]
        leg_links.append(lns)
    trace_len = int(T_meta / DT_R) if trace_cycle else N
    leg_traces = [ax3d.plot([], [], [], '-', color=LEG_COLORS[leg],
                             lw=1.2, alpha=0.6)[0] for leg in range(4)]
    trace_buf = [[[], [], []] for _ in range(4)]
    swing_dots = [ax3d.plot([], [], [], 'o', color=LEG_COLORS[leg],
                             markersize=9, alpha=0.9)[0] for leg in range(4)]
    link_com_markers = []
    for leg in range(4):
        nj = _NJ[leg]
        lm = _LM[leg]
        mks = [ax3d.plot([], [], [], '*', color='#ffd700',
                          markersize=4.0 + 3.5 * math.sqrt(float(lm[k])),
                          markeredgecolor='black', markeredgewidth=0.5,
                          alpha=0.9, zorder=15)[0] for k in range(nj)]
        link_com_markers.append(mks)

    # Frame indicator quivers (joint frame, recreated each call)
    FRAME_LEN = 0.035
    jf_quivers = [[[None, None, None] for _ in range(_NJ[leg] + 1)]
                   for leg in range(4)]

    info_text = ax3d.text2D(0.02, 0.98, "", transform=ax3d.transAxes,
                             color='white', fontfamily='monospace', fontsize=7.5, va='top')

    # Base frame quivers (origin, static)
    BASE_FRAME_LEN = 0.12
    for ax_i, lbl in enumerate(['X (fwd)', 'Y (lat)', 'Z (up)']):
        dv = np.zeros(3); dv[ax_i] = BASE_FRAME_LEN
        ax3d.quiver(0, 0, 0, dv[0], dv[1], dv[2],
                     color=_AX_COLORS[ax_i], linewidth=2.5, arrow_length_ratio=0.25)
        ax3d.text(dv[0]*1.15, dv[1]*1.15, dv[2]*1.15,
                   lbl, color=_AX_COLORS[ax_i], fontsize=8, fontweight='bold')

    # body transform helper (closure)
    def _body_T_at(fi):
        if viz_body_mode == 'static':
            return np.zeros(3), np.eye(3)
        if viz_body_mode == 'body_follow':
            return np.zeros(3), R.body_R_hist[fi]
        return R.body_pos_hist[fi], R.body_R_hist[fi]

    def init():
        for leg in range(4):
            for ln in leg_links[leg]:
                ln.set_data([], []); ln.set_3d_properties([])
            leg_traces[leg].set_data([], []); leg_traces[leg].set_3d_properties([])
            swing_dots[leg].set_data([], []); swing_dots[leg].set_3d_properties([])
            for mk in link_com_markers[leg]:
                mk.set_data([], []); mk.set_3d_properties([])
            trace_buf[leg][0].clear(); trace_buf[leg][1].clear(); trace_buf[leg][2].clear()
        info_text.set_text('')
        return []

    def animate(fi):
        t = fi * DT_R
        body_pos_v, body_R_v = _body_T_at(fi)

        bc_world = np.array([body_pos_v + body_R_v @ p for p in BC_BODY])
        body_chassis_line.set_data(bc_world[:, 0], bc_world[:, 1])
        body_chassis_line.set_3d_properties(bc_world[:, 2])
        body_com_marker.set_data([body_pos_v[0]], [body_pos_v[1]])
        body_com_marker.set_3d_properties([body_pos_v[2]])

        for leg in range(4):
            nj = _NJ[leg]
            q  = R.joint_hist[fi, leg, :nj]
            pts_dh = _fwk(q, dh=_LEG_DH[leg])
            pts    = [_d2s(p, front_leg=(leg < 2)) for p in pts_dh]
            hip_b  = LEG_HIP_OFFSETS[leg]
            hip_w  = body_pos_v + body_R_v @ hip_b
            hip_markers[leg].set_data([hip_w[0]], [hip_w[1]])
            hip_markers[leg].set_3d_properties([hip_w[2]])
            for k in range(nj):
                A_b = hip_b + pts[k];   B_b = hip_b + pts[k+1]
                A   = body_pos_v + body_R_v @ A_b
                B   = body_pos_v + body_R_v @ B_b
                leg_links[leg][k].set_data([A[0], B[0]], [A[1], B[1]])
                leg_links[leg][k].set_3d_properties([A[2], B[2]])
                mid = 0.5 * (A + B)
                link_com_markers[leg][k].set_data([mid[0]], [mid[1]])
                link_com_markers[leg][k].set_3d_properties([mid[2]])
            pe_b = R.foot_hist[fi, leg]
            pe   = body_pos_v + body_R_v @ pe_b
            if swing_flag[fi, leg]:
                swing_dots[leg].set_data([pe[0]], [pe[1]])
                swing_dots[leg].set_3d_properties([pe[2]])
            else:
                swing_dots[leg].set_data([], []); swing_dots[leg].set_3d_properties([])
            trace_buf[leg][0].append(pe[0])
            trace_buf[leg][1].append(pe[1])
            trace_buf[leg][2].append(pe[2])
            leg_traces[leg].set_data(trace_buf[leg][0][-trace_len:],
                                       trace_buf[leg][1][-trace_len:])
            leg_traces[leg].set_3d_properties(trace_buf[leg][2][-trace_len:])
            # Joint frame quivers (recreate)
            T_dh = np.eye(4)
            for j in range(nj + 1):
                orig_sim_b = _d2s(T_dh[:3, 3], front_leg=(leg < 2))
                pos_b      = hip_b + orig_sim_b
                pos        = body_pos_v + body_R_v @ pos_b
                for ax_i in range(3):
                    dv_b = _d2s(T_dh[:3, ax_i], front_leg=(leg < 2))
                    dv   = body_R_v @ dv_b
                    if jf_quivers[leg][j][ax_i] is not None:
                        jf_quivers[leg][j][ax_i].remove()
                    jf_quivers[leg][j][ax_i] = ax3d.quiver(
                        pos[0], pos[1], pos[2],
                        dv[0]*FRAME_LEN, dv[1]*FRAME_LEN, dv[2]*FRAME_LEN,
                        color=_AX_COLORS[ax_i], linewidth=1.0, arrow_length_ratio=0.3)
                if j < nj:
                    T_dh = T_dh @ _dhM(_LEG_DH[leg][j][0], _LEG_DH[leg][j][1],
                                         _LEG_DH[leg][j][2], float(q[j]))

        # Info text
        sw_str = "  ".join(
            f"{LEG_NAMES[l]}:{'SW' if swing_flag[fi, l] else 'ST'}" for l in range(4))
        deg = np.degrees(R.joint_hist[fi])
        jnt_lines = []; tau_lines = []; grf_des_lines = []; grf_calc_lines = []
        for leg in range(4):
            d  = deg[leg]
            tc = R.wbc_tau_cmd[fi, leg]            # 실제 motor τ_cmd (clip 후)
            ld = R.wbc_lam_des[fi, leg]            # MPC QP planned GRF
            lc = R.wbc_lam_calc[fi, leg]           # τ_cmd 가 만드는 등가 GRF (땅이 발에 가하는 추정 반력)
            jnt_lines.append(
                f"{LEG_NAMES[leg]} th1={d[0]:+5.1f}d th2={d[1]:+6.1f}d "
                f"th3={d[2]:+6.1f}d th4={d[3]:+5.1f}d th5={d[4]:+5.1f}d")
            tau_lines.append(
                f"{LEG_NAMES[leg]} tau_cmd=[{tc[0]:+5.1f} {tc[1]:+5.1f} "
                f"{tc[2]:+5.1f} {tc[3]:+5.1f} {tc[4]:+5.1f}]Nm")
            grf_des_lines.append(
                f"{LEG_NAMES[leg]} lam_des =[{ld[0]:+6.1f} {ld[1]:+6.1f} {ld[2]:+7.1f}]N")
            grf_calc_lines.append(
                f"{LEG_NAMES[leg]} lam_calc=[{lc[0]:+6.1f} {lc[1]:+6.1f} {lc[2]:+7.1f}]N")
        info_text.set_text(
            f"t={t:.3f}s\n{sw_str}\n\n"
            + "\n".join(jnt_lines)
            + "\n\n"
            + "\n".join(tau_lines)
            + "\n\n"
            + "\n".join(grf_des_lines)
            + "\n\n"
            + "\n".join(grf_calc_lines)
        )
        return []

    intv = int(interval_ms if interval_ms is not None else DT_R * 1000)
    anim = FuncAnimation(fig, animate, frames=N, init_func=init,
                          interval=intv, blit=False, repeat=True)
    return fig, anim
