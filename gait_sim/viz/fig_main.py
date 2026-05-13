"""gait_sim.viz.fig_main — Main visualization figures.

v13.2 Phase 5-b/h: fig5 (body state) + fig1 static portion 추출.
모든 figure 함수는 SimState 객체를 받아 plot.

함수:
  · plot_body_state(R, meta)    — Figure 5: body pos/vel x/y/z + orient + omega vs cmd
  · plot_main_static(R, meta)   — Figure 1 static portion:
                                    gait phase chart + foot Z/X/dZ/dX/d²Z/d²X (7 panels)
                                    3D animation 은 별도 plot_anim() 으로 추후 분리.
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
