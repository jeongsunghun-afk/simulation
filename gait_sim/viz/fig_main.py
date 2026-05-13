"""gait_sim.viz.fig_main — Main visualization figures.

v13.2 Phase 5-b: fig5 (body state vs cmd) 부터 pilot 추출.
모든 figure 함수는 SimState 객체를 받아 plot.

함수:
  · plot_body_state(R: SimState, meta: dict) → plt.Figure
        body pos/vel x/y/z + orientation + angular vel (vs cmd)
        (v13.py Figure 5 와 동일 layout)

후속:
  · plot_anim(R, meta)       — Figure 1 (3D 애니메이션)
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
