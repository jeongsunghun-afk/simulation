"""gait_sim.viz.fig_legs — Joint / foot analysis figures (per-leg).

v13.2 Phase 5-c/e: Figure 2 (FR/HL joint analysis) + Figure 6 (foot trajectory) 추출.

함수:
  · plot_joint_analysis(R, meta)   — FR/HL pos/vel/acc/jerk + opt-IK 진단 (Figure 2)
  · plot_foot_trajectory(R, meta)  — 4-leg world-frame foot cmd vs actual (Figure 6)
"""
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gait_sim.sim_state import SimState

LEG_COLORS  = ['#00d4ff', '#ff6b35', '#00ff99', '#c264ff']    # FR, FL, HR, HL
AXIS_COLORS = ['#ff6b6b', '#ffd166', '#06d6a0', '#4cc9f0', '#f72585']  # 5 joints


# ══════════════════════════════════════════════════════════════
# Dark theme helper
# ══════════════════════════════════════════════════════════════
def _style_ax(ax, title, xlabel='Frame', ylabel=''):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=10)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors='gray')
    ax.grid(True, alpha=0.25, color='gray')
    for sp in ax.spines.values():
        sp.set_edgecolor('gray')


def _leg_subplot(fig, gs_pos, fr, title, data, ylabel, n_frames):
    """5-joint overlay subplot for one leg."""
    ax = fig.add_subplot(gs_pos)
    _style_ax(ax, title, ylabel=ylabel)
    ax.set_xlim(0, n_frames)
    nj = data.shape[1]
    for j in range(nj):
        ax.plot(fr, data[:, j], lw=1.6,
                color=AXIS_COLORS[j % len(AXIS_COLORS)], label=f'th{j+1}')
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
              edgecolor='gray', ncol=5)
    return ax


# ══════════════════════════════════════════════════════════════
# Figure 2 — FR/HL joint pos/vel/acc/jerk + opt-IK diagnostics
# ══════════════════════════════════════════════════════════════
def plot_joint_analysis(R: SimState, meta: Optional[dict] = None) -> plt.Figure:
    """FR / HL joint analysis + opt-IK 진단 (v13.py Figure 2).

    Args:
        R:    SimState — joint_hist + joint_vel/acc/jrk_hist + opt_ik_*_hist 사용
        meta: dict — {gait_type, V, T, D, step_height, step_length, opt_ik_maxiter}

    Returns: plt.Figure (5 rows × 2 cols)
    """
    meta = meta or {}
    N = R.n_frames
    fr = np.arange(N)
    maxiter = meta.get('opt_ik_maxiter', 100)

    fig = plt.figure(figsize=(12, 13))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(5, 2, figure=fig, wspace=0.35, hspace=0.55,
                           left=0.07, right=0.97, top=0.94, bottom=0.04)

    # 4 rows: pos / vel / acc / jerk × (FR, HL)
    _leg_subplot(fig, gs[0, 0], fr, 'FR Joint Pos [deg]',
                 np.degrees(R.joint_hist[:, 0, :5]), '[deg]', N)
    _leg_subplot(fig, gs[0, 1], fr, 'HL Joint Pos [deg]',
                 np.degrees(R.joint_hist[:, 3, :5]), '[deg]', N)
    _leg_subplot(fig, gs[1, 0], fr, 'FR Joint Angular Velocity [rad/s]',
                 R.joint_vel_hist[:, 0, :5], '[rad/s]', N)
    _leg_subplot(fig, gs[1, 1], fr, 'HL Joint Angular Velocity [rad/s]',
                 R.joint_vel_hist[:, 3, :5], '[rad/s]', N)
    _leg_subplot(fig, gs[2, 0], fr, 'FR Joint Angular Acceleration [rad/s²]',
                 R.joint_acc_hist[:, 0, :5], '[rad/s²]', N)
    _leg_subplot(fig, gs[2, 1], fr, 'HL Joint Angular Acceleration [rad/s²]',
                 R.joint_acc_hist[:, 3, :5], '[rad/s²]', N)
    _leg_subplot(fig, gs[3, 0], fr, 'FR Joint Jerk [rad/s³]',
                 R.joint_jrk_hist[:, 0, :5], '[rad/s³]', N)
    _leg_subplot(fig, gs[3, 1], fr, 'HL Joint Jerk [rad/s³]',
                 R.joint_jrk_hist[:, 3, :5], '[rad/s³]', N)

    # row 4: opt-IK iterations + position error (4 legs overlay)
    ax_nit = fig.add_subplot(gs[4, 0])
    _style_ax(ax_nit, 'Opt-IK Iterations  (★=fallback frame)', ylabel='nit')
    ax_nit.set_xlim(0, N)
    ax_nit.axhline(maxiter, color='red', lw=0.8, ls='--', alpha=0.6,
                   label=f'maxiter={maxiter}')
    for c, name, color in zip(range(4), ['FR','FL','HR','HL'], LEG_COLORS):
        ax_nit.plot(fr, R.opt_ik_nit_hist[:, c], lw=1.2, color=color,
                    alpha=0.85, label=name)
        fb_idx = np.where(R.opt_ik_fallback_hist[:, c])[0]
        if len(fb_idx) > 0:
            ax_nit.plot(fb_idx, R.opt_ik_nit_hist[fb_idx, c], '*',
                        color=color, markersize=8, markeredgecolor='red',
                        markeredgewidth=0.7, alpha=0.95)
    ax_nit.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                  edgecolor='gray', ncol=5)

    ax_perr = fig.add_subplot(gs[4, 1])
    _style_ax(ax_perr, 'Opt-IK Position Error  (NaN=fallback)', ylabel='[mm]')
    ax_perr.set_xlim(0, N)
    ax_perr.set_yscale('log')
    for c, name, color in zip(range(4), ['FR','FL','HR','HL'], LEG_COLORS):
        perr_mm = np.sqrt(R.opt_ik_pos_err_hist[:, c]) * 1e3
        ax_perr.plot(fr, perr_mm, lw=1.2, color=color, alpha=0.85, label=name)
    ax_perr.axhline(0.1, color='red', lw=0.8, ls='--', alpha=0.6, label='0.1mm')
    ax_perr.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                   edgecolor='gray', ncol=5)

    gait     = meta.get('gait_type', '')
    V        = meta.get('V', '?')
    T        = meta.get('T', '?')
    D        = meta.get('D', '?')
    step_h   = meta.get('step_height', 0.0) * 1e3
    step_l   = meta.get('step_length', 0.0) * 1e3
    fig.suptitle(
        f'FR / HL Joint Analysis  |  {gait.upper()}  |  '
        f'v={V}m/s  T={T}s  D={D}  '
        f'step_h={step_h:.0f}mm  step_l={step_l:.0f}mm',
        color='white', fontsize=11)

    return fig


# ══════════════════════════════════════════════════════════════
# Figure 6 — Foot trajectory cmd vs actual (world frame, 4 legs)
# ══════════════════════════════════════════════════════════════
def plot_foot_trajectory(R: SimState, meta: Optional[dict] = None) -> plt.Figure:
    """4-leg world-frame foot trajectory cmd vs actual (v13.py Figure 6).

    Args:
        R:    SimState — foot_target_world_hist / foot_actual_world_hist 사용
        meta: dict — {gait_type, V, T, step_height, step_length, use_nmpc}

    Returns: plt.Figure  (3 rows × 4 cols = 12 panels)
    """
    meta = meta or {}
    N  = R.n_frames
    fr = np.arange(N)

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.32, hspace=0.55,
                           left=0.05, right=0.98, top=0.92, bottom=0.06)

    axis_lbl = ['x [m]', 'y [m]', 'z [m]']

    for li in range(4):
        for ai in range(3):
            ax = fig.add_subplot(gs[ai, li])
            actual = R.foot_actual_world_hist[:, li, ai]
            target = R.foot_target_world_hist[:, li, ai]
            err = actual - target
            valid = ~np.isnan(target)
            if valid.sum() > 0:
                err_max = float(np.max(np.abs(err[valid])))
                err_rms = float(np.sqrt(np.mean(err[valid]**2)))
                title = (f'{["FR","FL","HR","HL"][li]} {axis_lbl[ai].split()[0]}  '
                         f'(swing err: rms={err_rms*1e3:.1f}mm max={err_max*1e3:.1f}mm)')
            else:
                title = f'{["FR","FL","HR","HL"][li]}  foot {axis_lbl[ai].split()[0]}'

            ax.set_facecolor('#16213e')
            ax.set_title(title, color='white', fontsize=8)
            ax.set_xlabel('Frame', color='white', fontsize=7)
            ax.set_ylabel(axis_lbl[ai], color='white', fontsize=7)
            ax.tick_params(colors='gray', labelsize=7)
            ax.grid(True, alpha=0.25, color='gray')
            for sp in ax.spines.values():
                sp.set_edgecolor('gray')
            ax.set_xlim(0, N)
            ax.plot(fr, target, lw=1.6, color='#ff6b6b', ls='--', label='cmd (swing)')
            ax.plot(fr, actual, lw=1.4, color='#00d4ff',          label='actual')
            if ai == 0 and li == 0:
                ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                          edgecolor='gray', loc='upper left')

    gait    = meta.get('gait_type', '')
    V       = meta.get('V', '?')
    T       = meta.get('T', '?')
    step_h  = meta.get('step_height', 0.0) * 1e3
    step_l  = meta.get('step_length', 0.0) * 1e3
    mode    = 'NMPC' if meta.get('use_nmpc') else 'MPC + WBIC'
    fig.suptitle(
        f'Foot Trajectory Cmd vs Actual (world frame)  |  {gait.upper()}  |  '
        f'v={V}m/s  T={T}s  step_h={step_h:.0f}mm  step_l={step_l:.0f}mm  {mode}',
        color='white', fontsize=10)

    return fig
