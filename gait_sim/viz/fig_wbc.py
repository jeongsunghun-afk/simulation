"""gait_sim.viz.fig_wbc — WBC analysis figure.

v13.2 Phase 5-e: Figure 3 (WBC 분석 3×2 FR/HL) 추출.

함수:
  · plot_wbc(R, meta)  — FR/HL tau_cmd−tau_grf + GRF lam_z (des vs calc)
                          + GRF lam_x/lam_y + 마찰 추 한계 (v13.py Figure 3)
"""
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gait_sim.sim_state import SimState

LEG_NAMES = ['FR', 'FL', 'HR', 'HL']
N_JOINTS_PER_LEG = [5, 5, 5, 5]
_AX5_COLORS = ['#ff6b6b', '#ffd166', '#06d6a0', '#4cc9f0', '#f72585']


def _style_ax(ax, title, xlabel='Frame', ylabel=''):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors='gray')
    ax.grid(True, alpha=0.25, color='gray')
    for sp in ax.spines.values():
        sp.set_edgecolor('gray')


def plot_wbc(R: SimState, meta: Optional[dict] = None) -> plt.Figure:
    """WBC FR/HL τ_cmd 와 GRF (lam_z, lam_xy + friction cone) 시각화.

    Args:
        R:    SimState — wbc_tau_cmd/grf + wbc_lam_des/calc 사용
        meta: dict — {gait_type, V, T, D, use_mpc, n_mpc, mu_friction, total_mass}

    Returns: plt.Figure  (3 rows × 2 cols = 6 panels)
    """
    meta = meta or {}
    N  = R.n_frames
    fr = np.arange(N)
    mu = meta.get('mu_friction', 0.6)

    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.38, hspace=0.60,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)

    for col, leg in enumerate([0, 3]):   # FR=0, HL=3
        nj = N_JOINTS_PER_LEG[leg]

        # row 0: tau_cmd (실제 motor 명령 — clip 후, PD/imp/dyn/grf 모두 포함)
        ax_tc = fig.add_subplot(gs[0, col])
        _style_ax(ax_tc, f'{LEG_NAMES[leg]} tau_cmd [N·m]', ylabel='[N·m]')
        ax_tc.set_xlim(0, N)
        for j in range(nj):
            ax_tc.plot(fr, R.wbc_tau_cmd[:, leg, j], lw=1.4, color=_AX5_COLORS[j], label=f'th{j+1}')
        ax_tc.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
        ax_tc.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                     edgecolor='gray', ncol=5)

        # row 1: GRF lam_z (lam_des vs lam_calc)
        ax_fz = fig.add_subplot(gs[1, col])
        _style_ax(ax_fz, f'{LEG_NAMES[leg]} GRF lam_z [N]', ylabel='[N]')
        ax_fz.set_xlim(0, N)
        ax_fz.plot(fr, R.wbc_lam_des [:, leg, 2], lw=1.8, color='#00d4ff', label='lam_z des')
        ax_fz.plot(fr, R.wbc_lam_calc[:, leg, 2], lw=1.4, color='magenta',
                   ls='--', label='lam_z calc')
        ax_fz.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
        ax_fz.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                     edgecolor='gray')

        # row 2: GRF lam_x/lam_y + friction cone limit
        ax_fxy = fig.add_subplot(gs[2, col])
        _style_ax(ax_fxy, f'{LEG_NAMES[leg]} GRF lam_x/lam_y + Friction Cone [N]',
                  ylabel='[N]')
        ax_fxy.set_xlim(0, N)
        fric_limit = mu * np.abs(R.wbc_lam_des[:, leg, 2])
        ax_fxy.plot(fr, R.wbc_lam_des [:, leg, 0], lw=1.4, color='#ff6b6b', label='lam_x des')
        ax_fxy.plot(fr, R.wbc_lam_des [:, leg, 1], lw=1.4, color='#ffd166', label='lam_y des')
        ax_fxy.plot(fr, R.wbc_lam_calc[:, leg, 0], lw=1.2, color='#ff6b6b',
                    ls='--', label='lam_x calc')
        ax_fxy.plot(fr, R.wbc_lam_calc[:, leg, 1], lw=1.2, color='#ffd166',
                    ls='--', label='lam_y calc')
        ax_fxy.fill_between(fr, fric_limit, -fric_limit,
                            color='white', alpha=0.07, label=f'mu*lam_z (mu={mu})')
        ax_fxy.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
        ax_fxy.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                      edgecolor='gray', ncol=3)

    gait     = meta.get('gait_type', '')
    V        = meta.get('V', '?')
    T        = meta.get('T', '?')
    D        = meta.get('D', '?')
    use_mpc  = meta.get('use_mpc', True)
    n_mpc    = meta.get('n_mpc', 10)
    total_m  = meta.get('total_mass', 0)
    mode     = f'MPC QP (N={n_mpc})' if use_mpc else 'QP GRF'
    fig.suptitle(
        f'WBC Analysis  FR/HL  |  {gait.upper()}  |  v={V}m/s  T={T}s  D={D}  '
        f'{mode}  mu={mu}  total_mass={total_m:.2f}kg',
        color='white', fontsize=10)

    return fig
