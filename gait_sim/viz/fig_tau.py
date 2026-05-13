"""gait_sim.viz.fig_tau — τ decomposition figure.

v13.2 Phase 5-e: Figure 4 (FR/HL tau decompose th1~th4) 추출.

함수:
  · plot_tau_decompose(R, meta)  — FR/HL joint th1~th4 의 4종 τ component
                                    (tau_dyn / tau_pd / tau_imp / tau_grf)
                                    overlay (v13.py Figure 4)
"""
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gait_sim.sim_state import SimState

LEG_NAMES = ['FR', 'FL', 'HR', 'HL']


def _style_ax(ax, title, xlabel='Frame', ylabel=''):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors='gray')
    ax.grid(True, alpha=0.25, color='gray')
    for sp in ax.spines.values():
        sp.set_edgecolor('gray')


def plot_tau_decompose(R: SimState, meta: Optional[dict] = None) -> plt.Figure:
    """FR/HL joint τ decomposition (v13.py Figure 4).

    Args:
        R:    SimState — wbc_tau_dyn/pd/imp/grf 사용 (4 components × th1~th4)
        meta: dict — {gait_type, V, T, D, use_mpc, n_mpc}

    Returns: plt.Figure  (4 rows × 2 cols, 8 panels)
    """
    meta = meta or {}
    N = R.n_frames
    fr = np.arange(N)

    fig = plt.figure(figsize=(12, 13))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.38, hspace=0.58,
                           left=0.07, right=0.97, top=0.93, bottom=0.05)

    for col, leg in enumerate([0, 3]):   # FR=0, HL=3
        for row, ji in enumerate([0, 1, 2, 3]):
            ax = fig.add_subplot(gs[row, col])
            _style_ax(ax, f'{LEG_NAMES[leg]} tau decompose th{ji+1} [N·m]', ylabel='[N·m]')
            ax.set_xlim(0, N)
            ax.plot(fr, R.wbc_tau_dyn[:, leg, ji], lw=1.4, color='#00d4ff', label='tau_dyn')
            ax.plot(fr, R.wbc_tau_pd [:, leg, ji], lw=1.4, color='#ff6b35', label='tau_pd')
            ax.plot(fr, R.wbc_tau_imp[:, leg, ji], lw=1.4, color='#00ff99', label='tau_imp')
            ax.plot(fr, R.wbc_tau_grf[:, leg, ji], lw=1.4, color='#ffd166', label='tau_grf')
            ax.axhline(0, color='white', lw=0.5, ls='--', alpha=0.4)
            ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                      edgecolor='gray', ncol=2)

    gait    = meta.get('gait_type', '')
    V       = meta.get('V', '?')
    T       = meta.get('T', '?')
    D       = meta.get('D', '?')
    use_mpc = meta.get('use_mpc', True)
    n_mpc   = meta.get('n_mpc', 10)
    mode    = f'MPC QP (N={n_mpc})' if use_mpc else 'QP GRF'
    fig.suptitle(
        f'FR / HL tau decompose th1~th4  |  {gait.upper()}  |  '
        f'v={V}m/s  T={T}s  D={D}  {mode}',
        color='white', fontsize=10)

    return fig
