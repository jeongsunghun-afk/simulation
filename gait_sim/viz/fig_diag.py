"""gait_sim.viz.fig_diag — Diagnostic + gait diagram figures.

v13.2 Phase 5-b: Figure 7 (gait diagram) 추출.

함수:
  · plot_gait_diagram(R, meta)   — Hildebrand-style stance chart + Fz overlay
                                    (v13.py Figure 7)

후속:
  · plot_diagnostics(R, meta)    — Figure 8 (friction cone usage + CoT + slip + τ margin)
"""
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gait_sim.sim_state import SimState

LEG_NAMES   = ['FR', 'FL', 'HR', 'HL']
_LEG_COLORS = ['#ff6b6b', '#ffd166', '#06d6a0', '#4cc9f0']   # v13 fig7 color set


# ══════════════════════════════════════════════════════════════
# Figure 7 — Gait diagram (Hildebrand-style stance chart)
# ══════════════════════════════════════════════════════════════
def plot_gait_diagram(R: SimState, meta: Optional[dict] = None) -> plt.Figure:
    """Gait diagram + per-leg Fz line plot.

    Args:
        R:    SimState — phase_hist + wbc_lam_des 사용
        meta: dict — {gait_type, T, D, use_nmpc}.
                     D 는 swing_ratio (0~1) — phase<D 이면 swing, ≥D 이면 stance.

    Returns: plt.Figure
    """
    meta = meta or {}
    N = R.n_frames
    fr = np.arange(N)

    D_swing = meta.get('D', 0.5)
    gait    = meta.get('gait_type', '')
    T       = meta.get('T', '?')
    use_nmpc = meta.get('use_nmpc', False)

    fig = plt.figure(figsize=(13, 5))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.42,
                           left=0.07, right=0.98, top=0.86, bottom=0.10,
                           height_ratios=[3, 2])

    # phase_hist 기반 stance mask (phase ≥ D_swing 이면 stance)
    # phase_hist 가 0 이면 (alloc init) 모두 swing 처리됨 → meta D 가 0.5 인 trot case 기본 동작
    stance_mat = (R.phase_hist >= D_swing)

    # Top: gait diagram with Fz heatmap
    ax_g = fig.add_subplot(gs[0, 0])
    ax_g.set_facecolor('#16213e')
    ax_g.set_title(f'Gait Diagram  |  {gait.upper()}  |  T={T}s  D={D_swing}  '
                   f'(filled = stance, color depth = Fz [N])',
                   color='white', fontsize=9)
    ax_g.set_xlim(0, N)
    ax_g.set_ylim(-0.5, 3.5)
    ax_g.set_yticks([0, 1, 2, 3])
    ax_g.set_yticklabels(LEG_NAMES, color='white')
    ax_g.set_xlabel('Frame', color='white', fontsize=8)
    ax_g.tick_params(colors='gray')
    ax_g.grid(True, alpha=0.2, axis='x', color='gray')
    for sp in ax_g.spines.values():
        sp.set_edgecolor('gray')

    fz_max = max(1.0, float(np.max(R.wbc_lam_des[:, :, 2])))
    for li in range(4):
        fz = R.wbc_lam_des[:, li, 2]
        in_stance = False
        seg_start = 0
        for fi in range(N + 1):
            is_stance = stance_mat[fi, li] if fi < N else False
            if is_stance and not in_stance:
                seg_start = fi
                in_stance = True
            elif not is_stance and in_stance:
                seg_fz_avg = float(np.mean(fz[seg_start:fi]))
                alpha = 0.25 + 0.7 * min(1.0, seg_fz_avg / fz_max)
                ax_g.fill_between([seg_start, fi], li - 0.35, li + 0.35,
                                  color='#06d6a0', alpha=alpha, edgecolor='none')
                in_stance = False

    # Bottom: contact force Fz over time per leg
    ax_fz = fig.add_subplot(gs[1, 0])
    ax_fz.set_facecolor('#16213e')
    ax_fz.set_title('Contact Force Fz [N] per leg', color='white', fontsize=9)
    ax_fz.set_xlim(0, N)
    ax_fz.set_xlabel('Frame', color='white', fontsize=8)
    ax_fz.set_ylabel('Fz [N]', color='white', fontsize=8)
    ax_fz.tick_params(colors='gray')
    ax_fz.grid(True, alpha=0.25, color='gray')
    for sp in ax_fz.spines.values():
        sp.set_edgecolor('gray')
    for li, (lname, lc) in enumerate(zip(LEG_NAMES, _LEG_COLORS)):
        ax_fz.plot(fr, R.wbc_lam_des[:, li, 2], lw=1.2, color=lc, label=lname)
    ax_fz.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                 edgecolor='gray', ncol=4)

    mode = 'NMPC (FDDP)' if use_nmpc else 'MPC + WBIC'
    fig.suptitle(
        f'Gait Diagram  |  {gait.upper()}  |  {mode}',
        color='white', fontsize=10)

    return fig
