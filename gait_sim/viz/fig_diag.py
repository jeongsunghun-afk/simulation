"""gait_sim.viz.fig_diag — Diagnostic + gait diagram figures.

v13.2 Phase 5-b/g: Figure 7 (gait diagram) + Figure 8 (diagnostics) 추출.

함수:
  · plot_gait_diagram(R, meta)   — Hildebrand-style stance chart + Fz overlay (Figure 7)
  · plot_diagnostics(R, meta)    — friction cone + CoT + slip + τ margin (Figure 8)
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


# ══════════════════════════════════════════════════════════════
# Figure 8 — Diagnostics (friction cone + CoT + slip + τ margin)
# ══════════════════════════════════════════════════════════════
def plot_diagnostics(R: SimState, meta: Optional[dict] = None) -> plt.Figure:
    """Performance diagnostics (v13.py Figure 8) — 4 panels:
        (1) friction cone usage ratio  |F_xy| / (μ·F_z)  per leg
        (2) mechanical power Σ|τ·dq| + CoT (cost of transport)
        (3) stance foot slip velocity ‖v_foot‖ per leg
        (4) torque saturation margin max_j |τ_j|/τ_limit_j per leg

    Args:
        R:    SimState — wbc_lam_des, wbc_tau_cmd, joint_hist,
                          body_v_hist, foot_actual_world_hist, phase_hist 사용
        meta: dict — {gait_type, V, T, use_nmpc,
                       mu_friction, total_mass, g_acc, D,
                       joint_torque_limit (np.ndarray len ≥ 5)}

    Returns: plt.Figure (2 rows × 2 cols + twin axis on CoT panel = 5 axes)
    """
    meta = meta or {}
    N  = R.n_frames
    fr = np.arange(N)
    mu = meta.get('mu_friction', 0.6)
    total_m = meta.get('total_mass', 40.0)
    g_acc   = meta.get('g_acc', 9.81)
    D_swing = meta.get('D', 0.5)
    tau_lim = np.asarray(meta.get('joint_torque_limit',
                                  [60., 120., 120., 60., 60.]), dtype=float)
    n_jt_max = R.wbc_tau_cmd.shape[2]

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.32, hspace=0.50,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)

    # (1) Friction cone usage ratio
    ax_fc = fig.add_subplot(gs[0, 0])
    fc_ratio = np.zeros((N, 4))
    for li in range(4):
        fxy = np.linalg.norm(R.wbc_lam_des[:, li, :2], axis=1)
        fz  = R.wbc_lam_des[:, li, 2]
        safe = fz > 1.0
        fc_ratio[safe, li]  = fxy[safe] / (mu * fz[safe])
        fc_ratio[~safe, li] = np.nan
    fc_max = [float(np.nanmax(fc_ratio[:, li])) if np.any(~np.isnan(fc_ratio[:, li])) else 0.0
              for li in range(4)]
    ax_fc.set_facecolor('#16213e')
    ax_fc.set_title(
        f'Friction Cone Usage  |F_xy|/(μ·F_z)  μ={mu}  (>1 = slip)\n'
        f'peak: FR={fc_max[0]:.2f} FL={fc_max[1]:.2f} '
        f'HR={fc_max[2]:.2f} HL={fc_max[3]:.2f}',
        color='white', fontsize=9)
    ax_fc.set_xlabel('Frame', color='white', fontsize=8)
    ax_fc.set_ylabel('ratio', color='white', fontsize=8)
    ax_fc.tick_params(colors='gray')
    ax_fc.grid(True, alpha=0.25, color='gray')
    for sp in ax_fc.spines.values():
        sp.set_edgecolor('gray')
    ax_fc.set_xlim(0, N)
    for li, (lname, lc) in enumerate(zip(LEG_NAMES, _LEG_COLORS)):
        ax_fc.plot(fr, fc_ratio[:, li], lw=1.2, color=lc, label=lname)
    ax_fc.axhline(1.0, color='red', lw=1.0, ls='--', alpha=0.6, label='slip threshold')
    ax_fc.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                 edgecolor='gray', ncol=5)

    # (2) Mechanical power + CoT
    ax_pw = fig.add_subplot(gs[0, 1])
    DT = R.dt
    dq_num = np.gradient(R.joint_hist, axis=0) / DT
    power_per_leg = np.sum(np.abs(R.wbc_tau_cmd * dq_num), axis=2)
    power_total   = np.sum(power_per_leg, axis=1)
    v_actual = np.linalg.norm(R.body_v_hist, axis=1)
    cot_inst = power_total / np.maximum(total_m * g_acc * np.maximum(v_actual, 0.05), 1e-3)
    power_avg = float(np.mean(power_total))
    cot_avg   = float(np.mean(cot_inst))

    ax_pw.set_facecolor('#16213e')
    ax_pw.set_title(
        f'Mechanical Power [W]  +  CoT (right axis)\n'
        f'P_avg={power_avg:.1f}W  P_peak={float(np.max(power_total)):.1f}W  '
        f'CoT_avg={cot_avg:.2f}  (lower = more efficient)',
        color='white', fontsize=9)
    ax_pw.set_xlabel('Frame', color='white', fontsize=8)
    ax_pw.set_ylabel('Power [W]', color='white', fontsize=8)
    ax_pw.tick_params(colors='gray')
    ax_pw.grid(True, alpha=0.25, color='gray')
    for sp in ax_pw.spines.values():
        sp.set_edgecolor('gray')
    ax_pw.set_xlim(0, N)
    ax_pw.plot(fr, power_total, lw=1.4, color='#06d6a0', label='Σ |τ·dq|')
    ax_pw.axhline(power_avg, color='#06d6a0', lw=0.8, ls='--', alpha=0.5)
    ax_pw_r = ax_pw.twinx()
    ax_pw_r.plot(fr, cot_inst, lw=1.0, color='#ffd166', alpha=0.7, label='CoT')
    ax_pw_r.set_ylabel('CoT (-)', color='#ffd166', fontsize=8)
    ax_pw_r.tick_params(axis='y', colors='#ffd166')
    ax_pw.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                 edgecolor='gray', loc='upper left')
    ax_pw_r.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                   edgecolor='gray', loc='upper right')

    # (3) Stance foot slip velocity (phase_hist 기반 stance 필터)
    ax_sl = fig.add_subplot(gs[1, 0])
    v_foot = np.gradient(R.foot_actual_world_hist, axis=0) / DT
    slip_speed = np.linalg.norm(v_foot, axis=2)
    stance_mat = (R.phase_hist >= D_swing)
    slip_stance = np.where(stance_mat, slip_speed, np.nan)
    slip_max = [float(np.nanmax(slip_stance[:, li])) if np.any(~np.isnan(slip_stance[:, li])) else 0.0
                for li in range(4)]
    ax_sl.set_facecolor('#16213e')
    ax_sl.set_title(
        f'Stance Foot Slip Velocity ‖v_foot‖ [m/s]  (ideal: 0)\n'
        f'peak: FR={slip_max[0]:.2f} FL={slip_max[1]:.2f} '
        f'HR={slip_max[2]:.2f} HL={slip_max[3]:.2f}',
        color='white', fontsize=9)
    ax_sl.set_xlabel('Frame', color='white', fontsize=8)
    ax_sl.set_ylabel('|v| [m/s]', color='white', fontsize=8)
    ax_sl.tick_params(colors='gray')
    ax_sl.grid(True, alpha=0.25, color='gray')
    for sp in ax_sl.spines.values():
        sp.set_edgecolor('gray')
    ax_sl.set_xlim(0, N)
    for li, (lname, lc) in enumerate(zip(LEG_NAMES, _LEG_COLORS)):
        ax_sl.plot(fr, slip_stance[:, li], lw=1.2, color=lc, label=lname)
    ax_sl.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                 edgecolor='gray', ncol=4)

    # (4) Torque saturation margin
    ax_tm = fig.add_subplot(gs[1, 1])
    tau_ratio = np.zeros((N, 4, n_jt_max))
    for j in range(n_jt_max):
        if j < len(tau_lim) and tau_lim[j] > 0:
            tau_ratio[:, :, j] = np.abs(R.wbc_tau_cmd[:, :, j]) / tau_lim[j]
    tau_worst = np.max(tau_ratio, axis=2)
    tau_max = [float(np.max(tau_worst[:, li])) for li in range(4)]
    ax_tm.set_facecolor('#16213e')
    ax_tm.set_title(
        f'Torque Saturation: max_j |τ_j|/τ_limit_j per leg  (>1 = saturate)\n'
        f'peak: FR={tau_max[0]:.2f} FL={tau_max[1]:.2f} '
        f'HR={tau_max[2]:.2f} HL={tau_max[3]:.2f}',
        color='white', fontsize=9)
    ax_tm.set_xlabel('Frame', color='white', fontsize=8)
    ax_tm.set_ylabel('|τ|/τ_max', color='white', fontsize=8)
    ax_tm.tick_params(colors='gray')
    ax_tm.grid(True, alpha=0.25, color='gray')
    for sp in ax_tm.spines.values():
        sp.set_edgecolor('gray')
    ax_tm.set_xlim(0, N)
    for li, (lname, lc) in enumerate(zip(LEG_NAMES, _LEG_COLORS)):
        ax_tm.plot(fr, tau_worst[:, li], lw=1.2, color=lc, label=lname)
    ax_tm.axhline(1.0, color='red', lw=1.0, ls='--', alpha=0.6, label='saturation')
    ax_tm.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                 edgecolor='gray', ncol=5)

    gait    = meta.get('gait_type', '')
    V       = meta.get('V', '?')
    T       = meta.get('T', '?')
    mode    = 'NMPC (FDDP)' if meta.get('use_nmpc') else 'MPC + WBIC'
    fig.suptitle(
        f'Diagnostic  |  {gait.upper()}  |  '
        f'v={V}m/s  T={T}s  μ={mu}  {mode}',
        color='white', fontsize=10)

    return fig
