"""gait_sim.viz — Visualization (matplotlib figures using SimState).

각 figure 함수는 `SimState` 객체를 받아 `plt.Figure` 를 반환합니다.

Public API:
    plot_main_static(R, meta)    — Figure 1 static: gait phase + foot Z/X/dZ/dX/d²Z/d²X
    plot_anim(R, meta)           — Figure 1: 3D 애니메이션 (FuncAnimation, returns (fig, anim))
    plot_joint_analysis(R, meta) — Figure 2: FR/HL joint pos/vel/acc/jerk + opt-IK
    plot_wbc(R, meta)            — Figure 3: WBC FR/HL τ_cmd + GRF + friction cone
    plot_tau_decompose(R, meta)  — Figure 4: τ decomposition (tau_dyn/pd/imp/grf)
    plot_body_state(R, meta)     — Figure 5: body pos/vel/orient/omega vs cmd
    plot_foot_trajectory(R, meta)— Figure 6: foot world cmd vs actual (4 legs × xyz)
    plot_gait_diagram(R, meta)   — Figure 7: Hildebrand stance chart + Fz
    plot_diagnostics(R, meta)    — Figure 8: friction cone + CoT + slip + τ margin
"""
from gait_sim.viz.fig_main import plot_body_state, plot_main_static, plot_anim
from gait_sim.viz.fig_legs import plot_joint_analysis, plot_foot_trajectory
from gait_sim.viz.fig_wbc  import plot_wbc
from gait_sim.viz.fig_tau  import plot_tau_decompose
from gait_sim.viz.fig_diag import plot_gait_diagram, plot_diagnostics

__all__ = [
    'plot_main_static',
    'plot_anim',
    'plot_joint_analysis',
    'plot_wbc',
    'plot_tau_decompose',
    'plot_body_state',
    'plot_foot_trajectory',
    'plot_gait_diagram',
    'plot_diagnostics',
]
