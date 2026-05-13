"""gait_sim.viz — Visualization (matplotlib figures using SimState).

각 figure 함수는 `SimState` 객체를 받아 `plt.Figure` 를 반환합니다.

Public API:
    plot_body_state(R, meta)     — Figure 5: body pos/vel/orient/omega vs cmd
    plot_joint_analysis(R, meta) — Figure 2: FR/HL joint pos/vel/acc/jerk + opt-IK
    plot_gait_diagram(R, meta)   — Figure 7: Hildebrand stance chart + Fz

다음 세션 (TODO):
    plot_anim(R, meta)           — Figure 1: 3D 애니메이션
    plot_wbc(R, meta)            — Figure 3: WBC decomposition
    plot_tau(R, meta)            — Figure 4: τ analysis
    plot_foot(R, meta)           — Figure 6: foot trajectory cmd vs actual
    plot_diagnostics(R, meta)    — Figure 8: friction cone + CoT + slip + τ margin
"""
from gait_sim.viz.fig_main import plot_body_state
from gait_sim.viz.fig_legs import plot_joint_analysis
from gait_sim.viz.fig_diag import plot_gait_diagram

__all__ = ['plot_body_state', 'plot_joint_analysis', 'plot_gait_diagram']
