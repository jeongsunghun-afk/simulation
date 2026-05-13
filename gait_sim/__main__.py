"""gait_sim.__main__ — Entry point for `python -m gait_sim`.

v13.2 Phase 6-e: 통합 시뮬레이션 + viz orchestrator.

Usage:
    python -m gait_sim                       # default CFG (trot, v11 MPC+WBIC)
    python -m gait_sim --gait walk           # walk gait
    python -m gait_sim --nmpc                # NMPC mode
    python -m gait_sim --no-show             # 저장만, 창 안띄움 (CI / batch)
    python -m gait_sim --save /tmp/out       # save PNG to dir
"""
import argparse
import os
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog='gait_sim',
                                 description='Quadruped gait simulator (v13 package split)')
    p.add_argument('--gait', default=None,
                   help="Gait: 'walk' / 'amble' / 'pace' / 'trot' / 'canter' / 'gallop'")
    p.add_argument('--nmpc', action='store_true',
                   help='Use NMPC (crocoddyl FDDP) instead of MPC+WBIC')
    p.add_argument('--no-show', action='store_true',
                   help="Don't display figures (matplotlib Agg backend)")
    p.add_argument('--save', default=None, metavar='DIR',
                   help='Save figures as PNG to DIR (creates if missing)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.no_show or args.save:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from gait_sim.runner import run_simulation
    from gait_sim.viz import (
        plot_main_static, plot_joint_analysis, plot_wbc, plot_tau_decompose,
        plot_body_state, plot_foot_trajectory, plot_gait_diagram, plot_diagnostics,
    )

    # Run simulation
    R, meta = run_simulation(
        gait_type=args.gait,
        use_nmpc=args.nmpc if args.nmpc else None,
    )

    # Generate all 8 figures
    plots = {
        'fig1_main':   plot_main_static,
        'fig2_joints': plot_joint_analysis,
        'fig3_wbc':    plot_wbc,
        'fig4_tau':    plot_tau_decompose,
        'fig5_body':   plot_body_state,
        'fig6_foot':   plot_foot_trajectory,
        'fig7_gait':   plot_gait_diagram,
        'fig8_diag':   plot_diagnostics,
    }
    figs = {}
    for name, plot_fn in plots.items():
        try:
            figs[name] = plot_fn(R, meta)
        except Exception as e:
            print(f"  [WARN] {name} 플롯 실패: {e}")

    # Save if requested
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for name, fig in figs.items():
            out = os.path.join(args.save, f'{name}.png')
            fig.savefig(out, dpi=80, facecolor=fig.get_facecolor())
            print(f"  saved {out}")

    # Show if not suppressed
    if not (args.no_show or args.save):
        plt.show()


if __name__ == '__main__':
    main()
