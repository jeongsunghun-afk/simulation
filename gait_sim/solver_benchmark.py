"""gait_sim.solver_benchmark — QP solver 비교 (quadprog / osqp / proxqp).

v13.0a: 동일 시뮬레이션 (precompute 공유) 을 3 solver 로 비교.
        WBIC loop walltime + final body state metric 일치 검증.

Usage:
    python3 gait_sim/solver_benchmark.py            # 1 cycle trot (~50s)
    python3 gait_sim/solver_benchmark.py --short    # 동일
    python3 gait_sim/solver_benchmark.py --full     # 4 cycle (~3min)
"""
import argparse
import copy
import os
import sys
import time
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_THIS)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


def deepcopy_simstate(R):
    """SimState 의 np array fields 깊은 복사. dataclass + numpy 호환."""
    R_copy = copy.copy(R)
    for fname in R.__dataclass_fields__:
        val = getattr(R, fname)
        if isinstance(val, np.ndarray):
            setattr(R_copy, fname, val.copy())
        elif isinstance(val, list):
            setattr(R_copy, fname, list(val))
        elif isinstance(val, dict):
            setattr(R_copy, fname, dict(val))
    return R_copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--full', action='store_true', help='N_CYCLES=4 (default 1)')
    p.add_argument('--solvers', default='quadprog,osqp,proxqp',
                   help='comma-separated solver names')
    args = p.parse_args()

    from gait_sim import config
    if not args.full:
        config.CFG.n_cycles = 1
        config.N_CYCLES = 1
        config.N_FRAMES = int(config.N_CYCLES * config.T / config.DT)

    from gait_sim.runner import (
        precompute_trajectories, compute_derivatives,
        init_wbic_state, run_wbic_loop, postprocess_foot_world,
    )
    from gait_sim.sim_state import SimState
    from gait_sim.gait import GaitScheduler

    solvers = [s.strip() for s in args.solvers.split(',') if s.strip()]
    print("═" * 64)
    print(f"  QP Solver Benchmark  (N_FRAMES={config.N_FRAMES}, "
          f"solvers={solvers})")
    print("═" * 64)

    # 1) precompute + derivatives once (solver-independent)
    R_template = SimState.alloc(n_frames=config.N_FRAMES, dt=config.DT)
    sched = GaitScheduler()
    t0 = time.perf_counter()
    precompute_trajectories(R_template, sched)
    compute_derivatives(R_template)
    t_pre = time.perf_counter() - t0
    print(f"\n  precompute + derivatives : {t_pre:.1f}s "
          f"(공유, solver 영향 없음)\n")

    # 2) Per-solver: deepcopy + WBIC loop + foot world postprocess
    results = {}
    for solver in solvers:
        config.CFG.qp_solver = solver
        R = deepcopy_simstate(R_template)
        body_state, foot_z_home = init_wbic_state(R)

        t0 = time.perf_counter()
        try:
            diag = run_wbic_loop(R, sched, body_state, foot_z_home)
        except Exception as e:
            print(f"  [{solver:8s}] FAIL — {e}")
            results[solver] = None
            continue
        t_wbic = time.perf_counter() - t0

        postprocess_foot_world(R, sched)

        # Metrics for comparison
        roll = np.degrees(np.arctan2(R.body_R_hist[:, 2, 1], R.body_R_hist[:, 2, 2]))
        pitch = np.degrees(np.arcsin(np.clip(-R.body_R_hist[:, 2, 0], -1, 1)))
        tau_peak = float(np.max(np.abs(R.wbc_tau_cmd)))
        fz_peak = float(np.max(R.wbc_lam_des[:, :, 2]))
        x_final = float(R.body_pos_hist[-1, 0])
        y_final = float(R.body_pos_hist[-1, 1])
        vx_mean = float(R.body_v_hist[:, 0].mean())
        roll_max = float(np.max(np.abs(roll)))
        pitch_max = float(np.max(np.abs(pitch)))
        success_rate = float(np.mean(R.wbic_status_hist))
        results[solver] = dict(
            t_wbic=t_wbic, diag=diag, x_final=x_final, y_final=y_final,
            vx_mean=vx_mean, roll_max=roll_max, pitch_max=pitch_max,
            tau_peak=tau_peak, fz_peak=fz_peak, success_rate=success_rate,
        )
        print(f"  [{solver:8s}] WBIC loop {t_wbic*1e3:7.0f}ms  "
              f"x={x_final:.3f}m  τ={tau_peak:5.1f}Nm  "
              f"Fz={fz_peak:.0f}N  success={success_rate*100:.1f}%")

    # 3) Comparison table
    print()
    print("═" * 64)
    print("  Solver 비교 (vs quadprog baseline)")
    print("═" * 64)
    base = results.get('quadprog')
    header = f"  {'solver':10s} {'WBIC':>10s} {'speedup':>9s}  {'x_final':>9s} {'τ_peak':>9s}"
    print(header)
    for solver, r in results.items():
        if r is None:
            print(f"  {solver:10s} FAIL")
            continue
        speedup = (base['t_wbic'] / r['t_wbic']) if base else 1.0
        marker = '✓' if base and (
            abs(r['x_final'] - base['x_final']) < 0.01 and
            abs(r['tau_peak'] - base['tau_peak']) / max(base['tau_peak'], 1) < 0.10
        ) else '⚠'
        print(f"  {solver:10s} {r['t_wbic']*1e3:8.0f}ms {speedup:7.2f}x  "
              f"{r['x_final']:8.3f}m {r['tau_peak']:7.1f}Nm  {marker}")
    print("═" * 64)
    print(f"  Sim time = {config.N_FRAMES * config.DT * 1e3:.0f} ms "
          f"({config.N_FRAMES} frames @ {config.DT*1e3}ms)")


if __name__ == '__main__':
    main()
