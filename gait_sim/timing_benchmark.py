"""gait_sim.timing_benchmark — Isaac Lab 통합 평가용 timing 실측.

각 stage 별 walltime + per-call latency 측정.

Usage: python3 gait_sim/timing_benchmark.py
       python3 gait_sim/timing_benchmark.py --nmpc       # NMPC 모드만
       python3 gait_sim/timing_benchmark.py --short      # N_CYCLES=1 (빠른 측정)

출력:
  · precompute_trajectories 총 시간 + per-frame
  · compute_derivatives 시간
  · run_wbic_loop 총 시간 + per-frame breakdown (MPC / leg dynamics / WBIC QP / body integrate)
  · MPC QP solve time 분포 (mean / p50 / p95 / max)
  · WBIC QP solve time 분포
  · Real-time factor (sim_time / wall_time)
"""
import argparse
import os
import sys
import time

# Ensure gait_sim/ parent dir is on path (subprocess 호환)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import numpy as np


def measure_qp_calls(R, sched, body_state, foot_z_home, n_frames_sample=50):
    """MPC QP + WBIC QP per-call timing (representative sample)."""
    from gait_sim.config import CFG
    from gait_sim.controllers.mpc import mpc_qp_plan, DT_MPC, N_MPC
    from gait_sim.controllers.wbic import wbic_qp_leg, wbic_qp_full
    from gait_sim.kinematics import compute_jacobian_sim, compute_gravity_torque_sim
    from gait_sim.dynamics import rnea, compute_mh_leg
    from gait_sim.model import (
        LEG_DH, N_JOINTS_PER_LEG, LINK_MASS_PER_LEG,
        BODY_INERTIA, TOTAL_MASS, MU_FRICTION,
    )
    from gait_sim.config import G_ACC

    mpc_times = []
    wbic_leg_times = []
    wbic_full_times = []

    swing_flag = (R.phase_hist < sched.swing_ratio)
    stride = max(1, R.n_frames // n_frames_sample)

    for fi in range(0, R.n_frames, stride):
        t_cur = fi * R.dt
        contact_mask = ~swing_flag[fi]

        # ── MPC QP timing ────────────────────────────────
        cs = np.zeros((N_MPC, 4), dtype=bool)
        fp = np.zeros((N_MPC, 4, 3))
        for k in range(N_MPC):
            t_k = t_cur + k * DT_MPC
            for leg in range(4):
                cs[k, leg] = not sched.is_swing(leg, t_k)
                fp[k, leg] = R.foot_hist[fi, leg]
        x0 = np.array([0., 0., 0., 0., 0., -foot_z_home,
                        0., 0., 0., 1.0, 0., 0., -G_ACC])
        t0 = time.perf_counter()
        lam = mpc_qp_plan(x0, cs, fp, x_ref_step=None, ltv=False)
        mpc_times.append(time.perf_counter() - t0)

        # ── WBIC per-leg + per-leg M, h timing ───────────
        # M, h, J, RNEA per leg (1 leg sample for QP timing)
        leg = 0
        nj = N_JOINTS_PER_LEG[leg]
        dh = LEG_DH[leg]
        lm = LINK_MASS_PER_LEG[leg]
        q  = R.joint_hist[fi, leg, :nj]
        dq = R.joint_vel_hist[fi, leg, :nj]
        ddq = R.joint_acc_hist[fi, leg, :nj]
        J  = compute_jacobian_sim(q, dh, True)
        M, h = compute_mh_leg(q, dq, dh, lm)
        tau_dyn = rnea(q, dq, ddq, dh, lm)
        tau_ff  = tau_dyn - J.T @ lam[leg]

        t0 = time.perf_counter()
        wbic_qp_leg(M, h, ddq, tau_ff, lam[leg], J, contact=contact_mask[leg], nj=nj,
                     w_ddq=1.0, w_tau=0.01, w_lam=0.001, lamz_min=1.0, mu=MU_FRICTION)
        wbic_leg_times.append(time.perf_counter() - t0)

        # ── WBIC full body 6-DoF QP timing ───────────────
        M_legs = [M.copy() for _ in range(4)]
        h_legs = [h.copy() for _ in range(4)]
        ddq_des_legs = [ddq.copy() for _ in range(4)]
        tau_ff_legs = [tau_ff.copy() for _ in range(4)]
        J_legs = [J.copy() for _ in range(4)]
        foot_world = R.foot_hist[fi].copy()
        for li in range(4):
            foot_world[li] = body_state['pos'] + body_state['R'] @ R.foot_hist[fi, li]

        t0 = time.perf_counter()
        wbic_qp_full(
            M_legs, h_legs, ddq_des_legs, tau_ff_legs, lam,
            J_legs, contact_mask=contact_mask, nj_per_leg=N_JOINTS_PER_LEG,
            foot_world_all=foot_world, body_pos=body_state['pos'],
            v_dot_des_fb=np.zeros(6), M_total=TOTAL_MASS,
            I_body=BODY_INERTIA, omega_world=np.zeros(3),
            w_ddq=1.0, w_tau=0.01, w_lam=0.001, w_fb=0.1,
            lamz_min=1.0, mu=MU_FRICTION,
            stance_foot_J_v11=None,
        )
        wbic_full_times.append(time.perf_counter() - t0)

    return {
        'mpc_qp': np.array(mpc_times),
        'wbic_qp_leg': np.array(wbic_leg_times),
        'wbic_qp_full': np.array(wbic_full_times),
    }


def fmt_dist(label, times_us):
    """Distribution stats from microsecond array."""
    return (f"  {label:24s} n={len(times_us):3d}  "
            f"mean={times_us.mean():.0f}μs  "
            f"p50={np.percentile(times_us, 50):.0f}μs  "
            f"p95={np.percentile(times_us, 95):.0f}μs  "
            f"max={times_us.max():.0f}μs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nmpc', action='store_true', help='NMPC 모드 측정')
    p.add_argument('--short', action='store_true', help='N_CYCLES=1 짧은 측정')
    args = p.parse_args()

    from gait_sim import config
    if args.short:
        config.CFG.n_cycles = 1
        config.N_CYCLES = 1
        config.N_FRAMES = int(config.N_CYCLES * config.T / config.DT)
    if args.nmpc:
        config.CFG.use_nmpc = True
    else:
        config.CFG.use_nmpc = False

    from gait_sim.runner import (
        precompute_trajectories, compute_derivatives,
        init_wbic_state, run_wbic_loop, postprocess_foot_world,
    )
    from gait_sim.sim_state import SimState
    from gait_sim.gait import GaitScheduler

    print("═" * 60)
    print(f"  gait_sim timing benchmark  "
          f"(N_FRAMES={config.N_FRAMES}, "
          f"mode={'NMPC' if args.nmpc else 'MPC+WBIC'})")
    print("═" * 60)

    R = SimState.alloc(n_frames=config.N_FRAMES, dt=config.DT)
    sched = GaitScheduler()

    # 1) precompute
    t0 = time.perf_counter()
    precompute_trajectories(R, sched)
    t_precompute = time.perf_counter() - t0

    # 2) derivatives
    t0 = time.perf_counter()
    compute_derivatives(R)
    t_deriv = time.perf_counter() - t0

    if args.nmpc:
        from gait_sim.controllers.nmpc import (
            _CROCODDYL_AVAILABLE, solve_nmpc_receding, populate_simstate_from_nmpc,
        )
        if not _CROCODDYL_AVAILABLE:
            print("  crocoddyl 미설치 — NMPC 측정 불가")
            return
        print(f"\n[1] precompute_trajectories : {t_precompute*1e3:.0f} ms  "
              f"({t_precompute/config.N_FRAMES*1e6:.0f} μs/frame)")
        print(f"[2] compute_derivatives     : {t_deriv*1e3:.0f} ms")
        t0 = time.perf_counter()
        xs, us, forces, done, pin_m, pin_d = solve_nmpc_receding(sched)
        t_nmpc = time.perf_counter() - t0
        if not done:
            print("  NMPC 수렴 실패 — 통계 skip")
            return
        N_solve = max(1, len(us) // config.CFG.nmpc_rh_n_resolve)
        print(f"[3] solve_nmpc_receding    : {t_nmpc*1e3:.0f} ms  "
              f"({t_nmpc/N_solve*1e3:.0f} ms/solve, {N_solve} solves)")
        t0 = time.perf_counter()
        populate_simstate_from_nmpc(R, xs, us, forces, pin_m, pin_d,
                                       foot_z_home=float(-R.foot_hist[0, 0, 2]))
        t_pop = time.perf_counter() - t0
        print(f"[4] populate_simstate      : {t_pop*1e3:.0f} ms")
        total = t_precompute + t_deriv + t_nmpc + t_pop
    else:
        # 3) WBIC loop
        body_state, foot_z_home = init_wbic_state(R)
        t0 = time.perf_counter()
        diag = run_wbic_loop(R, sched, body_state, foot_z_home)
        t_wbic = time.perf_counter() - t0

        # 4) postprocess foot world
        t0 = time.perf_counter()
        postprocess_foot_world(R, sched)
        t_post = time.perf_counter() - t0

        print(f"\n[1] precompute_trajectories : {t_precompute*1e3:.0f} ms  "
              f"({t_precompute/config.N_FRAMES*1e6:.0f} μs/frame)")
        print(f"[2] compute_derivatives     : {t_deriv*1e3:.0f} ms")
        print(f"[3] run_wbic_loop           : {t_wbic*1e3:.0f} ms  "
              f"({t_wbic/config.N_FRAMES*1e6:.0f} μs/frame)")
        print(f"[4] postprocess_foot_world  : {t_post*1e3:.0f} ms")

        # Per-call QP distributions
        print("\n  Per-call QP latency (sample n=50):")
        d = measure_qp_calls(R, sched, body_state, foot_z_home, n_frames_sample=50)
        print(fmt_dist('MPC QP (N=10)',     d['mpc_qp'] * 1e6))
        print(fmt_dist('WBIC per-leg QP',   d['wbic_qp_leg'] * 1e6))
        print(fmt_dist('WBIC full body QP', d['wbic_qp_full'] * 1e6))

        total = t_precompute + t_deriv + t_wbic + t_post

    # Real-time factor
    sim_time = config.N_FRAMES * config.DT
    print("\n" + "═" * 60)
    print(f"  Total walltime : {total*1e3:.0f} ms")
    print(f"  Sim time       : {sim_time*1e3:.0f} ms ({config.N_FRAMES} frames @ DT={config.DT*1e3}ms)")
    print(f"  Real-time factor (sim/wall) : {sim_time/total:.3f}x  "
          f"({'real-time ✓' if sim_time/total >= 1.0 else 'slower than real-time'})")
    print("═" * 60)


if __name__ == '__main__':
    main()
