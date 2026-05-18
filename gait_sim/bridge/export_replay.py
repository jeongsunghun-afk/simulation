"""v14.3-b: Export gait_sim trajectory to .npz for Isaac Sim open-loop replay.

gait_sim core 무수정 — runner.run_simulation() 호출 → SimState 핵심 필드만 .npz 로 직렬화.

Saved fields:
  meta: dt, gait_type, n_frames, V, T, mass, joint_home (4,5)
  time   (N,)
  q      (N,4,5)  commanded joint angles
  qdot   (N,4,5)  commanded joint vel
  tau    (N,4,5)  WBC final torque command (wbc_tau_cmd)
  body_pos   (N,3)
  body_R     (N,3,3)
  body_vel   (N,3)
  body_omega (N,3)
  foot_world (N,4,3) actual foot world position
  phase  (N,4)    per-leg gait phase [0,1)
  lam    (N,4,3)  WBIC ground reaction force used
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np


def export(out_path: str, gait_type: str | None = None,
            use_nmpc: bool = False) -> dict:
    from gait_sim.runner import run_simulation
    from gait_sim.model import Q_HOME_PER_LEG, BODY_MASS, TOTAL_MASS
    from gait_sim.config import DT

    R, meta = run_simulation(gait_type=gait_type, use_nmpc=use_nmpc)
    N = R.n_frames
    dt = R.dt
    t = np.arange(N) * dt

    q_home = np.array(Q_HOME_PER_LEG, dtype=np.float64)   # (4,5)

    data = dict(
        dt=np.float64(dt),
        gait_type=str(meta['gait_type']),
        V=np.float64(meta['V']),
        T=np.float64(meta['T']),
        D=np.float64(meta['D']),
        n_frames=np.int64(N),
        total_mass=np.float64(TOTAL_MASS),
        body_mass=np.float64(BODY_MASS),
        q_home=q_home,
        time=t,
        q=R.joint_hist.astype(np.float64),                # (N,4,5)
        qdot=R.joint_vel_hist.astype(np.float64),         # (N,4,5)
        tau=R.wbc_tau_cmd.astype(np.float64),             # (N,4,5)
        body_pos=R.body_pos_hist.astype(np.float64),      # (N,3)
        body_R=R.body_R_hist.astype(np.float64),          # (N,3,3)
        body_vel=R.body_v_hist.astype(np.float64),        # (N,3)
        body_omega=R.body_omega_hist.astype(np.float64),  # (N,3)
        foot_world=R.foot_actual_world_hist.astype(np.float64),  # (N,4,3)
        phase=R.phase_hist.astype(np.float64),            # (N,4)
        lam=R.wbic_lam_used.astype(np.float64),           # (N,4,3)
        diverged=np.bool_(R.diverged),
    )
    np.savez_compressed(out_path, **data)
    return dict(out_path=out_path, N=N, dt=dt,
                 gait_type=meta['gait_type'],
                 diverged=R.diverged)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--out',  default='/mnt/c/Users/jsh/simulation/replay_trot_v13.npz')
    p.add_argument('--gait', default='trot')
    p.add_argument('--nmpc', action='store_true')
    a = p.parse_args()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    info = export(a.out, gait_type=a.gait, use_nmpc=a.nmpc)
    print("\n========== replay exported ==========")
    for k, v in info.items():
        print(f"  {k}: {v}")
    size = os.path.getsize(a.out)
    print(f"  file size: {size/1024:.1f} KiB")
