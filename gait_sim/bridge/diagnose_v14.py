"""v14.5+: 두 가지 진단
  (1) NMPC vs MPC+WBIC joint rate 비교 — NMPC fast joint motion 가설 검증
  (2) WBIC FB QP 93% 실패 원인 — closed-loop 에서 Isaac state vs controller-expected state 차이
"""
import os
import sys
import numpy as np

GAIT_WBIC_NPZ = '/mnt/c/Users/jsh/simulation/replay_trot_v13.npz'
GAIT_NMPC_NPZ = '/mnt/c/Users/jsh/simulation/replay_trot_nmpc_v13.npz'
CLOSED_LOOP_NPZ = '/mnt/c/Users/jsh/closed_loop_trace.npz'


def report_joint_rate(label: str, npz_path: str) -> None:
    d = np.load(npz_path, allow_pickle=True)
    q   = d['q']         # (N, 4, 5)
    qd  = d['qdot']      # (N, 4, 5)
    tau = d['tau']
    dt  = float(d['dt'])
    N   = q.shape[0]

    qd_per_joint = np.abs(qd).reshape(-1, 5)    # (N*4, 5)
    qd_max  = qd_per_joint.max(axis=0)
    qd_rms  = np.sqrt(np.mean(qd_per_joint**2, axis=0))
    qd_p99  = np.percentile(qd_per_joint, 99, axis=0)

    # angular acceleration (numerical derivative)
    qdd = np.diff(qd, axis=0) / dt              # (N-1, 4, 5)
    qdd_per_joint = np.abs(qdd).reshape(-1, 5)
    qdd_max = qdd_per_joint.max(axis=0)
    qdd_rms = np.sqrt(np.mean(qdd_per_joint**2, axis=0))

    tau_per_joint = np.abs(tau).reshape(-1, 5)
    tau_max = tau_per_joint.max(axis=0)
    tau_rms = np.sqrt(np.mean(tau_per_joint**2, axis=0))

    print(f"\n=== {label} ({npz_path}) ===")
    print(f"  N={N}, dt={dt}s, duration={N*dt:.2f}s")
    print(f"  per-joint |qdot| (rad/s):")
    print(f"    max  : {qd_max.round(2).tolist()}")
    print(f"    p99  : {qd_p99.round(2).tolist()}")
    print(f"    RMS  : {qd_rms.round(2).tolist()}")
    print(f"  per-joint |qddot| (rad/s²):")
    print(f"    max  : {qdd_max.round(1).tolist()}")
    print(f"    RMS  : {qdd_rms.round(1).tolist()}")
    print(f"  per-joint |tau| (Nm):")
    print(f"    max  : {tau_max.round(1).tolist()}")
    print(f"    RMS  : {tau_rms.round(1).tolist()}")


def report_closed_loop_state_gap(npz_path: str) -> None:
    d = np.load(npz_path, allow_pickle=True)
    isaac_q     = d['q']             # (N, 20)  Isaac order
    isaac_p     = d['base_pos']      # (N, 3)
    isaac_v     = d['base_vel']      # (N, 3)
    isaac_w     = d['base_omega']    # (N, 3)
    ctrl_pos    = d['ctrl_body_pos'] # (N, 3) controller's body_state.pos AFTER step
                                      # (overridden each frame with Isaac state, then
                                      #  integrate_body_state writes back)
    ctrl_qhist  = d['ctrl_joint_hist']  # (N, 4, 5) controller's joint_hist target (precomputed)
    ctrl_tau    = d['ctrl_wbc_tau_cmd'] # (N, 4, 5) controller's tau output
    N = isaac_q.shape[0]
    dt = float(d['dt'])

    # base_pos: Isaac actual vs controller's post-integration
    pos_drift = ctrl_pos - isaac_p
    print(f"\n=== closed-loop state gap (Isaac vs controller post-integration) ===")
    print(f"  N={N}, dt={dt}s")
    print(f"  controller post-integ body_pos drift from Isaac:")
    print(f"    max |Δx|={np.max(np.abs(pos_drift[:,0])):.4f} m")
    print(f"    max |Δy|={np.max(np.abs(pos_drift[:,1])):.4f} m")
    print(f"    max |Δz|={np.max(np.abs(pos_drift[:,2])):.4f} m")

    # joint target vs actual (Isaac is in isaac order, need to map)
    # Quick approach: convert isaac to gait_sim (4,5) using gait_isaac_bridge logic
    # We can just compute the per-frame norm
    isaac_q_g = isaac_q.reshape(N, 5, 4).transpose(0, 2, 1)  # rough — actually wrong order
    # The correct mapping requires the permutation; instead just use Q_HOME for sanity
    # Skip mapping for now; show only base divergence

    # Tau magnitude trend
    tau_norm = np.linalg.norm(ctrl_tau.reshape(N, -1), axis=1)
    print(f"  controller tau_cmd ||.||_2 trajectory:")
    print(f"    mean={tau_norm.mean():.2f}  max={tau_norm.max():.2f}  std={tau_norm.std():.2f}")
    # bins of 100 steps
    for k in range(0, N, 100):
        end = min(k+100, N)
        seg = tau_norm[k:end]
        print(f"    step {k:4d}~{end:4d}: tau_norm mean={seg.mean():6.2f} max={seg.max():6.2f}")


if __name__ == '__main__':
    print("# v14.5+ diagnostic\n")
    if os.path.exists(GAIT_WBIC_NPZ):
        report_joint_rate("MPC+WBIC", GAIT_WBIC_NPZ)
    if os.path.exists(GAIT_NMPC_NPZ):
        report_joint_rate("NMPC", GAIT_NMPC_NPZ)
    if os.path.exists(CLOSED_LOOP_NPZ):
        report_closed_loop_state_gap(CLOSED_LOOP_NPZ)
