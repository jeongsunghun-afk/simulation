"""v14.5.6: gait_sim ↔ MuJoCo closed-loop bridge

MuJoCo joint order:
   leg_FR_j1..j5, leg_FL_j1..j5, leg_HR_j1..j5, leg_HL_j5
== gait_sim LEG_NAMES=['FR','FL','HR','HL'] × j1..j5 (leg-major)
== R.joint_hist (4, 5).reshape(20)

→ joint mapping 단순 reshape (Isaac Sim 처럼 permutation 불필요)

Usage:
    python3 -m gait_sim.bridge.mujoco_runner --gait trot --headless
    python3 -m gait_sim.bridge.mujoco_runner --gait trot --viewer
"""
from __future__ import annotations
import argparse
import math
import os
import signal
import sys
import time
import numpy as np

import mujoco

from gait_sim.bridge.controller_step import GaitSimControllerStep

MJCF_PATH_DEFAULT = '/home/jsh/simulation/quadruped_v13_mujoco.mjcf'


def quat_wxyz_to_R(q):
    """MuJoCo quat (w,x,y,z) → 3x3 rotation matrix"""
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),    2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),    2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),  1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)


def R_to_quat_wxyz(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return np.array([0.25*s, (R[2,1]-R[1,2])/s,
                         (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s,
                         (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s,
                         0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s,
                         (R[1,2]+R[2,1])/s, 0.25*s])


def run_closed_loop(mjcf_path: str = MJCF_PATH_DEFAULT,
                     gait_type: str = 'trot',
                     n_frames: int | None = None,
                     mode: str = 'effort',
                     use_viewer: bool = False,
                     out_npz: str | None = None):
    """gait_sim controller closed-loop with MuJoCo physics.

    Args:
        mjcf_path: MJCF 파일 경로
        gait_type: 'trot', 'walk' 등
        n_frames: 시뮬레이션 frame 수 (None → controller's full horizon)
        mode: 'effort' (tau 직접) / 'position+effort' (hybrid)
        use_viewer: True 면 MuJoCo GUI viewer
        out_npz: 결과 trace 저장 path (None 이면 저장 안 함)
    """
    print(f"=== INIT controller (gait={gait_type}) ===", flush=True)
    ctrl = GaitSimControllerStep(gait_type=gait_type)
    dt    = ctrl.dt
    N_max = ctrl.n_frames
    N     = n_frames if n_frames else N_max
    print(f"=== controller ready: dt={dt}s N={N}/{N_max} ===", flush=True)

    print(f"=== LOAD MJCF: {mjcf_path} ===", flush=True)
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data  = mujoco.MjData(model)
    assert model.nu == 20, f"actuator count mismatch: {model.nu}"
    assert model.nq == 27, f"nq mismatch: {model.nq}"

    # MuJoCo dt 를 gait_sim 의 dt 와 매칭
    model.opt.timestep = dt
    print(f"=== mujoco dt set to {model.opt.timestep} ===", flush=True)

    # 초기 자세: controller 의 body_state 사용
    init_pos = ctrl.body_state['pos'].copy()
    init_R   = ctrl.body_state['R'].copy()
    init_v   = ctrl.body_state['v'].copy()
    init_w   = ctrl.body_state['omega'].copy()
    init_q   = ctrl.R.joint_hist[0]   # (4, 5)

    data.qpos[0:3] = init_pos
    data.qpos[3:7] = R_to_quat_wxyz(init_R)
    data.qpos[7:]  = init_q.reshape(20)   # leg-major, MuJoCo 와 같음
    data.qvel[0:3] = init_v
    data.qvel[3:6] = init_w
    data.qvel[6:]  = 0.0
    mujoco.mj_forward(model, data)  # state propagate (no integration)
    print(f"=== init pose: base_z={data.qpos[2]:.3f}, q_max={np.max(np.abs(data.qpos[7:])):.3f} ===",
          flush=True)

    # 기록 buffer
    rec = dict(
        q       = np.zeros((N, 20)),
        qdot    = np.zeros((N, 20)),
        base_p  = np.zeros((N, 3)),
        base_q  = np.zeros((N, 4)),
        base_v  = np.zeros((N, 3)),
        base_w  = np.zeros((N, 3)),
        tau     = np.zeros((N, 20)),
    )

    # viewer (옵션)
    viewer = None
    if use_viewer:
        try:
            from mujoco import viewer as _mj_viewer   # local alias — outer 'mujoco' scope 보존
            viewer = _mj_viewer.launch_passive(model, data)
            print("=== viewer launched ===", flush=True)
        except Exception as e:
            print(f"viewer 실패 (headless 로 진행): {e}", flush=True)

    # ── Ctrl+C handler ────────────────────────────────────────
    # 1st Ctrl+C → graceful: 현재 step 끝나고 cleanup + trace save
    # 2nd Ctrl+C → 즉시 os._exit (viewer hang 회피)
    _abort = {'n': 0, 'last_step': 0}
    def _sigint_handler(signum, frame):
        _abort['n'] += 1
        if _abort['n'] >= 2:
            print("\n!!! 2nd Ctrl+C — FORCE EXIT (os._exit)", flush=True)
            os._exit(130)
        print(f"\n!! Ctrl+C — finishing current step then cleanup (press again to FORCE)",
              flush=True)
    old_handler = signal.signal(signal.SIGINT, _sigint_handler)

    print(f"=== START CLOSED-LOOP ({mode}, {N} steps) ===", flush=True)
    t0 = time.monotonic()
    k = 0
    N_done = 0   # 실제 완료된 step 수 (early abort 시 < N)
    try:
        for k in range(N):
            # graceful abort check (Ctrl+C → handler 가 _abort['n']=1)
            if _abort['n'] > 0:
                print(f"!! aborted at step {k} — saving partial trace", flush=True)
                break

            # 1. MuJoCo state 읽기
            q_mj    = data.qpos[7:].copy()
            qdot_mj = data.qvel[6:].copy()
            body_p  = data.qpos[0:3].copy()
            body_q  = data.qpos[3:7].copy()
            body_v  = data.qvel[0:3].copy()
            body_w  = data.qvel[3:6].copy()
            body_R  = quat_wxyz_to_R(body_q)

            # 2. 변환 (단순 reshape)
            q_g    = q_mj.reshape(4, 5)
            qdot_g = qdot_mj.reshape(4, 5)

            # 3. controller 호출
            ctrl.step_closed_loop(k, q_g, qdot_g, body_p, body_R, body_v, body_w)
            tau_g  = ctrl.get_tau_cmd(k)
            tau_mj = tau_g.reshape(20)

            # 4. MuJoCo 에 torque 적용
            if mode == 'effort':
                data.ctrl[:] = tau_mj

            mujoco.mj_step(model, data)
            if viewer is not None:
                try:
                    viewer.sync()
                except Exception:
                    pass

            # 5. 기록
            rec['q'][k]      = q_mj
            rec['qdot'][k]   = qdot_mj
            rec['base_p'][k] = body_p
            rec['base_q'][k] = body_q
            rec['base_v'][k] = body_v
            rec['base_w'][k] = body_w
            rec['tau'][k]    = tau_mj
            N_done = k + 1

            if k % 100 == 0:
                print(f"  step {k:4d}: base_z={body_p[2]:+.3f}  "
                      f"q_max={np.max(np.abs(q_mj)):.3f}  "
                      f"tau_RMS={np.sqrt(np.mean(tau_mj**2)):.2f}", flush=True)
    except KeyboardInterrupt:
        # signal handler 가 못 잡은 경우 fallback
        print(f"!! KeyboardInterrupt at step {k} (handler 미작동)", flush=True)
    finally:
        # 1) signal handler 복구
        signal.signal(signal.SIGINT, old_handler)

        # 2) viewer cleanup
        if viewer is not None:
            try:
                viewer.close()
                print("=== viewer closed ===", flush=True)
            except Exception as e:
                print(f"viewer.close() 실패: {e}", flush=True)

        # 3) partial trace 라도 저장
        if out_npz and N_done > 0:
            partial = {k_: v[:N_done] for k_, v in rec.items()}
            np.savez_compressed(out_npz, dt=dt, N=N_done, gait=gait_type, mode=mode,
                                  aborted=(_abort['n'] > 0), **partial)
            print(f"=== TRACE SAVED ({N_done}/{N} steps): {out_npz} "
                  f"({os.path.getsize(out_npz)/1024:.1f} KiB) ===", flush=True)

        t1 = time.monotonic()
        print(f"=== DONE in {t1-t0:.2f}s (wall) for {N_done*dt:.2f}s (sim, {N_done}/{N} steps) ===",
              flush=True)

        # 4) summary
        if N_done > 0:
            print()
            print("=== FINAL STATE ===")
            print(f"  base pos : {rec['base_p'][N_done-1].round(4).tolist()}")
            print(f"  base vel : {rec['base_v'][N_done-1].round(3).tolist()}")
            print(f"  fail cnt : {ctrl.fail_counts}")

    return rec


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gait',     default='trot')
    p.add_argument('--mode',     default='effort')
    p.add_argument('--mjcf',     default=MJCF_PATH_DEFAULT)
    p.add_argument('--n',        type=int, default=0,
                    help='frame 수 (0=full horizon)')
    p.add_argument('--viewer',   action='store_true')
    p.add_argument('--out',      default='/tmp/mujoco_trace.npz')
    args = p.parse_args()

    run_closed_loop(
        mjcf_path  = args.mjcf,
        gait_type  = args.gait,
        n_frames   = args.n if args.n > 0 else None,
        mode       = args.mode,
        use_viewer = args.viewer,
        out_npz    = args.out,
    )
