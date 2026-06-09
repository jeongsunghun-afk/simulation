"""biped_wrapper.mjcf 를 MuJoCo passive viewer 로 실행.

URDF q=0 (모든 8 관절 0) = home/standing 자세. PD 컨트롤러로 그 자세를 유지.
사용:
    python3 run_biped_viewer.py            # GUI viewer
    python3 run_biped_viewer.py --headless # 창 없이 N초 시뮬 (검증용)
"""
from __future__ import annotations
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

MJCF = 'biped_wrapper.mjcf'

# 8 actuated joints, home target = 0 (standing)
Q_HOME = np.zeros(8)
KP = np.array([120, 120, 120, 60, 120, 120, 120, 60.0])   # hip/thigh/calf/foot
KD = np.array([4, 4, 4, 2, 4, 4, 4, 2.0])


def pd_control(m, d):
    q = d.qpos[7:7 + 8]     # skip freejoint (7)
    dq = d.qvel[6:6 + 8]    # skip freejoint (6)
    tau = KP * (Q_HOME - q) - KD * dq
    d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]) \
        if m.actuator_ctrllimited.any() else tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--seconds', type=float, default=5.0)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    # base 를 바닥 가까이 (홈 자세에서 발이 바닥에 닿도록)
    d.qpos[2] = 0.45
    mujoco.mj_forward(m, d)

    if args.headless:
        n = int(args.seconds / m.opt.timestep)
        for _ in range(n):
            pd_control(m, d)
            mujoco.mj_step(m, d)
        print('headless %.1fs done: base_z=%.3f joints_rms=%.4f rad'
              % (args.seconds, d.qpos[2], np.sqrt(np.mean((d.qpos[7:15]) ** 2))))
        return

    with mujoco.viewer.launch_passive(m, d) as viewer:
        print('viewer open — close window to exit')
        while viewer.is_running():
            t0 = time.time()
            pd_control(m, d)
            mujoco.mj_step(m, d)
            viewer.sync()
            dt = m.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == '__main__':
    main()
