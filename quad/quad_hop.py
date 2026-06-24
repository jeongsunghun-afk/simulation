"""02_Leg offline 점프(제자리 호핑) 궤적최적화 — crocoddyl FDDP.

개발방향(2026-06-23): 조이스틱=online MPC+WBIC(quad_sim), 점프=offline trajopt+추종(이 파일).
WBIC stance PD는 목표 근처서 감속 → 이륙속도 미발생(takeoff 실험 검증) → 폭발적 이륙은 offline 궤적 필요.

스케줄: 정착(stance) → 푸시(stance, base_z↑+vz 타깃) → 비행(stance=[]=탄도) → 착지(stance) → 정착.
quad_nmpc 의 NMPCModel/_contact_multiple 재사용(02_Leg 14-DOF FullDynamics). 오프라인 완전수렴.

실행: BASE_Z0=0.30 /home/jsh/miniforge3/envs/proxddp/bin/python quad_hop.py --apex 0.04 [--view]
"""
import os
import time
import argparse
import numpy as np
import pinocchio as pin
import crocoddyl
import quad_nmpc as QN

G = 9.81


def _hop_action(M, stance, x_ref, dt, w_base_z=50.0, w_base_vz=1.0, w_ori=150.0,
                w_joint=0.2, w_jvel=0.1, w_fric=2.0):
    """호핑 전용 per-step 액션 — 위상별 가중 조절(push=vz강조, flight=자세홀드).
       build_action 과 동일 구조지만 base_z/base_vz 가중을 인자로 노출."""
    cm = QN._contact_multiple(M, stance)
    cost = crocoddyl.CostModelSum(M.state, M.nu)
    w_state = np.array([0.0, 0.0, w_base_z, w_ori, w_ori, 10.0] +   # base pos(xy자유) + ori
                       [w_joint] * M.nu +                            # 관절각
                       [1.0, 1.0, w_base_vz, 1.0, 1.0, 1.0] +        # base vel (vz 강조 가능)
                       [w_jvel] * M.nu)                              # 관절속도
    cost.addCost('xreg', crocoddyl.CostModelResidual(
        M.state, crocoddyl.ActivationModelWeightedQuad(w_state ** 2),
        crocoddyl.ResidualModelState(M.state, x_ref, M.nu)), 1.0)
    cost.addCost('ureg', crocoddyl.CostModelResidual(
        M.state, crocoddyl.ResidualModelControl(M.state, M.nu)), 1e-3)
    for L in stance:        # 마찰콘 + 접촉력 정규화
        fc = crocoddyl.FrictionCone(np.eye(3), QN.MU_FRIC, 4, False)
        fc_act = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(fc.lb, fc.ub))
        fc_res = crocoddyl.ResidualModelContactFrictionCone(
            M.state, M.foot_fid[L], fc, M.nu, True)
        cost.addCost('fc_' + L, crocoddyl.CostModelResidual(M.state, fc_act, fc_res), w_fric)
    tau_act = crocoddyl.ActivationModelQuadraticBarrier(   # 토크 한계 배리어
        crocoddyl.ActivationBounds(-M.tau_lim, M.tau_lim))
    cost.addCost('taulim', crocoddyl.CostModelResidual(
        M.state, tau_act, crocoddyl.ResidualModelControl(M.state, M.nu)), 1.0)
    diff = crocoddyl.DifferentialActionModelContactFwdDynamics(
        M.state, M.actuation, cm, cost, 0.0, True)
    return crocoddyl.IntegratedActionModelEuler(diff, dt)


def _foot_z(M, q):
    """현 자세 q 에서 4발 최소 z (지면 접근/이탈 감지용)."""
    pin.forwardKinematics(M.model, M.data, q)
    pin.updateFramePlacements(M.model, M.data)
    return min(M.data.oMf[M.foot_fid[L]].translation[2] for L in M.legs)


def _tilt(x):
    q = x[3:7]   # xyzw
    R = pin.Quaternion(q[3], q[0], q[1], q[2]).toRotationMatrix()
    return np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))


def solve_hop(apex=0.04, dt=0.01, n_settle=30, n_push=25, n_land=70,
              z_takeoff=0.52, view=False, save=None):
    """제자리 호핑 궤적. apex=신전높이 위 정점(m). 반환 (M, xs, us, schedule)."""
    M = QN.NMPCModel('ours')
    x0 = M.x0.copy()
    nq, nv = M.model.nq, M.model.nv
    z_crouch = float(x0[2])
    vz_tk = float(np.sqrt(2 * G * apex))               # 이륙속도(탄도)
    n_flight = max(4, int(round(2 * vz_tk / G / dt)))  # 비행步 ≈ 왕복 탄도시간
    z_apex = z_takeoff + apex
    print('[HOP] crouch z=%.3f  takeoff z=%.3f  apex z=%.3f  vz_tk=%.2fm/s' %
          (z_crouch, z_takeoff, z_apex, vz_tk))
    print('[HOP] 스케줄: settle=%d push=%d flight=%d land=%d (dt=%.3f, T=%.2fs)' %
          (n_settle, n_push, n_flight, n_land, dt, (n_settle + n_push + n_flight + n_land) * dt))

    # 위상별 x_ref
    x_push = x0.copy(); x_push[2] = z_apex; x_push[nq + 2] = vz_tk      # 푸시: 정점 위 + 상승속도
    x_air = x0.copy(); x_air[2] = z_apex; x_air[nq + 2] = 0.0           # 비행: 정점 유지(자세홀드)

    actions, sched = [], []
    for _ in range(n_settle):                                          # 정착(stance)
        actions.append(_hop_action(M, M.legs, x0, dt)); sched.append('settle')
    for _ in range(n_push):                                            # 푸시(stance, vz강조)
        actions.append(_hop_action(M, M.legs, x_push, dt, w_base_z=200.0, w_base_vz=50.0))
        sched.append('push')
    for _ in range(n_flight):                                          # 비행(접촉제거=탄도, 자세홀드)
        actions.append(_hop_action(M, [], x_air, dt, w_base_z=0.0, w_base_vz=0.0, w_joint=2.0))
        sched.append('flight')
    for _ in range(n_land):                                            # 착지+정착(stance)
        actions.append(_hop_action(M, M.legs, x0, dt, w_base_z=80.0))
        sched.append('land')
    w_T = np.ones(2 * nv); w_T[:3] = [0, 0, 10]
    terminal = QN.build_terminal(M, x0, w=w_T)

    problem = crocoddyl.ShootingProblem(M.x0, actions, terminal)
    solver = crocoddyl.SolverFDDP(problem)
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    N = len(actions)
    t0 = time.time()
    done = solver.solve([M.x0] * (N + 1), [np.zeros(M.nu)] * N, 500, False, 1e-9)
    xs = np.array(solver.xs); us = np.array(solver.us)
    print('[HOP] done=%s iters=%d time=%.0fms cost=%.3e' %
          (done, solver.iter, (time.time() - t0) * 1e3, solver.cost))

    # 진단: base_z 궤적, 실제 비행(발 지면이탈) 창, apex, tilt, 토크
    bz = xs[:, 2]; bvz = xs[:, nq + 2]
    footz = np.array([_foot_z(M, x[:nq]) for x in xs])
    air = footz > 0.01                                  # 발 1cm 이상 이탈 = 공중
    air_knots = int(air.sum())
    tilts = np.array([_tilt(x) for x in xs])
    print('[HOP] base_z: 시작%.3f → 최대%.3f → 최소%.3f → 끝%.3f' %
          (bz[0], bz.max(), bz.min(), bz[-1]))
    print('[HOP] 실제 공중구간=%d knots(%.0fms)  발 최대이탈=%.3fm  최대상승vz=%.2fm/s' %
          (air_knots, air_knots * dt * 1e3, footz.max(), bvz.max()))
    print('[HOP] tilt 최대=%.1f°  토크 RMS=%.1f max=%.1f Nm (한계%.0f)' %
          (tilts.max(), np.sqrt(np.mean(us ** 2)), np.abs(us).max(), M.tau_lim[0]))
    ok = done and air_knots >= 4 and footz.max() > 0.015 and tilts.max() < 25
    print('[HOP] 결과:', '✅ 호핑 궤적 생성(공중구간 발생)' if ok else
          '❌ 미흡(공중구간/수렴/자세 확인)')

    if save:
        np.savez(save, xs=xs, us=us, dt=dt, sched=np.array(sched, dtype=object),
                 nq=nq, nv=nv)
        print('[HOP] 저장:', save)
    if view:
        QN.replay_mujoco(M, xs, dt, loops=6)
    return M, xs, us, sched


def solve_jump(stand_z=0.52, crouch_z=0.30, apex=0.04, dt=0.01,
               n_load=35, n_push=25, n_land=35, n_recover=50, z_takeoff=0.52,
               save='jump_stand.npz', view=False):
    """GUI용 자체완결 점프: 서기(stand_z)→웅크림(crouch_z)→푸시→비행→착지→서기.
       MuJoCo 액추에이터 순서로 export(q*/dq*/tau*) → quad_sim 이 pin 없이 직접 재생."""
    import sys
    _a = sys.argv; sys.argv = [_a[0]]
    os.environ['BASE_Z0'] = str(stand_z)
    try:
        import quad_sim
        import mujoco
        quad_sim._ROBOT = 'ours_sphere'
        M = QN.NMPCModel('ours')
        qs = quad_sim.QuadSim()
    finally:
        sys.argv = _a
    pin2mj, mj_to_pin = QN._bridge(M, qs)
    nq = M.model.nq; nv = M.model.nv; nu = M.nu

    def state_at(z):                          # 주어진 base 높이의 crouch 자세 → pin 전상태
        qs.crouch_home(z); mujoco.mj_forward(qs.m, qs.d)
        return mj_to_pin(qs.d).copy()
    x_stand = state_at(stand_z); x_crouch = state_at(crouch_z)
    vz_tk = float(np.sqrt(2 * G * apex)); n_flight = max(4, int(round(2 * vz_tk / G / dt)))
    x_push = x_crouch.copy(); x_push[2] = z_takeoff + apex; x_push[nq + 2] = vz_tk
    x_air = x_crouch.copy(); x_air[2] = z_takeoff + apex; x_air[nq + 2] = 0.0
    print('[JUMP] stand z=%.3f → crouch z=%.3f → apex z=%.3f (vz_tk=%.2f, flight=%d)' %
          (stand_z, crouch_z, z_takeoff + apex, vz_tk, n_flight))

    actions, sched = [], []
    for _ in range(n_load):                                          # 웅크림(squat down)
        actions.append(_hop_action(M, M.legs, x_crouch, dt, w_base_z=120.0)); sched.append('load')
    for _ in range(n_push):                                          # 푸시(이륙)
        actions.append(_hop_action(M, M.legs, x_push, dt, w_base_z=200.0, w_base_vz=50.0))
        sched.append('push')
    for _ in range(n_flight):                                        # 비행(접촉제거=탄도)
        actions.append(_hop_action(M, [], x_air, dt, w_base_z=0.0, w_base_vz=0.0, w_joint=2.0))
        sched.append('flight')
    for _ in range(n_land):                                          # 착지 흡수→웅크림
        actions.append(_hop_action(M, M.legs, x_crouch, dt, w_base_z=80.0)); sched.append('land')
    for _ in range(n_recover):                                       # 다시 서기
        actions.append(_hop_action(M, M.legs, x_stand, dt, w_base_z=80.0)); sched.append('recover')
    w_T = np.ones(2 * nv); w_T[:3] = [0, 0, 10]
    terminal = QN.build_terminal(M, x_stand, w=w_T)
    problem = crocoddyl.ShootingProblem(x_stand, actions, terminal)
    solver = crocoddyl.SolverFDDP(problem)
    N = len(actions); t0 = time.time()
    done = solver.solve([x_stand] * (N + 1), [np.zeros(nu)] * N, 500, False, 1e-9)
    xs = np.array(solver.xs); us = np.array(solver.us)
    bz = xs[:, 2]; footz = np.array([_foot_z(M, x[:nq]) for x in xs])
    tilts = np.array([_tilt(x) for x in xs]); air = (footz > 0.02).sum()
    print('[JUMP] done=%s iters=%d time=%.0fms  base_z 시작%.3f→정점%.3f→끝%.3f' %
          (done, solver.iter, (time.time() - t0) * 1e3, bz[0], bz.max(), bz[-1]))
    print('[JUMP] 공중=%d knots(%.0fms) 발최대%.3f tilt최대%.1f° 토크max%.1f' %
          (air, air * dt * 1e3, footz.max(), tilts.max(), np.abs(us).max()))

    # MuJoCo 액추에이터 순서로 변환·저장
    mnu = qs.m.nu
    qref = np.zeros((len(xs), mnu)); dqref = np.zeros((len(xs), mnu))
    tauff = np.zeros((len(us), mnu))
    for k in range(len(xs)):
        qref[k][pin2mj] = xs[k][7:7 + nu]; dqref[k][pin2mj] = xs[k][nq + 6:nq + 6 + nu]
    for k in range(len(us)):
        tauff[k][pin2mj] = us[k]
    path = save if os.path.isabs(save) else os.path.join(QN._HERE, save)
    np.savez(path, q=qref, dq=dqref, tau=tauff, dt=dt, base_z=bz,
             sched=np.array(sched, dtype=object), stand_z=stand_z)
    print('[JUMP] 저장(MuJoCo순서):', path)
    ok = done and air >= 4 and footz.max() > 0.015 and tilts.max() < 25 \
        and abs(bz[-1] - stand_z) < 0.05
    print('[JUMP] 결과:', '✅ 서기→점프→서기 자체완결 궤적' if ok else '❌ 미흡')
    if view:
        QN.replay_mujoco(M, xs, dt, loops=4)
    return path


def track_hop(npz='/tmp/hop_4cm.npz', kp=120.0, kd=3.0, base_z0=0.30,
              view=False, pre=200):
    """②단계: offline 호핑 궤적을 MuJoCo 동역학에서 토크추종(피드포워드 u* + 관절 PD).
       ours_sphere(sphere발) + 이름기반 _bridge(14-DOF 비균일) + 실제 per-joint 토크한계.
       점프는 짧은 피드포워드 기동이라 연속보행과 달리 pin↔MuJoCo 불일치 누적 적음."""
    import sys
    _a = sys.argv; sys.argv = [_a[0]]
    os.environ['BASE_Z0'] = str(base_z0)
    try:
        import quad_sim
        import mujoco
        import mujoco.viewer as mjv
        quad_sim._ROBOT = 'ours_sphere'
        M = QN.NMPCModel('ours')
        q = quad_sim.QuadSim(); q.crouch_home(base_z0); mujoco.mj_forward(q.m, q.d)
    finally:
        sys.argv = _a
    m, d = q.m, q.d
    nq = M.model.nq; nu = M.nu
    pin2mj, mj_to_pin = QN._bridge(M, q)
    tau_lim = q._tau_peak.copy() if hasattr(q, '_tau_peak') else np.full(m.nu, 80.0)
    dat = np.load(npz, allow_pickle=True)
    xs, us, dt = dat['xs'], dat['us'], float(dat['dt'])
    N = len(us); sub = max(1, round(dt / m.opt.timestep))

    def foot_contacts():
        return sum(1 for i in range(d.ncon)
                   if d.contact[i].geom1 in q.foot_gid or d.contact[i].geom2 in q.foot_gid)

    def apply(k):
        qpl = xs[k][7:7 + nu]; vpl = xs[k][nq + 6:nq + 6 + nu]; uff = us[k]
        for _ in range(sub):
            xp = mj_to_pin(d)
            qj = xp[7:7 + nu]; vj = xp[nq + 6:nq + 6 + nu]
            tau = uff + kp * (qpl - qj) + kd * (vpl - vj)
            ctrl = np.zeros(m.nu); ctrl[pin2mj] = tau
            d.ctrl[:] = np.clip(ctrl, -tau_lim, tau_lim)
            mujoco.mj_step(m, d)

    # 사전 정착(crouch 0.30 PD홀드) — 시작 평형
    qhold = xs[0][7:7 + nu]
    for _ in range(pre):
        xp = mj_to_pin(d); qj = xp[7:7 + nu]; vj = xp[nq + 6:nq + 6 + nu]
        ctrl = np.zeros(m.nu); ctrl[pin2mj] = 120 * (qhold - qj) - 3 * vj
        d.ctrl[:] = np.clip(ctrl, -tau_lim, tau_lim); mujoco.mj_step(m, d)

    def foot_z_min():
        return min(d.geom_xpos[g][2] - q.foot_r[i] for i, g in enumerate(q.foot_gid))

    maxz = d.qpos[2]; airborne = 0; touched_after_air = False
    def run(v=None):
        nonlocal maxz, airborne, touched_after_air
        for k in range(N):
            if v is not None and not v.is_running():
                return
            apply(k)
            maxz = max(maxz, d.qpos[2])
            if foot_z_min() > 0.02:          # 발 2cm 이상 이탈 = 공중
                airborne += 1
            elif airborne > 0:
                touched_after_air = True
            if v is not None:
                v.sync(); time.sleep(dt)
    if view:
        print('호핑 추종 뷰어 — offline 궤적 + 토크/PD, 물리 적분.')
        with mjv.launch_passive(m, d) as v:
            run(v)
    else:
        run(None)
    x, y = d.qpos[4], d.qpos[5]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
    ok = airborne >= 4 and maxz > base_z0 + 0.10 and d.qpos[2] > 0.25 and tilt < 25
    print('[TRACK] 사전정착후 시작 base_z=%.3f' % base_z0)
    print('[TRACK] 최대 base_z=%.3f  공중구간=%d knots(%.0fms)  착지복귀=%s' %
          (maxz, airborne, airborne * dt * 1e3,
           '✅' if touched_after_air else '✗'))
    print('[TRACK] 최종 base_z=%.3f tilt=%.1f°' % (d.qpos[2], tilt))
    print('[TRACK] 결과:', '✅ 실제로 뛰고 착지' if ok else '❌ 미흡(공중/높이/착지 확인)')
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', action='store_true', help='offline 궤적 MuJoCo 추종')
    ap.add_argument('--jump', action='store_true', help='GUI용 서기→점프→서기 궤적 생성·export')
    ap.add_argument('--npz', default='/tmp/hop_4cm.npz')
    ap.add_argument('--kp', type=float, default=120.0)
    ap.add_argument('--kd', type=float, default=3.0)
    ap.add_argument('--apex', type=float, default=0.04)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--settle', type=int, default=30)
    ap.add_argument('--push', type=int, default=25)
    ap.add_argument('--land', type=int, default=70)
    ap.add_argument('--takeoff', type=float, default=0.52)
    ap.add_argument('--view', action='store_true')
    ap.add_argument('--save', default=None)
    a = ap.parse_args()
    if a.jump:
        solve_jump(apex=a.apex, view=a.view)
    elif a.track:
        track_hop(npz=a.npz, kp=a.kp, kd=a.kd, view=a.view)
    else:
        solve_hop(apex=a.apex, dt=a.dt, n_settle=a.settle, n_push=a.push, n_land=a.land,
                  z_takeoff=a.takeoff, view=a.view, save=a.save)
