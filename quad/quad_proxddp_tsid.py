"""구조차용 proxDDP — simple-mpc의 2계층(NMPC 참조 + TSID-ID 1kHz) 구조를 우리에 이식.

기존 quad_proxddp.py(우리 전신 NMPC walk_loop) 는 그대로 두고, **DDP게인 추종을
TSID 역동역학 QP(1kHz)로 교체**한다. simple-mpc 의 KinodynamicsID 역할을 raw TSID 로 구현.

Phase 2a: 우리 수렴 사이클 NMPC + TSID-ID (RTI 아님).  Phase 2b(추후): NMPC를 RTI로.
검증 단계: ① tsid_stand(중력보상+자세+4접촉 QP) ② walk_loop_tsid(사이클 참조 추종).
실행: /home/jsh/miniforge3/envs/proxddp/bin/python quad_proxddp_tsid.py --test stand --robot go2
"""
import os
import numpy as np
import pinocchio as pin
import tsid
import mujoco

import quad_proxddp as Q
import quad_nmpc as QN

TAU_LIM = 80.0


class TSIDLayer:
    """우리 pinocchio 모델 위의 TSID 역동역학 QP. 발=ContactPoint, 자세·베이스 task.
       set_reference(q,v,a_ref, stance) → solve(q_meas,v_meas) → tau."""

    def __init__(self, P, q0, dt, mu=0.6,
                 w_post=float(os.environ.get('W_POST', '2.0')),       # best found(~5s)
                 w_base=float(os.environ.get('W_BASE', '10.0')),
                 w_force=float(os.environ.get('W_FORCE', '1.0')),
                 kp_post=float(os.environ.get('KP_POST', '30.0')),
                 kp_base=float(os.environ.get('KP_BASE', '50.0')),
                 kp_contact=float(os.environ.get('KP_CONTACT', '50.0'))):
        self.P = P
        self.rw = tsid.RobotWrapper(P.model, False)
        self.na = self.rw.na                       # 구동 관절수
        self.formulation = tsid.InverseDynamicsFormulationAccForce("tsid", self.rw, False)
        v0 = np.zeros(P.nv)
        self.formulation.computeProblemData(0.0, q0, v0)
        data = self.formulation.data()

        # ── 4발 점접촉 ──
        self.contacts = {}
        normal = np.array([0.0, 0.0, 1.0])
        for L in P.legs:
            fname = P.model.frames[P.M.foot_fid[L]].name      # "FL_foot" 등
            c = tsid.ContactPoint("contact_" + L, self.rw, fname, normal, mu, 5.0, 200.0)
            c.setKp(kp_contact * np.ones(3))
            c.setKd(2.0 * np.sqrt(kp_contact) * np.ones(3))
            c.useLocalFrame(False)
            ref = self.rw.framePosition(data, P.M.foot_fid[L])
            c.setReference(ref)
            self.formulation.addRigidContact(c, w_force)
            self.contacts[L] = c
        self.stance = set(P.legs)

        # ── 자세(관절) task ──
        self.postureTask = tsid.TaskJointPosture("posture", self.rw)
        self.postureTask.setKp(kp_post * np.ones(self.na))
        self.postureTask.setKd(2.0 * np.sqrt(kp_post) * np.ones(self.na))
        self.formulation.addMotionTask(self.postureTask, w_post, 1, 0.0)
        self.samplePosture = tsid.TrajectorySample(self.na)

        # ── 베이스 SE3 task (몸 자세·높이) ──
        base_fname = P.model.frames[1].name if P.model.frames[1].type == pin.FrameType.JOINT else "root_joint"
        # base frame = 첫 바디 프레임. pinocchio base body 이름 탐색
        self.base_fid = 1
        self.baseTask = tsid.TaskSE3Equality("base", self.rw, P.model.frames[self._base_body()].name)
        self.baseTask.setKp(kp_base * np.ones(6))
        self.baseTask.setKd(2.0 * np.sqrt(kp_base) * np.ones(6))
        mask = np.array([0, 0, 1, 1, 1, 1.0])      # z, roll, pitch, yaw (x,y 자유)
        self.baseTask.setMask(mask)
        self.formulation.addMotionTask(self.baseTask, w_base, 1, 0.0)
        self.sampleBase = tsid.TrajectorySample(12, 6)

        # ── CoM task (균형 핵심 — base 가라앉음 방지) ──
        self.comTask = tsid.TaskComEquality("com", self.rw)
        kp_com = float(os.environ.get('KP_COM', '100.0'))
        self.comTask.setKp(kp_com * np.ones(3))
        self.comTask.setKd(2.0 * np.sqrt(kp_com) * np.ones(3))
        self.w_com = float(os.environ.get('W_COM', '0.0'))   # 0=비활성(naive CoM task가 오히려 악화)
        if self.w_com > 0:
            self.formulation.addMotionTask(self.comTask, self.w_com, 1, 0.0)
        self.sampleCom = tsid.TrajectorySample(3)

        # ── 토크 한계 ──
        self.actBounds = tsid.TaskActuationBounds("act_bounds", self.rw)
        self.actBounds.setBounds(-TAU_LIM * np.ones(self.na), TAU_LIM * np.ones(self.na))
        self.formulation.addActuationTask(self.actBounds, 1.0, 0, 0.0)

        self.mass = sum(P.model.inertias[i].mass for i in range(1, P.model.njoints))
        self._comdata = P.model.createData()
        self.solver = tsid.SolverHQuadProgFast("qp")
        self.solver.resize(self.formulation.nVar, self.formulation.nEq, self.formulation.nIn)
        self.dt = dt
        self.q0 = q0.copy()

    def _base_body(self):
        """floating base 직후 바디 프레임 id."""
        for i, f in enumerate(self.P.model.frames):
            if f.type == pin.FrameType.BODY and f.parentJoint == 1:
                return i
        return 1

    def set_contacts(self, stance):
        """gait 접촉상태 갱신: stance 발만 접촉 활성."""
        for L in self.P.legs:
            want = L in stance
            have = L in self.stance
            if want and not have:
                ref = self.rw.framePosition(self.formulation.data(), self.P.M.foot_fid[L])
                self.contacts[L].setReference(ref)
                self.formulation.addRigidContact(self.contacts[L], 1e-3)
            elif have and not want:
                self.formulation.removeRigidContact(self.contacts[L].name, 0.0)
        self.stance = set(stance)

    def set_force_ref(self, forces):
        """NMPC 동적 접촉력 → TSID force ref. FZ_ONLY=1 이면 수직만(수평 NMPC력이 MuJoCo로 잘 안옮겨감)."""
        fz_only = int(os.environ.get('FZ_ONLY', '1'))
        for L, f in forces.items():
            if L in self.stance:
                ff = np.array([0.0, 0.0, float(f[2])]) if fz_only else np.asarray(f, float)
                self.contacts[L].setForceReference(ff)

    def set_posture_ref(self, q_ref, v_ref=None, a_ref=None):
        self.samplePosture.value(q_ref[7:7 + self.na])
        if v_ref is not None:
            self.samplePosture.derivative(v_ref[6:6 + self.na])    # 관절속도 피드포워드
        if a_ref is not None:
            self.samplePosture.second_derivative(a_ref[6:6 + self.na])  # 관절가속 피드포워드
        self.postureTask.setReference(self.samplePosture)

    def set_com_ref(self, q_ref, v_ref=None):
        if self.w_com <= 0:
            return
        com = pin.centerOfMass(self.P.model, self._comdata, q_ref[:self.P.nq])
        self.sampleCom.value(np.asarray(com))
        self.comTask.setReference(self.sampleCom)

    def set_base_ref(self, q_ref, v_ref=None):
        oMi = pin.XYZQUATToSE3(q_ref[:7])
        data = np.concatenate([oMi.translation, oMi.rotation.flatten()])
        self.sampleBase.value(data)
        if v_ref is not None:
            self.sampleBase.derivative(v_ref[:6])      # 베이스 트위스트 피드포워드
        self.baseTask.setReference(self.sampleBase)

    def solve(self, t, q, v):
        HQPData = self.formulation.computeProblemData(t, q, v)
        sol = self.solver.solve(HQPData)
        if sol.status != 0:
            return None
        tau = self.formulation.getActuatorForces(sol)
        return tau


def _mujoco_setup(exec_robot):
    import sys
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        import quad_sim; quad_sim._ROBOT = exec_robot; q = quad_sim.QuadSim(); q.crouch_home()
    finally:
        sys.argv = _a
    # ★ 접촉 강체화(simple-mpc 방식) — TSID 강체접촉 가정과 MuJoCo soft접촉 불일치 해소(발 침투→몸 가라앉음 방지)
    if os.environ.get('CONE'):
        q.m.opt.cone = int(os.environ['CONE'])              # 0=pyramidal
    if os.environ.get('STIFF'):
        q.m.geom_solref[:, 0] = float(os.environ['STIFF'])  # solref timeconst↓ = 강체
        q.m.geom_solref[:, 1] = 1.0
    return q


def tsid_stand(robot='go2', sim_T=5.0, view=False, exec_robot=None):
    """검증①: TSID standing — 중력보상+자세+4접촉 QP 로 제자리 유지(MuJoCo). NMPC 없이 ID만."""
    import time
    exec_robot = exec_robot or robot
    P = Q.ProxModel(robot)
    q0 = P.x0[:P.nq].copy()
    dt = 0.001
    tl = TSIDLayer(P, q0, dt)
    q = _mujoco_setup(exec_robot)
    m, d = q.m, q.d
    pin2mj, mj_to_pin = QN._bridge(P.M, q)
    m.opt.timestep = dt
    tl.set_posture_ref(q0); tl.set_base_ref(q0)
    nsteps = int(sim_T / dt); t0 = time.time(); tmax = 0.0
    viewer = None
    if view:
        import mujoco.viewer as mjv; viewer = mjv.launch_passive(m, d)
    for it in range(nsteps):
        x = mj_to_pin(d); qp = x[:P.nq]; vp = x[P.nq:]
        tau = tl.solve(it * dt, qp, vp)
        if tau is None:
            print('  TSID QP 실패 @%.2fs' % (it * dt)); break
        umj = np.zeros(P.nu); umj[pin2mj] = tau[:P.nu]
        d.ctrl[:] = np.clip(umj, -TAU_LIM, TAU_LIM); mujoco.mj_step(m, d)
        if viewer:
            viewer.sync(); time.sleep(dt)
        xx, yy = d.qpos[4], d.qpos[5]
        tmax = max(tmax, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))))
        if it % 500 == 0:
            print('  t=%.2f base_z=%.3f tilt=%.1f' % (it * dt, d.qpos[2],
                  np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1)))), flush=True)
        if d.qpos[2] < 0.15:
            print('  TSID standing ❌ 전복 @%.2fs' % (it * dt));
            if viewer: viewer.close()
            return
    if viewer: viewer.close()
    print('  TSID standing ✅ %.1fs base_z=%.3f tilt_max=%.0f° (벽시계%.0fs)' % (
        sim_T, d.qpos[2], tmax, time.time() - t0))


def _solve_cycle(P, V, T, sf, step_h, dt, cyc, settle_steps, n_warm, maxiter=100):
    """우리 전신 NMPC(ProxDDP) 정상사이클 풀이 — walk_loop 방식. 반환 (cyc_xs, cyc_stance)."""
    import aligator
    N = settle_steps + n_warm * cyc
    x_ref_fwd = P.x0.copy(); x_ref_fwd[P.nq + 0] = V
    stages = []
    for k in range(N):
        tg = (k - settle_steps) * dt
        if tg < 0:
            stages.append(P.stage(P.legs, P.x0, dt))
        else:
            st, sw = QN._trot_schedule(tg, T, sf, step_h, P.foot_home, P.legs, P.diag, V, march=V * tg)
            stages.append(P.stage(st, x_ref_fwd, dt, swing=sw))
    prob = aligator.TrajOptProblem(P.x0, stages, P.terminal_cost(x_ref_fwd))
    sol = aligator.SolverProxDDP(1e-5, 1e-8, max_iters=maxiter)
    sol.setup(prob); sol.run(prob, [P.x0] * (N + 1), [np.zeros(P.nu)] * N)
    xs = [np.array(x) for x in sol.results.xs]
    us = [np.array(u) for u in sol.results.us]
    b = settle_steps + (n_warm - 1) * cyc
    cyc_xs = [xs[b + i].copy() for i in range(cyc)]
    cyc_us = [us[b + i].copy() for i in range(cyc)]
    cyc_stance = []                                    # 사이클 스텝별 stance 발집합
    for i in range(cyc):
        st, _ = QN._trot_schedule(i * dt, T, sf, step_h, P.foot_home, P.legs, P.diag, V, march=0)
        cyc_stance.append(st)
    cyc_acc = []                                       # 가속 피드포워드 a[i]=(v[i+1]-v[i])/dt (주기적)
    for i in range(cyc):
        v_now = cyc_xs[i][P.nq:]; v_nxt = cyc_xs[(i + 1) % cyc][P.nq:]
        cyc_acc.append((v_nxt - v_now) / dt)
    cyc_forces = []                                    # ★ NMPC 동적 접촉력: lambda_c at (x,u), stance발별 3D
    for i in range(cyc):
        stance = cyc_stance[i]
        if not stance:
            cyc_forces.append({}); continue
        dyn, _ = P.dynamics(stance, dt)
        dd = dyn.createData()
        dyn.forward(cyc_xs[i], cyc_us[i], dd)
        lam = np.asarray(dd.continuous_data.pin_data.lambda_c)   # 3*len(stance), world-aligned
        cyc_forces.append({L: lam[3 * k:3 * k + 3].copy() for k, L in enumerate(stance)})
    return cyc_xs, cyc_stance, cyc_acc, cyc_forces, sol.results.conv


def walk_loop_tsid(robot='go2', V=0.3, T=0.5, sf=0.5, step_h=0.06, settle_steps=15, n_warm=6,
                   dt_nmpc=0.02, total_T=10.0, view=False, exec_robot=None):
    """★ 구조차용: 우리 전신 NMPC 사이클(q,v 참조 + 접촉스케줄) → TSID-ID 1kHz 추종(MuJoCo).
       DDP게인 대신 TSID QP가 토크 생성. simple-mpc 2계층 구조를 우리 NMPC 위에 이식."""
    import time
    exec_robot = exec_robot or robot
    P = Q.ProxModel(robot)
    cyc = round(T / dt_nmpc)
    cyc_xs, cyc_stance, cyc_acc, cyc_forces, conv = _solve_cycle(
        P, V, T, sf, step_h, dt_nmpc, cyc, settle_steps, n_warm)
    print('NMPC 사이클 conv=%s (cyc=%d) → TSID-ID 1kHz 추종' % (conv, cyc))
    q = _mujoco_setup(exec_robot); m, d = q.m, q.d
    pin2mj, mj_to_pin = QN._bridge(P.M, q)
    dt_sim = 0.001; m.opt.timestep = dt_sim
    sub = round(dt_nmpc / dt_sim)
    tl = TSIDLayer(P, P.x0[:P.nq], dt_sim)
    nouter = int(total_T / dt_nmpc); t0 = time.time(); tmax = 0.0
    viewer = None
    if view:
        import mujoco.viewer as mjv; viewer = mjv.launch_passive(m, d)
    for outer in range(nouter):
        i = outer % cyc
        q_ref = cyc_xs[i][:P.nq]; v_ref = cyc_xs[i][P.nq:]
        tl.set_contacts(cyc_stance[i])
        tl.set_force_ref(cyc_forces[i])                # ★ NMPC 동적 접촉력 reference
        tl.set_posture_ref(q_ref, v_ref, cyc_acc[i]); tl.set_base_ref(q_ref, v_ref); tl.set_com_ref(q_ref)
        for s in range(sub):
            x = mj_to_pin(d)
            tau = tl.solve((outer * sub + s) * dt_sim, x[:P.nq], x[P.nq:])
            if tau is None:
                tau = np.zeros(tl.na)
            umj = np.zeros(P.nu); umj[pin2mj] = tau[:P.nu]
            d.ctrl[:] = np.clip(umj, -TAU_LIM, TAU_LIM); mujoco.mj_step(m, d)
            if viewer:
                viewer.sync(); time.sleep(dt_sim)
        xx, yy = d.qpos[4], d.qpos[5]
        tmax = max(tmax, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))))
        if outer % 25 == 0:
            print('  t=%.2f base_z=%.3f x=%.2f y=%+.3f tilt=%.1f' % (
                outer * dt_nmpc, d.qpos[2], d.qpos[0], d.qpos[1],
                np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1)))), flush=True)
        if d.qpos[2] < 0.15:
            print('  TSID-ID 보행 ❌ 전복 @%.2fs (x=%.2f)' % (outer * dt_nmpc, d.qpos[0]))
            if viewer: viewer.close()
            return
    if viewer: viewer.close()
    print('  TSID-ID 보행 ✅ %.1fs base_z=%.3f tilt_max=%.0f° 전진=%.2fm(%.2fm/s) 벽시계%.0fs' % (
        total_T, d.qpos[2], tmax, d.qpos[0], d.qpos[0] / total_T, time.time() - t0))


def _stance_forces_at(P, x, u, gait_t, T, sf, step_h, V, dt):
    """gait_t 위상의 stance 집합 + 그 (x,u)에서의 동적 접촉력(lambda_c). settle(tg<0)=전발."""
    from quad_proxddp import _swing_turn
    if gait_t < 0:
        stance = list(P.legs)
    else:
        stance, _ = _swing_turn(gait_t, T, sf, step_h, P, 0.0, 0.0, 0.0, V)
    if not stance:
        return stance, {}
    dyn, _ = P.dynamics(stance, dt)
    dd = dyn.createData()
    dyn.forward(x, u, dd)
    lam = np.asarray(dd.continuous_data.pin_data.lambda_c)
    return stance, {L: lam[3 * k:3 * k + 3].copy() for k, L in enumerate(stance)}


def walk_rti_tsid(robot='go2', V=0.3, T=0.5, sf=0.5, step_h=0.06, settle=0.3,
                  horizon_cycles=2, total_T=10.0, dt=0.02, warm_iters=20,
                  view=False, exec_robot=None):
    """★ Part2b: RTI-NMPC(매틱 재계획) + TSID-ID(1kHz). RTI는 참조(xs[1])·접촉·힘만 제공,
       TSID가 토크 생성(모델불일치 흡수). 가설: 이전 RTI실패=ID부재였으니 TSID-ID 얹으면 안정."""
    import aligator, time
    from quad_proxddp import _xref_pose, _swing_turn, _solver_opts, _yaw_xyzw
    exec_robot = exec_robot or robot
    P = Q.ProxModel(robot)
    Hn = round(horizon_cycles * T / dt)
    q = _mujoco_setup(exec_robot); m, d = q.m, q.d
    pin2mj, mj_to_pin = QN._bridge(P.M, q)
    dt_sim = float(os.environ.get("DT_SIM","0.001")); m.opt.timestep = dt_sim; sub = round(dt / dt_sim)
    tl = TSIDLayer(P, P.x0[:P.nq], dt_sim)

    def mk_stage(tg, px, py, pyaw, v, w):
        xr = _xref_pose(P, px, py, pyaw, v, w)
        if tg < 0:
            return P.stage(P.legs, xr, dt)
        st, sw = _swing_turn(tg, T, sf, step_h, P, px, py, pyaw, v)
        return P.stage(st, xr, dt, swing=sw)

    x0 = mj_to_pin(d)
    px, py, pyaw = x0[0], x0[1], _yaw_xyzw(x0[3:7])
    gait_t = -settle; stages = []
    for k in range(Hn):
        stages.append(mk_stage(gait_t + k * dt, px, py, pyaw, V, 0.0))
        px += V * np.cos(pyaw) * dt; py += V * np.sin(pyaw) * dt
    end_px, end_py, end_pyaw = px, py, pyaw; end_t = gait_t + Hn * dt
    prob = aligator.TrajOptProblem(x0, stages, P.terminal_cost(_xref_pose(P, px, py, pyaw, V, 0.0)))
    sol = _solver_opts(aligator.SolverProxDDP(1e-3, 1e-6, max_iters=warm_iters))
    sol.setup(prob); sol.run(prob, [x0] * (Hn + 1), [np.zeros(P.nu)] * Hn)
    sol.max_iters = int(os.environ.get('RTI_ITERS', '1'))
    print('RTI 워밍업 conv=%s → RTI-NMPC + TSID-ID 1kHz' % sol.results.conv)

    tmax = 0.0; t0 = time.time()
    viewer = None
    if view:
        import mujoco.viewer as mjv; viewer = mjv.launch_passive(m, d)
    for ps in range(int(total_T / dt)):
        prob.x0_init = mj_to_pin(d)
        sol.run(prob, list(sol.results.xs), list(sol.results.us))
        xs0 = np.array(sol.results.xs[0]); xs1 = np.array(sol.results.xs[1])
        us0 = np.array(sol.results.us[0])
        # RTI 참조 → TSID: 다음 계획상태(xs1) 추종 + 현재 stance·동적힘
        stance, forces = _stance_forces_at(P, xs0, us0, gait_t, T, sf, step_h, V, dt)
        tl.set_contacts(stance); tl.set_force_ref(forces)
        a_ref = (xs1[P.nq:] - xs0[P.nq:]) / dt
        tl.set_posture_ref(xs1[:P.nq], xs1[P.nq:], a_ref)
        tl.set_base_ref(xs1[:P.nq], xs1[P.nq:]); tl.set_com_ref(xs1[:P.nq])
        for s in range(sub):
            x = mj_to_pin(d)
            tau = tl.solve((ps * sub + s) * dt_sim, x[:P.nq], x[P.nq:])
            if tau is None:
                tau = np.zeros(tl.na)
            umj = np.zeros(P.nu); umj[pin2mj] = tau[:P.nu]
            d.ctrl[:] = np.clip(umj, -TAU_LIM, TAU_LIM); mujoco.mj_step(m, d)
            if viewer:
                viewer.sync(); time.sleep(dt_sim)
        gait_t += dt; end_t += dt
        end_px += V * np.cos(end_pyaw) * dt; end_py += V * np.sin(end_pyaw) * dt
        ns = mk_stage(end_t, end_px, end_py, end_pyaw, V, 0.0)
        prob.replaceStageCircular(ns); sol.cycleProblem(prob, ns.createData())
        xx, yy = d.qpos[4], d.qpos[5]
        tmax = max(tmax, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))))
        if ps % 25 == 0:
            print('  t=%.2f base_z=%.3f x=%.2f tilt=%.1f' % (ps * dt, d.qpos[2], d.qpos[0],
                  np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1)))), flush=True)
        if d.qpos[2] < 0.15:
            print('  RTI+TSID 보행 ❌ 전복 @%.2fs (x=%.2f)' % (ps * dt, d.qpos[0]))
            if viewer: viewer.close()
            return
    if viewer: viewer.close()
    print('  RTI+TSID 보행 ✅ %.1fs base_z=%.3f tilt_max=%.0f° 전진=%.2fm(%.2fm/s) 벽시계%.0fs' % (
        total_T, d.qpos[2], tmax, d.qpos[0], d.qpos[0] / total_T, time.time() - t0))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', default='stand', choices=['stand', 'walk', 'rti'])
    ap.add_argument('--robot', default='go2', choices=['go2', 'ours'])
    ap.add_argument('--vel', type=float, default=0.3)
    ap.add_argument('--total-T', type=float, default=10.0)
    ap.add_argument('--noview', action='store_true')
    a = ap.parse_args()
    os.environ.setdefault('DISPLAY', ':0')
    if a.test == 'stand':
        tsid_stand(robot=a.robot, view=not a.noview)
    elif a.test == 'rti':
        walk_rti_tsid(robot=a.robot, V=a.vel, total_T=a.total_T, view=not a.noview)
    else:
        walk_loop_tsid(robot=a.robot, V=a.vel, total_T=a.total_T, view=not a.noview)
