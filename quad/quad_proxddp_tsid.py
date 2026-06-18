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


# ══════════════════════════════════════════════════════════════════════
# sim2real 하드웨어 추상화 (RobotInterface)
#   컨트롤러(NMPC+TSID)는 read_state()→x, apply_torque(tau) 두 함수만 본다.
#   시뮬=MujocoInterface / 실로봇=RealRobotInterface 로 교체해도 컨트롤러 코드 불변.
#   (CommandSource 가 '명령'의 seam 이라면, RobotInterface 는 '센서/액추에이터'의 seam.)
# ══════════════════════════════════════════════════════════════════════
class RobotInterface:
    """배포 경계. 컨트롤러는 아래 추상 API만 사용 → sim↔실 무변경 교체.
       nq/nv/nu = pinocchio 규약 차원(컨트롤러 모델 기준)."""
    nq = nv = nu = 0

    # ── 컨트롤러가 보는 상태/명령 (필수) ──
    def read_state(self):
        """전체 상태 x=[q_pin; v_pin] (pinocchio 규약). 실로봇=엔코더+IMU+추정."""
        raise NotImplementedError

    def apply_torque(self, tau_pin):
        """구동관절 토크(pinocchio 순서) 적용 + 1 제어틱 진행(sim)/송신(실)."""
        raise NotImplementedError

    # ── re-anchor/낙상감지에 필요한 베이스 평면자세 (실로봇은 추정 필요) ──
    def base_xy_yaw(self):
        raise NotImplementedError

    def base_height(self):
        raise NotImplementedError

    @property
    def time(self):
        raise NotImplementedError

    # ── 센서 raw (상태추정 단계서 사용; 기본은 read_state가 내부 처리) ──
    def read_imu(self):
        """(quat[w,x,y,z], gyro[3] local) — 실로봇 IMU. sim은 ground-truth 합성."""
        raise NotImplementedError

    def read_joints(self):
        """(q_joint[nu], v_joint[nu]) — 엔코더."""
        raise NotImplementedError


class MujocoInterface(RobotInterface):
    """시뮬 구현 — QuadSim(MuJoCo) 래핑 + pinocchio 브리지. ground-truth 상태 제공.
       (추후 read_state에 센서노이즈/지연을 주입하면 sim2real 강건성 단계로 확장)."""

    def __init__(self, P, exec_robot, dt_sim=0.001):
        self.P = P
        self.q = _mujoco_setup(exec_robot)
        self.m, self.d = self.q.m, self.q.d
        self.m.opt.timestep = dt_sim
        self.dt_sim = dt_sim
        self.pin2mj, self._mj_to_pin = QN._bridge(P.M, self.q)
        self.nq, self.nv, self.nu = P.nq, P.nv, P.nu

    def read_state(self):
        return self._mj_to_pin(self.d)

    def apply_torque(self, tau_pin):
        umj = np.zeros(self.nu)
        umj[self.pin2mj] = tau_pin[:self.nu]
        self.d.ctrl[:] = np.clip(umj, -TAU_LIM, TAU_LIM)
        mujoco.mj_step(self.m, self.d)

    def base_xy_yaw(self):
        d = self.d
        return float(d.qpos[0]), float(d.qpos[1]), Q._yaw_xyzw(d.qpos[[4, 5, 6, 3]])

    def base_height(self):
        return float(self.d.qpos[2])

    @property
    def time(self):
        return self.d.time

    def read_imu(self):
        return self.d.qpos[3:7].copy(), self.d.qvel[3:6].copy()   # quat wxyz, base gyro(local)

    def read_joints(self):
        return (self.d.qpos[7:7 + self.nu].copy(), self.d.qvel[6:6 + self.nu].copy())


class RealRobotInterface(RobotInterface):
    """실로봇(02_Leg) 자리표시 — ROS2 엔코더/IMU 구독 + 토크 송신 연결 지점.
       sim2real 단계서 구현: read_state는 read_joints+read_imu+상태추정으로 조립,
       apply_torque는 모터드라이버로 송신. base_xy_yaw는 추정기(odometry) 필요."""

    def __init__(self, *a, **k):
        raise NotImplementedError(
            "실로봇 인터페이스 미구현 — ROS2 구독/송신 + 상태추정 연결 필요(sim2real 다음 단계)")


class TSIDLayer:
    """우리 pinocchio 모델 위의 TSID 역동역학 QP. 발=ContactPoint, 자세·베이스 task.
       set_reference(q,v,a_ref, stance) → solve(q_meas,v_meas) → tau."""

    def __init__(self, P, q0, dt, mu=float(os.environ.get('MU_FRIC', '0.9')),   # 구조1과 동일 비보수 마찰(물리 1.3 내접)
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

        # ── swing 발 SE3 tracking task (★KinodynamicsID 구조: stance=contact, swing=발pose 직접추종) ──
        #   기존 raw-TSID엔 없어서 swing발이 관절posture로만 간접이동 → 발배치 부정확·단일지지 불안정(~5s).
        kp_feet = float(os.environ.get('KP_FEET', '150.0'))
        self.w_feet = float(os.environ.get('W_FEET', '0.0'))   # 0=off(기본,raw-TSID). >0=swing발 SE3추종(KinodynamicsID식)
        mask_pos = np.array([1, 1, 1, 0, 0, 0.0])      # 점접촉 = 위치(3D)만 추종
        self.trackTask = {}; self.sampleFeet = {}; self.tracking = set()
        for L in P.legs:
            fnm = P.model.frames[P.M.foot_fid[L]].name
            tt = tsid.TaskSE3Equality("track_" + L, self.rw, fnm)
            tt.setKp(kp_feet * np.ones(6)); tt.setKd(2.0 * np.sqrt(kp_feet) * np.ones(6))
            tt.setMask(mask_pos); tt.useLocalFrame(False)
            self.trackTask[L] = tt
            self.sampleFeet[L] = tsid.TrajectorySample(12, 6)

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
        """gait 접촉상태 갱신: stance발=rigid contact, swing발=SE3 tracking task(발pose 직접추종)."""
        for L in self.P.legs:
            want = L in stance
            have = L in self.stance
            if want and not have:                       # swing→stance: tracking 제거, contact 추가
                if L in self.tracking:
                    self.formulation.removeTask("track_" + L, 0.0); self.tracking.discard(L)
                ref = self.rw.framePosition(self.formulation.data(), self.P.M.foot_fid[L])
                self.contacts[L].setReference(ref)
                self.formulation.addRigidContact(self.contacts[L], 1e-3)
            elif have and not want:                     # stance→swing: contact 제거 (+ W_FEET>0이면 tracking)
                self.formulation.removeRigidContact(self.contacts[L].name, 0.0)
                if self.w_feet > 0:
                    ref = self.rw.framePosition(self.formulation.data(), self.P.M.foot_fid[L])  # 현재 발위치 초기화
                    val = np.zeros(12); val[:3] = ref.translation; val[3:] = ref.rotation.flatten()
                    self.sampleFeet[L].value(val); self.trackTask[L].setReference(self.sampleFeet[L])
                    self.formulation.addMotionTask(self.trackTask[L], self.w_feet, 1, 0.0)
                    self.tracking.add(L)
        self.stance = set(stance)

    def set_swing_ref(self, swing_pos):
        """swing 발 목표 위치(world 3D, NMPC 계획서 FK) → tracking task ref. 점접촉이라 위치만."""
        for L, p in swing_pos.items():
            if L not in self.tracking:
                continue
            val = np.zeros(12); val[:3] = np.asarray(p, float); val[3:] = np.eye(3).flatten()
            self.sampleFeet[L].value(val)
            self.trackTask[L].setReference(self.sampleFeet[L])

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

    def set_base_ref(self, q_ref, v_ref=None, a_ref=None):
        oMi = pin.XYZQUATToSE3(q_ref[:7])
        data = np.concatenate([oMi.translation, oMi.rotation.flatten()])
        self.sampleBase.value(data)
        if v_ref is not None:
            self.sampleBase.derivative(v_ref[:6])      # 베이스 트위스트 피드포워드
        if a_ref is not None:
            self.sampleBase.second_derivative(a_ref[:6])   # ★베이스 가속 FF(일관성: OCP 계획가속 → 반응PD 아님)
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
        pin.forwardKinematics(P.M.model, P.M.data, q_ref); pin.updateFramePlacements(P.M.model, P.M.data)
        tl.set_swing_ref({L: P.M.data.oMf[P.M.foot_fid[L]].translation.copy()
                          for L in P.legs if L not in cyc_stance[i]})   # swing발 목표=사이클 발위치
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
        pin.forwardKinematics(P.M.model, P.M.data, xs1[:P.nq]); pin.updateFramePlacements(P.M.model, P.M.data)
        tl.set_swing_ref({L: P.M.data.oMf[P.M.foot_fid[L]].translation.copy()   # ★swing발 목표=계획(xs1) 발위치
                          for L in P.legs if L not in stance})
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


def walk_teleop_tsid(robot='go2', V=0.3, T=0.5, sf=0.5, step_h=0.06, settle_steps=15,
                     n_warm=6, dt_nmpc=0.02, total_T=30.0, view=False, exec_robot=None,
                     cmd_source=None, log_path=None, plot=False):
    """★ 구조2 무한 teleop 보행: NMPC 사이클 + TSID-ID 1kHz, RobotInterface 경유(sim2real seam).
       Q._HUD/CommandSource/_se2_reanchor 재사용(구조1과 동일 조작). TSID 보행은 현재 ~5s 캡(인프라 우선)."""
    import time
    exec_robot = exec_robot or robot
    P = Q.ProxModel(robot)
    cyc = round(T / dt_nmpc)
    cyc_xs, cyc_stance, cyc_acc, cyc_forces, conv = _solve_cycle(
        P, V, T, sf, step_h, dt_nmpc, cyc, settle_steps, n_warm)
    print('NMPC 사이클 conv=%s (cyc=%d) → TSID-ID teleop(RobotInterface 경유)' % (conv, cyc))
    iface = MujocoInterface(P, exec_robot, dt_sim=0.001)
    m, d = iface.m, iface.d
    sub = round(dt_nmpc / iface.dt_sim)
    tl = TSIDLayer(P, P.x0[:P.nq], iface.dt_sim)
    hud = Q._HUD(m, d, sub, cmd=cmd_source, v0=V, foot_gids=iface.q.foot_gid) if view else None
    dlog = Q.DataLog(m, log_path) if log_path else None     # 각 축 토크/각속도 기록(--log)
    lplot = Q.LivePlot(m) if plot else None                 # 실시간 그래프(--plot)
    active_cmd = hud.cmd if hud else cmd_source
    # NOTE: struct1의 course-hold/carrot(2D 위치추종)은 TSID에 부적합(base x,y가 task-free).
    #   teleop 헤딩(yaw)만 직접 반영 + 베이스를 현재 위치에 re-anchor(불연속 제거). v/vy는 표시·후속용.
    rx, ry, yaw0 = iface.base_xy_yaw()
    yaw_ref = yaw0
    nouter = int(total_T / dt_nmpc); t0 = time.time(); tmax = 0.0
    for outer in range(nouter):
        if hud and not hud.running():
            break
        i = outer % cyc
        rx, ry, _ = iface.base_xy_yaw()
        if active_cmd is not None:
            v_cmd, w_cmd = active_cmd.read()
            yaw_ref += w_cmd * dt_nmpc                    # 운영자 선회(heading 적분)
        else:
            v_cmd, w_cmd = V, 0.0
        vy_cmd = getattr(active_cmd, 'vy', 0.0) if active_cmd is not None else 0.0
        turn = yaw_ref - yaw0
        if abs(turn) > 1e-9:                 # 선회 중에만 heading 회전(플랜 자연 yaw + turn)
            ref = cyc_xs[i].copy()
            Q._se2_reanchor([ref], 0, rx, ry, Q._yaw_xyzw(ref[3:7]) + turn)
        else:                               # 직진 = baseline 동일(절대 마칭 사이클 직접 → TSID 교란 없음)
            ref = cyc_xs[i]
        q_ref = ref[:P.nq]; v_ref = ref[P.nq:]
        tl.set_contacts(cyc_stance[i]); tl.set_force_ref(cyc_forces[i])
        pin.forwardKinematics(P.M.model, P.M.data, q_ref); pin.updateFramePlacements(P.M.model, P.M.data)
        tl.set_swing_ref({L: P.M.data.oMf[P.M.foot_fid[L]].translation.copy()
                          for L in P.legs if L not in cyc_stance[i]})   # swing발 목표=사이클 발위치
        tl.set_posture_ref(q_ref, v_ref, cyc_acc[i]); tl.set_base_ref(q_ref, v_ref); tl.set_com_ref(q_ref)
        if hud:
            hud.pre_step()
        for s in range(sub):                             # 1kHz TSID-ID — RobotInterface 경유
            x = iface.read_state()
            tau = tl.solve(iface.time, x[:P.nq], x[P.nq:])
            if tau is None:
                tau = np.zeros(tl.na)
            iface.apply_torque(tau)
            if dlog: dlog.add(d.time, d.ctrl, d.qvel[6:])   # 각 축 토크[Nm]·각속도[rad/s]
            if lplot: lplot.add(d.time, d.ctrl, d.qvel[6:])  # 실시간 그래프
        if hud:
            mk = Q._footstep_markers(P, v_cmd, T, sf, outer * dt_nmpc, d.qpos[0], d.qpos[1],
                                     Q._yaw_xyzw(d.qpos[[4, 5, 6, 3]]))
            zc = d.qpos[2] + 0.18; ln = 0.15 + 0.5 * abs(v_cmd); sgn = 1.0 if v_cmd >= 0 else -1.0
            frm = np.array([d.qpos[0], d.qpos[1], zc])
            to = frm + sgn * ln * np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
            hud.post_step(mk, arrow=(frm, to))
        xx, yy = d.qpos[4], d.qpos[5]
        tmax = max(tmax, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))))
        if os.environ.get('TRACE') and outer % 25 == 0:
            print('  t=%.2f base_z=%.3f x=%.2f y=%+.3f tilt=%.1f' % (
                outer * dt_nmpc, d.qpos[2], d.qpos[0], d.qpos[1],
                np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1)))), flush=True)
        if iface.base_height() < 0.15:
            print('  TSID teleop ❌ 전복 @%.2fs (x=%.2f)' % (outer * dt_nmpc, d.qpos[0]))
            if dlog: dlog.save()
            if lplot: lplot.close()
            if hud: hud.close()
            return
    if dlog: dlog.save()
    if lplot: lplot.close()
    if hud: hud.close()
    print('  TSID teleop ✅ %.1fs base_z=%.3f tilt_max=%.0f° 전진=%.2fm 벽시계%.0fs' % (
        total_T, d.qpos[2], tmax, d.qpos[0], time.time() - t0))


def hold_tsid(robot='go2', total_T=30.0, dt_nmpc=0.02, view=False, exec_robot=None):
    """★ 정지 버티기(TSID standing) — RobotInterface 경유. 보행X, nominal 자세 유지(외력 버팀)."""
    import time
    exec_robot = exec_robot or robot
    P = Q.ProxModel(robot)
    iface = MujocoInterface(P, exec_robot, dt_sim=0.001)
    m, d = iface.m, iface.d
    sub = round(dt_nmpc / iface.dt_sim)
    tl = TSIDLayer(P, P.x0[:P.nq], iface.dt_sim)
    q0 = P.x0[:P.nq].copy()
    tl.set_contacts(set(P.legs)); tl.set_posture_ref(q0); tl.set_base_ref(q0); tl.set_com_ref(q0)
    print('TSID 정지 버티기(보행X) → RobotInterface 경유')
    hud = Q._HUD(m, d, sub, foot_gids=iface.q.foot_gid) if view else None
    nouter = int(total_T / dt_nmpc); t0 = time.time(); tmax = 0.0
    for outer in range(nouter):
        if hud and not hud.running():
            break
        if hud:
            hud.pre_step()
        for s in range(sub):
            x = iface.read_state()
            tau = tl.solve(iface.time, x[:P.nq], x[P.nq:])
            if tau is None:
                tau = np.zeros(tl.na)
            iface.apply_torque(tau)
        if hud:
            hud.post_step()
        xx, yy = d.qpos[4], d.qpos[5]
        tmax = max(tmax, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))))
        if iface.base_height() < 0.15 or tmax > 45:
            print('  TSID 정지 ❌ 전복 @%.2fs (tilt=%.0f°)' % (outer * dt_nmpc, tmax))
            if hud: hud.close()
            return
    if hud: hud.close()
    print('  TSID 정지 ✅ %.1fs 버팀 base_z=%.3f tilt_max=%.0f° 벽시계%.0fs' % (
        total_T, d.qpos[2], tmax, time.time() - t0))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', default='stand', choices=['stand', 'hold', 'walk', 'loop', 'rti'])
    ap.add_argument('--robot', default='go2', choices=['go2', 'ours'])
    ap.add_argument('--exec-robot', default=None, choices=[None, 'go2', 'ours', 'ours_sphere'])
    ap.add_argument('--vel', type=float, default=0.3)
    ap.add_argument('--gait-T', type=float, default=0.5)
    ap.add_argument('--step-h', type=float, default=0.06)
    ap.add_argument('--total-T', type=float, default=10.0)
    ap.add_argument('--cmd', default='key', choices=['key', 'json', 'ros'],
                    help='조작기 명령원(loop 모드): key=뷰어키보드, json=/tmp/quad_cmd.json, ros=/cmd_vel')
    ap.add_argument('--noview', action='store_true')
    ap.add_argument('--log', default=None, help='loop 중 각 축 토크[Nm]/각속도[rad/s] CSV 저장 경로')
    ap.add_argument('--plot', action='store_true', help='loop 중 뷰어 옆 실시간 그래프')
    a = ap.parse_args()
    os.environ.setdefault('DISPLAY', ':0')
    if a.test == 'stand':
        tsid_stand(robot=a.robot, view=not a.noview, exec_robot=a.exec_robot)
    elif a.test == 'hold':                        # ★ 정지 버티기(보행X)
        hold_tsid(robot=a.robot, total_T=a.total_T, view=not a.noview, exec_robot=a.exec_robot)
    elif a.test == 'loop':                        # ★ 무한 teleop 보행(RobotInterface 경유)
        cs = Q.make_cmd_source(a.cmd, v0=a.vel) if a.cmd != 'key' else None
        walk_teleop_tsid(robot=a.robot, V=a.vel, T=a.gait_T, step_h=a.step_h, total_T=a.total_T,
                         view=not a.noview, exec_robot=a.exec_robot, cmd_source=cs, log_path=a.log, plot=a.plot)
    elif a.test == 'rti':
        walk_rti_tsid(robot=a.robot, V=a.vel, total_T=a.total_T, view=not a.noview, exec_robot=a.exec_robot)
    else:                                         # walk = 고정전진 검증(기존)
        walk_loop_tsid(robot=a.robot, V=a.vel, T=a.gait_T, step_h=a.step_h,
                       total_T=a.total_T, view=not a.noview, exec_robot=a.exec_robot)
