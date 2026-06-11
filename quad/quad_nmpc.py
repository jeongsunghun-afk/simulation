"""02_Leg 전신(whole-body) NMPC — crocoddyl FDDP.  4단계.

SRBD(단일강체) MPC 의 한계(leg-heavy 02_Leg 에서 다리질량 78% 무시) 를 극복하기 위해
pinocchio 전신 동역학 + crocoddyl OCP 로 푼다. MuJoCo(quad_sim) 와 동일 URDF → 단일 소스.

구성:
  · NMPCModel : URDF→pinocchio, sole 접촉프레임 추가, StateMultibody/actuation, MuJoCo↔pin 매핑
  · build_action / build_terminal : crocoddyl per-step OCP (gait_sim.nmpc 패턴 차용)
  · solve_standing : 정지균형 NMPC 검증 (1단계)

실행: python3 quad_nmpc.py --test stand
"""
import os
import glob
import time
import argparse
import numpy as np
import pinocchio as pin
import crocoddyl

_HERE = os.path.dirname(os.path.abspath(__file__))
LEGS_PIN = ['FL', 'FR', 'HL', 'HR']     # (기본=02_Leg; 실제는 NMPCModel.legs 사용)
_GO2_XML = os.path.join(_HERE, '..', 'mujoco_menagerie', 'unitree_go2', 'go2.xml')
ROBOTS_PIN = {
    'ours': dict(legs=['FL', 'FR', 'HL', 'HR'], dof=4,
                 diag={'FL': 0.0, 'HR': 0.0, 'FR': 0.5, 'HL': 0.5}),   # 대각쌍 FL+HR / FR+HL
    'go2':  dict(legs=['FL', 'FR', 'RL', 'RR'], dof=3,
                 diag={'FL': 0.0, 'RR': 0.0, 'FR': 0.5, 'RL': 0.5}),   # 대각쌍 FL+RR / FR+RL
}
MU_FRIC = 0.6 * 0.707                    # elliptic cone 내접 마진(quad_sim 과 동일)
# NMPC 비용 가중 (env 튜닝)
# 튜닝 winning config 기본값 (closed-loop trot 안정성 최적; 스윕으로 도출)
W_ORI = float(os.environ.get('W_ORI', '150'))      # base 자세(roll/pitch) reg
W_JOINT = float(os.environ.get('W_JOINT', '0.2'))  # 관절각 reg (낮춰 swing 덜 방해)
W_SWING = float(os.environ.get('W_SWING', '100'))  # swing 발 위치추종
W_FRIC = float(os.environ.get('W_FRIC', '2.0'))    # 마찰콘 (높여 soft 위반↓)
# aligator controlFeedbacks 부호 → control()의 u=u0−K0·δx 규약에 맞춤(croc 반대부호 가정)
_PROXDDP_KSGN = float(os.environ.get('PROXDDP_KSGN', '-1.0'))


def _aligator():
    """ProxDDP(aligator) 지연 import — FDDP 만 쓸 땐(시스템 env) 불필요."""
    import aligator
    return aligator


class NMPCModel:
    """pinocchio 전신 모델 + crocoddyl 상태/구동 + 접촉프레임. robot='ours'(URDF)/'go2'(MJCF)."""

    def __init__(self, robot=None):
        robot = robot or os.environ.get('NMPC_ROBOT', 'ours')
        cfg = ROBOTS_PIN[robot]
        self.robot = robot; self.legs = cfg['legs']; self.dof = cfg['dof']; self.diag = cfg['diag']
        if robot == 'ours':
            urdf = sorted(glob.glob(os.path.join(_HERE, 'urdf', '*.urdf')))[0]
            self.model = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
            self._add_foot_frames_ours()
        else:  # go2: MJCF 직접 로드(MuJoCo와 동일 모델) + 발프레임 추가
            self.model = pin.buildModelFromMJCF(_GO2_XML, pin.JointModelFreeFlyer(), 'root')[0]
            self._add_foot_frames_go2()
        self.foot_fid = {L: self.model.getFrameId(L + '_foot') for L in self.legs}
        self.data = self.model.createData()
        self.state = crocoddyl.StateMultibody(self.model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.nu = self.model.nv - 6        # = legs×dof (MJCF 로드시 actuation.nu 오계산 회피)
        self.tau_lim = np.full(self.nu, 80.0)
        self.q0, self.v0, self.x0 = self._crouch_state()
        pin.forwardKinematics(self.model, self.data, self.q0)
        pin.updateFramePlacements(self.model, self.data)
        self.foot_home = {L: self.data.oMf[self.foot_fid[L]].translation.copy()
                          for L in self.legs}

    def _add_foot_frames_ours(self):
        """02_Leg: foot_contact_link 메시 sole 최저점에 {L}_foot 프레임 추가."""
        import sys
        _a = sys.argv; sys.argv = [_a[0]]
        try:
            import quad_sim
            quad_sim._ROBOT = 'ours'
            q = quad_sim.QuadSim()
            sole_off = {L: q.sole_off[i].copy() for i, L in enumerate(q.legs)}
        finally:
            sys.argv = _a
        for L in self.legs:
            fr = self.model.frames[self.model.getFrameId(L + '_foot_contact_link')]
            placement = fr.placement * pin.SE3(np.eye(3), sole_off[L])
            self.model.addFrame(pin.Frame(L + '_foot', fr.parentJoint, fr.parentFrame,
                                          placement, pin.FrameType.OP_FRAME))

    def _add_foot_frames_go2(self):
        """Go2: calf 의 sphere 발 접촉점(calf 로컬 [-0.0196,0,-0.2262])에 {L}_foot 프레임."""
        off = pin.SE3(np.eye(3), np.array([-0.0196, 0.0, -0.2262]))
        for L in self.legs:
            jid = self.model.getJointId(L + '_calf_joint')
            self.model.addFrame(pin.Frame(L + '_foot', jid, 0, off, pin.FrameType.OP_FRAME))

    def _crouch_state(self):
        """MuJoCo crouch_home 자세를 pinocchio q0/v0/x0 로 변환."""
        import sys
        _a = sys.argv; sys.argv = [_a[0]]
        try:
            import quad_sim, mujoco
            quad_sim._ROBOT = self.robot
            q = quad_sim.QuadSim(); q.crouch_home(); mujoco.mj_forward(q.m, q.d)
            mq = q.d.qpos.copy()
            mleg = {L: list(q.legqp[i]) for i, L in enumerate(q.legs)}
        finally:
            sys.argv = _a
        q0 = np.zeros(self.model.nq)
        q0[0:3] = mq[0:3]
        w, x, y, z = mq[3:7]; q0[3:7] = [x, y, z, w]      # wxyz→xyzw
        ji = 7
        for L in self.legs:
            for idx in mleg[L]:
                q0[ji] = mq[idx]; ji += 1
        v0 = np.zeros(self.model.nv)
        return q0, v0, np.concatenate([q0, v0])


def _contact_multiple(M, stance):
    cm = crocoddyl.ContactModelMultiple(M.state, M.nu)
    for L in stance:
        c = crocoddyl.ContactModel3D(
            M.state, M.foot_fid[L], np.zeros(3),
            pin.LOCAL_WORLD_ALIGNED, M.nu, np.array([0.0, 50.0]))   # Baumgarte kp,kd
        cm.addContact('c_' + L, c)
    return cm


def build_action(M, stance, x_ref, dt, swing_targets=None):
    """per-step IntegratedActionModelEuler. stance=접촉다리 리스트,
       swing_targets={L: pos(3)} 면 해당 발 위치추종 cost."""
    swing_targets = swing_targets or {}
    cm = _contact_multiple(M, stance)
    cost = crocoddyl.CostModelSum(M.state, M.nu)
    # 상태 정규화(목표 x_ref 추종): base pose/자세 + 관절 + 속도
    w_state = np.array([0, 0, 50, W_ORI, W_ORI, 10] +    # base pos(xyz) + ori(rpy)
                       [W_JOINT] * M.nu +                 # 관절각
                       [1] * 6 + [0.1] * M.nu)            # base vel + 관절속도
    cost.addCost('xreg', crocoddyl.CostModelResidual(
        M.state, crocoddyl.ActivationModelWeightedQuad(w_state ** 2),
        crocoddyl.ResidualModelState(M.state, x_ref, M.nu)), 1.0)
    cost.addCost('ureg', crocoddyl.CostModelResidual(
        M.state, crocoddyl.ResidualModelControl(M.state, M.nu)), 1e-3)
    # 마찰콘 + 접촉력 정규화 (stance)
    for L in stance:
        fc = crocoddyl.FrictionCone(np.eye(3), MU_FRIC, 4, False)
        fc_act = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(fc.lb, fc.ub))
        fc_res = crocoddyl.ResidualModelContactFrictionCone(
            M.state, M.foot_fid[L], fc, M.nu, True)
        cost.addCost('fc_' + L, crocoddyl.CostModelResidual(M.state, fc_act, fc_res), W_FRIC)
    # 토크 한계 배리어
    tau_act = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(-M.tau_lim, M.tau_lim))
    cost.addCost('taulim', crocoddyl.CostModelResidual(
        M.state, tau_act, crocoddyl.ResidualModelControl(M.state, M.nu)), 1.0)
    # swing 발 위치추종
    for L, tgt in swing_targets.items():
        res = crocoddyl.ResidualModelFrameTranslation(M.state, M.foot_fid[L], tgt, M.nu)
        cost.addCost('sw_' + L, crocoddyl.CostModelResidual(M.state, res), W_SWING)
    diff = crocoddyl.DifferentialActionModelContactFwdDynamics(
        M.state, M.actuation, cm, cost, 0.0, True)
    return crocoddyl.IntegratedActionModelEuler(diff, dt)


def build_terminal(M, x_ref, w=None):
    cm = _contact_multiple(M, M.legs)
    cost = crocoddyl.CostModelSum(M.state, M.nu)
    res = crocoddyl.ResidualModelState(M.state, x_ref, M.nu)
    creg = (crocoddyl.CostModelResidual(M.state,
            crocoddyl.ActivationModelWeightedQuad(w ** 2), res)
            if w is not None else crocoddyl.CostModelResidual(M.state, res))
    cost.addCost('xreg_T', creg, 1e2)
    diff = crocoddyl.DifferentialActionModelContactFwdDynamics(
        M.state, M.actuation, cm, cost, 0.0, True)
    return crocoddyl.IntegratedActionModelEuler(diff, 0.0)


def solve_standing(N=30, dt=0.02):
    """정지균형 NMPC 검증 — 전 다리 stance, x0 유지."""
    M = NMPCModel()
    print('NMPCModel: nq=%d nv=%d nu=%d 총질량=%.1fkg' % (
        M.model.nq, M.model.nv, M.nu, sum(i.mass for i in M.model.inertias)))
    print('발 home:', {L: np.round(M.foot_home[L], 3).tolist() for L in LEGS_PIN})
    actions = [build_action(M, LEGS_PIN, M.x0, dt) for _ in range(N)]
    terminal = build_terminal(M, M.x0)
    problem = crocoddyl.ShootingProblem(M.x0, actions, terminal)
    solver = crocoddyl.SolverFDDP(problem)
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    t0 = time.time()
    done = solver.solve([M.x0] * (N + 1), [np.zeros(M.nu)] * N, 100, False, 1e-9)
    print('done=%s iters=%d time=%.0fms cost=%.3e' % (
        done, solver.iter, (time.time() - t0) * 1e3, solver.cost))
    xs = np.array(solver.xs)
    drift = np.linalg.norm(xs[-1][:3] - M.x0[:3])
    print('base 이동(드리프트)=%.4fm, 최종 base_z=%.3f (시작 %.3f)' % (
        drift, xs[-1][2], M.x0[2]))
    # 토크 크기
    us = np.array(solver.us)
    print('토크 RMS=%.1fNm max=%.1fNm' % (np.sqrt(np.mean(us ** 2)), np.abs(us).max()))
    return done


def solve_lift(lift_leg='FL', N=60, dt=0.02, hold=15, lift_h=0.08):
    """결정적 검증 — SRBD+WBIC 가 못한 '한 다리 들기'를 whole-body NMPC 가 푸나.
    스케줄: hold 스텝 4발 정착 → 이후 lift_leg swing(3발 지지) + 발 8cm 들기.
    NMPC 가 다리질량 반영해 CoM 이동/균형을 계획하면 base 가 안 넘어가야 함."""
    M = NMPCModel()
    stance_all = LEGS_PIN
    stance_3 = [L for L in LEGS_PIN if L != lift_leg]
    home = M.foot_home[lift_leg]
    actions = []
    for k in range(N):
        if k < hold:
            actions.append(build_action(M, stance_all, M.x0, dt))
        else:
            s = min(1.0, (k - hold) / (N - hold) * 2)            # 0→1 들어올림
            tgt = home + np.array([0, 0, lift_h * s])
            actions.append(build_action(M, stance_3, M.x0, dt, swing_targets={lift_leg: tgt}))
    terminal = build_terminal(M, M.x0)
    problem = crocoddyl.ShootingProblem(M.x0, actions, terminal)
    solver = crocoddyl.SolverFDDP(problem)
    t0 = time.time()
    done = solver.solve([M.x0] * (N + 1), [np.zeros(M.nu)] * N, 200, False, 1e-9)
    xs = np.array(solver.xs)
    # base 자세(roll/pitch) 최대 + 들린발 최종 높이
    M.data = M.model.createData()
    def tilt(x):
        quat = x[3:7]  # xyzw
        R = pin.Quaternion(quat[3], quat[0], quat[1], quat[2]).toRotationMatrix()
        return np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    tilts = [tilt(x) for x in xs]
    footz_max = 0.0
    for x in xs:
        pin.forwardKinematics(M.model, M.data, x[:M.model.nq])
        pin.updateFramePlacements(M.model, M.data)
        footz_max = max(footz_max, M.data.oMf[M.foot_fid[lift_leg]].translation[2])
    print('done=%s iters=%d time=%.0fms cost=%.2e' % (
        done, solver.iter, (time.time() - t0) * 1e3, solver.cost))
    lifted = footz_max - M.foot_home[lift_leg][2]
    print('%s 들기: 최대 발들림=%.3fm(목표%.3f) base_z=%.3f tilt_max=%.1f° %s' % (
        lift_leg, lifted, lift_h, xs[-1][2], max(tilts),
        '✅ NMPC 한다리 들기 성공!' if lifted > 0.04 and max(tilts) < 15 and done else '❌ 발 안들림'))
    return done


def _trot_schedule(tk, T, swing_frac, step_h, foot_home, legs, diag, V=0.0, march=0.0):
    """trot 대각쌍 교대. legs=다리목록, diag={L:위상오프셋}. tk=gait위상시각[s].
       march=발 착지 전방오프셋 — **몸의 실제 전진량 기준**(절대시간 아님 → 발이 몸 안떠남)."""
    stance, swing = [], {}
    for L in legs:
        off = diag[L]
        ph = (tk / T + off) % 1.0
        if ph < swing_frac:                          # swing
            s = ph / swing_frac
            z = 4 * step_h * s * (1 - s)              # 포물선 아치 (peak step_h)
            fwd = V * T * swing_frac * (s - 0.5)      # swing 중 전방 스윙(march 위에 추가)
            swing[L] = foot_home[L] + np.array([march + fwd, 0, z])
        else:
            stance.append(L)
    return stance, swing


def solve_trot(N=75, dt=0.02, T=0.5, swing_frac=0.5, step_h=0.07, V=0.0):
    """trot NMPC 궤적 풀이 (one-shot). 반환 (M, xs)."""
    M = NMPCModel()
    print('NMPC trot: N=%d dt=%.3f T=%.2f V=%.2f' % (N, dt, T, V))
    actions = []
    for k in range(N):
        st, sw = _trot_schedule(k * dt, T, swing_frac, step_h, M.foot_home, M.legs, M.diag, V)
        actions.append(build_action(M, st, M.x0, dt, swing_targets=sw))
    x_ref_T = M.x0.copy(); x_ref_T[0] += V * (N * dt)
    terminal = build_terminal(M, x_ref_T)
    problem = crocoddyl.ShootingProblem(M.x0, actions, terminal)
    solver = crocoddyl.SolverFDDP(problem)
    t0 = time.time()
    done = solver.solve([M.x0] * (N + 1), [np.zeros(M.nu)] * N, 300, False, 1e-9)
    print('done=%s iters=%d time=%.0fms cost=%.2e' % (
        done, solver.iter, (time.time() - t0) * 1e3, solver.cost))
    return M, np.array(solver.xs)


def replay_mujoco(M, xs, dt, loops=8):
    """NMPC 계획 궤적(xs)을 MuJoCo 뷰어에 운동학 재생 (pin q → mujoco qpos)."""
    import sys
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        import quad_sim, mujoco
        import mujoco.viewer as mjv
        quad_sim._ROBOT = 'ours'
        q = quad_sim.QuadSim()
    finally:
        sys.argv = _a
    midx = {L: list(q.legqp[q.legs.index(L)]) for L in LEGS_PIN}   # pin leg → mujoco qpos
    nq = M.model.nq

    def to_mujoco(qpin):
        mq = q.d.qpos.copy()
        mq[0:3] = qpin[0:3]
        x, y, z, w = qpin[3:7]; mq[3:7] = [w, x, y, z]       # xyzw→wxyz
        ji = 7
        for L in LEGS_PIN:
            for m_i in midx[L]:
                mq[m_i] = qpin[ji]; ji += 1
        return mq
    print('뷰어 재생 — 창 닫으면 종료. (NMPC 계획 운동학 재생, %d회 반복)' % loops)
    with mjv.launch_passive(q.m, q.d) as v:
        for _ in range(loops):
            for x in xs:
                if not v.is_running():
                    return
                q.d.qpos[:] = to_mujoco(x[:nq])
                mujoco.mj_forward(q.m, q.d)
                v.sync()
                time.sleep(dt)


def closed_loop(M, xs, us, dt=0.02, KP=60.0, KD=2.0, loops=10, view=True):
    """closed-loop: NMPC 계획 토크(feedforward) + 관절 PD 추종을 MuJoCo 물리에 적용.
       (1차 closed-loop — 재계획 없이 계획궤적 실행 + PD 안정화. 실제 토크제어/물리적분.)"""
    import sys
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        import quad_sim, mujoco
        import mujoco.viewer as mjv
        quad_sim._ROBOT = 'ours'
        q = quad_sim.QuadSim(); q.crouch_home()
    finally:
        sys.argv = _a
    m, d = q.m, q.d
    nq = M.model.nq
    # pin 관절 인덱스(FL,FR,HL,HR순) → mujoco 관절 인덱스
    pin2mj = np.zeros(M.nu, dtype=int)
    for pi, L in enumerate(LEGS_PIN):
        ml = q.legs.index(L)
        for j in range(4):
            pin2mj[pi * 4 + j] = ml * 4 + j
    qadr = m.jnt_qposadr[1]                      # base 이후 첫 관절 qpos 시작
    vadr = m.jnt_dofadr[1]
    # 초기 상태 = 계획 x0 (crouch) — 이미 crouch_home 으로 일치
    sub = max(1, round(dt / m.opt.timestep))     # 계획스텝당 sim스텝 (=10)
    N = len(us)

    def step_block(k):
        tau_pin = us[k]
        qj_plan = np.zeros(M.nu); qj_plan[pin2mj] = xs[k][7:7 + M.nu]
        vj_plan = np.zeros(M.nu); vj_plan[pin2mj] = xs[k][nq + 6:nq + 6 + M.nu]
        tau_ff = np.zeros(M.nu); tau_ff[pin2mj] = tau_pin
        for _ in range(sub):
            qj = d.qpos[qadr:qadr + M.nu]; vj = d.qvel[vadr:vadr + M.nu]
            d.ctrl[:] = np.clip(tau_ff + KP * (qj_plan - qj) + KD * (vj_plan - vj), -80, 80)
            mujoco.mj_step(m, d)

    def run(v=None):
        for _ in range(loops):
            for k in range(N):
                if v is not None and not v.is_running():
                    return
                step_block(k)
                if v is not None:
                    v.sync(); time.sleep(dt)
        if v is None:
            x, y, zq = d.qpos[4], d.qpos[5], d.qpos[6]
            ti = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
            print('closed-loop %d회: base_z=%.3f tilt=%.1f° %s' % (
                loops, d.qpos[2], ti, '✅' if d.qpos[2] > 0.3 and ti < 25 else '❌'))
    if view:
        print('closed-loop 뷰어 — NMPC 토크 + PD, 물리 적분.')
        with mjv.launch_passive(m, d) as v:
            run(v)
    else:
        run(None)


def _bridge(M, q):
    """pin↔mujoco 매핑/변환 헬퍼 반환."""
    import mujoco
    pin2mj = np.zeros(M.nu, dtype=int)
    for pi, L in enumerate(M.legs):
        ml = q.legs.index(L)
        for j in range(M.dof):
            pin2mj[pi * M.dof + j] = ml * M.dof + j

    def mj_to_pin(d):
        qp = np.zeros(M.model.nq); vp = np.zeros(M.model.nv)
        qp[0:3] = d.qpos[0:3]
        w, x, y, z = d.qpos[3:7]; qp[3:7] = [x, y, z, w]
        R = np.zeros(9); mujoco.mju_quat2Mat(R, d.qpos[3:7]); R = R.reshape(3, 3)
        vp[0:3] = R.T @ d.qvel[0:3]      # world lin → local
        vp[3:6] = d.qvel[3:6]            # ang (mujoco free joint = local)
        qp[7:] = d.qpos[7:7 + M.nu][pin2mj]
        vp[6:] = d.qvel[6:6 + M.nu][pin2mj]
        return np.concatenate([qp, vp])
    return pin2mj, mj_to_pin


def nmpc_mpc_loop(N=24, dt=0.02, T=0.5, sf=0.5, step_h=0.06, V=0.0,
                  resolve_every=1, sim_T=6.0, maxiter=15, view=True, solver_name='fddp',
                  settle=0.6):
    """receding-horizon NMPC closed-loop. 매 resolve_every sim스텝마다 짧은 호라이즌
       NMPC 재계획(warm-start, 위상 shift) → us0 적용(+필요시 DDP 피드백). MuJoCo 물리 적분.
       gait 는 접촉모드 변화 때문에 매 스텝 재계획(resolve_every=1) 필요(정지는 드물어도 OK)."""
    import sys
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        import quad_sim, mujoco
        import mujoco.viewer as mjv
        quad_sim._ROBOT = 'ours'
        q = quad_sim.QuadSim(); q.crouch_home()
    finally:
        sys.argv = _a
    m, d = q.m, q.d
    M = NMPCModel()
    pin2mj, mj_to_pin = _bridge(M, q)
    state = M.state
    sim_dt = m.opt.timestep
    xs_ws = [M.x0] * (N + 1); us_ws = [np.zeros(M.nu)] * N
    cur = dict(u0=np.zeros(M.nu), K0=np.zeros((M.nu, 2 * M.model.nv)), x0=M.x0.copy())
    # 전진 명령: x_ref 의 base 선속도 vx=V (pin local≈world, 직립 가정). terminal 은 base xy 자유.
    x_ref_fwd = M.x0.copy(); x_ref_fwd[M.model.nq + 0] = V
    x_ref_hold = M.x0.copy()                              # 정착용(속도 0, 제자리)
    ndx = 2 * M.model.nv
    w_T = np.ones(ndx); w_T[0] = 0.0; w_T[1] = 0.0       # base x,y 위치 자유(전진 허용)
    gait_start = [settle]                                 # gait 시작 시각(이전 settle 정착)

    def resolve(t, x):
        # warm-start 위상 shift (재계획 간 gait 전진만큼만; sub-dt 면 shift=0)
        shift = round(resolve_every * sim_dt / dt)
        if shift:
            xs_ws[:] = xs_ws[shift:] + [xs_ws[-1]] * shift
            us_ws[:] = us_ws[shift:] + [us_ws[-1]] * shift
        body_x = x[0]                                    # 현재 실제 base x (발 march 기준)
        actions = []
        for k in range(N):
            tg = (t + k * dt) - gait_start[0]            # gait 위상 시각(<0=정착)
            if tg < 0:                                   # ① 정착: 전 다리 stance, 제자리
                actions.append(build_action(M, LEGS_PIN, x_ref_hold, dt))
            else:                                        # ② gait: trot + 전진
                march = body_x + V * (k * dt)            # 실제 몸 위치 + 호라이즌 예측(절대시간 아님)
                st, sw = _trot_schedule(tg, T, sf, step_h, M.foot_home, M.legs, M.diag, V, march=march)
                actions.append(build_action(M, st, x_ref_fwd, dt, swing_targets=sw))
        x_ref_T = x_ref_hold if (t + N * dt) < gait_start[0] else x_ref_fwd
        prob = crocoddyl.ShootingProblem(x.copy(), actions, build_terminal(M, x_ref_T, w_T))
        if solver_name == 'fddp':
            solver = crocoddyl.SolverFDDP(prob)
            solver.solve(xs_ws, us_ws, maxiter, False, 1e-9)
            cur['u0'] = np.array(solver.us[0]); cur['K0'] = np.array(solver.K[0])
            cur['x0'] = np.array(solver.xs[0])
            return list(solver.xs), list(solver.us)
        elif solver_name in ('proxddp', 'aligator'):
            # ProxDDP(aligator): crocoddyl 문제 변환 → proximal 솔버. (현재 soft 제약 변환;
            #   추후 마찰콘/토크 hard constraint 네이티브化 — PROXDDP 논문 참조)
            aprob = _aligator().croc.convertCrocoddylProblem(prob)
            sol = _aligator().SolverProxDDP(1e-3, 1e-2, max_iters=maxiter)
            sol.setup(aprob); sol.run(aprob, xs_ws, us_ws)
            r = sol.results
            cur['u0'] = np.array(r.us[0])
            cur['K0'] = _PROXDDP_KSGN * np.array(r.controlFeedbacks()[0])  # 부호규약 보정
            cur['x0'] = np.array(r.xs[0])
            return list(r.xs), list(r.us)
        else:
            raise ValueError(solver_name)

    def control(it):
        x = mj_to_pin(d)
        if it % resolve_every == 0:
            xs_n, us_n = resolve(it * sim_dt, x)
            xs_ws[:] = xs_n; us_ws[:] = us_n
        dx = state.diff(cur['x0'], x)
        u = cur['u0'] - cur['K0'] @ dx               # DDP 피드백 정책
        umj = np.zeros(M.nu); umj[pin2mj] = u        # pin→mujoco 토크
        d.ctrl[:] = np.clip(umj, -80, 80)
        mujoco.mj_step(m, d)

    def reset(it):
        q.crouch_home()
        xs_ws[:] = [M.x0] * (N + 1); us_ws[:] = [np.zeros(M.nu)] * N
        cur['u0'] = np.zeros(M.nu); cur['K0'] = np.zeros((M.nu, 2 * M.model.nv))
        cur['x0'] = M.x0.copy()
        gait_start[0] = it * sim_dt + settle              # 리셋 후 다시 settle 정착 거쳐 gait

    nsteps = int(sim_T / sim_dt)
    if view:
        print('NMPC %s closed-loop 뷰어 (재계획 %dHz, 호라이즌 %.2fs, 전복시 자동리셋)' % (
            solver_name.upper(), 1 / (resolve_every * sim_dt), N * dt))
        with mjv.launch_passive(m, d) as v:
            for it in range(nsteps):
                if not v.is_running():
                    break
                control(it)
                if d.qpos[2] < 0.18:                  # 전복 → 리셋 후 재시도
                    reset(it)
                if it % resolve_every == 0:
                    x, y = d.qpos[4], d.qpos[5]
                    ti = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
                    phase = 'settle' if it * sim_dt < gait_start[0] else 'gait'
                    v.set_texts((mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                 'NMPC %s [%s]' % (solver_name.upper(), phase),
                                 'z=%.2f tilt=%.0f' % (d.qpos[2], ti)))
                    v.sync()
    else:
        tmax = 0.0
        for it in range(nsteps):
            control(it)
            x, y = d.qpos[4], d.qpos[5]
            tmax = max(tmax, np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
            if d.qpos[2] < 0.15:
                print('closed-loop ❌ 전복 @%.1fs (x=%.2fm)' % (it * sim_dt, d.qpos[0])); return
        print('closed-loop ✅ %.1fs 생존 base_z=%.3f tilt_max=%.0f° 전진x=%.2fm(평균%.2fm/s)' % (
            sim_T, d.qpos[2], tmax, d.qpos[0], d.qpos[0] / sim_T))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', default='stand', choices=['stand', 'lift', 'trot', 'closed', 'mpc'])
    ap.add_argument('--leg', default='FL', choices=LEGS_PIN)
    ap.add_argument('--vel', type=float, default=0.0, help='전진속도 m/s')
    ap.add_argument('--solver', default='fddp', choices=['fddp', 'proxddp'],
                    help='NMPC 솔버 (fddp=crocoddyl, proxddp=aligator 추후)')
    ap.add_argument('--noview', action='store_true')
    a = ap.parse_args()
    os.environ.setdefault('DISPLAY', ':0')
    if a.test == 'stand':
        solve_standing()
    elif a.test == 'lift':
        solve_lift(a.leg)
    elif a.test == 'trot':
        M, xs = solve_trot(V=a.vel)
        if not a.noview:
            replay_mujoco(M, xs, 0.02)
    elif a.test == 'closed':
        M = NMPCModel()
        N, dt = 75, 0.02
        actions = []
        for k in range(N):
            st, sw = _trot_schedule(k * dt, 0.5, 0.5, 0.07, M.foot_home, a.vel)
            actions.append(build_action(M, st, M.x0, dt, swing_targets=sw))
        x_ref_T = M.x0.copy(); x_ref_T[0] += a.vel * N * dt
        problem = crocoddyl.ShootingProblem(M.x0, actions, build_terminal(M, x_ref_T))
        solver = crocoddyl.SolverFDDP(problem)
        solver.solve([M.x0] * (N + 1), [np.zeros(M.nu)] * N, 300, False, 1e-9)
        print('NMPC trot 계획 done, cost=%.2e' % solver.cost)
        closed_loop(M, np.array(solver.xs), np.array(solver.us), dt, view=not a.noview)
    elif a.test == 'mpc':
        nmpc_mpc_loop(V=a.vel, view=not a.noview, solver_name=a.solver)
