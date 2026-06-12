"""02_Leg 전신 NMPC — aligator ProxDDP **네이티브 hard constraint**.

quad_nmpc.py(FDDP, crocoddyl, soft penalty) 와 달리, 마찰콘·토크를 ProxDDP 의
**explicit constraint**(proximal/AL) 로 푼다. PROXDDP 논문(quad/PROXDDP-*.pdf) 방향.

구성:
  · 접촉 동역학 = pinocchio RigidConstraintModel(발 3D) + MultibodyConstraintFwdDynamics
  · stage = state/control reg cost + swing 발 추종 + [마찰콘 NegativeOrthant + 토크 BoxConstraint]
  · SolverProxDDP

검증: conda env python. 모델은 quad_nmpc.NMPCModel(pinocchio) 재사용.
"""
import os
import time
import numpy as np
import mujoco
import pinocchio as pin
import aligator
from aligator import constraints as acon

import quad_nmpc as QN

LEGS = QN.LEGS_PIN
MU = 0.6 * 0.707
TAU_LIM = 80.0
# cost 가중 (env 튜닝): base z/자세(roll·pitch)/yaw, 관절, base속도, 관절속도, swing, 제어
PW = dict(z=float(os.environ.get('PW_Z', '50')), ori=float(os.environ.get('PW_ORI', '150')),
          yaw=float(os.environ.get('PW_YAW', '10')), j=float(os.environ.get('PW_J', '0.2')),
          bv=float(os.environ.get('PW_BV', '1')), jv=float(os.environ.get('PW_JV', '0.1')),
          sw=float(os.environ.get('PW_SW', '100')), u=float(os.environ.get('PW_U', '1e-3')))


def _wstate(nu):
    return np.concatenate([[0, 0, PW['z']], [PW['ori'], PW['ori'], PW['yaw']],
                           [PW['j']] * nu, [PW['bv']] * 6, [PW['jv']] * nu])
# 마찰 pyramid A(5x3): A·f ≤ 0  (fz≥0, |fx|≤μfz, |fy|≤μfz)
_FC_A = np.array([[0, 0, -1.0],
                  [1, 0, -MU], [-1, 0, -MU],
                  [0, 1, -MU], [0, -1, -MU]])


class ProxModel:
    """pinocchio 모델 + aligator phase space/actuation + 접촉 제약모델(발별)."""

    def __init__(self, robot='ours'):
        self.M = QN.NMPCModel(robot)
        self.model = self.M.model
        self.legs = self.M.legs; self.diag = self.M.diag; self.dof = self.M.dof
        self.nq, self.nv, self.nu = self.model.nq, self.model.nv, self.M.nu
        self.space = aligator.manifolds.MultibodyPhaseSpace(self.model)
        self.ndx = self.space.ndx
        self.B = np.zeros((self.nv, self.nu)); self.B[6:, :] = np.eye(self.nu)
        self.prox = pin.ProximalSettings(1e-9, 1e-10, 10)
        self.x0 = self.M.x0
        self.foot_home = self.M.foot_home
        # 발별 RigidConstraintModel(3D 점접촉, 발 프레임 parent joint + placement)
        self.rcm = {}
        for L in self.legs:
            fr = self.model.frames[self.M.foot_fid[L]]
            c = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D, self.model,
                                         fr.parentJoint, fr.placement, pin.LOCAL_WORLD_ALIGNED)
            c.name = L
            self.rcm[L] = c

    def _rcms(self, stance):
        v = pin.StdVec_RigidConstraintModel()
        for L in stance:
            v.append(self.rcm[L])
        return v

    def dynamics(self, stance, dt):
        rcms = self._rcms(stance)
        dyn = aligator.dynamics.MultibodyConstraintFwdDynamics(self.space, self.B, rcms, self.prox)
        return aligator.dynamics.IntegratorSemiImplEuler(dyn, dt), rcms

    def stage(self, stance, x_ref, dt, swing=None, w_state=None):
        swing = swing or {}
        disc, rcms = self.dynamics(stance, dt)
        # ── cost ──
        cost = aligator.CostStack(self.space, self.nu)
        if w_state is None:
            w_state = _wstate(self.nu)
        cost.addCost('xreg', aligator.QuadraticStateCost(
            self.space, self.nu, x_ref, np.diag(w_state ** 2)), 1.0)
        cost.addCost('ureg', aligator.QuadraticControlCost(
            self.space, np.zeros(self.nu), PW['u'] * np.eye(self.nu)), 1.0)
        for L, tgt in swing.items():
            res = aligator.FrameTranslationResidual(self.ndx, self.nu, self.model, tgt, self.M.foot_fid[L])
            cost.addCost('sw_' + L, aligator.QuadraticResidualCost(self.space, res, PW['sw'] * np.eye(3)), 1.0)
        stm = aligator.StageModel(cost, disc)
        # ── hard constraints ──
        # 마찰콘: A·f ≤ 0 (stance 발마다)
        for L in stance:
            cfr = aligator.ContactForceResidual(self.ndx, self.model, self.B, rcms, self.prox,
                                                np.zeros(3), L)
            fcone = aligator.linear_compose(cfr, _FC_A, np.zeros(5))
            stm.addConstraint(fcone, acon.NegativeOrthant())
        # 토크 한계: -TAU ≤ u ≤ TAU (BoxConstraint)
        ures = aligator.ControlErrorResidual(self.ndx, np.zeros(self.nu))
        stm.addConstraint(ures, acon.BoxConstraint(-TAU_LIM * np.ones(self.nu),
                                                   TAU_LIM * np.ones(self.nu)))
        return stm

    def terminal_cost(self, x_ref, w_state=None):
        if w_state is None:
            w_state = _wstate(self.nu)
        return aligator.QuadraticStateCost(self.space, self.nu, x_ref, np.diag((1e2 * w_state) ** 2))


def solve_standing(N=24, dt=0.02, maxiter=50, robot="ours"):
    P = ProxModel(robot)
    stages = [P.stage(P.legs, P.x0, dt) for _ in range(N)]
    problem = aligator.TrajOptProblem(P.x0, stages, P.terminal_cost(P.x0))
    solver = aligator.SolverProxDDP(1e-4, 1e-2, max_iters=maxiter)
    solver.setup(problem)
    import time
    t0 = time.time()
    solver.run(problem, [P.x0] * (N + 1), [np.zeros(P.nu)] * N)
    r = solver.results
    print('ProxDDP native standing: conv=%s iters=%d time=%.0fms primal_inf=%.1e dual_inf=%.1e' % (
        r.conv, r.num_iters, (time.time() - t0) * 1e3, r.primal_infeas, r.dual_infeas))
    xs = np.array(r.xs)
    print('base 드리프트=%.4fm 최종 base_z=%.3f(시작%.3f) 토크RMS=%.1f' % (
        np.linalg.norm(xs[-1][:3] - P.x0[:3]), xs[-1][2], P.x0[2],
        np.sqrt(np.mean(np.array(r.us) ** 2))))
    return r.conv


def solve_trot_oneshot(robot='go2', V=0.0, settle_steps=15, n_cycle=3, dt=0.02,
                       T=0.5, sf=0.5, step_h=0.06, maxiter=100):
    """① 검증: one-shot gait 계획(MPC 아님) + maxiter=100 + 예제 솔버옵션.
       계획 궤적이 안정적이면 → MPC 루프(maxiter8/warm-start)가 문제였음 확정."""
    import time
    P = ProxModel(robot)
    N = settle_steps + n_cycle * round(T / dt)
    x_ref_fwd = P.x0.copy(); x_ref_fwd[P.nq + 0] = V
    stages = []
    for k in range(N):
        tg = (k - settle_steps) * dt
        if tg < 0:
            stages.append(P.stage(P.legs, P.x0, dt))
        else:
            march = V * tg
            st, sw = QN._trot_schedule(tg, T, sf, step_h, P.foot_home, P.legs, P.diag, V, march=march)
            stages.append(P.stage(st, x_ref_fwd, dt, swing=sw))
    problem = aligator.TrajOptProblem(P.x0, stages, P.terminal_cost(x_ref_fwd))
    solver = aligator.SolverProxDDP(1e-5, 1e-8, max_iters=maxiter)
    for opt, val in [('rollout_type', getattr(aligator, 'ROLLOUT_LINEAR', None)),
                     ('sa_strategy', getattr(aligator, 'SA_FILTER', None))]:
        try:
            if val is not None:
                setattr(solver, opt, val)
        except Exception:
            pass
    try:
        solver.filter.beta = 1e-5
    except Exception:
        pass
    solver.setup(problem)
    t0 = time.time()
    solver.run(problem, [P.x0] * (N + 1), [np.zeros(P.nu)] * N)
    r = solver.results
    xs = np.array(r.xs)
    tilts = [np.degrees(np.arccos(np.clip(1 - 2 * (x[3] ** 2 + x[4] ** 2), -1, 1))) for x in xs]
    print('one-shot trot [%s V=%.1f]: conv=%s iters=%d time=%.0fms primal_inf=%.1e' % (
        robot, V, r.conv, r.num_iters, (time.time() - t0) * 1e3, r.primal_infeas))
    print('  계획 tilt_max=%.0f° base_z[%.3f,%.3f] 전진=%.2fm %s' % (
        max(tilts), min(x[2] for x in xs), max(x[2] for x in xs), xs[-1][0] - xs[0][0],
        '✅ 안정 계획' if max(tilts) < 20 and r.conv else '❌ 불안정/미수렴'))
    return r.conv


def track_oneshot(robot='go2', V=0.0, settle_steps=15, n_cycle=4, dt=0.02,
                  T=0.5, sf=0.5, step_h=0.06, maxiter=100, view=False, exec_robot=None,
                  reanchor=False):
    """one-shot 안정계획 + DDP 피드백 추종(재계획 없음). Go2는 pin=MuJoCo라 모델일치.
       exec_robot: MuJoCo 실행모델 분리(예: 계획='ours' pin, 실행='ours_sphere' 점접촉 정합).
       reanchor: 사이클 경계마다 남은 계획의 베이스 SE(2)를 로봇 실제 pose에 정렬(1단·드리프트 흡수)."""
    exec_robot = exec_robot or robot
    import sys, time, mujoco
    P = ProxModel(robot)
    N = settle_steps + n_cycle * round(T / dt)
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
    r = sol.results
    xs = [np.array(x) for x in r.xs]; us = [np.array(u) for u in r.us]
    Ks = [-np.array(k) for k in r.controlFeedbacks()]
    print('계획 conv=%s iters=%d → MuJoCo 추종 시작' % (r.conv, r.num_iters))
    # MuJoCo 추종
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        import quad_sim; quad_sim._ROBOT = exec_robot; q = quad_sim.QuadSim(); q.crouch_home()
    finally:
        sys.argv = _a
    m, d = q.m, q.d; pin2mj, mj_to_pin = QN._bridge(P.M, q)
    sub = max(1, round(dt / m.opt.timestep)); tmax = 0.0
    cyc = round(T / dt)
    ra_div = int(os.environ.get('RA_DIV', '0'))   # re-anchor 주기 = cyc/RA_DIV. 0=매스텝(최선·기본)
    ra_period = 1 if ra_div <= 0 else max(1, cyc // ra_div)
    gait_t0 = -settle_steps * dt                  # footstep 마커용 gait 위상 기준
    hud = _HUD(m, d, sub) if view else None
    for k in range(N):
        if hud and not hud.running():
            break
        # 1단: 사이클 경계마다 남은 계획을 로봇 실제 SE(2)에 re-anchor(드리프트 흡수)
        if reanchor and k > settle_steps and (k - settle_steps) % ra_period == 0:
            _se2_reanchor(xs, k, d.qpos[0], d.qpos[1], _yaw_xyzw(d.qpos[[4, 5, 6, 3]]))
        if hud: hud.pre_step()
        for _ in range(sub):
            x = mj_to_pin(d)
            dx = P.space.difference(xs[k], x)
            u = us[k] - Ks[k] @ dx
            umj = np.zeros(P.nu); umj[pin2mj] = u
            d.ctrl[:] = np.clip(umj, -TAU_LIM, TAU_LIM); mujoco.mj_step(m, d)
        if hud:
            gt = gait_t0 + k * dt
            mk = _footstep_markers(P, V, T, sf, gt, d.qpos[0], d.qpos[1],
                                   _yaw_xyzw(d.qpos[[4, 5, 6, 3]])) if gt >= 0 else ()
            hud.post_step(mk)
        xx, yy = d.qpos[4], d.qpos[5]
        tmax = max(tmax, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))))
        if os.environ.get('TRACE') and k % 5 == 0:
            til = np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1)))
            yaw = np.degrees(_yaw_xyzw(d.qpos[[4, 5, 6, 3]]))
            print('  k=%3d t=%.2f tilt=%4.1f yaw=%5.1f x=%.2f y=%+.3f z=%.3f' % (
                k, k * dt, til, yaw, d.qpos[0], d.qpos[1], d.qpos[2]), flush=True)
        if d.qpos[2] < 0.15:
            print('  추종 ❌ 전복 @plan스텝%d (%.2fs)' % (k, k * dt));
            if hud: hud.close()
            return
    if hud: hud.close()
    print('  추종 ✅ 계획끝까지 생존 base_z=%.3f tilt_max=%.0f° 전진=%.2fm' % (
        d.qpos[2], tmax, d.qpos[0]))


def _yaw_xyzw(q):
    x, y, z, w = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _qmul_xyzw(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def _se2_reanchor(xs, k0, rx, ry, ref_yaw):
    """xs[k0:] 의 베이스 위치(x,y)를 로봇 (rx,ry)에, 베이스 yaw 기준을 ref_yaw 로 정렬.
       속도는 pinocchio free-flyer local-frame 이라 re-anchor 불변(건드리지 않음).
       ref_yaw=robot_yaw → 완전 정렬(드리프트 중립). ref_yaw=heading명령 → heading 능동보정(직진/조향)."""
    px, py = xs[k0][0], xs[k0][1]
    pyaw = _yaw_xyzw(xs[k0][3:7])
    dpsi = ref_yaw - pyaw
    c, s = np.cos(dpsi), np.sin(dpsi)
    qz = np.array([0.0, 0.0, np.sin(dpsi / 2), np.cos(dpsi / 2)])
    for j in range(k0, len(xs)):
        dx, dy = xs[j][0] - px, xs[j][1] - py
        xs[j][0] = rx + c * dx - s * dy
        xs[j][1] = ry + s * dx + c * dy
        xs[j][3:7] = _qmul_xyzw(qz, xs[j][3:7])


# ══════════════════════════════════════════════════════════════
# 조작기(teleop) 명령 인터페이스 — 컨트롤러는 read()→(v,w)만 사용.
# 소스 교체로 시뮬↔실배포 동일: KeyboardCmd(뷰어) / JsonCmd(외부GUI·테스트) / RosTwistCmd(배포).
# ══════════════════════════════════════════════════════════════
class CommandSource:
    """전진 v[m/s] + 선회 w[rad/s] 명령. read()→(v,w). 하위클래스가 소스별 구현."""
    def __init__(self, v0=0.0, vmax=0.5, wmax=0.25):
        self.v, self.w, self.vmax, self.wmax = v0, 0.0, vmax, wmax
        self.vy = 0.0          # 측방(strafe) 선속도[m/s] — 전방향 조작용(기본 0)

    def read(self):
        return self.v, self.w

    def key(self, kc):                       # 키보드 소스만 사용(뷰어 key_callback 위임)
        return False


class KeyboardCmd(CommandSource):
    """MuJoCo 뷰어 키보드(화살표만 — WASD 는 뷰어 내장키와 충돌해 미사용):
       ↑/↓=전진±, ←/→=선회±, X=정지. 화살표 GLFW 코드(↑265 ↓264 ←263 →262)."""
    UP, DOWN, LEFT, RIGHT = 265, 264, 263, 262
    def key(self, kc):
        if kc == self.UP:
            self.v = min(self.vmax, self.v + 0.05)
        elif kc == self.DOWN:
            self.v = max(-self.vmax, self.v - 0.05)
        elif kc == self.LEFT:
            self.w = min(self.wmax, self.w + 0.1)
        elif kc == self.RIGHT:
            self.w = max(-self.wmax, self.w - 0.1)
        elif kc == ord('X'):
            self.v = self.w = 0.0
        else:
            return False
        print('[조작기] v_cmd=%+.2f m/s  w_cmd=%+.2f rad/s' % (self.v, self.w))
        return True


class JsonCmd(CommandSource):
    """외부 GUI/테스트가 JSON 파일에 {\"v\":..,\"w\":..} 기록 → 매 read 시 폴링.
       (배포 전 비ROS 환경에서 웹/Qt UI 연동, 또는 스크립트 자동주행 테스트용.)"""
    def __init__(self, path='/tmp/quad_cmd.json', v0=0.0, vmax=0.5, wmax=0.25):
        super().__init__(v0, vmax, wmax); self.path = path

    def read(self):
        try:
            import json
            with open(self.path) as f:
                d = json.load(f)
            self.v = float(np.clip(d.get('v', self.v), -self.vmax, self.vmax))
            self.w = float(np.clip(d.get('w', self.w), -self.wmax, self.wmax))
            self.vy = float(np.clip(d.get('vy', self.vy), -self.vmax, self.vmax))
        except Exception:
            pass
        return self.v, self.w


def make_cmd_source(kind, v0=0.0):
    """--cmd {key,json,ros} → CommandSource. ros 는 배포시 rclpy /cmd_vel(Twist) 구독(자리)."""
    if kind == 'json':
        return JsonCmd(v0=v0)
    if kind == 'ros':
        raise NotImplementedError('RosTwistCmd: 배포시 rclpy 로 /cmd_vel(Twist) 구독 구현 '
                                  '(linear.x=v, angular.z=w). conda proxddp env 에 rclpy 필요.')
    return KeyboardCmd(v0=v0)


class _HUD:
    """ProxDDP 뷰어 공통 HUD: 좌상단 sim time, 우상단 외력 N, 하단 조작기 명령,
       [/] 재생속도, Ctrl+드래그 외력, 다음 footstep 빨간점. 명령은 CommandSource(cmd) 위임."""

    def __init__(self, m, d, sub=1, cmd=None, v0=0.0, vmax=0.5, wmax=0.25):
        import mujoco.viewer as mjv
        self.m, self.d, self.speed, self.sub = m, d, 1.0, sub
        self.cmd = cmd if cmd is not None else KeyboardCmd(v0, vmax, wmax)  # 조작기 명령원
        self.v = mjv.launch_passive(m, d, key_callback=self._key)
        self.v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 1     # 외력 박스 표시
        print('조작기: ↑/↓=전진±  ←/→=선회±  X=정지 | [/]=재생속도  Ctrl+드래그=외력  창닫기=종료')

    def _key(self, kc):
        if kc == ord(']'):
            self.speed = min(16.0, self.speed * 2); print('[viewer] 재생속도 x%.3g' % self.speed)
        elif kc == ord('['):
            self.speed = max(1.0 / 16, self.speed / 2); print('[viewer] 재생속도 x%.3g' % self.speed)
        else:
            self.cmd.key(kc)                    # 조작기 키는 CommandSource 에 위임

    def running(self):
        return self.v.is_running()

    def pre_step(self):
        # Ctrl+드래그 외력 — active(드래그 중)일 때만 적용. 아니면 잔류 외력 0 클리어.
        if self.v.perturb.active:
            mujoco.mjv_applyPerturbForce(self.m, self.d, self.v.perturb)
        else:
            self.d.xfrc_applied[:] = 0.0

    def post_step(self, markers=(), arrow=None):
        m, d, v = self.m, self.d, self.v
        scn = v.user_scn; scn.ngeom = 0
        eye = np.eye(3).flatten()
        for p in markers:                                              # 다음 footstep(빨강 지면구)
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.03, 0, 0]), np.array([p[0], p[1], 0.012]),
                                eye, np.array([1, 0.1, 0.1, 0.9], np.float32))
            scn.ngeom += 1
        if arrow is not None and scn.ngeom < scn.maxgeom:             # 조작 방향(로봇 위 노랑 화살표)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                eye, np.array([1, 0.85, 0.1, 1.0], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012,
                                 np.asarray(arrow[0], float), np.asarray(arrow[1], float))
            scn.ngeom += 1
        fext = max((float(np.linalg.norm(d.xfrc_applied[b, :3]))
                    for b in range(1, m.nbody)), default=0.0)
        v.set_texts([
            (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT,
             'sim time', '%.2f s  (x%.3g)' % (d.time, self.speed)),
            (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
             'ext force', '%.0f N' % fext),
            (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
             'cmd  v / w', '%+.2f m/s  %+.2f rad/s' % (self.cmd.v, self.cmd.w))])
        v.sync()
        slp = m.opt.timestep * self.sub / self.speed                  # 제어스텝당 1회 호출 기준
        if slp > 0:
            time.sleep(slp)

    def close(self):
        self.v.close()


def _footstep_markers(P, V, T, sf, gait_t, bx, by, byaw):
    """현재 swing 다리의 다음 착지점(world xy). 로봇 베이스 SE(2)에 body-relative 발 nominal +
       전방 스텝을 회전·평행이동. 시각화용(아직 Raibert 보정 전)."""
    c, s = np.cos(byaw), np.sin(byaw)
    out = []
    for L in P.legs:
        ph = (gait_t / T + P.diag[L]) % 1.0
        if ph >= sf:                                # stance — 표시 안 함
            continue
        off = P.foot_home[L] - P.x0[:3]             # body-relative 발 nominal(평지 가정)
        lx = off[0] + V * T * sf * 0.5              # 착지 전방 스텝
        ly = off[1]
        out.append((bx + c * lx - s * ly, by + s * lx + c * ly))
    return out


def walk_loop(robot='go2', V=0.3, total_T=30.0, n_warm=6, dt=0.02, T=0.5, sf=0.5,
              step_h=0.06, settle_steps=15, maxiter=100, view=False, exec_robot=None,
              cmd_source=None):
    """★ 무한 연속보행: 정상상태 1사이클(us/Ks/xs)을 저장해 매스텝 SE(2) re-anchor로 무한 반복.
       절대 x,y,yaw 를 보지 않고 상대(높이·자세·관절·속도)만 추종 → 재최적화·드리프트 누적 없음.
       cmd_source: 조작기 명령원(CommandSource). 주면 그 v/w 명령 추종(뷰어 없이 teleop 테스트 가능),
       없으면 뷰어=키보드 / 헤드리스=자동 직선유지."""
    exec_robot = exec_robot or robot
    import sys, time, mujoco
    P = ProxModel(robot)
    cyc = round(T / dt)
    # ── warm 계획(settle + n_warm 사이클) 풀어 정상상태 1사이클 추출 ──
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
    r = sol.results
    xs = [np.array(x) for x in r.xs]; us = [np.array(u) for u in r.us]
    Ks = [-np.array(k) for k in r.controlFeedbacks()]
    if int(os.environ.get('PERIODIC', '0')):       # [실험] 하드 주기성 — pinocchio 수렴하나 MuJoCo 추종 깨짐(1s 전복)
        # ★ 주기적 사이클: 중간(대칭) 사이클 phase-0 을 시작점으로 주기성 하드제약
        #   x(cyc)=shift(x0)[base x만 +V·T] 로 단일사이클 재최적화 → 대칭+주기 둘 다 만족(쏠림·limp 제거).
        b_mid = settle_steps + (n_warm // 2) * cyc
        x_start = xs[b_mid].copy()
        cyc_xs, cyc_us, cyc_Ks, pconv = _solve_periodic_cycle(
            P, x_start, V, T, sf, step_h, dt, cyc, maxiter)
        print('계획 conv=%s → 주기사이클 재최적화 conv=%s(cyc=%d) → 무한루프 추종' % (r.conv, pconv, cyc))
    else:
        ec = int(os.environ.get('EXTRACT_CYC', str(n_warm - 1)))
        b = settle_steps + max(1, min(ec, n_warm - 1)) * cyc
        cyc_xs = [xs[b + i].copy() for i in range(cyc)]
        cyc_us = [us[b + i].copy() for i in range(cyc)]
        cyc_Ks = [Ks[b + i].copy() for i in range(cyc)]
        print('계획 conv=%s → 사이클 추출(b=%d,cyc=%d) → 무한루프 추종' % (r.conv, b, cyc))
    # ── MuJoCo 무한루프 추종 ──
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        import quad_sim; quad_sim._ROBOT = exec_robot; q = quad_sim.QuadSim(); q.crouch_home()
    finally:
        sys.argv = _a
    m, d = q.m, q.d; pin2mj, mj_to_pin = QN._bridge(P.M, q)
    sub = max(1, round(dt / m.opt.timestep)); tmax = 0.0
    nsteps = int(total_T / dt); t0 = time.time()
    # ── 상위 명령 루프(조작기 인터페이스). 명령 = 전진 v_cmd[m/s] + 선회 w_cmd[rad/s].
    #    뷰어=운영자 키(WASD)로 v_cmd/w_cmd, 헤드리스=자동 직선유지(steer-to-line, w 자동).
    #    heading 적분 yaw_ref → 2D 전진 carrot(yaw_ref 방향) + lag 포화. Raibert 없이 heading 권한만. ──
    K_LAT = float(os.environ.get('K_LAT', '1.2'))      # (헤드리스 자동) 측방 위치 → 조향 게인
    K_LATD = float(os.environ.get('K_LATD', '0.4'))    # (헤드리스 자동) 측방 속도 댐핑
    YAW_MAX = float(os.environ.get('YAW_MAX', '0.35'))  # (헤드리스 자동) 조향 한계(rad)
    MAXLAG = float(os.environ.get('MAXLAG', '0.06'))   # 후방 lag 포화(m) → 위치유지·runaway 방지
    LOOK_T = float(os.environ.get('LOOK_T', '0.4'))    # 전방 carrot 룩어헤드(s) → push 속도비례
    LEAD_MAX = float(os.environ.get('LEAD_MAX', '0.1'))  # 전방 carrot 상한(m) → 고속 과push 방지
    hud = _HUD(m, d, sub, cmd=cmd_source, v0=V) if view else None
    active_cmd = hud.cmd if hud else cmd_source        # 활성 명령원(뷰어 키보드 / 외부 cmd_source)
    y_prev = 0.0
    yaw_ref = _yaw_xyzw(d.qpos[[4, 5, 6, 3]])          # 명령 heading(운영자 선회 / 자동 조향)
    ax, ay = float(d.qpos[0]), float(d.qpos[1])        # course-hold rail anchor(직진 시 cross-track 고정점)
    xtgt, ytgt = float(d.qpos[0]), float(d.qpos[1])    # 2D 전진 가상타깃
    for s in range(nsteps):
        if hud and not hud.running():
            break
        i = s % cyc
        rx, ry = d.qpos[0], d.qpos[1]
        # 명령 결정: 명령원=운영자/외부 teleop(+직진 course-hold), 없으면 헤드리스 자동 직선유지
        if active_cmd is not None:
            v_cmd, w_cmd = active_cmd.read()
            yaw_ref += w_cmd * dt                       # 운영자 선회(heading 적분)
            if abs(w_cmd) > 1e-6:                        # 선회 중엔 rail 리셋(course-hold 간섭 X)
                ax, ay = rx, ry
            cross = -np.sin(yaw_ref) * (rx - ax) + np.cos(yaw_ref) * (ry - ay)   # heading 수직 드리프트
            yaw_use = yaw_ref + float(np.clip(-K_LAT * cross, -YAW_MAX, YAW_MAX))  # 직진 course-hold(crab 제거)
        else:
            v_cmd = V
            y = d.qpos[1]; vy = (y - y_prev) / dt; y_prev = y
            yaw_ref = yaw_use = float(np.clip(-K_LAT * y - K_LATD * vy, -YAW_MAX, YAW_MAX))  # 자동 직선유지
        # 2D carrot: body-frame [v_cmd 전진, vy_cmd 측방] → world(yaw_use 회전) + lag 포화
        vy_cmd = getattr(active_cmd, 'vy', 0.0) if active_cmd is not None else 0.0
        cu, su = np.cos(yaw_use), np.sin(yaw_use)
        xtgt += (v_cmd * cu - vy_cmd * su) * dt
        ytgt += (v_cmd * su + vy_cmd * cu) * dt
        dx, dy = xtgt - rx, ytgt - ry
        dist = float(np.hypot(dx, dy))
        lead = max(MAXLAG, min(np.hypot(v_cmd, vy_cmd) * LOOK_T, LEAD_MAX))
        if dist > lead and dist > 1e-9:
            xtgt = rx + dx / dist * lead; ytgt = ry + dy / dist * lead
        target = cyc_xs[i].copy()
        _se2_reanchor([target], 0, xtgt, ytgt, yaw_use)
        if hud: hud.pre_step()
        for _ in range(sub):
            x = mj_to_pin(d)
            u = cyc_us[i] - cyc_Ks[i] @ P.space.difference(target, x)
            umj = np.zeros(P.nu); umj[pin2mj] = u
            d.ctrl[:] = np.clip(umj, -TAU_LIM, TAU_LIM); mujoco.mj_step(m, d)
        if hud:
            mk = _footstep_markers(P, V, T, sf, s * dt, d.qpos[0], d.qpos[1],
                                   _yaw_xyzw(d.qpos[[4, 5, 6, 3]]))
            # 조작 방향 화살표(로봇 위): 명령 heading yaw_ref 방향, 길이 ∝ |v_cmd|
            zc = d.qpos[2] + 0.18
            ln = 0.15 + 0.5 * abs(v_cmd)
            sgn = 1.0 if v_cmd >= 0 else -1.0
            frm = np.array([d.qpos[0], d.qpos[1], zc])
            to = frm + sgn * ln * np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
            hud.post_step(mk, arrow=(frm, to))
        xx, yy = d.qpos[4], d.qpos[5]
        tmax = max(tmax, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))))
        if os.environ.get('TRACE') and s % 25 == 0:
            print('  t=%.1f tilt=%4.1f x=%.2f y=%+.3f z=%.3f' % (
                s * dt, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))),
                d.qpos[0], d.qpos[1], d.qpos[2]), flush=True)
        if d.qpos[2] < 0.15:
            print('  무한루프 ❌ 전복 @%.2fs (x=%.2f, 벽시계%.0fs)' % (
                s * dt, d.qpos[0], time.time() - t0))
            if hud: hud.close()
            return
    if hud: hud.close()
    print('  무한루프 ✅ %.1fs 완주 base_z=%.3f tilt_max=%.0f° 전진=%.2fm(평균%.2fm/s) 벽시계%.0fs' % (
        total_T, d.qpos[2], tmax, d.qpos[0], d.qpos[0] / total_T, time.time() - t0))


def _solver_opts(sol):
    for opt, val in [('rollout_type', getattr(aligator, 'ROLLOUT_LINEAR', None)),
                     ('sa_strategy', getattr(aligator, 'SA_FILTER', None))]:
        try:
            if val is not None:
                setattr(sol, opt, val)
        except Exception:
            pass
    try:
        sol.filter.beta = 1e-5
    except Exception:
        pass
    return sol


def walk_continuous(robot='go2', V=0.0, T=0.5, sf=0.5, step_h=0.06, settle=0.3,
                    horizon_cycles=2, replan_steps=15, total_T=6.0, maxiter=60,
                    view=False, exec_robot=None):
    """연속보행 — 주기적 re-plan(replan_steps 계획스텝마다 완전수렴 재계획) + DDP 추종.
       매 스텝 재계획(불안정)이 아니라 사이클 단위 재계획으로 누적오차 보정."""
    import sys, time, mujoco
    exec_robot = exec_robot or robot
    P = ProxModel(robot)
    dt = 0.02
    Hn = round(horizon_cycles * T / dt)                  # 계획 호라이즌(스텝)
    x_ref_fwd = P.x0.copy(); x_ref_fwd[P.nq + 0] = V
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        import quad_sim; quad_sim._ROBOT = exec_robot
        q = quad_sim.QuadSim(); q.crouch_home()
    finally:
        sys.argv = _a
    m, d = q.m, q.d; pin2mj, mj_to_pin = QN._bridge(P.M, q)
    sub = max(1, round(dt / m.opt.timestep))

    def plan(gait_t0, x, xs_ws=None, us_ws=None):
        body_x = x[0]; stages = []
        for k in range(Hn):
            tg = gait_t0 + k * dt
            if tg < 0:
                stages.append(P.stage(P.legs, P.x0, dt))
            else:
                st, sw = QN._trot_schedule(tg, T, sf, step_h, P.foot_home, P.legs, P.diag,
                                           V, march=body_x + V * (k * dt))
                stages.append(P.stage(st, x_ref_fwd, dt, swing=sw))
        x_ref_T = P.x0 if (gait_t0 + Hn * dt) < 0 else x_ref_fwd
        prob = aligator.TrajOptProblem(x.copy(), stages, P.terminal_cost(x_ref_T))
        sol = _solver_opts(aligator.SolverProxDDP(1e-4, 1e-7, max_iters=maxiter))
        sol.setup(prob)
        sol.run(prob, xs_ws or [x] * (Hn + 1), us_ws or [np.zeros(P.nu)] * Hn)
        r = sol.results
        return ([np.array(s) for s in r.xs], [np.array(u) for u in r.us],
                [-np.array(k) for k in r.controlFeedbacks()])

    gait_t = -settle
    xs, us, Ks = plan(gait_t, mj_to_pin(d)); pidx = 0
    viewer = None
    if view:
        import mujoco.viewer as mjv; viewer = mjv.launch_passive(m, d)
    tmax = 0.0; t0 = time.time(); n_plan_steps = int(total_T / dt)
    for ps in range(n_plan_steps):
        if pidx >= replan_steps:
            # 이전 해를 pidx 만큼 shift 해 warm-start(부드러운 재계획)
            s = pidx
            xw = xs[s:] + [xs[-1]] * s; uw = us[s:] + [us[-1]] * s
            xs, us, Ks = plan(gait_t, mj_to_pin(d), xw[:Hn + 1], uw[:Hn]); pidx = 0
        for _ in range(sub):
            x = mj_to_pin(d)
            u = us[pidx] - Ks[pidx] @ P.space.difference(xs[pidx], x)
            umj = np.zeros(P.nu); umj[pin2mj] = u
            d.ctrl[:] = np.clip(umj, -TAU_LIM, TAU_LIM); mujoco.mj_step(m, d)
            if viewer:
                viewer.sync(); time.sleep(m.opt.timestep)
        pidx += 1; gait_t += dt
        xx, yy = d.qpos[4], d.qpos[5]
        tmax = max(tmax, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))))
        if d.qpos[2] < 0.15:
            if viewer:
                viewer.close()
            print('연속보행 ❌ 전복 @%.1fs (x=%.2f, 벽시계%.0fs)' % (ps * dt, d.qpos[0], time.time() - t0))
            return
    if viewer:
        viewer.close()
    print('연속보행 ✅ %.1fs 생존 base_z=%.3f tilt_max=%.0f° 전진=%.2fm(평균%.2fm/s, 벽시계%.0fs)' % (
        total_T, d.qpos[2], tmax, d.qpos[0], d.qpos[0] / total_T, time.time() - t0))


def _xref_pose(P, px, py, pyaw, v, w):
    """기준 베이스 pose(px,py,pyaw) + body 트위스트(전진 v, yaw rate w) 의 상태 레퍼런스."""
    xr = P.x0.copy()
    xr[0], xr[1] = px, py
    xr[3:7] = _qmul_xyzw(np.array([0.0, 0.0, np.sin(pyaw / 2), np.cos(pyaw / 2)]), P.x0[3:7])
    xr[P.nq + 0] = v        # body-forward 속도
    xr[P.nq + 5] = w        # yaw rate
    return xr


def _swing_turn(tg, T, sf, step_h, P, px, py, pyaw, v):
    """선회 반영 swing 발 타깃: body-relative nominal 발 + 전진스윙을 기준 pose(px,py,pyaw)로 회전·평행이동."""
    c, s = np.cos(pyaw), np.sin(pyaw)
    stance, swing = [], {}
    for L in P.legs:
        ph = (tg / T + P.diag[L]) % 1.0
        if ph < sf:
            sp = ph / sf
            arc = 4 * step_h * sp * (1 - sp)
            fwd = v * T * sf * (sp - 0.5)
            ox = P.foot_home[L][0] - P.x0[0] + fwd        # body-frame (전진 스윙 포함)
            oy = P.foot_home[L][1] - P.x0[1]
            swing[L] = np.array([px + c * ox - s * oy, py + s * ox + c * oy,
                                 P.foot_home[L][2] + arc])
        else:
            stance.append(L)
    return stance, swing


def _solve_periodic_cycle(P, x_start, V, T, sf, step_h, dt, cyc, maxiter=100):
    """중간(대칭) 사이클 phase-0 x_start 에서 단일 주기를 재최적화하되 **주기성 하드제약**
       x(cyc) = shift(x_start)[base x,y 만 한 사이클 전진, 나머지(높이·자세·관절·전 속도) 동일] 부과.
       → 대칭(중간사이클 유래) + 주기(loop 이음매 없음) 둘 다 만족 → 측방 쏠림·RL limp 제거."""
    px0, py0, pyaw0 = x_start[0], x_start[1], _yaw_xyzw(x_start[3:7])
    px, py, pyaw = px0, py0, pyaw0
    stages = []
    for k in range(cyc):
        xr = _xref_pose(P, px, py, pyaw, V, 0.0)
        st, sw = _swing_turn(k * dt, T, sf, step_h, P, px, py, pyaw, V)
        stages.append(P.stage(st, xr, dt, swing=sw))
        px += V * np.cos(pyaw) * dt; py += V * np.sin(pyaw) * dt
    prob = aligator.TrajOptProblem(x_start, stages, P.terminal_cost(_xref_pose(P, px, py, pyaw, V, 0.0)))
    x_tgt = x_start.copy()                              # 주기성 목표: base x,y 만 +V·T 전진
    x_tgt[0] = px0 + V * T * np.cos(pyaw0); x_tgt[1] = py0 + V * T * np.sin(pyaw0)
    res = aligator.StateErrorResidual(P.space, P.nu, x_tgt)
    prob.addTerminalConstraint(res, acon.EqualityConstraintSet())
    sol = _solver_opts(aligator.SolverProxDDP(1e-4, 1e-7, max_iters=maxiter))
    sol.setup(prob); sol.run(prob, [x_start] * (cyc + 1), [np.zeros(P.nu)] * cyc)
    rr = sol.results
    return ([np.array(x) for x in rr.xs], [np.array(u) for u in rr.us],
            [-np.array(k) for k in rr.controlFeedbacks()], rr.conv)


def walk_replan(robot='go2', V=0.3, T=0.5, sf=0.5, step_h=0.06, settle=0.3,
                horizon_cycles=2, replan_steps=12, total_T=30.0, dt=0.02, maxiter=40,
                view=False, exec_robot=None, cmd_source=None):
    """[실험·불안정] 재계획 연속보행: 현재 상태+명령(v,w)으로 짧은 호라이즌 ProxDDP 재계획 →
       DDP 추종 → replan_steps 마다 warm-start 재계획. 선회를 계획에 반영(_swing_turn).
       ※ 검증결과 전환 transient 로 불안정(replan_steps 6/12/25 모두 <3s 전복) — 매 재계획 시
       us/Ks 가 바뀌며 제어 점프. = 메모리 기록된 receding-MPC 불안정 재확인. 안정 경로는
       walk_loop(plan-once + 매스텝 re-anchor). 진정한 재계획은 실시간 RTI(단일 뉴턴스텝/틱)나 MJPC 필요.
       (turning-aware planning + warm-start 구조는 추후 RTI 개발 참조용으로 보존.)"""
    import sys, time, mujoco
    exec_robot = exec_robot or robot
    P = ProxModel(robot)
    Hn = round(horizon_cycles * T / dt)
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        import quad_sim; quad_sim._ROBOT = exec_robot; q = quad_sim.QuadSim(); q.crouch_home()
    finally:
        sys.argv = _a
    m, d = q.m, q.d; pin2mj, mj_to_pin = QN._bridge(P.M, q)
    sub = max(1, round(dt / m.opt.timestep))
    K_LAT = float(os.environ.get('K_LAT', '1.0')); K_LATD = float(os.environ.get('K_LATD', '0.3'))

    def plan(gait_t0, x, v_cmd, w_cmd, xs_ws=None, us_ws=None):
        px, py, pyaw = x[0], x[1], _yaw_xyzw(x[3:7])
        stages = []
        for k in range(Hn):
            tg = gait_t0 + k * dt
            xr = _xref_pose(P, px, py, pyaw, v_cmd, w_cmd)
            if tg < 0:
                stages.append(P.stage(P.legs, xr, dt))
            else:
                st, sw = _swing_turn(tg, T, sf, step_h, P, px, py, pyaw, v_cmd)
                stages.append(P.stage(st, xr, dt, swing=sw))
            px += v_cmd * np.cos(pyaw) * dt; py += v_cmd * np.sin(pyaw) * dt; pyaw += w_cmd * dt
        xr_T = _xref_pose(P, px, py, pyaw, v_cmd, w_cmd)
        prob = aligator.TrajOptProblem(x.copy(), stages, P.terminal_cost(xr_T))
        sol = _solver_opts(aligator.SolverProxDDP(1e-4, 1e-7, max_iters=maxiter))
        sol.setup(prob)
        sol.run(prob, xs_ws or [x] * (Hn + 1), us_ws or [np.zeros(P.nu)] * Hn)
        r = sol.results
        return ([np.array(s) for s in r.xs], [np.array(u) for u in r.us],
                [-np.array(k) for k in r.controlFeedbacks()])

    gait_t = -settle
    xs, us, Ks = plan(gait_t, mj_to_pin(d), V, 0.0); pidx = 0
    hud = _HUD(m, d, sub, cmd=cmd_source, v0=V) if view else None
    active_cmd = hud.cmd if hud else cmd_source
    y_prev = 0.0; tmax = 0.0; t0 = time.time(); nps = int(total_T / dt)
    for ps in range(nps):
        if hud and not hud.running():
            break
        if active_cmd is not None:
            v_cmd, w_cmd = active_cmd.read()
        else:                                              # 헤드리스 자동 직선유지(→ plan yaw rate)
            v_cmd = V
            y = d.qpos[1]; vy = (y - y_prev) / dt; y_prev = y
            w_cmd = float(np.clip(-K_LAT * y - K_LATD * vy, -0.25, 0.25))
        if pidx >= replan_steps:                           # warm-start 재계획(이전해 shift)
            sh = pidx
            xw = xs[sh:] + [xs[-1]] * sh; uw = us[sh:] + [us[-1]] * sh
            xs, us, Ks = plan(gait_t, mj_to_pin(d), v_cmd, w_cmd, xw[:Hn + 1], uw[:Hn]); pidx = 0
        if hud: hud.pre_step()
        for _ in range(sub):
            x = mj_to_pin(d)
            u = us[pidx] - Ks[pidx] @ P.space.difference(xs[pidx], x)
            umj = np.zeros(P.nu); umj[pin2mj] = u
            d.ctrl[:] = np.clip(umj, -TAU_LIM, TAU_LIM); mujoco.mj_step(m, d)
        pidx += 1; gait_t += dt
        xx, yy = d.qpos[4], d.qpos[5]
        tmax = max(tmax, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))))
        if hud:
            byaw = _yaw_xyzw(d.qpos[[4, 5, 6, 3]])
            mk = _footstep_markers(P, v_cmd, T, sf, gait_t, d.qpos[0], d.qpos[1], byaw)
            zc = d.qpos[2] + 0.18; ln = 0.15 + 0.5 * abs(v_cmd); sg = 1.0 if v_cmd >= 0 else -1.0
            frm = np.array([d.qpos[0], d.qpos[1], zc])
            to = frm + sg * ln * np.array([np.cos(byaw), np.sin(byaw), 0.0])
            hud.post_step(mk, arrow=(frm, to))
        if os.environ.get('TRACE') and ps % 25 == 0:
            print('  t=%.1f tilt=%4.1f x=%.2f y=%+.3f z=%.3f' % (
                ps * dt, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))),
                d.qpos[0], d.qpos[1], d.qpos[2]), flush=True)
        if d.qpos[2] < 0.15:
            print('  재계획 ❌ 전복 @%.2fs (x=%.2f, 벽시계%.0fs)' % (ps * dt, d.qpos[0], time.time() - t0))
            if hud: hud.close()
            return
    if hud: hud.close()
    print('  재계획 ✅ %.1fs 완주 base_z=%.3f tilt_max=%.0f° 전진=%.2fm(평균%.2fm/s) 벽시계%.0fs' % (
        total_T, d.qpos[2], tmax, d.qpos[0], d.qpos[0] / total_T, time.time() - t0))


def walk_rti(robot='go2', V=0.3, T=0.5, sf=0.5, step_h=0.06, settle=0.3,
             horizon_cycles=2, total_T=30.0, dt=0.02, warm_iters=20,
             view=False, exec_robot=None, cmd_source=None):
    """★ RTI(Real-Time Iteration) 연속보행: 영속 솔버 + 매 제어틱 뉴턴 1스텝만 + 호라이즌 1칸 shift.
       이산 재계획(walk_replan)의 transient·무한루프 seam 을 동시 해결 — 궤적이 연속 진화(이음매 없음).
       구조: prob.x0_init(현재상태) → sol.run(1 iter, warm) → us[0] 적용 → replaceStageCircular+cycleProblem.
       명령(v,w)은 호라이즌 끝 새 stage 의 기준 pose 에 반영(선회). aligator cycleProblem MPC 패턴."""
    import sys, time, mujoco
    exec_robot = exec_robot or robot
    P = ProxModel(robot)
    Hn = round(horizon_cycles * T / dt)
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        import quad_sim; quad_sim._ROBOT = exec_robot; q = quad_sim.QuadSim(); q.crouch_home()
    finally:
        sys.argv = _a
    m, d = q.m, q.d; pin2mj, mj_to_pin = QN._bridge(P.M, q)
    sub = max(1, round(dt / m.opt.timestep))
    K_LAT = float(os.environ.get('K_LAT', '1.0')); K_LATD = float(os.environ.get('K_LATD', '0.3'))

    def mk_stage(tg, px, py, pyaw, v, w):
        xr = _xref_pose(P, px, py, pyaw, v, w)
        if tg < 0:
            return P.stage(P.legs, xr, dt)
        st, sw = _swing_turn(tg, T, sf, step_h, P, px, py, pyaw, v)
        return P.stage(st, xr, dt, swing=sw)

    # ── 초기 호라이즌 구성(현재 pose 기준 직진) + cold 워밍업 solve ──
    x0 = mj_to_pin(d)
    px, py, pyaw = x0[0], x0[1], _yaw_xyzw(x0[3:7])
    gait_t = -settle; stages = []
    for k in range(Hn):
        stages.append(mk_stage(gait_t + k * dt, px, py, pyaw, V, 0.0))
        px += V * np.cos(pyaw) * dt; py += V * np.sin(pyaw) * dt
    end_px, end_py, end_pyaw = px, py, pyaw
    end_t = gait_t + Hn * dt
    prob = aligator.TrajOptProblem(x0, stages, P.terminal_cost(_xref_pose(P, px, py, pyaw, V, 0.0)))
    sol = _solver_opts(aligator.SolverProxDDP(1e-3, 1e-6, max_iters=warm_iters))
    sol.setup(prob); sol.run(prob, [x0] * (Hn + 1), [np.zeros(P.nu)] * Hn)
    rti_iters = int(os.environ.get('RTI_ITERS', '1'))   # 틱당 뉴턴스텝(1=정통 RTI, 2~5=완화)
    print('RTI 워밍업 conv=%s iters=%d → 매 틱 %d스텝 RTI 시작' % (
        sol.results.conv, sol.results.num_iters, rti_iters))
    sol.max_iters = rti_iters                           # ← RTI: 이후 매 틱 소수 뉴턴스텝

    hud = _HUD(m, d, sub, cmd=cmd_source, v0=V) if view else None
    active_cmd = hud.cmd if hud else cmd_source
    y_prev = 0.0; ax_, ay_ = float(d.qpos[0]), float(d.qpos[1])
    tmax = 0.0; t0 = time.time(); nps = int(total_T / dt)
    for ps in range(nps):
        if hud and not hud.running():
            break
        # 명령(운영자/외부 teleop + 직진 course-hold, 또는 헤드리스 자동직선)
        rx, ry = d.qpos[0], d.qpos[1]
        if active_cmd is not None:
            v_cmd, w_cmd = active_cmd.read()
            if abs(w_cmd) > 1e-6:
                ax_, ay_ = rx, ry
            cross = -np.sin(end_pyaw) * (rx - ax_) + np.cos(end_pyaw) * (ry - ay_)
            w_eff = w_cmd + float(np.clip(-K_LAT * cross, -0.25, 0.25)) if abs(w_cmd) < 1e-6 else w_cmd
        else:
            v_cmd = V
            y = d.qpos[1]; vy = (y - y_prev) / dt; y_prev = y
            w_eff = float(np.clip(-K_LAT * y - K_LATD * vy, -0.25, 0.25))
        # ── RTI: 현재상태로 x0 갱신 → 뉴턴스텝 → us[0] 적용 ──
        # cycleProblem 이 results 를 내부 shift → 그대로 warm-start 전달(수동 shift 금지=이중shift)
        prob.x0_init = mj_to_pin(d)
        sol.run(prob, list(sol.results.xs), list(sol.results.us))
        u0 = np.array(sol.results.us[0]); xs0 = np.array(sol.results.xs[0])
        K0 = -np.array(sol.results.controlFeedbacks()[0])
        if hud: hud.pre_step()
        for _ in range(sub):
            x = mj_to_pin(d)
            u = u0 - K0 @ P.space.difference(xs0, x)
            umj = np.zeros(P.nu); umj[pin2mj] = u
            d.ctrl[:] = np.clip(umj, -TAU_LIM, TAU_LIM); mujoco.mj_step(m, d)
        # ── 호라이즌 1칸 전진: 새 terminal stage(끝 pose 를 명령 twist 로 적분) ──
        gait_t += dt; end_t += dt
        end_px += v_cmd * np.cos(end_pyaw) * dt; end_py += v_cmd * np.sin(end_pyaw) * dt
        end_pyaw += w_eff * dt
        ns = mk_stage(end_t, end_px, end_py, end_pyaw, v_cmd, w_eff)
        prob.replaceStageCircular(ns)
        sol.cycleProblem(prob, ns.createData())
        xx, yy = d.qpos[4], d.qpos[5]
        tmax = max(tmax, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))))
        if hud:
            byaw = _yaw_xyzw(d.qpos[[4, 5, 6, 3]])
            mk = _footstep_markers(P, v_cmd, T, sf, gait_t, d.qpos[0], d.qpos[1], byaw)
            zc = d.qpos[2] + 0.18; ln = 0.15 + 0.5 * abs(v_cmd); sg = 1.0 if v_cmd >= 0 else -1.0
            frm = np.array([d.qpos[0], d.qpos[1], zc])
            to = frm + sg * ln * np.array([np.cos(byaw), np.sin(byaw), 0.0])
            hud.post_step(mk, arrow=(frm, to))
        if os.environ.get('TRACE') and ps % 25 == 0:
            print('  t=%.1f tilt=%4.1f x=%.2f y=%+.3f z=%.3f' % (
                ps * dt, np.degrees(np.arccos(np.clip(1 - 2 * (xx * xx + yy * yy), -1, 1))),
                d.qpos[0], d.qpos[1], d.qpos[2]), flush=True)
        if d.qpos[2] < 0.15:
            print('  RTI ❌ 전복 @%.2fs (x=%.2f, 벽시계%.0fs)' % (ps * dt, d.qpos[0], time.time() - t0))
            if hud: hud.close()
            return
    if hud: hud.close()
    print('  RTI ✅ %.1fs 완주 base_z=%.3f tilt_max=%.0f° 전진=%.2fm(평균%.2fm/s) 벽시계%.0fs' % (
        total_T, d.qpos[2], tmax, d.qpos[0], d.qpos[0] / total_T, time.time() - t0))


def mpc_loop(N=20, dt=0.02, T=0.5, sf=0.5, step_h=0.06, V=0.0, settle=0.6, robot="ours",
             maxiter=8, sim_T=4.0, view=True):
    """네이티브 ProxDDP receding-horizon closed-loop (hard constraint). MuJoCo 물리."""
    import sys, time
    import mujoco
    import mujoco.viewer as mjv
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        import quad_sim
        quad_sim._ROBOT = robot
        q = quad_sim.QuadSim(); q.crouch_home()
    finally:
        sys.argv = _a
    m, d = q.m, q.d
    P = ProxModel(robot)
    pin2mj, mj_to_pin = QN._bridge(P.M, q)
    sim_dt = m.opt.timestep
    x_ref_fwd = P.x0.copy(); x_ref_fwd[P.nq + 0] = V
    x_ref_hold = P.x0.copy()
    cur = dict(u0=np.zeros(P.nu), K0=np.zeros((P.nu, P.ndx)), x0=P.x0.copy())
    xs_ws = [P.x0] * (N + 1); us_ws = [np.zeros(P.nu)] * N
    gait_start = [settle]

    def resolve(t, x):
        body_x = x[0]
        stages = []
        for k in range(N):
            tg = (t + k * dt) - gait_start[0]
            if tg < 0:
                stages.append(P.stage(P.legs, x_ref_hold, dt))
            else:
                march = body_x + V * (k * dt)
                st, sw = QN._trot_schedule(tg, T, sf, step_h, P.foot_home, P.legs, P.diag, V, march=march)
                stages.append(P.stage(st, x_ref_fwd, dt, swing=sw))
        x_ref_T = x_ref_hold if (t + N * dt) < gait_start[0] else x_ref_fwd
        prob = aligator.TrajOptProblem(x.copy(), stages, P.terminal_cost(x_ref_T))
        sol = aligator.SolverProxDDP(1e-3, 1e-2, max_iters=maxiter)
        sol.setup(prob); sol.run(prob, xs_ws, us_ws)
        r = sol.results
        cur['u0'] = np.array(r.us[0]); cur['x0'] = np.array(r.xs[0])
        cur['K0'] = -np.array(r.controlFeedbacks()[0])     # 부호규약(croc 반대)
        return list(r.xs), list(r.us)

    def control(it):
        x = mj_to_pin(d)
        xs_n, us_n = resolve(it * sim_dt, x)
        xs_ws[:] = xs_n; us_ws[:] = us_n
        dx = P.space.difference(cur['x0'], x)
        u = cur['u0'] - cur['K0'] @ dx
        umj = np.zeros(P.nu); umj[pin2mj] = u
        d.ctrl[:] = np.clip(umj, -TAU_LIM, TAU_LIM)
        mujoco.mj_step(m, d)

    def reset(it):
        q.crouch_home()
        xs_ws[:] = [P.x0] * (N + 1); us_ws[:] = [np.zeros(P.nu)] * N
        cur['x0'] = P.x0.copy(); cur['u0'] = np.zeros(P.nu); cur['K0'] = np.zeros((P.nu, P.ndx))
        gait_start[0] = it * sim_dt + settle

    nsteps = int(sim_T / sim_dt)
    if view:
        print('ProxDDP native closed-loop 뷰어 (hard constraint, 전복시 리셋)')
        with mjv.launch_passive(m, d) as v:
            for it in range(nsteps):
                if not v.is_running():
                    break
                control(it)
                if d.qpos[2] < 0.18:
                    reset(it)
                x, y = d.qpos[4], d.qpos[5]
                ti = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
                ph = 'settle' if it * sim_dt < gait_start[0] else 'gait'
                v.set_texts((mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             'ProxDDP [%s]' % ph, 'z=%.2f tilt=%.0f' % (d.qpos[2], ti)))
                v.sync()
    else:
        tmax = 0.0; t0 = time.time()
        for it in range(nsteps):
            control(it)
            x, y = d.qpos[4], d.qpos[5]
            ti = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
            tmax = max(tmax, ti)
            if os.environ.get('TRACE') and it % 125 == 0:
                print('  t=%.2f tilt=%.0f base_z=%.3f x=%.2f' % (it * sim_dt, ti, d.qpos[2], d.qpos[0]), flush=True)
            if d.qpos[2] < 0.15:
                print('ProxDDP native ❌ 전복 @%.1fs (x=%.2f, 벽시계%.0fs)' % (
                    it * sim_dt, d.qpos[0], time.time() - t0)); return
        print('ProxDDP native ✅ %.1fs 생존 base_z=%.3f tilt_max=%.0f° 전진x=%.2fm (벽시계%.0fs)' % (
            sim_T, d.qpos[2], tmax, d.qpos[0], time.time() - t0))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    # track = 작동하는 표준경로(one-shot 안정계획 + DDP 추종). mpc = 폐기된 매스텝 재계획.
    ap.add_argument('--test', default='track',
                    choices=['stand', 'track', 'walk', 'loop', 'replan', 'rti', 'mpc'])
    ap.add_argument('--total-T', type=float, default=30.0)
    ap.add_argument('--robot', default='go2', choices=['ours', 'go2', 'ours_sphere'])
    ap.add_argument('--vel', type=float, default=0.0)
    ap.add_argument('--settle', type=float, default=0.6)
    ap.add_argument('--cycles', type=int, default=4)
    ap.add_argument('--gait-T', type=float, default=0.5)
    ap.add_argument('--step-h', type=float, default=0.06)
    ap.add_argument('--exec-robot', default=None, choices=[None, 'ours', 'go2', 'ours_sphere'])
    ap.add_argument('--reanchor', action='store_true', help='1단 SE(2) re-anchoring(드리프트 흡수)')
    ap.add_argument('--cmd', default='key', choices=['key', 'json', 'ros'],
                    help='조작기 명령원: key=뷰어키보드, json=/tmp/quad_cmd.json 폴링, ros=/cmd_vel(배포)')
    ap.add_argument('--noview', action='store_true')
    a = ap.parse_args()
    os.environ.setdefault('DISPLAY', ':0')
    if a.test == 'stand':
        solve_standing(robot=a.robot)
    elif a.test == 'track':                       # ★ 권장 경로
        track_oneshot(robot=a.robot, V=a.vel, n_cycle=a.cycles, T=a.gait_T,
                      step_h=a.step_h, view=not a.noview, exec_robot=a.exec_robot,
                      reanchor=a.reanchor)
    elif a.test == 'loop':                        # ★ 무한 연속보행 + 조작기(teleop)
        # key=뷰어키보드(기본). json=외부GUI/스크립트 폴링(헤드리스 teleop 가능). 자동직선=명령원없음.
        cs = make_cmd_source(a.cmd, v0=a.vel) if a.cmd != 'key' else None
        walk_loop(robot=a.robot, V=a.vel, T=a.gait_T, step_h=a.step_h, total_T=a.total_T,
                  view=not a.noview, exec_robot=a.exec_robot, cmd_source=cs)
    elif a.test == 'replan':                      # [실험·불안정] 주기 완전수렴 재계획
        cs = make_cmd_source(a.cmd, v0=a.vel) if a.cmd != 'key' else None
        walk_replan(robot=a.robot, V=a.vel, T=a.gait_T, step_h=a.step_h, total_T=a.total_T,
                    view=not a.noview, exec_robot=a.exec_robot, cmd_source=cs)
    elif a.test == 'rti':                         # ★ RTI 연속보행 + 조작기(매 틱 1스텝, seam·transient 해결)
        cs = make_cmd_source(a.cmd, v0=a.vel) if a.cmd != 'key' else None
        walk_rti(robot=a.robot, V=a.vel, T=a.gait_T, step_h=a.step_h, total_T=a.total_T,
                 view=not a.noview, exec_robot=a.exec_robot, cmd_source=cs)
    elif a.test == 'walk':
        walk_continuous(robot=a.robot, V=a.vel, T=a.gait_T, step_h=a.step_h,
                        view=not a.noview, exec_robot=a.exec_robot)
    else:
        mpc_loop(robot=a.robot, V=a.vel, settle=a.settle, view=not a.noview)

