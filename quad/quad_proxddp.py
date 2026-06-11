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


class _HUD:
    """ProxDDP 뷰어 공통 HUD: 좌상단 sim time, 우상단 외력 N, [/] 속도조절,
       Ctrl+드래그 외력, 다음 footstep 빨간점."""

    def __init__(self, m, d, sub=1):
        import mujoco.viewer as mjv
        self.m, self.d, self.speed, self.sub = m, d, 1.0, sub
        self.v = mjv.launch_passive(m, d, key_callback=self._key)
        self.v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 1     # 외력 박스 표시
        print('viewer: [ 느리게 / ] 빠르게, Ctrl+드래그=외력, 창닫기=종료')

    def _key(self, kc):
        if kc == ord(']'):
            self.speed = min(16.0, self.speed * 2)
        elif kc == ord('['):
            self.speed = max(1.0 / 16, self.speed / 2)
        else:
            return
        print('[viewer] 재생속도 x%.3g' % self.speed)

    def running(self):
        return self.v.is_running()

    def pre_step(self):
        # Ctrl+드래그 외력 — active(드래그 중)일 때만 적용. 아니면 잔류 외력 0 클리어.
        if self.v.perturb.active:
            mujoco.mjv_applyPerturbForce(self.m, self.d, self.v.perturb)
        else:
            self.d.xfrc_applied[:] = 0.0

    def post_step(self, markers=()):
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
        fext = max((float(np.linalg.norm(d.xfrc_applied[b, :3]))
                    for b in range(1, m.nbody)), default=0.0)
        v.set_texts([
            (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT,
             'sim time', '%.2f s  (x%.3g)' % (d.time, self.speed)),
            (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
             'ext force', '%.0f N' % fext)])
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
              step_h=0.06, settle_steps=15, maxiter=100, view=False, exec_robot=None):
    """★ 무한 연속보행: 정상상태 1사이클(us/Ks/xs)을 저장해 매스텝 SE(2) re-anchor로 무한 반복.
       절대 x,y,yaw 를 보지 않고 상대(높이·자세·관절·속도)만 추종 → 재최적화·드리프트 누적 없음.
       (track_oneshot 의 매스텝 re-anchor 가 유한계획에서 검증된 것을 주기루프로 확장한 형태.)"""
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
    b = settle_steps + (n_warm - 1) * cyc          # 마지막(정상상태) 사이클 시작
    cyc_xs = [xs[b + i].copy() for i in range(cyc)]
    cyc_us = [us[b + i].copy() for i in range(cyc)]
    cyc_Ks = [Ks[b + i].copy() for i in range(cyc)]
    print('계획 conv=%s iters=%d → 정상사이클 추출(b=%d,cyc=%d) → 무한루프 추종' % (
        r.conv, r.num_iters, b, cyc))
    # ── MuJoCo 무한루프 추종 ──
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        import quad_sim; quad_sim._ROBOT = exec_robot; q = quad_sim.QuadSim(); q.crouch_home()
    finally:
        sys.argv = _a
    m, d = q.m, q.d; pin2mj, mj_to_pin = QN._bridge(P.M, q)
    sub = max(1, round(dt / m.opt.timestep)); tmax = 0.0
    nsteps = int(total_T / dt); t0 = time.time()
    # ── 상위 명령 루프: 직선(y=0) 추종. 측방 위치 → yaw 조향(steer-to-line, crab 제거),
    #    전진 carrot(고정 전방오프셋 → 일정 전진 push). Raibert 발 디딤 없이 heading 권한만 사용. ──
    K_LAT = float(os.environ.get('K_LAT', '1.2'))      # 측방 위치 → 조향 yaw 게인
    K_LATD = float(os.environ.get('K_LATD', '0.4'))    # 측방 속도 댐핑
    YAW_MAX = float(os.environ.get('YAW_MAX', '0.35'))  # 조향 한계(rad)
    MAXLAG = float(os.environ.get('MAXLAG', '0.06'))   # 후방 lag 포화(m) → 위치유지·runaway 방지
    LOOK_T = float(os.environ.get('LOOK_T', '0.4'))    # 전방 carrot 룩어헤드(s) → push 속도비례
    LEAD_MAX = float(os.environ.get('LEAD_MAX', '0.1'))  # 전방 carrot 상한(m) → 고속 과push 방지
    hud = _HUD(m, d, sub) if view else None
    y_prev = 0.0; xtgt = float(d.qpos[0])              # 전진 가상타깃(명령속도로 전진, V=0이면 위치유지)
    for s in range(nsteps):
        if hud and not hud.running():
            break
        i = s % cyc
        # 상위 명령 루프: 측방오차 y → 조향 yaw_cmd, 전진 = 가상타깃(속도전진 + lag 포화)
        y = d.qpos[1]; vy = (y - y_prev) / dt; y_prev = y
        yaw_cmd = float(np.clip(-K_LAT * y - K_LATD * vy, -YAW_MAX, YAW_MAX))
        xtgt += V * dt                                  # 비대칭 클립: 전방 carrot(속도비례·상한) / 후방 고정 lag
        lead = max(MAXLAG, min(V * LOOK_T, LEAD_MAX))    # lead 상한 → 고속서 과push 전복 방지
        xtgt = float(np.clip(xtgt, d.qpos[0] - MAXLAG, d.qpos[0] + lead))
        rx_ref = xtgt
        ry_ref = d.qpos[1] + MAXLAG * np.sin(yaw_cmd)   # 조향 방향으로 측방 carrot
        target = cyc_xs[i].copy()
        _se2_reanchor([target], 0, rx_ref, ry_ref, yaw_cmd)
        if hud: hud.pre_step()
        for _ in range(sub):
            x = mj_to_pin(d)
            u = cyc_us[i] - cyc_Ks[i] @ P.space.difference(target, x)
            umj = np.zeros(P.nu); umj[pin2mj] = u
            d.ctrl[:] = np.clip(umj, -TAU_LIM, TAU_LIM); mujoco.mj_step(m, d)
        if hud:
            mk = _footstep_markers(P, V, T, sf, s * dt, d.qpos[0], d.qpos[1],
                                   _yaw_xyzw(d.qpos[[4, 5, 6, 3]]))
            hud.post_step(mk)
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
    ap.add_argument('--test', default='track', choices=['stand', 'track', 'walk', 'loop', 'mpc'])
    ap.add_argument('--total-T', type=float, default=30.0)
    ap.add_argument('--robot', default='go2', choices=['ours', 'go2', 'ours_sphere'])
    ap.add_argument('--vel', type=float, default=0.0)
    ap.add_argument('--settle', type=float, default=0.6)
    ap.add_argument('--cycles', type=int, default=4)
    ap.add_argument('--gait-T', type=float, default=0.5)
    ap.add_argument('--step-h', type=float, default=0.06)
    ap.add_argument('--exec-robot', default=None, choices=[None, 'ours', 'go2', 'ours_sphere'])
    ap.add_argument('--reanchor', action='store_true', help='1단 SE(2) re-anchoring(드리프트 흡수)')
    ap.add_argument('--noview', action='store_true')
    a = ap.parse_args()
    os.environ.setdefault('DISPLAY', ':0')
    if a.test == 'stand':
        solve_standing(robot=a.robot)
    elif a.test == 'track':                       # ★ 권장 경로
        track_oneshot(robot=a.robot, V=a.vel, n_cycle=a.cycles, T=a.gait_T,
                      step_h=a.step_h, view=not a.noview, exec_robot=a.exec_robot,
                      reanchor=a.reanchor)
    elif a.test == 'loop':                        # ★ 무한 연속보행(정상사이클 + 매스텝 re-anchor)
        walk_loop(robot=a.robot, V=a.vel, T=a.gait_T, step_h=a.step_h, total_T=a.total_T,
                  view=not a.noview, exec_robot=a.exec_robot)
    elif a.test == 'walk':
        walk_continuous(robot=a.robot, V=a.vel, T=a.gait_T, step_h=a.step_h,
                        view=not a.noview, exec_robot=a.exec_robot)
    else:
        mpc_loop(robot=a.robot, V=a.vel, settle=a.settle, view=not a.noview)

