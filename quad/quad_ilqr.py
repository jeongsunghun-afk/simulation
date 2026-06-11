"""02_Leg 전신 MPC — iLQR + MuJoCo 유한차분 미분 (Zhang et al. 2025, "Whole-Body MPC
   of Legged Robots with MuJoCo", arXiv 2503.04613).

핵심: **MuJoCo 자체가 동역학 모델** → MPC 모델 = 시뮬 = MuJoCo (접촉 불일치 없음).
  · 선형화 A,B = mujoco.mjd_transitionFD (유한차분)
  · 접촉/마찰은 MuJoCo 물리가 처리 → 제약 정의 불필요. 비용은 base 추종 + swing 발만.
  · pinocchio/crocoddyl/aligator 불필요 (시스템 python).

quad_sim.QuadSim(MuJoCo 모델·발 헬퍼) 재사용. 실행: python3 quad_ilqr.py --test stand
"""
import os
import sys
import time
import argparse
import numpy as np
import mujoco

sys.argv = sys.argv  # placeholder
_a = sys.argv; sys.argv = [_a[0]]
try:
    import quad_sim
finally:
    sys.argv = _a

TAU = 80.0


class ILQR:
    def __init__(self, robot='ours'):
        quad_sim._ROBOT = robot
        self.q = quad_sim.QuadSim(); self.q.crouch_home()
        self.m, self.d = self.q.m, self.q.d
        self.m.opt.timestep = float(os.environ.get('ILQR_DT', '0.01'))  # 계획 타임스텝(호라이즌 확보)
        self.nv = self.m.nv; self.nu = self.m.nu; self.nx = 2 * self.nv
        self.dt = self.m.opt.timestep
        # 기준 상태(crouch) 저장
        mujoco.mj_forward(self.m, self.d)
        self.qpos0 = self.d.qpos.copy(); self.qvel0 = self.d.qvel.copy()
        # 비용 가중 (tangent: [base_lin3, base_ang3, joints], [vel...])
        self.Wp = np.concatenate([[0, 0, 400.], [300., 300., 100.],
                                  [5.0] * self.nu])              # pos-tangent
        self.Wv = np.concatenate([[5., 5., 5.], [3., 3., 3.],
                                  [0.2] * self.nu])              # vel
        self.R = 1e-3 * np.ones(self.nu)                         # control reg
        self.W_sw = 5e3                                          # swing 발 추종
        self.Vdes = 0.0

    # ── 상태 setter/getter ──
    def set_state(self, qpos, qvel):
        self.d.qpos[:] = qpos; self.d.qvel[:] = qvel
        mujoco.mj_forward(self.m, self.d)

    def tangent(self, qpos, qvel, qpos_r, qvel_r):
        """x ⊖ x_ref (tangent 2nv)."""
        dq = np.zeros(self.nv)
        mujoco.mj_differentiatePos(self.m, dq, 1.0, qpos_r, qpos)   # qpos ⊖ qpos_r
        return np.concatenate([dq, qvel - qvel_r])

    # ── 비용 + 미분 (현재 self.d 상태, 제어 u, gait 스케줄) ──
    def cost(self, u, qpos_r, qvel_r, swing, terminal=False):
        nv, nu = self.nv, self.nu
        dx = self.tangent(self.d.qpos, self.d.qvel, qpos_r, qvel_r)
        W = np.concatenate([self.Wp, self.Wv])
        wt = 1.0 if not terminal else 50.0
        l = 0.5 * wt * np.sum(W * dx * dx)
        lx = wt * (W * dx)
        lxx = np.diag(wt * W)
        lu = np.zeros(nu); luu = np.zeros((nu, nu))
        if not terminal:
            l += 0.5 * np.sum(self.R * u * u)
            lu = self.R * u; luu = np.diag(self.R)
        # swing 발 추종 (Gauss-Newton)
        for L, tgt in swing.items():
            i = self.q.legs.index(L)
            p = self.q.foot_point(i); r = p - tgt
            J = self.q.foot_jac(i)                                # 3 x nv
            l += 0.5 * self.W_sw * r @ r
            lx[:nv] += self.W_sw * (J.T @ r)
            lxx[:nv, :nv] += self.W_sw * (J.T @ J)
        return l, lx, lu, lxx, luu

    # ── 동역학 선형화 (MuJoCo 유한차분) ──
    def linearize(self, qpos, qvel, u):
        self.set_state(qpos, qvel); self.d.ctrl[:] = u
        A = np.zeros((self.nx, self.nx)); B = np.zeros((self.nx, self.nu))
        mujoco.mjd_transitionFD(self.m, self.d, 1e-6, True, A, B, None, None)
        return A, B

    # ── 롤아웃 ──
    def rollout(self, x0q, x0v, U, sched):
        self.set_state(x0q, x0v)
        Xq = [x0q.copy()]; Xv = [x0v.copy()]; J = 0.0
        N = len(U)
        for k in range(N):
            qr, vr, sw = sched(k)
            J += self.cost(U[k], qr, vr, sw)[0]
            self.d.ctrl[:] = np.clip(U[k], -TAU, TAU)
            mujoco.mj_step(self.m, self.d)
            Xq.append(self.d.qpos.copy()); Xv.append(self.d.qvel.copy())
        qr, vr, sw = sched(N)
        self.set_state(Xq[N], Xv[N])
        J += self.cost(None, qr, vr, sw, terminal=True)[0]
        return Xq, Xv, J

    # ── iLQR 1회 (backward + forward line search) ──
    def iterate(self, x0q, x0v, U, sched, reg=1e-6):
        N = len(U)
        Xq, Xv, J0 = self.rollout(x0q, x0v, U, sched)
        # backward
        qr, vr, sw = sched(N)
        self.set_state(Xq[N], Xv[N])
        _, Vx, _, Vxx, _ = self.cost(None, qr, vr, sw, terminal=True)
        K = [None] * N; kff = [None] * N
        for k in range(N - 1, -1, -1):
            qr, vr, sw = sched(k)
            A, B = self.linearize(Xq[k], Xv[k], U[k])
            self.set_state(Xq[k], Xv[k])
            l, lx, lu, lxx, luu = self.cost(U[k], qr, vr, sw)
            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B + reg * np.eye(self.nu)
            Qux = B.T @ Vxx @ A
            try:
                Quu_i = np.linalg.inv(Quu)
            except np.linalg.LinAlgError:
                return U, J0, False
            Kk = -Quu_i @ Qux; kk = -Quu_i @ Qu
            K[k] = Kk; kff[k] = kk
            Vx = Qx + Kk.T @ Quu @ kk + Kk.T @ Qu + Qux.T @ kk
            Vxx = Qxx + Kk.T @ Quu @ Kk + Kk.T @ Qux + Qux.T @ Kk
            Vxx = 0.5 * (Vxx + Vxx.T)
        # forward line search
        for alpha in [1.0, 0.5, 0.25, 0.1, 0.03]:
            self.set_state(x0q, x0v)
            Un = [None] * N; Jn = 0.0
            for k in range(N):
                dx = self.tangent(self.d.qpos, self.d.qvel, Xq[k], Xv[k])
                u = U[k] + alpha * kff[k] + K[k] @ dx
                u = np.clip(u, -TAU, TAU)
                qr, vr, sw = sched(k)
                Jn += self.cost(u, qr, vr, sw)[0]
                Un[k] = u; self.d.ctrl[:] = u; mujoco.mj_step(self.m, self.d)
            qr, vr, sw = sched(N)
            Jn += self.cost(None, qr, vr, sw, terminal=True)[0]
            if Jn < J0:
                return Un, Jn, True
        return U, J0, False


def wbic_init(il, N):
    """WBIC 정지제어를 N스텝 롤아웃해 유지 토크열 초기추정(warm-start)."""
    il.set_state(il.qpos0, il.qvel0)
    U = []
    for _ in range(N):
        il.q.wbic_stance()
        U.append(il.d.ctrl[:il.nu].copy())
        mujoco.mj_step(il.m, il.d)
    base_z = il.d.qpos[2]
    return U, base_z


def solve_standing(N=20, iters=30):
    """standing iLQR 검증 — x0 유지. WBIC warm-start."""
    il = ILQR()
    qpos0, qvel0 = il.qpos0.copy(), il.qvel0.copy()
    sched = lambda k: (qpos0, qvel0, {})        # 항상 crouch 유지, swing 없음
    U, bz_wbic = wbic_init(il, N)               # WBIC warm-start
    print('  WBIC warm-start 롤아웃 끝 base_z=%.3f' % bz_wbic)
    t0 = time.time(); reg = 1e-3; it = -1
    for it in range(iters):
        U, J, ok = il.iterate(qpos0, qvel0, U, sched, reg=reg)
        reg = max(1e-6, reg * 0.5) if ok else min(1e3, reg * 4)
    Xq, Xv, Jf = il.rollout(qpos0, qvel0, U, sched)
    drift = np.linalg.norm(Xq[-1][:3] - qpos0[:3])
    print('iLQR standing: iters=%d time=%.0fms cost=%.3e base_z=%.3f(시작%.3f) 드리프트=%.4fm 토크RMS=%.1f' % (
        it + 1, (time.time() - t0) * 1e3, Jf, Xq[-1][2], qpos0[2], drift,
        np.sqrt(np.mean(np.array(U) ** 2))))


def mpc_loop(N=20, V=0.0, sim_T=2.0, iters_per=2, view=False):
    """iLQR receding-horizon MPC. 별도 real MjData에 매 스텝 U[0] 적용 + 재계획.
       standing(현재) — sched=crouch 유지. (gait 는 추후 sched 확장)"""
    il = ILQR(); il.Vdes = V
    qpos0, qvel0 = il.qpos0.copy(), il.qvel0.copy()
    sched = lambda k: (qpos0, qvel0, {})
    U, _ = wbic_init(il, N)                       # warm-start
    real = mujoco.MjData(il.m)
    real.qpos[:] = qpos0; real.qvel[:] = qvel0; mujoco.mj_forward(il.m, real)
    nsteps = int(sim_T / il.dt); reg = 1e-3; tmax = 0.0; t0 = time.time()
    for it in range(nsteps):
        rq, rv = real.qpos.copy(), real.qvel.copy()
        for _ in range(iters_per):
            U, J, ok = il.iterate(rq, rv, U, sched, reg=reg)
            reg = max(1e-6, reg * 0.5) if ok else min(1e2, reg * 4)
        real.ctrl[:] = np.clip(U[0], -TAU, TAU)   # MPC: 첫 제어 적용(dx=0이라 U[0])
        mujoco.mj_step(il.m, real)
        U = U[1:] + [U[-1].copy()]                # warm-start shift
        x, y = real.qpos[4], real.qpos[5]
        tmax = max(tmax, np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
        if real.qpos[2] < 0.15:
            print('iLQR MPC ❌ 전복 @%.2fs (벽시계%.0fs)' % (it * il.dt, time.time() - t0)); return
    print('iLQR MPC ✅ %.1fs 생존 base_z=%.3f tilt_max=%.0f° (벽시계%.0fs)' % (
        sim_T, real.qpos[2], tmax, time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', default='stand', choices=['stand', 'mpc'])
    a = ap.parse_args()
    if a.test == 'stand':
        solve_standing()
    else:
        mpc_loop()
