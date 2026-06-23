"""quad_sim — 실제 4족(02_Leg_UFDF_260610_7) MuJoCo 통합 테스트/제어.

biped wbic_balance.py 와 동일한 '단일 파일 + --mode' 관리 방식.
모델: quad_real.mjcf  (build_real_quad.py 로 생성).

  python3 quad_sim.py --mode view    # 정적 기립 (physics 정지, 자세 확인)
  python3 quad_sim.py --mode stand   # 중력 하 PD(+중력보상) 기립
  (향후) stance(WBIC) → lipm → march → mpc → nmpc
  ※ check(모델/동역학 정합)는 stance(WBIC) 동작 시 자동 검증되므로 별도 단계 없음

시각화: 구조3(02leg9_fulldynamics) 스타일 — footstep타겟·지지다각형·CoM투영·명령화살표·base/발궤적·마찰콘+GRF·텍스트. 항상표시.
"""
import os
import json
import sys
import math
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from qpsolvers import solve_qp

KP, KD, TAU_MAX = 80.0, 4.0, 80.0
MU, LAMZ_MIN = 0.6, 1.0          # 마찰계수(물리), 최소 수직지지력
# 제어기 마찰 안전마진: box pyramid(|λx|,|λy|≤μλz)는 대각서 원뿔보다 √2배 큼 →
# μ_ctrl = μ/√2 로 elliptic cone 안쪽에 내접시켜 보수화. (추후 SOCP 로 정식 elliptic 대체)
MU_MARGIN = float(os.environ.get('MU_MARGIN', '0.707'))
# 통합 WBC(legged_control식) swing 발끝 작업공간 PD 게인/가중
SW_KP = float(os.environ.get('SW_KP', '800.0'))
SW_KD = float(os.environ.get('SW_KD', '80.0'))
W_SW = float(os.environ.get('W_SW', '30.0'))
_NOLIMIT = False                 # --nolimit: 관절 한계 해제 (가동범위 테스트용)

# 로봇 설정 — 우리 모델 / Go2 (--robot 으로 선택). 다리 인덱스 0,3 / 1,2 = 대각쌍(둘 다 동일)
_HERE = os.path.dirname(os.path.abspath(__file__))
ROBOTS = {
    'ours': dict(mjcf=os.path.join(_HERE, 'quad_real.mjcf'),
                 legs=['HL', 'HR', 'FL', 'FR'], dof=4,
                 foot_body='{L}_foot_contact_link', hip_body='{L}_hip_link',
                 foot_kind='mesh', base_z0=0.52, foot_z0=0.02, mu=0.6),   # _9 고정앞발목: 0.52서 무한안정(0.42=앞피칭 전복)
    'go2':  dict(mjcf=os.path.join(_HERE, '..', 'mujoco_menagerie', 'unitree_go2', 'scene.xml'),
                 legs=['FL', 'FR', 'RL', 'RR'], dof=3,
                 foot_geom='{L}', hip_body='{L}_hip',
                 foot_kind='sphere', base_z0=0.30, foot_z0=0.02, mu=0.6),
    # 02_Leg 발을 sphere 충돌로 교체(발목 자세 무관 점접촉) — box 모서리 rocking 회피 검증용
    'ours_sphere': dict(mjcf=os.path.join(_HERE, 'quad_real_sphere.mjcf'),
                        legs=['HL', 'HR', 'FL', 'FR'], dof=4,
                        foot_geom='{L}_sphere', hip_body='{L}_hip_link',
                        foot_kind='sphere', base_z0=0.52, mu=0.6),   # _9 고정앞발목: 0.52서 무한안정(0.42=앞피칭 전복)
}
_ROBOT = 'ours'
MJCF = ROBOTS['ours']['mjcf']    # (하위호환)


class QuadSim:
    def __init__(self, mjcf=None):
        cfg = ROBOTS[_ROBOT]
        self.cfg = cfg
        self.legs = cfg['legs']; self.dof = cfg['dof']
        self.foot_kind = cfg['foot_kind']; self.base_z0 = cfg['base_z0']
        self.foot_z0 = cfg.get('foot_z0', 0.0)   # nominal 자세 발바닥 목표 z(접지=0). 과거 0.02→20mm 부양 버그
        global MU; MU = cfg['mu']
        self.m = mujoco.MjModel.from_xml_path(mjcf or cfg['mjcf'])
        if _NOLIMIT:
            self.m.jnt_limited[:] = 0    # 관절 한계 해제 (테스트용)
        else:
            # 관절 한계를 단단하게: 기본 soft 제약(solreflimit 시정수 0.02s)은 동적 충격에
            # 한계를 뚫음(최대 0.3rad). 시정수↓+impedance↑ 로 실제 기계 스톱처럼 강화.
            lim = self.m.jnt_limited.astype(bool)
            self.m.jnt_solref[lim] = [0.004, 1.0]              # 2*timestep, 빠른 복원
            self.m.jnt_solimp[lim] = [0.95, 0.99, 0.001, 0.5, 2.0]  # 높은 impedance
        self.d = mujoco.MjData(self.m)
        self.nv = self.m.nv
        self.nu = self.m.nu          # = 4*dof
        N = lambda kind, nm: mujoco.mj_name2id(self.m, kind, nm)
        self.hip_bid = [N(mujoco.mjtObj.mjOBJ_BODY, cfg['hip_body'].format(L=L))
                        for L in self.legs]
        # 다리별 qpos/qvel 인덱스 — 관절 이름으로 조회(비균일 DOF 대응: 앞발목 fixed면 3, 뒤 4)
        _JT = ['hip', 'thigh', 'calf', 'foot']
        self.legqp = []; self.legqv = []
        for L in self.legs:
            jids = [N(mujoco.mjtObj.mjOBJ_JOINT, '%s_%s_joint' % (L, jt)) for jt in _JT]
            jids = [j for j in jids if j >= 0]
            self.legqp.append([int(self.m.jnt_qposadr[j]) for j in jids])
            self.legqv.append([int(self.m.jnt_dofadr[j]) for j in jids])
        self.leg_dof = [len(qp) for qp in self.legqp]   # 다리별 DOF(비균일)
        # 발 접촉 geom/body + 접촉점 정의 (mesh=최저정점 / sphere=중심−반지름)
        self.foot_bid = [0] * 4; self.foot_gid = [0] * 4
        self.sole_off = [None] * 4; self.foot_r = [0.0] * 4
        for i, L in enumerate(self.legs):
            if self.foot_kind == 'mesh':
                fb = N(mujoco.mjtObj.mjOBJ_BODY, cfg['foot_body'].format(L=L))
                gid = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == fb
                       and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                       and self.m.geom_contype[g] != 0][0]
                mid = self.m.geom_dataid[gid]
                adr = self.m.mesh_vertadr[mid]; nvt = self.m.mesh_vertnum[mid]
                Vv = self.m.mesh_vert[adr:adr + nvt].reshape(-1, 3)
                vlow = Vv[np.argmin(Vv[:, 2])]
                Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, self.m.geom_quat[gid])
                self.sole_off[i] = self.m.geom_pos[gid] + Rg.reshape(3, 3) @ vlow
            else:  # sphere
                gid = N(mujoco.mjtObj.mjOBJ_GEOM, cfg['foot_geom'].format(L=L))
                fb = self.m.geom_bodyid[gid]; self.foot_r[i] = float(self.m.geom_size[gid][0])
            self.foot_bid[i] = fb; self.foot_gid[i] = gid
        # ★물리환경을 구조3(02leg9_fulldynamics_mujoco.py line54-58)과 동일하게 ──
        #   timestep(1kHz)·CONE(마찰콘)·STIFF(접촉강성 전 geom solref)·FRIC(마찰). 발 침투(soft 접촉) 해결 포함.
        self.m.opt.timestep = float(os.environ.get('TIMESTEP', '0.001'))    # 구조3 dt_simu=1kHz
        if os.environ.get('CONE'):
            self.m.opt.cone = int(os.environ['CONE'])                       # 0=pyramidal 1=elliptic
        _stiff = float(os.environ.get('STIFF', '0.005'))                    # 기본 단단(발 침투 방지). 0=끄기
        if _stiff > 0:
            self.m.geom_solref[:, 0] = _stiff; self.m.geom_solref[:, 1] = 1.0   # 전 geom(구조3 동일)
        if os.environ.get('FRIC'):
            self.m.geom_friction[:, 0] = float(os.environ['FRIC'])
        self.q_home = None
        self.com_ref = None
        self.last_lam = None
        self.foot_targets = [None, None, None, None]   # 다음 착지 목표 (시각화용)
        self.contact_state = [False, False, False, False]   # detect_contact 결과 (색칠용)
        # 발 접촉 geom (접촉 시 빨강 색칠용) + 원래 색 저장
        self.foot_geoms = [[self.foot_gid[i]] for i in range(4)]
        self._foot_rgba0 = {g: self.m.geom_rgba[g].copy()
                            for gs in self.foot_geoms for g in gs}
        # ── 시각화 궤적(구조3 02leg9_fulldynamics 스타일 통일) ──
        from collections import deque as _deque
        _tn = int(os.environ.get('TRAIL_N', '300'))
        self.cmd_v = np.zeros(6)                          # 명령속도(화살표·텍스트용; mode가 설정)
        self.cmd_mode = 'move'                            # 현재 모드(GUI; move/stand_up/stand_down)
        # per-joint Peak토크(QP 한계+클립용): jnt_actfrcrange = hip/thigh84·calf126·foot168, qpos/ctrl 순서 일치
        self._tau_peak = np.array([self.m.jnt_actfrcrange[j, 1] if self.m.jnt_actfrcrange[j, 1] > 0 else 1e8
                                   for j in range(self.m.njnt) if self.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE])
        self.base_trail = _deque(maxlen=_tn)              # base 궤적(마젠타)
        self.foot_trail = [_deque(maxlen=_tn) for _ in range(4)]   # 발별 궤적

    # ── 운동학/동역학 헬퍼 ────────────────────────────
    def fullM(self):
        M = np.zeros((self.nv, self.nv)); mujoco.mj_fullM(self.m, M, self.d.qM); return M

    def foot_point(self, i):
        if self.foot_kind == 'mesh':
            return self.d.xpos[self.foot_bid[i]] + \
                self.d.xmat[self.foot_bid[i]].reshape(3, 3) @ self.sole_off[i]
        return self.d.geom_xpos[self.foot_gid[i]] - np.array([0, 0, self.foot_r[i]])  # sphere 바닥

    def foot_jac(self, i):
        jp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, self.d, jp, None, self.foot_point(i), self.foot_bid[i])
        return jp

    def _foot_pt(self, i, dat):
        if self.foot_kind == 'mesh':
            return dat.xpos[self.foot_bid[i]] + \
                dat.xmat[self.foot_bid[i]].reshape(3, 3) @ self.sole_off[i]
        return dat.geom_xpos[self.foot_gid[i]] - np.array([0, 0, self.foot_r[i]])

    def ik_leg(self, i, p_tgt, seed):
        """다리 i 의 발끝을 p_tgt(world)로 보내는 관절각 (damped-LS, scratch data)."""
        if not hasattr(self, '_scratch'):
            self._scratch = mujoco.MjData(self.m)
        s = self._scratch; s.qpos[:] = self.d.qpos[:]; qj = seed.copy()
        for _ in range(10):
            s.qpos[self.legqp[i]] = qj
            mujoco.mj_kinematics(self.m, s)
            mujoco.mj_comPos(self.m, s)        # ★ mj_jac 전제: comPos(cdof) 없으면 J=0
            e = p_tgt - self._foot_pt(i, s)
            jp = np.zeros((3, self.nv))
            mujoco.mj_jac(self.m, s, jp, None, self._foot_pt(i, s), self.foot_bid[i])
            J = jp[:, self.legqv[i]]
            qj = qj + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
        return qj

    def crouch_home(self, base_z=None):
        """q=0(또는 keyframe) 넓은 발 위치 유지한 채 base 낮춰 무릎굽힘 → q_home/com_ref."""
        d, m = self.d, self.m
        base_z = base_z if base_z is not None else float(os.environ.get('BASE_Z0', self.base_z0))
        if m.nkey > 0:
            mujoco.mj_resetDataKeyframe(m, d, 0)   # 모델 keyframe(home) 있으면 사용
        else:
            d.qpos[:] = 0; d.qpos[3] = 1
        d.qpos[2] = 0.60; mujoco.mj_forward(m, d)
        foot_xy = [self.foot_point(i)[:2].copy() for i in range(4)]
        d.qpos[2] = base_z
        # ★뒷다리(4DoF) 발목을 REAR_ANKLE로 두고 hip/thigh/calf로 같은 발 위치 IK
        #   → 같은 발 궤적, 더 웅크린 자세(발목 꺾고 thigh 펴짐). 4-DoF 여유(redundancy) 해소.
        _ra = float(os.environ.get('REAR_ANKLE', '0.0'))
        if _ra != 0.0:
            for i in range(4):
                if self.leg_dof[i] == 4:
                    d.qpos[self.legqp[i][3]] = _ra
        for _ in range(300):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            for i in range(4):
                tgt = np.array([foot_xy[i][0], foot_xy[i][1], self.foot_z0])
                e = tgt - self.foot_point(i)
                vi = self.legqv[i][:3]; qi = self.legqp[i][:3]   # hip/thigh/calf만 — 뒷발 foot관절은 0(직선) 유지해 앞발(용접)과 대칭
                J = self.foot_jac(i)[:, vi]
                d.qpos[qi] += 0.5 * (J.T @ np.linalg.solve(
                    J @ J.T + 1e-4 * np.eye(3), e))
        mujoco.mj_forward(m, d)
        self.q_home = d.qpos[7:7 + self.nu].copy()
        fc = np.mean([self.foot_point(i)[:2] for i in range(4)], axis=0)
        self.com_ref = np.array([fc[0], fc[1], d.subtree_com[0][2]])
        return self.q_home

    # ── WBIC stance QP (4발 접촉정합 균형) ─────────────
    def wbic_stance(self, contacts=(0, 1, 2, 3)):
        d, m, nv = self.d, self.m, self.nv
        K = len(contacts); nz = nv + 3 * K
        sl = lambda k: slice(nv + 3 * k, nv + 3 * k + 3)
        M = self.fullM(); h = d.qfrc_bias.copy(); qv = d.qvel.copy()
        P = np.zeros((nz, nz)); g = np.zeros(nz)
        # CoM xyz task
        Jc = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
        a_com = np.array([120, 120, 200]) * (self.com_ref - d.subtree_com[0]) \
            - np.array([20, 20, 25]) * (Jc @ qv)
        for r in range(3):
            P[:nv, :nv] += np.outer(Jc[r], Jc[r]); g[:nv] -= a_com[r] * Jc[r]
        # base orientation (upright)
        oerr = np.zeros(3); mujoco.mju_quat2Vel(oerr, d.qpos[3:7], 1.0)
        a_ori = 150 * (-oerr) - 20 * qv[3:6]
        for j in range(3):
            P[3 + j, 3 + j] += 5.0; g[3 + j] -= 5.0 * a_ori[j]
        # posture (crouch 유지)
        a_post = 60 * (self.q_home - d.qpos[7:7 + self.nu]) - 5 * qv[6:6 + self.nu]
        for j in range(self.nu):
            P[6 + j, 6 + j] += 1.0; g[6 + j] -= a_post[j]
        P[:nv, :nv] += 1e-4 * np.eye(nv)
        for k in range(K):
            P[sl(k), sl(k)] += 1e-3 * np.eye(3)
        # equality: floating-base 6 + 디딘발 무가속
        Js = [self.foot_jac(c) for c in contacts]
        A = np.zeros((6, nz)); b = -h[0:6]; A[:, :nv] = M[0:6, :]
        for k, J in enumerate(Js):
            A[:, sl(k)] = -J[:, 0:6].T
        for J in Js:
            Ac = np.zeros((3, nz)); Ac[:, :nv] = J
            A = np.vstack([A, Ac]); b = np.concatenate([b, np.zeros(3)])
        # ineq: 마찰추 + λz≥min
        lb = np.full(nz, -1e8); ub = np.full(nz, 1e8); Gl = []; hl = []
        for k in range(K):
            o = nv + 3 * k; lb[o + 2] = LAMZ_MIN
            for sx, sy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                row = np.zeros(nz); row[o] = sx; row[o + 1] = sy; row[o + 2] = -MU * MU_MARGIN
                Gl.append(row); hl.append(0.0)
        P = 0.5 * (P + P.T) + 1e-8 * np.eye(nz)
        z = solve_qp(P, g, np.vstack(Gl), np.array(hl), A, b, lb, ub, solver='quadprog')
        if z is None:
            self.last_lam = None
            return None, False
        qdd = z[:nv]; lam = [z[sl(k)] for k in range(K)]
        tau = M[6:6 + self.nu, :] @ qdd + h[6:6 + self.nu]
        for k, J in enumerate(Js):
            tau -= J[:, 6:6 + self.nu].T @ lam[k]
        self.last_lam = lam
        d.ctrl[:] = np.clip(tau, -self._tau_peak, self._tau_peak)   # per-joint Peak로 안전클립(QP가 이미 존중)
        return tau, True

    # ── MPC (Linear Convex, gait_sim.controllers.mpc 재사용) + WBIC 추종 ──
    def compute_Icom(self):
        """현재 자세의 CoM 기준 복합 관성 (world)."""
        com = self.d.subtree_com[0]; I = np.zeros((3, 3))
        for b in range(1, self.m.nbody):
            ms = self.m.body_mass[b]
            if ms <= 0:
                continue
            r = self.d.xipos[b] - com
            Rb = self.d.ximat[b].reshape(3, 3)
            Ib = Rb @ np.diag(self.m.body_inertia[b]) @ Rb.T
            I += Ib + ms * (r @ r * np.eye(3) - np.outer(r, r))
        return I

    def setup_mpc(self):
        """gait_sim 4족 SRBD MPC 를 실제 4족용으로 연결 (모듈상수 override)."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _argv = sys.argv; sys.argv = [_argv[0]]
        try:
            import gait_sim.controllers.mpc as MPC
        finally:
            sys.argv = _argv
        mujoco.mj_forward(self.m, self.d)
        TOT = float(self.m.body_subtreemass[0]); Ic = self.compute_Icom()
        MPC.TOTAL_MASS = TOT; MPC.BODY_INERTIA = Ic; MPC._I_BODY = Ic
        MPC.MU_FRICTION = MU * MU_MARGIN   # elliptic cone 내접 마진(A안)
        MPC.MPC_Q = np.diag([100., 100., 1., 1., 1., 50., 0., 0., 1., 1., 1., 1., 0.])
        MPC.MPC_R = 1e-6 * np.eye(3)
        MPC.LAMZ_MIN = 1.0
        MPC.LAMZ_MAX = float(os.environ.get('LAMZ_MAX_K', '2.0')) * TOT * 9.81  # 발당 수직력 상한(동적 trot)
        MPC.N_MPC = int(os.environ.get('N_MPC', '14'))   # 호라이즌 0.28s — 선형MPC 최적(>0.4s면 고정R 가정 깨져 발산)
        self.MPC = MPC; self.N_MPC = MPC.N_MPC

    def body_x0(self):
        """MuJoCo 상태 → MPC body state(13). [r,p,y, px,py,pz, ωxyz, vxyz, g]."""
        d, m = self.d, self.m
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, d.qpos[3:7]); Rm = Rm.reshape(3, 3)
        pitch = math.asin(max(-1, min(1, -Rm[2, 0])))
        roll = math.atan2(Rm[2, 1], Rm[2, 2]); yaw = math.atan2(Rm[1, 0], Rm[0, 0])
        com = d.subtree_com[0]
        Jc = np.zeros((3, m.nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
        vcom = Jc @ d.qvel; omega_w = Rm @ d.qvel[3:6]
        return np.array([roll, pitch, yaw, com[0], com[1], com[2],
                         omega_w[0], omega_w[1], omega_w[2],
                         vcom[0], vcom[1], vcom[2], -9.81])

    def mpc_grf(self, x_ref, contact_sched=None):
        """MPC 호출 → lam_des (4,3). contact_sched=None 이면 전부 stance."""
        com = self.d.subtree_com[0].copy(); N = self.N_MPC
        cs = np.ones((N, 4), dtype=bool) if contact_sched is None else contact_sched
        fp = np.zeros((N, 4, 3))
        foot_rel = [self.foot_point(i) - com for i in range(4)]
        for k in range(N):
            for i in range(4):
                fp[k, i] = foot_rel[i]
        return self.MPC.mpc_qp_plan(self.body_x0(), cs, fp, x_ref_step=x_ref, ltv=True)

    def wbic_track(self, lam_des, contacts=(0, 1, 2, 3), w_lam=10.0, swing=None):
        """통합 WBC (legged_control 식): 단일 QP z=[q̈; λ_stance] → τ(전 관절).

        stance: floating-base EOM(등식) + 디딘발 무가속(등식) + 마찰추(부등식)
                + λ 가 lam_des(MPC) 추종(soft).
        swing : swing={leg:(p_des, v_des)} 주면 발끝 작업공간 PD
                accel = SW_KP(p*−p)+SW_KD(v*−v) 를 J q̈ ≈ accel 로 같은 QP에 soft task 통합.
                → swing 다리도 stance 와 일관된 토크로 풀려 별도 IK-PD 루프 불필요."""
        d, m, nv = self.d, self.m, self.nv
        swing = swing or {}
        K = len(contacts); nz = nv + 3 * K
        sl = lambda k: slice(nv + 3 * k, nv + 3 * k + 3)
        M = self.fullM(); h = d.qfrc_bias.copy(); qv = d.qvel.copy()
        P = np.zeros((nz, nz)); g = np.zeros(nz)
        # swing 발끝 작업공간 PD task (legged_control formulateSwingLegTask)
        sw_vidx = set()
        for leg, (p_des, v_des) in swing.items():
            J = self.foot_jac(leg)
            accel = SW_KP * (p_des - self.foot_point(leg)) + SW_KD * (v_des - J @ qv)
            P[:nv, :nv] += W_SW * (J.T @ J); g[:nv] -= W_SW * (J.T @ accel)
            sw_vidx.update(self.legqv[leg])
        # base 안정화 task (legged_control BaseAccelTask 해당): 자세(upright)+높이(z)
        #   — swing 반력에 의한 pitch/sink 억제. xy 병진은 MPC λ 에 위임(전진 보행 유지).
        Jc = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
        oerr = np.zeros(3); mujoco.mju_quat2Vel(oerr, d.qpos[3:7], 1.0)
        a_ori = 150 * (-oerr) - 20 * qv[3:6]
        w_ori = float(os.environ.get('W_ORI', '5.0'))
        for j in range(3):
            P[3 + j, 3 + j] += w_ori; g[3 + j] -= w_ori * a_ori[j]
        a_z = 200 * (self.com_ref[2] - d.subtree_com[0][2]) - 25 * (Jc @ qv)[2]
        w_z = float(os.environ.get('W_Z', '150.0'))   # 높이 유지 강화(nose-dive 방지 핵심)
        P[:nv, :nv] += w_z * np.outer(Jc[2], Jc[2]); g[:nv] -= w_z * a_z * Jc[2]
        # posture — swing 관절엔 약한 규제(w=0.1)만: 발끝 3D task가 다리 DOF<4 면 여유도
        #   (4-DOF 발목)를 남기므로 null-space 규제 없으면 발목이 발산(flail). 약규제로 안정화.
        a_post = 60 * (self.q_home - d.qpos[7:7 + self.nu]) - 5 * qv[6:6 + self.nu]
        for j in range(self.nu):
            w_post = 0.1 if (6 + j) in sw_vidx else 1.0
            P[6 + j, 6 + j] += w_post; g[6 + j] -= w_post * a_post[j]
        P[:nv, :nv] += 1e-3 * np.eye(nv)
        for k in range(K):           # λ tracking
            P[sl(k), sl(k)] += w_lam * np.eye(3); g[sl(k)] -= w_lam * lam_des[contacts[k]]
        Js = [self.foot_jac(c) for c in contacts]
        A = np.zeros((6, nz)); b = -h[0:6]; A[:, :nv] = M[0:6, :]
        for k, J in enumerate(Js):
            A[:, sl(k)] = -J[:, 0:6].T
        for J in Js:
            Ac = np.zeros((3, nz)); Ac[:, :nv] = J
            A = np.vstack([A, Ac]); b = np.concatenate([b, np.zeros(3)])
        lb = np.full(nz, -1e8); ub = np.full(nz, 1e8); Gl = []; hl = []
        for k in range(K):
            o = nv + 3 * k; lb[o + 2] = LAMZ_MIN
            for sx, sy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                row = np.zeros(nz); row[o] = sx; row[o + 1] = sy; row[o + 2] = -MU * MU_MARGIN
                Gl.append(row); hl.append(0.0)
        P = 0.5 * (P + P.T) + 1e-8 * np.eye(nz)
        # ★per-joint 토크 한계: −τ_peak ≤ τ ≤ τ_peak,  τ = M_act·q̈ + h_act − Σ Jᵀλ (z의 선형식)
        if os.environ.get('TAU_LIM', '1') != '0':
            h_act = h[6:6 + self.nu]
            T_mat = np.zeros((self.nu, nz)); T_mat[:, :nv] = M[6:6 + self.nu, :]
            for k, J in enumerate(Js):
                T_mat[:, sl(k)] = -J[:, 6:6 + self.nu].T
            Gl.extend(list(T_mat));  hl.extend(list(self._tau_peak - h_act))    # τ ≤ +peak
            Gl.extend(list(-T_mat)); hl.extend(list(self._tau_peak + h_act))    # −τ ≤ +peak
        G = np.vstack(Gl) if Gl else None; hh = np.array(hl) if hl else None
        z = solve_qp(P, g, G, hh, A, b, lb, ub, solver='quadprog')
        if z is None:
            self.last_lam = None; return None, False
        qdd = z[:nv]; lam = [z[sl(k)] for k in range(K)]
        tau = M[6:6 + self.nu, :] @ qdd + h[6:6 + self.nu]
        for k, J in enumerate(Js):
            tau -= J[:, 6:6 + self.nu].T @ lam[k]
        self.last_lam = lam
        d.ctrl[:] = np.clip(tau, -self._tau_peak, self._tau_peak)   # per-joint Peak로 안전클립(QP가 이미 존중)
        return tau, True

    # ── 제어 ──────────────────────────────────────────
    def set_home(self):
        mujoco.mj_forward(self.m, self.d)
        self.q_home = self.d.qpos[7:7 + self.nu].copy()

    def pd_grav(self):
        """중력보상 + home 자세 PD → ctrl."""
        q = self.d.qpos[7:7 + self.nu]
        qd = self.d.qvel[6:6 + self.nu]
        grav = self.d.qfrc_bias[6:6 + self.nu]
        tau = grav + KP * (self.q_home - q) - KD * qd
        self.d.ctrl[:] = np.clip(tau, -TAU_MAX, TAU_MAX)

    # ── 발 GRF (MuJoCo 접촉에서) ───────────────────────
    def foot_grf(self):
        """각 발의 world GRF (3,) — 시각화용. 반환 list[4] of (3,) or None."""
        d, m = self.d, self.m
        grf = [np.zeros(3) for _ in range(4)]
        hit = [False] * 4
        for c in range(d.ncon):
            con = d.contact[c]
            g1, g2 = con.geom1, con.geom2
            bod = None
            for gid in (g1, g2):
                bb = m.geom_bodyid[gid]
                if bb in self.foot_bid:
                    bod = self.foot_bid.index(bb)
            if bod is None:
                continue
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, c, f6)
            # contact frame → world (con.frame 는 3x3 row-major, x축=법선)
            R = con.frame.reshape(3, 3)
            fw = R.T @ f6[:3]
            grf[bod] += fw
            hit[bod] = True
        return [grf[i] if hit[i] else None for i in range(4)]

    # ── 발끝 접촉 감지 (Di Carlo 2018 MIT Cheetah 3 식) ──────────
    def foot_fz(self):
        """각 발 추정 수직 접촉력 fz (force-sensor proxy = MuJoCo 접촉력).
        실하드웨어: 발 힘센서 또는 관절토크 기반 추정기(τ→Jᵀ⁻¹)로 대체."""
        grf = self.foot_grf()
        return np.array([g[2] if g is not None else 0.0 for g in grf])

    def detect_contact(self, scheduled, fz_on=10.0, fz_off=4.0, z_thr=0.03, alpha=0.6):
        """발끝 접촉 감지 — 주기 gait 스케줄(Di Carlo) + 측정 융합 + 히스테리시스.

        scheduled[i] : gait 가 예측한 stance 여부 (primary, Di Carlo periodic schedule).
        측정 융합     : 수직 GRF fz(저역통과) + 발끝 높이 z 로 조기/지연 착지 보정.
        히스테리시스 : 접촉중이면 낮은 임계(fz_off)로 유지, 비접촉이면 높은 임계(fz_on)로
                      진입 → 임계 근처 채터링(출렁임) 방지.
        반환: contact(4,) bool, fz_filt(4,)."""
        if not hasattr(self, '_fz_filt'):
            self._fz_filt = np.zeros(4)
        self._fz_filt = alpha * self._fz_filt + (1 - alpha) * self.foot_fz()
        contact = []
        for i in range(4):
            near = self.foot_point(i)[2] < z_thr
            thr = fz_off if self.contact_state[i] else fz_on   # 히스테리시스
            loaded = self._fz_filt[i] > thr
            c = (loaded or near) if scheduled[i] else (loaded and near)
            contact.append(bool(c))
        self.contact_state = contact
        return contact, self._fz_filt.copy()

    # ── 뷰어 오버레이 ─────────────────────────────────
    def _key_callback(self, keycode):
        pass                                            # 시각화는 항상 표시(구조3 스타일). 토글 없음.

    def draw_overlay(self, v):
        # ★구조3(02leg9_fulldynamics_mujoco.py) 스타일로 시각화 통일:
        #   빨강구=타겟footstep(swing) / 청록선=지지다각형 / 노랑구+선=CoM지면투영 / 노랑화살표=명령방향
        #   마젠타=base궤적 / 파란콘+초록화살표=마찰콘+GRF / 발별색선=발궤적
        d = self.d; m = self.m
        scn = v.user_scn; scn.ngeom = 0; eye = np.eye(3).flatten()
        def _sph(p, r, c):
            if scn.ngeom >= scn.maxgeom: return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([r, 0, 0]), np.asarray(p, float), eye, np.asarray(c, np.float32))
            scn.ngeom += 1
        def _ln(a, b, w, c, typ=mujoco.mjtGeom.mjGEOM_LINE):
            if scn.ngeom >= scn.maxgeom: return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, typ, np.zeros(3), np.zeros(3), eye, np.asarray(c, np.float32))
            mujoco.mjv_connector(g, typ, w, np.asarray(a, float), np.asarray(b, float)); scn.ngeom += 1
        ZC = 0.02                                            # 발 접지판별(발끝 z<ZC=접지)
        fp = [self.foot_point(i).copy() for i in range(4)]; fz = [p[2] for p in fp]
        # 발 접촉링: 접지면 빨강
        RED = np.array([0.9, 0.1, 0.1, 1.0], np.float32)
        for i in range(4):
            for g in self.foot_geoms[i]:
                self.m.geom_rgba[g] = RED if fz[i] < ZC else self._foot_rgba0[g]
        # ── 타겟 footstep(swing 발만, 빨강구) ──
        for i in range(4):
            ft = self.foot_targets[i]
            if ft is not None and fz[i] > ZC:
                _sph([ft[0], ft[1], 0.008], 0.012, [1, 0.1, 0.1, 0.9])
        # ── 지지다각형(접지 발 연결, 청록 지면선) ──
        _ord = [2, 3, 1, 0]                                  # FL,FR,HR,HL 둘레순(quad_sim legs=HL,HR,FL,FR)
        sp = [fp[i] for i in _ord if fz[i] < ZC]
        for k in range(len(sp)):
            if len(sp) < 2: break
            a = sp[k].copy(); a[2] = 0.003; b = sp[(k + 1) % len(sp)].copy(); b[2] = 0.003
            _ln(a, b, 0.006, [0.1, 0.9, 0.9, 1])
        # ── CoM 지면투영(노랑구+수직선) ──
        com = d.subtree_com[0].copy()
        _sph([com[0], com[1], 0.004], 0.020, [1, 0.9, 0.1, 0.95]); _ln([com[0], com[1], 0.0], com, 0.003, [1, 0.9, 0.1, 0.6])
        # ── 명령방향 화살표(로봇 위 노랑) ──
        cv = self.cmd_v
        if float(np.hypot(cv[0], cv[1])) > 1e-3:
            frm = d.qpos[0:3].copy() + np.array([0, 0, 0.20]); to = frm + np.array([cv[0], cv[1], 0.0]) * 0.4
            _ln(frm, to, 0.015, [1, 0.85, 0.1, 1], mujoco.mjtGeom.mjGEOM_ARROW)
        # ── base 궤적(마젠타·굵게) ──
        self.base_trail.append(d.qpos[0:3].copy()); bp = self.base_trail
        for k in range(1, len(bp)):
            if np.linalg.norm(bp[k] - bp[k - 1]) < 1e-4: continue
            _ln(bp[k - 1], bp[k], 0.010, [1, 0.15, 0.9, 1])
        # ── 마찰콘(파랑) + GRF(초록 화살표): GRF가 콘 벗어나면 슬립 ──
        mu = float(os.environ.get('CONE_MU', str(m.geom_friction[self.foot_gid[0]][0])))
        hh = 0.10; Ncn = 8
        if not os.environ.get('NOCONE'):
            for i in range(d.ncon):
                c = d.contact[i]
                if c.geom1 not in self.foot_gid and c.geom2 not in self.foot_gid: continue
                p = c.pos.copy()
                for k in range(Ncn):
                    a1 = 2 * np.pi * k / Ncn; a2 = 2 * np.pi * (k + 1) / Ncn
                    r1 = np.array([np.cos(a1), np.sin(a1), 0]) * hh * mu + np.array([0, 0, hh])
                    r2 = np.array([np.cos(a2), np.sin(a2), 0]) * hh * mu + np.array([0, 0, hh])
                    _ln(p, p + r1, 0.0015, [0.3, 0.5, 1, 0.5]); _ln(p + r1, p + r2, 0.0015, [0.3, 0.5, 1, 0.5])
                f6 = np.zeros(6); mujoco.mj_contactForce(m, d, i, f6)
                fw = c.frame.reshape(3, 3).T @ f6[:3]
                if fw[2] < 0: fw = -fw
                mag = np.linalg.norm(fw)
                if mag > 1.0: _ln(p, p + fw / mag * min(mag / 250.0, 0.15), 0.008, [0.1, 1, 0.2, 1], mujoco.mjtGeom.mjGEOM_ARROW)
        # ── 발 궤적(발별 색선) ──
        _tc = [[0.2, 0.6, 1, 1], [0.2, 1, 0.4, 1], [1, 0.6, 0.2, 1], [1, 0.3, 0.85, 1]]
        for i in range(4):
            self.foot_trail[i].append(fp[i].copy()); pts = self.foot_trail[i]
            for k in range(1, len(pts)):
                if np.linalg.norm(pts[k] - pts[k - 1]) < 1e-4: continue
                _ln(pts[k - 1], pts[k], 0.004, _tc[i % 4])

    def publish_state(self, path):
        """구조3와 동일 스키마로 상태 발행(원자적) → teleop_gui 모니터 패널(IMU/Actuator)."""
        d, m = self.d, self.m
        if not hasattr(self, '_jnames'):                  # q/dq 순서와 일치(MuJoCo 관절순)
            self._jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).replace('_joint', '')
                            for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, d.qpos[3:7]); Rm = Rm.reshape(3, 3)
        rpy = [math.degrees(math.atan2(Rm[2, 1], Rm[2, 2])),
               math.degrees(math.asin(max(-1, min(1, -Rm[2, 0])))),
               math.degrees(math.atan2(Rm[1, 0], Rm[0, 0]))]
        st = {'mode': self.cmd_mode, 'base_z': float(d.qpos[2]), 't': float(d.time),
              'rpy': rpy, 'gyro': [float(x) for x in d.qvel[3:6]], 'names': self._jnames,
              'q': [float(x) for x in d.qpos[7:7 + self.nu]],
              'dq': [float(x) for x in d.qvel[6:6 + self.nu]],
              'tau': [float(x) for x in d.ctrl[:self.nu]]}
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(st, f)
        os.replace(tmp, path)

    def run_viewer(self, control_fn, reset_on_fall=True, reset_fn=None):
        m, d = self.m, self.d
        reset_fn = reset_fn or self.set_home
        _sp = os.environ.get('STATE_PUB'); _pc = 0   # 상태발행 채널 + 프레임카운터
        if os.environ.get('HEADLESS'):                       # 헤드리스 정량테스트(뷰어 없이 N스텝)
            nsteps = int(os.environ.get('STEPS', '1500'))
            pe = int(os.environ.get('PRINT_EVERY', '100'))   # 출력 간격(스텝). 첫사이클 관찰은 20 등
            falls = 0
            for s in range(nsteps):
                control_fn(); mujoco.mj_step(m, d)
                if reset_on_fall and d.qpos[2] < 0.2:
                    falls += 1; mujoco.mj_resetData(m, d); reset_fn()
                if _sp and s % 30 == 0: self.publish_state(_sp)   # GUI 모니터 패널
                if s % pe == 0:
                    w, x, y, z = d.qpos[3:7]                  # base quat [w,x,y,z] → tilt(수직과의 각)
                    tilt = np.degrees(np.arccos(max(-1, min(1, 1 - 2 * (x * x + y * y)))))
                    pen = [0.0] * 4                           # 발별 접촉침투(mm, 음수=파고듦)
                    for ci in range(d.ncon):
                        c = d.contact[ci]
                        for fi, g in enumerate(self.foot_gid):
                            if c.geom1 == g or c.geom2 == g:
                                pen[fi] = min(pen[fi], c.dist)
                    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                    pr = min(pen[0], pen[1]); pf = min(pen[2], pen[3])  # 뒤/앞 최대침투
                    print('[hl] s=%d t=%.2f z=%.3f x=%+.3f y=%+.3f yaw=%+.0f tilt=%.1f 침투뒤/앞=%.1f/%.1fmm falls=%d'
                          % (s, d.time, d.qpos[2], d.qpos[0], d.qpos[1], yaw, tilt, pr*1000, pf*1000, falls), flush=True)
            print('[hl] 종료: %d스텝 falls=%d 최종 x=%+.3f' % (nsteps, falls, d.qpos[0]), flush=True)
            return
        with mujoco.viewer.launch_passive(m, d, key_callback=self._key_callback) as v:
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 1     # 외란 박스 ON
            _re = int(os.environ.get('RENDER_EVERY', '10'))     # 렌더당 물리스텝(1kHz라 매스텝 렌더는 과함→10=100Hz)
            _rate = float(os.environ.get('RATE', '1.0'))        # ★재생 배속(2=2배빠르게, 0=최대속도/sleep없음)
            print('viewer open — 창 닫으면 종료. (더블클릭+Ctrl+우드래그=외란) | RATE=%.1f배 RENDER_EVERY=%d' % (_rate, _re))
            while v.is_running():
                t0 = time.time()
                for _ in range(_re):                            # 물리 _re스텝 후 1회 렌더(빠르고 부드럽게)
                    control_fn()
                    mujoco.mjv_applyPerturbForce(m, d, v.perturb)
                    mujoco.mj_step(m, d)
                    if reset_on_fall and d.qpos[2] < 0.2:
                        mujoco.mj_resetData(m, d); reset_fn()
                _pc += 1
                if _sp and _pc % 3 == 0: self.publish_state(_sp)    # GUI 모니터 패널(~30Hz)
                self.draw_overlay(v)
                # 좌상단: 시뮬 시간 / 우상단: 외란 힘 N
                fext = max((float(np.linalg.norm(d.xfrc_applied[b, :3]))
                            for b in range(1, m.nbody)), default=0.0)
                cv = self.cmd_v
                v.set_texts([                                    # 구조3와 동일: 좌상=시간 우상=외력 좌하=명령
                    (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     'sim time', '%.2f s' % d.time),
                    (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                     'ext force', '%.0f N' % fext),
                    (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     'cmd vx/vy/wz', '%+.2f %+.2f %+.2f' % (cv[0], cv[1], cv[5]))])
                v.sync()
                if _rate > 0:                                   # RATE=0이면 sleep없이 최대속도
                    dt = _re * m.opt.timestep / _rate - (time.time() - t0)
                    if dt > 0:
                        time.sleep(dt)


# ══════════════════════════════════════════════════════════════
# 모드  (check 단계는 stance/WBIC가 동작하면 자동 검증되므로 별도 두지 않음)
# ══════════════════════════════════════════════════════════════
def mode_view():
    """정적 기립 (physics 정지) — 자세/모델 시각 확인."""
    q = QuadSim(); q.set_home()
    q.run_viewer(lambda: mujoco.mj_forward(q.m, q.d), reset_on_fall=False)


def mode_inspect():
    """관절 가동범위 확인 — 중력 OFF + 제어 OFF + 베이스 고정.
    뷰어에서 발/링크를 **Ctrl+드래그**하면 관절이 움직여 한계(joint limit)가 보임."""
    q = QuadSim(); q.crouch_home()
    q.m.opt.gravity[:] = 0.0                      # 중력 끔 (안 떨어짐)
    base0 = q.d.qpos[0:7].copy()
    # 관절 가동범위 출력 (구동 hinge 관절 전체)
    print('=== 관절 가동범위 ===')
    for j in range(q.m.njnt):
        if q.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        nm = mujoco.mj_id2name(q.m, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = q.m.jnt_range[j]
        print('  %-18s : [%+.2f, %+.2f] (%.0f~%.0f°)' %
              (nm, lo, hi, np.degrees(lo), np.degrees(hi)))

    def ctrl():
        q.d.ctrl[:] = 0                           # 관절 토크 0 (자유, 드래그로 가동)
        q.d.qpos[0:7] = base0; q.d.qvel[0:6] = 0  # 베이스 고정 (드리프트 방지)
    q.run_viewer(ctrl, reset_on_fall=False)


def mode_stand():
    """중력 하 PD(+중력보상) 기립 (제어 sanity)."""
    q = QuadSim(); q.set_home()
    q.run_viewer(q.pd_grav)


def mode_stance():
    """1단계 — WBIC 단독 정지 균형 (크라우치 자세, 4발 접촉정합 제어)."""
    q = QuadSim(); q.crouch_home()
    print('크라우치 base_z=%.3f  com_ref=%s' % (q.d.qpos[2], np.round(q.com_ref, 3)))
    q.run_viewer(q.wbic_stance, reset_fn=q.crouch_home)


def mode_mpc():
    """3단계 — MPC(Linear Convex) + WBIC 연동. 현재: 정지(전부 stance) GRF 계획→추종."""
    q = QuadSim(); q.crouch_home(); q.setup_mpc()
    x_ref = q.body_x0().copy()        # 현재 자세를 목표로 (정지 유지)
    lam0 = q.mpc_grf(x_ref)
    print('MPC Σλz=%.1f N (무게 %.1f)' %
          (lam0[:, 2].sum(), q.m.body_subtreemass[0] * 9.81))

    def ctrl():
        q.wbic_track(q.mpc_grf(x_ref))

    def reset():
        q.crouch_home()
    q.run_viewer(ctrl, reset_fn=reset)


def mode_walk():
    """3.b단계 — MPC+WBIC 정적 크롤 보행 (제자리). [개발중 — 뷰어 판단용]
    각 발 슬롯: 전반 CoM를 나머지3발 중심으로 이동(4발지지) → 후반 그 발 swing."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        from gait_sim.gait import swing_foot_pos
    finally:
        sys.argv = _a
    q = QuadSim(); q.crouch_home(); q.setup_mpc()
    T_GAIT, ORDER, SHIFT_FRAC, STEP_H = 2.4, [3, 0, 2, 1], 0.55, 0.05
    KP_SW, KD_SW = 40.0, 2.0
    state = {'nominal': [q.foot_point(i).copy() for i in range(4)],
             'liftoff': [q.foot_point(i).copy() for i in range(4)],
             'x0': q.body_x0().copy()}

    def schedule(t):
        ph = (t % T_GAIT) / T_GAIT; slot = int(ph * 4) % 4; local = (ph * 4) % 1.0
        leg = ORDER[slot]
        if local < SHIFT_FRAC:
            return leg, local / SHIFT_FRAC, [0, 1, 2, 3], []
        return leg, (local - SHIFT_FRAC) / (1 - SHIFT_FRAC), \
            [j for j in range(4) if j != leg], [leg]

    def csched(t):
        cs = np.zeros((q.N_MPC, 4), dtype=bool)
        for k in range(q.N_MPC):
            _, _, st, _ = schedule(t + k * q.MPC.DT_MPC)
            for i in st:
                cs[k, i] = True
        return cs

    def ctrl():
        t = q.d.time
        leg, s, st, sw = schedule(t)
        if sw and s < 0.02:
            state['liftoff'][leg] = q.foot_point(leg).copy()
        tri = [j for j in range(4) if j != leg]
        cen = np.mean([q.foot_point(j)[:2] for j in tri], axis=0)
        x_ref = state['x0'].copy(); x_ref[3] = cen[0]; x_ref[4] = cen[1]
        q.wbic_track(q.mpc_grf(x_ref, csched(t)), contacts=tuple(st))
        for i in sw:
            p_tgt = swing_foot_pos(s, state['liftoff'][i], state['nominal'][i],
                                   np.zeros(3), step_height=STEP_H, tau_land=1.0)
            qt = q.ik_leg(i, p_tgt, q.d.qpos[q.legqp[i]])
            q.d.ctrl[i * q.dof:(i + 1) * q.dof] = np.clip(
                KP_SW * (qt - q.d.qpos[q.legqp[i]]) - KD_SW * q.d.qvel[q.legqv[i]], -TAU_MAX, TAU_MAX)

    q.run_viewer(ctrl, reset_fn=q.crouch_home)


def mode_trot():
    """3.b단계 — MPC+WBIC trot 보행 (제자리). [개발중 — 뷰어 판단용]
    대각쌍(HL+FR / HR+FL) 교대. baseline 방식: x_ref=속도/자세/높이 추종(위치 안박음),
    MPC_Q 자세가중치↑. 동적 게이트라 roll 균형이 난관(튜닝 필요)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        from gait_sim.gait import swing_foot_pos
    finally:
        sys.argv = _a
    q = QuadSim()
    _tz = os.environ.get('TROT_Z')           # trot crouch 높이 override(낮추면 CoM↓ roll안정↑)
    q.crouch_home(float(_tz) if _tz else None); q.setup_mpc()
    TROT_Q = np.diag([200., 200., 100., 0., 100., 200., 0., 0., 1., 10., 10., 1., 0.])
    OFFSET = {0: 0.0, 3: 0.0, 1: 0.5, 2: 0.5}      # 대각쌍 A=HL,FR / B=HR,FL
    # SWING_FRAC<0.5 → 대각 전환에 double-support 겹침(착지 후 이륙) → 공중(flight) 방지
    SETTLE = 0.5
    T_TROT = float(os.environ.get('TROT_T', '0.50'))        # 레퍼런스 trot 프리셋
    SWING_FRAC = float(os.environ.get('TROT_SWF', '0.50'))  # D=swing 비율
    STEP_H = float(os.environ.get('TROT_STEPH', '0.08'))
    V = float(os.environ.get('TROT_V', '0.30'))     # 전진속도[m/s] 초기/기본
    VY = float(os.environ.get('TROT_VY', '0.0'))    # ★측방속도[m/s] (+좌 −우)
    WZ = float(os.environ.get('TROT_WZ', '0.0'))    # 선회각속도[rad/s] (+좌선회)
    ACC = float(os.environ.get('TROT_ACC', '0.6'))  # 명령 가속도제한[m/s²]: 시작램프+GUI 급조작 완화
    WARMUP = float(os.environ.get('TROT_WARMUP', '0.6'))  # 시작 제자리trot 시간[s]: 첫 사이클 리듬확립 후 이동(시작 lurch 완화)
    CMDFILE = os.environ.get('CMDFILE')             # ★GUI 연동: JSON(/tmp/quad_cmd.json) 폴링(v/vy/w)
    KP_SW = float(os.environ.get('TROT_KPSW', '40.0')); KD_SW = 2.0
    KCAP = float(os.environ.get('TROT_KCAP', '0.16'))   # capture 게인 ≈√(z/g) (LIPM)
    USE_DETECT = os.environ.get('DETECT', '1') == '1'   # detect_contact 조기착지 보정 on/off
    T_ST = T_TROT * (1 - SWING_FRAC)
    S = {'armed': False, 't0': 0.0, 'nominal': None, 'liftoff': None, 'x_ref': None,
         'ptgt_prev': [None, None, None, None], 'lam_des': None, 'mpc_t': -1.0, 'bx': 0.0,
         'settle_until': SETTLE,
         'Vt': V, 'Vyt': VY, 'Wt': WZ,                      # 목표명령(GUI가 갱신)
         'Vs': 0.0, 'Vys': 0.0, 'Ws': 0.0, 'cmd_t': -1.0,   # 스무딩 적용명령(0서 시작)
         'yaw_ref': 0.0}                                     # 선회 yaw각 참조(적분)

    def gait(i, tg):
        ph = (tg / T_TROT + OFFSET[i]) % 1.0
        return (ph >= SWING_FRAC, 0.0) if ph >= SWING_FRAC else (False, ph / SWING_FRAC)

    def csched(tg):
        cs = np.zeros((q.N_MPC, 4), dtype=bool)
        for k in range(q.N_MPC):
            for i in range(4):
                cs[k, i] = gait(i, tg + k * q.MPC.DT_MPC)[0]
        return cs

    def ctrl():
        t = q.d.time
        # ── GUI 명령 폴링(CMDFILE, ~20Hz): v/vy/w + mode ──
        if CMDFILE and (S['cmd_t'] < 0 or t - S['cmd_t'] > 0.05):
            S['cmd_t'] = t
            try:
                with open(CMDFILE) as _f: _c = json.load(_f)
                S['Vt'] = float(_c.get('v', S['Vt'])); S['Vyt'] = float(_c.get('vy', S['Vyt'])); S['Wt'] = float(_c.get('w', S['Wt']))
                q.cmd_mode = _c.get('mode', 'move')
            except Exception: pass
        # ── 모드: move 외(Ready 서기/Ground 눕기/STOP)는 제자리 WBIC stance ──
        if q.cmd_mode != 'move':
            q.wbic_stance(); S['armed'] = False; q.cmd_v[:] = 0.0
            return
        # ★ 리셋 감지(뷰어 Backspace/자동 낙상): 시간 역행 OR base 위치 급변(>0.3m) →
        #   상태기계 재초기화 + 재정착 타이머. 안 하면 낡은 world-frame nominal/liftoff 로 멈춤.
        #   settle_until 은 d.time 절대값 — 시간 0복귀 안 하는 passive 뷰어 케이스도 커버.
        if S['armed'] and (t < S['t0'] - 1e-6 or abs(q.d.qpos[0] - S['bx']) > 0.3):
            print('[trot] 리셋 감지(t=%.2f bx=%.2f→%.2f) → crouch 복원 후 재정착'
                  % (t, S['bx'], q.d.qpos[0]))
            q.crouch_home()                  # ★ 깨끗한 crouch-at-origin 복원(time→0)
            S['armed'] = False; S['lam_des'] = None
            S['ptgt_prev'] = [None, None, None, None]
            S['t0'] = 0.0; S['settle_until'] = SETTLE   # crouch_home 후 time=0
            S['bx'] = float(q.d.qpos[0]); return
        S['bx'] = float(q.d.qpos[0])
        if t < S['settle_until']:            # 1) WBIC stance 정착 (초기/리셋)
            q.wbic_stance(); return
        if not S['armed']:                   # 2) 정착/리셋 자세를 trot 기준으로 캡처
            S['armed'] = True; S['t0'] = t; S['yaw_ref'] = 0.0
            S['nominal'] = [q.foot_point(i).copy() for i in range(4)]
            S['liftoff'] = [q.foot_point(i).copy() for i in range(4)]
            xr = np.zeros(13); xr[5] = float(q.d.subtree_com[0][2]); xr[12] = -9.81  # vx/vy/wz는 매틱 갱신
            S['x_ref'] = xr
            # 발의 hip 기준 default offset (settle 시) — 이후 hip 따라 전방 배치
            S['hip_off'] = [S['nominal'][i][:2] - q.d.xpos[q.hip_bid[i]][:2] for i in range(4)]
            S['gz'] = [float(S['nominal'][i][2]) for i in range(4)]
            q.MPC.MPC_Q = TROT_Q
            print('[trot] 정착 완료 → trot 시작 v=%.2f vy=%.2f w=%.2f%s (base_z=%.3f)'
                  % (S['Vt'], S['Vyt'], S['Wt'], ' [GUI]' if CMDFILE else '', q.d.qpos[2]))
        tg = t - S['t0']                     # 3) trot
        # ── 가속도제한 스무딩 + 시작 warmup(첫 WARMUP초 제자리trot로 리듬확립 후 이동) ──
        dts = q.m.opt.timestep
        _go = tg > WARMUP                                   # warmup 지난 뒤에만 목표속도 적용
        _vt, _vyt, _wt = (S['Vt'], S['Vyt'], S['Wt']) if _go else (0.0, 0.0, 0.0)
        S['Vs']  += float(np.clip(_vt  - S['Vs'],  -ACC * dts, ACC * dts))
        S['Vys'] += float(np.clip(_vyt - S['Vys'], -ACC * dts, ACC * dts))
        S['Ws']  += float(np.clip(_wt  - S['Ws'],  -2.0 * dts, 2.0 * dts))
        V_eff, Vy_eff, W_eff = S['Vs'], S['Vys'], S['Ws']
        # 선회: yaw각 참조 적분 + 명령(body)→world 회전 (SRBD 상태는 world frame)
        S['yaw_ref'] += W_eff * dts
        _qq = q.d.qpos[3:7]                                                     # quat [w,x,y,z]
        yaw_m = float(np.arctan2(2 * (_qq[0]*_qq[3] + _qq[1]*_qq[2]), 1 - 2 * (_qq[2]**2 + _qq[3]**2)))
        cy, sy = np.cos(yaw_m), np.sin(yaw_m)
        vx_w = V_eff * cy - Vy_eff * sy; vy_w = V_eff * sy + Vy_eff * cy        # body→world 속도
        S['x_ref'][2] = S['yaw_ref']; S['x_ref'][8] = W_eff                     # yaw각·yaw rate 참조
        S['x_ref'][9] = vx_w; S['x_ref'][10] = vy_w                            # world vx,vy
        q.cmd_v[0] = V_eff; q.cmd_v[1] = Vy_eff; q.cmd_v[5] = W_eff             # 시각화(body명령)
        # ★ Di Carlo 식: gait 스케줄=primary, detect_contact=조기/지연 착지 보정.
        #   스케줄 stance → 힘제어. 스케줄 swing 후반(>0.7)+접촉감지 → 조기착지로 stance 승격.
        #   그 외 swing → 들어올림. (USE_DETECT=False 면 순수 스케줄, A/B 비교용)
        contact, _ = q.detect_contact([gait(i, tg)[0] for i in range(4)])
        st, sw = [], []
        for i in range(4):
            sch_stance, s_prog = gait(i, tg)
            early = USE_DETECT and (not sch_stance) and contact[i] and s_prog > 0.7
            if sch_stance or early:                         # 스케줄 stance OR 조기착지 → 힘제어
                st.append(i); S['ptgt_prev'][i] = None      # 다음 swing 첫 프레임 v_des=0
            else:                                           # 스케줄 swing → 위치제어(들어올림)
                sw.append(i)
                if s_prog < 0.03:                           # swing 시작 시 liftoff 캡처
                    S['liftoff'][i] = q.foot_point(i).copy()
        # 발 배치 = hip 기준 default + Raibert(전진 0.5·T_st·V + 피드백 KCAP·(v−v_des))
        Jc = np.zeros((3, q.nv)); mujoco.mj_jacSubtreeCom(q.m, q.d, Jc, 0)
        vcom = Jc @ q.d.qvel
        v_des = np.array([vx_w, vy_w])       # Raibert 발배치 = world 속도(전진+측방, 선회 회전반영)
        rai = np.clip(0.5 * T_ST * v_des + KCAP * (vcom[:2] - v_des), -0.14, 0.14)
        q.foot_targets = [None, None, None, None]
        dt = q.m.opt.timestep; swing = {}
        Rw = np.array([[cy, -sy], [sy, cy]])                 # body→world(현재 yaw)
        _sh = STEP_H * (0.2 + 0.8 * min(1.0, tg / WARMUP)) if WARMUP > 1e-6 else STEP_H  # 시작 step↑ ramp(첫스윙 작게→full)
        for i in sw:                                        # swing 발끝 작업공간 목표(p,v)
            hip_xy = q.d.xpos[q.hip_bid[i]][:2]
            r_xy = hip_xy - q.d.qpos[:2]                     # 몸중심→hip
            tw = W_eff * T_ST * np.array([-r_xy[1], r_xy[0]])  # 선회 접선 발배치(yaw)
            pe_xy = hip_xy + Rw @ S['hip_off'][i] + rai + tw  # nominal도 몸따라 회전 + Raibert + 선회
            s_ = gait(i, tg)[1]
            p_end = np.array([pe_xy[0], pe_xy[1], S['gz'][i]])
            q.foot_targets[i] = p_end                       # 착지 목표 시각화
            p_tgt = swing_foot_pos(s_, S['liftoff'][i], p_end,
                                   np.array([vcom[0], vcom[1], 0]), step_height=_sh, tau_land=1.0)
            pv = S['ptgt_prev'][i]                          # 목표속도(차분, 노이즈 억제 위해 clip)
            v_tgt = np.clip((p_tgt - pv) / dt, -1.0, 1.0) if pv is not None else np.zeros(3)
            S['ptgt_prev'][i] = p_tgt.copy(); swing[i] = (p_tgt, v_tgt)
        # ★ 표준 구조: MPC 저주파 재계획(DT_MPC=0.02s=50Hz), WBIC 풀주파(500Hz).
        #   MPC는 무겁고(긴 호라이즌 QP) 느리게 변하는 GRF 계획 → 매 스텝 풀 필요 없음.
        #   WBIC는 빠른 외란/접촉 반응 위해 매 스텝.  → 긴 호라이즌도 실시간 감당.
        # 재계획: 현재 MPC 윈도우[mpc_t, mpc_t+DT_MPC) 밖이면. t<mpc_t(시간역행/리셋)도 즉시 재계획.
        dmpc = t - S['mpc_t']
        if st and (S['lam_des'] is None or dmpc < 0 or dmpc >= q.MPC.DT_MPC):
            S['lam_des'] = q.mpc_grf(S['x_ref'], csched(tg)); S['mpc_t'] = t
        lam_des = S['lam_des'] if (st and S['lam_des'] is not None) else np.zeros((4, 3))
        q.wbic_track(lam_des, contacts=tuple(st), swing=swing)

    def reset():
        q.crouch_home(); S['armed'] = False

    q.run_viewer(ctrl, reset_fn=reset)


def main():
    ap = argparse.ArgumentParser(description='실제 4족 MuJoCo 통합 테스트')
    # 단계: stance(1) → mpc(3.a 정지) → walk(3.b 보행) → nmpc(4).  view/stand 는 sanity.
    ap.add_argument('--mode', default='stance',
                    choices=['view', 'inspect', 'stand', 'stance', 'mpc', 'walk', 'trot'])
    ap.add_argument('--nolimit', action='store_true', help='관절 한계 해제 (가동범위 테스트용)')
    ap.add_argument('--robot', default='ours', choices=list(ROBOTS), help='로봇 모델 선택')
    a = ap.parse_args()
    global _NOLIMIT, _ROBOT; _NOLIMIT = a.nolimit; _ROBOT = a.robot
    os.environ.setdefault('DISPLAY', ':0')
    {'view': mode_view, 'inspect': mode_inspect, 'stand': mode_stand, 'stance': mode_stance,
     'mpc': mode_mpc, 'walk': mode_walk, 'trot': mode_trot}[a.mode]()
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
