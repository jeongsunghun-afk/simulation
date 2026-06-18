"""biped WBIC balance — MuJoCo, point-contact + capture-point stepping.

설계 (biped/README "WBIC = mj_jac 직접 사용"):
  - 동역학:  M·q̈ + h = Sᵀ·τ + Σ Jᵢᵀ·λᵢ      (M=mj_fullM, h=qfrc_bias, Jᵢ=mj_jac@foot)
  - WBIC QP: 변수 z=[q̈(14); λ(3·K)]
       min  Σ task‖Jₜq̈ − aₜ‖² + w_λ‖λ‖² + w_reg‖q̈‖²
       s.t. floating-base 6행:  M[0:6]q̈ − Σ Jᵢ[:,0:6]ᵀλᵢ = −h[0:6]
            마찰추(피라미드), λ_z ≥ λz_min
       τ = M[6:]q̈ + h[6:] − Σ Jᵢ[:,6:]ᵀλᵢ   (post clamp ±τmax)

Stage 1: 양발 스탠스만 (pitch 표류 확인). Stage 2 에서 capture-point stepping 추가.
"""
from __future__ import annotations
import os
import sys
import math
import numpy as np
import mujoco
import mujoco.viewer
from qpsolvers import solve_qp

# gait_sim_v13 baseline swing 궤적 (Zeng 2019 Scheme I) 재사용
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gait_sim.gait import swing_foot_pos, STEP_HEIGHT  # noqa: E402

GROUND_Z = 0.003   # 발 collision box 안착 높이

JOINT_NAMES = ['HL_hip', 'HL_thigh', 'HL_calf', 'HL_foot',
               'HR_hip', 'HR_thigh', 'HR_calf', 'HR_foot']
FEET = ['HL_foot_collision', 'HR_foot_collision']
TAU_MAX = 100.0
MU = 0.7


class BipedWBIC:
    def __init__(self, mjcf):
        self.m = mujoco.MjModel.from_xml_path(mjcf)
        self.d = mujoco.MjData(self.m)
        self.nv = self.m.nv                      # 14
        self.foot_bid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FEET]
        self.foot_body = [self.m.geom_bodyid[g] for g in self.foot_bid]
        # 발바닥 접촉점 = collision 메시 최하단 정점 (geom 원점이 아님!)
        self.sole_local = []
        for g in self.foot_bid:
            mid = self.m.geom_dataid[g]
            adr = self.m.mesh_vertadr[mid]; n = self.m.mesh_vertnum[mid]
            V = self.m.mesh_vert[adr:adr + n].reshape(-1, 3)
            self.sole_local.append(V[np.argmin(V[:, 2])].copy())
        self.z_ref = None
        self.com_x_ref = None
        self.q_home = np.zeros(8)   # crouch posture ref (settle 에서 채움)
        self.com_mode = 0           # 0:OFF 1:전체CoM만 2:파트별CoM만 (뷰어 '1' 키로 순환)
        self.show_com = False       # com_mode 에서 파생
        self.show_part_com = False

    # ---- model quantities ----
    def M_full(self):
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.m, M, self.d.qM)
        return M

    def foot_world(self, i, data=None):
        """발바닥 접촉점 (메시 최하단 정점) world pos. geom 원점 아님."""
        d = data if data is not None else self.d
        g = self.foot_bid[i]
        R = d.geom_xmat[g].reshape(3, 3)
        return d.geom_xpos[g] + R @ self.sole_local[i]

    def foot_jac(self, i):
        jp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, self.d, jp, None, self.foot_world(i), self.foot_body[i])
        return jp

    def draw_grf_arrows(self, viewer, feet, forces, scale=0.004):
        """GRF 를 화살표로 시각화 (시작=발바닥, 방향=힘 방향, 길이∝크기)."""
        scn = viewer.user_scn
        scn.ngeom = 0
        for leg, F in zip(feet, forces):
            if F is None:
                continue
            F = np.asarray(F, float)
            if np.linalg.norm(F) < 1.0:
                continue
            p0 = self.foot_world(leg)
            p1 = p0 + F * scale
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                                np.zeros(3), np.zeros(3), np.zeros(9),
                                np.array([0.1, 0.9, 0.2, 1.0], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, p0, p1)
            scn.ngeom += 1

    def com_jac(self):
        jp = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.m, self.d, jp, 0)
        return jp

    # ---- 전신 CoM 모니터링 (뷰어) ----
    def _key_callback(self, keycode):
        """뷰어 '1' 키: CoM 표시 순환  OFF → 전체CoM만 → 파트별CoM만 → OFF.
        ('M'은 MuJoCo 내장 mjVIS_COM 단축키라 충돌 → '1' 사용)."""
        if keycode == ord('1'):
            self.com_mode = (self.com_mode + 1) % 3
            self.show_com = (self.com_mode == 1)
            self.show_part_com = (self.com_mode == 2)
            print('[viewer] CoM 표시: %s' % ['OFF', '전체CoM만', '파트별CoM만'][self.com_mode])

    def draw_com_marker(self, viewer):
        """전신 CoM(주황 구, show_com) + 각 파트 CoM(시안 구, show_part_com).
        draw_grf_arrows 뒤에 호출(ngeom 이어붙임)."""
        scn = viewer.user_scn
        eye = np.eye(3).flatten()
        # 전신 질량중심 (주황 구)
        if getattr(self, 'show_com', False) and scn.ngeom < scn.maxgeom:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.016, 0, 0]), self.d.subtree_com[0].copy(), eye,
                                np.array([1.0, 0.55, 0.0, 1.0], np.float32))
            scn.ngeom += 1
        # 각 파트(link) CoM (시안 구, 질량 비례 크기)
        if getattr(self, 'show_part_com', False):
            masses = self.m.body_mass
            mmax = float(masses[1:].max()) if self.m.nbody > 1 else 1.0
            for b in range(1, self.m.nbody):
                if masses[b] <= 0 or scn.ngeom >= scn.maxgeom:
                    continue
                r = 0.008 + 0.020 * (masses[b] / max(mmax, 1e-9))
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([r, 0, 0]), self.d.xipos[b].copy(), eye,
                                    np.array([0.1, 0.7, 1.0, 0.9], np.float32))
                scn.ngeom += 1

    # ---- WBIC stance QP ----
    def wbic_stance(self, contacts, com_ref=None, kp=(120, 120, 160), kd=(22, 22, 25),
                    kp_ori=120.0, kd_ori=18.0, kp_post=40.0, kd_post=5.0,
                    w_lam=1e-3, w_reg=1e-4):
        d, m, nv = self.d, self.m, self.nv
        K = len(contacts)
        nz = nv + 3 * K
        sl_lam = lambda k: slice(nv + 3 * k, nv + 3 * k + 3)
        if com_ref is None:
            com_ref = np.array([self.com_x_ref, 0.0, self.z_ref])

        M = self.M_full()
        h = d.qfrc_bias.copy()
        qvel = d.qvel.copy()

        # ---- tasks → desired generalized accel terms in cost ----
        P = np.zeros((nz, nz)); qv = np.zeros(nz)

        # CoM xy + height
        Jc = self.com_jac()
        p_com = d.subtree_com[0].copy()
        v_com = Jc @ qvel
        a_com = np.array([kp[i] * (com_ref[i] - p_com[i]) - kd[i] * v_com[i] for i in range(3)])
        wc = np.array([1.0, 1.0, 1.0])
        for r in range(3):
            P[:nv, :nv] += wc[r] * np.outer(Jc[r], Jc[r])
            qv[:nv]     -= wc[r] * a_com[r] * Jc[r]

        # base orientation (upright): error from quat → small-angle vec
        quat = d.qpos[3:7]
        ori_err = np.zeros(3); mujoco.mju_quat2Vel(ori_err, _quat_err(quat), 1.0)
        a_ori = kp_ori * (-ori_err) - kd_ori * qvel[3:6]
        w_ori = 5.0
        for j in range(3):
            P[3 + j, 3 + j] += w_ori
            qv[3 + j]       -= w_ori * a_ori[j]

        # posture (joints → crouch home)
        a_post = kp_post * (self.q_home - d.qpos[7:15]) - kd_post * qvel[6:14]
        for j in range(8):
            P[6 + j, 6 + j] += 1.0
            qv[6 + j]       -= a_post[j]

        # regularization + force reg
        P[:nv, :nv] += w_reg * np.eye(nv)
        for k in range(K):
            P[sl_lam(k), sl_lam(k)] += w_lam * np.eye(3)

        # ---- equality: floating-base 6-DoF dynamics ----
        Js = [self.foot_jac(c) for c in contacts]
        A = np.zeros((6, nz)); b = -h[0:6]
        A[:, :nv] = M[0:6, :]
        for k, J in enumerate(Js):
            A[:, sl_lam(k)] = -J[:, 0:6].T
        # ---- equality: 접촉 발 무가속 (J_c·q̈ = 0, 디딘 발 planted) ----
        for J in Js:
            Ac = np.zeros((3, nz)); Ac[:, :nv] = J
            A = np.vstack([A, Ac]); b = np.concatenate([b, np.zeros(3)])

        # ---- inequality: friction pyramid + λz>=λz_min ----
        G = []; hg = []
        lb = np.full(nz, -1e8); ub = np.full(nz, 1e8)
        for k in range(K):
            o = nv + 3 * k
            lb[o + 2] = 20.0  # λz_min (N)
            for sx, sy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                row = np.zeros(nz)
                row[o + 0] = sx; row[o + 1] = sy; row[o + 2] = -MU
                G.append(row); hg.append(0.0)
        G = np.vstack(G); hg = np.array(hg)

        P = 0.5 * (P + P.T) + 1e-8 * np.eye(nz)
        z = solve_qp(P, qv, G, hg, A, b, lb, ub, solver='quadprog')
        if z is None:
            self.last_lam = None
            return None, None, False
        ddq = z[:nv]
        lam = [z[sl_lam(k)] for k in range(K)]
        # τ from actuated rows
        tau = M[6:14, :] @ ddq + h[6:14]
        for k, J in enumerate(Js):
            tau -= J[:, 6:14].T @ lam[k]
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        self.last_lam = lam   # GRF 화살표 표시용
        return tau, lam, True


def _quat_err(q):
    """current quat → quat error vs upright [1,0,0,0]  (returns q since ref=identity)."""
    return q


# ────────────────────────────────────────────────────────────────
# Stage 2: capture-point marching (point-contact stepping balance)
# ────────────────────────────────────────────────────────────────
LEG_QVEL = {0: [6, 7, 8, 9], 1: [10, 11, 12, 13]}    # HL, HR  (q̇ idx)
LEG_QPOS = {0: [7, 8, 9, 10], 1: [11, 12, 13, 14]}   # HL, HR  (q idx)


# swing 궤적은 gait_sim_v13 baseline (gait_sim.gait.swing_foot_pos, Zeng 2019 Scheme I) 사용.
# X: 5차 spline(body_vel 피드포워드) · Y: 5차 smoothstep · Z: 6차 다항식(C2/jerk 연속).


def _R_to_euler_xyz(R):
    """world R → (roll, pitch, yaw) XYZ."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 0.99999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw  = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1]); yaw = 0.0
    return roll, pitch, yaw


class BipedMarch(BipedWBIC):
    def __init__(self, mjcf):
        super().__init__(mjcf)
        self.scratch = mujoco.MjData(self.m)
        self.nom_foot = {}     # nominal foot pos (base frame x,y) at q=0
        self.mpc = None        # gait_sim MPC 모듈 (setup_mpc 에서)
        self.I_com = None

    # ───────────────────── MPC 연결 ─────────────────────
    def compute_Icom(self):
        """현재 자세의 CoM 기준 복합 관성 (world frame, 3×3)."""
        m, d = self.m, self.d
        com = d.subtree_com[0]
        I = np.zeros((3, 3))
        for b in range(1, m.nbody):
            mass = m.body_mass[b]
            if mass <= 0:
                continue
            r = d.xipos[b] - com
            Rb = d.ximat[b].reshape(3, 3)
            Ib = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
            I += Ib + mass * (r @ r * np.eye(3) - np.outer(r, r))
        return I

    def setup_mpc(self):
        """gait_sim quadruped MPC 를 biped 용으로 연결 (모듈 상수 override)."""
        settle_and_set_ref(self)
        mujoco.mj_forward(self.m, self.d)
        self.I_com = self.compute_Icom()
        import gait_sim.controllers.mpc as mpc
        mpc.TOTAL_MASS = float(self.m.body_subtreemass[0])
        mpc.BODY_INERTIA = self.I_com
        mpc._I_BODY = self.I_com           # ★ LTV 경로가 쓰는 실제 관성 (line 217)
        mpc.MU_FRICTION = MU
        # biped 전용 가중치 (Cheetah3 논문 참조 + 직립 위해 roll/pitch 유지)
        #   [roll,pitch,yaw, px,py,pz, ωx,ωy,ωz, vx,vy,vz, g]
        mpc.MPC_Q = np.diag([100., 100., 1.,   1., 1., 50.,
                             0., 0., 1.,        1., 1., 1.,   0.])
        mpc.MPC_R = 1e-6 * np.eye(3)
        mpc.LAMZ_MIN = 10.0                # 최소 수직 지지력 [N]
        mpc.LAMZ_MAX = 350.0               # 최대 수직 지지력 [N] (힘 폭발 캡)
        self.mpc = mpc
        self.N_MPC = mpc.N_MPC
        self.DT_MPC = mpc.DT_MPC

    def body_x0(self):
        """MuJoCo state → MPC body state x0 (13)."""
        d = self.d
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, d.qpos[3:7]); Rm = Rm.reshape(3, 3)
        roll, pitch, yaw = _R_to_euler_xyz(Rm)
        com = d.subtree_com[0]
        vcom = self.com_jac() @ d.qvel
        omega_w = Rm @ d.qvel[3:6]
        return np.array([roll, pitch, yaw, com[0], com[1], com[2],
                         omega_w[0], omega_w[1], omega_w[2],
                         vcom[0], vcom[1], vcom[2], -9.81])

    def mpc_grf(self, stance_feet, x_ref=None):
        """MPC 호출 → lam_des (2,3)  biped HL/HR GRF (stance 만 비영)."""
        d = self.d
        com = d.subtree_com[0].copy()
        N = self.N_MPC
        cs = np.zeros((N, 4), dtype=bool)
        fp = np.zeros((N, 4, 3))
        foot_rel = [self.foot_world(0) - com, self.foot_world(1) - com]
        for leg in stance_feet:
            cs[:, leg] = True
        for k in range(N):
            fp[k, 0] = foot_rel[0]; fp[k, 1] = foot_rel[1]
        lam = self.mpc.mpc_qp_plan(self.body_x0(), cs, fp, x_ref_step=x_ref, ltv=True)
        return lam[:2]   # biped 2발

    def wbic_track(self, contacts, lam_des, kp_ori=80.0, kd_ori=14.0,
                   kp_post=40.0, kd_post=5.0, w_track=1.0, w_reg=2e-3):
        """WBIC: MPC 의 lam_des 를 추종하며 floating-base 동역학 일관 τ 산출.
        변수 z=[q̈(14); λ(3K)],  cost: w_track‖λ−λ_des‖² + posture/ori task + reg."""
        d, m, nv = self.d, self.m, self.nv
        K = len(contacts)
        nz = nv + 3 * K
        sl_lam = lambda k: slice(nv + 3 * k, nv + 3 * k + 3)
        M = self.M_full(); h = d.qfrc_bias.copy(); qvel = d.qvel.copy()
        P = np.zeros((nz, nz)); qv = np.zeros(nz)
        # track lam_des
        for k in range(K):
            P[sl_lam(k), sl_lam(k)] += w_track * np.eye(3)
            qv[sl_lam(k)]          -= w_track * lam_des[k]
        # orientation upright (base ang accel task)
        ori_err = np.zeros(3); mujoco.mju_quat2Vel(ori_err, d.qpos[3:7].copy(), 1.0)
        a_ori = kp_ori * (-ori_err) - kd_ori * qvel[3:6]
        for j in range(3):
            P[3 + j, 3 + j] += 2.0; qv[3 + j] -= 2.0 * a_ori[j]
        # posture → crouch home
        a_post = kp_post * (self.q_home - d.qpos[7:15]) - kd_post * qvel[6:14]
        for j in range(8):
            P[6 + j, 6 + j] += 1.0; qv[6 + j] -= a_post[j]
        P[:nv, :nv] += w_reg * np.eye(nv)
        # equality: floating-base dyn
        A = np.zeros((6, nz)); b = -h[0:6]; A[:, :nv] = M[0:6, :]
        Js = [self.foot_jac(c) for c in contacts]
        for k, J in enumerate(Js):
            A[:, sl_lam(k)] = -J[:, 0:6].T
        # friction
        G = []; hg = []; lb = np.full(nz, -1e8); ub = np.full(nz, 1e8)
        for k in range(K):
            o = nv + 3 * k; lb[o + 2] = 0.0
            for sx, sy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                row = np.zeros(nz); row[o] = sx; row[o + 1] = sy; row[o + 2] = -MU
                G.append(row); hg.append(0.0)
        G = np.vstack(G); hg = np.array(hg)
        P = 0.5 * (P + P.T) + 1e-8 * np.eye(nz)
        z = solve_qp(P, qv, G, hg, A, b, lb, ub, solver='quadprog')
        if z is None:
            return None, None, False
        ddq = z[:nv]; lam = [z[sl_lam(k)] for k in range(K)]
        tau = M[6:14, :] @ ddq + h[6:14]
        for k, J in enumerate(Js):
            tau -= J[:, 6:14].T @ lam[k]
        return np.clip(tau, -TAU_MAX, TAU_MAX), lam, True

    def foot_ik(self, leg, p_tgt, q_seed):
        """DLS IK: swing leg 4관절 → 목표 발 world pos. base 는 현재 상태 고정."""
        s = self.scratch
        s.qpos[:] = self.d.qpos[:]
        s.qvel[:] = 0
        q = np.array(q_seed, float)
        qidx, vidx = LEG_QPOS[leg], LEG_QVEL[leg]
        rng = self.m.jnt_range
        jadr = [self.m.jnt_qposadr[self.m.dof_jntid[v]] for v in vidx]
        for _ in range(12):
            s.qpos[qidx] = q
            mujoco.mj_kinematics(self.m, s)
            mujoco.mj_comPos(self.m, s)
            p = self.foot_world(leg, data=s)
            e = p_tgt - p
            if np.linalg.norm(e) < 1e-4:
                break
            jp = np.zeros((3, self.nv))
            mujoco.mj_jac(self.m, s, jp, None, p, self.foot_body[leg])
            J = jp[:, vidx]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
            q = q + dq
            for k, a in enumerate(qidx):
                lo, hi = self.m.jnt_range[self.m.dof_jntid[vidx[k]]]
                q[k] = min(max(q[k], lo), hi)
        return q

    # ─── gait params (튜닝 대상) ───
    T_DS = 0.06      # double-support 시간 (짧게)
    T_SS = 0.14      # single-support 시간 (< 낙하 시정수 1/ω≈0.21s)
    Y_SHIFT = 0.80   # CoM-y 를 지지발 y 의 몇 배까지 이동 (측면 체중이동)
    CP_KX = 0.05     # capture-point x 추가 피드백
    MAX_STEP = 0.16  # liftoff 기준 최대 보폭
    KP_SW, KD_SW = 90.0, 4.0

    def gait_phase(self, t):
        """반환 (mode, stance, swing, s)  mode∈{'DS','SS'}.
        순환: DS(→HL지지) · SS(HL지지/HR스윙) · DS(→HR지지) · SS(HR지지/HL스윙)."""
        T_cyc = 2 * (self.T_DS + self.T_SS)
        ph = t % T_cyc
        if ph < self.T_DS:
            return 'DS', 0, None, ph / self.T_DS              # → HL 지지 준비
        ph -= self.T_DS
        if ph < self.T_SS:
            return 'SS', 0, 1, ph / self.T_SS                 # HL 지지, HR 스윙
        ph -= self.T_SS
        if ph < self.T_DS:
            return 'DS', 1, None, ph / self.T_DS              # → HR 지지 준비
        ph -= self.T_DS
        return 'SS', 1, 0, ph / self.T_SS                     # HR 지지, HL 스윙

    def run_march(self, headless=True, seconds=6.0, push=None, viewer=False):
        import time
        m, d = self.m, self.d
        settle_and_set_ref(self)
        omega = np.sqrt(9.81 / self.z_ref)
        nom = {0: self.foot_world(0)[:2].copy(), 1: self.foot_world(1)[:2].copy()}
        liftoff = {0: nom[0].copy(), 1: nom[1].copy()}
        v = mujoco.viewer.launch_passive(m, d, key_callback=self._key_callback) if viewer else None
        if v is not None:
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0   # 내장 CoM 마커 OFF (커스텀과 중복 방지)
        if viewer:
            print('viewer open (march) — 창 닫으면 종료.')
        nfall = 0
        com_y_prev = 0.0
        for step in range(int(seconds / m.opt.timestep)):
            t = d.time
            mode, stance, swing, s = self.gait_phase(t)
            com = d.subtree_com[0].copy()
            Jc = self.com_jac(); vcom = Jc @ d.qvel

            # desired CoM-y: 지지발 쪽으로 체중이동 (측면 균형 핵심)
            y_stance = nom[stance][1] * self.Y_SHIFT
            if mode == 'DS':
                # 이전 지지 y → 다음 지지 y 로 부드럽게 이동
                sig = s * s * (3 - 2 * s)
                com_y_des = (1 - sig) * com_y_prev + sig * y_stance
            else:
                com_y_des = y_stance
                com_y_prev = y_stance
            com_ref = np.array([self.com_x_ref, com_y_des, self.z_ref])

            tau = np.zeros(8)
            if mode == 'DS':
                used_feet = [0, 1]
                tau_w, lam, ok = self.wbic_stance(contacts=[0, 1], com_ref=com_ref)
                if ok: tau[:] = tau_w
                else:  nfall += 1
            else:  # SS: 지지 WBIC + 스윙 IK-PD
                used_feet = [stance]
                if s < 0.02:
                    liftoff[swing] = self.foot_world(swing)[:2].copy()
                # capture-point/ICP 풋스텝: 발을 CoM 낙하 방향으로 전진 (forward walking 허용)
                xi = com[:2] + vcom[:2] / omega
                tgt_raw = np.array([xi[0] + self.CP_KX * vcom[0], nom[swing][1]])
                # liftoff 기준 보폭 제한
                step_vec = tgt_raw - liftoff[swing]
                n = np.linalg.norm(step_vec)
                if n > self.MAX_STEP:
                    step_vec *= self.MAX_STEP / n
                tgt = liftoff[swing] + step_vec
                p_start = np.array([liftoff[swing][0], liftoff[swing][1], GROUND_Z])
                p_end   = np.array([tgt[0], tgt[1], GROUND_Z])
                body_vel = np.array([vcom[0], vcom[1], 0.0])
                p_tgt = swing_foot_pos(s, p_start, p_end, body_vel,
                                       step_height=STEP_HEIGHT, tau_land=1.0)
                q_sw_tgt = self.foot_ik(swing, p_tgt, d.qpos[LEG_QPOS[swing]])

                tau_w, lam, ok = self.wbic_stance(contacts=[stance], com_ref=com_ref)
                if ok: tau[:] = tau_w
                else:  nfall += 1
                sl = slice(swing * 4, swing * 4 + 4)
                q_sw = d.qpos[LEG_QPOS[swing]]; dq_sw = d.qvel[LEG_QVEL[swing]]
                tau[sl] = np.clip(self.KP_SW * (q_sw_tgt - q_sw) - self.KD_SW * dq_sw,
                                  -TAU_MAX, TAU_MAX)

            if push and abs(t - push[0]) < m.opt.timestep:
                d.qvel[0] += push[1]
            d.ctrl[:] = tau
            t0 = time.time()
            mujoco.mj_step(m, d)
            if v is not None:
                if not v.is_running():
                    break
                if d.qpos[2] < 0.15 or d.qpos[2] > 0.9 or abs(d.qpos[3]) < 0.6:
                    settle_and_set_ref(self); self.scratch.qpos[:] = d.qpos[:]; com_y_prev = 0.0
                self.draw_grf_arrows(v, used_feet, lam if ok else [None] * len(used_feet))
                self.draw_com_marker(v)
                v.sync()
                dt = m.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
            if step % 100 == 0:
                print('t=%.2f bz=%.3f com=(%.3f,%.3f,%.3f) %s st=%s ncon=%d %s'
                      % (t, d.qpos[2], com[0], com[1], com[2], mode,
                         'HL' if stance == 0 else 'HR', d.ncon, '' if ok else 'QPfail'))
        if v is not None:
            try: v.close()
            except Exception: pass
        print('march done. QPfail=%d final bz=%.3f upright=%.3f com=(%.3f,%.3f)'
              % (nfall, d.qpos[2], d.qpos[3], d.subtree_com[0][0], d.subtree_com[0][1]))
        return d.qpos[2], d.qpos[3]

    # ───────────────────── MPC→WBIC 추종 루프 ─────────────────────
    def run_mpc(self, seconds=6.0, push=None, viewer=False):
        """MPC(GRF 계획) → WBIC(λ 추종 τ) 파이프라인. gait 스케줄 + 스윙 IK 동일."""
        import time
        m, d = self.m, self.d
        if self.mpc is None:
            self.setup_mpc()
        settle_and_set_ref(self)
        omega = np.sqrt(9.81 / self.z_ref)
        nom = {0: self.foot_world(0)[:2].copy(), 1: self.foot_world(1)[:2].copy()}
        liftoff = {0: nom[0].copy(), 1: nom[1].copy()}
        v = mujoco.viewer.launch_passive(m, d, key_callback=self._key_callback) if viewer else None
        if v is not None:
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0   # 내장 CoM 마커 OFF (커스텀과 중복 방지)
        if viewer:
            print('viewer open (mpc) — 창 닫으면 종료.')
        nfail = 0
        lam_des = np.zeros((2, 3))
        mpc_every = max(1, int(round(self.DT_MPC / m.opt.timestep)))   # 10
        # 고정 steady reference (closed-loop): 직립, 영속도, CoM 기준 높이
        x_ref = np.array([0, 0, 0, self.com_x_ref, 0.0, self.z_ref,
                          0, 0, 0, 0, 0, 0, -9.81])
        for step in range(int(seconds / m.opt.timestep)):
            t = d.time
            mode, stance, swing, s = self.gait_phase(t)
            com = d.subtree_com[0].copy()
            Jc = self.com_jac(); vcom = Jc @ d.qvel

            stance_feet = [0, 1] if mode == 'DS' else [stance]
            # MPC GRF 계획 (DT_MPC 주기로 갱신)
            if step % mpc_every == 0:
                try:
                    lam_des = self.mpc_grf(stance_feet, x_ref=x_ref)
                except Exception:
                    pass
            # WBIC: 해당 stance 발의 lam_des 추종
            ld = [lam_des[c] for c in stance_feet]
            tau_w, lam, ok = self.wbic_track(stance_feet, ld)
            tau = np.zeros(8)
            if ok: tau[:] = tau_w
            else:  nfail += 1

            if mode == 'SS':   # 스윙 발 IK-PD
                if s < 0.02:
                    liftoff[swing] = self.foot_world(swing)[:2].copy()
                xi = com[:2] + vcom[:2] / omega
                tgt_raw = np.array([xi[0] + self.CP_KX * vcom[0], nom[swing][1]])
                step_vec = tgt_raw - liftoff[swing]
                n = np.linalg.norm(step_vec)
                if n > self.MAX_STEP:
                    step_vec *= self.MAX_STEP / n
                tgt = liftoff[swing] + step_vec
                p_start = np.array([liftoff[swing][0], liftoff[swing][1], GROUND_Z])
                p_end   = np.array([tgt[0], tgt[1], GROUND_Z])
                p_tgt = swing_foot_pos(s, p_start, p_end,
                                       np.array([vcom[0], vcom[1], 0.0]),
                                       step_height=STEP_HEIGHT, tau_land=1.0)
                q_sw_tgt = self.foot_ik(swing, p_tgt, d.qpos[LEG_QPOS[swing]])
                sl = slice(swing * 4, swing * 4 + 4)
                q_sw = d.qpos[LEG_QPOS[swing]]; dq_sw = d.qvel[LEG_QVEL[swing]]
                tau[sl] = np.clip(self.KP_SW * (q_sw_tgt - q_sw) - self.KD_SW * dq_sw,
                                  -TAU_MAX, TAU_MAX)

            if push and abs(t - push[0]) < m.opt.timestep:
                d.qvel[0] += push[1]
            d.ctrl[:] = tau
            t0 = time.time()
            mujoco.mj_step(m, d)
            if v is not None:
                if not v.is_running():
                    break
                if d.qpos[2] < 0.15 or d.qpos[2] > 0.9 or abs(d.qpos[3]) < 0.6:
                    settle_and_set_ref(self); self.scratch.qpos[:] = d.qpos[:]
                self.draw_grf_arrows(v, stance_feet, lam if ok else [None] * len(stance_feet))
                self.draw_com_marker(v)
                v.sync()
                dt = m.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
            if step % 100 == 0:
                lz = sum(l[2] for l in lam) if ok else 0
                print('t=%.2f bz=%.3f com=(%.3f,%.3f,%.3f) %s Σλz=%.0fN(des %.0f) ncon=%d %s'
                      % (t, d.qpos[2], com[0], com[1], com[2], mode, lz,
                         sum(lam_des[c][2] for c in stance_feet), d.ncon,
                         '' if ok else 'QPfail'))
        if v is not None:
            try: v.close()
            except Exception: pass
        print('mpc done. QPfail=%d final bz=%.3f upright=%.3f' % (nfail, d.qpos[2], d.qpos[3]))
        return d.qpos[2], d.qpos[3]

    # ───────────────────── LIPM / ICP 풋스텝 플래너 ─────────────────────
    def run_lipm(self, seconds=8.0, T_step=0.34, viewer=False, push=None,
                 clamp_y=0.05, clamp_x=0.12):
        """LIPM + Capture-Point 풋스텝 안정화.

        핵심: 스탠스 WBIC 는 높이+직립만 유지(수평 CoM 규제 0 → CoM 이 LIPM 진자처럼
        자유롭게 흔들림). 수평 CoM 은 '발을 어디 딛느냐'로 제어.
        풋스텝 = 스텝 종료 시점 ICP 예측치에 배치 (deadbeat capture).
        """
        import time
        m, d = self.m, self.d
        settle_and_set_ref(self)
        omega = float(np.sqrt(9.81 / self.z_ref))
        foot_y = {0: self.foot_world(0)[1], 1: self.foot_world(1)[1]}   # +0.138 / -0.138
        foot_x_nom = self.foot_world(0)[0]                              # -0.231
        liftoff = {0: self.foot_world(0).copy(), 1: self.foot_world(1).copy()}
        swing_tgt = {0: self.foot_world(0).copy(), 1: self.foot_world(1).copy()}
        planned = -1
        v = mujoco.viewer.launch_passive(m, d, key_callback=self._key_callback) if viewer else None
        if v is not None:
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0   # 내장 CoM 마커 OFF (커스텀과 중복 방지)
        if viewer:
            print('viewer open (lipm) — 창 닫으면 종료.')
        nfall = 0
        for step in range(int(seconds / m.opt.timestep)):
            t = d.time
            k = int(t / T_step); s = (t - k * T_step) / T_step
            swing = k % 2; stance = 1 - swing
            com = d.subtree_com[0].copy()
            vcom = self.com_jac() @ d.qvel

            # 스텝 시작 시 풋스텝 플랜 (ICP 예측 → deadbeat capture)
            if k != planned:
                planned = k
                liftoff[swing] = self.foot_world(swing).copy()
                p_st = self.foot_world(stance)[:2]
                xi0 = com[:2] + vcom[:2] / omega
                xi_end = p_st + (xi0 - p_st) * np.exp(omega * T_step)   # 종료 시 ICP
                tx = np.clip(xi_end[0], foot_x_nom - clamp_x, foot_x_nom + clamp_x)
                ty = foot_y[swing] + np.clip(xi_end[1] - foot_y[swing], -clamp_y, clamp_y)
                swing_tgt[swing] = np.array([tx, ty, GROUND_Z])

            # 스윙 발 궤적 (gait_sim baseline)
            p_tgt = swing_foot_pos(s, liftoff[swing], swing_tgt[swing],
                                   np.array([vcom[0], vcom[1], 0.0]),
                                   step_height=STEP_HEIGHT, tau_land=1.0)
            q_sw_tgt = self.foot_ik(swing, p_tgt, d.qpos[LEG_QPOS[swing]])

            # 스탠스: 높이+직립만 (수평 CoM 규제 0 → LIPM 자유 진자)
            com_ref = np.array([com[0], com[1], self.z_ref])
            tau_w, lam, ok = self.wbic_stance(contacts=[stance], com_ref=com_ref,
                                              kp=(0., 0., 120.), kd=(0., 0., 22.))
            tau = np.zeros(8)
            if ok: tau[:] = tau_w
            else:  nfall += 1
            sl = slice(swing * 4, swing * 4 + 4)
            tau[sl] = np.clip(self.KP_SW * (q_sw_tgt - d.qpos[LEG_QPOS[swing]])
                              - self.KD_SW * d.qvel[LEG_QVEL[swing]], -TAU_MAX, TAU_MAX)

            if push and abs(t - push[0]) < m.opt.timestep:
                d.qvel[0] += push[1]
            d.ctrl[:] = tau
            t0 = time.time()
            mujoco.mj_step(m, d)
            if v is not None:
                if not v.is_running():
                    break
                if d.qpos[2] < 0.15 or d.qpos[2] > 0.9 or abs(d.qpos[3]) < 0.6:
                    settle_and_set_ref(self); self.scratch.qpos[:] = d.qpos[:]; planned = -1
                self.draw_grf_arrows(v, [stance], [lam[0]] if ok else [None])
                self.draw_com_marker(v)
                v.sync()
                dt = m.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
            if step % 100 == 0:
                xi = com[:2] + vcom[:2] / omega
                print('t=%.2f bz=%.3f com=(%.3f,%.3f) ICP=(%.3f,%.3f) sw=%s ncon=%d %s'
                      % (t, d.qpos[2], com[0], com[1], xi[0], xi[1],
                         'HL' if swing == 0 else 'HR', d.ncon, '' if ok else 'QPfail'))
        if v is not None:
            try: v.close()
            except Exception: pass
        print('lipm done. fall=%d final bz=%.3f upright=%.3f' % (nfall, d.qpos[2], d.qpos[3]))
        return d.qpos[2], d.qpos[3]


# ────────────────────────────────────────────────────────────────
def settle_and_set_ref(ctrl: BipedWBIC, base_z=0.46):
    """Crouch home 자세로 초기화 → q_home / CoM / 높이 ref 기록.

    MJCF 의 'home' keyframe(무릎 굽힌 crouch) 을 로드. 곧게 편 q=0 은 leg Jacobian
    특이점 근처라 힘제어 발산 → home keyframe 으로 회피. keyframe 없으면 IK fallback.
    """
    d, m = ctrl.d, ctrl.m
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(m, d, kid)
    else:
        d.qpos[:] = 0; d.qpos[3] = 1; d.qpos[2] = base_z
        mujoco.mj_forward(m, d)
        tgt = {leg: np.array([-0.231, 0.138 * (1 if leg == 0 else -1), GROUND_Z])
               for leg in range(2)}
        for _ in range(200):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            for leg in range(2):
                p = ctrl.foot_world(leg)
                jp = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp, None, p, ctrl.foot_body[leg])
                J = jp[:, LEG_QVEL[leg]]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(3), tgt[leg] - p)
                q = d.qpos[LEG_QPOS[leg]] + 0.5 * dq
                for k, a in enumerate(LEG_QPOS[leg]):
                    lo, hi = m.jnt_range[m.dof_jntid[LEG_QVEL[leg][k]]]
                    d.qpos[a] = min(max(q[k], lo), hi)
    mujoco.mj_forward(m, d)
    ctrl.q_home = d.qpos[7:15].copy()
    ctrl.z_ref = d.subtree_com[0][2]
    ctrl.com_x_ref = d.subtree_com[0][0]
    return ctrl


def run(seconds=5.0, viewer=False):
    """Stage 1: 양발 스탠스 WBIC (검증된 부분). viewer=True 면 GUI."""
    ctrl = settle_and_set_ref(BipedWBIC('biped_wrapper.mjcf'))
    m, d = ctrl.m, ctrl.d
    Wtot = m.body_subtreemass[0]
    print('total mass %.3f kg  weight %.1f N  z_ref=%.3f com_x_ref=%.3f'
          % (Wtot, Wtot * 9.81, ctrl.z_ref, ctrl.com_x_ref))

    def control():
        tau, lam, ok = ctrl.wbic_stance(contacts=[0, 1])
        d.ctrl[:] = tau if ok else np.zeros(8)
        return ok, (sum(l[2] for l in lam) if ok else 0.0)

    if viewer:
        _viewer_loop(m, d, control, seconds, ctrl, draw_grf=True, feet=[0, 1])
    else:
        for step in range(int(seconds / m.opt.timestep)):
            ok, lamz = control(); mujoco.mj_step(m, d)
            if step % 100 == 0:
                com = d.subtree_com[0]
                print('t=%.2f base_z=%.3f com=(%.3f,%.3f,%.3f) Σλz=%.1fN ncon=%d %s'
                      % (d.time, d.qpos[2], com[0], com[1], com[2], lamz, d.ncon,
                         '' if ok else 'FAIL'))
        print('done. final base_z=%.3f upright_w=%.3f' % (d.qpos[2], d.qpos[3]))


def _viewer_loop(m, d, control_fn, seconds, ctrl, draw_grf=False, feet=(0, 1)):
    """passive viewer 에서 control_fn() 호출하며 실시간 step. 넘어지면 reset.

    draw_grf=True 면 매 프레임 ctrl.last_lam 을 발바닥에서 GRF 화살표로 표시.
    """
    import time
    # (mujoco.viewer imported at module top)
    with mujoco.viewer.launch_passive(m, d, key_callback=ctrl._key_callback) as v:
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0   # 내장 CoM 마커 OFF (커스텀과 중복 방지)
        print('viewer open — 창 닫으면 종료. 넘어지면 자동 리셋.'
              + ('  (GRF 화살표 ON)' if draw_grf else ''))
        while v.is_running():
            t0 = time.time()
            control_fn()
            mujoco.mj_step(m, d)
            # GRF 화살표 (control_fn 이 ctrl.last_lam 갱신) + CoM 마커
            if draw_grf:
                lam = getattr(ctrl, 'last_lam', None)
                ctrl.draw_grf_arrows(v, list(feet),
                                     lam if lam is not None else [None] * len(feet))
            else:
                v.user_scn.ngeom = 0
            ctrl.draw_com_marker(v)
            # auto-reset on fall
            if d.qpos[2] < 0.15 or d.qpos[2] > 0.9 or abs(d.qpos[3]) < 0.6:
                settle_and_set_ref(ctrl)
                if hasattr(ctrl, 'scratch'):
                    ctrl.scratch.qpos[:] = d.qpos[:]
            v.sync()
            dt = m.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


def run_march_viewer(seconds=1e9, **kw):
    ctrl = BipedMarch('biped_wrapper.mjcf')
    ctrl.run_march(seconds=seconds, viewer=True, **kw)


# ────────────────────────────────────────────────────────────────
# Stage 0: 모델/동역학 정합 검증 (WBIC 게인과 무관한 sanity check)
#   A 낙하   : MJCF 물리(중력/질량/지면)        — τ=0 시 자연 붕괴
#   B Jac FD : 접촉점·foot_jac 정합             — ‖J_fd−J‖ < 1e-4
#   C 무게   : GRF 부호·접촉 모델                — Σλz ≈ mg ±2%
#   D 잔차   : M,h,J,τ 운동방정식 정합          — base6 잔차 ≈ 0
# ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    import argparse
    os.environ.setdefault('DISPLAY', ':0')           # GUI 자동
    ap = argparse.ArgumentParser(description='biped WBIC balance — 인자 없이 실행하면 viewer')
    # 단계 순서: stance(1) → lipm(2.a) → march(2.b) → mpc(3)
    ap.add_argument('--mode', choices=['stance', 'lipm', 'march', 'mpc'], default='stance')
    ap.add_argument('--headless', action='store_true', help='창 없이 콘솔 로그만')
    ap.add_argument('--seconds', type=float, default=5.0)
    ap.add_argument('--tstep', type=float, default=0.34, help='lipm 스텝시간 [s] (기본 0.34, 수정안 0.18)')
    ap.add_argument('--yclamp', type=float, default=0.05, help='lipm 측면 풋스텝 클램프 [m] (기본 0.05, 수정안 0.20)')
    a = ap.parse_args()
    use_viewer = not a.headless                       # 기본 = viewer ON
    secs = 1e9 if use_viewer else a.seconds
    if a.mode == 'stance':
        run(seconds=secs, viewer=use_viewer)
    elif a.mode == 'lipm':
        BipedMarch('biped_wrapper.mjcf').run_lipm(seconds=secs, viewer=use_viewer,
                                                  T_step=a.tstep, clamp_y=a.yclamp)
    elif a.mode == 'march':
        BipedMarch('biped_wrapper.mjcf').run_march(seconds=secs, viewer=use_viewer)
    else:  # mpc
        BipedMarch('biped_wrapper.mjcf').run_mpc(seconds=secs, viewer=use_viewer)
    # Wayland/glfw 종료단계 teardown segfault 회피 — 즉시 클린 종료
    sys.stdout.flush()
    if use_viewer:
        os._exit(0)
