"""biped MPC + WBIC (B3) — Di Carlo SRBD MPC(mpc.hpp)를 2발로 이식 + WBIC 추종 + capture-point.

mature quad와 동일 구조: MPC(50Hz)가 균형 GRF 계획 → WBIC(500Hz)가 추종 + 스윙 발.
MPC 상태 x∈R¹³=[rpy, CoM, ω_world, v_com, g]. x_ref 속도=0 → CoM 속도를 GRF로 감쇠(수평 균형).
Q=[200,200,100,0,0,200,0,0,1,10,10,1,0], R=1e-6, N=14, DT=0.02 (params.html 동일).
헤드리스: python biped_mpc_wbic.py · 뷰어: VIEW=1 python biped_mpc_wbic.py
"""
from __future__ import annotations
import os, time, numpy as np, mujoco, mujoco.viewer
from qpsolvers import solve_qp
import biped_step as BS
from biped_step import T_STEP, DS_FRAC, STEP_H, SW_KP, SW_KD, GVEC
from biped_wbic import (STANCE_KD, W_ORI, W_POST, W_ANKLE, MU, MU_MARGIN,
                        LAMZ_MIN, DRV_PEAK, Q_HOME, ANKLE_IDX, base_rpy, tau_to_drive)

MPC_N, MPC_DT = 14, 0.02
Q_DIAG = np.array([200,200,100, 0,0,200, 0,0,1, 10,10,1, 0.0])   # TROT_Q
R_DIAG = np.array([1e-6, 1e-6, 1e-6])
W_LAM  = 10.0          # WBIC의 MPC GRF 추종 가중(mature 동일)
MPC_DECIM = int(round(MPC_DT / 0.002))   # 10 (500Hz sim, 50Hz MPC)
# ★점발 발목(ankle) posture 튜닝 노브 (2026-09-21). 기본값=종전 하드코딩값 보존(60/5/W_ANKLE).
#   whip 원인 = kd 5·kp 60 → ζ=0.32 저감쇠. env 로 스윕해 임계감쇠(ζ≈1: kd≈2√kp) 찾는다.
ANK_KP = float(os.environ.get('ANK_KP', 60.0))
ANK_KD = float(os.environ.get('ANK_KD', 5.0))
ANK_W  = float(os.environ.get('ANK_W',  20.0))   # = W_ANKLE 기본


def euler_to_R(r, p, y):
    cr,sr,cp,sp,cy,sy = np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx
def skew(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])


class BipedMPCWBIC(BS.BipedStep):
    def __init__(self, mjcf=BS.BW.MJCF):
        super().__init__(mjcf)
        self.nfeet = 2
        self.lam = np.zeros((2, 3))     # MPC GRF (leg 0=HL,1=HR)
        self._k = 0
        self.vx_cmd = float(os.environ.get('VX', '0.0'))   # 전진 명령[m/s] (body +x)
        self.vy_cmd = float(os.environ.get('VY', '0.0'))   # ★좌우 명령[m/s] (body +y=좌)
        self.wz_cmd = float(os.environ.get('WZ', '0.0'))   # ★선회 명령[rad/s] (+=좌회전)
        self.yaw_des = 0.0                                  # heading 목표(wz로 갱신)
        self.yaw_hold = None                                # ★정지/직진 heading latch(자유 yaw 표류 방지, 17-DOF 방식)
        self.head_lead = 0.15                               # ★헤딩 lead clamp[rad](발이 실제보다 앞서는 최대)
        self.static_stand = os.environ.get('FLAT_STATIC', '1') != '0'  # ★평발 정지=정적 양발 지지(0=연속스텝 실험)
        # ★접촉모드 자세 램프(앉기/서기): q_home·높이를 목표로 서서히 이동(발목 눕힘=앉기, 세움=서기).
        # ★CZ_1PT 0.50→0.48 (2026-08-28 높이 스윕): 0.50 은 낙상 경계였다 — 0.52 는 vx0.30 낙상,
        #   0.50 은 vx0.35 낙상+0.30 미끄러짐(실효 0.175). 0.48 은 vx0.35 완주·실효 0.28~0.32,
        #   배터리 8/8 유지, 무릎 +2Nm 뿐. (0.46/0.44 도 완주하나 tilt 증가 — 0.48 이 최적점)
        self.CZ_2PT, self.CZ_1PT = 0.362, 0.48        # 모드별 목표 CoM 높이(평발 낮음/점발 crouch)
        # ★스윙 위치제어 하이브리드 게인(관절 Nm/rad · Nm·s/rad). 0 = 꺼짐(종전 동작)
        self.SWING_KP = float(os.environ.get('SWING_KP', 0.0))
        self.SWING_KD = float(os.environ.get('SWING_KD', 0.0))
        self._ik_scratch = None
        # WBIC 스윙 태스크 게인(기본 = biped_step 값). 하이브리드에선 낮춰 배분한다
        self.SW_KP_EFF = float(os.environ.get('SW_KP_EFF', SW_KP))
        self.SW_KD_EFF = float(os.environ.get('SW_KD_EFF', SW_KD))
        self.q_home_tgt = self.q_home.copy()
        self.com_z_tgt = self.CZ_2PT if self.contact_mode == '2pt' else self.CZ_1PT
        self.ramp_q = 0.9                              # 자세 램프 속도[rad/s](≈1.3s에 발목 눕힘/세움)
        self.ramp_z = 0.06                             # 높이 램프 속도[m/s]

    # ── MPC 셋업 ──
    def setup_mpc(self):
        mujoco.mj_forward(self.m, self.d)
        self.mpc_decim = max(1, int(round(MPC_DT / self.m.opt.timestep)))   # 제어율에 맞게(1kHz→20, 500Hz→10)
        self.mass = self.m.body_subtreemass[0]
        self.I_body = self.compute_Icom()
        self.mu = MU * MU_MARGIN
        self.lamz_max = 2.0 * self.mass * GVEC

    def compute_Icom(self):
        d, m = self.d, self.m
        com = d.subtree_com[0]; I = np.zeros((3, 3))
        for b in range(1, m.nbody):
            ms = m.body_mass[b]
            if ms <= 0: continue
            r = d.xipos[b] - com
            Rb = d.ximat[b].reshape(3, 3)
            Ib = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
            I += Ib + ms * (r @ r * np.eye(3) - np.outer(r, r))
        return I

    def body_x0(self):
        d = self.d
        R = d.xmat[1].reshape(3, 3).copy() if False else np.zeros((3,3))
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, d.qpos[3:7]); R = Rm.reshape(3, 3)
        pitch = np.arcsin(np.clip(-R[2,0], -1, 1))
        roll = np.arctan2(R[2,1], R[2,2]); yaw = np.arctan2(R[1,0], R[0,0])
        Jc = np.zeros((3, self.nv)); mujoco.mj_jacSubtreeCom(self.m, d, Jc, 0)
        vcom = Jc @ d.qvel
        omega_w = R @ d.qvel[3:6]
        return np.array([roll, pitch, yaw, *d.subtree_com[0], *omega_w, *vcom, -9.81]), yaw

    # ── Di Carlo SRBD MPC (mpc.hpp 이식, 2발) ──
    def mpc_qp_plan(self, x0, cs, fp, x_ref):
        nx, nf = 13, self.nfeet; nu = 3 * nf; N = MPC_N
        roll0, pitch0, yaw0 = x0[0], x0[1], x0[2]
        Rn = euler_to_R(roll0, pitch0, yaw0)
        Iw_inv = np.linalg.inv(Rn @ self.I_body @ Rn.T)
        Ac = np.zeros((13, 13))
        cyw, syw = np.cos(yaw0), np.sin(yaw0)
        Ac[0:3, 6:9] = np.array([[cyw,syw,0],[-syw,cyw,0],[0,0,1]])   # Rz(yaw)ᵀ
        Ac[3:6, 9:12] = np.eye(3); Ac[11, 12] = 1.0
        Ad = np.eye(13) + MPC_DT * Ac
        Adp = [np.eye(13)]
        for _ in range(N): Adp.append(Ad @ Adp[-1])
        Bc = []
        for k in range(N):
            B = np.zeros((13, nu))
            for i in range(nf):
                if cs[k][i]:
                    r = fp[k][i]
                    B[6:9, i*3:i*3+3] = Iw_inv @ skew(r)
                    B[9:12, i*3:i*3+3] = np.eye(3) / self.mass
            Bc.append(MPC_DT * B)
        Aq = np.zeros((N*nx, nx)); Bq = np.zeros((N*nx, N*nu))
        for i in range(N):
            Aq[i*nx:(i+1)*nx] = Adp[i+1]
            for j in range(i+1):
                Bq[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = Adp[i-j] @ Bc[j]
        X_ref = np.tile(x_ref, N)
        err0 = Aq @ x0 - X_ref
        Qbar = np.tile(Q_DIAG, N)
        QBq = Qbar[:, None] * Bq; Qerr = Qbar * err0
        Rbar = np.tile(R_DIAG, N * nf)
        H = 2.0 * (Bq.T @ QBq); H[np.diag_indices_from(H)] += 2.0 * Rbar
        H = 0.5 * (H + H.T); H[np.diag_indices_from(H)] += 1e-8
        f = 2.0 * (Bq.T @ Qerr)
        # 제약
        Grows=[]; hvals=[]; Arows=[]; bvals=[]
        for k in range(N):
            for i in range(nf):
                col = k*nu + i*3
                if cs[k][i]:
                    g=np.zeros(N*nu); g[col+2]=-1; Grows.append(g); hvals.append(-LAMZ_MIN)
                    for sx in (1,-1):
                        g=np.zeros(N*nu); g[col]=sx; g[col+2]=-self.mu; Grows.append(g); hvals.append(0.0)
                        g=np.zeros(N*nu); g[col+1]=sx; g[col+2]=-self.mu; Grows.append(g); hvals.append(0.0)
                    g=np.zeros(N*nu); g[col+2]=1; Grows.append(g); hvals.append(self.lamz_max)
                else:
                    for dax in range(3):
                        a=np.zeros(N*nu); a[col+dax]=1; Arows.append(a); bvals.append(0.0)
        G=np.array(Grows); h=np.array(hvals)
        A=np.array(Arows) if Arows else None; b=np.array(bvals) if bvals else None
        u = solve_qp(H, f, G, h, A, b, solver='quadprog')
        lam = np.zeros((nf, 3))
        if u is not None:
            for i in range(nf): lam[i] = u[i*3:i*3+3]
        else:                                    # fallback: 균등 수직력
            ns = sum(cs[0]);
            if ns > 0:
                fz = self.mass * GVEC / ns
                for i in range(nf):
                    if cs[0][i]: lam[i, 2] = fz
        return lam

    def mpc_grf(self, contacts):
        x0, yaw0 = self.body_x0()
        com = self.d.subtree_com[0]
        frel = [self.foot_center(i) - com for i in range(self.nfeet)]   # ★MPC moment arm=발 중심
        fp = [frel] * MPC_N
        cur = [1 if i in contacts else 0 for i in range(self.nfeet)]   # 현재 접촉 유지 가정(event-based)
        cs = [cur] * MPC_N
        # x_ref: 자세수평·heading(yaw_des)·wz선회·목표높이·body속도(vx전진·vy좌우)를 world로
        # ★속도명령=실제 base yaw 기준(base-relative, 17-DOF yaw_m 방식)·헤딩참조=yaw_des
        qc = self.d.qpos[3:7]
        yaw_act = np.arctan2(2*(qc[0]*qc[3]+qc[1]*qc[2]), 1-2*(qc[2]**2+qc[3]**2))
        cya, sya = np.cos(yaw_act), np.sin(yaw_act)
        vx_w = cya * self.vx_cmd - sya * self.vy_cmd
        vy_w = sya * self.vx_cmd + cya * self.vy_cmd
        x_ref = np.array([0, 0, self.yaw_des, com[0], com[1], self.com_ref[2], 0,0, self.wz_cmd, vx_w, vy_w, 0, -9.81])
        return self.mpc_qp_plan(x0, cs, fp, x_ref)

    # ── WBIC (MPC lam 추종) — biped_step.wbic_track 오버라이드 ──
    def wbic_track(self, contacts, swing, lam):
        d, m, nv, nu = self.d, self.m, self.nv, self.nu
        cpts = self.contact_pts(contacts)          # ★접촉점 [(geom,body,foot)] — 2접촉=stance 발당 2점
        Kc = len(cpts); nz = nv + 3*Kc; sl = lambda k: nv + 3*k
        cjac = [self.foot_jac_at(g, b) for (g, b, f) in cpts]
        foot_npts = {}                             # 발별 실제 접촉점 수(MPC GRF 분배용, 적응형)
        for (_g, _b, _f) in cpts: foot_npts[_f] = foot_npts.get(_f, 0) + 1
        M = np.zeros((nv, nv)); mujoco.mj_fullM(m, M, d.qM)
        h = d.qfrc_bias.copy(); qv = d.qvel.copy()
        P = np.zeros((nz, nz)); g = np.zeros(nz); sw_vidx = set()
        for leg, (ptgt, vtgt) in swing.items():
            J = self.foot_jac_center(leg)          # ★swing=발 중심 추종(2접촉이면 2구 평균)
            # ★하이브리드 배분(2026-08-28): 스윙을 관절 위치제어로 넘기면 WBIC 태스크는
            #   낮추거나 꺼야 한다 — 둘이 같은 다리를 동시에 몰면 서로 감쇠시킨다.
            accel = self.SW_KP_EFF*(ptgt - self.foot_center(leg)) + self.SW_KD_EFF*(vtgt - J @ qv)
            P[:nv,:nv] += 90.0*(J.T @ J); g[:nv] -= 90.0*(J.T @ accel)
            for t in range(4): sw_vidx.add(6 + leg*4 + t)
            if self.has_heel and self.contact_mode == '2pt':   # ★평발 walk 시만 swing 발 수평(점발 보행엔 부적용)
                qb = d.qpos[3:7]
                yw = np.arctan2(2*(qb[0]*qb[3]+qb[1]*qb[2]), 1-2*(qb[2]**2+qb[3]**2))
                qyaw = np.array([np.cos(yw/2), 0, 0, np.sin(yw/2)])
                ftgt = np.zeros(4); mujoco.mju_mulQuat(ftgt, qyaw, self.foot_home_quat[leg])
                oe = np.zeros(3); mujoco.mju_subQuat(oe, d.xquat[self.fbody[leg]], ftgt)
                jr = np.zeros((3, nv)); mujoco.mj_jac(m, d, None, jr, d.xpos[self.fbody[leg]], self.fbody[leg])
                a_rot = 120*(-oe) - 15*(jr @ qv)
                P[:nv,:nv] += 15.0*(jr.T @ jr); g[:nv] -= 15.0*(jr.T @ a_rot)   # swing 발 레벨링(weight 15)
        Jc = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0); self.Jc_cache = Jc
        qc = d.qpos[3:7]
        yaw = np.arctan2(2*(qc[0]*qc[3]+qc[1]*qc[2]), 1-2*(qc[2]**2+qc[3]**2))
        qlev = np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)])
        oerr = np.zeros(3); mujoco.mju_subQuat(oerr, qc, qlev)
        for j in range(2):
            a = 150*(-oerr[j]) - 20*qv[3+j]; P[3+j,3+j]+=W_ORI; g[3+j]-=W_ORI*a
        a_yaw = 150*(-oerr[2]) - 20*qv[5]; P[5,5]+=1.0; g[5]-=1.0*a_yaw
        zref = self.com_ref[2]; a_z = 300*(zref - d.subtree_com[0][2]) - 30*(Jc @ qv)[2]
        P[:nv,:nv]+=400.0*np.outer(Jc[2],Jc[2]); g[:nv]-=400.0*a_z*Jc[2]   # ★단일지지 sink 억제(150→400·200→300)
        for j in range(nu):
            is_ank = j in ANKLE_IDX
            kp_p = ANK_KP if is_ank else 60.0        # ★발목만 튜닝 노브(env). 나머지=종전 60/5
            kd_p = ANK_KD if is_ank else 5.0
            a = kp_p*(self.q_home[j]-d.qpos[7+j]) - kd_p*qv[6+j]
            w = ANK_W if is_ank else (5.0 if (6+j) in sw_vidx else W_POST)
            P[6+j,6+j]+=w; g[6+j]-=w*a
        P[:nv,:nv]+=1e-3*np.eye(nv)
        for k in range(Kc):                        # ★MPC GRF 추종 (발 GRF를 접촉점 수로 분배)
            P[sl(k):sl(k)+3, sl(k):sl(k)+3] += W_LAM*np.eye(3)
            g[sl(k):sl(k)+3] -= W_LAM*lam[cpts[k][2]]/foot_npts[cpts[k][2]]
        neq = 6+3*Kc; A=np.zeros((neq,nz)); bb=np.zeros(neq)
        A[:6,:nv]=M[:6,:]; bb[:6]=-h[:6]
        for k in range(Kc):
            A[:6, sl(k):sl(k)+3] = -cjac[k][:,:6].T
            A[6+3*k:6+3*k+3,:nv] = cjac[k]; bb[6+3*k:6+3*k+3] = -STANCE_KD*(cjac[k]@qv)
        rows=[]; hh=[]; mu=MU*MU_MARGIN
        for k in range(Kc):
            o=sl(k)
            for sx,sy in [(1,0),(-1,0),(0,1),(0,-1)]:
                r=np.zeros(nz); r[o]=sx; r[o+1]=sy; r[o+2]=-mu; rows.append(r); hh.append(0.0)
            r=np.zeros(nz); r[o+2]=-1; rows.append(r); hh.append(-LAMZ_MIN)
        Tm=np.zeros((nu,nz)); Tm[:,:nv]=M[6:,:]
        for k in range(Kc): Tm[:,sl(k):sl(k)+3]=-cjac[k][:,6:].T
        # ★토크한계를 **드라이브 공간**에 건다 (2026-08-13). 무릎 드라이브가 τ_calf−τ_foot 을
        #   짊어지므로 관절공간 박스는 실기가 못 내는 조합을 허용한다. calf 행만 전단한다.
        for i in range(nu):
            shear = (i % 4 == 2 and i + 1 < nu)                 # calf 축만
            ri = Tm[i] - Tm[i+1] if shear else Tm[i]
            oi = h[6+i] - h[6+i+1] if shear else h[6+i]
            rows.append(ri); hh.append(self.drv_peak[i]-oi)
            rows.append(-ri); hh.append(self.drv_peak[i]+oi)
        G=np.array(rows); hh=np.array(hh)
        P=0.5*(P+P.T)+1e-8*np.eye(nz)
        _multi = any(n > 1 for n in foot_npts.values())        # ★실제 발당 2점 접지 시만 rank-deficient
        _solver='proxqp' if _multi else 'quadprog'             # 평발=proxqp, 점발=quadprog(강건·결정적)
        x=solve_qp(P,g,G,hh,A,bb,solver=_solver)
        if x is None: return self.wbic_stance()
        qdd=x[:nv]; tau=M[6:,:]@qdd + h[6:]
        for k in range(Kc): tau -= cjac[k][:,6:].T @ x[sl(k):sl(k)+3]
        d.ctrl[:]=np.clip(self._foot_comp(tau_to_drive(tau)),-self.drv_peak,self.drv_peak); return True   # ★ctrl=드라이브 토크

    # ★★스윙 위치제어 하이브리드 (2026-08-28 · 기본 꺼짐 SWING_KP=0)
    #   구조: **지지=토크(WBIC) · 스윙=위치(IK 목표각 + 관절 PD)**.
    #   왜: 스윙은 무부하 궤적추종이라 힘제어가 필요 없고, 위치 피드백이 우리 플랜트의
    #   약점(α 0.85 전달손실·마찰 0.6~0.9Nm·foot 상수결손)을 **되먹임으로 걷어낸다**.
    #   또 현재 스윙은 kp=0·dq_des=0 이라 드라이브 kd 가 41Nm 로 제동하는 문제가 있는데,
    #   IK 목표를 주면 **진짜 q_des·q̇_des** 가 생겨 그 문제가 사라진다
    #   (종전 "q̇_cmd 전송" 안이 기각된 이유 = dq_ch 가 측정속도였기 때문).
    #   ⚠실기 전제: 백래쉬. 무릎 유격 7° = 발끝 42mm 라 **기어 재조임 전에는 무의미**하다.
    #   여기서는 드라이브가 할 PD 를 sim 에서 재현한다(τ += kp·err + kd·derr, 스윙 4축만).
    def _foot_pt(self, dat, leg):
        """접촉 기준점 — 평발(2점)이면 두 구의 중심, 점발이면 발끝 구. foot_center 와 같은 규약."""
        if self.has_heel and self.sph2 is not None:
            return 0.5 * (dat.geom_xpos[self.sph[leg]] + dat.geom_xpos[self.sph2[leg]])
        return dat.geom_xpos[self.sph[leg]]

    def _swing_ik(self, leg, p_tgt, iters=12):
        """DLS IK — 스윙 4관절로 발 중심을 p_tgt 에 (base 는 현재 상태 고정)."""
        m, d = self.m, self.d
        if self._ik_scratch is None:
            self._ik_scratch = mujoco.MjData(m)
        s = self._ik_scratch
        s.qpos[:] = d.qpos[:]; s.qvel[:] = 0.0
        vidx = list(range(6 + 4 * leg, 6 + 4 * leg + 4))
        qidx = list(range(7 + 4 * leg, 7 + 4 * leg + 4))
        q = np.array(d.qpos[qidx], float)
        for _ in range(iters):
            s.qpos[qidx] = q
            mujoco.mj_kinematics(m, s); mujoco.mj_comPos(m, s)
            p = self._foot_pt(s, leg)
            e = np.asarray(p_tgt, float) - p
            if np.linalg.norm(e) < 1e-4:
                break
            jp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, s, jp, None, p, self.fbody[leg])
            J = jp[:, vidx]
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
            for k, vi in enumerate(vidx):                      # 관절한계 클램프
                lo, hi = m.jnt_range[m.dof_jntid[vi]]
                if hi > lo: q[k] = min(max(q[k], lo), hi)
        # 목표속도: q̇ = J⁺ v  (같은 자세에서)
        s.qpos[qidx] = q; mujoco.mj_kinematics(m, s); mujoco.mj_comPos(m, s)
        jp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, s, jp, None, self._foot_pt(s, leg), self.fbody[leg])
        return q, jp[:, vidx]

    def _apply_swing_pd(self, leg, q_des, J, v_tgt):
        """드라이브측 관절 PD 를 재현 — 스윙 4축의 드라이브 토크에 가산."""
        if self.SWING_KP <= 0.0: return
        m, d = self.m, self.d
        qidx = list(range(7 + 4 * leg, 7 + 4 * leg + 4))
        vidx = list(range(6 + 4 * leg, 6 + 4 * leg + 4))
        dq_des = J.T @ np.linalg.solve(J @ J.T + 1e-4*np.eye(3), np.asarray(v_tgt, float))
        tau_pd = self.SWING_KP * (q_des - d.qpos[qidx]) + self.SWING_KD * (dq_des - d.qvel[vidx])
        full = np.zeros(8); full[4*leg:4*leg+4] = tau_pd
        d.ctrl[:] = np.clip(d.ctrl + tau_to_drive(full), -self.drv_peak, self.drv_peak)

    def reset(self):
        super().reset()
        self.q_home_tgt = self.q_home.copy()           # 램프 목표 = 스폰 자세(램프 없이 시작)
        self.com_z_tgt = self.CZ_2PT if self.contact_mode == '2pt' else self.CZ_1PT
        self.com_ref[2] = self.com_z_tgt

    def set_contact_mode(self, mode):
        """★접촉모드 전환(같은 모델·같은 제어기): 1pt=점발(동적보행)·2pt=평발(정적 rest).
        발목·높이 목표만 바꿈 → control()이 앉기/서기 동작으로 서서히 전환."""
        if mode not in ('1pt', '2pt') or mode == self.contact_mode:
            return
        self.contact_mode = mode
        self.q_home_tgt = (BS.BW.Q_HOME_FLAT if mode == '2pt' else BS.BW.Q_HOME).copy()
        self.com_z_tgt = self.CZ_2PT if mode == '2pt' else self.CZ_1PT

    def control(self, dt):
        qc = self.d.qpos[3:7]                        # 실제 yaw
        yaw_act = np.arctan2(2*(qc[0]*qc[3]+qc[1]*qc[2]), 1-2*(qc[2]**2+qc[3]**2))
        # ★자세 램프(앉기/서기 동작): q_home 발목·CoM 높이를 모드 목표로 서서히 이동
        self.q_home = self.q_home + np.clip(self.q_home_tgt - self.q_home, -self.ramp_q*dt, self.ramp_q*dt)
        self.com_ref[2] += float(np.clip(self.com_z_tgt - self.com_ref[2], -self.ramp_z*dt, self.ramp_z*dt))
        # ★정적 양발 지지 = 2점 목표 + 발이 실제 평발(양발 heel+toe 접지)일 때만. 앉기(1pt→2pt) 중엔
        #   아직 발끝이라 스텝 유지하며 발목을 내려 heel 착지 → 평발 되면 정적. 서기(2pt→1pt)는 스텝(발끝 기립).
        feet_flat = all(sum(1 for g in self.foot_spheres[f]
                            if self.d.geom_xpos[g][2] < self.m.geom_size[g][0] + 0.012) >= 2 for f in range(2))
        if self.has_heel and self.contact_mode == '2pt' and feet_flat:
            fc = 0.5 * (self.foot_center(0) + self.foot_center(1))
            self.com_ref[0], self.com_ref[1] = fc[0], fc[1]   # 현재 지지중심 홀드
            self.wbic_stance()
            self.t_ss = 0.0                                   # 보행 재개용 위상 초기화
            self.com0 = self.d.subtree_com[0][:2].copy()
            self.liftoff = [None, None]; self.yaw_hold = yaw_act
            self.t += dt; return
        if abs(self.wz_cmd) > 0.02:                  # ★선회 중: 명령 적분 + 리드 클램프(폭주방지)
            self.yaw_des += self.wz_cmd * dt
            lag = np.arctan2(np.sin(self.yaw_des - yaw_act), np.cos(self.yaw_des - yaw_act))
            self.yaw_des = yaw_act + np.clip(lag, -self.head_lead, self.head_lead)
            self.yaw_hold = None
        else:                                        # ★선회 외 전부(정지/전후진/측방): heading latch, 복원오차 head_lead 제한(base0.50서 측방도 안정)

            if self.yaw_hold is None: self.yaw_hold = yaw_act
            err = np.arctan2(np.sin(self.yaw_hold - yaw_act), np.cos(self.yaw_hold - yaw_act))
            self.yaw_des = yaw_act + np.clip(err, -self.head_lead, self.head_lead)
        cya, sya = np.cos(yaw_act), np.sin(yaw_act)   # ★속도명령=실제 base yaw 기준(base-relative)
        self.com0[0] += (cya * self.vx_cmd - sya * self.vy_cmd) * dt   # ★복귀목표를 body속도(vx전진·vy좌우)로 이동
        self.com0[1] += (sya * self.vx_cmd + cya * self.vy_cmd) * dt
        stance, swing_leg, s = self.step_gait(dt)   # event-based (측방 DCM 동기)
        if self._k % self.mpc_decim == 0:
            self.lam = self.mpc_grf(stance)          # 50Hz MPC (현재 stance 기준)
        self._k += 1
        if self.liftoff[swing_leg] is None:
            self.liftoff[swing_leg] = self.foot_center(swing_leg).copy()
        p, v = self.swing_traj(swing_leg, s)
        self.wbic_track(stance, {swing_leg: (p, v)}, self.lam)
        if self.SWING_KP > 0.0:                       # ★스윙만 위치제어 가산
            q_des, J = self._swing_ik(swing_leg, p)
            self._apply_swing_pd(swing_leg, q_des, J, v)
        self.t += dt


def main():
    c = BipedMPCWBIC(); c.reset(); c.setup_mpc()
    m, d = c.m, c.d
    print(f"biped MPC+WBIC · mass={c.mass:.2f} · I_body diag={np.round(np.diag(c.I_body),4)} · com_ref={np.round(c.com_ref,3)}")
    T = float(os.environ.get('T', 5.0)); dt = m.opt.timestep
    view = os.environ.get('VIEW','0')=='1'
    viewer = mujoco.viewer.launch_passive(m, d) if view else None
    z0 = d.qpos[2]; maxtilt = 0.0; fell = None
    for i in range(int(T/dt)):
        c.control(dt); mujoco.mj_step(m, d)
        rpy = base_rpy(d.qpos[3:7]); maxtilt = max(maxtilt, np.hypot(rpy[0], rpy[1]))
        if viewer is not None and i % 10 == 0: viewer.sync()
        if d.qpos[2] < 0.2: fell = i*dt; break
    rpy = base_rpy(d.qpos[3:7])
    if fell: print(f"❌ 낙상 @ {fell:.2f}s")
    print(f"t={min(i*dt,T):.2f}s · base z={d.qpos[2]:.3f}(Δ{d.qpos[2]-z0:+.3f}) xy=({d.qpos[0]:+.3f},{d.qpos[1]:+.3f}) tilt now={np.hypot(rpy[0],rpy[1]):.1f}° max={maxtilt:.1f}°")
    print("✅ MPC+WBIC 균형 유지" if fell is None and maxtilt < 25 and d.qpos[2] > 0.4 else "⚠️ 아직 불안정")
    if viewer is not None:
        while viewer.is_running():
            c.control(dt); mujoco.mj_step(m, d); viewer.sync(); time.sleep(dt)


if __name__ == '__main__':
    main()
