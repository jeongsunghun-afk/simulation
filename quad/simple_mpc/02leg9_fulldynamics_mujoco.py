import numpy as np
import mujoco as _mj
import os as _os0
_GO2_MJCF=_os0.environ.get("MJCF","/home/jsh/문서/jsh/simulation/quad/quad_real_pt.mjcf")  # MJCF=...quad_real_pt.mjcf = 단일구 점접촉(OCP와 일치)
_PIN2MJ=[8,9,10,11,12,13,0,1,2,3,4,5,6,7]  # pin(FL,FR,HL,HR)→mjcf(HL,HR,FL,FR)
class SportClient:
    """Unitree SDK 고수준 인터페이스(unitree_sdk2py SportClient) 흉내 — 연속 속도명령.
       저수준(LowCmd/LowState kp/kd/tau)은 MujocoRobot.execute가 담당."""
    def __init__(self): self.vx=0.0; self.vy=0.0; self.vyaw=0.0
    def Move(self, vx, vy, vyaw):          # 고수준 속도명령 (전진 m/s, 측방 m/s, 선회 rad/s)
        self.vx=float(vx); self.vy=float(vy); self.vyaw=float(vyaw)
    def StopMove(self): self.vx=self.vy=self.vyaw=0.0          # 정지
    def BalanceStand(self): self.StopMove()                    # 제자리 균형
    def velocity_base(self):               # → MPC velocity_base 6벡터 [vx,vy,vz,ωx,ωy,ωz]
        v=np.zeros(6); v[0]=self.vx; v[1]=self.vy; v[5]=self.vyaw; return v
class LowCmd:
    """Unitree 저수준 모터명령(unitree_sdk2py LowCmd) 흉내 — 관절별(pin 14-DOF).
       실토크 τ_i = kp_i·(q_i−q_meas_i) + kd_i·(dq_i−dq_meas_i) + tau_i  (Unitree 모터명령식)."""
    def __init__(self, nu):
        self.q=np.zeros(nu); self.dq=np.zeros(nu)         # 목표 위치/속도
        self.kp=np.zeros(nu); self.kd=np.zeros(nu)         # PD 게인
        self.tau=np.zeros(nu)                              # 피드포워드 토크
class LowState:
    """Unitree 저수준 상태(unitree_sdk2py LowState) 흉내 — 관절 q/dq/tau_est + IMU(quat,gyro)."""
    def __init__(self): self.q=None; self.dq=None; self.tau_est=None; self.quat=None; self.gyro=None
class MujocoRobot:
    """simple-mpc device(BulletRobot) 인터페이스를 MuJoCo로 구현. 토크는 KinodynamicsID(TSID) 출력.
       pin↔mujoco: go2 관절 순서 동일(재정렬 X), 베이스 quat [w,x,y,z]↔[x,y,z,w], lin world→local(R^T)."""
    def __init__(self, q0, dt_simu, view=False):
        self.m=_mj.MjModel.from_xml_path(_GO2_MJCF); self.m.opt.timestep=dt_simu
        import os as _o2                                   # 접촉모델 매칭(컨트롤러 강체가정 ↔ MuJoCo soft)
        if _o2.environ.get("CONE"): self.m.opt.cone=int(_o2.environ["CONE"])
        if _o2.environ.get("STIFF"): self.m.geom_solref[:,0]=float(_o2.environ["STIFF"]); self.m.geom_solref[:,1]=1.0
        if _o2.environ.get("FRIC"): self.m.geom_friction[:,0]=float(_o2.environ["FRIC"])   # 발 접촉마찰 override(속도레버)
        self.d=_mj.MjData(self.m); self.nu=self.m.nu
        self._set(q0); self.viewer=None; self.markers=[]; self.cmd_v=np.zeros(6)
        self.sport=SportClient()                                 # 고수준 속도명령(Unitree SportClient.Move)
        # 발 충돌구 geom id (foot_contact_link body의 SPHERE) — 궤적·슬립진단용(항상)
        self.foot_gids=[]
        for _L in ['FL','FR','HL','HR']:
            _bid=_mj.mj_name2id(self.m,_mj.mjtObj.mjOBJ_BODY,_L+'_foot_contact_link')
            for _g in range(self.m.ngeom):
                if self.m.geom_bodyid[_g]==_bid and self.m.geom_type[_g]==_mj.mjtGeom.mjGEOM_SPHERE:
                    self.foot_gids.append(_g); break
        if view:
            import mujoco.viewer as _v; self.viewer=_v.launch_passive(self.m,self.d,key_callback=self._key)
            self.viewer.opt.flags[_mj.mjtVisFlag.mjVIS_PERTOBJ]=1     # Ctrl+드래그 외력 박스 표시
            from collections import deque as _dq
            _tn=int(_o2.environ.get("TRAIL_N","300"))
            self.foot_trail=[_dq(maxlen=_tn) for _ in self.foot_gids]
            self.base_trail=_dq(maxlen=_tn)
    def _set(self,q):
        self.d.qpos[0:3]=q[0:3]; x,y,z,w=q[3:7]; self.d.qpos[3:7]=[w,x,y,z]
        import numpy as _np0;
        _tmp=_np0.zeros(self.nu); _tmp[_PIN2MJ]=q[7:7+self.nu]; self.d.qpos[7:7+self.nu]=_tmp; self.d.qvel[:]=0.0
        _mj.mj_forward(self.m,self.d)
    def initializeJoints(self,q0): self._set(q0)
    def resetState(self,q0): self._set(q0)
    def measureState(self):
        d=self.d; import numpy as _np
        qp=_np.zeros(self.m.nq); vp=_np.zeros(self.m.nv)
        qp[0:3]=d.qpos[0:3]; w,x,y,z=d.qpos[3:7]; qp[3:7]=[x,y,z,w]
        R=_np.zeros(9); _mj.mju_quat2Mat(R,d.qpos[3:7]); R=R.reshape(3,3)
        vp[0:3]=R.T@d.qvel[0:3]; vp[3:6]=d.qvel[3:6]
        qp[7:]=_np.asarray(d.qpos[7:7+self.nu])[_PIN2MJ]; vp[6:]=_np.asarray(d.qvel[6:6+self.nu])[_PIN2MJ]
        return qp, vp
    def read_low_state(self):                # Unitree 저수준 상태 읽기(관절 q/dq/tau_est + IMU quat/gyro)
        import numpy as _np
        qp, vp = self.measureState()
        st = LowState()
        st.q = qp[7:7+self.nu].copy(); st.dq = vp[6:6+self.nu].copy()      # 관절(pin order)
        st.quat = qp[3:7].copy(); st.gyro = vp[3:6].copy()                  # IMU(자세[xyzw]·각속도)
        st.tau_est = _np.asarray(self.d.qfrc_actuator[6:6+self.nu])[_PIN2MJ].copy()
        return st
    def write_low_cmd(self, cmd):            # Unitree 저수준 명령 적용: τ=kp(q-q_meas)+kd(dq-dq_meas)+tau
        import numpy as _np
        qp, vp = self.measureState()
        q = qp[7:7+self.nu]; dq = vp[6:6+self.nu]
        tau = _np.asarray(cmd.kp)*(cmd.q - q) + _np.asarray(cmd.kd)*(cmd.dq - dq) + _np.asarray(cmd.tau)
        self.execute(tau)                    # 토크 적용(_PIN2MJ 재정렬·step·뷰어)
    def execute(self,tau):
        import numpy as _np, os as _o3
        _um=_np.zeros(self.nu); _um[_PIN2MJ]=_np.asarray(tau).ravel()[:self.nu]; self.d.ctrl[:]=_um
        if self.viewer:                                          # Ctrl+드래그 외력 적용(드래그중만)
            if self.viewer.perturb.active: _mj.mjv_applyPerturbForce(self.m,self.d,self.viewer.perturb)
            else: self.d.xfrc_applied[:]=0.0
        _mj.mj_step(self.m,self.d)
        if self.viewer:
            self._vc=getattr(self,'_vc',0)+1
            _every=int(_o3.environ.get("RENDER_EVERY","10"))   # 서브스텝마다 sync 말고 N개마다(물리 교란↓)
            if self._vc % _every == 0:
                self._draw_viz()
                self.viewer.sync()
                if not _o3.environ.get("NOSLEEP"):
                    import time as _t; _t.sleep(self.m.opt.timestep*_every)
    def _draw_viz(self):
        import numpy as _np, os as _o4
        d=self.d; m=self.m
        scn=self.viewer.user_scn; scn.ngeom=0; eye=_np.eye(3).flatten()
        def _sph(p,r,c):
            if scn.ngeom>=scn.maxgeom: return
            _mj.mjv_initGeom(scn.geoms[scn.ngeom],_mj.mjtGeom.mjGEOM_SPHERE,_np.array([r,0,0]),_np.asarray(p,float),eye,_np.asarray(c,_np.float32)); scn.ngeom+=1
        def _ln(a,b,w,c,typ=_mj.mjtGeom.mjGEOM_LINE):
            if scn.ngeom>=scn.maxgeom: return
            g=scn.geoms[scn.ngeom]; _mj.mjv_initGeom(g,typ,_np.zeros(3),_np.zeros(3),eye,_np.asarray(c,_np.float32))
            _mj.mjv_connector(g,typ,w,_np.asarray(a,float),_np.asarray(b,float)); scn.ngeom+=1
        ZC=0.025                                                  # 발 접지판별(구중심 z)
        fz=[d.geom_xpos[g][2] for g in self.foot_gids]
        # ── 타겟 footstep(swing 발만, 빨강구) ──
        for fi,g in enumerate(self.foot_gids):
            if fi<len(self.markers) and fz[fi]>ZC:
                p=self.markers[fi]; _sph([p[0],p[1],0.008],0.012,[1,0.1,0.1,0.9])
        # ── 지지다각형(접지 발 연결, 청록 지면선) — CoM투영 벗어남 확인용 ──
        _ord=[0,1,3,2]                                            # FL,FR,HR,HL 둘레순(교차X)
        sp=[d.geom_xpos[self.foot_gids[i]][:3].copy() for i in _ord if fz[i]<ZC]
        for k in range(len(sp)):
            if len(sp)<2: break
            a=sp[k].copy(); a[2]=0.003; b=sp[(k+1)%len(sp)].copy(); b[2]=0.003
            _ln(a,b,0.006,[0.1,0.9,0.9,1])
        # ── 무게중심 지면투영(노랑구+수직선) ──
        com=d.subtree_com[0].copy()
        _sph([com[0],com[1],0.004],0.020,[1,0.9,0.1,0.95]); _ln([com[0],com[1],0.0],com,0.003,[1,0.9,0.1,0.6])
        # ── 명령방향 화살표(로봇 위 노랑) ──
        cv=self.cmd_v
        if float(_np.hypot(cv[0],cv[1]))>1e-3:
            frm=d.qpos[0:3].copy()+_np.array([0,0,0.20]); to=frm+_np.array([cv[0],cv[1],0.0])*0.4
            _ln(frm,to,0.015,[1,0.85,0.1,1],_mj.mjtGeom.mjGEOM_ARROW)
        # ── base 궤적: 3D(밝은 마젠타·굵게) ──
        self.base_trail.append(d.qpos[0:3].copy()); bp=self.base_trail
        for k in range(1,len(bp)):
            if _np.linalg.norm(bp[k]-bp[k-1])<1e-4: continue
            _ln(bp[k-1],bp[k],0.010,[1,0.15,0.9,1])                       # 3D base 궤적
        # ── 마찰콘 + GRF(접촉마다): GRF가 콘 벗어나면 슬립 ──
        mu=float(_o4.environ.get("CONE_MU", str(m.geom_friction[self.foot_gids[0]][0])))
        hh=0.10; Ncn=8
        if not _o4.environ.get("NOCONE"):
            for i in range(d.ncon):
                c=d.contact[i]
                if c.geom1 not in self.foot_gids and c.geom2 not in self.foot_gids: continue
                p=c.pos.copy()
                for k in range(Ncn):                              # 마찰콘(파랑 모선+림, 반각=atan(mu))
                    a1=2*_np.pi*k/Ncn; a2=2*_np.pi*(k+1)/Ncn
                    r1=_np.array([_np.cos(a1),_np.sin(a1),0])*hh*mu+_np.array([0,0,hh])
                    r2=_np.array([_np.cos(a2),_np.sin(a2),0])*hh*mu+_np.array([0,0,hh])
                    _ln(p,p+r1,0.0015,[0.3,0.5,1,0.5]); _ln(p+r1,p+r2,0.0015,[0.3,0.5,1,0.5])
                f6=_np.zeros(6); _mj.mj_contactForce(m,d,i,f6)    # GRF 화살표(초록)
                fw=c.frame.reshape(3,3).T@f6[:3]
                if fw[2]<0: fw=-fw
                mag=_np.linalg.norm(fw)
                if mag>1.0: _ln(p,p+fw/mag*min(mag/250.0,0.15),0.008,[0.1,1,0.2,1],_mj.mjtGeom.mjGEOM_ARROW)
        # ── 발 궤적(예산 남으면, 마지막=초과시 잘림): 발별 색선 ──
        _tc=[[0.2,0.6,1,1],[0.2,1,0.4,1],[1,0.6,0.2,1],[1,0.3,0.85,1]]
        for fi,gid in enumerate(self.foot_gids):
            self.foot_trail[fi].append(d.geom_xpos[gid].copy()); pts=self.foot_trail[fi]
            for k in range(1,len(pts)):
                if _np.linalg.norm(pts[k]-pts[k-1])<1e-4: continue
                _ln(pts[k-1],pts[k],0.004,_tc[fi%4])
        # ── 텍스트 오버레이(좌상=sim time, 우상=외력, 좌하=명령) ──
        _fext=max((float(_np.linalg.norm(d.xfrc_applied[b,:3])) for b in range(1,m.nbody)),default=0.0)
        cv=self.cmd_v
        self.viewer.set_texts([
            (_mj.mjtFont.mjFONT_BIG,_mj.mjtGridPos.mjGRID_TOPLEFT,'sim time','%.2f s'%d.time),
            (_mj.mjtFont.mjFONT_BIG,_mj.mjtGridPos.mjGRID_TOPRIGHT,'ext force','%.0f N'%_fext),
            (_mj.mjtFont.mjFONT_BIG,_mj.mjtGridPos.mjGRID_BOTTOMLEFT,'cmd vx/vy/wz','%+.2f %+.2f %+.2f'%(cv[0],cv[1],cv[5]))])
    def _key(self,kc):                                           # teleop → SportClient.Move: ↑↓=vx ←→=vy ,/.=yaw X=정지
        import os as _o5
        s=self.sport; vmx=float(_o5.environ.get("VMAX_X","0.4")); vmy=float(_o5.environ.get("VMAX_Y","0.2")); wmx=float(_o5.environ.get("WMAX","0.3"))
        if   kc==265: s.Move(min( vmx,s.vx+0.05), s.vy, s.vyaw)
        elif kc==264: s.Move(max(-vmx,s.vx-0.05), s.vy, s.vyaw)
        elif kc==263: s.Move(s.vx, min( vmy,s.vy+0.05), s.vyaw)
        elif kc==262: s.Move(s.vx, max(-vmy,s.vy-0.05), s.vyaw)
        elif kc==ord(','): s.Move(s.vx, s.vy, min( wmx,s.vyaw+0.05))
        elif kc==ord('.'): s.Move(s.vx, s.vy, max(-wmx,s.vyaw-0.05))
        elif kc==ord('X') or kc==ord('x'): s.StopMove()
        else: return
        print('[SportClient.Move] vx=%+.2f vy=%+.2f vyaw=%+.2f'%(s.vx,s.vy,s.vyaw),flush=True)
    def changeCamera(self,*a,**k): pass
    def showQuadrupedFeet(self,*a,**k): pass
    def moveQuadrupedFeet(self,*a,**k): pass
from simple_mpc import (
    RobotModelHandler,
    RobotDataHandler,
    FullDynamicsOCP,
    MPC,
    Interpolator,
)
import os as _os, pinocchio as _pin
class _ERD:
    PKG=_os.path.join(_os.environ["CONDA_PREFIX"],"share")            # package:// 루트
    SHARE=_os.path.join(PKG,"example-robot-data/robots")              # robots 디렉토리
    def load(self,name):
        rw=_pin.RobotWrapper.BuildFromURDF(self.SHARE+"/go2_description/urdf/go2.urdf",self.PKG,_pin.JointModelFreeFlyer())
        _pin.loadReferenceConfigurations(rw.model,self.SHARE+"/go2_description/srdf/go2.srdf",False)  # "standing" 자세
        return rw
    def getModelPath(self,sub):
        return self.SHARE
erd=_ERD()
import time
import copy

# ####### CONFIGURATION  ############
# Load robot
URDF = "/home/jsh/문서/jsh/simulation/quad/urdf/02_Leg_UFDF_260610_9.urdf"
base_joint_name = "root_joint"
_M = _pin.buildModelFromUrdf(URDF, _pin.JointModelFreeFlyer())
_qstand = np.array([0.0,0.0,0.52, 0,0,0,1,
    0.0,0.49223,-0.76893, 0.0,0.49223,-0.76893,
    0.0,-0.4749,0.72463,0.0, 0.0,-0.4749,0.72463,0.0])   # _9 crouch(앞발목 fixed: 앞3·뒤4)
_M.referenceConfigurations["standing"] = _qstand
_sole={'FL':[0.01452,0.0,-0.07802],'FR':[0.01452,0.0,-0.07802],
       'HL':[0.02455,0.0,-0.07467],'HR':[0.02455,0.0,-0.07467]}
for _L in ['FL','FR','HL','HR']:                          # 접촉프레임 {L}_foot = contact_link + sole_off
    _fr=_M.frames[_M.getFrameId(_L+"_foot_contact_link")]
    _pl=_fr.placement*_pin.SE3(np.eye(3), np.array(_sole[_L]))
    _M.addFrame(_pin.Frame(_L+"_foot", _fr.parentJoint, _fr.parentFrame, _pl, _pin.FrameType.OP_FRAME))
# ★스탠스 낮춤(CoM↓=물리적 안정마진): CROUCHZ 지정시 그 base높이로 다리 IK 재유도(발은 같은 지면위치 유지)
if _os.environ.get("CROUCHZ"):
    _bz = float(_os.environ["CROUCHZ"])
    _dk = _M.createData(); _fid = {L: _M.getFrameId(L + "_foot") for L in ['FL','FR','HL','HR']}
    _pin.forwardKinematics(_M, _dk, _qstand); _pin.updateFramePlacements(_M, _dk)
    _tgt = {L: _dk.oMf[_fid[L]].translation.copy() for L in ['FL','FR','HL','HR']}
    _qidx = {'FL':[7,8,9],'FR':[10,11,12],'HL':[13,14,15],'HR':[17,18,19]}   # 발목(16,20) 제외=0 고정
    _q = _qstand.copy(); _q[2] = _bz
    for _it in range(300):
        _pin.forwardKinematics(_M, _dk, _q); _pin.updateFramePlacements(_M, _dk); _pin.computeJointJacobians(_M, _dk, _q)
        for _L in ['FL','FR','HL','HR']:
            _err = _tgt[_L] - _dk.oMf[_fid[_L]].translation
            _J = _pin.getFrameJacobian(_M, _dk, _fid[_L], _pin.LOCAL_WORLD_ALIGNED)[:3]
            _dq = np.linalg.lstsq(_J[:, [i-1 for i in _qidx[_L]]], _err, rcond=None)[0]
            for _k, _c in enumerate(_qidx[_L]): _q[_c] += _dq[_k]
    _qstand = _q; _M.referenceConfigurations["standing"] = _qstand
    print("[MJ] CROUCHZ=%.2f 적용, qstand 재유도" % _bz)
model_handler = RobotModelHandler(_M, "standing", base_joint_name)
model_handler.addPointFoot("FL_foot", base_joint_name)
model_handler.addPointFoot("FR_foot", base_joint_name)
model_handler.addPointFoot("HL_foot", base_joint_name)
model_handler.addPointFoot("HR_foot", base_joint_name)
data_handler = RobotDataHandler(model_handler)

# ===== FullDynamics OCP + MPC (full-body MPC, RTI, 토크 직접) =====
nq = model_handler.getModel().nq
nv = model_handler.getModel().nv
nu = nv - 6
force_size = 3
nk = model_handler.getFeetNb()
gravity = np.array([0, 0, -9.81])
u0 = np.zeros(nu)
dt_mpc = 0.01

# ★go2_fulldynamics 작동값과 정확히 일치(WBORI/WBVRT 기본 0). base 위치/자세는 penalize 안 함 → 발프레임추종(w_frame)으로 안정화
# 02_Leg는 비대칭이라 측방(y)·yaw 모드가 약함 → 소량 위치가중으로 안정마진 보강(go2는 0이어도 대칭이라 OK)
w_basepos = [0, float(_os.environ.get("WBY", "0")), 0, float(_os.environ.get("WBORI", "0")), float(_os.environ.get("WBORI", "0")), float(_os.environ.get("WBYAW", "0"))]
w_basevel = [float(_os.environ.get("WBVX", "400")), float(_os.environ.get("WBVY", "200")), 10, 10, 10, float(_os.environ.get("WBWZ", "10"))]   # vx/vy/vz/wx/wy/wz 추종가중(WBVX전진·WBVY측방·WBWZ선회)
# 뒷발목(pin idx 9=HL_foot,13=HR_foot)은 point-foot서 floppy → posture/vel 가중치 강하게(핀고정)
_ankw = float(_os.environ.get("ANKLE_W", "50")); _ankdw = float(_os.environ.get("ANKLE_DW", "5"))
_wlp = [1.0] * nu; _wlv = [0.1] * nu
for _ia in (9, 13):
    _wlp[_ia] = _ankw; _wlv[_ia] = _ankdw
w_x = np.diag(np.array(w_basepos + _wlp + w_basevel + _wlv))   # _9: nu=14 비균일
w_u = np.eye(nu) * 1e-4
w_LFRF = float(_os.environ.get("WFRAME", "1000"))
w_cent = np.diag(np.array([0.04, 0.04, 0, 0, 0, 0]))   # go2와 동일
w_forces_lin = np.array([0.0001, 0.0001, 0.0001])

problem_conf = dict(
    timestep=dt_mpc, w_x=w_x, w_u=w_u, w_cent=w_cent, gravity=gravity, force_size=3,
    w_forces=np.diag(w_forces_lin), w_frame=np.eye(3) * w_LFRF,
    umin=-model_handler.getModel().effortLimit[6:]*3.0, umax=model_handler.getModel().effortLimit[6:]*3.0,   # ★Peak=Rated×3(구조1과 동일, 포화제거)
    qmin=model_handler.getModel().lowerPositionLimit[7:], qmax=model_handler.getModel().upperPositionLimit[7:],
    Kp_correction=np.array([0, 0, 0]), Kd_correction=np.array([0, 0, 0]),
    mu=float(_os.environ.get("MU", "0.8")), Lfoot=0.01, Wfoot=0.01,
    torque_limits=True, kinematics_limits=True,
    force_cone=_os.environ.get("FCONE","1")!="0", land_cstr=_os.environ.get("LAND","1")!="0",   # 표준: 기본 ON(FCONE=0/LAND=0로 끔)
)
T = int(_os.environ.get("T","50"))
dynproblem = FullDynamicsOCP(problem_conf, model_handler)
dynproblem.createProblem(model_handler.getReferenceState(), T, force_size, gravity[2], False)

T_ds = int(_os.environ.get("TDS", "8")); T_ss = int(_os.environ.get("TSS", "20"))   # 빠른cadence 기본=0.1~0.4 전범위 94~97%+전방향
mpc_conf = dict(support_force=-model_handler.getMass() * gravity[2], TOL=1e-4, mu_init=float(_os.environ.get("MUINIT","1e-8")),
                max_iters=int(_os.environ.get("ITERS", "1")), num_threads=int(_os.environ.get("NTH", "8")),
                swing_apex=float(_os.environ.get("APEX", "0.15")),
                T_fly=T_ss, T_contact=T_ds, timestep=dt_mpc)
mpc = MPC(mpc_conf, dynproblem)

cq = {"FL_foot": True, "FR_foot": True, "HL_foot": True, "HR_foot": True}
cFL = {"FL_foot": False, "FR_foot": True, "HL_foot": True, "HR_foot": False}
cFR = {"FL_foot": True, "FR_foot": False, "HL_foot": False, "HR_foot": True}
if _os.environ.get("STAND"):
    contact_phases = [cq] * (2 * T_ds + 2 * T_ss)   # 전스탠스(보행X) — base 제어 격리용
else:
    contact_phases = [cq] * T_ds + [cFL] * T_ss + [cq] * T_ds + [cFR] * T_ss
mpc.generateCycleHorizon(contact_phases)

N_simu = 10; dt_simu = dt_mpc / N_simu
interpolator = Interpolator(model_handler.getModel())

device = MujocoRobot(model_handler.getReferenceState()[:nq], dt_simu, view=bool(int(_os.environ.get("VIEW", "0"))))
device.initializeJoints(model_handler.getReferenceState()[:nq])
q_meas, v_meas = device.measureState()
x_measured = np.concatenate([q_meas, v_meas])

_vx0 = float(_os.environ.get("VX", "0.2"))
device.sport.Move(_vx0, float(_os.environ.get("VY", "0")), float(_os.environ.get("WZ", "0")))   # 초기 고수준 속도명령
v = device.sport.velocity_base()
mpc.velocity_base = v
_SLIP=bool(_os.environ.get("SLIP")); _slipacc=[0.0]*4; _netx=[0.0]*4; _prevf=[None]*4   # 발 슬립진단(접촉중 수평이동)
_itms=[]   # mpc.iterate 시간(ms) — 실시간성 측정
_lc = LowCmd(nu); _KP = np.full(nu, float(_os.environ.get("KP","0"))); _KD = np.full(nu, float(_os.environ.get("KD","0")))  # 저수준 LowCmd(기본 kp=kd=0=순수토크)
# ★비동기 모사: MPC를 K 제어주기(10ms)마다 풀고, 그 사이 plan을 advance하며 재사용 (K=4→25Hz, K=2→50Hz). 1=동기(매주기)
_DECIM = int(_os.environ.get("MPC_DECIM","1")); _pk = 0
print("[MJ] MPC_DECIM=%d → 재계획 %.0fHz (제어 %.0fHz)" % (_DECIM, 100.0/_DECIM, 100.0*N_simu), flush=True)
import numpy as _npd
# 뷰어=무한루프+키보드 teleop / 헤드리스=STEPS 유한
_INF = bool(device.viewer); _MAXSTEP = int(_os.environ.get("STEPS", "300"))
print("[MJ] FullDynamics 02_Leg _9 — %s, 초기 vx=%.2f" % ("무한(키보드 teleop)" if _INF else "STEPS=%d"%_MAXSTEP, _vx0), flush=True)
if _INF: print("[teleop] ↑↓=전진vx  ←→=측방vy  ,/.=선회yaw  X=정지  (뷰어 닫으면 종료)", flush=True)

# 공유메모리 레이아웃(비동기): [pver, stop, ms, xver] + x(NX) + vcmd(6) + plan(NP*PSTEP)
_NX = nq + nv; _NDX = 2 * nv; _NP = int(_os.environ.get("ASYNC_NP", "16")); _PSTEP = _NX + nu + nu * _NDX
_HDR = 4; _SHN = _HDR + _NX + 6 + _NP * _PSTEP; _XO = _HDR; _VO = _HDR + _NX; _PO = _HDR + _NX + 6

# ════ MPC 워커 프로세스(WORKER env): 독립 프로세스가 자기 MPC로 iterate만 (spawn식 → fork·GIL 무관) ════
if _os.environ.get("WORKER"):
    import sys as _sys2, time as _tm2
    from multiprocessing import shared_memory as _shmmod
    _shm = _shmmod.SharedMemory(name=_os.environ["WORKER"]); _b = np.ndarray(_SHN, dtype=np.float64, buffer=_shm.buf)
    print("[WORKER] MPC 워커 시작 — 자기 프로세스서 mpc.iterate 연속", flush=True)
    while _b[1] == 0.0:                                   # stop 플래그
        while True:                                       # x torn-read 방지(버전 재확인)
            _v1 = _b[3]; _x = np.array(_b[_XO:_XO+_NX]); _vc = np.array(_b[_VO:_VO+6])
            if _b[3] == _v1: break
        mpc.velocity_base = _vc
        try: _t0 = _tm2.perf_counter(); mpc.iterate(_x); _ms = (_tm2.perf_counter()-_t0)*1000.0
        except Exception: continue
        _o = _PO                                          # plan 쓰기
        for i in range(_NP):
            _b[_o:_o+_NX] = np.asarray(mpc.xs[i], float).ravel(); _o += _NX
            _b[_o:_o+nu] = np.asarray(mpc.us[i], float).ravel(); _o += nu
            _b[_o:_o+nu*_NDX] = np.asarray(mpc.Ks[i], float).ravel(); _o += nu*_NDX
        _b[2] = _ms; _b[0] += 1                           # ms, pver++ (마지막=발행)
    _shm.close(); _sys2.exit(0)

# ════ 비동기 제어 프로세스(ASYNC=1): 워커 subprocess launch + 1kHz 제어 (GIL·fork 무관 진짜 동시) ════
if _os.environ.get("ASYNC"):
    import sys as _sys, time as _tm, subprocess as _sp
    from multiprocessing import shared_memory as _shmmod
    if device.viewer: print("[ASYNC] VIEW=0 에서만"); _sys.exit(1)
    _shm = _shmmod.SharedMemory(create=True, size=_SHN*8); _b = np.ndarray(_SHN, dtype=np.float64, buffer=_shm.buf); _b[:] = 0.0
    _b[_XO:_XO+_NX] = x_measured; _b[_VO:_VO+6] = device.sport.velocity_base(); _b[3] = 1
    _env = dict(_os.environ); _env["WORKER"] = _shm.name
    _wk = _sp.Popen([_sys.executable, _os.path.abspath(__file__)], env=_env)
    print("[ASYNC-MP] MPC 워커 subprocess launch(name=%s) — 독립 프로세스" % _shm.name, flush=True)
    while _b[0] == 0:                                     # 첫 plan 대기
        _tm.sleep(0.01)
        if _wk.poll() is not None: print("[ASYNC] 워커 종료됨(빌드실패?)"); _shm.close(); _shm.unlink(); _sys.exit(1)
    def _unpack():
        while True:
            _v1 = _b[0]; _a = np.array(_b[_PO:_PO+_NP*_PSTEP]); _ms = _b[2]
            if _b[0] == _v1: break
        _xs=[]; _us=[]; _Ks=[]; _off=0
        for i in range(_NP):
            _xs.append(_a[_off:_off+_NX]); _off+=_NX
            _us.append(_a[_off:_off+nu]); _off+=nu
            _Ks.append(_a[_off:_off+nu*_NDX].reshape(nu,_NDX)); _off+=nu*_NDX
        return _xs,_us,_Ks,_v1,_ms
    _xs,_us,_Ks,_myver,_ms = _unpack(); _age=0; _cs=0; _NCTRL=_MAXSTEP*N_simu; _tnext=_tm.perf_counter()
    while _cs < _NCTRL:
        q_meas, v_meas = device.measureState(); x_measured = np.concatenate([q_meas, v_meas])
        _b[_XO:_XO+_NX] = x_measured; _b[_VO:_VO+6] = device.sport.velocity_base(); _b[3] += 1   # 상태 발행
        if _b[0] > _myver: _xs,_us,_Ks,_myver,_ms = _unpack(); _age=0
        _kp = min(_age // N_simu, _NP-2); _delay = (_age % N_simu) * dt_simu
        _xi = interpolator.interpolateState(_delay, dt_mpc, [_xs[_kp], _xs[_kp+1]])
        _ui = interpolator.interpolateLinear(_delay, dt_mpc, [_us[_kp], _us[_kp+1]])
        _tau = _ui - _Ks[_kp] @ model_handler.difference(x_measured, _xi)
        _lc.q=_xi[7:7+nu]; _lc.dq=_xi[nq+6:nq+6+nu]; _lc.kp=_KP; _lc.kd=_KD; _lc.tau=_tau
        device.write_low_cmd(_lc)
        _age+=1; _cs+=1
        _tnext+=dt_simu; _slp=_tnext-_tm.perf_counter()                # 1kHz 실시간 페이싱
        if _slp>0: _tm.sleep(_slp)
        if _cs % 300 == 0:
            _z=device.d.qpos[2]
            print("[ASYNC-MP] t=%.1fs base_z=%.3f MPC=%.1fms(~%.0fHz) plan_age=%d틱" % (_cs*dt_simu,_z,_ms,1000.0/max(_ms,1e-3),_age), flush=True)
            if _z<0.15: print("[ASYNC-MP] 전복 @%.1fs"%(_cs*dt_simu)); break
    _b[1]=1; _tm.sleep(0.1); _wk.terminate()
    try: _shm.close(); _shm.unlink()
    except Exception: pass
    print('[ASYNC-MP] 종료(전복없이 완주)' if _cs>=_NCTRL else '[ASYNC-MP] 종료', flush=True); _sys.exit(0)

step = 0
while True:
    if _INF:
        if not device.viewer.is_running(): break
    elif step >= _MAXSTEP: break
    v = device.sport.velocity_base()        # 고수준 SportClient → cmd_vel
    mpc.velocity_base = v
    if step % 30 == 0:
        _z = device.d.qpos[2]; _x = device.d.qpos[0]; _y = device.d.qpos[1]
        _t = _npd.degrees(_npd.arccos(_npd.clip(1 - 2 * (device.d.qpos[4]**2 + device.d.qpos[5]**2), -1, 1)))
        _qw,_qx,_qy,_qz = device.d.qpos[3:7]
        _yaw = _npd.degrees(_npd.arctan2(2*(_qw*_qz+_qx*_qy), 1-2*(_qy*_qy+_qz*_qz)))
        print("[MJ] step=%3d t=%.2f base_z=%.3f x=%+.3f y=%+.3f yaw=%+.1f tilt=%.1f" % (step, step * dt_mpc, _z, _x, _y, _yaw, _t), flush=True)
        if _SLIP and step > 0:
            print("[SLIP] 누적|이동| FL=%.3f FR=%.3f HL=%.3f HR=%.3f | 순dx(앞+/뒤-) FL=%+.3f FR=%+.3f HL=%+.3f HR=%+.3f" % (tuple(_slipacc)+tuple(_netx)), flush=True)
        if _os.environ.get("TIMING") and step>0 and _itms:
            import numpy as _n2; _a=_n2.array(_itms[-100:]); print("[TIMING] mpc.iterate 평균=%.2fms 최대=%.2fms (%.0fHz 가능)"%(_a.mean(),_a.max(),1000.0/_a.mean()), flush=True)
        if _z < 0.15:
            print("[MJ] FullDynamics 전복 @%.2fs" % (step * dt_mpc)); break
    if step % _DECIM == 0:                               # 비동기: K주기마다만 재계획
        _ti0=time.perf_counter(); mpc.iterate(x_measured); _itms.append((time.perf_counter()-_ti0)*1000.0); _pk = 0
        if device.viewer:                               # 타겟 footstep = 호라이즌 끝 발 레퍼런스(착지예측)
            device.cmd_v = v.copy()
            try: device.markers = [mpc.getReferencePose(T - 1, _fn).translation for _fn in ["FL_foot","FR_foot","HL_foot","HR_foot"]]
            except Exception: device.markers = []
    _pkc = min(_pk, T - 2)                               # stale plan을 advance하며 재사용
    xss = [mpc.xs[_pkc], mpc.xs[_pkc + 1]]; uss = [mpc.us[_pkc], mpc.us[_pkc + 1]]; _Ksk = mpc.Ks[_pkc]
    for j in range(N_simu):
        delay = j / float(N_simu) * dt_mpc
        x_interp = interpolator.interpolateState(delay, dt_mpc, xss)
        u_interp = interpolator.interpolateLinear(delay, dt_mpc, uss)
        q_meas, v_meas = device.measureState()
        x_measured = np.concatenate([q_meas, v_meas])
        mpc.getDataHandler().updateInternalData(x_measured, True)
        current_torque = u_interp - 1.0 * _Ksk @ model_handler.difference(x_measured, x_interp)
        _lc.q = x_interp[7:7+nu]; _lc.dq = x_interp[nq+6:nq+6+nu]   # plan 목표 관절 q/dq
        _lc.kp = _KP; _lc.kd = _KD; _lc.tau = current_torque        # 기본 kp=kd=0=순수토크(Riccati는 tau에 포함)
        device.write_low_cmd(_lc)                                   # Unitree 저수준 인터페이스로 적용
    if _SLIP:                                            # 접촉중(z<0.025) 수평이동: 누적절대 + 순방향dx
        for _fi, _gid in enumerate(device.foot_gids):
            _p = device.d.geom_xpos[_gid][:2].copy()
            if _prevf[_fi] is not None and device.d.geom_xpos[_gid][2] < 0.025:
                _slipacc[_fi] += float(np.linalg.norm(_p - _prevf[_fi]))
                _netx[_fi] += float(_p[0] - _prevf[_fi][0])     # 부호있는 전후 변위(앞+/뒤-)
            _prevf[_fi] = _p
    _pk += 1; step += 1
