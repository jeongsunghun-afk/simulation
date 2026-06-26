import numpy as np
import mujoco as _mj
import os as _os0
# ★점접촉 MJCF(OCP 점접촉 가정과 일치) — 메시발은 OCP와 불일치해 발산. fulldynamics와 동일.
_GO2_MJCF=_os0.environ.get("MJCF","/home/jsh/문서/jsh/simulation/quad/quad_real_pt.mjcf")
_PIN2MJ=[8,9,10,11,12,13,0,1,2,3,4,5,6,7]  # pin(FL,FR,HL,HR)→mjcf(HL,HR,FL,FR)
class MujocoRobot:
    """simple-mpc device(BulletRobot) 인터페이스를 MuJoCo로 구현. 토크는 KinodynamicsID(TSID) 출력.
       pin↔mujoco: go2 관절 순서 동일(재정렬 X), 베이스 quat [w,x,y,z]↔[x,y,z,w], lin world→local(R^T)."""
    def __init__(self, q0, dt_simu, view=False):
        self.m=_mj.MjModel.from_xml_path(_GO2_MJCF); self.m.opt.timestep=dt_simu
        import os as _o2                                   # 접촉모델 매칭(컨트롤러 강체가정 ↔ MuJoCo soft)
        _lms=float(_o2.environ.get('LEG_MASS_SCALE','1.0'))   # ★다리무게 가설: 물리(MuJoCo) 다리링크 질량/관성 스케일
        if _lms!=1.0:
            for _b in range(self.m.nbody):
                _bn=_mj.mj_id2name(self.m,_mj.mjtObj.mjOBJ_BODY,_b) or ''
                if any(_s in _bn for _s in ('hip','thigh','calf','foot')):
                    self.m.body_mass[_b]*=_lms; self.m.body_inertia[_b]*=_lms
            _mj.mj_setConst(self.m,_mj.MjData(self.m))
            print('[LEG_MASS-MJ] 다리링크 ×%.2f → 총질량 %.1fkg'%(_lms,self.m.body_mass.sum()),flush=True)
        _bad=float(_o2.environ.get('BODY_ADD','0'))   # ★바디무게 추가(다리비율↓, centroidal 검증)
        if _bad!=0.0:
            _bb=_mj.mj_name2id(self.m,_mj.mjtObj.mjOBJ_BODY,'base'); _m0=self.m.body_mass[_bb]; _mn=_m0+_bad
            self.m.body_inertia[_bb]*=(_mn/_m0); self.m.body_mass[_bb]=_mn; _mj.mj_setConst(self.m,_mj.MjData(self.m))
            print('[BODY_ADD-MJ] base %.2f→%.2fkg 총%.1fkg 다리비율%.0f%%'%(_m0,_mn,self.m.body_mass.sum(),100*(1-self.m.body_mass[_bb]/self.m.body_mass.sum())),flush=True)
        if _o2.environ.get("CONE"): self.m.opt.cone=int(_o2.environ["CONE"])
        if _o2.environ.get("STIFF"): self.m.geom_solref[:,0]=float(_o2.environ["STIFF"]); self.m.geom_solref[:,1]=1.0
        _rl=float(_o2.environ.get("REAR_LOCK","0"))   # ★뒷발목 물리잠금(4-DOF→3-DOF 대칭화, 강성)
        if _rl>0:
            for _jn in ("HL_foot_joint","HR_foot_joint"):
                _jid=_mj.mj_name2id(self.m,_mj.mjtObj.mjOBJ_JOINT,_jn)
                if _jid>=0:
                    self.m.jnt_stiffness[_jid]=_rl; self.m.dof_damping[self.m.jnt_dofadr[_jid]]=_rl*0.2
            print("[REAR_LOCK] 뒷발목 stiffness=%.0f (대칭3-DOF화)"%_rl,flush=True)
        self.d=_mj.MjData(self.m); self.nu=self.m.nu
        self._set(q0); self.viewer=None
        if view:
            import mujoco.viewer as _v; self.viewer=_v.launch_passive(self.m,self.d)
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
    def execute(self,tau):
        import numpy as _np
        _um=_np.zeros(self.nu); _um[_PIN2MJ]=_np.asarray(tau).ravel()[:self.nu]; self.d.ctrl[:]=_um
        _mj.mj_step(self.m,self.d)
        if self.viewer:
            self.viewer.sync()
            import time as _t; _t.sleep(self.m.opt.timestep)   # 실시간 페이싱
    def changeCamera(self,*a,**k): pass
    def showQuadrupedFeet(self,*a,**k): pass
    def moveQuadrupedFeet(self,*a,**k): pass
from simple_mpc import (
    RobotModelHandler,
    RobotDataHandler,
    KinodynamicsOCP,
    MPC,
    Interpolator,
    KinodynamicsID,
    KinodynamicsIDSettings,
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
_lms_pin = float(_os.environ.get('LEG_MASS_SCALE','1.0'))   # ★다리무게 가설: 모델(OCP+TSID) 다리링크 관성 스케일
if _lms_pin != 1.0:                                          # joint 0=universe,1=root_joint(base) 제외, 2..=다리링크
    for _ji in range(2, _M.njoints):
        _I = _M.inertias[_ji]
        _M.inertias[_ji] = _pin.Inertia(_I.mass*_lms_pin, _I.lever, _I.inertia*_lms_pin)
    print('[LEG_MASS-PIN] 다리링크 ×%.2f → pin 총질량 %.1fkg'
          % (_lms_pin, sum(_M.inertias[_j].mass for _j in range(1,_M.njoints))), flush=True)
_bad_pin = float(_os.environ.get('BODY_ADD','0'))   # ★바디무게 추가(OCP+TSID 모델, pinocchio base=joint1)
if _bad_pin != 0.0:
    _Ib = _M.inertias[1]; _mnb = _Ib.mass + _bad_pin
    _M.inertias[1] = _pin.Inertia(_mnb, _Ib.lever, _Ib.inertia * (_mnb/_Ib.mass))
    print('[BODY_ADD-PIN] base +%.1fkg → %.2fkg'%(_bad_pin, _mnb), flush=True)
model_handler = RobotModelHandler(_M, "standing", base_joint_name)
model_handler.addPointFoot("FL_foot", base_joint_name)
model_handler.addPointFoot("FR_foot", base_joint_name)
model_handler.addPointFoot("HL_foot", base_joint_name)
model_handler.addPointFoot("HR_foot", base_joint_name)
data_handler = RobotDataHandler(model_handler)

nq = model_handler.getModel().nq
nv = model_handler.getModel().nv
nu = nv - 6
nf = 12
force_size = 3
nk = model_handler.getFeetNb()
gravity = np.array([0, 0, -9.81])
fref = np.zeros(force_size)
fref[2] = -model_handler.getMass() / nk * gravity[2]
u0 = np.concatenate((fref, fref, fref, fref, np.zeros(model_handler.getModel().nv - 6)))
dt_mpc = 0.01

_wbp = float(_os.environ.get("WBPOS", "0"))   # ★base x,y 위치 가중(기본0=자유=보행용). STAND/드리프트엔 >0로 앵커
_wbz = float(_os.environ.get("WBZ", "100"))   # base z 가중(FullDynamics=0=발프레임에 위임)
w_basepos = [_wbp, _wbp, _wbz, float(_os.environ.get("WBORI","200")), float(_os.environ.get("WBORI","200")), 0]
w_legpos = [1, 1, 1, 1]

w_basevel = [float(_os.environ.get("WBVX","60")), 10, 10, 10, 10, 10]
w_legvel = [0.1, 0.1, 0.1, 0.1]
# ★FullDynamics 참조: 뒷발목(pin idx 9=HL_foot,13=HR_foot)은 point-foot서 floppy → posture/vel 강하게 핀고정
_ankw = float(_os.environ.get("ANKLE_W", "50")); _ankdw = float(_os.environ.get("ANKLE_DW", "5"))
_wlp = [1.0]*nu; _wlv = [0.1]*nu
for _ia in (9, 13):
    _wlp[_ia] = _ankw; _wlv[_ia] = _ankdw
w_x = np.array(w_basepos + _wlp + w_basevel + _wlv)   # _9: nu=14 비균일
w_x = np.diag(w_x)
w_linforce = np.array([0.01, 0.01, 0.01])
w_u = np.concatenate(
    (
        w_linforce,
        w_linforce,
        w_linforce,
        w_linforce,
        np.ones(model_handler.getModel().nv - 6) * 1e-5,
    )
)
w_u = np.diag(w_u)
w_LFRF = 2000
_wcap = float(_os.environ.get("WCENT_ANG_P","0.1"))   # ★pitch/roll 각운동량 가중(02_Leg 다리79%=pitch각모멘텀 폭증, 0.1로 못잡음)
_wcdp = float(_os.environ.get("WCENTDER_ANG_P","0.1"))
w_cent_lin = np.array([0.0, 0.0, 1])
w_cent_ang = np.array([_wcap, _wcap, 10])
w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))
w_centder_lin = np.ones(3) * 0.0
w_centder_ang = np.array([_wcdp, _wcdp, 0.1])
w_centder = np.diag(np.concatenate((w_centder_lin, w_centder_ang)))

problem_conf = dict(
    timestep=dt_mpc,
    w_x=w_x,
    w_u=w_u,
    w_cent=w_cent,
    w_centder=w_centder,
    gravity=gravity,
    force_size=3,
    w_frame=np.eye(3) * w_LFRF,
    qmin=model_handler.getModel().lowerPositionLimit[7:],
    qmax=model_handler.getModel().upperPositionLimit[7:],
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    kinematics_limits=True,
    force_cone=_os.environ.get("FCONE","0")!="0",    # FullDynamics는 ON
    land_cstr=_os.environ.get("LAND","0")!="0",       # FullDynamics는 ON
)
T = 50

dynproblem = KinodynamicsOCP(problem_conf, model_handler)
dynproblem.createProblem(
    model_handler.getReferenceState(), T, force_size, gravity[2], False
)

T_ds = 10
T_ss = 30

mpc_conf = dict(
    support_force=-model_handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=float(_os.environ.get("MU_INIT", "1e-8")),     # 정규화(↑=안정·보수). 02_Leg 발산 완화
    max_iters=int(_os.environ.get("MAXITER", "1")),        # RTI 반복(↑=수렴↑·느림)
    num_threads=8,
    swing_apex=0.15,
    T_fly=T_ss,
    T_contact=T_ds,
    timestep=dt_mpc,
    capture_gain=float(_os.environ.get("KCAP","0")), alip_gain=float(_os.environ.get("ALIP","0")),  # ★반응형 발배치
    predict_foot=float(_os.environ.get("PREDFOOT","0")),   # ★OCP 예측 발배치
)

mpc = MPC(mpc_conf, dynproblem)

""" Define contact sequence throughout horizon"""
contact_phase_quadru = {
    "FL_foot": True,
    "FR_foot": True,
    "HL_foot": True,
    "HR_foot": True,
}
contact_phase_lift_FL = {
    "FL_foot": False,
    "FR_foot": True,
    "HL_foot": True,
    "HR_foot": False,
}
contact_phase_lift_FR = {
    "FL_foot": True,
    "FR_foot": False,
    "HL_foot": False,
    "HR_foot": True,
}
contact_phase_lift = {
    "FL_foot": False,
    "FR_foot": False,
    "HL_foot": False,
    "HR_foot": False,
}
if _os.environ.get("STAND"):                              # ★서있기: 전 스탠스(stepping 없음) — 지지 격리·튜닝용
    contact_phases = [contact_phase_quadru] * (2 * (T_ds + T_ss))
else:
    contact_phases = [contact_phase_quadru] * T_ds
    contact_phases += [contact_phase_lift_FL] * T_ss
    contact_phases += [contact_phase_quadru] * T_ds
    contact_phases += [contact_phase_lift_FR] * T_ss
mpc.generateCycleHorizon(contact_phases)

""" Interpolation """
N_simu = 10  # Number of substep the simulation does between two MPC computation
dt_simu = dt_mpc / N_simu
interpolator = Interpolator(model_handler.getModel())

""" Inverse Dynamics """
kino_ID_settings = KinodynamicsIDSettings()
kino_ID_settings.kp_base = float(_os.environ.get("KP_BASE","7.0"))
kino_ID_settings.kp_posture = float(_os.environ.get("KP_POSTURE","10.0"))
kino_ID_settings.kp_contact = float(_os.environ.get("KP_CONTACT","10.0"))
kino_ID_settings.w_base = float(_os.environ.get("W_BASE","100.0"))
kino_ID_settings.w_posture = float(_os.environ.get("W_POSTURE","1.0"))
kino_ID_settings.w_contact_force = float(_os.environ.get("W_CFORCE","1.0"))
kino_ID_settings.w_contact_motion = float(_os.environ.get("W_CMOTION","1.0"))   # ★발 고정(미끄럼방지). ↑=firm
# ★TSID 제약 완화 실험: 마찰콘·발고정등식을 풀어 실현 자유도↑ (사용자 "제약 프리하게")
kino_ID_settings.friction_coefficient = float(_os.environ.get("FRICOEF","0.8"))   # ↑=마찰콘 넓힘(전단력 자유)
if _os.environ.get("CME") is not None:
    kino_ID_settings.contact_motion_equality = _os.environ.get("CME") != "0"       # 0=발고정 부등식/soft(slip 허용)

kino_ID = KinodynamicsID(model_handler, dt_simu, kino_ID_settings)


""" Initialize simulation"""
device = MujocoRobot(
    model_handler.getReferenceState()[: model_handler.getModel().nq],
    dt_simu,
    view=bool(int(_os.environ.get("VIEW","0"))),
)

device.initializeJoints(
    model_handler.getReferenceState()[: model_handler.getModel().nq]
)
device.changeCamera(1.0, 60, -15, [0.6, -0.2, 0.5])

q_meas, v_meas = device.measureState()
x_measured = np.concatenate([q_meas, v_meas])

device.showQuadrupedFeet(
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("FL_foot")),
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("FR_foot")),
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("HL_foot")),
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("HR_foot")),
)

force_FL = []
force_FR = []
force_RL = []
force_RR = []
FL_measured = []
FR_measured = []
RL_measured = []
RR_measured = []
FL_references = []
FR_references = []
RL_references = []
RR_references = []
x_multibody = []
u_multibody = []
u_riccati = []
com_measured = []
solve_time = []
L_measured = []

v = np.zeros(6); v[0]=float(_os.environ.get("VX","0.0"))  # 전진속도 명령(env)
v[0] = float(_os.environ.get("VX","0.2"))
mpc.velocity_base = v
import numpy as _npd
_fell=False
print("[MJ] velocity_base 명령 =", list(v))
for step in range(int(_os.environ.get("STEPS","300"))):
    mpc.velocity_base = v
    if step % 30 == 0 or step == 299:
        _z=device.d.qpos[2]; _x=device.d.qpos[0]; _y=device.d.qpos[1]
        _t=_npd.degrees(_npd.arccos(_npd.clip(1-2*(device.d.qpos[4]**2+device.d.qpos[5]**2),-1,1)))
        print("[MJ] step=%3d t=%.2f base_z=%.3f x=%+.3f y=%+.3f tilt=%.1f"%(step,step*0.01,_z,_x,_y,_t),flush=True)
        _nq=model_handler.getModel().nq
        _ocpvx=mpc.xs[1][_nq] if len(mpc.xs)>1 else 0.0   # OCP 계획 base 전진속도(pin local)
        _measvx=v_meas[0]                                  # 측정 base 전진속도(pin local)
        print("    OCP계획vx=%.3f 측정vx=%.3f (명령 %.2f)"%(_ocpvx,_measvx,v[0]),flush=True)
        if _os.environ.get("DIAG"):
            def _rp(qw,qx,qy,qz):   # roll,pitch [deg]
                r=_npd.degrees(_npd.arctan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy)))
                p=_npd.degrees(_npd.arcsin(_npd.clip(2*(qw*qy-qz*qx),-1,1)))
                return r,p
            _mw,_mx,_my,_mz=device.d.qpos[3:7]          # 측정 quat [w,x,y,z]
            _mr,_mp=_rp(_mw,_mx,_my,_mz)
            _xs0=mpc.xs[0]; _r0,_p0=_rp(_xs0[6],_xs0[3],_xs0[4],_xs0[5])         # 초기(=측정) pin quat[x,y,z,w]
            _xsT=mpc.xs[-1]; _rT,_pT=_rp(_xsT[6],_xsT[3],_xsT[4],_xsT[5])         # ★terminal(OCP가 가려는 목표)
            _nqv=model_handler.getModel().nq
            _vx0=_xs0[_nqv]; _vxT=_xsT[_nqv]; _wy0=_xs0[_nqv+4]; _wyT=_xsT[_nqv+4]  # base vx, pitch각속도(local)
            print("    [DIAG] 초기 pitch=%+.1f roll=%+.1f vx=%+.2f | ★terminal pitch=%+.1f roll=%+.1f vx=%+.2f wy=%+.2f"
                  %(_p0,_r0,_vx0,_pT,_rT,_vxT,_wyT),flush=True)
        if _z<0.15:
            print("[MJ] ❌ 전복 @%.2fs"%(step*0.01)); _fell=True; break
    # print("Time " + str(step))
    start = time.time()
    mpc.iterate(x_measured)
    end = time.time()
    solve_time.append(end - start)

    force_FL.append(mpc.us[0][:3])
    force_FR.append(mpc.us[0][3:6])
    force_RL.append(mpc.us[0][6:9])
    force_RR.append(mpc.us[0][9:12])

    FL_measured.append(
        mpc.getDataHandler()
        .getFootPose(mpc.getModelHandler().getFootNb("FL_foot"))
        .translation
    )
    FR_measured.append(
        mpc.getDataHandler()
        .getFootPose(mpc.getModelHandler().getFootNb("FR_foot"))
        .translation
    )
    RL_measured.append(
        mpc.getDataHandler()
        .getFootPose(mpc.getModelHandler().getFootNb("HL_foot"))
        .translation
    )
    RR_measured.append(
        mpc.getDataHandler()
        .getFootPose(mpc.getModelHandler().getFootNb("HR_foot"))
        .translation
    )
    FL_references.append(mpc.getReferencePose(0, "FL_foot").translation)
    FR_references.append(mpc.getReferencePose(0, "FR_foot").translation)
    RL_references.append(mpc.getReferencePose(0, "HL_foot").translation)
    RR_references.append(mpc.getReferencePose(0, "HR_foot").translation)
    com_measured.append(mpc.getDataHandler().getData().com[0].copy())
    L_measured.append(mpc.getDataHandler().getData().hg.angular.copy())

    a0 = mpc.getStateDerivative(0)[nv:].copy()
    a1 = mpc.getStateDerivative(1)[nv:].copy()

    a0[6:] = mpc.us[0][nk * force_size :]
    a1[6:] = mpc.us[1][nk * force_size :]
    forces0 = mpc.us[0][: nk * force_size]
    forces1 = mpc.us[1][: nk * force_size]
    contact_states = mpc.ocp_handler.getContactState(0)

    forces = [forces0, forces1]
    ddqs = [a0, a1]
    xss = [mpc.xs[0], mpc.xs[1]]
    uss = [mpc.us[0], mpc.us[1]]

    device.moveQuadrupedFeet(
        mpc.getReferencePose(0, "FL_foot").translation,
        mpc.getReferencePose(0, "FR_foot").translation,
        mpc.getReferencePose(0, "HL_foot").translation,
        mpc.getReferencePose(0, "HR_foot").translation,
    )

    for sub_step in range(N_simu):
        t = step * dt_mpc + sub_step * dt_simu

        delay = sub_step / float(N_simu) * dt_mpc
        xs_interp = interpolator.interpolateState(delay, dt_mpc, xss)
        acc_interp = interpolator.interpolateLinear(delay, dt_mpc, ddqs)
        force_interp = interpolator.interpolateLinear(delay, dt_mpc, forces).reshape(
            (4, 3)
        )

        q_interp = xs_interp[: mpc.getModelHandler().getModel().nq]
        v_interp = xs_interp[mpc.getModelHandler().getModel().nq :]
        force_interp = [force_interp[i, :] for i in range(4)]

        q_meas, v_meas = device.measureState()
        x_measured = np.concatenate([q_meas, v_meas])

        kino_ID.setTarget(q_interp, v_interp, acc_interp, contact_states, force_interp)
        tau_cmd = kino_ID.solve(t, q_meas, v_meas)

        device.execute(tau_cmd)
        u_multibody.append(copy.deepcopy(tau_cmd))
        x_multibody.append(x_measured)


force_FL = np.array(force_FL)
force_FR = np.array(force_FR)
force_RL = np.array(force_RL)
force_RR = np.array(force_RR)
solve_time = np.array(solve_time)
if len(solve_time):
    print("[KINO_TIMING] mpc.iterate 평균=%.2fms 최대=%.2fms (%.0fHz 가능)"
          % (solve_time.mean()*1000, solve_time.max()*1000, 1000.0/(solve_time.mean()*1000)), flush=True)
FL_measured = np.array(FL_measured)
FR_measured = np.array(FR_measured)
RL_measured = np.array(RL_measured)
RR_measured = np.array(RR_measured)
FL_references = np.array(FL_references)
FR_references = np.array(FR_references)
RL_references = np.array(RL_references)
RR_references = np.array(RR_references)
com_measured = np.array(com_measured)
L_measured = np.array(L_measured)

""" save_trajectory(x_multibody, u_multibody, com_measured, force_FL, force_FR, force_RL, force_RR, solve_time,
                FL_measured, FR_measured, RL_measured, RR_measured,
                FL_references, FR_references, RL_references, RR_references, L_measured, "kinodynamics") """
