import numpy as np
import mujoco as _mj
_GO2_MJCF="/home/jsh/문서/jsh/simulation/mujoco_menagerie/unitree_go2/scene.xml"
class MujocoRobot:
    """simple-mpc device(BulletRobot) 인터페이스를 MuJoCo로 구현. 토크는 KinodynamicsID(TSID) 출력.
       pin↔mujoco: go2 관절 순서 동일(재정렬 X), 베이스 quat [w,x,y,z]↔[x,y,z,w], lin world→local(R^T)."""
    def __init__(self, q0, dt_simu, view=False):
        self.m=_mj.MjModel.from_xml_path(_GO2_MJCF); self.m.opt.timestep=dt_simu
        import os as _o
        if _o.environ.get("CONE"):   # 0=pyramidal,1=elliptic (마찰콘 모델)
            self.m.opt.cone=int(_o.environ["CONE"])
        if _o.environ.get("STIFF"):  # 접촉 강체화(solref timeconst↓) → pybullet rigid 근사
            import numpy as _n
            self.m.geom_solref[:,0]=float(_o.environ["STIFF"]); self.m.geom_solref[:,1]=1.0
        _lms=float(_o.environ.get("LEG_MASS_SCALE","1.0"))   # ★go2 다리질량 스케일(>1=무겁게=02_Leg화, MuJoCo)
        if _lms!=1.0:
            for _b in range(self.m.nbody):
                _bn=_mj.mj_id2name(self.m,_mj.mjtObj.mjOBJ_BODY,_b) or ''
                if any(_s in _bn for _s in ('hip','thigh','calf')):
                    self.m.body_mass[_b]*=_lms; self.m.body_inertia[_b]*=_lms
            _mj.mj_setConst(self.m,_mj.MjData(self.m))
            _bb=_mj.mj_name2id(self.m,_mj.mjtObj.mjOBJ_BODY,'base')
            print("[LEG_MASS-MJ] go2 다리×%.2f → 총%.1fkg 다리비율%.0f%%"%(_lms,self.m.body_mass.sum(),100*(1-self.m.body_mass[_bb]/self.m.body_mass.sum())),flush=True)
        self.d=_mj.MjData(self.m); self.nu=self.m.nu
        self._set(q0); self.viewer=None
        if view:
            import mujoco.viewer as _v; self.viewer=_v.launch_passive(self.m,self.d)
    def _set(self,q):
        self.d.qpos[0:3]=q[0:3]; x,y,z,w=q[3:7]; self.d.qpos[3:7]=[w,x,y,z]
        self.d.qpos[7:7+self.nu]=q[7:7+self.nu]; self.d.qvel[:]=0.0
        _mj.mj_forward(self.m,self.d)
    def initializeJoints(self,q0): self._set(q0)
    def resetState(self,q0): self._set(q0)
    def measureState(self):
        d=self.d; import numpy as _np
        qp=_np.zeros(self.m.nq); vp=_np.zeros(self.m.nv)
        qp[0:3]=d.qpos[0:3]; w,x,y,z=d.qpos[3:7]; qp[3:7]=[x,y,z,w]
        R=_np.zeros(9); _mj.mju_quat2Mat(R,d.qpos[3:7]); R=R.reshape(3,3)
        vp[0:3]=R.T@d.qvel[0:3]; vp[3:6]=d.qvel[3:6]
        qp[7:]=d.qpos[7:7+self.nu]; vp[6:]=d.qvel[6:6+self.nu]
        return qp, vp
    def execute(self,tau):
        import numpy as _np
        self.d.ctrl[:]=_np.asarray(tau).ravel()[:self.nu]
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
        _lmsp=float(_os.environ.get("LEG_MASS_SCALE","1.0"))   # ★go2 다리질량 스케일(OCP모델, 다리=joint2~)
        if _lmsp!=1.0:
            for _ji in range(2, rw.model.njoints):
                _I=rw.model.inertias[_ji]; rw.model.inertias[_ji]=_pin.Inertia(_I.mass*_lmsp,_I.lever,_I.inertia*_lmsp)
            print("[LEG_MASS-PIN] go2 다리×%.2f"%_lmsp, flush=True)
        return rw
    def getModelPath(self,sub):
        return self.SHARE
erd=_ERD()
import time
import copy

# ####### CONFIGURATION  ############
# Load robot
URDF_SUBPATH = "/go2_description/urdf/go2.urdf"
base_joint_name = "root_joint"
robot_wrapper = erd.load("go2")

# Create Model and Data handler
model_handler = RobotModelHandler(robot_wrapper.model, "standing", base_joint_name)
model_handler.addPointFoot("FL_foot", base_joint_name)
model_handler.addPointFoot("FR_foot", base_joint_name)
model_handler.addPointFoot("RL_foot", base_joint_name)
model_handler.addPointFoot("RR_foot", base_joint_name)
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

w_basepos = [0, 0, 100, float(_os.environ.get("WBORI","200")), float(_os.environ.get("WBORI","200")), 0]
w_legpos = [1, 1, 1]

w_basevel = [float(_os.environ.get("WBVX","60")), 10, 10, 10, 10, 10]
w_legvel = [0.1, 0.1, 0.1]
w_x = np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4)
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
w_cent_lin = np.array([0.0, 0.0, 1])
w_cent_ang = np.array([0.1, 0.1, 10])
w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))
w_centder_lin = np.ones(3) * 0.0
w_centder_ang = np.ones(3) * 0.1
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
    force_cone=False,
    land_cstr=False,
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
    mu_init=1e-8,
    max_iters=1,
    num_threads=8,
    swing_apex=0.15,
    T_fly=T_ss,
    T_contact=T_ds,
    timestep=dt_mpc,
)

mpc = MPC(mpc_conf, dynproblem)

""" Define contact sequence throughout horizon"""
contact_phase_quadru = {
    "FL_foot": True,
    "FR_foot": True,
    "RL_foot": True,
    "RR_foot": True,
}
contact_phase_lift_FL = {
    "FL_foot": False,
    "FR_foot": True,
    "RL_foot": True,
    "RR_foot": False,
}
contact_phase_lift_FR = {
    "FL_foot": True,
    "FR_foot": False,
    "RL_foot": False,
    "RR_foot": True,
}
contact_phase_lift = {
    "FL_foot": False,
    "FR_foot": False,
    "RL_foot": False,
    "RR_foot": False,
}
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
kino_ID_settings.w_posture = 1.0
kino_ID_settings.w_contact_force = 1.0
kino_ID_settings.w_contact_motion = 1.0

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
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("RL_foot")),
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("RR_foot")),
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
        if _z<0.15:
            print("[MJ] ❌ 전복 @%.2fs"%(step*0.01)); _fell=True; break
    # print("Time " + str(step))
    land_LF = mpc.getFootLandCycle("FL_foot")
    land_RF = mpc.getFootLandCycle("RL_foot")
    takeoff_LF = mpc.getFootTakeoffCycle("FL_foot")
    takeoff_RF = mpc.getFootTakeoffCycle("RL_foot")
    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )
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
        .getFootPose(mpc.getModelHandler().getFootNb("RL_foot"))
        .translation
    )
    RR_measured.append(
        mpc.getDataHandler()
        .getFootPose(mpc.getModelHandler().getFootNb("RR_foot"))
        .translation
    )
    FL_references.append(mpc.getReferencePose(0, "FL_foot").translation)
    FR_references.append(mpc.getReferencePose(0, "FR_foot").translation)
    RL_references.append(mpc.getReferencePose(0, "RL_foot").translation)
    RR_references.append(mpc.getReferencePose(0, "RR_foot").translation)
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
        mpc.getReferencePose(0, "RL_foot").translation,
        mpc.getReferencePose(0, "RR_foot").translation,
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
