#!/usr/bin/env python3
"""점발 stand 지그 sim — 좌우·앞뒤 쓰러짐 구속, 상하(z) 자유 (2026-09-22).

유저 물리 지그(수직 레일 가이드) 재현: 로봇 base 의 x·y·roll·pitch·yaw 를 매 스텝
초기값으로 사영(=강체 가이드가 주는 반력) 하고 **z(상하)만 자유**로 둔다.
WBIC 는 6-DOF 부동베이스를 그대로 유지(코드 무수정) — 사영은 sim 밖 구속일 뿐이다.
발목 whip 을 균형과 분리해 관찰·튜닝하기 위한 도구.

  python biped_stand_guide_sim.py                 # 헤드리스, 발목 whip 통계
  VIEW=1 python biped_stand_guide_sim.py          # MuJoCo 뷰어로 눈으로 보기
  ANK_KD=30 python biped_stand_guide_sim.py       # A: 발목 posture kd 튜닝
  ANK_HARD=1 ANK_KD=25 python ...                 # B: 발목 QP 분리(위치서보)
  ANK_DAMP=8 python ...                           # C: 발목 직접 속도감쇠
  FREE_YAW=1 python ...                           # yaw 도 자유(가이드가 yaw 안 잡을 때)
  T=6 python ...                                  # 시간[s]
env 발목 노브는 biped_mpc_wbic.py 가 읽는다(ANK_KP/ANK_KD/ANK_W/ANK_HARD/ANK_DAMP).
"""
import os, numpy as np, mujoco
os.environ.setdefault('VX','0'); os.environ.setdefault('VY','0'); os.environ.setdefault('WZ','0')
os.environ.setdefault('FLAT_STATIC','1'); os.environ.setdefault('ALPHA_AXIS','0.834')
VIEW = os.environ.get('VIEW','0') != '0'
FREE_YAW = os.environ.get('FREE_YAW','0') != '0'
T = float(os.environ.get('T','5.0'))
import biped_mpc_wbic as BM

c = BM.BipedMPCWBIC(); c.set_contact_mode('1pt'); c.reset(); c.setup_mpc()
m, d = c.m, c.d; dt = m.opt.timestep
JN=['HL_hip','HL_thigh','HL_calf','HL_foot','HR_hip','HR_thigh','HR_calf','HR_foot']
def jid(n):
    for s in ('_joint',''):
        i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n+s)
        if i>=0: return i
qad={n:m.jnt_qposadr[jid(n)] for n in JN}; dad={n:m.jnt_dofadr[jid(n)] for n in JN}
x0,y0 = d.qpos[0], d.qpos[1]

def guide():
    """base 5 DOF(x·y·roll·pitch[·yaw]) 를 초기값으로 사영, z 만 자유 = 강체 가이드."""
    d.qpos[0]=x0; d.qpos[1]=y0
    if FREE_YAW:
        qb=d.qpos[3:7]; yaw=np.arctan2(2*(qb[0]*qb[3]+qb[1]*qb[2]),1-2*(qb[2]**2+qb[3]**2))
        d.qpos[3]=np.cos(yaw/2); d.qpos[4]=0; d.qpos[5]=0; d.qpos[6]=np.sin(yaw/2)
        d.qvel[3]=0; d.qvel[4]=0   # roll·pitch 속도만 0 (yaw 속도 유지)
    else:
        d.qpos[3]=1; d.qpos[4]=0; d.qpos[5]=0; d.qpos[6]=0
        d.qvel[3]=0; d.qvel[4]=0; d.qvel[5]=0
    d.qvel[0]=0; d.qvel[1]=0
    mujoco.mj_forward(m,d)

guide()
print("지그 sim — 좌우·앞뒤 구속 %s· z 자유 | mass=%.1fkg cmode=%s | ANK_KP=%.0f KD=%.0f W=%.0f HARD=%s DAMP=%.1f"%(
    ("(yaw 자유)" if FREE_YAW else ""), c.mass, c.contact_mode, BM.ANK_KP, BM.ANK_KD, BM.ANK_W, BM.ANK_HARD, BM.ANK_DAMP))

def step():
    c.control(dt); mujoco.mj_step(m,d); guide()

if VIEW:
    import mujoco.viewer, time
    with mujoco.viewer.launch_passive(m,d) as v:
        t=0.0
        while v.is_running() and t<T*20:   # 뷰어는 오래 돌린다
            step(); v.sync(); time.sleep(max(0,dt-0.0005)); t+=dt
else:
    fv=[]; fr={'HL_foot':[],'HR_foot':[]}; zz=[]
    n=int(T/dt); s0=int(min(1.0,T*0.2)/dt)
    for i in range(n):
        step()
        if i>=s0:
            for f in ('HL_foot','HR_foot'):
                fv.append(abs(d.qvel[dad[f]]*57.3)); fr[f].append(d.qpos[qad[f]]*57.3)
            zz.append(d.qpos[2])
    hr=fr['HR_foot']; hl=fr['HL_foot']
    print("=== 정상상태(%.1f~%.1fs) 발목 whip ==="%(s0*dt,T))
    print("  HR_foot 범위 %.1f°  HL_foot 범위 %.1f°  max|발목속도| %.0f dps"%(
        max(hr)-min(hr), max(hl)-min(hl), max(fv)))
    print("  base z %.3f~%.3f m (Δ%.0fmm)  ← 상하 자유"%(min(zz),max(zz),(max(zz)-min(zz))*1000))
    print("  판정: HR_foot 범위 <12° 면 실기 stand_runaway 트립 안 걸림")
