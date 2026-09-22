#!/usr/bin/env python3
"""biped_ref_squat_export.py — 1점 점발 위 수직 스쿼트(앉았다-일어서기) 궤적 생성.
   접촉점(발 sphere)의 x·y 를 고정하고 z 만 Δz 올려(=몸통 Δz 하강) 관절각을 IK 로 푼다.
   → **접촉점이 지면에 붙어있는** 준정적 스쿼트(비행 위상 없음 → 위치제어 재생 가능).

   walk 재생과 같은 포맷(ref_lib/*.npz, q[N,8] rad · dt · JOG_NAMES 순서)으로 저장 →
   GUI 재생 콤보(_WALK_FILES)에 '스쿼트(1점)' 로 얹으면 jog 위치제어로 재생된다.

   사용: python3 biped_ref_squat_export.py [depth_m] [cycle_s] [cycles]
         기본 depth=0.08m · 한 사이클 4s(내림2+올림2, 홈→홈 매끈해 반복 이음매 없음)."""
import sys, os, numpy as np, mujoco

MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'biped_flatfoot.mjcf')
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref_lib', 'biped_ref_squat.npz')
NM   = ['HL_hip','HL_thigh','HL_calf','HL_foot','HR_hip','HR_thigh','HR_calf','HR_foot']
# jog 안전범위(deg) — GUI JOG_LIM 과 동일(mjcf range×0.5). IK 결과를 이 안으로 클램프.
JOG_LIM = [(-17,17),(-67,32),(-27,32),(-40,20)]*2

depth  = float(sys.argv[1]) if len(sys.argv)>1 else 0.08     # [m] 최대 하강
Tcyc   = float(sys.argv[2]) if len(sys.argv)>2 else 4.0      # [s] 한 사이클(내림+올림)
cycles = int(sys.argv[3])   if len(sys.argv)>3 else 1        # 반복은 GUI 재생 loop 로도 됨
fs, dt = 50.0, 0.02

m = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(m)
jid  = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n+'_joint') for n in NM]
qadr = [int(m.jnt_qposadr[j]) for j in jid]; dadr = [int(m.jnt_dofadr[j]) for j in jid]
sph  = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ['HL_sphere','HR_sphere']]
fbody= [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ['HL_foot_contact_link','HR_foot_contact_link']]
nv = m.nv

def fwd():
    mujoco.mj_forward(m, d)

# 홈(1점 점발 = 전축 0) 에서 발 접촉구 기준 위치
for a in qadr: d.qpos[a] = 0.0
fwd()
home_foot = [d.geom_xpos[sph[k]].copy() for k in range(2)]   # 발 sphere 월드좌표(base 고정 기준)

def ik_leg(k, target):
    """다리 k(0=HL,1=HR)의 thigh/calf/foot 3관절로 발 sphere 를 target 에 맞춘다(hip=0 고정).
       damped LS Gauss-Newton. 관절한계 클램프. 현재 d.qpos 를 갱신."""
    leg = k*4
    dofs = [dadr[leg+1], dadr[leg+2], dadr[leg+3]]           # thigh,calf,foot
    for _ in range(60):
        fwd()
        err = target - d.geom_xpos[sph[k]]
        if np.linalg.norm(err) < 1e-5: break
        jacp = np.zeros((3,nv))
        mujoco.mj_jac(m, d, jacp, None, d.geom_xpos[sph[k]], fbody[k])
        J = jacp[:, dofs]                                    # 3×3
        dq = J.T @ np.linalg.solve(J@J.T + 1e-6*np.eye(3), err)   # damped LS
        for i,jj in enumerate([leg+1,leg+2,leg+3]):
            lo,hi = np.deg2rad(JOG_LIM[jj][0]), np.deg2rad(JOG_LIM[jj][1])
            d.qpos[qadr[jj]] = float(np.clip(d.qpos[qadr[jj]] + dq[i], lo, hi))
    return np.linalg.norm(target - d.geom_xpos[sph[k]])

nper = int(round(Tcyc*fs)); N = nper*cycles
Q = np.zeros((N,8)); zoff = np.zeros(N)
maxres = 0.0
for n in range(N):
    t = (n % nper)/fs
    dz = depth * 0.5*(1 - np.cos(2*np.pi*t/Tcyc))           # 홈→깊이→홈 매끈(raised cosine)
    zoff[n] = dz
    for a in qadr: d.qpos[a] = 0.0                          # hip 등 0, IK 는 leg 관절만 움직임
    for k in range(2):
        tgt = home_foot[k].copy(); tgt[2] += dz             # 발을 base 쪽으로 dz 올림 = 몸통 dz 하강
        r = ik_leg(k, tgt); maxres = max(maxres, r)
    Q[n] = [d.qpos[qadr[i]] for i in range(8)]

# 실제 실현 깊이 = 최저점 관절각으로 재확인
imin = int(np.argmax(zoff))
Qdeg = np.rad2deg(Q)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, q=Q, dt=np.float64(dt), t=np.arange(N)/fs, depth=np.float64(depth),
         zoff=zoff, note='1pt vertical squat, contact fixed (x,y), IK on thigh/calf/foot')
print('■ 저장:', OUT, '· N=%d(%.1fs) · depth요청=%.3fm · IK최대잔차=%.2fmm'%(N, N/fs, depth, maxres*1e3))
print('  관절각 범위[deg] (min/max) — jog 한계 초과 없어야:')
for i,n in enumerate(NM):
    lo,hi = Qdeg[:,i].min(), Qdeg[:,i].max()
    ok = (lo>=JOG_LIM[i][0]-0.1 and hi<=JOG_LIM[i][1]+0.1)
    print('   %-9s %+7.1f ~ %+7.1f  (한계 %+.0f~%+.0f) %s'%(n,lo,hi,JOG_LIM[i][0],JOG_LIM[i][1],'' if ok else '⚠초과'))
print('  최저점 q°:', ' '.join('%+.1f'%x for x in Qdeg[imin]))
