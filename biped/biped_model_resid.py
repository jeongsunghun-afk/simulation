#!/usr/bin/env python3
"""biped_model_resid.py — ExpLog CSV 로 모델오차 국소화. **매달림(자유공간) 전제**.
   실측토크 τ_real(cur_a) vs 모델 역동역학 τ_model = M(q)q̈ + C(q,q̇)q̇ + g(q) + 마찰 을
   관절별로 대조하고, **잔차의 q/q̇/q̈ 의존성**으로 어느 축·어느 항이 틀렸는지 짚는다.

   ⚠전제: 로그 구간이 **발 공중(접촉0)** 이어야 함(접지면 미지 GRF 가 잔차에 섞임).
   ⚠τ_real 은 SCALE(≈6.4) 얽힘 — 상수배 오차는 전 축 공통으로 보인다(절대검증=저울).
   ⚠고정베이스(크레인) 가정: mj_fullM 대각+커플링 · armature(ROTOR_I·N²) 주입.

   사용: python3 biped_model_resid.py exp_logs/squat_YYYYMMDD_HHMMSS.csv [SCALE]
"""
import csv, json, os, sys, numpy as np, mujoco

if len(sys.argv) < 2: print(__doc__); sys.exit(1)
CSVP = sys.argv[1]
MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'biped_flatfoot.mjcf')
NM   = ['HL_hip','HL_thigh','HL_calf','HL_foot','HR_hip','HR_thigh','HR_calf','HR_foot']; N=8
SIGN = np.array([-1,1,-1,-1,-1,-1,1,1], float)
GEAR = np.array([7,7,10.5,8.4,7,7,10.5,8.4], float); KT=0.2
JFRIC= np.array([0.827,0.604,0.871,0.639]*2); FRIC_V0=0.20
ROTOR_I = 5.29e-4
SCALE = float(sys.argv[2]) if len(sys.argv)>2 else (
        float(json.load(open('/tmp/grf_verify_base.json'))['SCALE'])
        if os.path.exists('/tmp/grf_verify_base.json') else 6.4)

rows = list(csv.DictReader(open(CSVP)))
if len(rows) < 30: print("표본 부족(%d)"%len(rows)); sys.exit(1)
def col(key): return np.array([[float(r['%s_%s'%(key,n)]) for n in NM] for r in rows], float)
t   = np.array([float(r['t']) for r in rows])
q   = np.deg2rad(col('q'))            # 측정 관절각[rad]
dq  = np.deg2rad(col('dq'))           # 측정 속도[rad/s]
cur = col('cur')                      # 실측 전류[A] (채널≈관절)
tau_real = SIGN*cur*KT*GEAR/SCALE     # 실측 관절토크[Nm]
def smooth(x,k=5): return np.convolve(x, np.ones(k)/k, mode='same') if k>1 else x
qdd = np.vstack([np.gradient(smooth(dq[:,i]), t) for i in range(N)]).T   # q̈[rad/s²]

m = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(m)
jid  = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n+"_joint") for n in NM]
qadr = [int(m.jnt_qposadr[j]) for j in jid]; dadr = [int(m.jnt_dofadr[j]) for j in jid]
for i in range(N): m.dof_armature[dadr[i]] = ROTOR_I*GEAR[i]**2
Mf = np.zeros((m.nv,m.nv))

tau_model = np.zeros((len(rows),N)); g_only = np.zeros((len(rows),N))
for k in range(len(rows)):
    for i in range(N): d.qpos[qadr[i]] = q[k,i]
    for i in range(m.nv): d.qvel[i]=0.0
    mujoco.mj_forward(m,d); g_only[k]=[d.qfrc_bias[dadr[i]] for i in range(N)]
    for i in range(N): d.qvel[dadr[i]] = dq[k,i]
    mujoco.mj_forward(m,d); bias=np.array([d.qfrc_bias[dadr[i]] for i in range(N)])  # C q̇ + g
    mujoco.mj_fullM(m, Mf, d.qM)
    qdd_full = np.zeros(m.nv)
    for i in range(N): qdd_full[dadr[i]] = qdd[k,i]
    Mqdd = np.array([ (Mf[dadr[i]] @ qdd_full) for i in range(N)])   # (M q̈)[관절] 커플링 포함
    fric = JFRIC*np.tanh(dq[k]/FRIC_V0)
    tau_model[k] = Mqdd + bias + fric

resid = tau_real - tau_model
def corr(a,b):
    if np.std(a)<1e-9 or np.std(b)<1e-9: return float('nan')
    return float(np.corrcoef(a,b)[0,1])

print("■ 모델 잔차 분석 — %s  (SCALE=%.2f · %d표본)"%(os.path.basename(CSVP),SCALE,len(rows)))
print("  자극: |q̇|max=%.2f rad/s · |q̈|max=%.1f rad/s²  (작으면 M·C 자극 부족)"%(np.abs(dq).max(),np.abs(qdd).max()))
print("%-9s %8s %8s %8s %5s | %7s %7s %7s  %s"%("관절","실측τσ","모델τσ","잔차RMS","잔차%","corr_q","corr_q̇","corr_q̈","판정"))
print("-"*90)
for i in range(N):
    rr = resid[:,i]; rms=float(np.sqrt(np.mean(rr**2)))
    sreal=float(np.std(tau_real[:,i])); rel=rms/max(sreal,1e-6)   # 잔차 상대크기
    cq=corr(rr,q[:,i]); cv=corr(rr,dq[:,i]); ca=corr(rr,qdd[:,i])
    cand=[]
    if rel>0.15:                              # ★잔차가 유의미할 때만 항 지목(수치노이즈 오탐 방지)
        if abs(cq)>0.4: cand.append("중력/질량")
        if abs(cv)>0.4: cand.append("마찰/damping")
        if abs(ca)>0.4: cand.append("관성M")
    tag = ("모델 OK" if rel<=0.15 else (" · ".join(cand) if cand else "잔차有(항 불명)"))
    print("%-9s %8.3f %8.3f %8.3f %4.0f%% | %+6.2f %+6.2f %+6.2f  %s"
          %(NM[i],sreal,float(np.std(tau_model[:,i])),rms,rel*100,cq,cv,ca,tag))
print("-"*90)
print("→ 잔차RMS 큰 축 = 모델오차 큼. corr_q↑=중력/질량, corr_q̇↑=마찰, corr_q̈↑=관성M 이 유력.")
print("  ⚠접지구간이면 GRF가 잔차에 섞임 — 매달림(발공중) 로그여야 순수 모델오차.")
print("  ⚠전 축 잔차가 같은 부호·비율이면 SCALE 오차(저울로 절대검증).")
