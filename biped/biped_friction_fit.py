#!/usr/bin/env python3
"""biped_friction_fit.py — 매달림 스쿼트/스윙 ExpLog 로 **각축 마찰 실측 추정**.
   τ_real(cur) 에서 동역학(M q̈ + C q̇ + g)을 빼면 남는 게 마찰토크:
       τ_fric = τ_real − (M q̈ + C q̇ + g)
   이걸 관절별로  τ_fric = τ_c·tanh(q̇/v0) + b·q̇  (쿨롱 τ_c + 점성 b) 로 최소제곱 적합.
   → 축별 실측 쿨롱마찰[Nm]·점성[Nm·s/rad] + R² + 모델 JFRIC 대조.

   ⚠전제: **매달림(발 공중, 접촉0)** — 접지면 GRF가 섞임.
   ⚠저속 jog면 q̇ 범위가 작아 점성(b)은 부정확, 쿨롱(τ_c)은 반전만 충분하면 OK.
   ⚠τ_real 은 SCALE(6.4) 얽힘 → 절대값은 그만큼 배율오차. **좌우/축간 상대비교는 신뢰.**
   ⚠거의 안 움직인 축(hip 등)은 추정 불가(반전·q̇ 부족 → 표시).

   사용: python3 biped_friction_fit.py exp_logs/squat_*.csv [SCALE]
"""
import csv, json, os, sys, numpy as np, mujoco
if len(sys.argv) < 2: print(__doc__); sys.exit(1)
CSVP = sys.argv[1]
MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'biped_flatfoot.mjcf')
NM   = ['HL_hip','HL_thigh','HL_calf','HL_foot','HR_hip','HR_thigh','HR_calf','HR_foot']; N=8
SIGN = np.array([-1,1,-1,-1,-1,-1,1,1], float)
GEAR = np.array([7,7,10.5,8.4,7,7,10.5,8.4], float); KT=0.2
JFRIC_MODEL = np.array([0.827,0.604,0.871,0.639]*2); V0=0.20; ROTOR_I=5.29e-4
SCALE = float(sys.argv[2]) if len(sys.argv)>2 else (
        float(json.load(open('/tmp/grf_verify_base.json'))['SCALE'])
        if os.path.exists('/tmp/grf_verify_base.json') else 6.4)

rows = list(csv.DictReader(open(CSVP)))
if len(rows) < 40: print("표본 부족(%d)"%len(rows)); sys.exit(1)
if ('cur_%s'%NM[0]) not in rows[0]:
    print("✗ 이 CSV 에 cur_(실측전류) 열이 없음 — **구버전 GUI**로 찍은 로그입니다.")
    print("  → 최신 GUI(스윙 로거 cur_a 추가분)로 **재기동** 후 calf 스윙을 다시 찍으세요.")
    print("     ./run_all.sh gui 로 재기동 → 스윙처프 HL_calf/HR_calf ▶실행 → 새 CSV")
    sys.exit(1)
def col(k): return np.array([[float(r['%s_%s'%(k,n)]) for n in NM] for r in rows], float)
t = np.array([float(r['t']) for r in rows])
q = np.deg2rad(col('q')); dq = np.deg2rad(col('dq')); cur = col('cur')
tau_real = SIGN*cur*KT*GEAR/SCALE
def smooth(x,k=5): return np.convolve(x,np.ones(k)/k,mode='same')
qdd = np.vstack([np.gradient(smooth(dq[:,i]), t) for i in range(N)]).T

m = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(m)
jid  = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n+"_joint") for n in NM]
qadr = [int(m.jnt_qposadr[j]) for j in jid]; dadr = [int(m.jnt_dofadr[j]) for j in jid]
for i in range(N): m.dof_armature[dadr[i]] = ROTOR_I*GEAR[i]**2
Mf = np.zeros((m.nv,m.nv))
# τ_dyn = M q̈ + C q̇ + g  (마찰 없이)
tau_dyn = np.zeros((len(rows),N))
for k in range(len(rows)):
    for i in range(N): d.qpos[qadr[i]] = q[k,i]
    for i in range(m.nv): d.qvel[i]=0.0
    for i in range(N): d.qvel[dadr[i]] = dq[k,i]
    mujoco.mj_forward(m,d); bias=np.array([d.qfrc_bias[dadr[i]] for i in range(N)])  # C q̇ + g
    mujoco.mj_fullM(m, Mf, d.qM); qf=np.zeros(m.nv)
    for i in range(N): qf[dadr[i]]=qdd[k,i]
    Mqdd=np.array([Mf[dadr[i]]@qf for i in range(N)])
    tau_dyn[k] = Mqdd + bias
tau_fric = tau_real - tau_dyn   # 남는 = 마찰(+측정오차)

print("■ 각축 마찰 추정 — %s  (SCALE=%.2f · %d표본)"%(os.path.basename(CSVP),SCALE,len(rows)))
print("%-9s %6s %7s %8s %7s %6s | %8s %s"%("관절","쿨롱τc","점성b","모델JFRIC","τc/모델","R²","|q̇|max","신뢰"))
print("-"*80)
for i in range(N):
    y = tau_fric[:,i]; v = dq[:,i]
    revs = int(np.sum(np.diff(np.sign(v))!=0)); vmax=float(np.abs(v).max())
    # 적합: y = tc·tanh(v/V0) + b·v
    A = np.column_stack([np.tanh(v/V0), v])
    reliable = (vmax>0.12 and revs>=4 and np.std(np.tanh(v/V0))>0.08)   # 움직인 축만(hip 등 저속 제외)
    if reliable:
        coef,_,_,_ = np.linalg.lstsq(A, y, rcond=None); tc,b = float(coef[0]), float(coef[1])
        pred = A@coef; ss=np.sum((y-y.mean())**2); r2 = 1-np.sum((y-pred)**2)/ss if ss>0 else float('nan')
        rel = tc/JFRIC_MODEL[i] if JFRIC_MODEL[i] else float('nan')
        flag = "✓" if r2>0.5 else "약(R²낮음)"
        print("%-9s %6.2f %7.3f %8.2f %7.2f %6.2f | %8.2f  %s"
              %(NM[i], tc, b, JFRIC_MODEL[i], rel, r2, vmax, flag))
    else:
        print("%-9s %6s %7s %8.2f %7s %6s | %8.2f  ⚠자극부족(반전%d)"
              %(NM[i], "--","--",JFRIC_MODEL[i],"--","--",vmax,revs))
print("-"*80)
print("→ 쿨롱τc = 실측 정마찰[Nm]. τc/모델>1 = 모델보다 뻑뻑(재텐션 과함 후보). 점성b = 속도비례분.")
print("  ⚠절대값은 SCALE(6.4) 배율오차 — **좌우(HL vs HR)·축간 상대비**로 판정. calf 좌우 비교가 핵심.")
