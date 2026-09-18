#!/usr/bin/env python3
"""biped_payload_scale.py — **known 페이로드(접촉점 추)** 로 토크스케일(SCALE) 실측 캘리브레이션.
   원리: 접촉점에 known 질량 m 을 달면 각 관절 중력토크가 **정확히 아는 값**만큼 변한다.
     Δτ_model_j = −m·g·Jz(접촉점)_j     (MJCF 정기구학 야코비안, 자세별)
     Δτ_real_j  = SIGN·Δcur·KT·GEAR / SCALE
     같아야 하므로  SCALE_j = SIGN·Δcur·KT·GEAR / Δτ_model_j     (축·자세 무관하게 일치해야 검증)
   → cur_a 를 **진짜 Nm 로 고정**(SCALE 불확실 해소). 마찰·중력 절대값이 그때부터 신뢰.

   전제: 추 전/후가 한 로그에 있고(나중에 달았음), **같은 자세**에서 차분(자세로 bin → sag 상쇄).
   ⚠매달림(발 공중) 가정·베이스 upright(크레인). 왼다리 영점 ~1.5° 틀어짐은 차분서 상쇄(모멘트암엔 미소영향).

   사용: python3 biped_payload_scale.py exp_logs/squat_*.csv [질량kg=3.0] [접촉바디=HL_foot_contact_link]
"""
import csv, os, sys, numpy as np, mujoco
if len(sys.argv) < 2: print(__doc__); sys.exit(1)
CSVP = sys.argv[1]
MASS = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
CBODY = sys.argv[3] if len(sys.argv) > 3 else 'HL_foot_contact_link'
SIDE = CBODY[:2]                                   # HL or HR
MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'biped_flatfoot.mjcf')
G = 9.81; KT = 0.2
NM = ['HL_hip','HL_thigh','HL_calf','HL_foot','HR_hip','HR_thigh','HR_calf','HR_foot']
SIGN = dict(zip(NM, [-1,1,-1,-1,-1,-1,1,1]))
GEAR = dict(zip(NM, [7,7,10.5,8.4,7,7,10.5,8.4]))
LEG = [n for n in NM if n.startswith(SIDE)]        # 부하 실린 다리 4축

rows = list(csv.DictReader(open(CSVP)))
if ('cur_%s'%NM[0]) not in rows[0]: sys.exit("✗ cur_ 열 없음 — 최신 GUI 로그 필요")
t   = np.array([float(r['t']) for r in rows])
cur = {n: np.array([float(r['cur_%s'%n]) for r in rows]) for n in NM}
q   = {n: np.array([float(r['q_%s'%n])   for r in rows]) for n in NM}
def sm(x,k=25): return np.convolve(x,np.ones(k)/k,mode='same')

# ── 추 단 시점 자동탐지: 부하 최대축(|cur| 상승 최대) ──
key = SIDE+'_calf'
ac = sm(np.abs(cur[key])); i_ev = int(np.argmax(sm(np.gradient(ac)))); t_ev = t[i_ev]
m_bef = t < t_ev-3; m_aft = t > t_ev+3
print("■ 페이로드 SCALE 캘리브 — %s · %s 에 %.2fkg" % (os.path.basename(CSVP), CBODY, MASS))
print("  추 단 시점 t=%.1fs · before %d행 · after %d행" % (t_ev, m_bef.sum(), m_aft.sum()))
if m_aft.sum() < 100: sys.exit("✗ after 표본 부족 — 추 단 뒤 구간이 너무 짧다")

# ── MuJoCo: fixed-base, 접촉점 야코비안 ──
mo = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(mo)
cbid = mujoco.mj_name2id(mo, mujoco.mjtObj.mjOBJ_BODY, CBODY)
if cbid < 0: sys.exit("✗ 바디 '%s' 없음"%CBODY)
jqadr = {n: int(mo.jnt_qposadr[mujoco.mj_name2id(mo,mujoco.mjtObj.mjOBJ_JOINT,n+'_joint')]) for n in NM}
jdadr = {n: int(mo.jnt_dofadr[mujoco.mj_name2id(mo,mujoco.mjtObj.mjOBJ_JOINT,n+'_joint')]) for n in NM}
# 베이스 free joint 있으면 upright 고정
free = [j for j in range(mo.njnt) if mo.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]

def dtau_model(qpose):
    for n in NM: d.qpos[jqadr[n]] = np.deg2rad(qpose[n])
    for fj in free:
        a=int(mo.jnt_qposadr[fj]); d.qpos[a:a+7]=[0,0,0,1,0,0,0]
    mujoco.mj_forward(mo,d)
    jacp = np.zeros((3,mo.nv)); mujoco.mj_jac(mo,d,jacp,None,d.xpos[cbid],cbid)
    return {n: -MASS*G*jacp[2, jdadr[n]] for n in NM}   # Δτ = −m g Jz

# ── 자세 bin (부하축 q 로) ──
FIX = float(os.environ.get('FIX_SCALE', '7.16'))          # grf SCALE 고정값(기본 7.16)
qc = q[key]; lo,hi = np.percentile(qc[m_aft],[5,95]); edges = np.linspace(lo,hi,7)
print("\n  [A] SCALE 실측 — 자세정합 bin (%s q %.0f~%.0f°) · SCALE=SIGN·Δcur·KT·GEAR/Δτ_model"%(key,lo,hi))
print("  %-9s %7s %8s %8s   자세별 SCALE→"%("관절","Δcur","Δτmodel","SCALE"))
perj={}; allx=[]; ally=[]
for n in LEG:
    scs=[]; dcs=[]; dtm=[]
    for i in range(len(edges)-1):
        a,b=edges[i],edges[i+1]
        mb=m_bef&(qc>=a)&(qc<b); ma=m_aft&(qc>=a)&(qc<b)
        if mb.sum()<5 or ma.sum()<5: continue
        dcur = np.median(cur[n][ma])-np.median(cur[n][mb])
        qpose = {k2: (np.median(q[k2][ma])+np.median(q[k2][mb]))/2 for k2 in NM}
        dtm_j = dtau_model(qpose)[n]
        if abs(dtm_j) < 1e-4: continue
        y = SIGN[n]*dcur*KT*GEAR[n]                    # 실측 Δτ·SCALE
        scs.append(y/dtm_j); dcs.append(dcur); dtm.append(dtm_j)
        allx.append(dtm_j); ally.append(y)
    if scs:
        perj[n]=dict(scale=float(np.median(scs)), dcur=float(np.mean(dcs)), dtm=float(np.mean(dtm)))
        print("  %-9s %7.1f %8.3f %8.2f   [%s]"%(n, perj[n]['dcur'], perj[n]['dtm'], perj[n]['scale'],
              " ".join("%.1f"%s for s in scs)))
    else:
        print("  %-9s  (자극/표본 부족)"%n)
allx=np.array(allx); ally=np.array(ally)
if len(allx)>=3:
    scale = float(allx@ally/(allx@allx))
    r2 = 1-np.sum((ally-scale*allx)**2)/np.sum((ally-ally.mean())**2)
    print("  ▶ 전체 회귀 SCALE = %.2f  (R²=%.3f · %d점)   [참고 grf=%.2f · 가정 6.4]"%(scale,r2,len(allx),FIX))

# ── [B] grf SCALE 고정 → 관절별 '겉보기 질량' · J 잔차 ──
#   SCALE 을 grf(FIX)로 고정하면 남는 관절별 편차 = J(q) 오차(+실제질량 오차).
#   겉보기 m_j = MASS·|SCALE_j|/FIX  (실제 MASS 이어야 정합). m_j>MASS = 모델 J 과소(예측토크 낮음).
print("\n  [B] grf SCALE=%.2f 고정 → 겉보기 질량 = %.1fkg·|SCALE_j|/%.2f  (실제 %.1fkg 이어야)"%(FIX,MASS,FIX,MASS))
print("  %-9s %10s %9s   판정"%("관절","겉보기m","J잔차%"))
ms=[]
for n in LEG:
    if n not in perj: continue
    m_app = MASS*abs(perj[n]['scale'])/FIX
    res = (m_app/MASS-1)*100
    if abs(perj[n]['dcur']) < 2.0:                       # 무부하/저SNR 축 = 겉보기질량 무의미
        print("  %-9s %9.2fkg %8s   ⚠무부하/저SNR(무시, Δcur=%.1f)"%(n, m_app, "--", perj[n]['dcur'])); continue
    ms.append(m_app)
    tag = "J 과대(예측토크↑→겉보기↓)" if res<-8 else ("J 과소(예측토크↓→겉보기↑)" if res>8 else "≈정합")
    print("  %-9s %9.2fkg %8.0f%%   %s"%(n, m_app, res, tag))
if len(ms)>=2:
    print("  ▶ 겉보기질량 %.2f~%.2fkg (%.2f×) = **관절간 J 상대오차**(grf·실제질량과 무관·robust)."%(min(ms),max(ms),max(ms)/min(ms)))
    print("    절대수준은 grf SCALE·실제질량에 의존 → 스프레드가 순수 J(q) 진단. (FIX_SCALE=env 로 조정)")
