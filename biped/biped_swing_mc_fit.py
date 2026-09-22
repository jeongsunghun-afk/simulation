#!/usr/bin/env python3
"""biped_swing_mc_fit.py — 자유공간 스윙 로그(swing_logs/*.csv)로 M·C(관성·코리올리) 검증.
   GUI 스윙처프 버튼이 남긴 전관절 q·q̇·τ 를 읽어, MuJoCo 역동역학으로
     τ_model = M(q)q̈ + C(q,q̇)q̇ + g(q) + 마찰(JFRIC·tanh(q̇/v0))
   을 재구성하고 **흔든 관절**의 측정 τ 와 대조한다.

   ⚠전제: 스윙 구간은 **발 공중(접촉0)** 이라 외력=0 → τ_meas 는 순수 관절동역학+마찰.
   ⚠τ_leg_nm 은 fCurrent 규약상 SCALE(≈6.4) 배 커서, 실 Nm 로 쓰려면 /SCALE. grf_verify_base.json
     있으면 그 SCALE 사용(없으면 6.4). 이 SCALE 얽힘은 미해결 항([[project_biped_ethercat_footcable]]).

   사용: python3 biped_swing_mc_fit.py swing_logs/swing_HL_thigh_YYYYMMDD_HHMMSS.csv [SCALE]"""
import sys, os, csv, json, numpy as np, mujoco

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
CSVP = sys.argv[1]
MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'biped_flatfoot.mjcf')
NM   = ['HL_hip','HL_thigh','HL_calf','HL_foot','HR_hip','HR_thigh','HR_calf','HR_foot']
JFRIC = np.array([0.827,0.604,0.871,0.639]*2)   # biped_control.hpp:205 (쿨롱마찰 Nm)
FRIC_V0 = 0.20                                    # biped_control.hpp FRIC_V0
# SCALE (fCurrent 단위계수) — 인자 > grf_verify_base.json > 기본 6.4
SCALE = None
if len(sys.argv) > 2:
    SCALE = float(sys.argv[2])
elif os.path.exists('/tmp/grf_verify_base.json'):
    try: SCALE = float(json.load(open('/tmp/grf_verify_base.json'))['SCALE'])
    except Exception: SCALE = None
if SCALE is None: SCALE = 6.4

rows = list(csv.DictReader(open(CSVP)))
if len(rows) < 30: print('표본 부족(%d)'%len(rows)); sys.exit(1)
swung = rows[0].get('swung') or ''
if swung not in NM:      # 파일명에서 유추
    for n in NM:
        if n in os.path.basename(CSVP): swung = n; break
j = NM.index(swung) if swung in NM else 1
t  = np.array([float(r['t']) for r in rows])
q  = np.deg2rad(np.array([[float(r['q_%s'%n])  for n in NM] for r in rows]))     # rad
dq = np.deg2rad(np.array([[float(r['dq_%s'%n]) for n in NM] for r in rows]))     # rad/s
tau= np.array([[float(r['tau_%s'%n]) for n in NM] for r in rows]) / SCALE        # 실 Nm

# q̈ = d(q̇)/dt (측정 q̇ 미분; 노이즈면 살짝 평활)
def smooth(x, k=5):
    if k < 2: return x
    ker = np.ones(k)/k; return np.convolve(x, ker, mode='same')
qdd = np.gradient(smooth(dq[:,j]), t)            # 흔든 관절 q̈ [rad/s²]

m = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(m)
jid  = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n+'_joint') for n in NM]
qadr = [int(m.jnt_qposadr[jj]) for jj in jid]; dadr = [int(m.jnt_dofadr[jj]) for jj in jid]
dj = dadr[j]
# ★MJCF 는 armature 를 비워둠(배포가 로드시 주입) → 반사 로터관성을 여기서 넣어야 물리적.
#   armature = ROTOR_I·N²  (N=GEAR, 벨트비 포함 [7,7,10.5,8.4]).  ROTOR_I=5.29e-4([[reference_cubemars_ro100_motor]])
ROTOR_I = 5.29e-4; GEARv = np.array([7,7,10.5,8.4,7,7,10.5,8.4], float)
for k2 in range(8): m.dof_armature[dadr[k2]] = ROTOR_I * GEARv[k2]**2
# ★크레인이 몸통을 잡음 → **고정베이스** 가정. 자유부동 베이스로 mj_inverse 하면 비일관(베이스
#   반력 0 가정 위반)이라, 관성토크는 질량행렬 대각 M[dj,dj]·q̈ 로 직접 계산(고정베이스=그 부분행렬).
Mfull = np.zeros((m.nv, m.nv))
g_c=[]; c_c=[]; mI=[]; full=[]
for i in range(len(rows)):
    for k2 in range(8): d.qpos[qadr[k2]] = q[i,k2]
    # 중력만 (q̇=0)
    for k2 in range(m.nv): d.qvel[k2]=0.0
    mujoco.mj_forward(m, d); g_only = d.qfrc_bias[dj]
    # 중력+코리올리 (q̇ 실값)
    for k2 in range(8): d.qvel[dadr[k2]] = dq[i,k2]
    mujoco.mj_forward(m, d); bias = d.qfrc_bias[dj]      # = C q̇ + g
    cori = bias - g_only
    mujoco.mj_fullM(m, Mfull, d.qM)                      # 질량행렬(armature 포함)
    Mqdd = Mfull[dj, dj] * qdd[i]                        # 흔든 관절만 가속 → 대각항이 관성토크
    fric = JFRIC[j]*np.tanh(dq[i,j]/FRIC_V0)
    g_c.append(g_only); c_c.append(cori); mI.append(Mqdd); full.append(Mqdd+bias+fric)
g_c=np.array(g_c); c_c=np.array(c_c); mI=np.array(mI); full=np.array(full)
meas=tau[:,j]

def corr(a,b):
    if np.std(a)<1e-9 or np.std(b)<1e-9: return float('nan')
    return float(np.corrcoef(a,b)[0,1])
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

# 실측 동역학분(중력·마찰 제거) vs 모델 관성+코리올리
fric_all = JFRIC[j]*np.tanh(dq[:,j]/FRIC_V0)
meas_dyn = meas - g_c - fric_all                 # ≈ M q̈ + C q̇
mdl_dyn  = mI + c_c
# 유효관성: meas_dyn ≈ I_eff·q̈ (저주파라 C 작음) 최소제곱
A = qdd.reshape(-1,1); Ieff,_,_,_ = np.linalg.lstsq(A, meas_dyn, rcond=None); Ieff=float(Ieff[0])
Mjj = float(mI @ qdd / (qdd@qdd)) if qdd@qdd>0 else float('nan')   # 모델 유효관성(회귀)

print('■ 스윙 M·C 적합 — 흔든 관절 %s  (SCALE=%.2f 적용, %d표본)'%(NM[j], SCALE, len(rows)))
print('  자극: |q̇|max=%.2f rad/s · |q̈|max=%.1f rad/s²  (저주파=저대역, 관성자극 약할 수 있음)'
      %(np.abs(dq[:,j]).max(), np.abs(qdd).max()))
print('  성분 기여(표준편차, Nm):  g=%.3f  Cq̇=%.3f  Mq̈=%.3f  마찰=%.3f'
      %(np.std(g_c), np.std(c_c), np.std(mI), np.std(fric_all)))
print('  전체 τ:   측정 vs 모델(M q̈+Cq̇+g+마찰)   corr=%.2f  RMSE=%.3f Nm  비(측정/모델 slope)=%.2f'
      %(corr(meas,full), rmse(meas,full), float(meas@full/(full@full)) if full@full>0 else float('nan')))
print('  동역학분(중력·마찰 제거) 측정 vs 모델:     corr=%.2f  RMSE=%.3f Nm'
      %(corr(meas_dyn, mdl_dyn), rmse(meas_dyn, mdl_dyn)))
print('  유효관성 I_eff:  측정 %.4f  vs 모델 %.4f  kg·m²  (비 %.2f)'
      %(Ieff, Mjj, Ieff/Mjj if Mjj and not np.isnan(Mjj) else float('nan')))
print('  → corr↑·비≈1 이면 M(+C) 검증. Mq̈ 표준편차가 g·마찰 대비 너무 작으면 자극부족(슬루완화 필요).')
