#!/usr/bin/env python3
"""grf_verify.py — 알려진 추 하중으로 모델 검증. deploy hold 중 **읽기전용** 실행(단일writer 무관).
   각 실행 = 한 하중점 스냅샷(N샘플 median) → /tmp/grf_verify.csv 누적 + 베이스라인 대비 검증.

   사용:
     발 빈 상태:            grf_verify.py baseline
     footX 에 W kg 매단 뒤:  grf_verify.py <W_kg> <HL|HR>   (예: grf_verify.py 3 HL)
     요약표:                grf_verify.py summary

   검증 2축(차분=중력상쇄):
     [GRF] ΔF_z_추정 vs W·g       — GRF 추정 정확도(역방향)
     [토크] Δτ_측정 vs Jᵀ·(W·g)   — 관절토크 vs 모델(정방향, 하중다리 4관절)
   부호규약: 추 하중 → +GRF (지지력 +). grf_est 와 동일."""
import ctypes as C, numpy as np, sys, os, time, json, csv
import mujoco
LIB="/home/rpetubt/simulation/biped/emb/hal/libbipedshm.so"
MJCF="/home/rpetubt/simulation/biped/biped_flatfoot.mjcf"
STATE=os.environ.get("QUAD_STATE","/dev/shm/biped_state.json")
BASE="/tmp/grf_verify_base.json"; CSV="/tmp/grf_verify.csv"
N=8; G=9.81; NSAMP=int(os.environ.get("NSAMP","600"))
NM=["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
SIGN=np.array([-1,1,-1,-1,-1,-1,1,1],float); GEAR=np.array([7,7,10.5,8.4,7,7,10.5,8.4],float); KT=0.2

arg = sys.argv[1] if len(sys.argv)>1 else "baseline"
if arg=="summary":
    if not os.path.exists(CSV): print("아직 데이터 없음"); sys.exit(0)
    rows=list(csv.DictReader(open(CSV)))
    print("%-6s %-4s %8s %8s %8s | %8s"%("W_kg","foot","Wg_N","dFz_N","오차%","dFz/Wg"))
    for r in rows:
        if r["type"]!="load": continue
        W=float(r["W_kg"]); Wg=W*G; dF=float(r["dFz"])
        print("%-6s %-4s %8.1f %8.1f %8.0f | %8.2f"%(r["W_kg"],r["foot"],Wg,dF,(dF/Wg-1)*100 if Wg else 0, dF/Wg if Wg else 0))
    print("→ 선형이고 dFz/Wg≈1 이면 GRF 모델 검증됨. (계수 c 로 일정하면 SCALE 보정으로 흡수 가능)")
    sys.exit(0)

is_base = (arg=="baseline")
W = 0.0 if is_base else float(arg)
foot = (sys.argv[2] if len(sys.argv)>2 else "HL") if not is_base else "-"
leg = 0 if foot=="HL" else 1

m=mujoco.MjModel.from_xml_path(MJCF); d=mujoco.MjData(m)
jid=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n+"_joint") for n in NM]
qadr=[int(m.jnt_qposadr[j]) for j in jid]; dadr=[int(m.jnt_dofadr[j]) for j in jid]
sph=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f) for f in ['HL_sphere','HR_sphere']]
fbody=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,b) for b in ['HL_foot_contact_link','HR_foot_contact_link']]
nv=m.nv
lib=C.CDLL(LIB); F=C.POINTER(C.c_float); I=C.POINTER(C.c_int)
lib.bridge_init.restype=C.c_int; lib.bridge_init.argtypes=[C.c_int]
lib.bridge_read.restype=C.c_int; lib.bridge_read.argtypes=[F]*7+[I,I]
def p(a):return a.ctypes.data_as(F)
def ip(a):return a.ctypes.data_as(I)
if lib.bridge_init(500)!=0: print("bridge_init 실패 — RobotEmbedded 미기동?"); sys.exit(1)
b=[np.zeros(16,np.float32) for _ in range(7)]; conn=np.zeros(16,np.int32); stt=np.zeros(16,np.int32)
def read_cur():
    lib.bridge_read(p(b[0]),p(b[1]),p(b[2]),p(b[3]),p(b[4]),p(b[5]),p(b[6]),ip(conn),ip(stt))
    return b[3][:N].copy()

print("→ %s: %d샘플 median 수집 …" % ("베이스라인(발빈)" if is_base else "%s %.1fkg"%(foot,W), NSAMP), flush=True)
CUR=[]; Q=[]; t0=time.time()
try: mode=json.load(open(STATE)).get("mode")
except Exception: mode=None
while len(CUR)<NSAMP and time.time()-t0<NSAMP/250+4:
    CUR.append(read_cur())
    try: q=json.load(open(STATE)).get("q_leg_deg") or [0.0]*N
    except Exception: q=[0.0]*N
    Q.append(q); time.sleep(0.003)
cur=np.median(np.array(CUR),axis=0); q=np.median(np.array(Q),axis=0)
if mode not in ("hold","stand"):
    print("  ⚠ 현재 mode=%s (hold/stand 아님) — 하중지지 상태서 측정할 것"%mode, flush=True)
for i in range(N): d.qpos[qadr[i]]=np.deg2rad(q[i])
mujoco.mj_forward(m,d)
tau_meas=SIGN*cur*KT*GEAR
tau_grav=np.array([d.qfrc_bias[dadr[i]] for i in range(N)])
def jac(k):
    jacp=np.zeros((3,nv)); mujoco.mj_jac(m,d,jacp,None,d.geom_xpos[sph[k]],fbody[k]); return jacp
def leg_Fz(tm,SC):
    out=[0.,0.]
    for lg in range(2):
        cols=[dadr[lg*4+kk] for kk in range(4)]; J=jac(lg)[:,cols]
        Fsol,*_=np.linalg.lstsq(J.T, tm[lg*4:lg*4+4]/SC - tau_grav[lg*4:lg*4+4], rcond=None)
        out[lg]=float(Fsol[2])
    return out

def append_csv(row):
    new=not os.path.exists(CSV)
    with open(CSV,"a",newline="") as f:
        w=csv.writer(f)
        if new: w.writerow(["t","type","W_kg","foot","mode","SCALE","dFz"]+["q_%s"%n for n in NM]+["tau_%s"%n for n in NM])
        w.writerow(row)

if is_base:
    mask=np.abs(tau_grav)>0.5
    SCALE=float(np.median(np.abs(tau_meas[mask])/np.abs(tau_grav[mask]))) if mask.any() else 7.16
    SCALE=min(max(SCALE,0.05),20)
    Fz=leg_Fz(tau_meas,SCALE)
    json.dump({"SCALE":SCALE,"tau_meas":tau_meas.tolist(),"q":q.tolist()}, open(BASE,"w"))
    append_csv([("%.1f"%time.time()),"baseline",0,"-",mode,"%.3f"%SCALE,0]+["%.2f"%x for x in q]+["%.3f"%x for x in tau_meas])
    print("■ 베이스라인 저장: SCALE=%.3f · F_z(HL/HR)=%.1f/%.1f N (발 빈이라 ~0 이어야)"%(SCALE,Fz[0],Fz[1]),flush=True)
    print("  q°:", " ".join("%.1f"%x for x in q),flush=True)
    print("  → 이제 추 매달고: grf_verify.py <kg> <HL|HR>",flush=True)
else:
    if not os.path.exists(BASE): print("✗ 베이스라인 없음 — 먼저 grf_verify.py baseline"); sys.exit(1)
    base=json.load(open(BASE)); SCALE=base["SCALE"]; tau0=np.array(base["tau_meas"])
    Fz0=leg_Fz(tau0,SCALE); Fz1=leg_Fz(tau_meas,SCALE)
    Wg=W*G; dF=Fz1[leg]-Fz0[leg]
    cols=[dadr[leg*4+kk] for kk in range(4)]; J=jac(leg)[:,cols]
    tau_model = J.T @ np.array([0,0,Wg])                 # 모델 예측 Δτ (real Nm, +GRF 규약)
    dtau_meas = (tau_meas[leg*4:leg*4+4]-tau0[leg*4:leg*4+4])/SCALE
    append_csv([("%.1f"%time.time()),"load","%.2f"%W,foot,mode,"%.3f"%SCALE,"%.2f"%dF]+["%.2f"%x for x in q]+["%.3f"%x for x in tau_meas])
    print("\n■ %s 에 %.2fkg (%.1f N) — 모델 검증 (SCALE=%.2f, mode=%s)"%(foot,W,Wg,SCALE,mode),flush=True)
    print("  [GRF 역방향] ΔF_z_추정 = %+.1f N  vs  적용 %.1f N  → 오차 %+.0f%% (비 %.2f)"
          %(dF,Wg,(dF/Wg-1)*100 if Wg else 0, dF/Wg if Wg else 0),flush=True)
    print("  [토크 정방향] Δτ (real Nm) — 하중다리 %s"%foot,flush=True)
    print("    %-9s %10s %10s %8s"%("joint","측정Δτ","모델Jᵀ·Wg","비"),flush=True)
    for kk in range(4):
        r = dtau_meas[kk]/tau_model[kk] if abs(tau_model[kk])>1e-3 else float("nan")
        print("    %-9s %10.3f %10.3f %8.2f"%(NM[leg*4+kk],dtau_meas[kk],tau_model[kk],r),flush=True)
    print("  → ΔF_z/Wg≈1 이고 Δτ측정≈모델이면 이 하중점 검증 OK. 여러 무게로 선형성은 summary 로.",flush=True)
