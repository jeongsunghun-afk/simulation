#!/usr/bin/env python3
"""backlash_jog.py — per-joint 백래시(lost motion) 측정. 삼각파 저속 스윕 + aux 히스테리시스.
   원리: 1차(q_ch=모터측)를 정→역 반전시키며 aux(출력측)를 관측. 같은 q_ch 에서 정/역 pass 의
        aux 차 = 백래시(출력 데드밴드). 노이즈는 q_ch 구간별 **median**(0-드롭아웃 제거)으로 억제.

   ★능동 모션이다. 실행 전 필수:
     - 팬 장착 후 온도 안정(<70°C). MAXTEMP 초과면 자동 중단.
     - 크레인 매달림 상태. AMP 작게(기본 3°), SPEED 느리게(기본 12dps).
     - 단일 writer(deploy/emb/gui 동시 금지). RobotEmbedded 는 떠 있어야 함.
     - ⚠HL_calf(ch2)=왼무릎 벨트슬립 → 결과는 백래시 아니라 벨트슬립일 수 있음(비신뢰).

   먼저 DRY=1 로 계획·현재자세·노이즈만 확인하고, 이상 없으면 DRY=0 로 실측.
   env: JOINTS=all|"2"|"0,1,2,3"  AMP_DEG=3  SPEED_DPS=12  CYCLES=2  N_SETTLE=400
        DRY=1  MAXTEMP=88  KP="60,40,60,25,60,40,60,25" KD="3,2,3,1.5,3,2,3,1.5"
   실행: AUX_MODE=1 /home/rpetubt/.venv-mujoco/bin/python backlash_jog.py"""
import ctypes as C, numpy as np, sys, time, os
LIB = "/home/rpetubt/simulation/biped/emb/hal/libbipedshm.so"
TEMPF = "/sys/class/thermal/thermal_zone0/temp"
N = 8
NM = ["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]

AMP   = float(os.environ.get("AMP_DEG", "3"))
SPEED = float(os.environ.get("SPEED_DPS", "12"))
CYC   = int(os.environ.get("CYCLES", "2"))
NSET  = int(os.environ.get("N_SETTLE", "400"))
DRY   = os.environ.get("DRY", "1") == "1"
MAXT  = float(os.environ.get("MAXTEMP", "88"))
HZ    = 200.0
def _pl(s, d):
    v = os.environ.get(s);
    return [float(x) for x in v.split(",")] if v else d
KP = _pl("KP", [60,40,60,25,60,40,60,25])
KD = _pl("KD", [3,2,3,1.5,3,2,3,1.5])
jsel = os.environ.get("JOINTS", "all")
JOINTS = list(range(N)) if jsel == "all" else [int(x) for x in jsel.split(",")]

def temp():
    try: return int(open(TEMPF).read())/1000.0
    except Exception: return float("nan")

lib = C.CDLL(LIB); F = C.POINTER(C.c_float); I = C.POINTER(C.c_int)
lib.bridge_init.restype = C.c_int; lib.bridge_init.argtypes = [C.c_int]
lib.bridge_read.restype = C.c_int; lib.bridge_read.argtypes = [F]*7 + [I, I]
lib.bridge_aux.restype = C.c_int; lib.bridge_aux.argtypes = [F, F]
lib.bridge_write_mit.restype = C.c_int; lib.bridge_write_mit.argtypes = [F]*5 + [C.c_int]
lib.bridge_enable.restype = C.c_int; lib.bridge_enable.argtypes = [C.c_int]
def p(a): return a.ctypes.data_as(F)
def ip(a): return a.ctypes.data_as(I)
if lib.bridge_init(int(HZ*3)) != 0: print("bridge_init 실패 — RobotEmbedded?"); sys.exit(1)
b = [np.zeros(16, np.float32) for _ in range(7)]
conn = np.zeros(16, np.int32); stt = np.zeros(16, np.int32)
ax = np.zeros(16, np.float32); av = np.zeros(16, np.float32); z = np.zeros(16, np.float32)

def read():
    lib.bridge_read(p(b[0]),p(b[1]),p(b[2]),p(b[3]),p(b[4]),p(b[5]),p(b[6]), ip(conn), ip(stt))
    lib.bridge_aux(p(ax), p(av))
    return b[0].copy(), ax.copy()                       # q_ch(1차), aux
q0, aux0 = read()

def rmed(v):
    """median + 0-드롭아웃/이상치 제거."""
    v = np.asarray(v, float); v = v[v != 0.0]
    if len(v) < 3: return float("nan"), float("nan"), 0
    m = np.median(v); s = 1.4826*np.median(np.abs(v-m))    # robust std(MAD)
    good = v[np.abs(v-m) < 5*max(s,1e-6)]
    return float(np.median(good)), float(good.std()), len(good)

print("== backlash_jog == AMP±%.1f° SPEED %.0fdps CYC %d N_settle %d %s" %
      (AMP, SPEED, CYC, NSET, "· ★DRY(무모션)" if DRY else "· 실측(능동모션)"), flush=True)
print("현재 온도 %.1f°C (MAXTEMP %.0f)" % (temp(), MAXT), flush=True)
print("현재 자세 q_ch° :", " ".join("%.1f" % q0[i] for i in range(N)), flush=True)
print("현재 aux°       :", " ".join("%.1f" % aux0[i] for i in range(N)), flush=True)

if DRY:
    print("\n[DRY] enable·토크0(limp·무모션)로 aux 노이즈 0.8s 샘플 …", flush=True)
    A = [[] for _ in range(N)]
    lib.bridge_enable(1)
    try:
        t0=time.time()
        while time.time()-t0 < 0.8:
            lib.bridge_write_mit(p(q0),p(z),p(z),p(z),p(z),N)   # 토크0 = 무모션
            _, a = read()
            for i in range(N): A[i].append(a[i])
            time.sleep(0.005)
    finally:
        lib.bridge_enable(0)
        for _ in range(15): lib.bridge_write_mit(p(q0),p(z),p(z),p(z),p(z),N); time.sleep(0.01)
    print("%-9s %9s %9s" % ("ch","aux_med°","aux_std°"), flush=True)
    for i in range(N):
        med,std,ng = rmed(A[i]); print("%-9s %9.3f %9.3f" % (NM[i], med, std), flush=True)
    print("\n[DRY] 계획: 대상 %s · 각 축 q0±%.1f° 삼각파 %d회. 이상없으면 DRY=0 로 실측." %
          ([NM[j] for j in JOINTS], AMP, CYC), flush=True)
    sys.exit(0)

if temp() > MAXT:
    print("✗ 온도 %.1f°C > MAXTEMP %.0f — 중단(냉각 먼저)" % (temp(), MAXT)); sys.exit(1)

kp = np.array(KP, np.float32); kd = np.array(KD, np.float32)
qdes = q0.copy().astype(np.float32)
results = {}
lib.bridge_enable(1)
try:
    # 초기 홀드 안정화
    t0=time.time()
    while time.time()-t0 < 1.0:
        lib.bridge_write_mit(p(qdes),p(z),p(z),p(kp),p(kd),N); read(); time.sleep(1/HZ)
    for j in JOINTS:
        if temp() > MAXT: print("✗ 온도 초과 — 중단"); break
        print("\n── %s(ch%d) 스윕 …" % (NM[j], j), flush=True)
        # 삼각파 목표열: q0 → +AMP → -AMP → q0, CYC회
        segs = []
        for _ in range(CYC): segs += [q0[j]+AMP, q0[j]-AMP]
        segs += [q0[j]]
        log = []   # (qdes_j, q_ch_j, aux_j, dir)
        cur = q0[j]
        for tgt in segs:
            direction = 1 if tgt > cur else -1
            step = SPEED/HZ*direction
            while (direction>0 and cur < tgt) or (direction<0 and cur > tgt):
                cur = min(tgt,cur+step) if direction>0 else max(tgt,cur-step)
                qdes[j] = cur
                lib.bridge_write_mit(p(qdes),p(z),p(z),p(kp),p(kd),N)
                qc, a = read()
                log.append((cur, qc[j], a[j], direction))
                time.sleep(1/HZ)
        L = np.array(log)
        qc = L[:,1]; auxj = L[:,2]; dr = L[:,3]
        # 히스테리시스: 겹치는 q_ch 구간을 bin, 정/역 median aux 차 = 백래시
        lo,hi = np.percentile(qc,5), np.percentile(qc,95)
        bins = np.linspace(lo,hi,12); bl=[]
        for k in range(len(bins)-1):
            m = (qc>=bins[k])&(qc<bins[k+1])
            up = auxj[m&(dr>0)]; dn = auxj[m&(dr<0)]
            mu,_,nu = rmed(up); md,_,nd = rmed(dn)
            if nu>=5 and nd>=5 and np.isfinite(mu) and np.isfinite(md): bl.append(mu-md)
        blk = float(np.median(np.abs(bl))) if bl else float("nan")
        results[NM[j]] = blk
        print("  백래시(aux 히스테리시스) ≈ %.3f°  (bin %d개, 샘플 %d)" % (blk, len(bl), len(L)), flush=True)
        # 원자료 저장(오프라인 플롯용)
        np.savetxt("/tmp/backlash_%s.csv"%NM[j], L, delimiter=",",
                   header="qdes,q_ch,aux,dir", comments="")
        # 원위치 복귀
        qdes[j]=q0[j]
        t0=time.time()
        while time.time()-t0<0.5: lib.bridge_write_mit(p(qdes),p(z),p(z),p(kp),p(kd),N); read(); time.sleep(1/HZ)
finally:
    lib.bridge_enable(0)
    for _ in range(30): lib.bridge_write_mit(p(q0),p(z),p(z),p(z),p(z),N); time.sleep(0.01)

print("\n════ 백래시 요약 ════", flush=True)
for nmj, v in results.items():
    flag = " ⚠벨트슬립(비신뢰)" if nmj=="HL_calf" else ""
    print("  %-9s %.3f°%s" % (nmj, v, flag), flush=True)
print("  (원자료 /tmp/backlash_*.csv · aux 히스테리시스 폭 = 출력측 백래시)", flush=True)
print("종료 온도 %.1f°C" % temp(), flush=True)
