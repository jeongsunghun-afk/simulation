#!/usr/bin/env python3
"""jog_zero_aux.py — 0x5A(AUX_MODE)로 전축을 채널영점(q_ch=0)으로 천천히 jog 후 1차(q) vs 2차(aux) 측정.
   ★단일 writer. RobotEmbedded 떠 있어야. 크레인 매달림 전제.
   안전: ①현자세 hold 로 0x5A 제어 먼저 검증(무모션) ②kp 램프인 ③느린 코사인 램프
        ④추종오차/속도/토크/드라이버에러 abort ⑤예외·종료 시 항상 limp.
   실행: MOT_BASE_MODE 불요(스크립트가 AUX_MODE 로 0x5A). RobotEmbedded 만 떠 있으면 됨."""
import ctypes as C, numpy as np, sys, os, time
os.environ["AUX_MODE"] = "1"   # bridge_init 에서 0x5A 선택
LIB = "/home/rpetubt/simulation/biped/emb/hal/libbipedshm.so"
N = 8
NM = ["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
KP = np.array([100,50,80,30,100,50,80,30], np.float32)   # config 값(채널프레임)
KD = np.array([6,4,3.5,2,6,4,3.5,2], np.float32)
ERR_ABORT, VEL_ABORT, TAU_ABORT = 18.0, 220.0, 8.0
DT, T_JOG = 0.006, 5.0

lib = C.CDLL(LIB)
F = C.POINTER(C.c_float); I = C.POINTER(C.c_int)
lib.bridge_init.restype = C.c_int; lib.bridge_init.argtypes = [C.c_int]
lib.bridge_read.restype = C.c_int; lib.bridge_read.argtypes = [F]*7 + [I, I]
lib.bridge_aux.restype = C.c_int; lib.bridge_aux.argtypes = [F, F]
lib.bridge_write_mit.restype = C.c_int; lib.bridge_write_mit.argtypes = [F]*5 + [C.c_int]
lib.bridge_enable.restype = C.c_int; lib.bridge_enable.argtypes = [C.c_int]
def p(a): return a.ctypes.data_as(F)
def ip(a): return a.ctypes.data_as(I)
buf = [np.zeros(16, np.float32) for _ in range(7)]
conn = np.zeros(16, np.int32); stt = np.zeros(16, np.int32)
ax = np.zeros(16, np.float32); av = np.zeros(16, np.float32)
Z = np.zeros(16, np.float32)

def rd():
    lib.bridge_read(p(buf[0]),p(buf[1]),p(buf[2]),p(buf[3]),p(buf[4]),p(buf[5]),p(buf[6]), ip(conn), ip(stt))
    lib.bridge_aux(p(ax), p(av))
def wr(qd, kp, kd):
    q=Z.copy(); k=Z.copy(); d=Z.copy(); q[:N]=qd; k[:N]=kp; d[:N]=kd
    lib.bridge_write_mit(p(q), p(Z), p(Z), p(k), p(d), N)   # dq_des=0, tau_ff=0
def limp():
    lib.bridge_enable(0)
    for _ in range(40):
        lib.bridge_write_mit(p(Z), p(Z), p(Z), p(Z), p(Z), N); time.sleep(0.005)
def check(qd):
    q=buf[0][:N]; dq=buf[1][:N]; tau=buf[2][:N]; e=np.abs(q-qd)
    if e.max() > ERR_ABORT:  return "추종오차 %.1f° @%s" % (e.max(), NM[int(e.argmax())])
    if np.abs(dq).max() > VEL_ABORT: return "속도 %.0fdps @%s" % (np.abs(dq).max(), NM[int(np.abs(dq).argmax())])
    if np.abs(tau).max() > TAU_ABORT: return "토크 %.1fNm @%s" % (np.abs(tau).max(), NM[int(np.abs(tau).argmax())])
    bad = [NM[i] for i in range(N) if int(stt[i]) != 0]
    if bad: return "드라이버에러 %s" % bad
    return None

if lib.bridge_init(800) != 0:
    print("bridge_init 실패 — RobotEmbedded 미기동?"); sys.exit(1)
rd(); q0 = buf[0][:N].copy()
print("시작 q_ch =", [round(float(x),1) for x in q0], flush=True)
abort = None; measured = None
try:
    lib.bridge_enable(1)
    # ── Phase A: 현자세 hold, kp 램프인 1.2s (0x5A 제어 검증·무모션) ──
    t0 = time.time()
    while (t := time.time()-t0) < 1.2:
        wr(q0, KP*(t/1.2), KD*(t/1.2)); rd()
        if (abort := check(q0)): break
        time.sleep(DT)
    # ── Phase A2: full-kp hold 0.6s ──
    if not abort:
        t0 = time.time()
        while time.time()-t0 < 0.6:
            wr(q0, KP, KD); rd()
            if (abort := check(q0)): break
            time.sleep(DT)
        if not abort: print("  ✓ 0x5A 제어 hold 정상 — 영점으로 jog 시작", flush=True)
    # ── Phase B: q0 → 0, 코사인 T_JOG ──
    if not abort:
        t0 = time.time()
        while (t := time.time()-t0) < T_JOG:
            s = 0.5*(1-np.cos(np.pi*t/T_JOG)); qd = q0*(1-s)
            wr(qd, KP, KD); rd()
            if (abort := check(qd)): break
            time.sleep(DT)
    # ── Phase C: hold 0 + 측정(정착 후 0.8s 평균) ──
    if not abort:
        aq = np.zeros(N); aa = np.zeros(N); n = 0; t0 = time.time()
        while time.time()-t0 < 1.6:
            wr(Z[:N], KP, KD); rd()
            if (abort := check(Z[:N])): break
            if time.time()-t0 > 0.8:
                aq += buf[0][:N]; aa += ax[:N]; n += 1
            time.sleep(DT)
        if not abort and n: measured = (aq/n, aa/n, n)
finally:
    limp()
if abort:
    print("⛔ ABORT:", abort, "→ limp 종료", flush=True); sys.exit(2)
qm, am, n = measured
print("\n=== 영점(q_ch≈0) 1차 vs 2차(aux) — %d샘플 평균 ===" % n)
print("%-9s %8s %8s %9s" % ("ch","q_ch","aux","aux-q_ch"))
for i in range(N):
    print("%-9s %8.2f %8.2f %9.2f" % (NM[i], qm[i], am[i], am[i]-qm[i]))
print("→ aux 영점차(=2차 자기영점 offset): 위 aux-q_ch 열 (q_ch≈0 이므로 aux 값 ≈ offset)")
