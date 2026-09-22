#!/usr/bin/env python3
"""aux_avg.py — aux 소프트웨어 평균 실증. 0x5A(AUX_MODE=1) enable·토크0(limp·무모션)로
   정지자세서 aux·q_ch N샘플 수집 → 채널별 원시 std(노이즈) vs 평균 불확실도(σ/√N) 대조.
   RobotEmbedded 떠 있어야 함. 단일 writer(deploy/emb 동시 금지). 저발열."""
import ctypes as C, numpy as np, sys, time, os
LIB = "/home/rpetubt/simulation/biped/emb/hal/libbipedshm.so"
N = 8
NM = ["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
lib = C.CDLL(LIB); F = C.POINTER(C.c_float); I = C.POINTER(C.c_int)
lib.bridge_init.restype = C.c_int; lib.bridge_init.argtypes = [C.c_int]
lib.bridge_read.restype = C.c_int; lib.bridge_read.argtypes = [F]*7 + [I, I]
lib.bridge_aux.restype = C.c_int; lib.bridge_aux.argtypes = [F, F]
lib.bridge_write_mit.restype = C.c_int; lib.bridge_write_mit.argtypes = [F]*5 + [C.c_int]
lib.bridge_enable.restype = C.c_int; lib.bridge_enable.argtypes = [C.c_int]
def p(a): return a.ctypes.data_as(F)
def ip(a): return a.ctypes.data_as(I)
if lib.bridge_init(600) != 0: print("bridge_init 실패 — RobotEmbedded 미기동?"); sys.exit(1)
b = [np.zeros(16, np.float32) for _ in range(7)]
conn = np.zeros(16, np.int32); stt = np.zeros(16, np.int32)
ax = np.zeros(16, np.float32); av = np.zeros(16, np.float32); z = np.zeros(16, np.float32)
lib.bridge_read(p(b[0]),p(b[1]),p(b[2]),p(b[3]),p(b[4]),p(b[5]),p(b[6]), ip(conn), ip(stt))
q0 = b[0].copy()
A = [[] for _ in range(N)]; Q = [[] for _ in range(N)]
print("→ 0x5A(AUX_MODE) enable·토크0(limp·무모션) %.0fs aux 수집 …" % DUR, flush=True)
lib.bridge_enable(1)
try:
    t0 = time.time()
    while time.time() - t0 < DUR:
        lib.bridge_write_mit(p(q0), p(z), p(z), p(z), p(z), N)   # q_des=현재,dq=tau=kp=kd=0 → limp
        lib.bridge_read(p(b[0]),p(b[1]),p(b[2]),p(b[3]),p(b[4]),p(b[5]),p(b[6]), ip(conn), ip(stt))
        lib.bridge_aux(p(ax), p(av))
        for i in range(N):
            A[i].append(float(ax[i])); Q[i].append(float(b[0][i]))
        time.sleep(0.005)
finally:
    lib.bridge_enable(0)
    for _ in range(20):
        lib.bridge_write_mit(p(q0), p(z), p(z), p(z), p(z), N); time.sleep(0.01)

print("\n%-9s %5s %9s %10s %11s %9s %9s" %
      ("ch","N","aux평균°","원시std°","평균σ/√N°","ptp°","q_ch std°"), flush=True)
print("-"*70, flush=True)
worst_raw = 0.0; worst_sem = 0.0
for i in range(N):
    a = np.array(A[i]); q = np.array(Q[i]); n = len(a)
    if n < 10 or np.count_nonzero(a) == 0:
        print("%-9s aux 0/데이터부족(n=%d) — AUX_MODE=1/0x5A 확인" % (NM[i], n)); continue
    raw = float(a.std()); sem = raw/np.sqrt(n); qstd = float(q.std())
    print("%-9s %5d %9.3f %10.3f %11.4f %9.3f %9.4f" %
          (NM[i], n, a.mean(), raw, sem, a.ptp(), qstd), flush=True)
    worst_raw = max(worst_raw, raw); worst_sem = max(worst_sem, sem)
print("-"*70, flush=True)
if worst_raw > 0:
    print("■ 원시 노이즈 최악 %.3f° → %d샘플 평균 후 %.4f° (%.0f배 개선)" %
          (worst_raw, len(A[0]), worst_sem, worst_raw/max(worst_sem,1e-9)), flush=True)
    print("  σ/√N 이라 백래시 0.05° 분해하려면 N≈%d 필요(원시 std %.2f° 기준)" %
          (int((worst_raw/0.05)**2), worst_raw), flush=True)
