#!/usr/bin/env python3
"""aux_check_en.py — 0x50 enable(토크0·limp)로 MCU를 ENA 상태에 두고 1차 vs 2차(aux) 확인.
   ★토크0(kp=kd=tau=0)이라 여자되지만 능동토크 없음 = 무모션. RobotEmbedded 떠 있어야 함.
   단일 writer여야 함(deploy/emb 동시 금지)."""
import ctypes as C, numpy as np, sys, time
LIB = "/home/rpetubt/simulation/biped/emb/hal/libbipedshm.so"
N = 8
NM = ["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
lib = C.CDLL(LIB)
F = C.POINTER(C.c_float); I = C.POINTER(C.c_int)
lib.bridge_init.restype = C.c_int; lib.bridge_init.argtypes = [C.c_int]
lib.bridge_read.restype = C.c_int; lib.bridge_read.argtypes = [F]*7 + [I, I]
lib.bridge_aux.restype = C.c_int; lib.bridge_aux.argtypes = [F, F]
lib.bridge_write_mit.restype = C.c_int; lib.bridge_write_mit.argtypes = [F]*5 + [C.c_int]
lib.bridge_enable.restype = C.c_int; lib.bridge_enable.argtypes = [C.c_int]
def p(a): return a.ctypes.data_as(F)
def ip(a): return a.ctypes.data_as(I)
if lib.bridge_init(600) != 0:
    print("bridge_init 실패 — RobotEmbedded 미기동?"); sys.exit(1)
b = [np.zeros(16, np.float32) for _ in range(7)]
conn = np.zeros(16, np.int32); stt = np.zeros(16, np.int32)
ax = np.zeros(16, np.float32); av = np.zeros(16, np.float32)
z = np.zeros(16, np.float32)
# 현재 q 래치 (q_des=현재각 · kp=0 이라 효과 없지만 안전차원)
lib.bridge_read(p(b[0]),p(b[1]),p(b[2]),p(b[3]),p(b[4]),p(b[5]),p(b[6]), ip(conn), ip(stt))
q0 = b[0].copy()
print("→ 0x50 enable(토크0·limp) 2.5s 유지하며 aux 관측 …")
lib.bridge_enable(1)
try:
    t0 = time.time()
    while time.time() - t0 < 2.5:
        lib.bridge_write_mit(p(q0), p(z), p(z), p(z), p(z), N)   # q_des,dq,tau,kp,kd = current,0,0,0,0
        lib.bridge_read(p(b[0]),p(b[1]),p(b[2]),p(b[3]),p(b[4]),p(b[5]),p(b[6]), ip(conn), ip(stt))
        lib.bridge_aux(p(ax), p(av))
        time.sleep(0.01)
finally:
    # 확실히 limp 로 종료
    lib.bridge_enable(0)
    for _ in range(20):
        lib.bridge_write_mit(p(q0), p(z), p(z), p(z), p(z), N); time.sleep(0.01)
nz = int(np.count_nonzero(ax[:N]))
print("%-9s %9s %9s %9s %5s %6s" % ("ch","q_ch","aux","aux-q_ch","conn","stt"))
for i in range(N):
    print("%-9s %9.2f %9.2f %9.2f %5d  0x%02X" % (NM[i], b[0][i], ax[i], ax[i]-b[0][i], conn[i], int(stt[i])))
print("→ aux nonzero %d/%d  %s" % (nz, N,
      "✅ aux 수신(0x50 enable) — .19 MCU 업데이트 확정" if nz >= N-1 else
      "❌ 0x50 enable서도 aux 0 — MCU가 여전히 aux 미전송(RGA 확인)"))
