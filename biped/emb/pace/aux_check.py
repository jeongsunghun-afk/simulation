#!/usr/bin/env python3
"""aux_check.py — 1차(q_ch) vs 2차(aux) read-only 확인. RobotEmbedded만 떠 있으면 됨."""
import ctypes as C, numpy as np, sys, time
LIB = "/home/rpetubt/simulation/biped/emb/hal/libbipedshm.so"
N = 8
NM = ["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
lib = C.CDLL(LIB)
F = C.POINTER(C.c_float); I = C.POINTER(C.c_int)
lib.bridge_init.restype = C.c_int; lib.bridge_init.argtypes = [C.c_int]
lib.bridge_read.restype = C.c_int; lib.bridge_read.argtypes = [F]*7 + [I, I]
lib.bridge_aux.restype = C.c_int; lib.bridge_aux.argtypes = [F, F]
def p(a): return a.ctypes.data_as(F)
def ip(a): return a.ctypes.data_as(I)
if lib.bridge_init(600) != 0:
    print("bridge_init 실패 — RobotEmbedded 미기동?"); sys.exit(1)
b = [np.zeros(16, np.float32) for _ in range(7)]
conn = np.zeros(16, np.int32); stt = np.zeros(16, np.int32)
ax = np.zeros(16, np.float32); av = np.zeros(16, np.float32)
for _ in range(12):
    lib.bridge_read(p(b[0]),p(b[1]),p(b[2]),p(b[3]),p(b[4]),p(b[5]),p(b[6]), ip(conn), ip(stt))
    lib.bridge_aux(p(ax), p(av)); time.sleep(0.05)
nz = int(np.count_nonzero(ax[:N]))
print("%-9s %9s %9s %9s %5s %6s" % ("ch","q_ch","aux","aux-q_ch","conn","stt"))
for i in range(N):
    print("%-9s %9.2f %9.2f %9.2f %5d  0x%02X" % (NM[i], b[0][i], ax[i], ax[i]-b[0][i], conn[i], int(stt[i])))
print("→ aux nonzero %d/%d  %s" % (nz, N, "✅ aux 수신(0x50 상시)" if nz >= N-1 else "❌ aux 0 — MCU가 0x50에서 aux 미전송(0x5A 필요?)"))
