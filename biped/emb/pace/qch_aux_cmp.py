#!/usr/bin/env python3
"""qch_aux_cmp.py — 1차(q_ch) vs 2차(aux) 비교. 0x5A·토크0(limp·무모션)로 N샘플 median.
   채널별 q_ch·aux·(aux−q_ch) 출력. aux−q_ch = 감속단 비틀림/백래시(부하 있을 때) 또는 정렬차.
   RobotEmbedded 만 떠 있으면 됨. 단일 writer. 실행: AUX_MODE=1 python qch_aux_cmp.py [dur]"""
import ctypes as C, numpy as np, sys, time
LIB="/home/rpetubt/simulation/biped/emb/hal/libbipedshm.so"; N=8
NM=["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
DUR=float(sys.argv[1]) if len(sys.argv)>1 else 6.0
lib=C.CDLL(LIB); F=C.POINTER(C.c_float); I=C.POINTER(C.c_int)
lib.bridge_init.restype=C.c_int; lib.bridge_init.argtypes=[C.c_int]
lib.bridge_read.restype=C.c_int; lib.bridge_read.argtypes=[F]*7+[I,I]
lib.bridge_aux.restype=C.c_int; lib.bridge_aux.argtypes=[F,F]
lib.bridge_write_mit.restype=C.c_int; lib.bridge_write_mit.argtypes=[F]*5+[C.c_int]
lib.bridge_enable.restype=C.c_int; lib.bridge_enable.argtypes=[C.c_int]
def p(a):return a.ctypes.data_as(F)
def ip(a):return a.ctypes.data_as(I)
if lib.bridge_init(500)!=0: print("bridge_init 실패 — RobotEmbedded?"); sys.exit(1)
b=[np.zeros(16,np.float32) for _ in range(7)];conn=np.zeros(16,np.int32);stt=np.zeros(16,np.int32)
ax=np.zeros(16,np.float32);av=np.zeros(16,np.float32);z=np.zeros(16,np.float32)
lib.bridge_read(p(b[0]),p(b[1]),p(b[2]),p(b[3]),p(b[4]),p(b[5]),p(b[6]),ip(conn),ip(stt));q0=b[0].copy()
Q=[[] for _ in range(N)];A=[[] for _ in range(N)]
print("→ 0x5A·토크0(limp·무모션) %.0fs 수집 …"%DUR,flush=True)
lib.bridge_enable(1)
try:
    t0=time.time()
    while time.time()-t0<DUR:
        lib.bridge_write_mit(p(q0),p(z),p(z),p(z),p(z),N)
        lib.bridge_read(p(b[0]),p(b[1]),p(b[2]),p(b[3]),p(b[4]),p(b[5]),p(b[6]),ip(conn),ip(stt))
        lib.bridge_aux(p(ax),p(av))
        for i in range(N): Q[i].append(float(b[0][i])); A[i].append(float(ax[i]))
        time.sleep(0.005)
finally:
    lib.bridge_enable(0)
    for _ in range(15): lib.bridge_write_mit(p(q0),p(z),p(z),p(z),p(z),N); time.sleep(0.01)
def med(v):
    v=np.array(v); v=v[v!=0.0] if (v!=0).any() else v
    return float(np.median(v)) if len(v) else float("nan")
print("%-9s %9s %9s %10s"%("ch","q_ch°(1차)","aux°(2차)","aux−q_ch°"),flush=True)
print("-"*42,flush=True)
mx=0
for i in range(N):
    qm=med(Q[i]); am=med(A[i]); d=am-qm
    print("%-9s %9.3f %9.3f %10.3f"%(NM[i],qm,am,d),flush=True); mx=max(mx,abs(d))
print("-"*42,flush=True)
print("→ 최대 |aux−q_ch| = %.3f°.  limp 면 ~0(추종), hold+부하 면 비틀림/백래시."%mx,flush=True)
