#!/usr/bin/env python3
"""read_qch.py — 채널 q_ch(1차, fPosition) 순수 읽기(무여자·무명령). 영점 검증용.
   RobotEmbedded 만 떠 있으면 됨. bridge_read 만(enable/write 없음)."""
import ctypes as C, numpy as np, time
LIB="/home/rpetubt/simulation/biped/emb/hal/libbipedshm.so"; N=8
NM=["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
PRE=[-1.30,3.76,-10.83,0.24,-1.24,1.17,1.03,-0.06]   # 영점 전 q_ch_deg(참고)
lib=C.CDLL(LIB); F=C.POINTER(C.c_float); I=C.POINTER(C.c_int)
lib.bridge_init.restype=C.c_int; lib.bridge_init.argtypes=[C.c_int]
lib.bridge_read.restype=C.c_int; lib.bridge_read.argtypes=[F]*7+[I,I]
def p(a):return a.ctypes.data_as(F)
def ip(a):return a.ctypes.data_as(I)
if lib.bridge_init(500)!=0: print("bridge_init 실패 — RobotEmbedded?"); raise SystemExit(1)
b=[np.zeros(16,np.float32) for _ in range(7)]; conn=np.zeros(16,np.int32); stt=np.zeros(16,np.int32)
Q=[]
for _ in range(60):
    lib.bridge_read(p(b[0]),p(b[1]),p(b[2]),p(b[3]),p(b[4]),p(b[5]),p(b[6]),ip(conn),ip(stt))
    Q.append(b[0][:N].copy()); time.sleep(0.02)
q=np.median(np.array(Q),axis=0)
print("%-9s %10s %10s %8s"%("ch","q_ch_now°","영점전°","|Δ|"))
mx=0
for i in range(N):
    print("%-9s %10.3f %10.2f %8.2f"%(NM[i],q[i],PRE[i],abs(q[i])))
    mx=max(mx,abs(q[i]))
print("→ 최대 |q_ch| = %.3f° (영점 성공이면 전부 ~0)"%mx)
