import ctypes as C, numpy as np, sys, os, time
LIB="/home/rpetubt/simulation/biped/emb/hal/libbipedshm.so"; N=8
NM=["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
mode=int(os.environ.get("MOT_BASE_MODE","0x5A"),0); ena=1 if (mode & 0x10) else 0
T=float(sys.argv[1]) if len(sys.argv)>1 else 4.0
lib=C.CDLL(LIB); F=C.POINTER(C.c_float); I=C.POINTER(C.c_int)
for nm,rt,at in [("bridge_init",C.c_int,[C.c_int]),("bridge_read",C.c_int,[F]*7+[I,I]),
                 ("bridge_aux",C.c_int,[F,F]),("bridge_write_mit",C.c_int,[F]*5+[C.c_int]),
                 ("bridge_enable",C.c_int,[C.c_int])]:
    getattr(lib,nm).restype=rt; getattr(lib,nm).argtypes=at
def p(a):return a.ctypes.data_as(F)
def ip(a):return a.ctypes.data_as(I)
b=[np.zeros(16,np.float32) for _ in range(7)];conn=np.zeros(16,np.int32);stt=np.zeros(16,np.int32)
ax=np.zeros(16,np.float32);av=np.zeros(16,np.float32);Z=np.zeros(16,np.float32)
if lib.bridge_init(700)!=0: print("bridge_init 실패");sys.exit(1)
A=[[] for _ in range(N)];Q=[[] for _ in range(N)]
lib.bridge_enable(ena)
try:
    t0=time.time()
    while time.time()-t0<T:
        lib.bridge_write_mit(p(Z),p(Z),p(Z),p(Z),p(Z),N)
        lib.bridge_read(p(b[0]),p(b[1]),p(b[2]),p(b[3]),p(b[4]),p(b[5]),p(b[6]),ip(conn),ip(stt))
        lib.bridge_aux(p(ax),p(av))
        for i in range(N): A[i].append(float(ax[i])); Q[i].append(float(b[0][i]))
        time.sleep(0.02)
finally:
    lib.bridge_enable(0)
    for _ in range(20): lib.bridge_write_mit(p(Z),p(Z),p(Z),p(Z),p(Z),N);time.sleep(0.005)
print("=== mode=0x%02X (%s) · N=%d ==="%(mode,"모터ON/PWM" if ena else "모터OFF/PWM없음",len(A[0])))
print("%-9s %9s %9s %9s"%("ch","aux_std","aux_ptp","q_ch_std"))
for i in range(N):
    a=np.array(A[i]);q=np.array(Q[i])
    print("%-9s %9.3f %9.3f %9.3f"%(NM[i],a.std(),a.max()-a.min(),q.std()))
