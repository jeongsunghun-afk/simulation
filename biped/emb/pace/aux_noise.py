import json,time,sys,numpy as np
STATE="/tmp/biped_state.json";N=8
NM=["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
T=float(sys.argv[1]) if len(sys.argv)>1 else 4.0
Q=[[] for _ in range(N)];A=[[] for _ in range(N)];nb=0
t0=time.time()
while time.time()-t0<T:
    try:
        d=json.load(open(STATE));q=d.get("q_ch_deg") or [];a=d.get("aux_deg") or []
        if d.get("aux_on"):
            for i in range(N):
                if i<len(q):Q[i].append(float(q[i]))
                if i<len(a):A[i].append(float(a[i]))
        else: nb+=1
    except: pass
    time.sleep(0.02)
if nb>5: print("⚠ aux_on=False 구간 — deploy AUX_MODE=1 확인")
print("%-9s %6s %9s %9s %10s %9s"%("ch","N","aux_std","aux_ptp","aux_step","q_ch_std"))
for i in range(N):
    a=np.array(A[i]);q=np.array(Q[i])
    if len(a)<10:print("%-9s 데이터부족(%d)"%(NM[i],len(a)));continue
    ua=np.unique(np.round(a,5));st=np.diff(np.sort(ua));step=st[st>1e-6].min() if (st>1e-6).any() else 0.0
    print("%-9s %6d %9.3f %9.3f %10.4f %9.3f"%(NM[i],len(a),a.std(),a.max()-a.min(),step,q.std()))
print("\naux_std=정지노이즈RMS° · aux_ptp=진폭° · aux_step=양자화(분해능)° · q_ch_std=1차노이즈°(비교)")
