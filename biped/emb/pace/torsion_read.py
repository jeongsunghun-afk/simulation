#!/usr/bin/env python3
"""torsion_read.py — deploy hold 중 READ-ONLY. state JSON 의 q_ch·aux·tau 를 N초 median →
   채널별 q_ch·aux·(aux−q_ch)=감속단 비틀림·tau. 단일 writer 무관(state 읽기만)."""
import json, time, numpy as np, sys, os
ST=os.environ.get("QUAD_STATE","/dev/shm/biped_state.json")
DUR=float(sys.argv[1]) if len(sys.argv)>1 else 4.0
NM=["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
Q=[[] for _ in range(8)]; A=[[] for _ in range(8)]; T=[[] for _ in range(8)]
t0=time.time()
while time.time()-t0<DUR:
    try:
        d=json.load(open(ST)); q=d.get("q_ch_deg") or []; a=d.get("aux_deg") or []; ta=d.get("tau_leg_nm") or []
        for i in range(8):
            if i<len(q): Q[i].append(q[i])
            if i<len(a) and a[i]!=0.0: A[i].append(a[i])
            if i<len(ta): T[i].append(ta[i])
    except Exception: pass
    time.sleep(0.02)
d=json.load(open(ST))
print("mode=%s loop_hz=%s" % (d.get("mode"), d.get("loop_hz")))
def med(v): return float(np.median(v)) if v else float("nan")
print("%-9s %9s %9s %10s %9s" % ("ch","q_ch°","aux°","aux−q_ch°","tau_Nm"))
print("-"*52)
mx=0.0
for i in range(8):
    qm=med(Q[i]); am=med(A[i]); dm=am-qm; tm=med(T[i])
    print("%-9s %9.3f %9.3f %10.3f %9.2f" % (NM[i],qm,am,dm,tm))
    if np.isfinite(dm): mx=max(mx,abs(dm))
print("-"*52)
print("→ 최대 |aux−q_ch| = %.3f°  (hold 부하 하 감속단 비틀림)" % mx)
print("  (q_ch≈0 이면 영점자세 · aux−q_ch = 그 유지토크 하의 출력측 지연=비틀림)")
