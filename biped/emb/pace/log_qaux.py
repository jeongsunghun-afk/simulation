#!/usr/bin/env python3
"""log_qaux.py — 무중력 손스윕 동안 q_ch·aux 로깅 → 관절별 offset(aux@q_ch=0)·scale·offset보정 추종 분석.
   read-only(state JSON 만 읽음, 아무것도 안 씀). deploy 가 AUX_MODE=1(0x5A)로 떠 있어야 aux 옴."""
import json, time, sys, numpy as np
STATE = "/tmp/biped_state.json"; N = 8
NM = ["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
T = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
Q = [[] for _ in range(N)]; A = [[] for _ in range(N)]
print("→ %g초 동안 **무중력에서 각 관절을 손으로 천천히 훑으세요**(특히 q_ch=0 을 지나도록). 로깅 중…" % T, flush=True)
t0 = time.time(); noaux = 0
while time.time() - t0 < T:
    try:
        d = json.load(open(STATE)); q = d.get("q_ch_deg") or []; a = d.get("aux_deg") or []
        if d.get("aux_on"):
            for i in range(N):
                if i < len(q) and i < len(a): Q[i].append(float(q[i])); A[i].append(float(a[i]))
        else:
            noaux += 1
    except Exception:
        pass
    time.sleep(0.035)
if noaux > 10:
    print("⚠ aux_on=False 구간 많음 — deploy 를 AUX_MODE=1 로 띄웠는지 확인");
print("\n%-9s %6s %7s %8s %9s %8s %8s" % ("ch","N","범위°","scale","offset°","잔차RMS°","최대|Δ|°"))
print("  (scale=aux증분/q_ch증분·1.0=완벽추종 | offset=aux@q_ch=0(2차영점) | 잔차=편심 | 최대|Δ|=offset보정후 aux−q_ch)")
for i in range(N):
    q = np.array(Q[i]); a = np.array(A[i])
    rng = (q.max() - q.min()) if len(q) else 0.0
    if len(q) < 25 or rng < 4.0:
        print("%-9s %6d  범위부족(%.1f°) — 더 크게 움직이세요" % (NM[i], len(q), rng)); continue
    M = np.vstack([q, np.ones_like(q)]).T
    (scale, offset), *_ = np.linalg.lstsq(M, a, rcond=None)
    resid = a - (scale * q + offset)
    dev = (a - offset) - q          # offset보정 aux 와 q_ch 의 차 = (scale-1)*q + 편심
    print("%-9s %6d %7.1f %8.3f %9.2f %8.3f %8.2f" % (NM[i], len(q), rng, scale, offset, resid.std(), np.abs(dev).max()))
