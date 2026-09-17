#!/usr/bin/env python3
"""biped_tremble_check.py — ExpLog CSV 로 '부르르 떨림' 진단(스틱슬립 vs PD링잉 vs 명령).
   관절별: 추종오차 · 고주파 진동폭 · 속도반전율(스틱슬립) · 떨림 주파수 · 전류변동.
   MuJoCo 불필요(순수 신호분석).  사용: python3 biped_tremble_check.py exp_logs/squat_*.csv
"""
import csv, sys, os, numpy as np
if len(sys.argv) < 2: print(__doc__); sys.exit(1)
NM=['HL_hip','HL_thigh','HL_calf','HL_foot','HR_hip','HR_thigh','HR_calf','HR_foot']; N=8
SIGN=np.array([-1,1,-1,-1,-1,-1,1,1],float); GEAR=np.array([7,7,10.5,8.4,7,7,10.5,8.4],float); KT=0.2; SCALE=6.4
rows=list(csv.DictReader(open(sys.argv[1])))
if len(rows)<40: print("표본 부족(%d)"%len(rows)); sys.exit(1)
def col(k):
    try: return np.array([[float(r['%s_%s'%(k,n)]) for n in NM] for r in rows],float)
    except KeyError: return None
t=np.array([float(r['t']) for r in rows]); dt=float(np.median(np.diff(t))); fs=1.0/dt
q=col('q'); qcmd=col('qcmd'); dq=col('dq'); cur=col('cur')
mode=rows[len(rows)//2].get('mode','?')
def smooth(x,k=7): return np.convolve(x,np.ones(k)/k,mode='same')
def domfreq(x):
    x=x-x.mean(); n=len(x)
    if n<16 or x.std()<1e-9: return 0.0
    P=np.abs(np.fft.rfft(x*np.hanning(n))); f=np.fft.rfftfreq(n,dt)
    P[f<0.8]=0                                   # 0.8Hz 미만(주모션)은 무시 → 떨림대역만
    return float(f[np.argmax(P)])
print("■ 떨림 진단 — %s  (fs=%.0fHz · %d표본 · mode=%s · SCALE=%.1f)"%(os.path.basename(sys.argv[1]),fs,len(rows),mode,SCALE))
print("%-9s %8s %8s %8s %9s | %8s  %s"%("관절","추종RMS°","진동폭°","떨림Hz","반전/s","전류σA","판정"))
print("-"*82)
worst=(-1,0)
for i in range(N):
    e = (q[:,i]-qcmd[:,i]) if (q is not None and qcmd is not None) else np.zeros(len(rows))
    hf = q[:,i]-smooth(q[:,i]) if q is not None else np.zeros(len(rows))   # 고주파 진동분
    hfa=float(hf.std())
    dqi=dq[:,i] if dq is not None else np.zeros(len(rows))
    rev = int(np.sum(np.diff(np.sign(dqi))!=0))/ (t[-1]-t[0])              # 속도반전율[/s]
    fpk = domfreq(dqi)
    cs = float(cur[:,i].std()) if cur is not None else 0.0
    # 판정
    tag=[]
    if hfa>0.15 and rev>3:  tag.append("스틱슬립?")     # 고주파+잦은반전
    if fpk>3 and hfa>0.1:   tag.append("PD링잉/한계주기 %0.1fHz"%fpk)
    if hfa<=0.1:            tag.append("조용")
    if hfa>worst[1]: worst=(i,hfa)
    print("%-9s %8.2f %8.3f %8.1f %9.1f | %8.3f  %s"
          %(NM[i],float(np.sqrt(np.mean(e**2))),hfa,fpk,rev,cs," · ".join(tag)))
print("-"*82)
print("→ 진동폭° 가장 큰 축 = **%s**. 반전/s 높음+진동폭 큼 = **스틱슬립(마찰)**,"%NM[worst[0]])
print("  떨림Hz 높음(>3) = **PD 링잉/저감쇠**. 명령(qcmd)에도 같은 진동 있으면 = 명령쪽.")
print("  calf가 최대면 재조립 뻑뻑(마찰)·jog 마찰보상無 유력. 전류σ 큰 축이 힘을 많이 쓰는 축.")
