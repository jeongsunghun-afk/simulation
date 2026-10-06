import numpy as np, pandas as pd, sys
from scipy.signal import butter, sosfiltfilt, welch
N=['HL_hip','HL_thigh','HL_calf','HL_foot','HR_hip','HR_thigh','HR_calf','HR_foot']
def band(x,lo,hi): return np.sqrt(np.mean(sosfiltfilt(butter(4,[lo,hi],'bandpass',fs=500,output='sos'),x)**2))
for n in sys.argv[1:]:
    D=pd.read_csv(n+'.csv'); fl=D.fall.values; t=D.t.values
    f1=t[np.argmax(fl>0)] if fl.max()>0 else None
    W=D[(D.t>3)&(D.t<(f1-0.3 if f1 else 99))]
    print(f'== {n}: 첫 낙상 {f1} · 분석 {len(W)/500:.1f}s · roll/pitch std {W.roll.std():.2f}/{W.pitch.std():.2f}')
    if len(W)<1000: continue
    for g,idx in (('hip',[0,4]),('thigh',[1,5]),('calf',[2,6]),('foot',[3,7])):
        m=[ [band(W[f'dqm{j}'].values,a,b) for a,b in ((1,5),(12,17),(17,22),(22,35))] for j in idx]
        l=[ [band(W[f'dql{j}'].values,a,b) for a,b in ((1,5),(12,17),(17,22),(22,35))] for j in idx]
        m=np.mean(m,0); l=np.mean(l,0)
        f,P=welch(W[f'dqm{idx[0]}'].values,fs=500,nperseg=1024); pk=f[(f>8)][np.argmax(P[f>8])]
        print(f'   {g:5s} 모터측 dq 1-5/12-17/17-22/22-35: {m[0]:6.1f} {m[1]:5.1f} {m[2]:5.1f} {m[3]:5.1f} · 링크/모터 {l[1]/m[1]:.2f} {l[2]/m[2]:.2f} · 8Hz↑ 봉우리 {pk:.1f}Hz')
    g=[band(W[c].values,12,17) for c in ('gx','gy')]; print(f'   몸통 각속도 12-17Hz roll {g[0]:.2f} pitch {g[1]:.2f} °/s')
