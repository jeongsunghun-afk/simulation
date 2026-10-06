import numpy as np, pandas as pd, sys, glob, re
from scipy.signal import butter, sosfiltfilt
def band(x,lo,hi): return np.sqrt(np.mean(sosfiltfilt(butter(4,[lo,hi],'bandpass',fs=500,output='sos'),x)**2))
HW=dict(hip=(11.7,3.85,4.5),thigh=(23,13,12.2),calf=(83,24.3,17.4),foot=(69,21.6,12.8),groll=(5.35,3.79),gpitch=(1.87,2.10))
def metr(n):
    D=pd.read_csv(n+'.csv'); fl=D.fall.values; t=D.t.values
    f1=t[np.argmax(fl>0)] if fl.max()>0 else None
    W=D[(D.t>3)&(D.t<(f1-0.3 if f1 else 99))]
    o=dict(falls=int(fl.max()))
    if len(W)<1500: return o
    # 실기 트레이스 dq 는 **채널** 좌표: calf_ch = 1.5·calf · foot_ch = 1.2·(calf+foot) — 같은 좌표로 바꿔 비교
    ch={}
    for b in (0,4):
        ch[b]=W[f'dqm{b}'].values; ch[b+1]=W[f'dqm{b+1}'].values
        ch[b+2]=1.5*W[f'dqm{b+2}'].values; ch[b+3]=1.2*(W[f'dqm{b+2}'].values+W[f'dqm{b+3}'].values)
    for g,idx in (('hip',[0,4]),('thigh',[1,5]),('calf',[2,6]),('foot',[3,7])):
        o[g]=tuple(np.mean([[band(ch[j],a,b) for a,b in ((1,5),(12,17),(17,22))] for j in idx],0))
        o[g+'_lr']=np.mean([band(W[f'dql{j}'].values,12,22)/max(band(W[f'dqm{j}'].values,12,22),1e-6) for j in idx])
    o['groll']=(band(W.gx.values,12,17),band(W.gx.values,17,22)); o['gpitch']=(band(W.gy.values,12,17),band(W.gy.values,17,22))
    return o
pref=sys.argv[1]
names=sorted(set(re.sub(r'_s\d+$','',p[:-4]) for p in glob.glob(pref+'*.csv')))
print('설정              낙상 | hip 12-17/17-22 | thigh 12-17/17-22 | calf 12-17/17-22 | foot 12-17/17-22 | 몸통 roll·pitch 12-17 | hip 링크/모터 | 오차점수')
print(f'{"실기 ⑲":17s}  -   | {HW["hip"][1]:4.1f} {HW["hip"][2]:4.1f}       | {HW["thigh"][1]:4.1f} {HW["thigh"][2]:4.1f}         | {HW["calf"][1]:4.1f} {HW["calf"][2]:4.1f}        | {HW["foot"][1]:4.1f} {HW["foot"][2]:4.1f}        | {HW["groll"][0]:4.1f} {HW["gpitch"][0]:4.1f}             | 1.45~2.1')
for nm in names:
    R=[metr(p[:-4]) for p in sorted(glob.glob(nm+'_s*.csv'))]
    fs=sum(r['falls'] for r in R); R=[r for r in R if 'hip' in r]
    if not R: print(f'{nm:17s} {fs:3d}  | (분석 불가)'); continue
    g=lambda k,i: np.mean([r[k][i] for r in R]); lr=np.mean([r['hip_lr'] for r in R])
    sc=0
    for k in ('hip','thigh','calf','foot'):
        for i in (1,2): sc+=abs(np.log(g(k,i)/HW[k][i]))
    sc+=abs(np.log(g('groll',0)/HW['groll'][0]))+abs(np.log(g('gpitch',0)/HW['gpitch'][0]))
    print(f'{nm:17s} {fs:3d}  | {g("hip",1):4.1f} {g("hip",2):4.1f}       | {g("thigh",1):4.1f} {g("thigh",2):4.1f}         | {g("calf",1):4.1f} {g("calf",2):4.1f}        | {g("foot",1):4.1f} {g("foot",2):4.1f}        | {g("groll",0):4.1f} {g("gpitch",0):4.1f}             | {lr:4.2f}          | {sc:5.2f}')
