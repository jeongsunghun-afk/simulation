"""τ smoothness 비교 (pre vs post smoothness patches).

각 scenario 의 wbc_tau_cmd 시계열을 분석:
  · τ peak                — 최대값
  · τ jerk = d(τ)/dt peak — 시간 미분 최대
  · τ jerk RMS            — 전체적 부드러움
  · τ stdev               — 변동성

git stash + 두 버전 모두 실행하여 비교.
"""
import os, sys, json, subprocess
import numpy as np

SRC = '/home/jsh/simulation/gait_sim_v12.py'
SCENARIOS = [
    ('NMPC trot', {'use_nmpc': True,  'gait_type': 'trot'}),
    ('NMPC walk', {'use_nmpc': True,  'gait_type': 'walk'}),
    ('v11 trot',  {'use_nmpc': False, 'gait_type': 'trot'}),
    ('v11 walk',  {'use_nmpc': False, 'gait_type': 'walk'}),
]

WORKER = r'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt
_plt.show = lambda *a, **k: None
import sys, json, re
import numpy as np

SRC = sys.argv[1]
overrides = json.loads(sys.argv[2])
src = open(SRC).read()
for key, val in overrides.items():
    if isinstance(val, bool):
        for tf in (True, False):
            old = f'{key}: bool = {tf}'
            if old in src:
                src = src.replace(old, f'{key}: bool = {val}'); break
    elif isinstance(val, str):
        src = re.sub(rf"{key}: str = '[a-z]+'", f"{key}: str = '{val}'", src)

ns = {'__name__': '__main__'}
exec(compile(src, SRC, 'exec'), ns)
tau = ns['wbc_tau_cmd']
DT  = ns['DT']
# τ jerk = dτ/dt (numerical)
tau_jerk = np.gradient(tau, DT, axis=0)
# stats
result = {
    'tau_peak':     float(np.max(np.abs(tau))),
    'tau_std':      float(np.std(tau)),
    'tau_jerk_peak': float(np.max(np.abs(tau_jerk))),
    'tau_jerk_rms': float(np.sqrt(np.mean(tau_jerk**2))),
    'tau_jerk_p95': float(np.percentile(np.abs(tau_jerk), 95)),
}
print('__RESULT__' + json.dumps(result))
'''


def run_one(label, overrides):
    proc = subprocess.run(
        ['python3', '-c', WORKER, SRC, json.dumps(overrides)],
        capture_output=True, text=True, cwd='/home/jsh/simulation',
        env={**os.environ, 'MPLBACKEND': 'Agg'}, timeout=600,
    )
    for line in proc.stdout.splitlines():
        if line.startswith('__RESULT__'):
            r = json.loads(line[len('__RESULT__'):])
            r['label'] = label
            return r
    return {'label': label, 'error': proc.stderr[-200:] if proc.stderr else 'unknown'}


if __name__ == '__main__':
    results = []
    for label, ov in SCENARIOS:
        print(f"━━ Running {label} (τ smoothness analysis) ━━", flush=True)
        r = run_one(label, ov)
        results.append(r)
        if 'error' in r:
            print(f"  ERROR: {r['error']}", flush=True)
        else:
            print(f"  τ peak={r['tau_peak']:.1f}Nm  std={r['tau_std']:.2f}  "
                  f"jerk peak={r['tau_jerk_peak']:.0f}Nm/s  rms={r['tau_jerk_rms']:.1f}",
                  flush=True)

    print('\n━━ τ smoothness 비교 (현재 v12.py — patched) ━━', flush=True)
    print('| Scenario   | τ peak | τ std | jerk peak  | jerk rms | jerk p95 |')
    print('|------------|-------:|------:|-----------:|---------:|---------:|')
    for r in results:
        if 'error' in r: continue
        print(f"| {r['label']:10s} | {r['tau_peak']:6.1f} | {r['tau_std']:5.2f} | "
              f"{r['tau_jerk_peak']:10.0f} | {r['tau_jerk_rms']:8.1f} | {r['tau_jerk_p95']:8.1f} |")
