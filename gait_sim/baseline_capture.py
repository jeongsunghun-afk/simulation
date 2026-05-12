"""v13 분할 전 baseline metric 캡처 (subprocess 격리 버전).

각 scenario 를 독립 subprocess 로 실행 (matplotlib/pinocchio 상태 분리).
Phase 1+ 후 재실행 → metric 비교 → regression 감지.

Usage: python3 gait_sim/baseline_capture.py
"""
import sys, os, subprocess, json, re

SRC = '/home/jsh/simulation/gait_sim_v12.py'
OUT_MD = '/home/jsh/simulation/gait_sim/BASELINE.md'
SCENARIOS = [
    ('NMPC trot',  {'use_nmpc': True,  'gait_type': 'trot'}),
    ('NMPC walk',  {'use_nmpc': True,  'gait_type': 'walk'}),
    ('v11 trot',   {'use_nmpc': False, 'gait_type': 'trot'}),
    ('v11 walk',   {'use_nmpc': False, 'gait_type': 'walk'}),
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

bp = ns['body_pos_hist']; bv = ns['body_v_hist']; bR = ns['body_R_hist']
tau = ns['wbc_tau_cmd']; lam = ns['wbc_lam_des']
V_ = ns['V']; T_ = ns['T']; N_C = ns['N_CYCLES']
target_x = V_ * T_ * N_C
roll = np.degrees(np.arctan2(bR[:,2,1], bR[:,2,2]))
pitch = np.degrees(np.arcsin(np.clip(-bR[:,2,0], -1, 1)))
diverged = (np.max(np.abs(roll)) > 60) or (np.max(np.abs(pitch)) > 60) or \
           (np.max(np.abs(bp[:, 2])) > 2.0)

result = {
    'V': V_, 'T': T_, 'N_CYCLES': N_C,
    'body_x_final':   float(bp[-1, 0]),
    'body_x_target':  float(target_x),
    'body_y_final':   float(bp[-1, 1]),
    'body_z_min':     float(bp[:, 2].min()),
    'body_z_max':     float(bp[:, 2].max()),
    'body_z_mean':    float(bp[:, 2].mean()),
    'vx_mean':        float(bv[:, 0].mean()),
    'vx_target':      float(V_),
    'roll_max_deg':   float(np.max(np.abs(roll))),
    'pitch_max_deg':  float(np.max(np.abs(pitch))),
    'tau_peak':       float(np.max(np.abs(tau))),
    'Fz_peak_FR':     float(np.max(np.abs(lam[:, 0, 2]))),
    'Fz_peak_FL':     float(np.max(np.abs(lam[:, 1, 2]))),
    'Fz_peak_HR':     float(np.max(np.abs(lam[:, 2, 2]))),
    'Fz_peak_HL':     float(np.max(np.abs(lam[:, 3, 2]))),
    'diverged':       bool(diverged),
    'mode':           f"_USE_NMPC_ACTIVE={ns['_USE_NMPC_ACTIVE']}",
}
print('__RESULT__' + json.dumps(result))
'''


def run_one(label, overrides):
    proc = subprocess.run(
        ['python3', '-c', WORKER, SRC, json.dumps(overrides)],
        capture_output=True, text=True, cwd='/home/jsh/simulation',
        env={**os.environ, 'MPLBACKEND': 'Agg'}, timeout=240,
    )
    if proc.returncode != 0:
        return {'label': label, 'error': proc.stderr.splitlines()[-3:] if proc.stderr else 'unknown'}
    for line in proc.stdout.splitlines():
        if line.startswith('__RESULT__'):
            r = json.loads(line[len('__RESULT__'):])
            r['label'] = label
            return r
    return {'label': label, 'error': 'no __RESULT__ marker'}


def fmt_md(results):
    lines = ['# Baseline Metrics (v12.8, pre v13 분할)\n',
             '각 scenario subprocess 격리 실행. Phase 1+ 후 재실행 → 동일값 = no regression.\n']
    lines.append('| Scenario | mode | x_final/target | y_final | z_range | vx mean/tgt | roll/pitch max | τ peak | Fz peak max | diverged |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|')
    for r in results:
        if 'error' in r:
            lines.append(f"| **{r['label']}** | ERROR | {r['error']} |")
            continue
        fz_max = max(r['Fz_peak_FR'], r['Fz_peak_FL'], r['Fz_peak_HR'], r['Fz_peak_HL'])
        lines.append(
            f"| {r['label']} "
            f"| {r['mode']} "
            f"| {r['body_x_final']:.3f} / {r['body_x_target']:.3f} "
            f"| {r['body_y_final']*1000:+.0f} mm "
            f"| {r['body_z_min']*1000:.0f}~{r['body_z_max']*1000:.0f} mm "
            f"| {r['vx_mean']:.3f} / {r['vx_target']:.2f} "
            f"| {r['roll_max_deg']:.2f}° / {r['pitch_max_deg']:.2f}° "
            f"| {r['tau_peak']:.1f} Nm "
            f"| {fz_max:.0f} N "
            f"| {'❌' if r['diverged'] else '✅'} |"
        )
    return '\n'.join(lines)


if __name__ == '__main__':
    results = []
    for label, ov in SCENARIOS:
        print(f"━━ Running {label} ━━", flush=True)
        r = run_one(label, ov)
        results.append(r)
        if 'error' in r:
            print(f"  ERROR: {r['error']}", flush=True)
        else:
            print(f"  OK: mode={r['mode']}, x={r['body_x_final']:.3f}, "
                  f"vx={r['vx_mean']:.3f}, τ={r['tau_peak']:.1f}, diverged={r['diverged']}",
                  flush=True)
    md = fmt_md(results)
    print('\n' + md, flush=True)
    open(OUT_MD, 'w').write(md + '\n')
    print(f'\nSaved to {OUT_MD}', flush=True)
