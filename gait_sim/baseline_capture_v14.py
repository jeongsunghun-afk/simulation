"""v14 분할 baseline 회귀 검증 (subprocess 격리).

각 scenario 를 독립 subprocess 로 실행 — gait_sim.runner 사용.
v13.1 BASELINE_v13.md 와 동일 metric 비교 → 회귀 없음 검증.

Usage: python3 gait_sim/baseline_capture_v14.py
"""
import os, sys, subprocess, json

SRC_TAG = 'gait_sim.runner (v13.6+)'
OUT_MD = '/home/jsh/simulation/gait_sim/BASELINE_v14.md'
SCENARIOS = [
    ('NMPC trot',     {'use_nmpc': True,  'gait_type': 'trot'}),
    ('NMPC walk',     {'use_nmpc': True,  'gait_type': 'walk'}),
    ('MPC+WBIC trot', {'use_nmpc': False, 'gait_type': 'trot'}),
    ('MPC+WBIC walk', {'use_nmpc': False, 'gait_type': 'walk'}),
]

WORKER = r'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
_plt.show = lambda *a, **k: None
import sys, json
import numpy as np

overrides = json.loads(sys.argv[1])
from gait_sim import config
for k, v in overrides.items():
    setattr(config.CFG, k, v)

# Refresh derived aliases when gait_type changes
if "gait_type" in overrides:
    config.GAIT_TYPE = config.CFG.gait_type
    p = config.GAIT_PRESETS[config.GAIT_TYPE]
    config.V, config.T, config.D = p["V"], p["T"], p["D"]
    config.STEP_HEIGHT = p["STEP_HEIGHT"]
    config.T_SW = config.T * config.D
    config.T_ST = config.T * (1.0 - config.D)
    config.STRIDE_D = config.V * config.T + 2.0 * config.V * config.T_SW
    config.STEP_LENGTH = config.STRIDE_D / 2.0 - config.V * config.T_SW
    config.N_FRAMES = int(config.N_CYCLES * config.T / config.DT)

from gait_sim.runner import run_simulation

R, meta = run_simulation()

bp  = R.body_pos_hist
bv  = R.body_v_hist
bR  = R.body_R_hist
tau = R.wbc_tau_cmd
lam = R.wbc_lam_des
V_  = meta["V"]; T_ = meta["T"]; N_C = config.N_CYCLES

target_x = V_ * T_ * N_C
roll  = np.degrees(np.arctan2(bR[:, 2, 1], bR[:, 2, 2]))
pitch = np.degrees(np.arcsin(np.clip(-bR[:, 2, 0], -1, 1)))
diverged = (np.max(np.abs(roll)) > 60) or (np.max(np.abs(pitch)) > 60) or \
           (np.max(np.abs(bp[:, 2])) > 2.0)

result = {
    "V": V_, "T": T_, "N_CYCLES": N_C,
    "body_x_final":   float(bp[-1, 0]),
    "body_x_target":  float(target_x),
    "body_y_final":   float(bp[-1, 1]),
    "body_z_min":     float(bp[:, 2].min()),
    "body_z_max":     float(bp[:, 2].max()),
    "vx_mean":        float(bv[:, 0].mean()),
    "vx_target":      float(V_),
    "roll_max_deg":   float(np.max(np.abs(roll))),
    "pitch_max_deg":  float(np.max(np.abs(pitch))),
    "tau_peak":       float(np.max(np.abs(tau))),
    "Fz_peak_FR":     float(np.max(np.abs(lam[:, 0, 2]))),
    "Fz_peak_FL":     float(np.max(np.abs(lam[:, 1, 2]))),
    "Fz_peak_HR":     float(np.max(np.abs(lam[:, 2, 2]))),
    "Fz_peak_HL":     float(np.max(np.abs(lam[:, 3, 2]))),
    "diverged":       bool(diverged),
    "mode":           meta.get("mode", ""),
}
print("__RESULT__" + json.dumps(result))
'''


def run_one(label, overrides):
    proc = subprocess.run(
        ['python3', '-c', WORKER, json.dumps(overrides)],
        capture_output=True, text=True, cwd='/home/jsh/simulation',
        env={**os.environ, 'MPLBACKEND': 'Agg'}, timeout=900,
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
    lines = [
        f'# Baseline Metrics (v14 = gait_sim.runner 기반, src: {SRC_TAG})\n',
        '각 scenario subprocess 격리 실행. v13.1 BASELINE_v13.md 와 비교 → 회귀 없음 검증.\n'
    ]
    lines.append('| Scenario | use_nmpc | x_final/target | y_final | z_range | '
                  'vx mean/tgt | roll/pitch max | τ peak | Fz peak max | diverged |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|')
    for r in results:
        if 'error' in r:
            lines.append(f"| **{r['label']}** | ERROR | {r['error']} |")
            continue
        fz_max = max(r['Fz_peak_FR'], r['Fz_peak_FL'], r['Fz_peak_HR'], r['Fz_peak_HL'])
        nmpc_str = 'True' if 'NMPC' in r['label'] else 'False'
        lines.append(
            f"| {r['label']} "
            f"| {nmpc_str} "
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
            print(f"  OK: x={r['body_x_final']:.3f}, vx={r['vx_mean']:.3f}, "
                  f"τ={r['tau_peak']:.1f}, diverged={r['diverged']}", flush=True)
    md = fmt_md(results)
    print('\n' + md, flush=True)
    open(OUT_MD, 'w').write(md + '\n')
    print(f'\nSaved to {OUT_MD}', flush=True)
