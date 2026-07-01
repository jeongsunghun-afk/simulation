"""plot_log — quad 로그 CSV(--log) 시각화 + 최대 토크/각속도 체크.

각 축(관절) 토크[Nm]·각속도[rad/s]를 시간축으로 플롯하고, 축별/전체 최대값을
02_Leg 모터 한계(Peak토크·최대각속도) 대비 %와 함께 출력. PNG도 CSV 옆에 저장.

실행: python plot_log.py [logs/quad_log.csv]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg' if os.environ.get('DISPLAY') else 'Agg')
import matplotlib.pyplot as plt

# 02_Leg 관절 모터 한계 (suffix → Peak토크[Nm], 최대각속도[rad/s]). [[02leg-motor-spec]]
LIMITS = {'hip': (84, 29.6), 'thigh': (84, 29.6), 'calf': (126, 19.7), 'foot': (168, 14.8)}

path = sys.argv[1] if len(sys.argv) > 1 else 'logs/quad_log.csv'
d = np.genfromtxt(path, delimiter=',', names=True)
names = d.dtype.names
t = d['t']
tau_cols = [c for c in names if c.startswith('tau')]   # genfromtxt가 ':' 제거 → tauHL_hip
wj_cols = [c for c in names if c.startswith('wj')]
joints = [c[3:] for c in tau_cols]                      # HL_hip ...
is_02leg = any(j.endswith('foot') for j in joints)      # foot 관절 있으면 02_Leg

# ── 최대값 요약 (표) ──
print('\n로그: %s   (%d행, %.2f초, %d축)' % (path, len(t), t[-1] - t[0], len(joints)))
print('─' * 64)
print('%-10s %12s %16s' % ('축(관절)', '|τ|max[Nm]', '|ω|max[rad/s]')
      + ('   (한계% )' if is_02leg else ''))
print('─' * 64)
tau_max_all = (0.0, ''); w_max_all = (0.0, '')
for tc, wc, j in zip(tau_cols, wj_cols, joints):
    tm = float(np.abs(d[tc]).max()); wm = float(np.abs(d[wc]).max())
    suf = j.split('_')[-1]
    if tm > tau_max_all[0]:
        tau_max_all = (tm, j)
    if wm > w_max_all[0]:
        w_max_all = (wm, j)
    if is_02leg and suf in LIMITS:
        tl, wl = LIMITS[suf]
        print('%-10s %12.1f %16.2f   (τ%3.0f%% ω%3.0f%%)' %
              (j, tm, wm, 100 * tm / tl, 100 * wm / wl))
    else:
        print('%-10s %12.1f %16.2f' % (j, tm, wm))
print('─' * 64)
print('▶ 전체 최대 토크 : %.1f Nm   (%s)' % (tau_max_all[0], tau_max_all[1]))
print('▶ 전체 최대 각속도: %.2f rad/s (%s)' % (w_max_all[0], w_max_all[1]))

# ── 플롯 (위=토크, 아래=각속도) ──
fig, (ax_t, ax_w) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
cmap = plt.cm.tab20(np.linspace(0, 1, max(2, len(joints))))
for i, (tc, wc, j) in enumerate(zip(tau_cols, wj_cols, joints)):
    ax_t.plot(t, d[tc], lw=0.8, color=cmap[i], label=j)
    ax_w.plot(t, d[wc], lw=0.8, color=cmap[i])
# 모터 한계 기준선
if is_02leg:
    for val, c, lab in [(84, 'gray', 'hip/thigh 84'), (126, 'orange', 'calf 126'), (168, 'red', 'foot 168')]:
        ax_t.axhline(val, ls='--', lw=0.7, color=c, alpha=0.6); ax_t.axhline(-val, ls='--', lw=0.7, color=c, alpha=0.6)
    for val, c in [(29.6, 'gray'), (19.7, 'orange'), (14.8, 'red')]:
        ax_w.axhline(val, ls='--', lw=0.7, color=c, alpha=0.6); ax_w.axhline(-val, ls='--', lw=0.7, color=c, alpha=0.6)
ax_t.set_ylabel('torque [Nm]'); ax_t.set_title('Torque per joint (dashed = 02_Leg motor Peak limit)' if is_02leg else 'Torque per joint')
ax_w.set_ylabel('joint vel [rad/s]'); ax_w.set_xlabel('t [s]')
ax_w.set_title('Joint velocity (dashed = max velocity limit)' if is_02leg else 'Joint velocity')
ax_t.grid(alpha=0.3); ax_w.grid(alpha=0.3)
ax_t.legend(fontsize=6, ncol=4, loc='upper right')
fig.suptitle('max τ=%.1f Nm (%s)   max ω=%.2f rad/s (%s)' %
             (tau_max_all[0], tau_max_all[1], w_max_all[0], w_max_all[1]))
fig.tight_layout()

png = os.path.splitext(path)[0] + '.png'
fig.savefig(png, dpi=110)
print('▶ 그래프 저장: %s' % png)
if os.environ.get('DISPLAY'):
    plt.show()
