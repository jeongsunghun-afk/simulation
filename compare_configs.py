"""
gait_sim_v10 — 다리 간격 비교 (Q_HOME_HIND 원본 vs +50mm 확장).

사용:
    python3 compare_configs.py                # 두 변형 모두 실행 + 비교
    python3 compare_configs.py --skip-run     # 기존 pickle만 로드해서 비교 (재실행 X)

출력:
    1. 콘솔: 항목별 비교 표
    2. /tmp/v10_compare.png: overlay plot (τ_cmd, dq, λ, body pitch, fz_sum 등)

내부:
    - subprocess로 v10을 두 번 실행 (HIND_VARIANT=orig | ext, COMPARE_MODE=1)
    - 각 실행이 /tmp/v10_metrics_{variant}.pkl 덤프
    - wrapper가 pickle 로드 후 비교 → 표 + figure
"""
import os
import sys
import pickle
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V10_PATH = Path(__file__).parent / 'gait_sim_v10.py'
PKL_DIR = Path('/tmp')
VARIANTS = ['orig', 'ext']
LEG_NAMES = ['FR', 'FL', 'HR', 'HL']
LEG_COLORS = ['#00d4ff', '#ff6b35', '#00ff99', '#c264ff']


def run_variant(variant: str, timeout_s: int = 600) -> None:
    """v10을 subprocess로 실행 (COMPARE_MODE=1, HIND_VARIANT=variant)."""
    env = os.environ.copy()
    env['HIND_VARIANT'] = variant
    env['COMPARE_MODE'] = '1'
    env['MPLBACKEND']   = 'Agg'   # figure 코드까지 가더라도 비-인터랙티브
    print(f"\n━━━ Running v10 [HIND_VARIANT={variant}] ━━━")
    proc = subprocess.run(
        [sys.executable, str(V10_PATH)],
        env=env, timeout=timeout_s,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout[-2000:])
        print("STDERR:", proc.stderr[-2000:])
        raise RuntimeError(f"v10 [{variant}] failed (exit={proc.returncode})")
    # 마지막 진단 라인만 표시
    for line in proc.stdout.splitlines()[-25:]:
        print("  " + line)


def load_metrics(variant: str) -> dict:
    p = PKL_DIR / f'v10_metrics_{variant}.pkl'
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — variant={variant}")
    with open(p, 'rb') as f:
        return pickle.load(f)


def fmt_pct(a, b):
    if abs(b) < 1e-12:
        return '   N/A'
    return f'{(a-b)/abs(b)*100:+6.1f}%'


def print_table(m_orig: dict, m_ext: dict) -> None:
    print()
    print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
    print(f'  비교: HIND 원본 (간격 401mm)  vs  HIND 확장 (간격 451mm)')
    print(f'  Gait={m_orig["gait"]}  V={m_orig["V"]}m/s  N_FRAMES={m_orig["N_FRAMES"]}')
    print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')

    Mg = m_orig['TOTAL_MASS'] * m_orig['G_ACC']

    rows = []
    for leg in range(4):
        nj = 5
        for j in range(nj):
            tc_o = float(np.max(np.abs(m_orig['wbc_tau_cmd'][:, leg, j] - m_orig['wbc_tau_grf'][:, leg, j])))
            tc_e = float(np.max(np.abs(m_ext ['wbc_tau_cmd'][:, leg, j] - m_ext ['wbc_tau_grf'][:, leg, j])))
            if tc_o > 0.5 or tc_e > 0.5:    # 자잘한 항목 숨김
                rows.append((f'{LEG_NAMES[leg]} τ_cmd−τ_grf peak th{j+1} [Nm]', tc_o, tc_e))
    for leg in range(4):
        for j in range(5):
            dq_o = float(np.max(np.abs(m_orig['joint_vel_hist'][:, leg, j])))
            dq_e = float(np.max(np.abs(m_ext ['joint_vel_hist'][:, leg, j])))
            if dq_o > 0.5 or dq_e > 0.5:
                rows.append((f'{LEG_NAMES[leg]} dq peak th{j+1} [rad/s]', dq_o, dq_e))
    for leg in range(4):
        for axis, lbl in enumerate(('Fx','Fy','Fz')):
            f_o = float(np.max(np.abs(m_orig['wbc_lam_des'][:, leg, axis])))
            f_e = float(np.max(np.abs(m_ext ['wbc_lam_des'][:, leg, axis])))
            rows.append((f'{LEG_NAMES[leg]} λ_des peak {lbl} [N]', f_o, f_e))
    for leg in range(4):
        fz_o = float(np.mean(m_orig['wbc_lam_des'][:, leg, 2]))
        fz_e = float(np.mean(m_ext ['wbc_lam_des'][:, leg, 2]))
        rows.append((f'{LEG_NAMES[leg]} λ_z mean [N]', fz_o, fz_e))

    res_o = float(np.mean(m_orig['wbic_residual_hist']))
    res_e = float(np.mean(m_ext ['wbic_residual_hist']))
    rows.append(('WBIC residual mean [-]', res_o, res_e))

    fz_sum_err_o = float(np.mean(np.abs(m_orig['fz_sum_des'] - Mg)))
    fz_sum_err_e = float(np.mean(np.abs(m_ext ['fz_sum_des'] - Mg)))
    rows.append(('|Σλz_des − Mg| mean [N]', fz_sum_err_o, fz_sum_err_e))

    energy_o = float(np.sum(np.abs(m_orig['wbc_tau_cmd'] * m_orig['joint_vel_hist'])) * m_orig['DT'])
    energy_e = float(np.sum(np.abs(m_ext ['wbc_tau_cmd'] * m_ext ['joint_vel_hist'])) * m_ext ['DT'])
    rows.append(('Σ|τ·q̇|·dt total [J·s? proxy]', energy_o, energy_e))

    print(f'{"항목":<46}{"orig":>11}{"ext":>11}  Δ%')
    print('-'*82)
    last_group = ''
    for label, vo, ve in rows:
        # 그룹 (FR/FL/HR/HL) 사이에 줄 추가
        leg_pfx = label.split()[0]
        if leg_pfx in LEG_NAMES and leg_pfx != last_group and last_group != '':
            print()
        last_group = leg_pfx if leg_pfx in LEG_NAMES else last_group
        print(f'{label:<46}{vo:>11.3f}{ve:>11.3f}  {fmt_pct(ve, vo)}')
    print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')


def make_overlay_plot(m_orig: dict, m_ext: dict, out_path: str = '/tmp/v10_compare.png') -> None:
    DT = m_orig['DT']
    N = m_orig['N_FRAMES']
    t = np.arange(N) * DT

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        f'Q_HOME_HIND compare - {m_orig["gait"]}, V={m_orig["V"]}m/s   '
        f'(orig=red, ext=cyan)',
        color='white', fontsize=11)

    # 1. HR/HL τ_cmd−τ_grf peak (가장 중요)
    for row, leg in enumerate([2, 3]):  # HR, HL
        ax = axes[row, 0]
        for j, c_o, c_e in [(1, '#ff5555', '#00d4ff'),
                            (2, '#ff7777', '#44ddff'),
                            (3, '#ff9999', '#88e6ff')]:
            tau_o = m_orig['wbc_tau_cmd'][:, leg, j] - m_orig['wbc_tau_grf'][:, leg, j]
            tau_e = m_ext ['wbc_tau_cmd'][:, leg, j] - m_ext ['wbc_tau_grf'][:, leg, j]
            ax.plot(t, tau_o, '-', color=c_o, lw=0.8, alpha=0.7, label=f'orig th{j+1}')
            ax.plot(t, tau_e, '-', color=c_e, lw=0.8, alpha=0.7, label=f'ext  th{j+1}')
        ax.set_title(f'{LEG_NAMES[leg]} τ_cmd − τ_grf [Nm]', color='white', fontsize=9)
        ax.set_xlabel('t [s]', color='white'); ax.set_ylabel('Nm', color='white')
        ax.legend(fontsize=6, ncol=2, loc='best'); ax.grid(alpha=0.3)
        ax.set_facecolor('#16213e'); ax.tick_params(colors='gray')

    # 2. HR/HL dq peak
    for row, leg in enumerate([2, 3]):
        ax = axes[row, 1]
        for j, c_o, c_e in [(1, '#ff5555', '#00d4ff'),
                            (2, '#ff7777', '#44ddff'),
                            (3, '#ff9999', '#88e6ff')]:
            ax.plot(t, m_orig['joint_vel_hist'][:, leg, j], '-', color=c_o, lw=0.8, alpha=0.7,
                    label=f'orig th{j+1}')
            ax.plot(t, m_ext ['joint_vel_hist'][:, leg, j], '-', color=c_e, lw=0.8, alpha=0.7,
                    label=f'ext  th{j+1}')
        ax.set_title(f'{LEG_NAMES[leg]} dq [rad/s]', color='white', fontsize=9)
        ax.set_xlabel('t [s]', color='white'); ax.set_ylabel('rad/s', color='white')
        ax.legend(fontsize=6, ncol=2, loc='best'); ax.grid(alpha=0.3)
        ax.set_facecolor('#16213e'); ax.tick_params(colors='gray')

    # 3. HR/HL λ_z (Fz)
    for col, leg in enumerate([2, 3]):
        ax = axes[2, col] if col == 0 else axes[2, col]
    # Reuse rows 2,3 for λz — overwrite
    for row, leg in enumerate([2, 3]):
        ax = axes[2 + (1 if row else 0) , 0] if False else None  # not used
    ax_lz_HR = axes[2, 0]; ax_lz_HL = axes[2, 1]
    for ax, leg, name in [(ax_lz_HR, 2, 'HR'), (ax_lz_HL, 3, 'HL')]:
        ax.cla()
        ax.plot(t, m_orig['wbc_lam_des'][:, leg, 2], '-', color='#ff5555', lw=0.9, label='orig λ_z')
        ax.plot(t, m_ext ['wbc_lam_des'][:, leg, 2], '-', color='#00d4ff', lw=0.9, label='ext  λ_z')
        ax.set_title(f'{name} λ_z (Fz) [N]', color='white', fontsize=9)
        ax.set_xlabel('t [s]', color='white'); ax.set_ylabel('N', color='white')
        ax.legend(fontsize=7, loc='best'); ax.grid(alpha=0.3)
        ax.set_facecolor('#16213e'); ax.tick_params(colors='gray')

    # 4. Σλz_des / fz balance
    Mg = m_orig['TOTAL_MASS'] * m_orig['G_ACC']
    ax = axes[3, 0]
    ax.plot(t, m_orig['fz_sum_des'], '-', color='#ff5555', lw=0.9, label='orig Σλz_des')
    ax.plot(t, m_ext ['fz_sum_des'], '-', color='#00d4ff', lw=0.9, label='ext  Σλz_des')
    ax.axhline(Mg, color='white', ls='--', lw=0.8, alpha=0.6, label=f'Mg={Mg:.1f}N')
    ax.set_title('Sum lambda_z (total GRF) [N]', color='white', fontsize=9)
    ax.set_xlabel('t [s]', color='white'); ax.set_ylabel('N', color='white')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_facecolor('#16213e'); ax.tick_params(colors='gray')

    # 5. λ_z 분담 비율 (front vs hind)
    ax = axes[3, 1]
    front_o = np.sum(m_orig['wbc_lam_des'][:, :2, 2], axis=1)
    hind_o  = np.sum(m_orig['wbc_lam_des'][:, 2:, 2], axis=1)
    front_e = np.sum(m_ext ['wbc_lam_des'][:, :2, 2], axis=1)
    hind_e  = np.sum(m_ext ['wbc_lam_des'][:, 2:, 2], axis=1)
    ax.plot(t, front_o/(front_o+hind_o+1e-9), '-', color='#ff5555', lw=0.9, label='orig front share')
    ax.plot(t, front_e/(front_e+hind_e+1e-9), '-', color='#00d4ff', lw=0.9, label='ext  front share')
    ax.set_title('Front lambda_z share ratio', color='white', fontsize=9)
    ax.set_xlabel('t [s]', color='white'); ax.set_ylabel('ratio', color='white')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_facecolor('#16213e'); ax.tick_params(colors='gray')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, facecolor='#1a1a2e', dpi=110)
    print(f'\noverlay plot 저장 → {out_path}')


if __name__ == '__main__':
    skip_run = '--skip-run' in sys.argv
    if not skip_run:
        for v in VARIANTS:
            run_variant(v)
    metrics = {v: load_metrics(v) for v in VARIANTS}
    print_table(metrics['orig'], metrics['ext'])
    make_overlay_plot(metrics['orig'], metrics['ext'])
