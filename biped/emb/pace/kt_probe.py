#!/usr/bin/env python3
"""kt_probe.py — 토크상수 k_t 검증 (전류 실측이 **진짜**일 때만).

배경(2026-08-12 확인, hwio.py:173-179):
  SHM 의 fCurrent 슬롯은 과거 펌웨어에서 **fTorque 의 비트단위 복제**였다. 그 "전류" 로
  τ/i 를 계산하면 k_t 가 아니라 **1/kt 항등식**이 재현될 뿐이다(가짜 검증). 그래서 이 도구는
  **먼저 fCurrent 가 fTorque 와 독립인지 판정**하고, 독립일 때만 k_t 회귀를 돌린다.

전제:
  · MD80/RO100 데이터시트: k_t(모터)=0.2 Nm/A · 기어 **7:1 확정** · η≈0.9~1.0
    → k_t(관절) = 0.2 × 7 × η = **1.26~1.40 Nm/A** (컨트롤러 prior).
    → 이미 교차검증: 0.2×7×62A = 86.8 ≈ tau_peak(hip) 84 Nm (η≈0.97). 물리사슬 정합.
  · ground-truth 토크 = 중력 τ_out(θ)=m·g·L·cos(θ−q0). 전류 없이도 α(명령→출력 충실도)는
    bench_actuator_full 의 alpha 페이즈가 이미 준다. 이 도구는 **오직 전류기반 k_t** 전용.

측정법(위치홀드 = 폭주 없음):
  여러 각도에 정지홀드 → 각에서 τ(fTorque, 신뢰) 와 i(fCurrent) 를 동시 수집.
  중력으로 τ 가 mgL→0 스윕된다. 독립이면 i 도 같이 변한다.
    k_t(관절) = Δτ/Δi 회귀기울기,  k_t(모터) = k_t(관절)/7,  선형성 = R².

사용(.46):
  cd ~/simulation/biped/emb/pace
  AUX_MODE=1 python3 kt_probe.py --ch 2 --mass 2.0 --lever 0.13
  python3 kt_probe.py --selftest        # 하드웨어 없이 판정로직만
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

G = 9.80665
GEAR = 7.0                     # ★확정 감속비 (사용자 확인 2026-09-11)
KT_MOTOR_SPEC = 0.2            # RO100 데이터시트 [Nm/A]
IDENT_EPS = 1e-4              # |i-τ| 최대차가 이 아래면 '복제'(독립 아님)


def _sweep(hw, ch, mass, lever, kp, span_deg, n_ang, log):
    """각도별 정지홀드 → (τ_report, i_report, q) 평균 수집. 중력으로 τ 스윕."""
    kd = 2.0
    hw.arm(ch, kp, kd)
    q_home = hw.read(ch)[0]
    targets = q_home + np.linspace(-span_deg, span_deg, n_ang)
    rows = []                                   # (q_rad, tau_mean, cur_mean, maxdiff)
    for tgt in targets:
        hw.goto(ch, tgt, kp, kd, speed_dps=8.0)
        hold = hw.run(ch, lambda t, g=tgt: g, 1.0, kp, kd)   # 1s 정착유지
        taus = np.array([s.tau for s in hold]); curs = np.array([s.cur for s in hold])
        qs = np.array([s.q_deg for s in hold])
        m = np.arange(len(taus)) > len(taus) // 2            # 후반부(정착후)
        rows.append((np.deg2rad(qs[m].mean()), taus[m].mean(), curs[m].mean(),
                     float(np.max(np.abs(taus - curs)))))
    hw.goto(ch, q_home, kp, kd, speed_dps=8.0)
    return np.array(rows)


def _verdict(R, mass, lever, log):
    q, tau, cur = R[:, 0], R[:, 1], R[:, 2]
    mgl = mass * G * lever
    maxdiff = float(np.max(np.abs(tau - cur)))

    # ── (A) 무게추 역산 = 토크체인 검증 (전류 불요) ──────────────────────────
    #   정지홀드에서 전달토크=mgL cosθ(중력균형). 드라이버 보고토크 fTorque 를 각도로 피팅:
    #     τ_rep = a·cosθ + b·sinθ → 보고진폭 A_rep, 수평 q0.
    #   토크스케일 α = mgL / A_rep  (=명령→실현 비, bench alpha 와 일치해야).
    #   ★이건 k_t·기어·η·손실을 **한 덩어리**로 검증한다 — k_t 단독분리는 아님.
    coef, *_ = np.linalg.lstsq(np.column_stack([np.cos(q), np.sin(q)]), tau, rcond=None)
    A_rep = float(np.hypot(*coef)); q0 = float(np.arctan2(coef[1], coef[0]))
    alpha = mgl / A_rep if A_rep > 1e-9 else float("nan")
    resid = tau - np.column_stack([np.cos(q), np.sin(q)]) @ coef
    ss_tot = float(np.sum((tau - tau.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 1e-12 else float("nan")
    log("  [A] 무게추 역산 — 토크체인 검증(전류 불요):")
    log(f"      보고토크진폭 {A_rep:.3f} Nm vs 물리 mgL {mgl:.3f} Nm → **토크스케일 α={alpha:.3f}**"
        f"  ·  R²={r2:.4f}  ·  수평 q0={np.rad2deg(q0):+.1f}°")
    ok = 0.6 < alpha < 1.15
    log(f"      ⇒ {'OK — 명령토크가 물리적으로 실현됨(토크체인 정합). bench alpha 와 대조.' if ok else '⚠α 이상 — 부호/레버/질량 재확인'}")
    for eta in (1.0, 0.9):                          # datasheet 전제 역산전류(외부프로브 대조용)
        i_h = mgl / (GEAR * KT_MOTOR_SPEC * eta)
        log(f"      역산 전류(수평·η={eta}) i = mgL/(7·0.2·η) = {i_h:.2f} A")

    # ── (B) 절대 k_t (Nm/A) = 독립 전류 필요 ─────────────────────────────────
    log(f"  [B] 절대 k_t(Nm/A) — fCurrent vs fTorque 비트차 {maxdiff:.6g}:")
    if maxdiff < IDENT_EPS:
        log("      복제(독립전류 없음) → **Nm/A 절대측정 불가**. datasheet 사용:")
        log(f"      k_t(모터)={KT_MOTOR_SPEC}, k_t(관절)=0.2×{GEAR:.0f}×η=1.26~1.40 [피크토크 86.8≈84Nm 로 이미 정합].")
        return dict(alpha=alpha, r2=r2, q0_deg=float(np.rad2deg(q0)), independent=False, maxdiff=maxdiff)
    # 독립전류 → τ vs i 회귀. slope = τ/i 상수[Nm/A].
    A = np.column_stack([cur, np.ones_like(cur)])
    (slope, intercept), *_ = np.linalg.lstsq(A, tau, rcond=None)
    kt = float(slope)
    pred = A @ np.array([slope, intercept])
    ss_t = float(np.sum((tau - tau.mean()) ** 2))
    r2b = 1.0 - float(np.sum((tau - pred) ** 2)) / ss_t if ss_t > 1e-12 else float("nan")
    i_dead = -intercept / slope if abs(slope) > 1e-9 else float("nan")
    # ★프레임 모호성: fTorque·fCurrent 기준이 모터냐 관절이냐에 따라 slope 가 k_t,motor 또는
    #   k_t,joint. datasheet 모터 0.2 / 관절 0.2×7=1.4 **양쪽과 대조**해 어느 프레임인지 판정.
    kt_j_spec = KT_MOTOR_SPEC * GEAR
    d_motor = abs(kt - KT_MOTOR_SPEC); d_joint = abs(kt - kt_j_spec)
    log(f"      ✔독립전류 → **τ/i 기울기 k_t = {kt:.4f} Nm/A** · R²={r2b:.4f} · 전류데드존 {i_dead:+.3f} A")
    log(f"        datasheet 대조: 모터 {KT_MOTOR_SPEC} (편차 {100*(kt-KT_MOTOR_SPEC)/KT_MOTOR_SPEC:+.0f}%)"
        f" · 관절 {kt_j_spec:.2f} (편차 {100*(kt-kt_j_spec)/kt_j_spec:+.0f}%)")
    if d_motor < d_joint:
        log(f"        ⇒ **모터프레임**: k_t,motor≈{kt:.3f}(=datasheet), k_t,joint=slope×7={kt*GEAR:.2f}")
    else:
        log(f"        ⇒ **관절프레임**: k_t,joint≈{kt:.3f}, k_t,motor=slope/7={kt/GEAR:.3f}")
    return dict(alpha=alpha, r2=r2, q0_deg=float(np.rad2deg(q0)), independent=True,
                maxdiff=maxdiff, kt_slope=kt, r2_kt=r2b)


def _selftest():
    print("  [selftest] 판정로직 — 무게추 역산(α) + 복제/독립 두 경우:")
    mass, lever = 2.0, 0.13; mgl = mass * G * lever
    q = np.deg2rad(np.linspace(-70, 70, 9)); q0 = np.deg2rad(10.0)
    tau = mgl * np.cos(q - q0)                       # 보고토크=중력균형 → 토크스케일 α≈1
    # (1) 복제: cur = tau
    R_dup = np.column_stack([q, tau, tau.copy(), np.zeros_like(tau)])
    r1 = _verdict(R_dup, mass, lever, lambda m: print("   " + m))
    assert r1["independent"] is False, "복제를 독립으로 오판"
    assert abs(r1["alpha"] - 1.0) < 0.05, f"역산 토크스케일 오류 {r1['alpha']}"
    # (2) 독립: cur = tau/1.4 + noise
    kt = 1.40; cur = tau / kt + np.random.default_rng(0).normal(0, 0.002, tau.shape)
    R_ind = np.column_stack([q, tau, cur, np.abs(tau - cur)])
    r2 = _verdict(R_ind, mass, lever, lambda m: print("   " + m))
    assert r2["independent"] is True, "독립을 복제로 오판"
    assert abs(r2["kt_slope"] - 1.40) < 0.05, f"k_t 기울기 복원 실패 {r2['kt_slope']}"
    print(f"  [selftest] OK — 역산 α({r1['alpha']:.3f}≈1.0) · 복제/독립 판정 · k_t 기울기 복원({r2['kt_slope']:.3f}≈1.40)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ch", type=int)
    ap.add_argument("--mass", type=float, default=2.0)
    ap.add_argument("--lever", type=float, default=0.13)
    ap.add_argument("--kp", type=float, default=100.0)
    ap.add_argument("--span-deg", type=float, default=70.0, help="홀드 각도범위 ±[°] (중력 τ 스윕폭)")
    ap.add_argument("--n-ang", type=int, default=9)
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    ap.add_argument("--zero", type=float, default=0.0, help="궤적 전/후 복귀할 0점[deg]")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.ch is None:
        ap.error("--ch 필요 (또는 --selftest)")

    import yaml
    from bench_actuator_full import open_hw, goto_zero
    spec = yaml.safe_load(open(a.spec, encoding="utf-8"))
    mgl = a.mass * G * a.lever
    print("=" * 72)
    print(f"  k_t 검증 · ch{a.ch} · {a.mass}kg @ {a.lever}m · 기어 {GEAR:.0f}:1 확정")
    print(f"  중력 ground-truth: τ_out(수평)=mgL={mgl:.3f} Nm → τ 를 {mgl:.2f}→0 으로 스윕")
    print("=" * 72)
    hw = open_hw(spec)
    try:
        with hw:
            print("[시작] 0점 복귀")
            goto_zero(hw, a.ch, a.kp, 2.0, a.zero)          # ★궤적 전 0점 복귀
            R = _sweep(hw, a.ch, a.mass, a.lever, a.kp, a.span_deg, a.n_ang, log=print)
            res = _verdict(R, a.mass, a.lever, log=print)
            print("\n[종료] 0점 복귀")                       # ★정상 완료 → 0점 복귀 후 종료
            goto_zero(hw, a.ch, a.kp, 2.0, a.zero)
    finally:
        hw.limp()
    return 0


if __name__ == "__main__":
    sys.exit(main())
