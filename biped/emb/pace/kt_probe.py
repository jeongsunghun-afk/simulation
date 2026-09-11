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
    q, tau, cur, _ = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
    maxdiff = float(np.max(np.abs(tau - cur)))
    log(f"  fCurrent vs fTorque 최대차 = {maxdiff:.6g}  (전 홀드 {len(R)}점)")
    if maxdiff < IDENT_EPS:
        log("  ⇒ **fCurrent = fTorque 복제** — SHM 에 독립 전류 없음(구펌웨어와 동일).")
        log(f"     k_t 전류측정 불가. 데이터시트값 사용: k_t(모터)={KT_MOTOR_SPEC} Nm/A,")
        log(f"     k_t(관절)=0.2×{GEAR:.0f}×η = 1.26(η0.9)~1.40(η1.0) Nm/A [교차검증 완료: 86.8≈84Nm].")
        log("     진짜 전류가 필요하면 → 외부 전류프로브 또는 MCU 펌웨어의 실 Iq 노출.")
        log("     (명령→출력 토크충실도 α 는 bench_actuator_full 의 alpha 페이즈가 전류 없이 준다.)")
        return dict(independent=False, maxdiff=maxdiff)
    # 독립 — k_t 회귀. τ_report(신뢰) vs i. 기울기 = k_t(관절).
    log("  ⇒ **fCurrent 가 fTorque 와 독립** — 진짜 전류로 판단, k_t 회귀 수행.")
    A = np.column_stack([cur, np.ones_like(cur)])
    (slope, intercept), *_ = np.linalg.lstsq(A, tau, rcond=None)
    pred = A @ np.array([slope, intercept])
    ss_res = float(np.sum((tau - pred) ** 2)); ss_tot = float(np.sum((tau - tau.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    kt_joint = float(slope); kt_motor = kt_joint / GEAR
    i_dead = -intercept / slope if abs(slope) > 1e-9 else float("nan")   # τ=0 일 때 전류(데드존)
    log(f"  → k_t(관절)={kt_joint:.4f} Nm/A  ·  k_t(모터)={kt_motor:.4f} Nm/A"
        f"  (데이터시트 {KT_MOTOR_SPEC}, 편차 {100*(kt_motor-KT_MOTOR_SPEC)/KT_MOTOR_SPEC:+.0f}%)")
    log(f"  → 선형성 R²={r2:.4f}  ·  전류 데드존(τ=0 절편)={i_dead:+.3f} A")
    if r2 < 0.98:
        log("    ⚠R² 낮음 — 비선형/데드존/열드리프트 의심. 표본각·홀드시간 늘려 재측정.")
    return dict(independent=True, maxdiff=maxdiff, kt_joint=kt_joint, kt_motor=kt_motor,
                r2=r2, i_deadzone_a=i_dead)


def _selftest():
    print("  [selftest] 판정로직 — 두 경우 합성:")
    # (1) 복제: cur = tau
    tau = np.linspace(0, 2.55, 9); q = np.deg2rad(np.linspace(-60, 60, 9))
    R_dup = np.column_stack([q, tau, tau.copy(), np.zeros_like(tau)])
    r1 = _verdict(R_dup, 2.0, 0.13, lambda m: print("   " + m))
    assert r1["independent"] is False, "복제를 독립으로 오판"
    # (2) 독립: cur = tau/1.4 + noise
    kt = 1.40; cur = tau / kt + np.random.default_rng(0).normal(0, 0.002, tau.shape)
    R_ind = np.column_stack([q, tau, cur, np.abs(tau - cur)])
    r2 = _verdict(R_ind, 2.0, 0.13, lambda m: print("   " + m))
    assert r2["independent"] is True, "독립을 복제로 오판"
    assert abs(r2["kt_joint"] - 1.40) < 0.02, f"k_t 복원 실패 {r2['kt_joint']}"
    print(f"  [selftest] OK — 복제·독립 판정 + k_t 복원({r2['kt_joint']:.3f}≈1.40) 통과")
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
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.ch is None:
        ap.error("--ch 필요 (또는 --selftest)")

    import yaml
    from bench_actuator_full import open_hw
    spec = yaml.safe_load(open(a.spec, encoding="utf-8"))
    mgl = a.mass * G * a.lever
    print("=" * 72)
    print(f"  k_t 검증 · ch{a.ch} · {a.mass}kg @ {a.lever}m · 기어 {GEAR:.0f}:1 확정")
    print(f"  중력 ground-truth: τ_out(수평)=mgL={mgl:.3f} Nm → τ 를 {mgl:.2f}→0 으로 스윕")
    print("=" * 72)
    hw = open_hw(spec)
    try:
        with hw:
            R = _sweep(hw, a.ch, a.mass, a.lever, a.kp, a.span_deg, a.n_ang, log=print)
            res = _verdict(R, a.mass, a.lever, log=print)
    finally:
        hw.limp()
    return 0


if __name__ == "__main__":
    sys.exit(main())
