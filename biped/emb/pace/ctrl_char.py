#!/usr/bin/env python3
"""ctrl_char.py — 제어/루프 특성 (아티팩트 ①위치제어 + Phase0 지연).

bench_actuator_full 이 **플랜트 파라미터**(α·마찰·b·I·백래시)를 뽑는다면, 이 도구는
**닫힌루프 위치제어의 응답특성**을 뽑는다 — 브링업 순서 Phase 0~1 의 아래 항목:

    ┌──────────┬───────────────────────────────────────────────┬──────────────┐
    │ 페이즈    │ 방법                                            │ 얻는 것       │
    ├──────────┼───────────────────────────────────────────────┼──────────────┤
    │ step     │ 위치 스텝(±) · 타이트폴링(sleep 없음)           │ 지연(dead-time)·│
    │          │   hwio.step_response 이 t=0 에 한 번만 명령      │ 상승·오버슛·   │
    │          │                                                 │ 정착·정상오차·부호│
    │ bw       │ 위치 처프 f0→f1 · 닫힌루프 FRF H=Q/Qcmd          │ −3dB 대역폭·   │
    │          │   (Welch tfestimate, numpy만)                    │ 대역서 위상    │
    └──────────┴───────────────────────────────────────────────┴──────────────┘

★지연(dead-time)은 스텝 트레이스의 **첫 이탈시각**으로 잰다 = EMB→EtherCAT→CAN→MD80
  왕복 전송+연산 지연. 과거 PACE 식별값 8.39ms 와 대조.
★대역폭은 **게인 의존**이다 — 배포 루프를 특성화하려면 그 축의 배포게인(--kp/--kd)을 줄 것.

전제:
  · deploy 종료(모터 writer 하나) · RobotEmbedded relay 기동 · 한 번에 한 축.
  · **추는 없어도 된다**(닫힌루프 추종시험이라 지그만으로 충분·더 안전). 달아도 동작한다.
  · 위치제어라 폭주 없음(kp·err 상한). 처프 진폭은 작게(기본 3°).

사용(.46):
  cd ~/simulation/biped/emb/pace
  python3 ctrl_char.py --ch 2                         # step,bw 둘 다 (기본게인)
  python3 ctrl_char.py --ch 2 --kp 100 --kd 2 --phases step
  python3 ctrl_char.py --ch 2 --phases bw --chirp 0.5,10,24,3
  python3 ctrl_char.py --selftest                     # 하드웨어 없이 metric 로직만
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ══════════════════════════════════════════════════════════════════════════
#  metric 로직 (하드웨어 무관 — selftest 로 검증)
# ══════════════════════════════════════════════════════════════════════════
def step_metrics(t, q, q0, delta, dead_band_deg=0.1) -> dict:
    """스텝 트레이스(t: 명령시각 기준 상대) → 지연·상승·오버슛·정착·정상오차·부호."""
    t = np.asarray(t, float); q = np.asarray(q, float)
    d = float(delta); qf = q0 + d
    resp = q - q0                                    # 시작점 기준 변위
    frac = resp / d if d != 0 else resp              # 목표 대비 진행률(부호 정규화)
    # 지연(온셋): |변위| 가 노이즈밴드를 처음 넘어 **2연속** 유지되는 시각.
    #   = 전송(EMB→EtherCAT→CAN→MD80 왕복) + 검출램프. 스텝직후는 정지→가속이라
    #     밴드 도달에 수 ms 가 더 붙는다 → 이 값은 **순수 전송지연의 상한**이다.
    #     (밴드를 엔코더 노이즈 바로 위로 잡을수록 전송지연에 근접.)
    over = np.abs(resp) > dead_band_deg
    dead = float("nan")
    for i in range(len(over) - 1):
        if over[i] and over[i + 1]:
            dead = float(t[i]); break
    # 상승시간 10→90%
    def _cross(fr):
        idx = np.where(frac >= fr)[0]
        return float(t[idx[0]]) if len(idx) else float("nan")
    t10, t90 = _cross(0.1), _cross(0.9)
    rise = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else float("nan")
    # 오버슛 %(목표 초과 최대), 부호 정규화
    peak = float(np.max(frac)) if d != 0 else 0.0
    overshoot = max(0.0, (peak - 1.0)) * 100.0
    # 정상오차: 마지막 20% 평균과 목표 차
    tail = q[int(len(q) * 0.8):]
    sse = float(qf - np.mean(tail)) if len(tail) else float("nan")
    # 정착시간 ±2%: |q-qf| 가 밴드 밖인 **마지막** 시각
    band = 0.02 * abs(d)
    outb = np.where(np.abs(q - qf) > band)[0]
    settle = float(t[outb[-1]]) if len(outb) else 0.0
    sign_ok = (np.sign(np.mean(tail) - q0) == np.sign(d)) if len(tail) and d != 0 else False
    return dict(dead_ms=dead * 1e3 if np.isfinite(dead) else float("nan"),
                rise_ms=rise * 1e3 if np.isfinite(rise) else float("nan"),
                overshoot_pct=overshoot, settle_ms=settle * 1e3,
                sse_deg=sse, peak_frac=peak, sign_ok=bool(sign_ok))


def tfestimate(u, y, fs, nperseg=None):
    """Welch tfestimate H=Pxy/Pxx (numpy만). 반환 (f[Hz], H[complex])."""
    u = np.asarray(u, float); y = np.asarray(y, float); n = len(u)
    if nperseg is None:
        nperseg = int(2 ** np.floor(np.log2(max(64, n // 6))))
    nperseg = min(nperseg, n)
    stepn = max(1, nperseg // 2)
    win = np.hanning(nperseg); nfb = nperseg // 2 + 1
    Pxy = np.zeros(nfb, complex); Pxx = np.zeros(nfb)
    cnt = 0
    for s in range(0, n - nperseg + 1, stepn):
        uu = (u[s:s + nperseg] - u[s:s + nperseg].mean()) * win
        yy = (y[s:s + nperseg] - y[s:s + nperseg].mean()) * win
        U = np.fft.rfft(uu); Y = np.fft.rfft(yy)
        Pxy += np.conj(U) * Y; Pxx += (np.conj(U) * U).real; cnt += 1
    H = Pxy / np.where(Pxx > 0, Pxx, 1.0)
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return f, H


def bandwidth(f, H, f_lo, f_hi) -> dict:
    """[f_lo,f_hi] 대역 내 −3dB 대역폭. 저역게인 기준 |H|/|H0|=0.707 첫 하향교차."""
    f = np.asarray(f); mag = np.abs(np.asarray(H))
    band = (f >= f_lo) & (f <= f_hi) & (f > 0)
    fb, mb = f[band], mag[band]
    if len(fb) < 4:
        return dict(bw_hz=float("nan"), gain_dc=float("nan"), phase_deg=float("nan"))
    g0 = float(np.mean(mb[:max(2, len(mb) // 10)]))  # 저역(대역 하단) 게인
    thr = g0 / np.sqrt(2.0)
    bw = float("nan")
    for i in range(len(fb) - 1):
        if mb[i] >= thr > mb[i + 1]:                 # 하향 교차 선형보간
            bw = float(fb[i] + (fb[i + 1] - fb[i]) * (mb[i] - thr) / (mb[i] - mb[i + 1]))
            break
    ph = float("nan")
    if np.isfinite(bw):
        j = int(np.argmin(np.abs(f - bw)))
        ph = float(np.rad2deg(np.angle(H[j])))
    return dict(bw_hz=bw, gain_dc=g0, phase_deg=ph, thr=thr)


# ══════════════════════════════════════════════════════════════════════════
#  하드웨어 페이즈
# ══════════════════════════════════════════════════════════════════════════
def phase_step(hw, ch, kp, kd, delta, window_s, log):
    from hwio import samples_to_arrays  # noqa
    hw.arm(ch, kp, kd)
    rows = []
    for d in (+abs(delta), -abs(delta)):
        r = hw.step_response(ch, d, kp, kd, window_s=window_s, settle_s=0.5)
        m = step_metrics(r["t"], r["q"], r["q0"], r["delta"])
        rows.append(m)
        log(f"  Δ{d:+.0f}°: 지연 {m['dead_ms']:.1f}ms · 상승 {m['rise_ms']:.0f}ms · "
            f"오버슛 {m['overshoot_pct']:.0f}% · 정착 {m['settle_ms']:.0f}ms · "
            f"정상오차 {m['sse_deg']:+.2f}° · 부호 {'OK' if m['sign_ok'] else '✗반대'}")
    dead = np.nanmean([r["dead_ms"] for r in rows])
    rise = np.nanmean([r["rise_ms"] for r in rows])
    ov = np.nanmax([r["overshoot_pct"] for r in rows])
    sse = np.nanmean([np.abs(r["sse_deg"]) for r in rows])
    signs = all(r["sign_ok"] for r in rows)
    log(f"  → 지연(온셋≈전송상한) {dead:.1f}ms (PACE 8.39ms 대조) · 상승 {rise:.0f}ms · "
        f"오버슛peak {ov:.0f}% · |정상오차| {sse:.2f}° · 부호 {'OK' if signs else '✗'}")
    if ov > 25:
        log("    ⚠오버슛 큼 — kd 부족/게인 과다. 발진 위험.")
    return dict(dead_ms=float(dead), rise_ms=float(rise), overshoot_pct=float(ov),
                sse_deg=float(sse), sign_ok=signs)


def phase_bw(hw, ch, kp, kd, f0, f1, T, amp, log):
    from hwio import samples_to_arrays
    hw.arm(ch, kp, kd)
    center = hw.read(ch)[0]

    def qcmd(t):
        f = f0 + (f1 - f0) * t / (2 * T)
        return center + amp * np.sin(2 * np.pi * f * t)
    # ★대역폭 측정은 고주파에서 q 가 **의도적으로 감쇠**(그게 −3dB)라, 측정값이 거의 안 변해
    #   stale 검사가 오탐한다(실제 EtherCAT 정지가 아님). 처프 동안만 stale 끔(tau/vel/err 유지).
    _stale = hw.lim.stale_ms
    hw.lim.stale_ms = 1e9
    try:
        ss = hw.run(ch, qcmd, T, kp, kd, progress="  bw-chirp")
    finally:
        hw.lim.stale_ms = _stale
    hw.goto(ch, center, kp, kd, speed_dps=8.0)
    a = samples_to_arrays(ss)
    t, q, qcmd_a = a["t"], a["q"], a["q_cmd"]
    fs = 1.0 / float(np.median(np.diff(t)))
    f, H = tfestimate(qcmd_a, q, fs)
    bw = bandwidth(f, H, f0, f1)
    # 주파수별 |H|/저역 롤오프 표 (−3dB=0.707 위치가 보이게)
    mag = np.abs(H); g0 = bw["gain_dc"] if bw["gain_dc"] > 1e-9 else 1.0
    log("  |H|/저역 롤오프 (−3dB=0.707):")
    for ftgt in (1, 2, 4, 6, 8, 10, 12, 15, 20):
        if ftgt <= f1 + 0.5:
            j = int(np.argmin(np.abs(f - ftgt)))
            r = mag[j] / g0
            ph = np.rad2deg(np.angle(H[j]))
            log(f"      {f[j]:5.1f}Hz: {r:.3f} ({ph:+4.0f}°){'  ← −3dB' if abs(r - 0.707) < 0.06 else ''}")
    log(f"  → −3dB 대역폭 {bw['bw_hz']:.2f} Hz · 위상 {bw['phase_deg']:+.0f}° · 저역게인 {bw['gain_dc']:.3f}"
        f" (fs={fs:.0f}Hz, 스윕 {f0}-{f1}Hz)")
    if not np.isfinite(bw["bw_hz"]):
        log(f"    ⚠대역 내 −3dB 없음 → 대역폭 > {f1}Hz (루프가 빠름). step 상승시간으로 교차확인.")
    return bw


# ══════════════════════════════════════════════════════════════════════════
def _selftest():
    print("  [selftest] metric 로직 합성검증:")
    fs = 1000.0
    # (1) 2차 언더댐프 스텝: ωn=40rad/s, ζ=0.5 → 오버슛≈16%, +5ms dead-time
    wn, z = 40.0, 0.5
    t = np.arange(0, 0.6, 1 / fs); d = 10.0; L = 0.005
    wd = wn * np.sqrt(1 - z**2)
    resp = np.where(t < L, 0.0,
                    d * (1 - np.exp(-z * wn * (t - L)) *
                         (np.cos(wd * (t - L)) + z / np.sqrt(1 - z**2) * np.sin(wd * (t - L)))))
    q = 5.0 + resp
    m = step_metrics(t, q, 5.0, d)
    print(f"    step: 지연온셋 {m['dead_ms']:.1f}ms(전송5ms+가속램프≈8~10) · 상승 {m['rise_ms']:.0f}ms · "
          f"오버슛 {m['overshoot_pct']:.0f}%(≈16) · 정상오차 {m['sse_deg']:+.2f}°(≈0) · 부호 {m['sign_ok']}")
    assert 5.0 <= m["dead_ms"] < 12.0, f"지연온셋 범위밖 {m['dead_ms']}"
    assert 8 < m["overshoot_pct"] < 25, "오버슛 오류"
    assert m["sign_ok"], "부호 오류"
    # (2) 1차 저역통과 fc=4Hz 를 처프에 통과 → 대역폭 복원
    T = 24.0; t2 = np.arange(0, T, 1 / fs); f0, f1 = 0.5, 15.0
    ph = 2 * np.pi * (f0 * t2 + (f1 - f0) * t2**2 / (2 * (2 * T)))
    u = 3.0 * np.sin(ph)
    fc = 4.0; a1 = np.exp(-2 * np.pi * fc / fs)      # 1차 IIR 저역
    y = np.zeros_like(u)
    for i in range(1, len(u)):
        y[i] = (1 - a1) * u[i] + a1 * y[i - 1]
    f, H = tfestimate(u, y, fs)
    bw = bandwidth(f, H, f0, f1)
    print(f"    bw: 복원 대역폭 {bw['bw_hz']:.2f}Hz (심음 fc=4.0)")
    assert abs(bw["bw_hz"] - 4.0) < 1.0, f"대역폭 복원 실패 {bw['bw_hz']}"
    print("  [selftest] OK — step metric·대역폭 FRF 모두 통과")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ch", type=int)
    ap.add_argument("--kp", type=float, default=100.0, help="위치게인(배포루프 특성화면 배포값)")
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--delta", type=float, default=10.0, help="스텝 크기 ±[°]")
    ap.add_argument("--window", type=float, default=0.5, help="스텝 관측창[s]")
    ap.add_argument("--chirp", default="0.5,15,24,2.5", help="bw: f0[Hz],f1[Hz],T[s],amp[°]")
    ap.add_argument("--phases", default="step,bw")
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    ap.add_argument("--zero", type=float, default=0.0, help="각 궤적 전/후 복귀할 0점[deg]")
    ap.add_argument("--pause", action="store_true", help="각 페이즈 전 Enter 대기")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.ch is None:
        ap.error("--ch 필요 (또는 --selftest)")

    import yaml
    from bench_actuator_full import open_hw, goto_zero
    spec = yaml.safe_load(open(a.spec, encoding="utf-8"))
    phases = [p.strip() for p in a.phases.split(",") if p.strip()]
    f0, f1, T, amp = (float(x) for x in a.chirp.split(","))
    print("=" * 72)
    print(f"  제어/루프 특성 · ch{a.ch} · kp{a.kp}/kd{a.kd} · 페이즈 {phases}")
    print("=" * 72)
    hw = open_hw(spec)
    res = {}

    def gate(header, danger=False):
        """헤더 → (Enter 대기) → **0점 복귀 후** 페이즈 진행."""
        print(header)
        if a.pause or danger:
            try:
                input("  ⏎ Enter 로 진행 · Ctrl+C 로 중단 …")
            except (EOFError, KeyboardInterrupt):
                raise KeyboardInterrupt
        goto_zero(hw, a.ch, a.kp, a.kd, a.zero)
    try:
        with hw:
            if "step" in phases:
                gate("\n[①] 위치 스텝응답 + 지연 (±스텝, 타이트폴링)")
                res["step"] = phase_step(hw, a.ch, a.kp, a.kd, a.delta, a.window, log=print)
            if "bw" in phases:
                gate("\n[②] 대역폭 (위치 처프 · 닫힌루프 FRF)")
                res["bw"] = phase_bw(hw, a.ch, a.kp, a.kd, f0, f1, T, amp, log=print)
            print("\n[종료] 0점 복귀")            # ★정상 완료 → 마지막 0점 복귀 후 종료
            goto_zero(hw, a.ch, a.kp, a.kd, a.zero)
    finally:
        hw.limp()

    print("\n" + "=" * 72 + "\n  제어특성 요약\n" + "=" * 72)
    st = res.get("step", {}); bw = res.get("bw", {})
    if st:
        print(f"  지연(온셋)     {st['dead_ms']:.1f} ms   (전송지연 상한 · PACE 8.39ms 대조)")
        print(f"  상승시간       {st['rise_ms']:.0f} ms · 오버슛 {st['overshoot_pct']:.0f}% · "
              f"|정상오차| {st['sse_deg']:.2f}° · 부호 {'OK' if st['sign_ok'] else '✗'}")
    if bw:
        print(f"  −3dB 대역폭    {bw['bw_hz']:.2f} Hz · 대역서 위상 {bw['phase_deg']:+.0f}°")
    return 0


if __name__ == "__main__":
    sys.exit(main())
