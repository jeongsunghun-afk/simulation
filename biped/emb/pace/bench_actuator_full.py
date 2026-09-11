#!/usr/bin/env python3
"""bench_actuator_full.py — 무게추 벤치로 구동기 **전(全) 파라미터** 추출 예제.

떼어낸 구동기 하나를 강체 지그에 고정하고, 레버(길이 L)에 무게추(질량 m)를 달아
**같은 EMB→EtherCAT→MCU→CAN→MD80 체인**(hwio.Hardware)으로 구동한다.
알려진 부하 τ_L(θ)=m·g·L·cos θ 를 **ground-truth 토크**로 써서, i_q(실전류) 없이도
아래를 전부 뽑는다.

    ┌────────────┬──────────────────────────────────────────────┬──────────────┐
    │ 파라미터   │ 궤적(프로파일)                                │ 레버 상태    │
    ├────────────┼──────────────────────────────────────────────┼──────────────┤
    │ α(토크스케일)│ **각도별 정지**(step-hold, 위·아래 양방향접근) │ 자유+무게추  │
    │ Fs(정지마찰)│ 위 hold 의 상행/하행 정착토크 반차             │ 자유+무게추  │
    │ b(점성)·I  │ **처프**(위치 정현, 주파수 스윕) 1회 회귀       │ 자유+무게추  │
    │  (armature) │   α·τ−mgL cosθ = I·θ̈ + b·θ̇ + Fc·sgn(θ̇)      │              │
    │ I_act 교차  │ 자유진동 주기(α 무관)                          │ 자유+무게추  │
    │ T-N 선도    │ **자유가속**(최대토크·알려진관성)→τ_out(ω)      │ 자유+무게추  │
    │ 백래시·강성 │ **양방향 토크왕복**(0 통과, 기존 backlash 툴)   │ **클램프**   │
    └────────────┴──────────────────────────────────────────────┴──────────────┘

★T-N 선도(토크-속도)의 한계
  · **뽑히는 것**: 저속~보행스윙(~700dps)까지의 T-N 포락선. 최대토크로 알려진 관성을
    자유가속시켜 θ̈(ω)→τ_out(ω). **α 의 속도의존성**(고속 토크하락=역기전력 제한)까지 검증.
  · **안 뽑히는 것**: 무부하속도까지의 전체 포락선. 레버 가동폭 때문에 도달속도가 제한되고,
    그 이상은 **다이나모미터(제어가능 부하)** 나 전기계측(i_q·V_bus)이 필요.

★프로파일이 파라미터마다 다른 이유
  · α·정지마찰 → **정지(θ̈=θ̇=0)** 여야 α·τ=mgL cosθ 가 순수 성립. 처프·회전은 관성·
    점성이 명령토크에 섞여 α 를 오염시킨다. 그래서 각도별 step-hold(또는 준정적 저속).
  · 점성 b·관성 I → 본질적으로 **동적**이라 속도·가속이 필요 → **처프**(=r26/PACE 방식,
    직접 비교 보너스) 또는 등속스윕+자유진동.
  · 백래시 → **부호가 바뀌는 전달토크**로만 유격을 횡단한다. 매단 추는 단방향(중력)이라
    메시가 한 flank 에 늘 물려 **백래시를 못 지나간다**(추 무거울수록 더 안 보임).
    ⇒ 무게추로는 백래시 측정 불가. 출력을 **클램프**하고 ±토크 0-통과로만 잰다.

★장착
  · α·마찰·점성·관성: 레버 **자유**(무게추가 관절을 돌릴 수 있게). 절대영점 몰라도 됨 —
    cos 피팅의 위상 q0 이 수평기준각을 자동 추정.
  · 백래시·강성: 레버를 **하드스톱에 클램프**(출력 고정). 무게추는 떼거나 무관.

★검증 메모(2026-09-04, 워크플로 어드버서리)
  · 백래시 토크축은 **실물 데드웨이트/로드셀** 권장 — MD80 명령토크는 강성 k 를 17~20%
    틀리게 한다. lash 비교는 **joint-side 엔코더(hip/thigh)** 로만(벨트 aux 는 과소보고).
  · α 는 각도축이라 명령토크 스케일에 둔감 → step-hold 로 안전하게 측정 가능.

★안전
  · α·마찰: **위치제어**라 폭주 없음(복원력 있음). 명령토크는 kp·err 로 계산.
  · 처프: 위치제어(복원력 유지). f1·진폭을 폐루프 대역 아래로.
  · 관성: **limp 자유진동**(무여자)이라 폭주 없음.
  · 한 번에 한 축(벤치엔 이 구동기뿐).

사용:
  # 레버 자유 + 2kg@0.2m, thigh(ch1) — α·마찰·점성·관성
  python3 bench_actuator_full.py --ch 1 --mass 2.0 --lever 0.20
  # 오프라인 자기검사(하드웨어 없이 로직만)
  python3 bench_actuator_full.py --ch 1 --selftest
  # 레버 클램프에서 백래시·강성만
  python3 bench_actuator_full.py --ch 1 --clamped --phases backlash
"""
from __future__ import annotations

import argparse
import ctypes as C
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tests"))
sys.path.insert(0, HERE)

from hwio import Hardware, Limits, SafetyAbort, samples_to_arrays  # noqa: E402

G = 9.80665


# ══════════════════════════════════════════════════════════════════════════
#  하드웨어
# ══════════════════════════════════════════════════════════════════════════
def open_hw(spec: dict) -> Hardware:
    shm = spec["shm"]; sf = spec.get("safety", {}); gn = spec.get("gains", {})
    # 벤치 리밋 — 추종오차(err_max)는 크게. 이 시험은 kp·err 로 부하를 읽으므로
    # 큰 추종오차가 정상이다(수평·2kg·kp100 → err ≈ 2.7°).
    lim = Limits(
        q_min=-180.0, q_max=180.0,
        tau_trip=float(max(sf.get("tau_trip_nm", 8.0), 11.0)),
        tau_trip_ms=float(sf.get("tau_trip_ms", 50.0)),
        vel_trip=float(max(sf.get("vel_trip_dps", 200.0), 400.0)),
        err_max=20.0,
        stale_ms=float(sf.get("stale_ms", 500.0)),
        kp_max=float(gn.get("kp_max", 100.0)),
        kd_max=float(gn.get("kd_max", 6.0)),
    )
    return Hardware(shm["lib"], int(shm["n_channel"]), float(shm["rate_hz"]),
                    lim, recv_wait_ms=int(shm.get("recv_wait_ms", 3000)),
                    enable_ramp_s=float(gn.get("enable_ramp_s", 0.3)))


def bind_aux(hw: Hardware):
    """.so 의 bridge_aux(pos,vel) 바인딩(파이썬 래퍼엔 없음). AUX_MODE=1 일 때만 유효."""
    try:
        fn = hw.lib.bridge_aux
        fn.restype = C.c_int
        fn.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float)]
    except Exception:
        return None
    pos = (C.c_float * hw.n)(); vel = (C.c_float * hw.n)()

    def read_aux(ch):
        return (float(pos[ch]), float(vel[ch])) if fn(pos, vel) == 1 else None
    return read_aux


def _cols(samples):
    """Sample 리스트 → (q[rad], dq[rad/s], tau_cmd[Nm], t[s], q_cmd[rad])."""
    a = samples_to_arrays(samples)
    q = np.deg2rad(a["q_deg"]); dq = np.deg2rad(a["dq_dps"])
    tau_cmd = a["kp"] * np.deg2rad(a["q_cmd_deg"] - a["q_deg"])   # 명령토크[Nm]
    return q, dq, tau_cmd, a["t"], np.deg2rad(a["q_cmd_deg"])


def goto_zero(hw, ch, kp, kd, zero_deg=0.0, speed_dps=10.0, log=print):
    """궤적 전/후 **0점 복귀**(에너자이즈 상태로 등속 이동 후 그 자리 유지).
    ⚠--clamped(레버 하드스톱 고정)에서는 호출 금지 — 스톱과 충돌해 스톨/과전류."""
    hw.arm(ch, kp, kd)
    hw.goto(ch, zero_deg, kp, kd, speed_dps=speed_dps)
    log(f"  0점 복귀 → {zero_deg:+.1f}° (현재 q={hw.read(ch)[0]:+.2f}°)")


# ══════════════════════════════════════════════════════════════════════════
#  1.  α + 정지마찰 Fs  — 각도별 step-and-hold (위·아래 양방향 접근)
# ══════════════════════════════════════════════════════════════════════════
def phase_alpha(hw, ch, mass, lever, kp, span_deg, n_ang, log):
    """각 목표각을 **위에서 내려오며** / **아래서 올라오며** 접근해 정지시킨 뒤,
       정착 명령토크 τ=kp·(q_cmd−q) 를 읽는다(θ̈=θ̇≈0 → 순수 정적).
         두 접근 평균 τ̄ 에서:  α·τ̄ = m·g·L·cos(q−q0)
           τ = a·cos q + b·sin q  최소제곱 →  A=√(a²+b²)=mgL/α,  q0=atan2(b,a)
           α = m·g·L / A
         위/아래 반차:  2·Fs/α  →  정지마찰 Fs = α·(τ_up−τ_dn)/2
    """
    mgl = mass * G * lever; kd = 2.0
    hw.arm(ch, kp, kd)
    q_home = hw.read(ch)[0]
    targets = q_home + np.linspace(-span_deg, span_deg, n_ang)
    rows = []                          # (q_rad, tau, appr, tgt)
    for tgt in targets:
        for appr in (+1.0, -1.0):      # +1 위에서, −1 아래서
            hw.goto(ch, tgt + appr * 12.0, kp, kd, speed_dps=12.0)
            hw.goto(ch, tgt, kp, kd, speed_dps=3.0)            # 준정적 접근
            hold = hw.run(ch, lambda t, g=tgt: g, 1.2, kp, kd)  # 정착 유지
            q, dq, tau, t, _ = _cols(hold)
            m = t > (t[-1] - 0.6)                              # 정착 후 0.6s
            rows.append((float(np.mean(q[m])), float(np.mean(tau[m])), appr, tgt))
    hw.goto(ch, q_home, kp, kd, speed_dps=12.0)

    R = np.array(rows)
    q, tau, appr = R[:, 0], R[:, 1], R[:, 2]
    coef, *_ = np.linalg.lstsq(np.column_stack([np.cos(q), np.sin(q)]), tau, rcond=None)
    A = float(np.hypot(*coef)); q0 = float(np.arctan2(coef[1], coef[0]))
    alpha = mgl / A if A > 1e-6 else float("nan")
    up, dn = tau[appr > 0], tau[appr < 0]
    Fs = float(alpha * np.mean(np.abs(up - dn)) / 2.0)
    resid = tau - np.column_stack([np.cos(q), np.sin(q)]) @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    # α 상수성 자기검증 — 각도 3구간 국소 α 편차
    thc = np.abs(np.cos(q - q0)) > 0.3
    log(f"  → α = {alpha:.3f}  ·  정지마찰 Fs = {Fs:.3f} Nm  ·  수평기준 q0 = {np.rad2deg(q0):+.1f}°"
        f"  ·  적합 RMS {rms:.3f} Nm  (hold {len(R)}점)")
    if rms > 0.15 * A:
        log(f"    ⚠ RMS 큼 → α 가 각도에 따라 변함(드라이버가 단순 스칼라 아님) 의심")
    return dict(alpha=alpha, Fs=Fs, q0_deg=float(np.rad2deg(q0)), A=A, rms=rms, n=len(R), mgl=mgl)


# ══════════════════════════════════════════════════════════════════════════
#  2.  점성 b + 반사관성 I_act  — 위치 처프 (r26/PACE 방식)
# ══════════════════════════════════════════════════════════════════════════
def phase_chirp(hw, ch, mass, lever, kp, alpha, q0_deg, amp_deg, f0, f1, T, log):
    """바닥(수직, 부하 최소) 중심으로 위치 정현을 f0→f1 스윕. 부하를 빼고 회귀:
         α·τ_cmd − m·g·L·cos(q−q0) = I_tot·θ̈ + b·θ̇ + Fc·sgn(θ̇)
       설계행렬 [θ̈, θ̇, sgn θ̇] 최소제곱 → I_tot, b, Fc.  I_act = I_tot − m·L².
       (관성·점성을 한 번에 — r26 처프와 같은 물리라 직접 비교 가능.)
    """
    mgl = mass * G * lever; mL2 = mass * lever ** 2; kd = 2.0
    q0 = np.deg2rad(q0_deg)
    hw.arm(ch, kp, kd)
    center = q0_deg - 90.0                       # 수직 아래(부하 최소)
    hw.goto(ch, center, kp, kd, speed_dps=12.0)

    def qcmd(t):
        f = f0 + (f1 - f0) * t / (2 * T)         # 순간주파수 선형증가
        return center + amp_deg * np.sin(2 * np.pi * f * t)
    ss = hw.run(ch, qcmd, T, kp, kd, progress="  chirp")
    hw.goto(ch, center, kp, kd, speed_dps=12.0)

    q, dq, tau, t, _ = _cols(ss)
    ddq = np.gradient(dq, t)                       # 각가속[rad/s²]
    load_free = alpha * tau - mgl * np.cos(q - q0)
    good = np.abs(dq) > np.deg2rad(8.0)            # 정지부근 제외(마찰부호 모호)
    Xd = np.column_stack([ddq[good], dq[good], np.sign(dq[good])])
    coef, *_ = np.linalg.lstsq(Xd, load_free[good], rcond=None)
    I_tot, b, Fc = float(coef[0]), float(coef[1]), float(coef[2])
    I_act = I_tot - mL2
    log(f"  → I_tot={I_tot:.4f}  I_load={mL2:.4f}  ⇒ **I_act={I_act:.4f} kg·m²**"
        f"  ·  점성 b={b:.4f} Nm·s/rad  ·  운동마찰 Fc={Fc:.3f} Nm  (n={int(good.sum())})")
    return dict(I_tot=I_tot, I_act=I_act, b=b, Fc_kin=Fc, mL2=mL2, n=int(good.sum()))


# ══════════════════════════════════════════════════════════════════════════
#  3.  I_act 교차확인  — 자유진동 (α·마찰 무관)
# ══════════════════════════════════════════════════════════════════════════
def phase_freeswing(hw, ch, mass, lever, kp, q0_deg, log):
    mgl = mass * G * lever; mL2 = mass * lever ** 2; kd = 2.0
    hw.arm(ch, kp, kd)
    bottom = q0_deg - 90.0
    hw.goto(ch, bottom + 40.0, kp, kd, speed_dps=15.0)
    time.sleep(0.4)
    hw.limp()
    t0 = time.monotonic(); ts, qs = [], []
    while time.monotonic() - t0 < 6.0:
        qs.append(hw.read(ch)[0]); ts.append(time.monotonic() - t0)
        time.sleep(0.005)
    t = np.array(ts); q = np.array(qs) - np.mean(np.array(qs)[-100:])
    zc = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
    if len(zc) >= 2:
        Tosc = float(np.median(np.diff(t[zc])))
        I_tot = mgl * (Tosc / (2 * np.pi)) ** 2
        log(f"  → 자유진동 T={Tosc:.3f}s ⇒ I_act={I_tot - mL2:.4f} kg·m²(교차확인, {len(zc)}회)")
        return dict(T=Tosc, I_act_free=I_tot - mL2, oscillated=True)
    log("  ⚠ 과감쇠(마찰↑) — 자유진동 미검출. 처프 관성만 사용.")
    return dict(oscillated=False)


# ══════════════════════════════════════════════════════════════════════════
#  5.  T-N 선도  — 알려진 관성 자유가속 (최대토크, ~보행속도까지)
# ══════════════════════════════════════════════════════════════════════════
def phase_tn(hw, ch, mass, lever, kp, q0_deg, dyn, tau_cmd, span_deg, dur, log):
    """순수토크 tau_cmd 로 **알려진 관성**(I_tot)을 자유가속시키고 θ̈(ω) 를 잰다.
         τ_out(ω) = I_tot·θ̈ + m·g·L·cosθ + Fc·sgn(ω) + b·ω     (전부 기지/측정)
       τ_out vs ω = **T-N 선도**. 평탄=전류제한(α 일정)·고속하락=역기전력/전압 제한.
       유효 α(ω)=τ_out/tau_cmd → α 의 속도의존성 검증(foot '드라이브 전류제한' 가설 직결).
       ⚠ 무게추 자유가속 → **하드스톱·여유공간·캐치 필수.** tau_cmd 를 낮은 값부터 올릴 것.
    """
    if not dyn or "I_tot" not in dyn:
        log("  ⚠ chirp 결과(I_tot) 없음 → T-N 건너뜀. --phases 에 chirp 먼저."); return dict(ok=False)
    mgl = mass * G * lever
    I_tot, b, Fc = dyn["I_tot"], dyn["b"], dyn["Fc_kin"]
    q0 = np.deg2rad(q0_deg); kd = 2.0
    hw.arm(ch, kp, kd)
    start = q0_deg - 90.0 - span_deg                    # 바닥보다 span 아래(한쪽 끝)
    hw.goto(ch, start, kp, kd, speed_dps=12.0)
    ss = hw.run_torque(ch, lambda t: tau_cmd, dur, tau_max=tau_cmd + 1.0,
                       drift_max_deg=2 * span_deg + 15.0, progress="  T-N")
    hw.brake(ch, kp, kd, 0.4)                           # 가속 후 잡기
    hw.goto(ch, q0_deg - 90.0, kp, kd, speed_dps=12.0)
    q, dq, _tau, t, _ = _cols(ss)
    ddq = np.gradient(dq, t)
    tau_out = I_tot * ddq + mgl * np.cos(q - q0) + Fc * np.sign(dq) + b * dq
    keep = dq > np.deg2rad(20.0)                        # 가속 중(정지부근 제외)
    w, Tq = dq[keep], tau_out[keep]
    if len(w) < 20:
        log("  ⚠ T-N 표본 부족 — dur/tau_cmd 키우거나 span 확대"); return dict(ok=False)
    bins = np.linspace(w.min(), w.max(), 8); idx = np.digitize(w, bins)
    curve = [(float(np.rad2deg(np.mean(w[idx == i]))), float(np.mean(Tq[idx == i])))
             for i in range(1, 8) if (idx == i).sum() > 3]
    lo, hi = curve[0][1], curve[-1][1]
    droop = (1 - hi / lo) * 100 if lo else 0.0
    log("  T-N 선도 (ω[dps] → τ_out[Nm] · α_eff):")
    for wd, tq in curve:
        log(f"      {wd:7.1f} → {tq:6.3f}   α_eff={tq / tau_cmd:.3f}")
    log(f"  → 저속 {lo:.2f} → 고속 {hi:.2f} Nm, 하락 {droop:.0f}%  "
        + ("(전류제한·α 속도무관)" if droop < 8 else "(고속 토크하락 = 역기전력/전압 제한)"))
    return dict(ok=True, curve=curve, droop_pct=droop, tau_cmd=tau_cmd)


# ══════════════════════════════════════════════════════════════════════════
#  4.  백래시 + 강성  — 양방향 토크왕복 (기존 툴, 레버 클램프)
# ══════════════════════════════════════════════════════════════════════════
def phase_backlash(hw, spec, ch, out, log):
    from act_measure_backlash import measure_backlash
    joint = next(j for j in spec["joints"] if int(j["ch"]) == ch)
    _html, res = measure_backlash(hw, spec, joint, out, log=log)
    log(f"  → 백래시 {res.get('backlash_deg')}°  ·  기어강성 {res.get('stiffness_nm_per_deg')} Nm/deg")
    return res


# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ch", type=int, required=True)
    ap.add_argument("--mass", type=float, default=2.0, help="무게추[kg]")
    ap.add_argument("--lever", type=float, default=0.20, help="레버[m]")
    ap.add_argument("--kp", type=float, default=100.0)
    ap.add_argument("--span-deg", type=float, default=60.0, help="α hold 각도범위 ±[°]")
    ap.add_argument("--n-ang", type=int, default=9, help="α hold 각도 개수")
    ap.add_argument("--chirp", default="0.3,2.0,25,12", help="f0[Hz],f1[Hz],T[s],amp[°]")
    ap.add_argument("--tn", default="5.0,40,0.5", help="T-N: tau_cmd[Nm],span[°],dur[s]")
    ap.add_argument("--q0", type=float, default=None,
                    help="수평기준각[deg]. alpha 없이 chirp/tn 만 돌릴 때 지정(안 하면 부하모델 부정확)")
    ap.add_argument("--clamped", action="store_true", help="레버 클램프(백래시용)")
    ap.add_argument("--phases", default="alpha,chirp,freeswing")
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--selftest", action="store_true", help="하드웨어 없이 로직만 검증")
    ap.add_argument("--pause", action="store_true",
                    help="각 페이즈 시작 전 Enter 대기(리그 점검·재배치용). tn(자유가속)은 이 옵션과 무관하게 항상 확인.")
    ap.add_argument("--zero", type=float, default=0.0,
                    help="각 궤적 전/후 복귀할 0점[deg]. --clamped 에서는 무시(하드스톱 충돌 방지).")
    a = ap.parse_args()

    spec = yaml.safe_load(open(a.spec, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    phases = [p.strip() for p in a.phases.split(",") if p.strip()]
    name = next((j["name"] for j in spec["joints"] if int(j["ch"]) == a.ch), f"ch{a.ch}")
    mgl = a.mass * G * a.lever
    f0, f1, T, amp = (float(x) for x in a.chirp.split(","))

    print("=" * 72)
    print(f"  구동기 전파라미터 벤치 · {name}(ch{a.ch}) · {a.mass}kg @ {a.lever}m")
    print(f"  τ_L(수평)=mgL={mgl:.3f} Nm · I_load=mL²={a.mass*a.lever**2:.4f} kg·m²")
    print(f"  예상 명령토크(수평,α0.83)≈{mgl/0.83:.2f} Nm → tau_trip 11Nm 여유 OK")
    print("=" * 72)

    if a.selftest:
        print("  [selftest] 합성데이터로 회귀 로직만 검증(하드웨어 미접속) …")
        _selftest(a.mass, a.lever)
        return 0

    hw = open_hw(spec)
    read_aux = bind_aux(hw)
    if read_aux and read_aux(a.ch) is not None:
        print("  aux(출력엔코더) 활성 — 교차확인 가능(AUX_MODE=1)")
    results = {"meta": dict(ch=a.ch, name=name, mass=a.mass, lever=a.lever, mgl=mgl)}

    def gate(header, danger=False):
        """페이즈 헤더 출력 → (Enter 대기) → **0점 복귀 후** 페이즈 진행. 중단은 Ctrl+C.
        ⚠--clamped 면 0점 복귀 생략(레버가 하드스톱에 고정돼 있어 충돌)."""
        print(header)
        if a.pause or danger:
            try:
                input("  ⏎ Enter 로 이 페이즈 진행 · Ctrl+C 로 전체 중단 …")
            except (EOFError, KeyboardInterrupt):
                raise KeyboardInterrupt
        if not a.clamped:
            goto_zero(hw, a.ch, a.kp, a.kd, a.zero)

    try:
        with hw:
            if "backlash" in phases:
                if not a.clamped:
                    print("\n[백래시] ⚠ --clamped 아님 → 건너뜀(무게추론 백래시 측정불가). "
                          "레버를 하드스톱에 고정하고 `--clamped --phases backlash` 로.")
                else:
                    gate("\n[4] 백래시·강성 (클램프, 양방향 토크왕복)")
                    results["backlash"] = phase_backlash(hw, spec, a.ch, a.out, log=print)

            free = [p for p in phases if p in ("alpha", "chirp", "freeswing")]
            if free and a.clamped:
                print("\n⚠ --clamped 인데 자유페이즈 요청 — 레버 풀고 무게추 달 것. 중단.")
                return 2

            q0 = a.q0 if a.q0 is not None else 0.0
            if "alpha" in phases:
                gate("\n[1] α + 정지마찰 (각도별 step-hold, 양방향접근)")
                r = phase_alpha(hw, a.ch, a.mass, a.lever, a.kp, a.span_deg, a.n_ang, log=print)
                results["alpha"] = r; q0 = r["q0_deg"]
            elif a.q0 is None and any(p in phases for p in ("chirp", "freeswing", "tn")):
                print("  ⚠ alpha 미실행·--q0 없음 → 수평기준각 q0=0 가정(부하모델 부정확).")
                print("    → --phases 에 alpha 포함하거나, 앞선 alpha 결과의 q0 을 --q0 로 지정할 것.")
            if "chirp" in phases:
                gate("\n[2] 점성 b + 관성 I_act (위치 처프)")
                results["chirp"] = phase_chirp(hw, a.ch, a.mass, a.lever, a.kp,
                                               results.get("alpha", {}).get("alpha", 0.83),
                                               q0, amp, f0, f1, T, log=print)
            if "freeswing" in phases:
                gate("\n[3] I_act 교차확인 (자유진동)")
                results["freeswing"] = phase_freeswing(hw, a.ch, a.mass, a.lever, a.kp, q0, log=print)
            if "tn" in phases:
                gate("\n[5] T-N 선도 (자유가속·최대토크) — ⚠하드스톱·여유공간 확인", danger=True)
                tau_cmd, span, dur = (float(x) for x in a.tn.split(","))
                results["tn"] = phase_tn(hw, a.ch, a.mass, a.lever, a.kp, q0,
                                         results.get("chirp", {}), tau_cmd, span, dur, log=print)
            # ★정상 완료 → 마지막 0점 복귀 후 종료(limp 전). clamped 면 생략(하드스톱).
            if not a.clamped:
                print("\n[종료] 0점 복귀")
                goto_zero(hw, a.ch, a.kp, a.kd, a.zero)
    except SafetyAbort as e:
        print(f"\n✗ 안전중단: {e}"); return 1
    except KeyboardInterrupt:
        print("\n중단(Ctrl+C)"); return 1
    finally:
        hw.limp()

    # 요약
    print("\n" + "=" * 72 + "\n  전파라미터 요약\n" + "=" * 72)
    al = results.get("alpha", {}); ch = results.get("chirp", {})
    fs = results.get("freeswing", {}); bl = results.get("backlash", {})
    if al: print(f"  α (토크스케일)   {al['alpha']:.3f}       [무게추 기준·직접측정]")
    if al: print(f"  Fs(정지마찰)     {al['Fs']:.3f} Nm")
    if ch: print(f"  b (점성)         {ch['b']:.4f} Nm·s/rad")
    if ch: print(f"  Fc(운동마찰)     {ch['Fc_kin']:.3f} Nm")
    if ch: print(f"  I_act(armature)  {ch['I_act']:.4f} kg·m²   [처프]"
                 + (f" · 자유진동 {fs.get('I_act_free'):.4f}" if fs.get("oscillated") else ""))
    tn = results.get("tn", {})
    if tn.get("ok"):
        print(f"  T-N 하락        {tn['droop_pct']:.0f}%  @ tau_cmd {tn['tau_cmd']:.1f}Nm"
              + ("  (전류제한·α 속도무관)" if tn['droop_pct'] < 8 else "  (역기전력 제한)"))
    if bl: print(f"  백래시 {bl.get('backlash_deg')}° · 강성 {bl.get('stiffness_nm_per_deg')} Nm/deg")

    npz = os.path.join(a.out, f"bench_full_ch{a.ch:02d}.npz")
    np.savez(npz, **{f"{k}_{kk}": vv for k, d in results.items() if isinstance(d, dict)
                     for kk, vv in d.items() if np.isscalar(vv)})
    print(f"\n  저장: {npz}")
    return 0


def _selftest(mass, lever):
    """합성 데이터로 α/처프 회귀가 심은 값을 복원하는지 확인(하드웨어 불필요)."""
    mgl = mass * G * lever; mL2 = mass * lever ** 2
    a_true, Fs_true, I_true, b_true, Fc_true = 0.83, 0.30, 0.036, 0.09, 0.60
    q0 = np.deg2rad(12.0)
    # α step-hold 합성
    q = np.deg2rad(np.linspace(-60, 60, 9))
    tau_hold = (mgl * np.cos(q - q0)) / a_true
    coef, *_ = np.linalg.lstsq(np.column_stack([np.cos(q), np.sin(q)]), tau_hold, rcond=None)
    a_rec = mgl / np.hypot(*coef)
    # 처프 합성
    t = np.linspace(0, 25, 25000)
    f = 0.3 + (2.0 - 0.3) * t / 50
    qq = np.deg2rad(-90 + 12 * np.sin(2 * np.pi * f * t)) + q0
    dq = np.gradient(qq, t); ddq = np.gradient(dq, t)
    load_free = (I_true + mL2) * ddq + b_true * dq + Fc_true * np.sign(dq)
    good = np.abs(dq) > np.deg2rad(8)
    c, *_ = np.linalg.lstsq(np.column_stack([ddq[good], dq[good], np.sign(dq[good])]),
                            load_free[good], rcond=None)
    print(f"    α  심음 {a_true:.3f} → 복원 {a_rec:.3f}   {'OK' if abs(a_rec-a_true)<0.01 else 'FAIL'}")
    print(f"    I  심음 {I_true:.3f} → 복원 {c[0]-mL2:.3f}   {'OK' if abs(c[0]-mL2-I_true)<0.005 else 'FAIL'}")
    print(f"    b  심음 {b_true:.3f} → 복원 {c[1]:.3f}   {'OK' if abs(c[1]-b_true)<0.02 else 'FAIL'}")
    print(f"    Fc 심음 {Fc_true:.3f} → 복원 {c[2]:.3f}   {'OK' if abs(c[2]-Fc_true)<0.03 else 'FAIL'}")


if __name__ == "__main__":
    sys.exit(main())
