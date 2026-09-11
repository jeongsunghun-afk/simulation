#!/usr/bin/env python3
"""act_measure_backlash.py — 토션 히스테리시스로 백래시·컴플라이언스 분리 측정.

★이 시험이 가능해진 경위 (2026-08-05):
  아침에 백래시를 못 쟀던 근본 이유는 **토크가 독립 입력이 아니었기** 때문이다.
  위치+게인 모드에서 드라이버가 보고하는 tau 는 `Kp·err` 로 위치에서 계산되는 값이라,
  "토크가 낮은 구간 = 유격" 이라는 판정이 "위치오차가 작은 구간 = 위치오차가 작은 구간"
  이라는 동어반복이 된다. 실제로 그 순환논리로 틀린 결론을 냈다.
  그 뒤 **순수 토크모드(Kp=Kd=0, fTorque)가 동작하고 크기도 ±0.005 Nm 로 정확함**을
  확인했다 → 토크가 독립 입력이 되었고, 표준 토션 히스테리시스 시험이 성립한다.

원리 — 관절이 **회전하지 않는 범위**(기동토크 이하)에서 토크를 왕복시키면
q-τ 평면에 히스테리시스 루프가 그려진다. 그 모양으로 두 성분이 갈린다:

    백래시        : 강성이 ~0 인 **평탄 구간**. 토크는 거의 안 변하는데 q 가 움직인다.
                    (기어 이빨이 반대면에 닿기 전까지 무저항으로 통과)
    마찰 히스테리시스: 기울기가 같은 두 분기의 **평행 이동**. q 가 같아도 τ 가 다르다.
    컴플라이언스   : 물린 구간의 기울기 dτ/dq (드라이브트레인 탄성)

⇒ 국소강성 |dτ/dq| 이 물린 구간 대비 임계 이하로 떨어지는 구간의 **폭이 백래시**다.
  평탄 구간이 없으면 백래시는 유의미하지 않고 루프 폭은 마찰 히스테리시스다.

⚠ 안전: τ_max 를 **실측 기동토크보다 확실히 낮게** 둔다(회전 금지). 기동 최저값이
  0.528 Nm 였으므로 기본 0.45 Nm. 회전이 감지되면(누적 이동이 탄성범위 초과) 즉시 중단.
  다리 미장착 상태가 이 시험에 가장 안전하다.
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template

TEMPLATE = Template("""
<h2>{{ title }}</h2>
<p>순수 토크모드(Kp=Kd=0)로 <b>회전하지 않는 범위</b>에서 토크를 왕복시켜 q-τ 히스테리시스
루프를 얻고, <b>국소강성이 떨어지는 평탄 구간</b>을 백래시로, 물린 구간 기울기를
컴플라이언스로 분리한다.</p>

<table>
  <tr><th colspan="2">시험 조건</th></tr>
  <tr><td>일시</td><td>{{ datetime }}</td></tr>
  <tr><td>축</td><td>{{ joint }} (SHM ch{{ ch }})</td></tr>
  <tr><td>토크 왕복</td><td class="numeric">±{{ '%.2f' % tau_max }} Nm @ {{ '%.2f' % ramp }} Nm/s × {{ cycles }}주기</td></tr>
  <tr><td>기동토크(참조)</td><td class="numeric">{{ '%.2f' % tau_break }} Nm — 이보다 낮게 유지해 회전 방지</td></tr>

  <tr><th colspan="2">결과</th></tr>
  <tr><td><b>백래시</b> (평탄구간 폭)</td><td class="numeric">{{ lash_str }}</td></tr>
  <tr><td><b>컴플라이언스</b> (물린구간 강성)</td><td class="numeric">{{ stiff_str }}</td></tr>
  <tr><td>루프 전체 폭 (@τ=0)</td><td class="numeric">{{ '%0.4f' % loop_w }} deg
      <span class="dim">= 백래시 + 마찰 히스테리시스</span></td></tr>
  <tr><td>총 변형 범위</td><td class="numeric">{{ '%0.4f' % q_span }} deg</td></tr>
  <tr><td>회전 발생</td><td class="numeric">{{ '예 (무효)' if rotated else '아니오 (정상)' }}</td></tr>
</table>

{{ warnings }}

<p><img src="{{ plot }}"></p>
""")


def _analyze(q, tau, tau_max, log):
    """q-τ 루프에서 백래시(평탄구간)·컴플라이언스(물린구간 기울기) 분리."""
    # 물린 구간 = |τ| 가 큰 영역. 여기서 기준 강성을 구한다.
    eng = np.abs(tau) > 0.55 * tau_max
    if eng.sum() < 20:
        return None, None, float("nan"), "물린 구간 샘플 부족"
    # 상·하 분기를 나눠 각각 선형적합 → 기울기 평균 = 컴플라이언스 강성
    dtau = np.gradient(tau)
    up, dn = eng & (dtau > 0), eng & (dtau < 0)
    slopes = []
    for m in (up, dn):
        if m.sum() >= 10:
            A = np.polyfit(q[m], tau[m], 1)
            if A[0] > 0:
                slopes.append(A[0])
    if not slopes:
        return None, None, float("nan"), "물린 구간 기울기 추정 실패"
    k_deg = float(np.mean(slopes))                     # Nm/deg

    # 국소강성: 토크 변화 대비 위치 변화. 평탄 구간 = 강성이 물린구간의 일부 이하
    win = 15
    ks = []
    for i in range(len(q)):
        a, b = max(0, i - win), min(len(q), i + win)
        if b - a < 8: ks.append(np.nan); continue
        dq = q[b-1] - q[a]
        ks.append(abs((tau[b-1] - tau[a]) / dq) if abs(dq) > 1e-6 else np.inf)
    ks = np.array(ks, float)
    flat = (ks < 0.25 * k_deg) & (np.abs(tau) < 0.5 * tau_max)   # 저강성 + 영교차 부근
    lash = float(q[flat].ptp()) if flat.sum() >= 8 else None
    return lash, k_deg, float(flat.sum()), None


def per_cycle_estimates(q, tau, tau_max, seg_bounds, warmup, log):
    """★주기별 추정 + **워밍업 제외** + 반쪽구간 제외.

    2026-08-05 실측이 드러낸 두 가지:
      (a) 첫 상승램프·마지막 하강램프는 **반쪽 구간**이라 온전한 루프가 없다
          → 강성이 3.84 / 2.63 처럼 엉뚱하게 나온다. 경계를 피크로 추측하던 초기
          구현의 문제였고, 이제 가진 시점에 기록한 seg_bounds 를 쓴다.
      (b) 강성이 주기마다 **단조 증가**한다(8.53 → 10.56 → 11.27 → 11.48).
          기계적 안정화(프리로드·물림 자리잡기)라 **첫 주기들은 워밍업으로 버려야** 한다.
    이 둘을 안 빼면 산포가 44~49% 로 벌어져 좌우 비교 자체가 불가능해진다.
    """
    # segs 구조: [0]=상승램프, 그 뒤 (down,up) 쌍 × cycles, 마지막=하강램프
    cyc = []
    for i in range(1, len(seg_bounds) - 2, 2):        # (down,up) 쌍 = 온전한 1주기
        a_, b_ = seg_bounds[i], seg_bounds[i + 2] if i + 2 < len(seg_bounds) else len(q)
        if b_ - a_ >= 60:
            cyc.append((a_, b_))
    used = cyc[warmup:] if len(cyc) > warmup else cyc
    ks, ls = [], []
    for a_, b_ in used:
        lash, k, _, err = _analyze(q[a_:b_], tau[a_:b_], tau_max, log)
        if k: ks.append(k)
        if lash is not None: ls.append(lash)
    return ks, ls, len(cyc), len(used)


def _unused_peak_segmentation(q, tau, tau_max, log):
    """★주기별로 따로 추정해 **산포**를 낸다.

    2026-08-05: 점추정만 내던 초기 구현은 같은 축을 두 번 재면 강성이 8.2 → 11.9 (+44%)
    로 튀었고, 좌우 대소관계까지 뒤집혔다. 즉 **반복 산포가 좌우 차이보다 컸다.**
    산포를 함께 내지 않으면 그 사실이 드러나지 않아 실재하지 않는 좌우 차이를
    보고하게 된다. 주기 경계는 tau 가 +tau_max 를 찍는 지점으로 잡는다.
    """
    peaks = [i for i in range(1, len(tau) - 1)
             if tau[i] > 0.9 * tau_max and tau[i] >= tau[i-1] and tau[i] > tau[i+1]]
    bounds = [0] + peaks + [len(tau)]
    ks, ls = [], []
    for a_, b_ in zip(bounds[:-1], bounds[1:]):
        if b_ - a_ < 60:
            continue
        lash, k, _, err = _analyze(q[a_:b_], tau[a_:b_], tau_max, log)
        if k: ks.append(k)
        if lash is not None: ls.append(lash)
    return ks, ls


def stiffness_vs_threshold(q, tau, tau_max, fracs=(0.4, 0.55, 0.7, 0.85)):
    """★강성 추정이 '물린 구간' 임계에 얼마나 민감한지 — 분석 인공물 판별용.

    유격이 큰 축은 전이구간이 물린 구간에 섞여 기울기를 끌어내릴 수 있다. 그러면
    '유격 크다 → 강성 낮다' 는 **가짜 상관**이 생긴다. 임계를 올려도 값이 안정적이면
    실제 차이, 임계에 따라 크게 변하면 인공물이다.
    """
    out = {}
    dtau = np.gradient(tau)
    for f in fracs:
        eng = np.abs(tau) > f * tau_max
        sl = []
        for m in (eng & (dtau > 0), eng & (dtau < 0)):
            if m.sum() >= 10:
                A = np.polyfit(q[m], tau[m], 1)
                if A[0] > 0: sl.append(A[0])
        out[f] = float(np.mean(sl)) if sl else float("nan")
    return out


def measure_backlash(hw, spec, joint, plotdir, log=print) -> tuple[str, dict]:
    ch = int(joint["ch"])
    name = joint["name"]
    cfg = spec.get("backlash", {})
    tau_max = float(cfg.get("tau_max_nm", 0.45))
    ramp = float(cfg.get("ramp_nm_per_s", 0.12))
    cycles = int(cfg.get("cycles", 2))
    tau_break = float(cfg.get("tau_break_ref_nm", 0.62))
    q_elastic_max = float(cfg.get("q_elastic_max_deg", 1.5))
    warn: list[str] = []

    if tau_max >= 0.85 * tau_break:
        warn.append(f"τ_max({tau_max}) 가 기동토크({tau_break})에 너무 가깝다 — 회전 위험")
    log(f"  [{name}] 토션 히스테리시스 — ±{tau_max} Nm @ {ramp} Nm/s × {cycles}주기")
    log(f"           (기동 {tau_break} Nm 이하로 유지 → 회전 없이 탄성·유격만 본다)")

    # 토크 파형: 0 → +max → −max → +max → … → 0
    segs = [(0.0, tau_max)]
    for _ in range(cycles):
        segs += [(tau_max, -tau_max), (-tau_max, tau_max)]
    segs += [(tau_max, 0.0)]

    hw.arm(ch, 0.0, 0.0)
    time.sleep(0.3)
    q0 = hw.read(ch)[0]
    # ★aux(2차 출력축 엔코더, 벨트 전단계) 동시 캡처 — q(1차,모터측)−aux = 감속기 유격,
    #   aux 이동폭 = 클램프 견고성(출력이 진짜 고정됐나) 검증.
    import ctypes as _C
    _rd_aux = None
    try:
        hw.lib.bridge_aux.restype = _C.c_int
        hw.lib.bridge_aux.argtypes = [_C.POINTER(_C.c_float), _C.POINTER(_C.c_float)]
        _ap = (_C.c_float * hw.n)(); _av = (_C.c_float * hw.n)()
        def _rd_aux():
            hw.lib.bridge_aux(_ap, _av); return float(_ap[ch])
        aux0 = _rd_aux()
    except Exception:
        aux0 = 0.0
    qs, ts, tc_all, auxs, rotated = [], [], [], [], False
    seg_bounds = []                                  # ★경계를 추측하지 말고 가진 시점에 기록
    for a, b in segs:
        seg_bounds.append(len(qs))
        T = abs(b - a) / ramp
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= T: break
            tc = a + (b - a) * (t / T)
            s = hw.step_torque(ch, tc, tau_max)
            qs.append(s.q_deg - q0); ts.append(s.tau); tc_all.append(tc)
            auxs.append((_rd_aux() - aux0) if _rd_aux else float("nan"))
            if abs(s.q_deg - q0) > q_elastic_max:      # 탄성범위 초과 = 회전 시작
                rotated = True; break
            time.sleep(hw.dt)
        if rotated: break
    hw.limp()

    q = np.array(qs); tau = np.array(ts); tcmd = np.array(tc_all); aux = np.array(auxs)
    if rotated:
        warn.append(f"<b>관절이 회전했다</b>(이동 {abs(q).max():.2f}° > {q_elastic_max}°) — "
                    f"τ_max 를 낮춰 재시험할 것. 이 결과는 무효다.")
        lash = k_deg = None; nflat = 0
    else:
        lash, k_deg, nflat, err = _analyze(q, tau, tau_max, log)
        if err: warn.append(err)

    # 루프 전체 폭(@τ≈0) — 백래시+마찰 히스테리시스 합
    near0 = np.abs(tau) < 0.08 * tau_max
    loop_w = float(q[near0].ptp()) if near0.sum() > 4 else float("nan")

    if lash is not None:
        log(f"  [{name}] → 백래시 {lash:.4f}° · 강성 {k_deg:.3f} Nm/deg "
            f"({k_deg*180/np.pi:.0f} Nm/rad) · 루프폭 {loop_w:.4f}°")
        if lash < 0.02:
            warn.append(f"백래시가 측정 노이즈 수준({lash:.4f}°) — 유의미한 유격 없음으로 볼 것")
    else:
        log(f"  [{name}] → 평탄 구간 미검출. 루프폭 {loop_w:.4f}° (= 마찰 히스테리시스로 해석)")
        warn.append("저강성 평탄 구간이 검출되지 않았다 → <b>유의미한 백래시 없음</b>. "
                    "루프 폭은 마찰 히스테리시스로 해석해야 한다.")

    # ── 플롯 ────────────────────────────────────────────────────────────────
    p = f"{plotdir}/backlash_ch{ch:02d}.png"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(q, tau, lw=.9)
    ax[0].axhline(0, c="k", lw=.5); ax[0].axvline(0, c="k", lw=.5)
    ax[0].set_xlabel("deflection (deg)"), ax[0].set_ylabel("measured torque (Nm)")
    ax[0].set_title("torsional hysteresis loop"), ax[0].grid(alpha=.3)
    ax[1].plot(tcmd, tau, lw=.8, label="measured")
    ax[1].plot([-tau_max, tau_max], [-tau_max, tau_max], "k:", lw=.8, label="ideal")
    ax[1].set_xlabel("commanded torque (Nm)"), ax[1].set_ylabel("measured (Nm)")
    ax[1].set_title("torque command fidelity"), ax[1].legend(), ax[1].grid(alpha=.3)
    fig.suptitle(f"Backlash / compliance — {name}")
    plt.savefig(p, dpi=110, bbox_inches="tight"), plt.close()

    wh = ('<div class="warn"><b>주의</b><ul>' + "".join(f"<li>{w}</li>" for w in warn)
          + "</ul></div>") if warn else ""
    html = TEMPLATE.render(
        title=f"Backlash & Compliance — {name}", datetime=time.strftime("%Y-%m-%d %H:%M:%S"),
        joint=name, ch=ch, tau_max=tau_max, ramp=ramp, cycles=cycles, tau_break=tau_break,
        lash_str=(f"{lash:.4f} deg" if lash is not None else "평탄구간 미검출 → 유의미한 백래시 없음"),
        stiff_str=(f"{k_deg:.3f} Nm/deg ({k_deg*180/np.pi:.0f} Nm/rad)" if k_deg else "—"),
        loop_w=loop_w, q_span=float(q.ptp()) if q.size else 0.0, rotated=rotated,
        warnings=wh, plot=p.replace(plotdir, "plots"))
    # ★원시데이터 저장 — 임계 민감도 등 사후분석용(하드웨어 재구동 없이)
    npz = f"{plotdir}/../backlash_raw_ch{ch:02d}.npz"
    np.savez(npz, q=q, tau=tau, tau_cmd=tcmd, tau_max=tau_max, ch=ch, name=name)
    ksens = stiffness_vs_threshold(q, tau, tau_max) if not rotated else {}
    warmup = int(cfg.get("warmup_cycles", 2))
    if not rotated:
        ks_cyc, ls_cyc, n_all, n_used = per_cycle_estimates(q, tau, tau_max, seg_bounds, warmup, log)
        log(f"    주기 {n_all}개 중 워밍업 {n_all-n_used}개 제외 → {n_used}개 사용")
    else:
        ks_cyc, ls_cyc = [], []
    if len(ks_cyc) >= 2:
        km, ksd = float(np.mean(ks_cyc)), float(np.std(ks_cyc))
        log(f"    주기별 강성 {len(ks_cyc)}개: " + " ".join(f"{v:.2f}" for v in ks_cyc)
            + f"  → {km:.2f} ± {ksd:.2f} Nm/deg (산포 {100*ksd/max(km,1e-9):.0f}%)")
        if ls_cyc:
            lm, lsd = float(np.mean(ls_cyc)), float(np.std(ls_cyc))
            log(f"    주기별 백래시 {len(ls_cyc)}개: " + " ".join(f"{v:.4f}" for v in ls_cyc)
                + f"  → {lm:.4f} ± {lsd:.4f}°")
        if ksd > 0.15 * km:
            warn.append(f"주기간 강성 산포가 {100*ksd/km:.0f}% 다 — 점추정을 좌우 비교에 쓰면 안 된다. "
                        f"주기를 늘려 평균±산포로 볼 것.")
    if ksens:
        log("    임계 민감도(물린구간 |tau|> f·tau_max 일 때 강성 Nm/deg):")
        log("      " + " · ".join(f"f={f}: {v:.2f}" for f, v in ksens.items()))
    # ── aux(2차) 교차: 클램프 견고성(2차 이동폭) + 감속기 유격(q−aux) ──
    res_aux = {}
    if aux.size == q.size and np.isfinite(aux).all() and not rotated:
        aux_pp = float(aux.ptp())
        qa = q - aux                                     # 1차−2차 = 감속기 구간 유격+비틀림
        qa_lash, qa_k, _n, _e = _analyze(qa, tau, tau_max, lambda *a: None)
        log(f"    [aux] 2차(출력축) 이동폭 {aux_pp:.4f}° (클램프 견고성 — 작을수록 출력이 잘 고정된 것)")
        log(f"          q−aux(감속기 유격) 백래시 {('%.4f°' % qa_lash) if qa_lash is not None else '미검출(유의미한 유격 없음)'}"
            + (f" · 감속기강성 {qa_k:.2f} Nm/deg" if qa_k else ""))
        res_aux = dict(aux_ptp_deg=aux_pp, backlash_q_minus_aux_deg=qa_lash,
                       stiffness_q_minus_aux=qa_k)
    return html, {"backlash_deg": lash, "stiffness_nm_per_deg": k_deg, "loop_width_deg": loop_w,
                  "rotated": rotated, "ch": ch, "name": name, "k_vs_threshold": ksens,
                  "npz": npz, **res_aux}
