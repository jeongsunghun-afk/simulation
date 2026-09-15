#!/usr/bin/env python3
"""monitor_state.py — 관절별 **측정 vs 명령** 실시간 모니터 (위치·속도·토크).

왜 터미널인가 (2026-08-13):
  Pi 4 에서 이미 MuJoCo 뷰어(biped_monitor) + dearpygui GUI + 500Hz 제어루프가 돈다.
  이 저장소엔 **루프가 밀리면 jog 속도제한이 20dps → 500dps 로 뚫린다**는 실측 기록이
  있다(app/biped_emb.py 페이싱 주석). 세 번째 무거운 GUI 는 그 위험을 키운다.
  ⇒ 표시는 stdlib + ANSI 로만 한다. 의존성 0, CPU 무시할 수준, SSH 로도 보인다.

읽는 것: /tmp/biped_state.json (QUAD_STATE) — **읽기 전용**. 아무것도 쓰지 않는다.
  제어기가 발행하는 키를 그대로 쓴다:
    q_leg_deg  dq_leg_dps  tau_leg_nm      측정 (모델각 deg · deg/s · 관절 Nm)
    q_cmd_deg  dq_cmd_dps  tau_cmd_nm      명령 (같은 단위)
    kp_raw kd_raw health installed mode loop_hz ...
  ⚠단위는 전부 **모델각**이다(채널각 아님). q_ch_deg 는 --ch 로 따로 볼 수 있다.

없는 키는 `—` 로 뜬다. 두 경우가 있다:
  · 제어기가 구버전 → 재시작하면 나온다
  · **C++ 배포(biped_deploy)는 순수 토크모드**라 위치·속도 "명령" 이 원래 없다
    (kp=kd=0). 그때 q명령/dq명령 이 비는 건 정상이고, 토크만 비교하면 된다.

사용:
  python3 monitor_state.py                 # 기본 10Hz
  python3 monitor_state.py --hz 20         # 갱신율
  python3 monitor_state.py --ch            # 채널각도 같이 표시(캘리브레이션용)
  python3 monitor_state.py --no-color
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
import time

STATE = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")
FALLBACK_NAMES = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot",
                  "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]

# ── 경고 임계 (env 로 조정) ────────────────────────────────────────────────
#   ★위치오차는 그 자체보다 **토크로 환산했을 때** 의미가 있다 — 드라이버 MIT 법칙이
#     τ = kp·err[rad] + kd·derr 라 kp 1 당 0.0175 Nm/deg 다. kp100 축에서 2° 면 3.5 Nm.
#     그래서 축마다 다른 kp 를 곱해 **토크 기여**로 판정한다. 고정 각도임계는 hip 과
#     foot 을 같은 잣대로 재게 되어 한쪽이 늘 빨갛거나 늘 조용해진다.
WARN_TAU = float(os.environ.get("MON_WARN_TAU", "3.0"))    # 위치오차의 토크환산[Nm] 경고
BAD_TAU = float(os.environ.get("MON_BAD_TAU", "8.0"))      # 〃 위험
WARN_DQ = float(os.environ.get("MON_WARN_DQ", "30"))       # 속도[dps] 경고
BAD_DQ = float(os.environ.get("MON_BAD_DQ", "120"))        # 〃 (속도트립 200 의 60%)
WARN_TAU_M = float(os.environ.get("MON_WARN_TAU_M", "8.0"))   # 측정토크[Nm] 경고
BAD_TAU_M = float(os.environ.get("MON_BAD_TAU_M", "15.0"))    # 〃 (토크트립 15 와 동일)
# ★CPU·온도 — GUI(teleop_gui_biped)와 **같은 모듈**을 쓴다. 양쪽에 복사하면 갈라진다.
_sysload, _sys_t, _sys_txt = None, [0.0], ("", 0)
try:
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sysload as _sysload
    _sys_txt = [("", 0)]
except Exception:
    _sys_txt = [("", 0)]

STALE_MS = float(os.environ.get("MON_STALE_MS", "500"))    # 이보다 오래된 상태 = 끊김
# ★추종실패 판정 — kp 가 걸려 있는데 이만큼 벌어지면 그 축은 명령을 수행하지 못한다.
#   10° 는 "정상 추종오차(수 °)" 와 "아예 안 감(수십 °)" 사이에 넉넉히 놓은 값이다.
STUCK_DEG = float(os.environ.get("MON_STUCK_DEG", "10"))   # 위치오차[deg]
STUCK_TAU = float(os.environ.get("MON_STUCK_TAU", "0.5"))  # 이보다 작으면 "토크를 안 낸다"
KP_TO_NM_PER_DEG = 0.0175                                  # [실측 2026-08-05] kp 1 = 0.0175 Nm/deg

C = {"r": "\033[31m", "y": "\033[33m", "g": "\033[32m", "d": "\033[2m",
     "b": "\033[1m", "c": "\033[36m", "x": "\033[0m", "R": "\033[41m\033[97m"}


def dw(s: str) -> int:
    """터미널 **표시폭**. 한글·기호는 2칸을 먹는다 — len() 으로 맞추면 열이 어긋난다."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, w: int, right: bool = True) -> str:
    """표시폭 기준 정렬. f-string 의 :>w 는 len() 기준이라 한글에서 깨진다."""
    g = max(0, w - dw(s))
    return (" " * g + s) if right else (s + " " * g)


def paint(s, col, on=True):
    return f"{C[col]}{s}{C['x']}" if (on and col) else s


def lvl(v, warn, bad):
    a = abs(v)
    return "r" if a >= bad else ("y" if a >= warn else "")


def fmt(v, w=7, p=2):
    return "—".rjust(w) if v is None else f"{v:+{w}.{p}f}"


def get(st, key, n):
    """길이 n 의 실수 리스트로. 없거나 길이가 모자라면 None 리스트."""
    v = st.get(key)
    if not isinstance(v, list) or len(v) < n:
        return [None] * n
    try:
        return [float(x) for x in v[:n]]
    except (TypeError, ValueError):
        return [None] * n


def load_names():
    try:
        import yaml
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "emb", "config", "biped_emb.yaml")
        return [j["name"] for j in yaml.safe_load(open(p))["joints"]]
    except Exception:
        return FALLBACK_NAMES          # config 를 못 읽어도 모니터는 떠야 한다


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=10.0, help="갱신율(기본 10)")
    ap.add_argument("--ch", action="store_true", help="채널각도 같이 표시")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--once", action="store_true",
                    help="한 프레임만 그리고 끝낸다(시험·스크립트용. 커서제어 없음)")
    a = ap.parse_args()
    col = not a.no_color and sys.stdout.isatty()
    names = load_names()
    nj = len(names)
    period = 1.0 / max(a.hz, 0.5)

    peak_dq = [0.0] * nj          # 세션 중 최대 |Δ| — 순간 스파이크는 눈으로 못 잡는다
    peak_eq = [0.0] * nj
    peak_tm = [0.0] * nj
    if not a.once:
        sys.stdout.write("\033[2J")
    try:
        while True:
            t0 = time.perf_counter()
            try:
                age_ms = (time.time() - os.path.getmtime(STATE)) * 1e3
                st = json.load(open(STATE))
            except Exception as e:
                sys.stdout.write("\033[H\033[J")
                print(paint(f"  상태파일을 못 읽는다: {STATE}", "r", col))
                print(f"  ({e})\n  제어기가 떠 있는지 확인:  pgrep -f 'biped_emb.py|biped_deploy'")
                if a.once:
                    return 1
                time.sleep(0.5)
                continue

            qm = get(st, "q_leg_deg", nj);   qc = get(st, "q_cmd_deg", nj)
            dm = get(st, "dq_leg_dps", nj);  dc = get(st, "dq_cmd_dps", nj)
            tm = get(st, "tau_leg_nm", nj);  tc = get(st, "tau_cmd_nm", nj)
            # ★kp_raw = kp_ch·gear_k² (raw 좌표). 옛 발행자는 `kp_leg` 를 냈는데 그건
            #   C++/Python 이 서로 다른 좌표였다 — 폴백은 두되 값은 못 믿는다.
            #   ⚠get() 은 없을 때 [None]*n(=truthy)을 주므로 `or` 폴백이 안 걸린다. 키로 본다.
            kp = get(st, "kp_raw" if "kp_raw" in st else "kp_leg", nj)
            kd = get(st, "kd_raw" if "kd_raw" in st else "kd_leg", nj)
            qch = get(st, "q_ch_deg", nj)
            # ★ucStatus 원값 = MD80 ERROR VECTOR 하위 8bit(벤더 확인 2026-08-14).
            #   'fault' 라는 것만 알고 왜인지 못 보던 걸 숫자로 드러낸다.
            stt = get(st, "stt_raw", nj)
            health = st.get("health") or ["?"] * nj
            inst = st.get("installed") or [True] * nj

            out = [] if a.once else ["\033[H"]
            stale = age_ms > STALE_MS
            hdr = (f" mode={st.get('mode','?'):<6} backend={st.get('backend','?'):<6} "
                   f"loop={st.get('loop_hz',0):6.1f}Hz  tilt={st.get('tilt_deg',0):5.2f}°  "
                   f"age={age_ms:6.1f}ms")
            out.append(paint(f"{hdr:<92}", "R" if stale else "b", col) + "\n")
            # ── ★CPU·온도 (2026-08-24) ─────────────────────────────────────
            #   왜 여기 있나: 500Hz 루프가 CPU·온도에 직접 물려 있다. 열로 클럭이
            #   내려가면 루프가 밀리고(실측 28~51ms 스톨), EtherCAT 동결 추적에서는
            #   "Emb 가 CPU 100% 로 살아 있었다" 가 핵심 증거였다. 상시로 보여 준다.
            #   ★계측은 sysload.py 가 한다 — GUI 와 **같은 코드**다(복사하지 않는다).
            #   ⚠1초에 한 번만 갱신한다. 표는 훨씬 빨리 도는데 매 프레임 /proc 을
            #     훑으면 모니터가 제 몫의 CPU 를 먹어 측정 대상을 오염시킨다.
            if _sysload is not None:
                if time.time() - _sys_t[0] >= 1.0:
                    _sys_t[0] = time.time()
                    try: _sys_txt[0] = _sysload.line()
                    except Exception: _sys_txt[0] = ("", 0)
                _t, _sev = _sys_txt[0]
                if _t:
                    # ★패딩은 dw() 로 — '온도'·'제어기' 같은 한글이 2칸을 먹어서
                    #   f-string 의 len() 패딩으로는 열이 어긋난다(이 파일이 dw 를 둔 이유).
                    out.append(paint(" " + _t + " " * max(0, 91 - dw(_t)),
                                     ("d", "y", "R")[_sev], col) + "\n")
            flags = []
            if stale:
                # ★"제어기" 가 Emb 로 오해받았다(2026-08-20). 둘은 다른 프로세스다:
                #   Emb(RobotEmbedded)  = SHM ↔ EtherCAT 중계. **상태파일을 안 쓴다.**
                #   제어기(biped_emb.py / biped_deploy) = 이 파일을 발행하는 쪽.
                #   ⚠제어기가 죽어도 모터는 안 멈춘다 — Emb 가 SHM 의 마지막 명령을
                #     1kHz 로 계속 재전송한다. 그래서 종료 시 kp=kd=τ=0 을 써야 한다.
                flags.append(paint(
                    f"*** STALE {age_ms:.0f}ms — 상태파일이 안 갱신된다 "
                    f"(Emb 아님. biped_emb.py / biped_deploy 확인) ***", "R", col))
            if st.get("estop_latched") or st.get("estop"):
                flags.append(paint(f"E-STOP 래치: {st.get('estop_reason','?')}", "R", col))
            if st.get("wd_trip"):
                flags.append(paint("워치독 트립", "r", col))
            if st.get("write_fail"):
                flags.append(paint(f"write_fail={st['write_fail']}", "r", col))
            if st.get("tilt_estop_ok") is False:
                flags.append(paint("IMU 죽음 → tilt E-stop 무력", "y", col))
            # ★QP 건강도 = **접지 판정**. 발이 덜 닿으면 QP 가 매 틱 실패하고 중력보상
            #   폴백으로 떨어지는데 겉보기엔 안정돼 보인다(매달림 실측 95% vs 접지 0.05%).
            _qf = st.get("qp_fail_pct")
            if _qf is not None:
                _ce = st.get("qp_cerr") or [0, 0, 0]
                _cn = (sum(v * v for v in _ce)) ** 0.5 * 1e3
                _t = (f"QP실패 {_qf:.0f}% · K={st.get('qp_K','?')} · com_err {_cn:.0f}mm")
                flags.append(paint(_t + ("  ← 접지 불량(폐루프 죽음)" if _qf >= 50 else ""),
                                   "R" if _qf >= 50 else ("y" if _qf >= 20 else "g"), col))
            # ★지연보상 — **꺼져 있어도 겉보기엔 똑같이 돈다.** 그래서 표시한다.
            #   9일간 "켜져 있다" 고 믿은 채로 실제로는 꺼져 있었던 적이 있다(RUNBOOK §6-c).
            _lc = st.get("lat_comp_ms")
            if _lc is not None:
                _sk = st.get("lc_skip_pct", 0.0)
                if _lc <= 0:
                    flags.append(paint("지연보상 OFF", "y", col))
                elif _sk >= 1:
                    flags.append(paint(f"지연보상 {_lc:.1f}ms · 폴백 {_sk:.0f}%  ← 예측 버려지는 중",
                                       "R" if _sk >= 20 else "y", col))
                else:
                    flags.append(paint(f"지연보상 {_lc:.1f}ms", "g", col))
            # ★위치모드 강성 배율(GUI 조절). `pos_kp_scale`=지금 나가는 값 · `_target`=요구값.
            #   둘이 다르면 **아직 램프 중**이다 — 그 사이 토크가 변하므로 보여야 한다.
            _ks, _kt = st.get("pos_kp_scale"), st.get("pos_kp_target")
            if _ks is not None and (abs(_ks - 1.0) > 1e-3 or (_kt is not None and abs(_kt - 1.0) > 1e-3)):
                if _kt is not None and abs(_kt - _ks) > 1e-3:
                    flags.append(paint(f"강성 kp×{_ks:.2f}→×{_kt:.2f} 램프중", "y", col))
                else:
                    flags.append(paint(f"강성 kp×{_ks:.2f}", "y" if _ks > 3.0 else "g", col))
            # ★★**명령했는데 안 따라오는 축**을 직접 잡는다 (2026-08-20).
            #   실기에서 HL_foot 이 명령 −59.8° 인데 +23.0° 에 머물고 토크가 0.03Nm 였는데
            #   health=ok · err=0 이라 모니터가 **아무 경고도 안 냈다.** 엔코더가 살아 있으면
            #   상태보고로는 안 드러난다 — "명령 vs 측정" 으로만 보인다.
            #   ⚠kp 가 0 이면 판정하지 않는다(순수 토크모드에선 위치오차가 정상이다).
            _bad = []
            for _i in range(nj):
                if not (kp[_i] and kp[_i] > 1):
                    continue
                if qm[_i] is None or qc[_i] is None:
                    continue
                _e = abs(qm[_i] - qc[_i])
                if _e < STUCK_DEG:
                    continue
                _t = abs(tm[_i]) if tm[_i] is not None else None
                _why = ("토크 0 — 구동 안 됨" if (_t is not None and _t < STUCK_TAU)
                        else "토크는 나오는데 안 움직임 — 기구 걸림")
                _bad.append(f"{names[_i]} Δ{_e:.0f}° ({_why})")
            if _bad:
                flags.append(paint("⛔ 추종실패: " + " · ".join(_bad), "R", col))
            out.append(("  " + "   ".join(flags) if flags else "  " + paint("이상 없음", "g", col)) + "\n")

            # ★★터미널 폭에 맞춰 열을 고른다 (2026-08-20).
            #   종전은 14열 고정 90+자라 좁은 창에서 줄이 접히며 **숫자와 이름이 어긋났다**
            #   — 무슨 값인지 못 읽는다. 폭이 모자라면 파생값(Δ·환산)부터 버린다.
            #   버리는 순서에 이유가 있다: Δ 와 ≈Nm 은 **원값에서 다시 계산할 수 있다**.
            #   원값(측정·명령)과 판단에 직접 쓰는 것(kp·kd·상태)은 끝까지 남긴다.
            term_w = shutil.get_terminal_size((100, 40)).columns
            COLS = [                       # (키, 머리말, 폭, 소수, 필수)
                ("name", "축",      10, None, True),
                ("qm",   "q측정",    8, 2, True),
                ("qc",   "q명령",    8, 2, True),
                ("eq",   "Δq",       7, 2, False),
                ("enm",  "≈Nm",      6, 1, False),
                ("dm",   "dq측정",   8, 1, True),
                ("ed",   "Δdq",      7, 1, False),
                ("tm",   "τ측정",    8, 2, True),
                ("tc",   "τ명령",    8, 2, True),
                ("et",   "Δτ",       7, 2, False),
                ("kp",   "kp",       6, 1, True),
                ("kd",   "kd",       5, 1, True),
                ("qch",  "q_ch",     8, 1, False),
                ("st",   "상태",     7, None, True),
            ]
            if not a.ch:
                COLS = [c2 for c2 in COLS if c2[0] != "qch"]
            def _fits(cs): return 2 + sum(c2[2] for c2 in cs)
            use = list(COLS)
            for key in ("Δτ", "Δdq", "≈Nm", "q_ch", "Δq"):     # 버리는 순서
                if _fits(use) <= term_w: break
                use = [c2 for c2 in use if c2[1] != key]
            UNITS = {"name": "", "qm": "deg", "qc": "deg", "eq": "deg", "enm": "Nm",
                     "dm": "dps", "ed": "dps", "tm": "Nm", "tc": "Nm", "et": "Nm",
                     "kp": "Nm/r", "kd": "-", "qch": "deg", "st": "health"}
            out.append("\n" + paint(
                "  " + "".join(pad(c2[1], c2[2], c2[0] != "name") for c2 in use) + "\n", "c", col))
            out.append(paint(
                "  " + "".join(pad(UNITS[c2[0]], c2[2], c2[0] != "name") for c2 in use) + "\n",
                "d", col))
            out.append("  " + "─" * min(term_w - 2, _fits(use)) + "\n")

            for i in range(nj):
                eq = None if (qm[i] is None or qc[i] is None) else qm[i] - qc[i]
                ed = None if (dm[i] is None or dc[i] is None) else dm[i] - dc[i]
                et = None if (tm[i] is None or tc[i] is None) else tm[i] - tc[i]
                # 위치오차 → 토크환산(축별 kp 반영). kp 가 없으면 환산 불가.
                enm = None if (eq is None or kp[i] is None) else eq * kp[i] * KP_TO_NM_PER_DEG
                if eq is not None:
                    peak_eq[i] = max(peak_eq[i], abs(eq))
                if dm[i] is not None:
                    peak_dq[i] = max(peak_dq[i], abs(dm[i]))
                if tm[i] is not None:
                    peak_tm[i] = max(peak_tm[i], abs(tm[i]))
                c_eq = lvl(enm, WARN_TAU, BAD_TAU) if enm is not None else ""
                c_dm = lvl(dm[i], WARN_DQ, BAD_DQ) if dm[i] is not None else ""
                c_tm = lvl(tm[i], WARN_TAU_M, BAD_TAU_M) if tm[i] is not None else ""
                h = str(health[i]) if i < len(health) else "?"
                c_h = {"ok": "g", "fault": "r", "dead": "r", "absent": "d"}.get(h, "y")
                nm = names[i] if (i >= len(inst) or inst[i]) else paint(names[i], "d", col)
                ev = "" if stt[i] is None else f"0x{int(stt[i]):02X}"
                c_ev = "r" if (stt[i] and int(stt[i])) else "d"
                vals = {"qm": (qm[i], None), "qc": (qc[i], None), "eq": (eq, c_eq),
                        "enm": (enm, c_eq), "dm": (dm[i], c_dm), "ed": (ed, None),
                        "tm": (tm[i], c_tm), "tc": (tc[i], None), "et": (et, None),
                        "kp": (kp[i], None), "kd": (kd[i], None), "qch": (qch[i], None)}
                cells = []
                for key, _lbl, w, dec, _req in use:
                    if key == "name":
                        cells.append(pad(nm, w, False))
                    elif key == "st":
                        raw = h[:1] + (" " + ev if ev else "")
                        body = paint(h[:1], c_h, col) + (" " + paint(ev, c_ev, col) if ev else "")
                        cells.append(" " * max(0, w - dw(raw)) + body)
                    else:
                        v, cc = vals[key]
                        txt = fmt(v, w, dec)
                        cells.append(paint(txt, cc, col) if cc else txt)
                out.append("  " + "".join(cells) + "\n")

            # ★★자세유지 토크: hold(위치제어) vs stand(WBIC) — 2026-08-21 사용자 요청.
            #   묻고 있는 것: **같은 자세를 버티는 데 두 제어기가 내는 토크가 얼마나 다른가.**
            #   hold 는 "이 자세를 지키는 데 실제로 필요한 토크" 를 실측으로 알려준다
            #   (위치제어라 모델을 안 쓴다). stand 는 WBIC 가 **모델로 계산한** 토크다.
            #   ⇒ Δ 가 곧 **모델 오차**다. 음수면 stand 가 덜 내고 있고, 그만큼 처진다.
            #   ⚠비어 있으면(`[]`) 아직 안 잡힌 것이다 — 0 Nm 이 아니다.
            th, ts_ = st.get("tau_hold_nm") or [], st.get("tau_stand_nm") or []
            if th:
                W = 9
                out.append("\n  " + paint("자세유지 τ[Nm] │ hold=위치제어 실측 · stand=WBIC 계산 · "
                                          "Δ<0 이면 stand 가 덜 낸다", "c", col) + "\n")
                out.append("  " + pad("", 8, False)
                           + "".join(pad(names[i], W, True) for i in range(nj)) + "\n")
                def _trow(lbl, vec, colr=None):
                    cs = "".join(fmt(vec[i] if i < len(vec) else None, W, 2) for i in range(nj))
                    return "  " + paint(pad(lbl, 8, False), "d", col) + (paint(cs, colr, col) if colr else cs) + "\n"
                out.append(_trow("hold", th))
                if ts_:
                    out.append(_trow("stand", ts_))
                    dv = [ts_[i] - th[i] if (i < len(ts_) and i < len(th)) else None for i in range(nj)]
                    worst = max((abs(v) for v in dv if v is not None), default=0.0)
                    out.append(_trow("Δ", dv, lvl(worst, 0.5, 1.5)))
                else:
                    out.append("  " + paint(pad("stand", 8, False), "d", col)
                               + paint("아직 안 잡혔다 — stand 로 전환하고 블렌드+1.5s 기다릴 것", "d", col) + "\n")

            # ★1차(q_ch) vs 2차(aux) 출력엔코더 — AUX_MODE=1(0x5A) 수신 시만. (2026-09-15)
            #   hip/thigh: aux=진짜 링크각 · calf/foot: aux=벨트 앞단(감속단만·벨트유격 안보임).
            aux = get(st, "aux_deg", nj)
            if st.get("aux_on") and any((x is not None) and abs(x) > 1e-9 for x in aux):
                Wa = 9
                out.append("\n  " + paint("1차 q_ch vs 2차 aux[°] │ Δ=aux−q_ch (감속단 비틀림+2차 영점/스케일) · "
                                          "calf/foot 은 감속단만", "c", col) + "\n")
                out.append("  " + pad("", 8, False)
                           + "".join(pad(names[i], Wa, True) for i in range(nj)) + "\n")
                def _arow(lbl, vec, colr=None):
                    cs = "".join(fmt(vec[i] if i < len(vec) else None, Wa, 2) for i in range(nj))
                    return "  " + paint(pad(lbl, 8, False), "d", col) + (paint(cs, colr, col) if colr else cs) + "\n"
                dqa = [aux[i] - qch[i] if (i < len(aux) and i < len(qch) and aux[i] is not None and qch[i] is not None) else None
                       for i in range(nj)]
                out.append(_arow("q_ch", qch))
                out.append(_arow("aux", aux))
                _wa = max((abs(v) for v in dqa if v is not None), default=0.0)
                out.append(_arow("Δ aux-q", dqa, lvl(_wa, 1.0, 3.0)))

            out.append("\n  " + paint("세션 최대 │ ", "d", col)
                       + paint(f"|Δq| {max(peak_eq):5.2f}°  |dq| {max(peak_dq):6.1f}dps  "
                               f"|τ| {max(peak_tm):5.2f}Nm", "d", col) + "\n")
            out.append(paint("  Ctrl-C 종료 · 읽기전용(아무것도 쓰지 않는다) · 단위=모델각\n", "d", col))
            if not a.once:
                out.append("\033[J")
            # ★잔상 제거 (2026-08-25 실기에서 당함). \033[H 는 커서만 홈으로 보내고
            #   화면을 안 지운다 — 이전 프레임이 더 길었으면(모드 전환·경고줄·자세유지 블록)
            #   그 아랫부분이 그대로 남아 "HR_foot 이 두 줄" 같은 유령 행이 보였다.
            #   ⇒ 각 줄 끝에 \033[K(줄 끝까지 지움), 프레임 끝에 \033[J(화면 끝까지 지움).
            #   \033[2J(전체 클리어)를 매 프레임 쓰지 않는 이유: 깜빡임이 생긴다.
            frame = "".join(out)
            if not a.once:
                frame = frame.replace("\n", "\033[K\n") + "\033[J"
            sys.stdout.write(frame)
            sys.stdout.flush()
            if a.once:
                return 0
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
