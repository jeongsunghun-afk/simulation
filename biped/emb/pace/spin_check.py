#!/usr/bin/env python3
"""spin_check.py — 지그만(무게추 없이) 단일축 안전 스핀/스윕 sanity 체크.

  무게추를 달기 **전에**, 지그만 물린 상태로 축을 느리게 왕복시켜 다음을 본다:
    (1) 모터가 명령을 따라오나(추종오차)  (2) 지그가 어디 걸리지 않나(간섭)
    (3) 방향/부호가 맞나                  (4) 기계 바인딩(τ 급증)
  ★특성분석 아님(bench_actuator_full 은 무게추 전제). 이건 그 전 예비 확인.

  전제:
    · deploy(biped_deploy) 종료 후 — 모터 명령 writer 는 하나. (안 죽이면 거부)
    · RobotEmbedded 기동돼 SHM 있어야 함(open_hw 가 상태 수신 대기).
    · 신펌웨어면 브리지가 Enable(0x50) 전송 → 스핀 중 모터 켜짐.
    · ⚠여유공간·하드스톱 확인. 처음엔 작은 --amp 로.

  사용(.46):
    cd ~/simulation/biped/emb/pace
    python3 spin_check.py --ch 3 --amp 90 --speed 15 --cycles 2
    python3 spin_check.py --ch 3 --amp 170 --speed 20     # 거의 한바퀴(관절한계 확인)
    python3 spin_check.py --selftest                       # 하드웨어 없이 구조확인
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run(a) -> int:
    import yaml
    from bench_actuator_full import open_hw
    if subprocess.run(["pgrep", "-f", "build/biped_deploy"], capture_output=True).returncode == 0:
        print("✗ biped_deploy 가 떠 있음 — 모터 writer 는 하나. 먼저: pkill -f build/biped_deploy")
        return 2
    spec = yaml.safe_load(open(a.spec, encoding="utf-8"))
    name = next((j["name"] for j in spec.get("joints", []) if int(j.get("ch", -1)) == a.ch), f"ch{a.ch}")
    kp, kd, amp = a.kp, a.kd, abs(a.amp)
    print(f"[spin] {name}(ch{a.ch})  ±{amp}° @ {a.speed}dps · kp{kp}/kd{kd} · {a.cycles}회 (지그만·무게추X)")
    print("  ⚠여유공간·하드스톱 확인. 3초 후 시작(Ctrl+C 중단)…")
    time.sleep(3)
    hw = open_hw(spec)
    tau_pk = 0.0
    err_pk = 0.0
    try:
        c = hw.arm(a.ch, kp, kd)                      # 측정각 래치 + 게인 램프(=Enable)
        print(f"  중심 q0={c:.1f}°")
        for cyc in range(a.cycles):
            for tgt in (c + amp, c - amp, c):
                hw.goto(a.ch, tgt, kp, kd, speed_dps=a.speed)
                q, dq, tau, _ = hw.read(a.ch)
                e = abs(q - tgt)
                tau_pk = max(tau_pk, abs(tau)); err_pk = max(err_pk, e)
                print(f"  cyc{cyc} → 목표{tgt:+7.1f}°   실측 q={q:+7.1f}  τ={tau:+6.2f}  추종오차={e:5.1f}°")
        verdict = "⚠추종오차 큼 — 바인딩/과부하/한계 의심" if err_pk > 10 else "OK — 정상 추종"
        print(f"  결과: τpeak={tau_pk:.2f} Nm · 추종오차peak={err_pk:.1f}°  → {verdict}")
    except KeyboardInterrupt:
        print("\n  중단 — 안전정지")
    finally:
        try:
            hw.safe_hold()
        except Exception:
            pass
    return 0


def _selftest() -> int:
    # 하드웨어 없이 import·인자 구조만 확인
    import importlib
    importlib.import_module("bench_actuator_full")
    print("  [selftest] import(bench_actuator_full)·구조 OK — 하드웨어 없이 검증 완료")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ch", type=int, help="채널 (연결된 축)")
    ap.add_argument("--amp", type=float, default=90.0, help="스윕 진폭 ±[deg] (처음엔 작게)")
    ap.add_argument("--speed", type=float, default=15.0, help="등속 램프 속도[dps]")
    ap.add_argument("--cycles", type=int, default=2, help="왕복 횟수")
    ap.add_argument("--kp", type=float, default=40.0, help="위치게인(지그만이라 낮게)")
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.ch is None:
        ap.error("--ch 필요 (또는 --selftest)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
