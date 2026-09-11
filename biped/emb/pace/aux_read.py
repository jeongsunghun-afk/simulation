#!/usr/bin/env python3
"""aux_read.py — 모터 disable(자유)로 두고 1차(q_ch)·2차(aux) 를 같이 읽는다 (손스핀 유격/백래시용).

  MOT_BASE_MODE 로 브리지가 보낼 ucMode 를 정한다:
    0x4A = DIS_MOT_WITH_AUX_ENC (모터 끔=자유 + aux)  ← 손으로 돌려 출력엔코더 읽기
    0x5A = ENA_MOT_WITH_AUX_ENC (켬=유지 + aux)
  kp=kd=0 으로 계속 쓰므로(=명령토크 0), 0x4A 면 모터는 자유. 그동안 q(1차)·aux(2차)를 읽는다.

  전제: RobotEmbedded 가 relay/master 로 떠 있어야 함(우리는 SHM 명령만 줌 — 모터 writer 중복 아님).
  사용(.46):
    cd ~/simulation/biped/emb/pace
    MOT_BASE_MODE=0x4A python3 aux_read.py --ch 2      # 자유+aux, 손으로 ch2 돌리며 q·aux 관찰
"""
from __future__ import annotations
import os, sys, time, yaml
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def main() -> int:
    from bench_actuator_full import open_hw, bind_aux
    ch = 2
    if "--ch" in sys.argv:
        ch = int(sys.argv[sys.argv.index("--ch") + 1])
    mode = os.environ.get("MOT_BASE_MODE", "(미설정→기본0x50)")
    hw = open_hw(yaml.safe_load(open(os.path.join(HERE, "spec.yaml"), encoding="utf-8")))
    ra = bind_aux(hw)
    if ra is None:
        print("⚠ bind_aux None (.so 미지원) — aux 못 읽음");
    q0 = hw.read(ch)[0]
    print("[aux_read] ch%d  MOT_BASE_MODE=%s  (0x4A=disable+aux=자유) — ch%d 를 손으로 돌리세요. Ctrl+C 종료."
          % (ch, mode, ch))
    print("  %-10s %-10s %-8s" % ("q(1차)", "aux(2차)", "aux−q(유격)"))
    try:
        while True:
            s = hw.step(ch, q0, 0.0, 0.0)              # kp=kd=0 + mode=MOT_BASE_MODE → 0x4A 면 자유
            a = ra(ch) if ra else None
            aux = a[0] if a else None
            print("  %-10.2f %-10s %-8s" % (
                s.q_deg,
                ("%.2f" % aux) if aux is not None else "0/None",
                ("%+.2f" % (aux - s.q_deg)) if aux is not None else "-"))
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        try: hw.safe_hold()
        except Exception: pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
