#!/usr/bin/env python3
"""catch_off.py — 무경고 off 순간 포착·판정. read-only. cmd 신선도(GUI 재발행)까지 감시."""
import json, time, sys, subprocess, os
import os
STATE = os.environ.get("QUAD_STATE", "/dev/shm/biped_state.json")
CMD = os.environ.get("QUAD_CMD", "/dev/shm/biped_cmd.json"); N = 8
NM = ["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
T = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
BUF = []; MAXB = 200

def _astale_hi(a):
    try: return any((v or 0) for v in a[:N])
    except Exception: return False

def snap():
    d = json.load(open(STATE))
    h = d.get("health") or []; er = d.get("err") or []
    try: cmt = os.path.getmtime(CMD); cmd = json.load(open(CMD))
    except Exception: cmt = None; cmd = {}
    return dict(t=time.time(), mode=d.get("mode"), mon=bool(d.get("motors_on")),
                lhz=d.get("loop_hz"), estop=bool(d.get("estop") or d.get("estop_latched")),
                ereason=d.get("estop_reason"), nd=d.get("n_dead") or 0, nf=d.get("n_fault") or 0,
                astale=d.get("ack_stale"), qpf=d.get("qp_fail_pct"),
                cmd_age=(time.time()-cmt) if cmt else None, cmd_seq=cmd.get("seq"), cmd_mode=cmd.get("mode"),
                errmax=max((int(x) for x in er), default=0),
                hbad=[NM[i] for i in range(min(N, len(h))) if h[i] not in ("ok", "absent")])

print("→ 감시 %gs. '확 풀리는' 상황 재현하세요. 드롭 잡으면 판정." % T, flush=True)
prev = None; t0 = time.time(); dropped = False
while time.time() - t0 < T:
    try:
        s = snap()
    except Exception:
        time.sleep(0.03); continue
    BUF.append(s); BUF[:] = BUF[-MAXB:]
    if prev is not None:
        drop = ((prev["mon"] and not s["mon"]) or
                (prev["mode"] not in ("off", None) and s["mode"] == "off") or
                (not prev["estop"] and s["estop"]))
        if drop:
            dropped = True
            print("\n⚠⚠ DROP @ %s — 전이: mode %s→%s · mon %s→%s · estop %s→%s"
                  % (time.strftime("%H:%M:%S"), prev["mode"], s["mode"], prev["mon"], s["mon"], prev["estop"], s["estop"]), flush=True)
            print("  직전 1s (상대t·mode·mon·loop_hz·estop·cmd_age·cmd_seq·cmd_mode·errmax·dead):")
            for x in BUF[-22:]:
                print("    %+.2f %-5s mon=%d lhz=%s es=%d cmd_age=%s cmd_seq=%s cmd_mode=%s err=0x%02X nd=%d"
                      % (x["t"]-s["t"], x["mode"], x["mon"], x["lhz"], x["estop"],
                         ("%.2f"%x["cmd_age"]) if x["cmd_age"] is not None else "?", x["cmd_seq"], x["cmd_mode"], x["errmax"], x["nd"]))
            pre = BUF[-22:-1] or [prev]
            lhz_pre = [x["lhz"] for x in pre if x["lhz"]]; lhz_min = min(lhz_pre) if lhz_pre else None
            cage = [x["cmd_age"] for x in pre if x["cmd_age"] is not None]
            cage_max = max(cage) if cage else None
            seqs = [x["cmd_seq"] for x in pre if x["cmd_seq"] is not None]
            seq_frozen = (len(set(seqs)) <= 1) if len(seqs) >= 3 else False
            cmd_off = any(x["cmd_mode"] == "off" for x in pre)
            print("\n  ▶ 판정:")
            if s["estop"] or s["ereason"]:
                print("    · estop=%s reason=%s → tau_trip/estop" % (s["estop"], s["ereason"]))
            if s["errmax"] or s["nd"] or s["hbad"]:
                print("    · err=0x%02X dead=%d %s → 드라이버/MCU 폴트" % (s["errmax"], s["nd"], s["hbad"]))
            if (lhz_min and lhz_min < 300) or _astale_hi(s["astale"] or []):
                print("    · loop_hz최저=%s ack_stale=%s → deploy 루프스톨" % (lhz_min, s["astale"]))
            if cmd_off:
                print("    · **cmd 파일 mode=off 수신** → 누군가(GUI/__pub) off 를 보냄")
            elif (cage_max and cage_max > 0.45) or seq_frozen:
                print("    · **cmd 신선도 멎음**(직전 cmd_age최대=%s · seq_frozen=%s) → **명령-워치독**(GUI seq++ 재발행 멎음)"
                      % (("%.2f"%cage_max) if cage_max else "?", seq_frozen))
            if not (s["estop"] or s["ereason"] or s["errmax"] or s["nd"] or s["hbad"] or (lhz_min and lhz_min<300)
                    or cmd_off or (cage_max and cage_max>0.45) or seq_frozen):
                print("    · 신호 다 정상 → cmd 신선한데 off = deploy 내부 안전로직/기타 (deploy 로그 참조)")
            print("\n  === deploy 로그 tail ===", flush=True)
            try:
                print(subprocess.run(["tail","-10","/tmp/biped_deploy.log"], capture_output=True, text=True, timeout=5).stdout)
            except Exception as e:
                print("   (로그 못읽음: %s)" % e)
            break
    prev = s; time.sleep(0.02)
if not dropped:
    print("→ 시간초과 — 드롭 못 잡음. 재현 후 다시.")
