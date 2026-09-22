#!/usr/bin/env python3
"""blackbox.py — state 블랙박스 레코더(read-only). /dev/shm state를 회전 CSV로 상시 기록.
   매크로 flood 없이 디버깅 기록 확보. QUAD_STATE/QUAD_CMD/BB_OUT/BB_HZ/BB_CAP_MB env."""
import json, time, os, sys
ST = os.environ.get("QUAD_STATE", "/dev/shm/biped_state.json")
CMD = os.environ.get("QUAD_CMD", "/dev/shm/biped_cmd.json")
OUT = os.environ.get("BB_OUT", "/dev/shm/blackbox.csv")
HZ = float(os.environ.get("BB_HZ", "50"))
CAP = int(os.environ.get("BB_CAP_MB", "30")) * 1024 * 1024
NM = ["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
HDR = ("t,wall,mode,mon,loop_hz,cmd_seq,cmd_age,estop,n_dead,errmax,"
       + ",".join("q_%s" % n for n in NM) + "," + ",".join("tau_%s" % n for n in NM))
f = open(OUT, "w"); f.write(HDR + "\n"); f.flush()
print("[blackbox] %s → %s @%gHz (cap %dMB, 회전)" % (ST, OUT, HZ, CAP // 1024 // 1024), flush=True)
t0 = time.time()
while True:
    try:
        d = json.load(open(ST))
        try:
            cage = "%.2f" % (time.time() - os.path.getmtime(CMD)); seq = json.load(open(CMD)).get("seq")
        except Exception:
            cage = ""; seq = ""
        q = d.get("q_ch_deg") or [0]*8; tl = d.get("tau_leg_nm") or [0]*8
        er = d.get("err") or []; em = max((int(x) for x in er), default=0)
        row = ["%.3f" % (time.time()-t0), time.strftime("%H:%M:%S"), d.get("mode"),
               int(bool(d.get("motors_on"))), d.get("loop_hz"), seq, cage,
               int(bool(d.get("estop") or d.get("estop_latched"))), d.get("n_dead") or 0, em] \
              + ["%.2f" % (q[i] if i < len(q) else 0) for i in range(8)] \
              + ["%.2f" % (tl[i] if i < len(tl) else 0) for i in range(8)]
        f.write(",".join(str(x) for x in row) + "\n"); f.flush()
        if f.tell() > CAP:
            f.close(); os.replace(OUT, OUT + ".1"); f = open(OUT, "w"); f.write(HDR + "\n")
    except Exception:
        pass
    time.sleep(1.0 / HZ)
