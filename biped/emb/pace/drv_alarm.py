#!/usr/bin/env python3
"""drv_alarm.py — 드라이버 상태바이트(ucStatus) read-only 알람.
   deploy 가 돌아가는 채로도 안전(bridge_init+bridge_read 만, 명령 안 씀).
   사용: python3 drv_alarm.py            # 1회 스냅샷
        python3 drv_alarm.py --watch     # 변화/폴트만 실시간 출력(Ctrl-C 종료)
"""
import ctypes as C, numpy as np, sys, time

LIB = "/home/rpetubt/simulation/biped/emb/hal/libbipedshm.so"
N = 8
NAMES = ["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
BITS = ["과전류","과전압","저전압","모터과온","MOSFET과온","ADC오프셋"]  # ucStatus bit0..5

def why(s):
    s = int(s) & 0xFF
    if s == 0:
        return "-"
    w = [BITS[b] for b in range(6) if s & (1 << b)]
    hi = [b for b in (6, 7) if s & (1 << b)]
    if hi:
        w.append("보호셧다운(bit%s·전원사이클로만 클리어)" % ",".join(map(str, hi)))
    return "·".join(w) if w else "0x%02X(정의밖)" % s

lib = C.CDLL(LIB)
F32P = C.POINTER(C.c_float); I32P = C.POINTER(C.c_int)
lib.bridge_init.restype = C.c_int;  lib.bridge_init.argtypes = [C.c_int]
lib.bridge_read.restype = C.c_int;  lib.bridge_read.argtypes = [F32P]*7 + [I32P, I32P]
def _p(a): return a.ctypes.data_as(F32P)
def _ip(a): return a.ctypes.data_as(I32P)

if lib.bridge_init(300) != 0:
    print("bridge_init 실패 — RobotEmbedded 미기동이거나 halGait 초기화 미완료.")
    sys.exit(1)

bufs = [np.zeros(16, np.float32) for _ in range(7)]  # q,dq,tau,cur,+3
conn = np.zeros(16, np.int32); stt = np.zeros(16, np.int32)
def rd():
    lib.bridge_read(_p(bufs[0]), _p(bufs[1]), _p(bufs[2]), _p(bufs[3]),
                    _p(bufs[4]), _p(bufs[5]), _p(bufs[6]), _ip(conn), _ip(stt))

if "--watch" not in sys.argv:
    for _ in range(5):  # 몇 틱 읽어 안정화
        rd(); time.sleep(0.05)
    print("%-4s %-9s %5s %6s %9s  %s" % ("ch","name","conn","stt","q_ch[°]","해석"))
    bad = 0
    for i in range(N):
        s = int(stt[i]); c = int(conn[i])
        flag = "⚠" if (s != 0 or c == 0) else " "
        if s != 0 or c == 0: bad += 1
        note = why(s) if s else ("연결끊김(conn=0)" if c == 0 else "정상")
        print("%s%-4d %-9s %5d  0x%02X %9.2f  %s" % (flag, i, NAMES[i], c, s, bufs[0][i], note))
    print("→ 이상 축 %d 개" % bad)
else:
    print("[drv_alarm] watch 시작 — 상태변화/폴트만 출력(0.1s 폴링). Ctrl-C 종료.", flush=True)
    prev = None
    while True:
        rd()
        cur = tuple((int(stt[i]), int(conn[i])) for i in range(N))
        if cur != prev:
            ts = time.strftime("%H:%M:%S")
            faults = [(i, s, c) for i, (s, c) in enumerate(cur) if s != 0 or c == 0]
            if faults:
                print("\n⚠⚠ [%s] 드라이버 이상:" % ts, flush=True)
                for i, s, c in faults:
                    print("   ch%d %-9s  stt=0x%02X conn=%d  → %s" % (i, NAMES[i], s, c, why(s)), flush=True)
            else:
                print("[%s] 전 축 정상 복귀" % ts, flush=True)
            prev = cur
        time.sleep(0.1)
