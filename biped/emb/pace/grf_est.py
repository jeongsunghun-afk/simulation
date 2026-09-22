#!/usr/bin/env python3
"""grf_est.py v2 — 자세무관 HL/HR 수직 GRF.
   fCurrent(state SHM cur_a)×kt → 모델프레임 τ_meas.  MuJoCo qfrc_bias = 자세별 중력토크.
   τ_ext = τ_meas/SCALE − τ_grav(현 자세) → F=Jᵀ⁺τ_ext → F_z = GRF.  ⇒ **다리무게 자세무관 보상.**
   ★hold(mode→hold) 누르면 재영점: 그 자세(발 빈)에서 SCALE(fCurrent 스케일) 자동보정 + 잔차 0.
   → /dev/shm/grf.json 기록 + 출력.  실행: /home/rpetubt/.venv-mujoco/bin/python grf_est.py"""
import ctypes as C, numpy as np, sys, os, time, json
import mujoco
LIB = "/home/rpetubt/simulation/biped/emb/hal/libbipedshm.so"
MJCF = "/home/rpetubt/simulation/biped/biped_flatfoot.mjcf"
STATE = os.environ.get("QUAD_STATE", "/dev/shm/biped_state.json")
GRF_OUT = os.environ.get("GRF_OUT", "/dev/shm/grf.json")
N = 8
NM = ["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
SIGN = np.array([-1,1,-1,-1,-1,-1,1,1], float)
GEAR = np.array([7,7,10.5,8.4,7,7,10.5,8.4], float)
KT = 0.2

m = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(m)
jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_joint") for n in NM]
qadr = [int(m.jnt_qposadr[j]) for j in jid]; dadr = [int(m.jnt_dofadr[j]) for j in jid]
sph = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f) for f in ['HL_sphere','HR_sphere']]
fbody = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ['HL_foot_contact_link','HR_foot_contact_link']]
nv = m.nv

lib = C.CDLL(LIB); F = C.POINTER(C.c_float); I = C.POINTER(C.c_int)
lib.bridge_init.restype = C.c_int; lib.bridge_init.argtypes = [C.c_int]
lib.bridge_read.restype = C.c_int; lib.bridge_read.argtypes = [F]*7 + [I, I]
def p(a): return a.ctypes.data_as(F)
def ip(a): return a.ctypes.data_as(I)
if lib.bridge_init(800) != 0: print("bridge_init 실패 — RobotEmbedded?"); sys.exit(1)
b = [np.zeros(16, np.float32) for _ in range(7)]; conn = np.zeros(16, np.int32); stt = np.zeros(16, np.int32)
def read_cur():
    lib.bridge_read(p(b[0]),p(b[1]),p(b[2]),p(b[3]),p(b[4]),p(b[5]),p(b[6]), ip(conn), ip(stt))
    return b[3][:N].copy()
def fjac(k):
    jacp = np.zeros((3, nv)); mujoco.mj_jac(m, d, jacp, None, d.geom_xpos[sph[k]], fbody[k]); return jacp

def sample():
    cur = read_cur()
    try:
        s = json.load(open(STATE)); q = s.get("q_leg_deg") or [0.0]*N; mode = s.get("mode")
    except Exception:
        q = [0.0]*N; mode = None
    for i in range(N): d.qpos[qadr[i]] = np.deg2rad(q[i])
    mujoco.mj_forward(m, d)
    tau_meas = SIGN * cur * KT * GEAR                                # 스케일된 측정토크
    tau_grav = np.array([d.qfrc_bias[dadr[i]] for i in range(N)])    # 자세별 중력토크(실단위)
    return tau_meas, tau_grav, mode

def leg_Fz(tau):
    out = [0.0, 0.0]
    for leg in range(2):
        cols = [dadr[leg*4 + k] for k in range(4)]
        J = fjac(leg)[:, cols]
        Fsol, *_ = np.linalg.lstsq(J.T, tau[leg*4:leg*4+4], rcond=None)
        out[leg] = float(Fsol[2])
    return out

SCALE = 1.0; resid = [0.0, 0.0]; ema = [0.0, 0.0]; A = 0.35
def calibrate(dur=1.3):
    """발 빈 정지자세에서 SCALE(=fCurrent 스케일)·잔차 보정. τ_meas ≈ SCALE·τ_grav."""
    global SCALE
    TM, TG = [], []
    t0 = time.time()
    while time.time()-t0 < dur:
        tm, tg, _ = sample(); TM.append(tm); TG.append(tg); time.sleep(0.03)
    TM = np.mean(TM, axis=0); TG = np.mean(TG, axis=0)
    mask = np.abs(TG) > 0.5                                          # 중력 유의미한 관절만
    if mask.any():
        SCALE = float(np.median(np.abs(TM[mask]) / np.abs(TG[mask])))
        SCALE = min(max(SCALE, 0.05), 20.0)                          # 가드
    tau_ext = TM/SCALE - TG
    resid[:] = leg_Fz(tau_ext)
    ema[:] = [0.0, 0.0]
    print("■ 재영점: SCALE=%.2f (fCurrent 스케일) · 잔차 HL=%.1f HR=%.1f N (발 빈 가정)" % (SCALE, resid[0], resid[1]), flush=True)

print("GRF v2 — 시작 자동보정(발 빈 가정). 이후 **hold 누르면 재영점**. Ctrl-C 종료.", flush=True)
calibrate()
prev_mode = None
while True:
    tau_meas, tau_grav, mode = sample()
    if mode == "hold" and prev_mode != "hold":                      # ★hold 누름 → 재영점
        print("(hold 감지 → 재영점 중…)", flush=True); calibrate()
    prev_mode = mode
    tau_ext = tau_meas/SCALE - tau_grav                             # 자세별 중력 제거(자세무관)
    Fz = leg_Fz(tau_ext)
    hl, hr = Fz[0]-resid[0], Fz[1]-resid[1]
    ema[0] += A*(hl-ema[0]); ema[1] += A*(hr-ema[1])
    try:
        with open(GRF_OUT+".tmp", "w") as gf:
            json.dump({"HL_GRF": round(ema[0],1), "HR_GRF": round(ema[1],1),
                       "sum": round(ema[0]+ema[1],1), "scale": round(SCALE,2), "t": time.time()}, gf)
        os.replace(GRF_OUT+".tmp", GRF_OUT)
    except Exception: pass
    print("HL_GRF=%+7.1f  HR_GRF=%+7.1f  N (합 %+.1f · 자세무관·SCALE%.2f)" % (ema[0], ema[1], ema[0]+ema[1], SCALE), flush=True)
    time.sleep(0.12)
