#!/usr/bin/env python3
"""biped_float_gravcheck.py — float(무중력) 발 들림 진단. **읽기전용**(deploy 병행 안전).
   float 모드에서 적용 feedforward(tau_cmd_nm) vs 모델중력 G_model(q) vs 실측토크(cur_a) 를
   관절별로 대조 → 어느 축이 과보상(적용>중력)이라 drift 하는지 국소화.

   float 은 kp=0 이라 적용토크가 실제중력보다 크면 평형점이 없어 그 방향으로 계속 돈다.
   ⇒ 발이 들리면 발목(foot) tau_cmd 가 실제 중력보다 큰 것. 그 크기를 여기서 본다.

   사용: (float 모드로 둔 상태에서)  python3 biped_float_gravcheck.py [N샘플=200]
"""
import json, os, sys, time, numpy as np, mujoco

MJCF  = "/home/rpetubt/simulation/biped/biped_flatfoot.mjcf"
STATE = os.environ.get("QUAD_STATE", "/dev/shm/biped_state.json")
BASE  = "/tmp/grf_verify_base.json"
NM    = ['HL_hip','HL_thigh','HL_calf','HL_foot','HR_hip','HR_thigh','HR_calf','HR_foot']
N     = 8
SIGN  = np.array([-1,1,-1,-1,-1,-1,1,1], float)
GEAR  = np.array([7,7,10.5,8.4,7,7,10.5,8.4], float); KT = 0.2
NS    = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SCALE = 6.4
if os.path.exists(BASE):
    try: SCALE = float(json.load(open(BASE))["SCALE"])
    except Exception: pass

# N샘플 median 수집
Q=[]; TC=[]; CUR=[]; mode=None
for _ in range(NS):
    try:
        st = json.load(open(STATE)); mode = st.get("mode")
        q = st.get("q_leg_deg") or [0.0]*N; tc = st.get("tau_cmd_nm") or [0.0]*N
        cu = st.get("cur_a") or [0.0]*N
        if len(q)>=N and len(tc)>=N and len(cu)>=N:
            Q.append(q[:N]); TC.append(tc[:N]); CUR.append(cu[:N])
    except Exception: pass
    time.sleep(0.005)
if len(Q) < 10:
    print("state 수집 실패 — deploy·QUAD_STATE 확인"); sys.exit(1)
q = np.median(np.array(Q,float), axis=0)
tau_cmd = np.median(np.array(TC,float), axis=0)      # 적용 feedforward[Nm] (float=중력보상)
cur = np.median(np.array(CUR,float), axis=0)         # 실측 전류[A]
tau_real = SIGN*cur*KT*GEAR/SCALE                    # 실측 관절토크[Nm]

# 모델 중력 G(q) = qfrc_bias @ qvel=0
m = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(m)
jid  = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n+"_joint") for n in NM]
qadr = [int(m.jnt_qposadr[j]) for j in jid]; dadr = [int(m.jnt_dofadr[j]) for j in jid]
for i in range(N): d.qpos[qadr[i]] = np.deg2rad(q[i])
for i in range(m.nv): d.qvel[i] = 0.0
mujoco.mj_forward(m, d)
G = np.array([d.qfrc_bias[dadr[i]] for i in range(N)])  # 모델 중력토크[Nm]

if mode != "float":
    print("⚠ 현재 mode=%s — float 에서 재실행해야 의미 있음(무중력 발들림 진단)" % mode)
print("■ float 중력보상 진단 (SCALE=%.2f · %d샘플 median · mode=%s)" % (SCALE, len(Q), mode))
print("%-9s %7s %9s %9s %8s | %8s %9s %s" % ("관절","q°","적용τ_cmd","모델G","비(cmd/G)","cur[A]","실측τ","판정"))
print("-"*82)
for i in range(N):
    r = tau_cmd[i]/G[i] if abs(G[i])>0.05 else float("nan")
    over = tau_cmd[i]-G[i]                             # >0 = 적용>중력 → 그 방향 drift
    flag = ""
    if abs(G[i])<0.15 and abs(tau_cmd[i])>0.2: flag="⚠과보상?(중력≈0인데 적용큼)"
    elif not np.isnan(r) and (r>1.4 or r<0.6):  flag="⚠비 벗어남"
    print("%-9s %7.1f %9.3f %9.3f %8s | %8.3f %9.3f  %+.2f %s"
          % (NM[i], q[i], tau_cmd[i], G[i], ("%.2f"%r if not np.isnan(r) else "  n/a"),
             cur[i], tau_real[i], over, flag))
print("-"*82)
print("→ '적용τ_cmd'가 '모델G'보다 크면(비>1) 그 축을 그 방향으로 민다. 발목(foot)은 중력이 작아")
print("  적용이 조금만 커도 평형이 없어 발이 들린다 — foot 행의 적용τ_cmd·비를 볼 것.")
print("  실측τ(cur유도)와 적용τ_cmd가 비슷하면 전류루프가 명령대로 realize 중(SCALE 정합).")
