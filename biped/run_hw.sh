#!/usr/bin/env bash
# run_hw.sh — biped 모드 전환 + 상태 관찰 헬퍼.
#   run_deploy_hw.sh(=producer 소비자, RT 루프)가 **떠 있는 상태**에서 쓴다.
#   cmd/state JSON 을 손으로 안 만지게 감싼다.
#
#   사용:
#     ./run_hw.sh hold                # 현자세 유지(IMU-free, 크레인 내려도 됨)
#     ./run_hw.sh stand               # ★base 피드백 균형 — 크레인 남긴 채!
#     ./run_hw.sh off                 # limp 정지
#     ./run_hw.sh home|float|jog|push|walk
#     ./run_hw.sh gui                 # ★teleop GUI (모드버튼·중력지지 슬라이더·LED) — 테스트는 이걸로
#     ./run_hw.sh watch [N]           # 상태 N회(기본30) 관찰(mode·rpy·tilt·estop)
#     ./run_hw.sh status              # 1회 스냅샷
#     ./run_hw.sh enc                 # 1차(q_ch) vs 2차(aux 출력축) 엔코더 — AUX_MODE=1 deploy 필요
#
#   ⚠ 순서: 터미널A RobotEmbedded → 터미널B run_deploy_hw.sh → (여기) run_hw.sh
#   ⚠ stand/walk 는 **크레인** 받친 채. 불안정하면 즉시 `./run_hw.sh hold` 또는 off.
set -u
CMD="${QUAD_CMD:-/tmp/biped_cmd.json}"
STATE="${QUAD_STATE:-/tmp/biped_state.json}"

usage(){ echo "사용: $0 [무인자=런처: GUI+3D뷰어+값모니터] | {up|down|gui|off|hold|stand|home|float|jog|push|walk|log [file]|watch [N]|status|enc}"
         echo "  런처 옵션: NOVIEW=1(3D뷰어 끔) · MON=1|graph(그래프)·text(표·SSH)·0(끔) · MON_ARGS=..."; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"

set_mode(){
  # hold [pct] [split]:  중력지지 %(0~100) + 좌우트림 split(20~80, →HL). hold 외엔 무시됨.
  # ★deploy 워치독(500ms) 때문에 **한 번만 쓰면 off 로 되돌아간다** — GUI 처럼 seq++ 로
  #   계속 재발행해야 유지된다. 백그라운드 발행자(__pub)를 띄우고, 모드 바꿀 때 교체한다.
  local m="$1" pct="${2:-0}" split="${3:-50}"
  pkill -f "run_hw.sh __pub" 2>/dev/null; sleep 0.05
  setsid "$0" __pub "$m" "$pct" "$split" </dev/null >/dev/null 2>&1 &
  pgrep -f build/biped_deploy >/dev/null || echo "  ⚠ biped_deploy 안 떠 있음 — run_deploy_hw.sh 먼저"
  echo "→ mode=$m  중력지지=${pct}%  split=${split}(→HL)  [연속발행 중]"
}

case "${1:-launch}" in
  launch)
    # ★무인자 = 원래 실기 런처 복원 (2026-09-10). GUI(입력) + 3D 뷰어(biped_monitor) + 값 모니터.
    #   d47e017 에서 이걸 모드전환 헬퍼로 덮어썼던 걸 되살린다(서브커맨드는 그대로 유지).
    #   ⚠제어기(run_deploy_hw.sh)는 **별도**로 띄운다 — 이 런처는 표시/입력만(모터 writer 아님).
    #   옵션: NOVIEW=1 뷰어끔 · MON=1|graph 그래프(기본)·text 표(SSH)·0 끔 · MON_ARGS 옵션.
    set +u; source "$HERE/lib_display.sh"; setup_display; _drc=$?; set -u   # lib_display 는 set -e 용이라 set -u 잠시 해제
    [ "$_drc" = 0 ] || { echo "✗ 디스플레이 설정 실패 — lib_display.sh 확인 (ls /run/user/$(id -u)/.mutter-Xwaylandauth.*)"; exit 1; }
    PY_GUI=""                                    # dearpygui python (venv 우선)
    for p in "${PY:-}" "$HOME/.venv/bin/python" "$HOME/.venvs/gui/bin/python" python3; do
      [ -n "$p" ] || continue
      { command -v "$p" >/dev/null 2>&1 || [ -x "$p" ]; } || continue
      "$p" -c "import dearpygui" 2>/dev/null && { PY_GUI="$p"; break; }
    done
    [ -z "$PY_GUI" ] && { echo "✗ dearpygui python 없음 (설치: python3 -m venv --system-site-packages ~/.venv && ~/.venv/bin/pip install dearpygui)"; exit 1; }
    MJ="${BIPED_MJCF:-$HERE/biped_from_quad.mjcf}"; CFG="${BIPED_CFG:-$HERE/emb/config/biped_emb.yaml}"
    pgrep -f "build/biped_deploy|biped_emb.py" >/dev/null || echo "⚠ 제어기 안 보임 — GUI 조작해도 로봇 안 움직임. 별도 터미널에서 run_deploy_hw.sh 먼저."
    for _p in teleop_gui_biped biped_monitor monitor_state.py monitor_plot.py; do pkill -f "$_p" 2>/dev/null; done   # 중복 정리
    sleep 0.5
    [ -f "$CMD" ] || echo '{"mode":"off","seq":0,"v":0,"vy":0,"w":0,"body_h":0.42}' > "$CMD"   # 무장 방지
    VIEW=1; [ "${NOVIEW:-0}" = "1" ] && VIEW=0
    [ -x "$HERE/cpp/build/biped_monitor" ] || { echo "⚠ cpp/build/biped_monitor 없음 → 3D 뷰어 생략"; VIEW=0; }
    if [ "$VIEW" = "1" ]; then
      setsid bash -c "cd '$HERE/cpp'; STATE='$STATE' CONFIG='$CFG' DISPLAY='$DISPLAY' XAUTHORITY='$XAUTHORITY' LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-$HOME/mujoco/lib}' ./build/biped_monitor '$MJ' >/tmp/biped_monitor.log 2>&1" </dev/null &
      sleep 2; pgrep -f biped_monitor >/dev/null && echo "✅ 3D 뷰어 RUNNING(표시 전용)" || { echo "❌ 3D 뷰어 DEAD"; tail -4 /tmp/biped_monitor.log; }
    fi
    setsid bash -c "cd '$HERE'; QUAD_CMD='$CMD' QUAD_STATE='$STATE' DISPLAY='$DISPLAY' XAUTHORITY='$XAUTHORITY' '$PY_GUI' teleop_gui_biped.py >/tmp/teleop_gui_biped.log 2>&1" </dev/null &
    sleep 2; pgrep -f teleop_gui_biped >/dev/null && echo "✅ GUI RUNNING" || { echo "❌ GUI DEAD"; tail -4 /tmp/teleop_gui_biped.log; }
    case "${MON:-1}" in
      1|graph) setsid bash -c "cd '$HERE'; QUAD_STATE='$STATE' DISPLAY='$DISPLAY' XAUTHORITY='$XAUTHORITY' '$PY_GUI' monitor_plot.py ${MON_ARGS:-} >/tmp/biped_monitor_plot.log 2>&1" </dev/null &
               sleep 2; pgrep -f monitor_plot.py >/dev/null && echo "✅ 값모니터(그래프) RUNNING" || { echo "❌ 값모니터 DEAD"; tail -4 /tmp/biped_monitor_plot.log; }
               echo "종료: $0 down" ;;
      text)    echo " 값모니터(표) — Ctrl-C 로 모니터만 종료(뷰어·GUI 는 계속)"; sleep 1
               cd "$HERE" && QUAD_STATE="$STATE" exec python3 monitor_state.py ${MON_ARGS:-} ;;
      *)       echo "종료: $0 down" ;;
    esac
    ;;
  watch)
    python3 - "$STATE" "${2:-30}" <<'PY' 2>/dev/null
import json,sys,time
path,n=sys.argv[1],int(sys.argv[2])
NAMES=["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
BITS=["과전류","과전압","저전압","모터과온","MOSFET과온","ADC오프셋"]  # ucStatus bit0..5
def why(e):
    e=int(e)
    if e==0: return ""
    w=[BITS[b] for b in range(6) if e&(1<<b)]
    return "("+"·".join(w)+")" if w else "(0x%02x)"%e
for _ in range(n):
    try:
        d=json.load(open(path)); r=d['rpy_deg']
        h=d.get('health',[]); er=d.get('err',[])
        bad=[(NAMES[i] if i<len(NAMES) else "ch%d"%i, h[i], er[i] if i<len(er) else 0)
             for i in range(len(h)) if h[i] not in ("ok","absent")]
        # 정상=요약(8o/0f/0d), 이상=어느 채널이 왜(에러이름) 죽었는지
        mot=("mot %do/%df/%dd"%(d.get('n_ok',0),d.get('n_fault',0),d.get('n_dead',0))
             if not bad else "⚠ "+" ".join("%s=%s%s"%(nm,st,why(e)) for nm,st,e in bad))
        print("mode %-5s roll %+5.1f pitch %+5.1f tilt %4.1f estop %s tiltOK %s | %s"
              %(d['mode'], r[0], r[1], d['tilt_deg'], d['estop'], d['tilt_estop_ok'], mot))
    except Exception as e:
        print("state 못읽음(deploy 떠 있나?):", e)
    time.sleep(0.4)
PY
    ;;
  log)
    # 균형 신호를 타임스탬프 CSV 로 기록(GUI 로 조작하며 다른 터미널에서 실행). Ctrl+C 종료.
    OUT="${2:-/tmp/biped_hw_$(date +%Y%m%d_%H%M%S).csv}"
    echo "→ 로깅: $OUT  (mode·rpy·tilt·estop·foot_tau, 10Hz · Ctrl+C 종료)"
    echo "  (deploy 는 자체 세션로그도 /tmp/hold_session.csv 에 남긴다)"
    python3 - "$STATE" "$OUT" <<'PY'
import json,sys,time
state,out=sys.argv[1],sys.argv[2]
t0=time.time()
with open(out,'w') as f:
    f.write("t,mode,roll,pitch,yaw,tilt,estop,tiltOK,footL_tau,footR_tau\n")
    while True:
        try:
            d=json.load(open(state)); r=d['rpy_deg']; tl=d.get('tau_leg_nm',[0]*8)
            f.write("%.2f,%s,%.2f,%.2f,%.2f,%.2f,%s,%s,%.2f,%.2f\n"%(
                time.time()-t0, d['mode'], r[0],r[1],r[2], d['tilt_deg'],
                d['estop'], d['tilt_estop_ok'],
                tl[3] if len(tl)>3 else 0.0, tl[7] if len(tl)>7 else 0.0)); f.flush()
        except Exception: pass
        time.sleep(0.1)
PY
    ;;
  status)
    python3 - "$STATE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); r=d['rpy_deg']
NAMES=["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
BITS=["과전류","과전압","저전압","모터과온","MOSFET과온","ADC오프셋"]  # ucStatus bit0..5
def why(e):
    e=int(e)
    if e==0: return "-"
    w=[BITS[b] for b in range(6) if e&(1<<b)]
    return "·".join(w) if w else "0x%02x(정의밖)"%e
print("mode=%s  rpy=%s  tilt=%.1f  estop=%s  tiltOK(imu)=%s  loop_hz=%s"
      %(d['mode'],[round(x,1) for x in r],d['tilt_deg'],d['estop'],d['tilt_estop_ok'],d.get('loop_hz')))
print("모터  ok=%d fault=%d dead=%d absent=%d  (설치 %d)"
      %(d.get('n_ok',0),d.get('n_fault',0),d.get('n_dead',0),d.get('n_absent',0),d.get('n_installed',0)))
h=d.get('health',[]); er=d.get('err',[]); tl=d.get('tau_leg_nm',[]); tc=d.get('tau_cmd_nm',[])
print("  %-9s %-7s %-7s %8s %8s %s"%("ch","health","err","τ실측","τ명령","원인"))
for i in range(len(h)):
    nm=NAMES[i] if i<len(NAMES) else "ch%d"%i
    e=er[i] if i<len(er) else 0
    mk="  ⚠" if h[i] not in("ok","absent") else "   "
    print("%s%-9s %-7s %-7s %8s %8s %s"%(mk,nm,h[i],
          ("0x%02x"%int(e)) if e else "0",
          ("%.1f"%tl[i]) if i<len(tl) else "-",
          ("%.1f"%tc[i]) if i<len(tc) else "-",
          why(e) if h[i]!="ok" else ""))
PY
    ;;
  enc)
    # ★1차(모터측 q_ch) vs 2차(출력축 aux) 엔코더 비교 — deploy 를 AUX_MODE=1 로 띄웠을 때만 aux 유효.
    #   aux_deg = 채널 원시각·자기 영점(q_ch 와 영점 다름) → 절대차엔 상수 오프셋 포함.
    #   의미: (a) 무하중(float/off)↔home 사이 diff 의 **변화** = 감속단 비틀림/백래시
    #         (b) hip/thigh 는 aux=관절각(진짜 링크) · calf/foot 은 aux=벨트 앞단 → 벨트 슬립은 안 보임.
    #   ⚠AUX_MODE=1 은 **매달린 채만**(0x5A 프레임 영향 RGA 미확인). home 은 공중이라 적합.
    python3 - "$STATE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
NAMES=["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
GEAR=[1,1,1.5,1.2,1,1,1.5,1.2]   # 벨트비(채널→관절). hip/thigh 1 = 직결
q=d.get('q_ch_deg',[]); a=d.get('aux_deg'); on=d.get('aux_on')
if a is None: print("aux 키 없음 — 구 .so/mock (bridge 미지원)"); sys.exit()
if not on: print("⚠ aux_on=false — deploy 를 AUX_MODE=1 로 다시 띄울 것 (매달린 채!)")
print("mode=%s  aux_on=%s   (단위 deg · 채널 원시각)"%(d.get('mode'),on))
print("  %-9s %9s %9s %9s  %s"%("ch","1차 q_ch","2차 aux","aux-q_ch","비고"))
for i in range(min(8,len(q))):
    av=a[i] if i<len(a) else 0.0
    if av==0: note="2차 없음/미채움"
    elif GEAR[i]==1: note="aux=관절각(진짜 링크)"
    else: note="aux=벨트앞단 (÷%.1f=명목링크, 벨트슬립 안보임)"%GEAR[i]
    print("  %-9s %9.2f %9.2f %9.2f  %s"%(NAMES[i],q[i],av,av-q[i],note))
PY
    ;;
  __pub)
    # 내부용 — cmd 를 seq++ 로 100ms 마다 재발행(watchdog 만족). set_mode 가 백그라운드로 띄운다.
    m="$2"; pct="${3:-0}"; split="${4:-50}"; sq=0
    while :; do
      printf '{"mode":"%s","seq":%d,"jog_deg":[0,0,0,0,0,0,0,0],"v":0,"vy":0,"w":0,"body_h":0.42,"hold_ff_pct":%s,"hold_ff_split":%s}\n' \
             "$m" "$sq" "$pct" "$split" > "$CMD"
      sq=$((sq+1)); sleep 0.1
    done ;;
  up)
    # ★전체 스택 기동: EMB(producer) → deploy(RT루프) → GUI.  MJCF=flat|<경로> 로 모델 선택.
    echo "① EMB 기동…"; ( cd "$HERE/emb" && diag/emb_ctl.sh start ) || exit 1
    sleep 6
    pgrep -x RobotEmbedded >/dev/null || { echo "✗ EMB 기동 실패 → tail /tmp/emb.log"; exit 1; }
    echo "② deploy 기동(${MJCF:-flat})…"
    setsid bash -c "cd '$HERE'; exec bash run_deploy_hw.sh ${MJCF:-flat}" </dev/null >/tmp/biped_deploy.log 2>&1 &
    sleep 4
    pgrep -f build/biped_deploy >/dev/null || { echo "✗ deploy 기동 실패 → tail /tmp/biped_deploy.log"; exit 1; }
    grep -m1 -E "\[deploy\] IMU" /tmp/biped_deploy.log 2>/dev/null || echo "  (IMU 줄 대기 중 — tail /tmp/biped_deploy.log)"
    echo "③ GUI 기동(가능하면)…"
    "$0" gui || echo "  → GUI 없이 진행. CLI 로 조작: ./run_hw.sh home | hold | stand | watch | log" ;;
  down)
    pkill -f "run_hw.sh __pub" 2>/dev/null
    printf '{"mode":"off","jog_deg":[0,0,0,0,0,0,0,0],"v":0,"vy":0,"w":0,"body_h":0.42}\n' > "$CMD" 2>/dev/null; sleep 0.3
    for _p in teleop_gui_biped biped_monitor monitor_plot.py monitor_state.py; do pkill -f "$_p" 2>/dev/null; done
    pkill -f build/biped_deploy 2>/dev/null
    ( cd "$HERE/emb" && diag/emb_ctl.sh stop )
    echo "→ 전체 종료(GUI·3D뷰어·값모니터·deploy·EMB)" ;;
  gui)
    # teleop GUI(dearpygui) — 모드버튼·중력지지 슬라이더·LED. cmd 만 쓴다(모터 writer 아님).
    #   ⚠ dearpygui + **DISPLAY** 필요. SSH(무화면)에선 못 뜬다 → 그땐 CLI 를 쓸 것.
    GPY=""
    for p in "${PY:-}" python3 /home/rpetubt/miniforge3/envs/*/bin/python \
             /home/rpetubt/miniconda3/envs/*/bin/python /home/rpetubt/.venv/bin/python; do
      [ -n "$p" ] || continue
      { command -v "$p" >/dev/null 2>&1 || [ -x "$p" ]; } || continue
      "$p" -c "import dearpygui" 2>/dev/null && { GPY="$p"; break; }
    done
    if [ -z "$GPY" ]; then
      echo "✗ dearpygui 있는 python 없음 → GUI 불가. (설치: <python> -m pip install dearpygui)"
      echo "   → CLI 로 조작: ./run_hw.sh home | hold | stand | watch | log"; exit 1
    fi
    if [ -z "${DISPLAY:-}" ]; then
      echo "✗ DISPLAY 없음(SSH?) → dearpygui 창을 못 그림 → GUI 불가."
      echo "   → Pi 모니터 세션에서 실행, 또는 CLI: ./run_hw.sh home|hold|stand|watch|log"; exit 1
    fi
    pgrep -f build/biped_deploy >/dev/null || echo "  ⚠ biped_deploy 안 떠 있음 — run_deploy_hw.sh 먼저"
    echo "→ GUI 기동 ($GPY · DISPLAY=$DISPLAY)"
    exec env QUAD_CMD="$CMD" QUAD_STATE="$STATE" "$GPY" "$HERE/teleop_gui_biped.py" ;;
  off|hold|stand|home|float|jog|push|walk)
    case "$1" in stand|walk) echo "⚠ $1 — 크레인 받친 채인지 확인!";; esac
    set_mode "$@" ;;
  *) usage ;;
esac
