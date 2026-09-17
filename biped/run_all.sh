#!/usr/bin/env bash
# run_all.sh — biped 실기 스택 통합 런처 (Emb·deploy·GUI 기동/종료). 2026-09-17
#   터미널1(제어): ./run_all.sh ctrl     # Emb + deploy (deploy 로그 포그라운드)
#   터미널2(GUI):  ./run_all.sh gui      # 전체화면 GUI
#   종료:          ./run_all.sh down     # GUI→deploy→Emb 역순
#   상태:          ./run_all.sh status
#   개별:          ./run_all.sh emb | deploy
#
# 채널은 두 서브커맨드 다 /dev/shm(RAM) 통일 — SD I/O 격리(홍수 사고 예방).
# 실험 토글은 앞에 붙여 실행:  AUX_MODE=0 ./run_all.sh ctrl   ·   JOG_SPEED_DPS=80 ./run_all.sh ctrl
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 공통 채널 = RAM(/dev/shm) ──
export QUAD_CMD="${QUAD_CMD:-/dev/shm/biped_cmd.json}"
export QUAD_STATE="${QUAD_STATE:-/dev/shm/biped_state.json}"
# ── 실험 토글 (기본값) ──
export AUX_MODE="${AUX_MODE:-1}"      # 2차엔코더 로깅(매달림). 떨림/0x5A 의심 시 AUX_MODE=0 으로.
MJCF="${MJCF:-flat}"                  # flat=2점평발 · point=1점점발
# JOG_SPEED_DPS 는 **설정된 경우에만** deploy 로 넘어감(스윙 고대역 전용, [5,150]). 평시 미설정.

_emb_up(){ pgrep -f "app/biped_emb|RobotEmbedded" >/dev/null; }

start_emb(){
  echo "① Emb 기동… (halGait 초기화 ≈5s)"
  ( cd "$HERE/emb" && diag/emb_ctl.sh start ) || { echo "✗ Emb 기동 실패 — tail /tmp/emb.log"; return 1; }
}
start_deploy(){
  _emb_up || echo "⚠ Emb 안 떠 있음 — 먼저 ./run_all.sh emb (deploy 는 Emb SHM 을 읽는다)"
  echo "② deploy 기동(${MJCF}) — 중복 자동정리 · 포그라운드(Ctrl+C=deploy 종료)"
  [ -n "${JOG_SPEED_DPS:-}" ] && echo "   ⚠JOG_SPEED_DPS=${JOG_SPEED_DPS} (스윙 고대역 — 실험 후 빼고 재기동)"
  cd "$HERE" && exec bash run_deploy_hw.sh "$MJCF"    # exec = 이 셸이 곧 deploy
}
start_gui(){
  export DISPLAY="${DISPLAY:-:0}"
  if [ -z "${XAUTHORITY:-}" ]; then
    XAUTHORITY="$(ls -t /run/user/$(id -u)/.mutter-Xwaylandauth.* 2>/dev/null | head -1)"
    [ -z "$XAUTHORITY" ] && XAUTHORITY="$HOME/.Xauthority"
    export XAUTHORITY
  fi
  echo "③ GUI 기동 — DISPLAY=$DISPLAY · XAUTHORITY=$XAUTHORITY"
  _emb_up || echo "⚠ 제어기(Emb/deploy) 안 보임 — GUI 조작해도 로봇 안 움직임"
  cd "$HERE" && exec ./run_hw.sh gui
}
stop_all(){
  echo "종료 (GUI→deploy→Emb 역순)…"
  pkill -f "[t]eleop_gui_biped"   2>/dev/null && echo "  ✓ GUI 종료"
  pkill -f "build/[b]iped_deploy" 2>/dev/null && echo "  ✓ deploy 종료"
  ( cd "$HERE/emb" && diag/emb_ctl.sh stop )   && echo "  ✓ Emb 종료(EtherCAT 정리)"
}
status(){
  _emb_up && echo "Emb    ✅" || echo "Emb    ✗"
  pgrep -f "build/biped_deploy" >/dev/null && echo "deploy ✅" || echo "deploy ✗"
  pgrep -f "teleop_gui_biped"   >/dev/null && echo "GUI    ✅" || echo "GUI    ✗"
  python3 - <<'PY' 2>/dev/null || echo "state ✗ (/dev/shm/biped_state.json 없음)"
import json; d=json.load(open("/dev/shm/biped_state.json"))
print("state  ✅  mode=%s · loop_hz=%s · cur_a=%s" % (d.get('mode'), d.get('loop_hz'), (d.get('cur_a') or [None])[:1]!=[None]))
PY
}

case "${1:-help}" in
  emb)       start_emb ;;
  deploy)    start_deploy ;;
  ctrl|up)   start_emb && sleep 1 && start_deploy ;;   # 터미널1
  gui)       start_gui ;;                              # 터미널2
  down|stop) stop_all ;;
  status)    status ;;
  *) echo "사용: $0 {ctrl|gui|down|status}   개별: {emb|deploy}"
     echo "  터미널1: ./run_all.sh ctrl   ·   터미널2: ./run_all.sh gui   ·   종료: ./run_all.sh down"
     echo "  토글:    AUX_MODE=0 ./run_all.sh ctrl   ·   JOG_SPEED_DPS=80 ./run_all.sh ctrl" ;;
esac
