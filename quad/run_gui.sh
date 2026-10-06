#!/usr/bin/env bash
# 02_Leg 17-DOF GUI 텔레옵 원샷 런처 (C++ 뷰어 + dearpygui GUI)
#   사용: bash run_gui.sh [map]        map = course(기본)|flat(16-DOF)|flat17|stairs|rough|friction|gap|stepping|soft
#   기본 맵 = 종합코스(마찰→험지→계단, perceptive 자동 ON)
#
# ★견고화(한 번에 확실히 실행): ①이전 인스턴스 SIGTERM 후 "실제 종료까지 폴링"(SIGKILL은 GL/X
#   리소스를 안 풀어 다음 실행이 컨텍스트를 못 얻는 간헐 크래시 유발 → 부드럽게 종료+대기).
#   ②뷰어는 "state 파일이 실제로 갱신(렌더 루프 생존)"될 때까지 검증하고, 실패 시 자동 재시도.
#   ③뷰어가 확인된 뒤에만 GUI 실행. ④pgrep은 bash wrapper까지 잡으므로 실제 바이너리/렌더로 판정.
# ★CMDFILE/STATE_PUB 필수(누락 시 GUI 명령 무시하고 저절로 전진) — 이 스크립트가 항상 붙임.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # simulation/quad
CPP="$HERE/cpp"
PXI="${PXI:-/home/jsh/miniforge3/envs/proxddp/bin/python}"
[ -x "$PXI" ] || PXI="$(command -v python3)"                   # conda env 없는 PC(노트북 WSL) → 시스템 python3(dearpygui 필요)
export DISPLAY="${DISPLAY:-:0}"
# ★WSL(노트북 RTX4050, 2026-10-06 실측): ①Mesa d3d12 기본 어댑터면 MuJoCo 창이 검정(픽셀 0)·소프트웨어 GL 은 1fps
#   → NVIDIA 어댑터 지정. ②화면전송(스왑)이 프레임당 ~90ms → 7fps·프레임당 60스텝 상한 탓에 sim 0.36배속
#   → 그림자·반사 끔(VIEW_LITE, 스왑 31ms)+따라잡기 상한 200(VIEW_MAXSTEP) = ~17fps·실시간 0.99. WSL 아니면 안 건드림.
if grep -qi microsoft /proc/version 2>/dev/null; then
  export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
  export VIEW_LITE="${VIEW_LITE:-1}" VIEW_MAXSTEP="${VIEW_MAXSTEP:-200}"
fi
CMD=/tmp/quad_cmd.json; STATE=/tmp/quad_state.json
# ★HWSIM=1 — 실기 유사 조건 묶음(2026-10-06, tools/plant/qrun.sh 와 동일 + 플랜트): 추정기 폐루프·지연 센서4+구동6ms·
#   엔코더/자이로 잡음·로터 반사관성(GEARBOX) + 플랜트(드라이버 PD 추종 kp 100/50/180/43.2·kd×1.5 · FF 0.75·LPF10·노치15).
#   각 값은 앞에 env 로 덮어쓰기. MIT 게인 시험: HWSIM=1 WBIC_MIT=1 SW_KP=500 SW_KD=10 bash run_gui.sh flat
if [ "${HWSIM:-0}" = 1 ]; then
  export EST_CTRL="${EST_CTRL:-1}" SENSE_LAT_MS="${SENSE_LAT_MS:-4}" ACT_LAT_MS="${ACT_LAT_MS:-6}"
  export ENCQ_N="${ENCQ_N:-7.6e-5}" ENCDQ_N="${ENCDQ_N:-0.037}" GYRO_N="${GYRO_N:-0.002}" GEARBOX="${GEARBOX:-1}"
  export PLANT="${PLANT:-1}" WALK_TRACK="${WALK_TRACK:-1}" TRK_KD="${TRK_KD:-1.5}" WALK_FF_SCALE="${WALK_FF_SCALE:-0.75}"
  export WALK_FF_LPF_HZ="${WALK_FF_LPF_HZ:-10}" WALK_FF_NOTCH_HZ="${WALK_FF_NOTCH_HZ:-15}" WALK_FF_NOTCH_Q="${WALK_FF_NOTCH_Q:-4}"
fi

case "${1:-course}" in
  course)   MJCF=mjcf/quad_terrain_course.mjcf ;;
  flat)     MJCF=mjcf/quad_real_16dof_sphere.mjcf ;;         # ★2026-10-06 실물 허리 제거 → 16-DOF 기본
  flat17)   MJCF=mjcf/quad_real_17dof_waist_sphere.mjcf ;;   # 구 허리 모델(지형 맵들은 아직 이 모델 include)
  stairs)   MJCF=mjcf/quad_terrain_stairs.mjcf ;;
  rough)    MJCF=mjcf/quad_terrain_rough.mjcf ;;
  friction) MJCF=mjcf/quad_terrain_friction.mjcf ;;
  gap)      MJCF=mjcf/quad_terrain_gap.mjcf ;;
  stepping) MJCF=mjcf/quad_terrain_stepping.mjcf ;;
  soft)     MJCF=mjcf/quad_terrain_soft.mjcf ;;
  *)        MJCF="$1" ;;                                       # 임의 mjcf 경로 허용(mjcf/ 포함해 전달)
esac

# ── ① 이전 인스턴스 부드럽게 종료 후 실제 종료까지 폴링(최대 5초, 안 죽으면 SIGKILL) ──
pkill -TERM -f 'build/trot_view'   2>/dev/null
pkill -TERM -f teleop_gui_17dof.py 2>/dev/null
for _ in $(seq 1 25); do
  pgrep -f 'build/trot_view' >/dev/null || pgrep -f teleop_gui_17dof.py >/dev/null || break
  sleep 0.2
done
pkill -9 -f 'build/trot_view'   2>/dev/null   # 잔존 시에만
pkill -9 -f teleop_gui_17dof.py 2>/dev/null
sleep 0.5; rm -f "$CMD" "$STATE"

# 실제 뷰어 바이너리 실행 중인가(bash wrapper 제외)
viewer_alive(){ pgrep -f 'build/trot_view ' >/dev/null; }
# 렌더 루프 생존(state mtime 갱신) 확인
viewer_rendering(){
  local a b; a=$(stat -c %Y.%N "$STATE" 2>/dev/null) || return 1
  sleep 0.8; b=$(stat -c %Y.%N "$STATE" 2>/dev/null) || return 1
  [ "$a" != "$b" ]
}

# ── ② 뷰어: 렌더 확인될 때까지 최대 4회 재시도 ──
VOK=0
for try in 1 2 3 4; do
  rm -f "$STATE"
  setsid bash -c "cd '$CPP'; env RATE=1.0 CMDFILE='$CMD' STATE_PUB='$STATE' \
    ./build/trot_view '../$MJCF' > /tmp/trot_view.log 2>&1" </dev/null &
  # ★렌더(STATE 갱신) 최대 ~11초 폴링(구 3초 단발은 warmup+GL init 느린 머신서 오탐 킬)
  for w in $(seq 1 6); do
    sleep 1
    if ! viewer_alive; then echo "viewer 조기종료(try $try) — 로그:"; tail -4 /tmp/trot_view.log 2>/dev/null; break; fi
    if viewer_rendering; then VOK=1; break; fi
  done
  [ $VOK = 1 ] && { echo "viewer RUNNING (try $try)"; break; }
  echo "viewer 재시도 $try…"; pkill -9 -f 'build/trot_view' 2>/dev/null; sleep 1
done
if [ $VOK = 0 ]; then echo "viewer DEAD (4회 실패)"; tail -5 /tmp/trot_view.log 2>/dev/null; exit 1; fi

# ── ③ 뷰어 확인 후 GUI ──
setsid bash -c "cd '$HERE'; DISPLAY='$DISPLAY' '$PXI' teleop_gui_17dof.py > /tmp/teleop_gui.log 2>&1" </dev/null &
sleep 4

echo "map=$MJCF"
pgrep -f teleop_gui_17dof.py >/dev/null && echo "gui RUNNING" || { echo "gui DEAD"; tail -5 /tmp/teleop_gui.log; }
echo "✅ 준비 완료 — 뷰어+GUI 실행 중"
