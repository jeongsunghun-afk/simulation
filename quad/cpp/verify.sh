#!/usr/bin/env bash
# ── 컨트롤러 품질 게이트 (헤드리스 회귀 배터리) ─────────────────────────────
#   빌드 → 평지(walk/trot/run) + 지형(course, perceptive) 표준 검증.
#   판정: falls=0 (하드) · max_tilt ≤ 한계 · 전진거리 x ≥ 최소.  전부 PASS면 exit 0.
#   사용: ./verify.sh            (C++ 회귀, ~10초)
#         ./verify.sh --python   (+ Python 파리티 스팟체크, 느림)
#   ★임계값은 정상 동작 여유값(회귀=falls 발생/tilt 급증만 잡음). 물리 캡 아님.
set -uo pipefail
cd "$(dirname "$0")"                       # simulation/quad/cpp
MJ=../mjcf/quad_real_16dof_sphere.mjcf   # ★2026-10-06 실물 허리 제거 → 16-DOF
COURSE=../mjcf/quad_terrain_verify.mjcf     # ★회귀검증=순차 직진코스(마찰→험지→계단). 병렬 course는 GUI용(갭/스테핑=실패허용)
# foot 감속비=8.4:1 실값(MJCF actuatorfrcrange=100.8, gear[3]=8.4). 재기어 불요
PY=${PY:-/home/jsh/miniforge3/envs/proxddp/bin/python}
DOPY=0; [ "${1:-}" = "--python" ] && DOPY=1

echo "▶ 빌드(trot_sim)…"
cmake --build build --target trot_sim -j >/dev/null 2>&1 || { echo "❌ BUILD FAIL"; exit 1; }

pass=0; fail=0
_num(){ echo "$1" | grep -oE "$2" | grep -oE '[+-]?[0-9.]+' | head -1; }
run(){ # name model gait v steps tilt_max min_x
  local name=$1 model=$2 gait=$3 v=$4 steps=$5 tmax=$6 minx=$7
  local o=$(GAIT=$gait TROT_V=$v STEPS=$steps PRINT_EVERY=9999999 ./build/trot_sim "$model" 2>/dev/null | grep "종료")
  local f=$(_num "$o" 'falls=[0-9]+') t=$(_num "$o" 'max_tilt=[0-9.]+') x=$(_num "$o" 'x=[+-][0-9.]+')
  local ok=1
  [ "${f:-1}" = "0" ] || ok=0
  awk "BEGIN{exit !(${t:-99} <= $tmax)}" || ok=0
  awk "BEGIN{exit !(${x:-0} >= $minx)}"  || ok=0
  if [ $ok = 1 ]; then printf "  ✅ %-22s falls=%s tilt=%s° x=%s\n" "$name" "$f" "$t" "$x"; pass=$((pass+1))
  else            printf "  ❌ %-22s falls=%s tilt=%s° x=%s  (한계 tilt≤%s x≥%s)\n" "$name" "$f" "$t" "$x" "$tmax" "$minx"; fail=$((fail+1)); fi
}

echo "▶ 평지 (quad_real_16dof_sphere)"
run "walk v0.6" $MJ walk 0.6 6000 8  2.0
run "trot v1.2" $MJ trot 1.2 6000 8  3.5
run "run  v2.0" $MJ run  2.0 6000 12 5.0
echo "▶ 지형 perceptive (verify: 험지→계단 높이적응)"
run "course walk v0.5" $COURSE walk 0.5 20000 10 7.0

if [ $DOPY = 1 ]; then
  echo "▶ Python 파리티 스팟체크 (느림)"
  po=$(MJCF="$COURSE" HEADLESS=1 GAIT=walk TROT_V=0.5 STEPS=20000 PRINT_EVERY=9999999 \
       "$PY" ../quad_mpc_wbic_17dof.py --mode trot --robot ours_17dof_waist_sphere 2>/dev/null | grep -iE "종료")
  pf=$(_num "$po" 'falls=[0-9]+'); px=$(_num "$po" 'x=[+-][0-9.]+')
  if [ "${pf:-1}" = "0" ] && awk "BEGIN{exit !(${px:-0} >= 7.0)}"; then
    printf "  ✅ %-22s falls=%s x=%s\n" "PY course walk v0.5" "$pf" "$px"; pass=$((pass+1))
  else printf "  ❌ %-22s falls=%s x=%s\n" "PY course walk v0.5" "$pf" "$px"; fail=$((fail+1)); fi
fi

echo "──────────── PASS=$pass  FAIL=$fail ────────────"
[ $fail = 0 ] && { echo "✅ ALL PASS — 회귀 없음"; exit 0; } || { echo "❌ REGRESSION — 위 실패 항목 확인"; exit 1; }
