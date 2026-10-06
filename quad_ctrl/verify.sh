#!/usr/bin/env bash
# ── quad_ctrl 이관 회귀 게이트 ──────────────────────────────────────────────
#   sim_bridge(배포경로: HAL+Estimator+TrotCtrl)가 trot_sim(레퍼런스)과 bit-동등한지 락인.
#   GT · EST(clean) · EST(노이즈+지연) · 자세(sit/stand_up) · config(로드==기본·적용·env override).
#   판정: 각 케이스서 sim_bridge의 (x·z·max_tilt·falls) == trot_sim 동일값이면 PASS. 전부 PASS면 exit 0.
#   사용: ./verify.sh   (~10초). 미래 변경(quad/cpp 또는 quad_ctrl)이 조용히 깨면 여기서 잡음.
set -uo pipefail
QC="$(cd "$(dirname "$0")" && pwd)"          # simulation/quad_ctrl
TS="$QC/../quad/cpp"
MJ="$QC/../quad/mjcf/quad_real_16dof_sphere.mjcf"

echo "▶ 빌드(sim_bridge + trot_sim)…"
cmake --build "$QC/build" --target sim_bridge -j >/dev/null 2>&1 || { echo "❌ sim_bridge BUILD FAIL"; exit 1; }
cmake --build "$TS/build" --target trot_sim  -j >/dev/null 2>&1 || { echo "❌ trot_sim BUILD FAIL"; exit 1; }

pass=0; fail=0
_key(){ sed 's/°//g' | grep -oE 'x=[+-][0-9.]+ z=[0-9.]+ max_tilt=[0-9.]+ falls=[0-9]+' | head -1; }

# sim_bridge(배포) == trot_sim(레퍼런스, +NO_JUMP_WARMUP). 공유 env를 양쪽에 동일 적용.
cmp_case(){ local name=$1; shift
  local sb=$( cd "$QC" && env "$@" ./build/sim_bridge          2>/dev/null | _key )
  local ts=$( cd "$TS" && env NO_JUMP_WARMUP=1 "$@" ./build/trot_sim "$MJ" 3000 2>/dev/null | _key )   # ★방금 빌드한 것(추적된 ./trot_sim 은 09-02 구 바이너리)
  if [ -n "$sb" ] && [ "$sb" = "$ts" ]; then printf "  ✅ %-24s %s\n" "$name" "$sb"; pass=$((pass+1))
  else printf "  ❌ %-24s\n     sim_bridge: %s\n     trot_sim  : %s\n" "$name" "${sb:-<empty>}" "${ts:-<empty>}"; fail=$((fail+1)); fi
}

echo "▶ GT (컨트롤러가 d_phys=실상태로 계산)"
cmp_case "GT v0.5"   MODE=move GAIT=trot TROT_V=0.5
echo "▶ EST clean (컨트롤러가 KF 추정 d_est로 계산)"
cmp_case "EST v0.0"  EST_CTRL=1 MODE=move GAIT=trot TROT_V=0.0
cmp_case "EST v0.5"  EST_CTRL=1 MODE=move GAIT=trot TROT_V=0.5
cmp_case "EST v1.0"  EST_CTRL=1 MODE=move GAIT=trot TROT_V=1.0
echo "▶ EST + 노이즈·지연(sim2real 에뮬)"
cmp_case "EST noise+lat5ms" EST_CTRL=1 MODE=move GAIT=trot TROT_V=0.5 \
         GYRO_N=0.02 QUAT_N=0.01 ENCQ_N=0.002 ENCDQ_N=0.02 ACC_N=0.2 SENSE_LAT_MS=5 ACT_LAT_MS=5
echo "▶ EST 자세 모드"
cmp_case "EST sit"       EST_CTRL=1 MODE=sit
cmp_case "EST stand_up"  EST_CTRL=1 MODE=stand_up

echo "▶ config (원칙③ + PACE 실측 물리)"
CFG="$QC/config/deploy_16dof.yaml"
# deploy config(실측 ROTOR_I/JFRIC/JDAMP 활성) 로드 → 보행 falls=0 (placeholder와 다름=의도된 현실 주입)
dc=$( cd "$QC" && env EST_CTRL=1 GAIT=walk TROT_V=0.5 QC_CONFIG="$CFG" ./build/sim_bridge 2>/dev/null | _key )
if [ -n "$dc" ] && [ "$(echo "$dc" | grep -oE 'falls=[0-9]+')" = "falls=0" ]; then printf "  ✅ %-24s %s\n" "deploy config 보행" "$dc"; pass=$((pass+1))
else printf "  ❌ %-24s %s\n" "deploy config 보행" "${dc:-<empty>}"; fail=$((fail+1)); fi
# config 값 적용 + env override: SENSE_LAT 8ms를 config로 → 붕괴(falls>0), env로 0 덮으면 falls=0
TMP="$(mktemp)"; printf 'GAIT: trot\nTROT_V: 0.5\nSENSE_LAT_MS: 8\nACT_LAT_MS: 8\n' > "$TMP"
fa=$( cd "$QC" && env EST_CTRL=1 QC_CONFIG="$TMP" ./build/sim_bridge 2>/dev/null | grep -oE 'falls=[0-9]+' )
fb=$( cd "$QC" && env EST_CTRL=1 SENSE_LAT_MS=0 ACT_LAT_MS=0 QC_CONFIG="$TMP" ./build/sim_bridge 2>/dev/null | grep -oE 'falls=[0-9]+' )
rm -f "$TMP"
if [ "$fa" != "falls=0" ] && [ "$fb" = "falls=0" ]; then printf "  ✅ %-24s 적용:%s env override:%s\n" "config 적용/override" "$fa" "$fb"; pass=$((pass+1))
else printf "  ❌ %-24s 적용:%s(붕괴기대) override:%s(falls=0기대)\n" "config 적용/override" "$fa" "$fb"; fail=$((fail+1)); fi

echo "────────────────────────────────────────"
if [ $fail = 0 ]; then echo "✅ 전부 PASS ($pass) — sim_bridge == trot_sim bit-동등, config OK"; exit 0
else echo "❌ FAIL $fail / PASS $pass"; exit 1; fi
