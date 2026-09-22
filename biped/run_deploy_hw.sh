#!/usr/bin/env bash
# run_deploy_hw.sh — 실기 배포기(biped_deploy)를 **식별된 파라미터로** 띄운다.
#
# ★왜 이 파일인가 (2026-08-24): 무중력 브래킷 측정으로 축별 중력 부족분이 확정됐는데,
#   그 값들이 env-var 라 매번 손으로 치면 반드시 빠뜨리거나 옛값을 쓴다.
#   측정의 출처와 값을 **한 곳에** 박아 둔다. 값을 바꾸려면 여기를 고칠 것.
#
# ── 측정 요약 (2026-08-24 · 점발 Qhome8 자세 · 브래킷 판독 = 마찰 소거) ──
#     축          g_lo    g_hi     g*     밴드   τ_c/α    비고
#     HL_hip      1.10    1.30    1.20    0.20            (평발자세 측정. hip 중력은 자세무관)
#     HR_hip      1.00    1.35    1.18    0.35    0.92
#     HL_thigh    1.10    1.90   ★1.50    0.80    0.68    ⚠측정값 — 운용은 1.10 (아래)
#     HR_thigh    0.70    1.50    1.10    0.80    0.66    (PACE 식별 0.604 와 일치 = 판독 검증)
#     calf/foot   측정 불가(중력 ≪ 마찰). 1.00 유지.
#
# ★★핵심 발견 — **밴드(마찰폭)는 좌우 동일한데 g* 만 1.36배 밀려 있다.**
#     밴드 = 2τc/(α·G_CAD) 가 같다  ⇒  α 도 마찰도 좌우 같다
#     g*   = G_real/(α·G_CAD) 만 다르다  ⇒  **왼다리 실제 중력토크가 36% 크다**
#   질량이 36% 다를 수 없으니 기하다: 왼무릎(calf) 실제각이 보고각과 25~30° 어긋나야
#   나오는 크기고, 육안 관찰("초기자세 좌우 다름")·HL_calf 영점 4.25° 이동·커플링 잔차
#   3.11° 미설명·2026-08-11 풀리 재조임 이력이 전부 같은 곳을 가리킨다.
#   ⇒ **왼쪽 무릎 벨트/풀리 미끄러짐.**
#
# ★★2026-08-25 확정 — **HL_thigh 의 g* 는 자세를 탄다 = 기하 오차** (α 아님):
#     측정 중립:  0° 자세 HL 1.225 / HR 1.05   ·   Qhome8 HL 1.50 / HR 1.10
#     비율이 1.17 → 1.36 으로 자세 따라 변함 — α(전기)는 자세 무관이므로 배제.
#     밴드는 좌우/자세에서 1/G 스케일 그대로(마찰 상수·물리 정합 ✓).
#   ⇒ **운용값은 중립이 아니라 "전 자세 밴드 안" 기준으로 고른다: thigh 1.10 (좌우 공통)**
#     1.25(0° 중립)를 쓰면 +27° 이상(중력→0 구간)에서 초과분이 마찰을 넘어 **떠오른다**
#     (실기 관측). 1.10 은 0° 밴드(0.90~1.55) 안이라 안 흐르고 고각에서도 유지된다.
#     실사용 확인: "잘 유지" (2026-08-24 저녁 · 08-25).
#   남은 원인 후보(수리 대상): 다리 위 케이블 루프의 스프링 토크 / CAD CoM 오차.
#     스테이터 유격 의심은 철회(로터 회전이었음). 판별: +35° 넘어도 오르는지(케이블) ·
#     좌우 케이블 루프 여유 비교 · 분해 기회에 정강이+발 저울.
#
# ★★2026-08-25 오후 — push(발밀기) 저울 시험 반영: **calf 1.00 → 1.22**
#   저울 힘시험(외부 기준·4세션·좌우 동일)이 calf 경로 전달비 r = 0.82±0.02 를 실측
#   → 보정 = 1/0.82 = 1.22. calf 는 g* 로 원리적으로 못 재던 축(중력≪마찰)이라
#   지금껏 무보정이었다 — stand/walk 에서 무릎 토크의 18% 가 조용히 새고 있던 것.
#   float 에는 거의 무해(calf 중력이 마찰 밴드 안).
#   세 경로 종합(hip 0.84 g* · calf 0.82 저울 · thigh 0.95 g*)의 최단 해석:
#     **α ≈ 0.83 전 축 공통 + thigh 모델중력 ~13% 과대(CAD CoM)**
#   ⇒ hip 값은 1/α 와 이미 일치(유지) · thigh 는 α 몫과 모델과대가 상쇄돼 1.10 유지.
#   foot 은 평발 스윕 전까지 1.00 보류(1점 walk 은 발목토크 ~0 이라 영향 미미).
#
# (기록) 2026-08-25 오전 판정 과정:
#     HL_thigh g* 1.25 · HR_thigh 1.05  (밴드 0.70 동일)
#   다섯 경로가 한 방향을 가리킨다: **α_HL_thigh 가 HR 보다 10~19% 작다** (구동기 개체차).
#     ①오늘 g* 비 1.19  ②PD 처짐 3.8° vs 1.6° (실효강성 α·kp)
#     ③손맛: 무여자에선 좌우 마찰 대칭·PD on 에선 HR 뻑뻑  ④벤치 마찰 HL/α 13% 과대
#     ⑤벤치 ROTOR_I(=I/α) HL 7.5% 과대 — 조립 전부터 같은 방향.
#   ⇒ 축별 배율이 정확히 그 처방이다(α 작은 축에 자동으로 큰 배율). 아래 값은 0° 자세 실측.
#   ⚠어제(08-24) 오후 1.50 → 저녁 과잉 → 오늘 1.25 의 변동은 α 만으로 설명이 안 되는
#     **일시적 기하 변동**이 겹쳐 있었다는 뜻 — 무릎 마킹이 감시자다. 값이 또 흔들리면 무릎.
#   ⚠walk 는 EtherCAT 케이블 교체 전까지 계속 금지. home 도 측정용 0° 상태다(yaml 주석).
#   후속: HL_thigh 상 커넥터 재삽입→재스윕 · MD80 설정 대조(전류한계·kt) — 실기팀.
#
# 사용:  ./run_deploy_hw.sh            # 1점 점발(기본. walk 용)
#        ./run_deploy_hw.sh flat       # 2점 평발(stand 용. ⚠foot gear_k 1.2/1.6 미해결)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)

MJCF="$HERE/biped_from_quad.mjcf"                  # 1점 점발
[ "${1:-}" = "flat" ] && MJCF="$HERE/biped_flatfoot.mjcf"
[ "${1:-}" = "point" ] && MJCF="$HERE/biped_pointfoot_payload.mjcf"  # 1점 점발 + 실물 payload(16.25kg)
[ -f "${1:-}" ] && MJCF="$(realpath "$1")"         # ★임의 MJCF 경로 (무게추 변형 등)
                                                   #   절대경로화 필수 — 아래에서 cpp/ 로 cd 한다

# ── ①float(무중력) 축별 중력배율 — 측정된 중립점 g* 그대로 ──────────────────
#   이 값이면 무중력에서 전 축이 중립이다(뜨지도 지지도 않음). GUI 배율은 이 위에
#   공통 계수로 곱해진다(×1.00 이 이 값 그대로라는 뜻).
# ★2026-09-17 재튜닝 — ff_comp(1/α) 이중계상 해소 + 재조립 후 float 브래킷.
#   원인: g*(1.20…)가 이미 α 보상하는데 ff_comp(2026-09-12 신설)가 α 를 또 보상 → float 이
#     realized ~1.5×중력으로 과보상, 다리가 떠서 고정지그와 충돌.
#   해법: ACT_ALPHA=1 로 ff_comp 끔(옛 g* 가 α 담당) + g* 를 float 브래킷값으로 재튜닝.
#     실기 브래킷(다리 놓아 안 뜨는 값): hip 만 약간 높음(HL 1.00[09-18 0.97→상향] / HR 0.95), 나머지 0.90.
#   둘 다 env 로 덮어쓰기 가능. 옛 방식 복원:
#     ACT_ALPHA=0.834 GRAV_SCALE_JOINT="1.20,1.10,1.22,1.00,1.18,1.10,1.22,1.00" ./run_all.sh ctrl
export ACT_ALPHA="${ACT_ALPHA:-1.0}"                 # ff_comp(1/α) 끔 — g* 가 α 담당(이중계상 방지)
export GRAV_SCALE_JOINT="${GRAV_SCALE_JOINT:-1.00,0.90,0.90,0.90,0.95,0.90,0.90,0.90}"  # 2026-09-18 HL_hip 0.97→1.00 (놓으면 미끌려처져 약간 상향)
# ★foot 상수결손 보상 (2026-08-27 무게추 캠페인 → 실기 검증: E4 blend 0.66→0.77)
#   r_foot(G)=α−k/G 의 상수항 k 를 토크부호 기반 k·tanh(τ_ch/τ0) 로 전방보상.
#   끄려면 FOOT_COMP_NM=0. 근거: data/push/PLAN_0826.md 최종표.
export FOOT_COMP_NM="${FOOT_COMP_NM:-0.36}"

# ── ②stand/walk 토크보정 — 같은 부족분을 WBIC 토크에 건다 ──────────────────
#   1/g* = α·(G_CAD/G_real) 이므로 τ 에 g* 를 곱하면 실제 출력이 모델 의도값이 된다.
#   ⚠HL_thigh 1.50 경고는 위와 동일 — 벨트 수리 전 walk 금지.
#   ★foot 1.00 → **1.30** (2026-09-03). "안 잰 축은 무보정" 규칙이었는데 이제 쟀다:
#     저울 r_foot(G) 포화 0.77 + hold 자립(발끝적용·1.30 등가)이 하중 대역에서 실증.
#     stand-lite 1차에서 foot 만 1.00 이라 ±7.5° 처짐 — hold 초기와 같은 병리였다.
export STAND_TAU_SCALE_JOINT="${STAND_TAU_SCALE_JOINT:-1.20,1.10,1.22,1.30,1.18,1.10,1.22,1.30}"
# ★2026-09-18 hold FF 클램프 해제 — HL_thigh/HL_calf 가 기본 14Nm 에 포화해 왼다리 지지토크
#   부족(몸 못 듦). 전축 50Nm 로 상향(유저 요청). ⚠접지 안전망 약화 — 매달림/브링업 한정.
export HOLD_FF_TAU_MAX="${HOLD_FF_TAU_MAX:-50}"

# ── ③walk 묶음 (2026-08-27 · sim 정량화 tools/walk_demand_check.py) ─────────
#   walk 모드 **한정** 트립 상향(실측 플랜트 스윙 요구 calf 673dps·kd제동 41Nm — 고정
#   200/15 는 스윙 즉살) + kd 축소(제동 제거 — sim 8/8 검증 플랜트와 정합).
#   타 모드는 cfg 200dps/15Nm·kd 전량 그대로. C++ 기본값과 동일 — 여기선 가시화 목적.
export WALK_VEL_TRIP_DPS="${WALK_VEL_TRIP_DPS:-900}"
export WALK_TAU_TRIP_NM="${WALK_TAU_TRIP_NM:-25}"
export WALK_KD_FLOOR="${WALK_KD_FLOOR:-0.15}"

# ── ④hold 중력지지 — 자립 확정 설정 (2026-09-03 실기: 크레인 프리 25s+) ──────
#   적용점 toe: 뒤꿈치는 발목축 위라 발목토크 기여 0 — 발끝 전량이 발목 FF ≈2배.
#   배율 1.0: toe 적용이면 1.3(midfoot 보정)은 과보정이었다.
#   그날의 배분은 GUI [배분(HL%)] 60 이었다(자세·크레인에 따라 재트림).
export HOLD_FF_POINT="${HOLD_FF_POINT:-toe}"
export HOLD_FF_FOOT="${HOLD_FF_FOOT:-1.0}"

# ── ⑤stand-lite 레시피 (2026-09-03 설계 · **기본 꺼짐** — 명시로만 켠다) ─────
#   hold(자립 성공) vs stand(8Hz 자려진동) 의 갈림 = 지연 낀 위치의존 피드백
#   (CoM kp120/200 · 레벨링 kp150 — IMU 사망 중엔 유령오차). 그 루프를 다 끄면
#   stand = "발 힘 분배만 QP 가 하는 hold" 가 되어 제어방식 A/B 가 성립한다:
#     hold      : λ 고정(50:50·toe) + 드라이브 PD          ← 실증됨
#     stand-lite: λ = QP 분배      + 드라이브 PD(전량)     ← 이걸 검증
#     stand-full: + CoM/자세 피드백                        ← IMU 복구 후
#   실행:
#     STAND_LITE=1 ./run_deploy_hw.sh flat
#   ⚠크레인 건 채 시작. 8Hz 떨림이 재발하면 즉시 hold 로 — 그 자체가 판정 데이터다.
if [ "${STAND_LITE:-0}" = "1" ]; then
    export STAND_COM_KP="0,0,0";  export STAND_COM_KD="0,0,0"
    export STAND_ORI_KP="0";      export STAND_ORI_KD="0"
    export STAND_KP_FLOOR="1.0"   # 드라이브 PD 전량(hold 와 동일) — WBIC 목표=Qflat8 라 안 싸움
    export FRIC_COMP="0"          # 포화 tanh = 음의 감쇠 8~39% — 미시험 항 제거
    # ★lite 1차 실기(09-03): 과제항을 0 으로 하자 QP code4(REDUNDANT_EQUALITIES)가
    #   **41~50%** 로 폭발 — 과제항이 QP 를 정칙화해주고 있었다. 틱의 절반이 QP 해,
    #   절반이 중력보상 폴백 = 두 토크 해 사이 채터링(떨림 유력 원인).
    #   ⇒ lite 에서는 등식 프루닝을 켠다(sim 18.5%→0% 검증). lite 밖 기본은 여전히 OFF.
    export WBIC_EQ_PRUNE="${WBIC_EQ_PRUNE:-1}"
    echo "[run_deploy_hw] ★STAND_LITE — CoM/레벨링 OFF · kp floor 1.0 · FRIC_COMP 0 · EQ_PRUNE 1"
fi

echo "[run_deploy_hw] MJCF=$MJCF"
echo "[run_deploy_hw] GRAV_SCALE_JOINT=$GRAV_SCALE_JOINT"
echo "[run_deploy_hw] STAND_TAU_SCALE_JOINT=$STAND_TAU_SCALE_JOINT"
echo "[run_deploy_hw] ⚠walk 는 왼무릎 벨트 수리 전 금지 — 헤더 주석 참조"
# ★중복 writer 방지 (2026-09-04) — 모터 명령 writer 는 하나여야 한다. 기존 deploy 를 죽인다.
#   (run_hw.sh up + 수동 run_deploy_hw.sh 처럼 두 번 뜨면 둘이 SHM 을 다퉈 반응이 이상해진다.)
if pgrep -f build/biped_deploy >/dev/null 2>&1; then
    echo "[run_deploy_hw] 기존 biped_deploy 종료(중복 writer 방지)"; pkill -f build/biped_deploy; sleep 1
fi
# ★기동 잔여명령 잠금(mode_locked) 방지 (2026-09-04) — cmd 파일에 off 아닌 모드가 남아 있으면
#   deploy 가 그 모드를 잠그고 무시한다(같은 모드 재전송해도 안 풀림). run_emb.sh 처럼 off 로 비운다.
# ★단, 이전 run_hw.sh 의 백그라운드 __pub(home/hold/stand 를 100ms마다 재발행) 가 살아 있으면
#   아래 off-쓰기를 곧바로 덮어써서 deploy 가 boot 에서 그 모드를 보고 잠긴다(mode=home 인데 계속
#   off 로 강하되는 트랩). 그래서 off 로 비우기 **전에** 잔여 발행자를 먼저 죽인다. (2026-09-04)
pkill -f "run_hw.sh __pub" 2>/dev/null; sleep 0.2
echo '{"mode":"off","jog_deg":[0,0,0,0,0,0,0,0],"v":0,"vy":0,"w":0,"body_h":0.42}' > "${QUAD_CMD:-/tmp/biped_cmd.json}"
cd "$HERE/cpp"
# ★2026-09-22 deploy RT 루프를 코어 2,3 에 고정 — Emb(코어 0,1)와 분리해 루프 스톨 방지.
#   taskset 이 biped_deploy 를 exec 하므로 파일 capability(cap_sys_nice=RT)는 그대로 적용된다.
exec taskset -c 2,3 ./build/biped_deploy --mjcf "$MJCF" --start-mode off
