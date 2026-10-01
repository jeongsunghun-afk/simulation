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
case "$MJCF" in *flatfoot*) IS_FLAT=1 ;; *) IS_FLAT=0 ;; esac   # ★2026-09-30 모드별 기본값 분기

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
# ★★2026-09-22 승리 조합 기본값 — 웅크림 Qflat8(CoM z0.34) 자립(6주 forward-tip 해결). env 로 덮어쓰기 가능.
export WBIC_SOFT_CONTACT="${WBIC_SOFT_CONTACT:-1}"      # 2점 평발 QP rank결손(code4) 해결
export FLAT_WLEG="${FLAT_WLEG:-25}"                     # thigh/calf posture pin (25=stand Δq<1°)
# ★2026-09-30 발목 posture 가중은 모드별: 평발 200(발 drift 억제) · 점발 20(200 이 점발로 새면 발목이 굳어 낙상 — sim)
if [ "$IS_FLAT" = "1" ]; then export STAND_WANKLE="${STAND_WANKLE:-200}"; else export STAND_WANKLE="${STAND_WANKLE:-20}"; fi
export STAND_RUNAWAY_DEG="${STAND_RUNAWAY_DEG:-25}"    # 양성 처짐이 기본12°를 넘어 헛트립하던 것 완화
export STAND_KP_FLOOR="${STAND_KP_FLOOR:-0.8}"        # 2026-09-22 stand 떨림 완화 - 위치서보 감쇠 0.30->0.6 (WBIC 목표=Qflat8 라 안싸움). 더 매끈=이값 up(최대 1.0) 또는 GUI kp 슬라이더 up
# ★2026-09-18 hold FF 클램프 해제 — HL_thigh/HL_calf 가 기본 14Nm 에 포화해 왼다리 지지토크
#   부족(몸 못 듦). 전축 50Nm 로 상향(유저 요청). ⚠접지 안전망 약화 — 매달림/브링업 한정.
export HOLD_FF_TAU_MAX="${HOLD_FF_TAU_MAX:-50}"

# ── ③walk 묶음 (2026-08-27 · sim 정량화 tools/walk_demand_check.py) ─────────
#   walk 모드 **한정** 트립 상향(실측 플랜트 스윙 요구 calf 673dps·kd제동 41Nm — 고정
#   200/15 는 스윙 즉살) + kd 축소(제동 제거 — sim 8/8 검증 플랜트와 정합).
#   타 모드는 cfg(biped_emb.yaml safety) 400dps/50Nm·kd 전량. VEL·KD 는 C++ 기본값과 동일(가시화 목적).
export WALK_VEL_TRIP_DPS="${WALK_VEL_TRIP_DPS:-900}"
# ★2026-09-30 25→50Nm (2점 stand/hold 의 cfg tau_trip_nm 50 과 같게). 이유: cfg 가 15→50 으로 오른 뒤
#   walk(25)가 오히려 stand/hold(50)보다 엄격해진 역전 — 점발 walk sim 에서 thigh 채널이 25Nm 를
#   50ms+ 연속 초과(DQ_ZERO=2+WALK_KD_FLOOR=1.0 시험 전 확인). ⚠전 축 공통 한계라 calf/foot 벨트
#   허용 토크도 같이 오른다. 원복: WALK_TAU_TRIP_NM=25 (C++ 기본값은 여전히 25).
export WALK_TAU_TRIP_NM="${WALK_TAU_TRIP_NM:-50}"
export WALK_KD_FLOOR="${WALK_KD_FLOOR:-0.15}"

# ── ③-b ★★2026-09-30 기본값 확정 (실기 결과 · env 로 덮어쓰기 가능 · 10-01 시험 후 재검토) ─────
#   ◆2점 평발 stand (T1 5회 + RobotEmbeddedNew, biped/hw_traces/arm_trace_stand_16*)
#     STANCE_KD=0     접촉 발속도 감쇠 K_D 폐기 — 7Hz 떨림 주경로(발목 80→12°/s). 관절속도는 그대로 씀.
#     FRIC_COMP=0     마찰보상(음의 감쇠) 끔 — 잔여 7Hz 12→3°/s. NEW Emb 에서 60~80s 자립 확인
#                     (deploy 의 "~20s 에 넘어진다" 경고는 IMU 1.8s 지연 시절 기록).
#     STAND_FF_NOTCH_HZ / STAND_FF_LPF_HZ  (선택·기본 없음) stand WBIC FF 노치(Q=STAND_FF_NOTCH_Q 기본 2)/1차 LPF.
#                     10-01 평발 stand 약 40Hz 자려진동(툭 치면 부르르)은 WBIC tff 가 kd 보다 많이 주입해서 — 시험값 40Hz.
#     STAND_BLEND_S=5 진입 35Hz 버스트 5→2s(T1-e). STAND_KD_FLOOR 는 1.0 유지(1.3 은 최고치만 −35%·총량 같음).
#   ◆1점 점발 walk (09-30 T2~T3 계열 5회, arm_trace_walk_17*)
#     WBIC_MIT=2 STANCE_KD=0      하이브리드(가중QP+J̇q̇, K_D 없음) + KinWBC 계획 q_des·q̇_des
#     WALK_TRACK=1 TRK_KP/KD=1.0  드라이버 추종 — 기준선(kd 15%)은 hip roll 25~28Hz 발산
#     WALK_FF_LPF_HZ=10           WBIC FF 널뛰기 억제: 부호반전 42.7→7.6/s · 실측↔명령 상관 0.01→0.6~0.7
#     FLAT_STEPH=0.03             발 드는 높이 6→3cm(관절속도 −26%, sim 0낙상)
#     STAND_BLEND_S=5             walk 진입 토크 인수 5s
#     IMU_PITCH_OFS_DEG=12.7      IMU pitch ↔ 관절기구학 +12.7° 불일치 보정(적용 시 제자리 유지 4~5s·전엔 앞으로 달림).
#                                 ⚠수평계 확인 전 · 평발 stand 에는 아직 미적용(미시험)
#     MPC_ASYNC=1                 MPC QP 를 워커 스레드로(제어 루프 2ms 보장). 10-01 실기: 틱>3ms 7~8.5%→0.22%
#     WALK_FF_SCALE=0.5           WBIC FF 비중 상한. 10-01 실기(FF1.0→0.5): 20~50Hz 떨림 32~34→13~15°/s ·
#                                 드라이버 출력0 26~33→2~3% · 명령↔실측 상관 0.1→0.6 · walk 56.5s. 0.6 과 사실상 같음.
#     WALK_FF_SCALE_CH            (선택·기본 없음) 채널별 FF 비중, 미지정 채널=WALK_FF_SCALE. 예 0.5,0.5,1,1,0.5,0.5,1,1
#                                 지연된 WBIC 가 떨림에 넣는 에너지는 hip roll 이 주범(calf/foot 는 kd 가 이김) · 전 축 0.5 는 foot 7~10° 처짐.
#     WALK_FF_NOTCH_HZ            (선택·기본 없음) walk WBIC FF 노치, 쉼표 목록 최대 3개(Q=WALK_FF_NOTCH_Q 기본 2). 시험값 15,19.5
#                                 무릎–발목 15Hz·hip roll 19.5Hz 에서 지연된 WBIC FF 가 kd 보다 많이 주입(stand 40Hz 와 같은 원리).
#                                 ★기본 15Hz·Q4(10-01 185634 walk 117s: 12–18Hz 35~43→16°/s, 주입 +9~13→+1W, 균형 이상 없음).
#                                 Q2·15,19.5 는 sim 에서 밀기 회복 악화(FF1.0 은 붕괴) — 보행대역(4–5Hz) 위상을 깎으므로 넓히지 말 것.
#     RET_TAU=1                   발디딤 복귀앵커(com0) 누설 1s. 10-01 실기 50s walk: 추정 xy 드리프트(1~2m)×K_RETURN 0.15 로
#                                 착지점이 몸통 대비 ~20cm 앞으로(CAP_CLAMP 포화) → 주저앉음·발목 꺾임·미끌림. sim 재현·해소(밀림 12.8→3.2cm).
#                                 실기 10-01 171117: walk 133s·발위치/종아리각/높이 내내 유지. ⚠좌우는 사람이 살짝 받침 — 혼자서는 못 선다(측방 균형 미해결).
#     ⚠MD80 속도한계 10 rad/s(=채널 573°/s)에 calf/foot 가 걸려 fault(0xC0) — RGA 10-01 08:14 20 rad/s 로 상향(fault 0 확인).
#     ⚠MD80 max current 20A ≈ 채널 28Nm 상한 — 크레인 느슨(체중 지지) 시 calf 수요 p99.5 ~45Nm(sim) → 포화 주의.
if [ "$IS_FLAT" = "1" ]; then
    export STANCE_KD="${STANCE_KD:-0}"; export FRIC_COMP="${FRIC_COMP:-0}"; export STAND_BLEND_S="${STAND_BLEND_S:-5}"
    export STAND_FF_NOTCH_HZ="${STAND_FF_NOTCH_HZ:-40}"   # ★10-01 stand WBIC FF 40Hz 노치(Q2) — 툭 치면 부르르(40Hz 자려진동) 25–50Hz 20→1°/s. 0=끔
    DEF_MSG="2점 평발 stand: STANCE_KD=$STANCE_KD FRIC_COMP=$FRIC_COMP STAND_BLEND_S=$STAND_BLEND_S FF_NOTCH=${STAND_FF_NOTCH_HZ}Hz"
else
    export WBIC_MIT="${WBIC_MIT:-2}"; export STANCE_KD="${STANCE_KD:-0}"
    export WALK_TRACK="${WALK_TRACK:-1}"; export TRK_KP="${TRK_KP:-1.0}"; export TRK_KD="${TRK_KD:-1.0}"
    export WALK_FF_LPF_HZ="${WALK_FF_LPF_HZ:-10}"; export FLAT_STEPH="${FLAT_STEPH:-0.03}"
    export STAND_BLEND_S="${STAND_BLEND_S:-5}"; export IMU_PITCH_OFS_DEG="${IMU_PITCH_OFS_DEG:-12.7}"
    export MPC_ASYNC="${MPC_ASYNC:-1}"   # ★10-01 MPC 워커 스레드 — 실기 틱>3ms 7~8.5%→0.22%(sim 0낙상 동일)
    export WALK_FF_SCALE="${WALK_FF_SCALE:-0.5}"   # ★10-01 WBIC FF 비중 50%(나머지 드라이버 PD 추종) — 실기 떨림 −50%·출력0 30→2%·56.5s walk
    export RET_TAU="${RET_TAU:-1}"   # ★10-01 발디딤 복귀앵커 누설 1s — 추정 드리프트에 착지가 끌려가 앉던 것 해소(0=종전)
    export WALK_FF_NOTCH_HZ="${WALK_FF_NOTCH_HZ:-15}"; export WALK_FF_NOTCH_Q="${WALK_FF_NOTCH_Q:-4}"   # ★10-01 walk WBIC FF 15Hz 좁은 노치 — 무릎–발목 12–18Hz 떨림 −55~62%·주입 −90%(185634). 0=끔
    DEF_MSG="1점 점발 walk: WBIC_MIT=$WBIC_MIT STANCE_KD=$STANCE_KD WALK_TRACK=$WALK_TRACK TRK=$TRK_KP/$TRK_KD FF_LPF=${WALK_FF_LPF_HZ}Hz STEPH=$FLAT_STEPH BLEND=${STAND_BLEND_S}s IMU_OFS=$IMU_PITCH_OFS_DEG MPC_ASYNC=$MPC_ASYNC FF_SCALE=$WALK_FF_SCALE RET_TAU=$RET_TAU FF_NOTCH=${WALK_FF_NOTCH_HZ}Hz/Q$WALK_FF_NOTCH_Q"
fi

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
echo "[run_deploy_hw] 자립 기본값: SOFT_CONTACT=$WBIC_SOFT_CONTACT FLAT_WLEG=$FLAT_WLEG STAND_WANKLE=$STAND_WANKLE RUNAWAY=$STAND_RUNAWAY_DEG (웅크림 Qflat8)"
echo "[run_deploy_hw] ★09-30 확정 기본값 — $DEF_MSG"
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
