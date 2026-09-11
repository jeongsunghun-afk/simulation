# TAMOLS Whole-Body 온라인 추종 — 개발 리포트

> **상태: 종결(구조적 막다른길) → 험지 크로싱 RL 피벗 확정 (2026-07-30 기준).** ETH식 완전 TAMOLS 계획(base pose+발판+GIAC 안정)을 A의 실행층(SRBD MPC + WBIC)으로 online receding-horizon 추종하려던 시도. 평지 안정화 진단(P0~P2)까지 수행했으나 **full TAMOLS 단발 WBIC 재생은 구조적 막다른길로 확정(2026-07-15, perceptive-nav-tamols)**. 아래 진단·수정 내역은 종결 근거이자 재사용 자산으로 보존한다.
>
> ※ 범위: 이 리포트는 **A 실행층 위 whole-body 추종**(§2~6, 평지)과 **갭 크로싱 + C++ TAMOLS 솔버**(§7)를 다룬다. 모델기반 지형계획(TOWR) 트랙은 `TOWR_개발리포트.md` 참조.

## 1. 결론 (상위 최종 위치)

- **full TAMOLS → WBIC 단발 재생 = 구조적 막다른길.** 저수준(WBIC)은 문제없으나(임의 plan 추종 가능), 계획층에 **3가지가 부재**: ①스텝길이 적응 X(Raibert=속도×stance 고정) ②접촉타이밍 적응 X(고정 게이트 클럭=속도공명 근본원인) ③base-발판 협조 최적화 X. tamols-rl은 발판 1개/다리=수신 horizon 플래너라 단발 2s 재생 자체가 부적합.
- **★확정 피벗**: 단발 WBIC-TAMOLS 정리 → **험지 크로싱 = RL(하이브리드)**. A(보행/기립/점프 배포)는 모델기반 유지, 험지 크로싱만 RL이 발판/타이밍 정책을 담당하되 **WBIC·지형맵·footScore를 실행층/관측으로 재사용**.
- 아래 진단(④ lateral=W_AM, ⑤ online=re-anchor death spiral)과 자산은 위 결론의 실측 근거이며, 향후 합성 정상-cadence plan으로 추종기 재검증 시(B2/B3 모델기반 실시간 OCP 착수 시) 출발점이다.

## 2. 시도한 것 (목표와 단계)

**목표**: 발판만 주입(injection)이 아닌 **base·자세·발판 통합 추종**(ETH TAMOLS+WBC 방식). 발판만 주입하는 injection(A gait + TAMOLS 발판)은 falls=0으로 이미 작동 — A의 검증된 gait가 lateral 드리프트를 우회한다. 순수 whole-body 추종은 그 문제에 정면으로 부딪힌다.

**단계 정의**:
- **P0**: offline 단일 사이클(0.8s) 추종 falls=0 · tilt<5°
- **P1**: offline 다사이클(체인) 추종 falls=0 (연속 3s+)
- **P2**: online receding-horizon 평지 falls=0 · 3s+ (death spiral 제거)
- **P3**: online + 지형(heightmap) 크로싱, injection과 성능 비교

## 3. 진단 (평지 online 추종이 ~1초 내 낙상)

| # | 문제 | 원인 | 상태 |
|---|---|---|---|
| 1 | z 침하 | RSL이 SRBD MPC 우회(gravity-comp만, base z-hold 약함) | ✅ `TAM_MPC`로 해결(offline 확인) |
| 2 | phantom stance | `SW_DUR`(0.4) > 계획 swing 위상(0.2) → 발 반만 스윙 | ✅ `SW_DUR` env 매칭 |
| 3 | base 후진 lurch | 오염 입력 시 solve_fast 후진 vx | ✅ `TAM_CLEANV`(X clean) |
| 4 | **lateral/yaw 드리프트** | CoM 지지폴리곤 유지 실패(body-sway 추종 부족) | ✅ P0에서 `W_AM`으로 해결(§4) |
| 5 | death spiral | 추종오차→오염상태→나쁜 replan→악화 | ❌ online 고유(§5) |

**핵심**: 개별요인(1·2·3)은 downstream 손잡이로 해결. 솔버 계획은 **깨끗한 입력엔 양호**(후진 없음)이므로 병목은 계획이 아니라 **추종의 lateral 로버스트니스**(우리 WBC의 CoM-지지 유지 성숙도 < RSL). 검증된 가설: **H2(각운동량 미제어)=참**(단일지지 스윙 시 발 flail·다리 반력이 yaw/roll 각운동량 유발) / **H4(발판 비대칭)=거의 무효** / **H5(SRBD 실행층 한계)=online에서 재확인**.

## 4. Phase 0 결과 — 각운동량 task(W_AM)가 ④의 핵심 (offline, 2026-07-29)

offline 첫계획(정지 덤프 `TAM_DUMP`) 추종에서:
- **W_AM 기본→30**: t=0.75 tilt **49.9 → 19.2°**(절반↓). **t=0.5까지 tilt<6.3°**(이전 ~50). → P0 목표 tilt<5° 근접.
- 각운동량 감쇠(KD_AM)·W_ORI 추가는 미미(plateau ~19). 마지막 스윙(FL)에서 tilt 19로 튐 = 사이클 경계 효과(체인에서 해소 기대).
- **H4(발판 대칭화) 거의 무효**(tilt 49.8, yaw만 8→2 약간). foot 비대칭은 부차.

→ **Phase 0 대체로 성공**: W_AM=30으로 offline 단일사이클 t=0.5까지 tilt<6.3°.

## 5. online(P2) 결과 — ⑤ re-anchoring death spiral (online 고유 벽)

- **online은 W_AM으로 안 고쳐짐.** online+W_AM=30: 여전히 낙상(**yaw −143° 스핀** · z 침하). W_YAW=30 · REPLAN_DT 0.4/0.8 모두 실패(yaw −132° / −47°).
- **⑤ re-anchoring death spiral이 online 고유 병목**(offline엔 없음). yaw 급발산 = 재anchor가 yaw 참조/발판을 오염시켜 누적. WBIC yaw task(W_YAW)로 안 잡힘 = **참조 자체가 오염**되는 문제.
- 시도한 완충책: swing foot commitment(절대시간+target 동결)로 재anchor 대응, replan 초기조건을 측정상태 대신 이전 계획상태 blend(오염 완충) 검토 — 그러나 **death spiral이 구조적**(④가 유발, ⑤가 online에서 증폭)이라 downstream 손잡이로 못 풂.

## 6. 종결 근거와 재사용 자산

**왜 막다른길인가**: P0(offline W_AM)은 확보됐으나 P2(online 폐루프)에서 re-anchoring death spiral을 못 풂. 근본은 §1의 **계획층 3부재** — tamols-rl 플래너가 수신 horizon(1스텝) 구조라 단발 재생 정합이 불가하고, 우리 실행층은 그 참조의 오염을 흡수할 계획층이 없다. downstream WBC 게인(W_AM·W_YAW·KD_AM)으로는 참조 오염을 교정할 수 없음이 실측으로 확정.

**재사용 자산**(이번 세션 구축):
- `TAM_MPC=1`: SRBD MPC 기반 추종(z 침하 해결)
- `TAM_CLEANV=1`: X 전진 clean(후진 회피, Y sway 유지)
- `TAM_DUMP=<file>`: online 계획 → load_tamols 포맷 덤프(offline 격리·대칭성 진단)
- `SW_DUR` env: offline도 읽음(계획 위상 매칭)
- `W_AM=30`: 각운동량 task(④ lateral/yaw 억제 근거)
- swing foot commitment: 절대시간+target 동결(재anchor 대응)
- 발판 대칭화·깨끗계획 후처리 스크립트, 지형(hsteps·dsteps·dsteps2·trench·gapcourse)

**이관**: 험지 크로싱은 RL(하이브리드) 트랙으로. 위 WBIC 실행층·지형맵·footScore는 RL의 실행층/관측으로 재사용(콜드스타트 아님). 모델기반 실시간 OCP(B2/B3)는 살아있으나 현재 미착수 — 착수 시 합성 정상-cadence plan으로 이 추종기를 재검증한다.

## 6.1 재확인 — Phase 4 교차 벤치 + D1 참조 평가 (2026-07-31)

사용자 요청("TAMOLS+RSL를 D1 참조로 재앵커·base_z 침하 근본해결")에 따라 **독립 재조사**했고, §3~§6 결론을 **정량 재확인**했다.

**Phase 4 교차 벤치**(동일 17-DOF·동일 지형·연속지형, VX=0.2, C++ trot_sim, 첫 tilt>90° base_x / 최종 상태):
| 지형 | A(반응형·plain) | injection(TAMOLS 발판주입) | pure-online RSL | D1(perceptive NMPC) |
|---|---|---|---|---|
| slope 15° | **x2.17·z0.838·falls0 완주** | x0.6·z0.50 **stall(미등반)** | x0.34·**z0.166 붕괴** | x2.65·z0.84(crest 전복) |
| rough | **x2.14·falls0 완주** | x0.6 **stall** | x0.21·**z0.156 붕괴** | x2.12(이후 전복) |

**진단(TAM_DBG 실측)**: pure-online RSL 붕괴는 base_z droop이 아니라 **yaw 급발산(0→−50°)+횡드리프트(comy 0→−0.27)=붕괴**(t≈0.4s부터). §3·§5의 ④lateral/⑤death-spiral·H2(각운동량) 재확인. **injection "stall"의 진짜 원인=gap-slow 오발동**: 이 지형들은 무한 floor plane(group0)+group2 블록이라, group2 미커버 영역서 `!tmap.valid`→"void 오판"→0.3× 상시 감속(평지도 x0.37/8s). `GAP_SLOW=1.0`으로 끄면 injection 전진 회복(rough x0.6→2.52)하나 그 뒤 지형서 **전복**(slope1.03/rough2.29)=injection+MPC가 연속지형선 A보다 약함.

## 6.2 ★injection+RSL — 돌파(사용자 지적 반영, 2026-07-31, 커밋 629209e)

사용자 통찰: **pure-online RSL은 injection 성능 향상 수단이지 별도 컨트롤러가 아니다.** RSL-WBC 직접추종을 **injection(A gait) 경로에 배선**(`RSL_TRACK`, move 경로 env-gated)하니 pure-online의 붕괴가 사라진다 — injection이 **A gait의 깨끗한 참조**(Raibert 발판+측방 capture)를 주므로 재앵커 오염·nominal 발판이 없기 때문.

**실측**:
| | flat | slope 15° | rough |
|---|---|---|---|
| injection+MPC(기존) | 정상 | x1.03 **전복** | x2.29 **전복** |
| pure-online RSL | z0.16 **붕괴** | 붕괴 | 붕괴 |
| **injection+RSL(신규)** | **x1.5·z0.55·falls0** | **falls0·전복 없음** | **falls0·전복 없음** |

- **붕괴/전복 제거**: injection+RSL은 flat·지형 모두 falls=0, 전복 없음. pure-online의 붕괴도, injection+MPC의 지형 전복도 없다(RSL이 더 안정).
- **핵심 튜닝**: `W_BASE_XY=40`(A Raibert 발판이 전진 주도; 80은 base task가 발판과 싸워 backward drift)·`KP_BASE=0`(xy 속도제어)·z=`x_ref[5]` 지형적응·λ=중력보상 baseline.
- **한계(정직·추가튜닝 후 확정, d146603)**: 지형 전방 traversal은 **게인튜닝으로 해결 불가=구조적**. 전방 lead 자가앵커(RSL_LEAD+KP_BASE)로 **평지는 강전진**(lead0.1·KP60=x2.85·falls0)이나 **지형선 전복**: TAMOLS 발판편차↔RSL base 위치앵커 **충돌**(TAM_BASE 무관·속도↑도 무효·두 지형 동일결과=플랫 접근부서 실패=TAMOLS 발판주입+RSL 비호환, A Raibert 발판만이면 안정). 근본=RSL base task가 **예측힘(모멘텀률 base FF) 없어** TAMOLS 발판과 강건협조 불가. → **"안정성"=해결(붕괴/전복 제거), "지형 전방 도달거리"=구조적 한계(D1식 aBaseFF 필요)**.
- 기본값=보수(KP_BASE=0=안정, 전방은 A Raibert 담당). 전방 lead는 env 실험용(평지 전용). 기본 A walk(`RSL_TRACK` off) 불변(회귀 clean).
- **다음(선택)**: D1식 모멘텀률 base 6D FF(aBaseFF=Ab⁻¹(m·ḣ−Ȧv−Aj q̈)) 이식 = RSL이 TAMOLS 발판과 예측적으로 협조하는 유일 경로(§6.1 A-WBIC엔 부재 확인). 복잡·고위험이나 지형 traversal의 근본해결.

**정정**: §6.1의 "D1 재구축과 동치" 결론은 **pure-online 기준**이었고, injection+RSL 경로는 그와 별개로 **붕괴/전복 없이 작동**(사용자 지적이 옳았음). 남은 건 D1식 모멘텀률 FF로 전방 progression 강건화(선택).

**결론 재확정**: 연속지형=**A(반응형) 또는 D1(perceptive)**가 강건(실측). pure-online TAMOLS+RSL은 구조적 막다른길(재확인). 이산험지=RL. D1 참조의 실질 = "연속지형은 D1/A 쓰라"이며, 그것이 이미 결론.

## 7. 별도 실행 시도 — 갭 크로싱 + C++ TAMOLS 솔버 (2026-07-27~28)

§2~6은 A 실행층 위 whole-body **평지** 추종이었고, 이건 **불연속 지형(갭) 크로싱**을 TAMOLS 계획 + C++ 전용 솔버로 실행한 별도 시도다. 결론은 같다 — **계획(TAMOLS)은 OK, 추종기(executor)가 벽**. (구 `모델기반_갭크로싱_탐색리포트.html` 통합.)

### 7.1 계획 — TAMOLS는 깨끗한 갭 크로싱 계획을 낸다
tamols-rl(ianpedroza, Drake)에 02_Leg 파라미터(m=37.9·nominal 0.52·μ0.6·sphere발) 적응, GAP=0.20m서 feasible solve. `add_gap_avoid_footholds`(앞발=갭 너머·뒷발=갭 앞) 하나로 발이 solid에 straddle(뒤 0.42·앞 0.87), base pitch ±13.4→±10.4°, 전진 0→0.73(갭 통과). 단 이건 **옵티마이저 해의 품질**이지 로봇이 건넜다는 실행 검증이 아님.

### 7.2 실시간 측정 — Drake는 583× 느려 불가
2-phase(최소 0.8s) cold solve 383ms=2.6Hz(실시간 50Hz와 19×), warm-start도 4~13Hz로 미달. Drake는 연구·오프라인용(범용 NLP) → **실시간 TAMOLS = 튜닝이 아니라 C++ 전용 솔버 재구현**(583× 못 메움).

### 7.3 ★C++ TAMOLS 솔버 — 완성 (planner)
`quad/cpp/tamols/`, Drake 단계별 정합 검증:
- **정식화**: 스플라인·지형·제약5(초기·연속·friction·kinematic·GIAC Eq17)·비용3(track·foothold·nominal). Drake residual 0~e-6·비용 rel 1e-9 ✅
- **솔버**: SQP-RTI(eiquadprog QP + ℓ1 merit + 적응형 LM). Drake해=고정점·섭동서 feasible 수렴 ✅
- **실시간**: 해석 Jacobian(cost/eq/비GIAC + GIAC block-sparse FD) → **<20ms(5-iter 11.7ms) 실시간 달성** ✅
- **cold-start**: elastic-mode globalization + 동역학일관 Hermite init → **Drake seed 없이 self-contained 수렴**(eq 1e-16, ineq 1e-6) ✅

### 7.4 폐루프 크로싱 — ❌ 추종기서 낙상
플랜→WBIC 참조 변환(`tamols_track/export_traj`)→A(WBIC+SRBD MPC) 추종. **접근(t=0~1.25s)은 깨끗이 추종**(yaw~0°·tilt<15°·falls=0), 그러나 **크로싱(t=1.25~1.75)서 계획을 1.7× 오버슛**(로봇 x=1.037 vs 계획 종단 0.724): 앞발 원거리 착지→몸통 급전진→**CoM이 앞 스탠스발 넘어 전방tip**→yaw 스핀 19→80°→낙상.
- **근본원인=추종기**: A의 SRBD MPC는 **x,y 위치를 추종 안 함(Qdiag px=py=0), 속도만 추종하는 정상상태 조절기**라 크로싱의 감속 GRF를 2발 대각지지서 못 만듦. 변형(yaw홀드 W_YAW·1.6× 느린재생·발판/base 대칭 solve) 전부 다른방식 실패.
- ★이전 "falls=0 크로싱" 보고는 **marginal·재현 불가**였음을 정정.

### 7.5 자산·결론
자산: `quad/cpp/tamols/`(C++ 솔버)·`quad/cpp/src/terrain_map.hpp`(footScore/edgeSDF/slope)·`quad/tamols/tamols_02leg.py`(Drake 레퍼런스)·`quad/mjcf/quad_tamols_gap.mjcf`(갭 검증 씬). TOWR(오프라인 발판최적화)는 우리 로봇 ROM서 미수렴(상세=`TOWR_개발리포트.md`).
**결론**: C++ TAMOLS 솔버는 완성(실시간+cold-start)이나 폐루프 크로싱은 A 추종기서 실패 = §1과 동일하게 **병목=추종기** → TAMOLS/TOWR 계획 재사용 + **RL 추종(DTC)** 이 남은 길.

## 8. Go2 이식 교차검증 (2026-08-03) — 종합: `pipeline_tamols.html` §6

"우리 로봇(02_Leg)이 문제냐, 접근법이 문제냐"를 가리려 제어기를 **Unitree Go2**(12-DOF·점발·15kg)에 이식. 핵심만:

- **로봇은 병목 아님**: A제어기 Go2 전속도 완주. "①(TAMOLS+WBC)이 서기만 한다"는 **오판**이었고, **버그 3건**이 원인:
  - **base-z 하드코딩**(계획 z=0.52를 다리0.42 Go2에 명령→붕괴, TAM_DUMP로 규명) → `tam_z0=base_z0`.
  - **HQP over-determination 크래시**(저-DOF서 nsw≥3면 strict등식>변수→heap corruption) → 가드.
  - **발판 y_min 하드코딩**(0.10=02_Leg값, Go2 hip폭0.093에 과폭→앞뒤까지 뭉침 WB 0.036) → hip폭 매칭, WB 0.244·tilt 7.4→1.7°.
- **HQP vs TSID**: 저-DOF(Go2 12관절)엔 **TSID(weighted)가 HQP(strict)보다 강건**. 논문표준 HQP는 여유부족으로 상한 낮음(V≤0.2).
- **★발판 = solve_fast 버리고 Raibert+footScore(selectFoot)**: solve_fast가 발판 통째 최적화하면 뭉침·충돌 → **Raibert 기본 + 지형 nudge만**(FOOT_NUDGE)로 02_Leg 스테핑 **완주 falls=0**. (사용자 통찰)
- **험지 = 직교 2축**: **배치**(스테핑·갭 = selectFoot 해결) / **등반**(경사·오르막 = 추진 필요, 미해결). Go2 10°램프·계단 전복은 배치 아닌 **등반**(base-pitch 협조·추진) 문제 = 모델기반의 진짜 벽 → RL.

**커밋**: cea6632·c4c1722·ea3d9d0·e5e80e7·ed90307. 모두 env-gated, 02_Leg 배포경로 무영향. 상세=`pipeline_tamols.html` §6.


---

## 부록 · TAMOLS 논문 한계·검증 지형조건
> (구 `TAMOLS_논문_한계_검증조건.md` 통합, 2026-09-11)

> 원전: F. Jenelten, R. Grandia, F. Farshidian, M. Hutter, **"TAMOLS: Terrain-Aware Motion Optimization for Legged Systems"**, IEEE RA-L 2022, arXiv:2206.14049v2. 로봇=ANYmal(12 DOF). 우리 세션(2026-07-31) 논문 정독 발췌.
> **용도: 우리 모델기반 full-TAMOLS 개발(full-dyn OCP + TAMOLS)의 참조** — ①목표 지형범위 설정(어디까지 노리나) ②기대 한계 파악(모델기반이 근본적으로 못 넘는 선). 우리가 실측한 "full-dyn OCP도 slope stall·gap fall"이 이 논문 한계와 어떻게 맞물리는지 대조.

---

## 1. 명시적 한계 (§VII-G Limitations)

1. **★동역학 모델 근사 (major drawback)** — TAMOLS의 GIAC 안정성 보장(weak contact stable)은 **수평 접촉면**에서만 성립. **발 하나라도 기울어진 평면**에 있으면 WBC가 no-slip 조건 위해 계획 궤적에서 이탈. 일반 지형(비수평)으로 확장은 이론상 prop 1·2를 hard 제약화하면 가능하나 **그런 발판 찾기가 매우 어렵다**고 명시.
2. **full kinematics 미반영** — 간이 운동학 제약(task-space)이 다리 과신장은 막으나 **무릎 관절 충돌(knee joint collision)은 여전히 문제**. → DTC 논문(2309.15462) 기준 **TAMOLS 결합 최대 극복 높이 ≈ 0.40 m**(이 한계 때문).
3. **elevation map 품질 강의존** — 상태추정 드리프트가 map을 odometry 대비 이동시킴 → 실제 발위치와 계획 발위치 **불일치** → **극단적 경우 시스템 불안정화**.

(추가: GM observer 실측서도 정지 시 힘 추정치가 0으로 수렴 안 함 = 부정확한 질량/CoM 모델오차. GM=실기 외란/모델오차용, sim은 무의미.)

---

## 2. 검증 지형조건 (§VII Results)

- **로봇**: ANYmal (ruggedized quadruped, 12 actuated DOF), 실기+sim. LiDAR 2개, elevation map 20Hz(GPU), 상태추정 400Hz(CPU).
- **예측 호라이즌**: **1 gait cycle**(다리당 1스텝). 계산: trot **6.3 ms**(SOTA CMO 48배)·fast trot 2.1 ms·SQP 1~2 iter 최다 수렴.
- **검증 gait**: trot · fast trot · running trot · amble · pace · running pace · crawl (7종).

| 지형 | 조건 | 결과(성공률/거동) |
|---|---|---|
| **계단(sim)** | 12 tread, 18회 승/강, trot. 다리 12.7% 연장 변형·tread 1.27× | TAMOLS **18/18**(trot/amble/running trot/pace) · crawl 16/18↑·14/18↓(대보폭=downstairs 접촉불일치·upstairs 무릎충돌 취약) · (구 batch search trot 10/18↑·14/18↓) |
| **계단(실기)** | **20 tread, 29×17 cm, 36° 경사**, trot, 명령 0.45(실현 0.37 m/s) | 스텝당 0/1/2 tread 클리어. tracking 오차=계단 기하로 feasible space 축소 |
| **속도변조(sim)** | 12스텝 승, fast trot | >0.9 m/s=2 tread/step · <0.9=2·1 교대 · **~0.45~0.55 m/s=1 tread/step(최적)** |
| **갭(실기)** | pallet+slope 갭, ambling, 장애물 30 cm 간격. **갭 높이 20 cm·폭 27 cm**, 명령 0.7 m/s | 급경사부 회피·**갭 안 밟음**(5회 중 RH발 1회만 갭 진입) |
| **stepping stone(실기)** | 경사 나무벽돌, trot, 명령 0.4 m/s. **벽돌 20×20×50 cm, 인접 갭 20 cm** | 발판을 돌 중앙 배치(h_s1 gradient 페널티) |

**요지(§VIII 결론)**: GIAC=미분가능·접촉력 free 동적안정 척도(ZMP만큼 복잡하나 SRBD의 큰 유효범위) + graduated optimization으로 발판·base pose를 rough 지형서 공동최적화. 일반화=rough·human-engineered 환경(계단·갭·stepping stone).

---

## 3. 모델기반 개발 함의 (우리가 쓸 것)

**(a) 우리 모델기반이 노릴 지형범위 (검증조건이 상한 제시)** — TAMOLS(모델기반)가 실기서 넘은 범위 = 우리 목표 상한:
- 계단 **~29×17cm·36° 경사**(1 tread/step 최적 0.45~0.55 m/s, 2 tread는 >0.9 m/s)
- 갭 **폭~27cm·높이~20cm** · stepping stone **20cm 벽돌·갭 20cm**
- 예측호라이즌 = **1 gait cycle**, gait 7종(trot~crawl). 이보다 큰 이산/비수평은 모델기반 단독 곤란.

**(b) 우리 실측 findings ↔ 논문 한계 대조 (2026-07-31)**:
- full-dynamics OCP = **평지 base 안정화(underactuated 닫기) 해결** ✅.
- 지형서 **slope stall·gap fall** = 논문 **한계①(수평 접촉면 가정, 기울어진 발판서 WBC 이탈)의 실증**. 즉 우리 실패가 TAMOLS 근본 한계와 정확히 동종 → **모델기반 추종의 공통 하드월**임을 논문이 사전 예고.
- ⇒ 모델기반 개발 시: **비수평 접촉(경사·계단 tread)·전진력이 근본 취약점**. prop 1·2를 hard 제약화(비수평 발판 허용)하거나 full-kinematics(무릎충돌)까지 넣어야 논문 한계①②를 넘음 = 대공사.

**(c) 모델기반의 근본 한계선 (넘기 어려운 것)**:
- ①비수평 접촉면(기울어진 발판) — WBC no-slip 이탈. ②무릎충돌 ~0.4m(간이 운동학). ③map 품질 의존(상태추정 드리프트→불안정).
- 이 선 너머(임의 이산험지·큰 단차)는 모델기반 한계 → RL(DTC) 영역. 참조=[[full-tamols-modelbased-tracker]]·`DTC_개발리포트.md`.
