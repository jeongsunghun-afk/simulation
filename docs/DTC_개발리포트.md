# DTC 개발리포트 — Deep Tracking Control (MPC 교사 + RL 추종)

> **상태: 활성 (DTC 트랙 정본) · 2026-08-04.** 17-DOF quad(02_Leg) 위 DTC(모델기반 planner → RL 추종). **P0·P1·P2 완료 · P2.5 오프라인 TAMOLS 캐시+커리큘럼 end-to-end 검증(걷기+gap 크로싱). ★★P2.7 DTC 가치 통제검증 = task-dependent 확정: 평지 gap(관용지형)엔 캐시가 방해(RL이 emergent로 이김), 계단(정밀지형)엔 full-TAMOLS-DTC가 도움 — **확실한 이득=샘플효율(iter4000서 2.8× 빠름, 2-seed 일관)**, 최종성능은 seed편차(clean seed42 동률~6.4·seed7만 DTC우세=baseline 정체)라 점근이득 아님. DTC는 정밀 발배치 지형서 '빠르게 배우는' 레버.**
> 이 문서 = DTC 트랙의 개발 기록·설계 근거. (전신 = MPC–RL 하이브리드 전략 리포트; 실행 트랙이 DTC로 좁혀져 이 이름으로 통합.) 학습법 심화는 `rl_module_train.html`, 작업 메모리는 `dtc-17dof-development`.

---

## 1. DTC란 — 패턴 A(MPC가 교사) (★우리가 택한 하이브리드)

모델기반 planner(TO/MPC)가 최적 레퍼런스를 생성하고 RL 정책이 강건하게 **추종(tracking)**. 원전 = **DTC(Jenelten et al., Science Robotics 2024, arXiv 2309.15462)**. TO의 정확한 발 배치·계획 + RL의 강건성·반사를 결합.

```
[온라인 planner: TAMOLS]              [RL 정책 (PPO, 오프라인 학습)]
지형 elevation map + 명령              관측 = 고유수용감각 + 지형 샘플
  → 발디딤 + base 포즈 동시 최적화          + 레퍼런스 "작은 서브셋":
  → touch-down 시점마다 재계획              · 2D 발디딤 좌표 · IK 목표 관절각 · 접촉 스케줄
       └──────── 레퍼런스 ──────────────┘  → 관절 명령 (고주파 추종 + 반사)
```

DTC가 잘 되는 4가지 (우리가 그대로 가져올 것):
1. **레퍼런스는 "작은 서브셋"만 노출** — 발디딤 2D·**IK 목표 관절각**·접촉 스케줄. 이 정보 병목이 모델기반 취약점(상태추정·비전 오차)을 가려 강건성·planner 교체 불변성을 만든다(다른 NMPC로 zero-shot 교체해도 발디딤 오차 3cm).
2. **IK 목표 관절각을 관측에 포함**한 것이 학습 수렴의 결정타(Fig 7D ablation 실증). ← ★우리 P2가 처음에 빠뜨렸다가 재발견한 바로 그 신호(§3.P2).
3. **가변 업데이트** — touch-down 순간에만 재계산. 50Hz로 올리면 실패율 7.11%p 추가 감소.
4. **레퍼런스 다양성 = 일반화** — 지형·명령·초기조건 randomize를 레퍼런스 생성 단계에서 확보.
- **성능(원전)**: 발디딤 평균 오차 **2.3cm**, 흔들 널빤지(맵오차 0.4m)·비전차단 계단 생존, 0.6m 틈새·1.8m 외나무다리 100%.

**왜 이 패턴인가**: 모델기반 폐루프의 상한은 **연속·구조화 지형**(A footScore/B heightmap = 슬로프 등반 OK, discrete stepping stones 붕괴). 임의 불연속 험지 강건 크로싱은 **RL 몫**으로 확정(2026-07-22 경계, `TOWR_개발리포트.md`). DTC = 그 둘을 잇는 다리(모델기반 정밀 계획 + RL 강건 추종).

---

## 2. 목표 아키텍처 — Kim 2025 (Raibo, Hwangbo/KAIST)

DTC(2024, ETH)보다 우리 목표(**사족·불연속지형**)에 더 가까운 **Kim et al. 2025 (arXiv 2506.02835, Raibo)** 를 목표 아키텍처로 삼는다. planner+tracker = DTC 구조를 갭·stepping·벽·4m/s로 발전. 핵심 3:
1. **tracker** = MLP actor + **GRU state estimator**(= 우리 ActorCriticRMA 골격) · 발판 추종 · target-index 접촉 0.06s 갱신 · 뒷발=앞발자리 · 보상(target/slip/impact/foot_gather 25cm/bound).
2. **competitive CVAE map generator**(적대적 커리큘럼) — tracker가 9.3/10 넘으면 CVAE 재학습해 더 어려운 지형 자동생성(벽주행·1.6m 갭). → P3 커리큘럼 도입 검토.
3. **planner** = 샘플링+순차필터(performance·spike·collision)+8후보 물리 롤아웃, 충돌=경계추정 MLP(RMS 2.27cm).
- ★한계 = 온보드 지각 없음(Vicon+사전맵) → 우리 perception/SLAM 트랙이 채우는 상보 갭.

---

## 3. 개발 현황 — P0 ~ P3

### P0 — 17-DOF quad 자산 (USD + cfg) ✅
MJCF(`quad_real_17dof_waist_sphere.mjcf`, 17관절=4다리×hip/thigh/calf/foot + FB_waist) → USD 변환 + `rga.py` `QUAD_17DOF_CFG`(Peak토크 hip/thigh84·calf126·foot100.8·waist84, self-collision on). 검증: **17관절·22바디·spawn z 0.52·물리안정** — 통과.
- USD 디버깅 3건: ①worldBody spurious ArticulationRoot 제거 ②baked z 0.5235 중복→init z=0 ③meshdir 절대경로 중복→상대경로. + 접촉 3건(instanceable 콜라이더·발 sphere·`/Robot/Base/.*` prim). ★근본해결=MJCF→USD 재변환서 지면·중첩·발콜라이더 처리(현재 런타임 workaround).
- ★교훈(사용자): 학습 전엔 "기립 안정" 판단 무의미 = P0 기준은 로드·관절·높이·물리안정뿐.

### P1 — 속도 워커 (velocity tracking) ✅
`Quad17-Velocity-Direct-v0`(hind_leg env 2→4다리+허리, quad trot, ActorCriticRMA/PPOParkour). action=17·obs=proj_grav3+cmd3+joint 17×3+trot clock4. 크라우치 default. **수렴: model_31200, reward~31, epLen 999**.
- ★즉시종료 버그(epLen 1.00·무학습): USD 내 **유령 바닥면 `/Robot/worldBody/floor`** 가 step0 스퓨리어스 접촉→base_contact 종료. 수정=`_disable_baked_floor_collision()` + init pos z=0.57. → epLen 999 회복. **교훈: 스모크 통과 ≠ 작동**(감시가 잡음).

### P2 — 발판 추종 tracker (진행 중) ★오늘의 핵심
DTC tracker를 **절차적 랜덤 발판**(실제 TAMOLS의 대역, from-scratch)으로 구현. obs에 발판 블록 +28, touchdown XY `−log(err²+eps)` 추종보상, **ablation gate**(발판 obs를 scramble했을 때 발 배치가 나빠져야 = 진짜 추종)로 검증.

**진단 여정 (전부 gate로 검증, 코드 위험 0으로 원인 좁힘):**

| 시험 | 방법 | gate(none vs scramble) | 결론 |
|---|---|---|---|
| 1차 구현 | 발판 obs+28·touchdown 보상 | 17.18 ≈ 17.22 cm (model_1500), 16.63 ≈ 16.59 cm (model_3000) | ❌ **obs 완전 무시** |
| ② 탈상관 | jitter 0.05→0.18 | 25.56 ≈ 25.57 cm | ❌ gap 0 |
| ③ 인센티브 | foothold scale 1→10 | 21.94 ≈ 21.84 cm | ❌ gap 0 |

- **1차 실패 원인 (DTC 논문 대조로 확정)**: ①**lazy-agent** — DTC가 명시 경고한 실패("optimizer가 상태에 적응→에이전트가 게으르게 보상 획득"). 우리 절차적 발판이 자연 걸음새 위치(±5cm)라 obs 안 읽어도 공짜 보상. ②**IK 목표 관절각 obs 누락** — DTC Fig 7D의 #1 학습가능성 레버("foothold accuracy is intrinsically complicated to learn"). §1의 그 신호를 우리가 빠뜨렸음.
- ②·③ 반증 → 원인 = **탈상관·인센티브가 아니라 학습가능성**(직교좌표 발판 obs를 관절 액션으로 매핑하는 걸 정책이 못 배움).

**① IK 목표 관절각 obs 구현 + 물리 검증 ✅ (2026-07-30):**
- Jacobian 1차 IK: 다리별 `Δq = J⁺·(p_target − p_foot)`(hip/thigh/calf 3관절) → obs **89→101(+12)**. ablation을 IK 블록까지 확장(cartesian+IK 동시 blind = gate 유효성).
- ★검증(스모크≠작동이라 값 검증): FD vs 해석 Jacobian **MATCH(rel 0.13~1.2%)** · 다리 간 누설 0 · e2e(dq가 발을 목표로?) **cos=1.000·mag 1.00** = delta_q가 발을 정확히 목표로 이동.
- **Run-D**(① IK + jitter 0.18 + scale 10, from-scratch) 학습 중 — Run-C(scale 10, IK 없음, gap 0)와 **IK만 다른 A/B**. gate@1500 대기.

**분기 (Run-D gate)**: gap 열리면 → **IK가 enabler = 평지 발판추종 해결**, 수렴·정확도(cm) 튜닝. gap 0이면 → 평지+소프트보상 추종은 근본 부족 확정 → **P3의 구조적 발판+지형 물리압력(DTC-identical env)** 으로 조기 이행.
- ★참고: DTC도 tracking을 **지형 커리큘럼**(갭·stepping)에서 학습했다(평지는 배포 결과). 평지 소프트보상 단독 추종은 물리적 압력이 없어 본질적으로 어려울 수 있음.

### P2.5 — 오프라인 TAMOLS 캐시 파이프라인 (실제 TAMOLS 발판) ★2026-07-31 핵심
P2의 "절차적 랜덤 발판"을 **실제 C++ TAMOLS 플랜**으로 대체 = DTC 논문 그대로 "full TAMOLS planner → RL 추종". 단 **온라인 solve가 IsaacLab 스케일서 불가**해 오프라인 캐시로 실현(= APT-RL 방식).

**아키텍처 결정 — in-loop 불가 → 오프라인 캐시:**
```
[in-loop DTC 원전]  4096 env × touchdown마다 TAMOLS solve
   벤치(순수 C++ OpenMP, 20코어, tamols/bench_batch.cpp): warm solve 13ms → 배치 ~400 solves/s
   4096 env @0.4s 재풀이 = 10,240 solves/s 필요 → 25× 부족 (서버 64코어도 alloc경합 포함 ~500 env 상한)
   = DTC가 CPU 클러스터+2주 쓴 이유. 우리 자원선 불가.
        ↓ pivot
[오프라인 캐시 = APT-RL]  TAMOLS를 학습 전 격자로 사전풀이 → 학습중 조회+재앵커(solve 0)
```
- **핵심 통찰**: TAMOLS 플랜은 로컬 프레임(base 원점·전방+x)이라 절대 위치/yaw 무관 → 플랜 모양은 `(vx, gap폭, gap거리)`만으로 결정 → 격자 사전풀이 가능. 학습중 env가 `world = base_pos + Rz(yaw)·local`로 재앵커. 재앵커가 로봇을 따라가 reactive·나머지 균형(capture-point)은 RL이 = **DTC 분업**. (in-loop이 pure-online서 실패한 근본원인 = capture-point 부재였고, RL이 그걸 채우는 게 DTC 존재이유 → 오프라인 캐시도 동일 분업.)

**캐시 (`tamols/cache_gen.cpp` → `cache/{footholds,base,contacts}.bin`+meta.json)**: `vx{0.2~0.6} × width{0.06~0.26} × gapd{-0.2~0.8}` = **1230셀**, 셀당 로컬 발판(4×3)+base궤적(51×12)+접촉(51×4). 50초 생성, 실패 0.
- **도달성 수정**(중요): aggressive straddle-init 끔(`OnlineCfg.straddle_init` 신규 플래그, 기본 true=컨트롤러 불변) + **foot-in-gap 게이트**(nominal 발판[앞~0.51·뒤~0.06]이 실제 gap에 빠질 때만 회피, 아니면 nominal). → 먼 gap의 도달불가 straddle(FL 1.0m 앞) 제거. 검증: straddle이 두 밴드(앞발 gapd~0.4·뒷발~-0.1)에만·전부 도달가능(최대 hip 앞 ~0.29m).
- **width 축**: env gap폭(0.05~0.26 커리큘럼 램프)을 캐시가 커버 → 폭 불일치 해소.

**env 배선 (`RobotSW_IsaacLab/.../quad17/quad17_env.py`, worker 위임)**: `_regen_footholds`의 절차적 Raibert를 캐시 lookup+재앵커로 대체(토글 `use_tamols_cache`, Raibert는 A/B용 보존). leg 매핑(cache FL,FR,RL,RR ↔ env sole 기하 부호매칭 `cache2sole=[2,3,0,1]`, 검증 assert), 3축 lookup(vx·per-gap폭[스트립간격서 도출]·gap거리[searchsorted]), forward-vx 한정(vy=yaw=0, TAMOLS가 vx-only). **obs=101 불변**(발판 출처만 교체). 스모크(64-env) 통과: 캐시 로드·매핑·무오류·foothold_track 산출.

**학습 착수 (GPU2 4096-env, tmux `dtc_gap`)**: 20000 iter, **2.4s/iter → ~13h**(주말 완주).

**★iter 1500 진단 — 두 층으로 갈림(정직):**
- ✅ **추종 층 = 해결**: **foothold_track 0.016→0.27**(iter7→1500, ~15× 상승) + **foot_in_gap ≈ −0.0005**(발이 gap에 거의 안 빠짐=straddle 작동) + Mean reward −7→**+6.5**(양전환). = **P2의 근본 실패(base상대 발판=obs 무시)를 지형유래 발판이 극복** = DTC 오프라인 메커니즘 검증.
- ⚠️ **균형/보행 층 = 현 병목**: **종료=base_contact 지배**(~4초 후 토르소 낙상, time_out 0) + **progress 0.033**(전진 거의 못함) + **epLen ~200/1000**(생존 20%). 발판은 맞추나 **안정된 보행으로 실행 못함** = parkour서 본 무거운 로봇 균형 문제. **spawn 근처(거의 평지)서 이미 낙상**.

**★Go2 파이프라인 대조 (현 병목 관련)**: 현재 quad17 DTC env는 Go2 파쿠르의 두 요소가 없음 — ①**heightmap 지각**(`num_heights=0`, DTC가 의도적으로 foothold 참조로 대체) ②**정식 IsaacLab 적응형 터레인 커리큘럼**(현재=`_build_gap_terrain` 손수 공간 gap 램프, Go2=`TerrainGeneratorCfg(curriculum=True)` level 승강). ②가 병목 기여 후보(평지 걷기 마스터 전에 gap 마주). 단 parkour 포팅(①②다 갖춤)도 tilt했음=원인은 **gait-free가 무거운 로봇엔 too hard**였고, 현재 고정 trot은 그 문제 없음(그래서 추종 성공).

**★iter 5000 균형 판정 = 정체 확정**: foothold_track 0.28→**0.34**(계속 상승, 추종 우수) but **epLen ~200 평탄**(190~218 진동, 상승無)·**progress ~0.03 평탄**·**base_contact ~20 평탄**(낙상 안 줆). = 3200 iter 동안 **발판 추종만 개선·균형/생존은 정체**. → 정책대로 **커리큘럼 전환**.

**★적응형 터레인 level 커리큘럼 전환 (2026-07-31, 자율)**: dtc_gap 중단(체크포인트 저장) → **커리큘럼 학습 발사**(tmux `dtc_curric`, `QUAD17_TERRAIN_CURRICULUM=1`). env 배선(worker): `_build_gap_terrain`을 **10 난이도 lane**으로 확장(level0=평지→level9 gap 0.20m, +y로 lane 배치), per-env `_terrain_level` 진행거리 기반 승강(promote>0.8·demote<0.4·corridor 8m), per-level gap 테이블로 캐시 통합 유지, `Metrics/terrain_level` 로그, 토글 OFF=현재 동작 불변. 스모크 통과(10 lane 빌드·terrain_level 로그·무오류). **가설**: 잘 되는 것(고정 trot+TAMOLS 발판 추종) + 빠진 것(평지부터 걷기 배우는 적응형 커리큘럼) 조합 → 균형 병목 해소.

**★★iter 1500 결과 = 커리큘럼 정공 검증(2026-08-01)**: iter ~1100 breakthrough(RL 전형 slow-start→급상승) — **epLen 150→~500**(4→10초 생존)·**progress 0.008→0.22**(전진!)·**base_contact 종료 28→7**(낙상 4×↓)·foothold_track→0.56·**Mean reward 22.7**·**★terrain_level 0→0.57**(평지 마스터 후 **gap 레벨 승급 시작**). = **"빠진 것=적응형 커리큘럼"이 균형 병목의 원인**이었음이 검증. dtc_gap(정체 epLen200·progress0.03)와 극명 대조. **정책대로 개선이니 지속**(gait 튜닝 불요).

**★★iter 8751 = 강한 성공 궤적**: **terrain_level 2.17→~4.6-5.5**(level 5 ≈ 11cm gap까지 크로싱, 계속 상승)·epLen 567·progress 0.22·falls 7·reward ~26 유지. = 로봇이 **걷기 + 점점 넓은 gap 크로싱을 순차 학습**. DTC 오프라인+커리큘럼 정공 작동.

**★크래시→robustness 복구 (iter ~11800)**: 물리 blowup(wide gap 극한상태)→NaN이 정책 std를 NaN화 → `RuntimeError: normal expects std>=0` 크래시(cfg에 obs/reward NaN-guard 없었음). terrain_level ~6.5까지 달성·체크포인트 보존. **자율 복구**: ①model_11700 재개 ②NaN-guard robustness 추가(always-on: obs/reward `nan_to_num`+clamp, blowup 종료[비유한/폭주속도/고도이탈], 체크포인트 호환) ③fix 포함 재재개(dtc_curric3). **검증: fix가 크래시난 terrain_level ~6.5 지점을 무크래시 통과**(재climb 0→6.3, epLen562·progress0.26 유지) = robustness가 std-NaN 방지 실증. ★resume 특성: iter카운터·terrain_level은 0 재시작하나 로드된 정책이 실력유지→빠르게 재승급. 커밋=RobotSW_IsaacLab 9c1a2d917(커리큘럼+robustness). 완주까지 terrain_level 최대 도달(→level9=20cm gap) 모니터.

**★★★최종 결과 (dtc_curric3 완주 iter 20000, 2026-08-01)**: **무크래시**(robustness 완전 유지, std 0회). **terrain_level frontier ~6**(오실 5-7.6, **최대 7.6 ≈ 17cm gap 크로싱**)·epLen ~510·progress 0.23·falls 낮음. 최종 체크포인트 model_19700+. = **DTC 오프라인 캐시 + 적응형 커리큘럼 end-to-end 검증 완결**: P2 obs무시(정체)→P2.5 오프라인캐시(발판추종 해결)→커리큘럼(걷기+gap 크로싱 해결). **핵심 교훈**: ①in-loop TAMOLS는 스케일 불가→오프라인 캐시(APT-RL)가 실전 해법 ②"발판 추종"과 "걷기"는 별개층=발판은 캐시로 풀리나 균형/보행은 **적응형 커리큘럼(평지→gap 승급)이 필수**(무거운 로봇) ③high terrain서 물리 blowup=NaN-guard 필수. **frontier ~level6-7(≈13-17cm gap)**: level9(20cm)는 미달=kinematic/gait 한계(발 reach·고정 trot 케이던스). **다음 실험(frontier 돌파)**: ①발 straddle/reach 확대(캐시 landing-relative 재생성·발판 최적화) ②가변 접촉타이밍 gait(고정 trot→적응) ③3D 지형(계단/슬로프)=full base 플랜 활용(캐시에 base궤적 이미 저장). ★자율운영(사용자 2일 부재): pipeline→진단→커리큘럼전환→크래시복구→디스크대응 전부 정책기반 자율수행+기록.
- **남은 것(구조)**: base궤적·접촉 캐시는 저장했으나 첫 배선은 발판만(base=env 램프·gait=고정 trot). 3D 지형(계단/슬로프)=full base 플랜 후속. in-loop 재검토=서버 코어 확대/GPU-batched QP 시.

### P2.7 — DTC 가치 통제검증(ablation) + 3D 계단 실험 ★★2026-08-03~04 핵심
P2.5가 "파이프라인 작동"(커리큘럼+캐시로 걷기+gap크로싱)을 확립했으나 **DTC(발판 planner)가 실제로 가치를 더하는가**는 미검증. 통제 ablation으로 답함.

**① 1차 ablation (평지 gap) — ★캐시가 오히려 방해:** 동일 커리큘럼·robustness서 발판 출처만 다르게(env-var `QUAD17_USE_TAMOLS_CACHE`, worker):
- **DTC-OFF**(Raibert, 무구속 발판): terrain_level **~8.9**(거의 max=20cm gap)·epLen 724·progress 0.34
- **DTC-ON**(TAMOLS 캐시 추종): terrain_level **~6**·epLen 570·progress 0.23
- = **캐시 발판 prescription이 gap 크로싱을 방해**. RL이 캐시 없이 emergent하게 더 잘·빨리 넘음. **이전 "frontier~6=kinematic한계"는 틀렸음**(캐시가 인위적 제한). 원인=gap은 "안 빠지기"라 정밀 발판 불필요→캐시 추종(foothold_track 0.5-0.6)이 정책을 suboptimal 배치에 구속. Raibert는 도달가능+무구속이라 정책이 무시하고 자유 학습.

**② 재설계 (★사용자 정정)**: 1차는 의도한 DTC 프레임워크가 아니었음. 정본 = baseline은 **heightmap-RL**(지형지각+emergent, 발판obs 없음), DTC는 **full TAMOLS**(base+발판+접촉 전부 추종), 비교는 **평지 gap이 아니라 3D 계단**서(평지선 base궤적·접촉이 trivial→full TAMOLS≈발판만). → 계단 실험 재구축.

**③ 계단 실험 배선** (worker+C++): heightmap RayCaster(187ray·**rigid-body 부착**=41s/iter버그 회피)·계단 커리큘럼(`_build_stair_terrain_curriculum` level0평지→level9 step0.15m·step_depth0.35)·**3D 계단 캐시**(`cache_gen_stairs.cpp`: 앞발 step-up tread·뒷발 현단·base z상승, z0_terrain, `test_stairs.cpp`로 TAMOLS 계단 feasibility 사전확인)·full TAMOLS 배선(tread발판+base-z ref[Eq1 z확장] 추종). 명령범위 균등([0.2,0.4] 양쪽). 스모크 통과(obs baseline248·DTC288·계단캐시로드).

**④ 계단 결과 (4-병렬 2-seed, GPU0-3, iter10000)** — ★DTC가 계단선 도움:
| | baseline(heightmap-RL) | DTC(full-TAMOLS) |
|---|---|---|
| iter4000 (matched) | 1.82 (2seed평균) | **5.12** = **2.8× 빠름**(두 seed 일관·겹침無) |
| iter10000 최종 | 5.57 (base42 **6.36**/base7 4.78) | 6.94 (dtc42 **6.54**/dtc7 7.34) — 평균 +25%지만 **seed편차**(clean seed42 동률·seed7만 우세) |
- **DTC 확실한(robust) 가치 = 학습 속도(샘플효율)뿐**: iter4000서 2.8× 앞섬(두 seed 일관·겹침無). **최종 점근성능은 동률** — clean seed42는 baseline이 6.36까지 따라잡아 dtc 6.54와 사실상 같음(tee-log 최종 base42 6.43·dtc42 6.3~6.8). "+25%"는 seed7 하나(base7 4.78 정체)가 끌어올린 것.
- **DTC가 stuck-seed 방지 가능성(잠정)**: base7이 4.78서 정체, DTC 두 seed는 6.5-7.3 도달. 단 **2-seed라 'reliable' 단정 불가**(seed42는 baseline도 6.36 도달). (4-병렬 CPU 경합 無, 전부 ~2.2s/iter.)

**★★핵심 결론 — DTC 가치는 task-dependent:**
| 지형 | planner(DTC) 효과 | 이유 |
|---|---|---|
| **평지 gap**(관용) | ❌ **방해** | gap="안빠지기"→RL emergent 쉬움, 발판 prescription 구속 |
| **계단**(정밀) | ✅ **도움**(샘플효율 2.8× @iter4000·2seed일관; 최종 점근은 동률·seed편차) | tread 정밀발판+base상승 필요→RL 혼자 느림, planner가 초기 학습 가속 |

= **DTC는 "무용"도 "만능"도 아님. 정밀 발배치 지형(계단·stepping stone)엔 크게 기여(특히 샘플효율)·관용 지형(gap)엔 방해.** 이 nuance는 3D 재설계로만 드러남.

**★★한계(2026-08-04, GIAC 퇴화) — P2.7은 "balance-aware full DTC"가 아니었음:** P2.7의 TO-on 참조 = **trot 캐시**인데, trot의 2발(대각) 국면은 지지영역이 **선분(1D)으로 붕괴** → 두 접촉력이 대각선 축 모멘트를 못 만듦 → **GIAC가 roll/전복 DOF를 최적화 못 함(퇴화)**. 즉 P2.7 발판은 **지형-aware이지 균형-최적화가 아님**. → P2.7이 검증한 건 **"저-균형 지형(계단)서 지형 발판이 샘플효율 도움"**까지고, **GIAC/균형의 가치는 미검증**("full-TAMOLS DTC 검증"이라 부른 건 과함). 계단은 준-1D(높이만 상승)라 2D 균형 스트레스가 작아 GIAC 퇴화가 덜 물어 trot 참조로도 성립했으나, **GIAC 퇴화가 진짜 무는 건 stepping(2D 정밀균형)** — 실제로 trot+snap-발판은 **평지서도 낙상**(place_err↑·base_contact 15). **GIAC가 유효(삼각 지지)한 건 3발+ = walk**이므로 balance-aware DTC의 진짜 검증 = **walk RL(정적크롤) + full-TAMOLS-stepping(walk 정식화 캐시)** = 현재 진행중(§P2.9 예정). cmp_gait서 walk 발판이 trot보다 tread 10× 정밀했던 것도 이와 일관.

**자산·주의**: `tamols/{cache_gen_stairs,test_stairs}.cpp`·`tamols_stair_cache/`·env 토글 `QUAD17_{HEIGHTMAP,FOOTHOLD_OBS,TERRAIN_KIND,FULL_TAMOLS,USE_TAMOLS_CACHE}`. 학습 dir=`2026-08-03_{14-52-36 base42·17-34-56 dtc42}`. **비교영상=`simulation/docs/stair_compare.{mp4,gif}`** — iter4000(동일 학습량) 두 정책을 같은 level-4 계단(step 6.7cm)서 나란히: DTC-ON=계단 등반 / DTC-OFF=계단 밑 정체(샘플효율 실증). 녹화용 서버패치(RobotSW_IsaacLab, 미커밋): play.py obs-CSV 비활성(num_envs>1서 죽던 것), env `QUAD17_PLAY_TERRAIN_LEVEL`(play시 지형레벨 고정+커리큘럼 freeze), viewer asset_root 로봇추적. ★런치 주의: 동일-초 발사 시 IsaacLab이 같은 타임스탬프 dir 충돌(base7/dtc7 겹침)→발사 간 sleep 필요. GPU0/1/3=사용자 임시허가(내일까지)로 4-병렬, 이후 GPU2-only 복귀.

### P2.8 — TO 정식화 선택: walk-TO 레퍼런스 (★2026-08-04, Go2 이식 세션서 도출)

Go2 이식 조사(`pipeline_tamols.html` §6)서 solve_fast 정식화 버그 4건(base-z·y_min·HQP·walk-CoM)을 규명하며 **DTC용 TO 정식화 방향이 정해짐**:

- **base-z·실행은 RL/WBC 담당, TO는 참조만.** 순수-TAMOLS(TO→WBC직접)는 base 궤적이 실행가능해야 했지만, **DTC는 RL이 실행·robustify**하므로 TO는 실행가능성 부담이 없다. 발견한 base-z 버그(계획이 불가능한 0.52 명령)는 TO config 버그였고, base 높이 추종·안정화 자체는 실행층 몫. → **DTC TO는 base z 레퍼런스(명목+지형)만 내면 됨.**
- **정밀 험지(계단)엔 walk 정식화 > trot.** walk(정적안정·한 발씩)가 발판을 정밀하게 놓는 깔끔한 레퍼런스를 줌(A가 계단 오른 게 walk). trot 레퍼런스는 동적이라 RL이 정밀 발배치 배우기 어려움. **P2.7 결론(계단=정밀배치서 DTC 가치)과 정합.**
- **walk-CoM·크롤순서 수정이 valid walk 레퍼런스의 전제**(커밋 11ad476). solve_fast가 walk서 CoM을 한쪽 0.32m로 몰던 근본(GIAC slack 느슨·CoM centering 부재) → cost_residuals에 **지지발 centroid 추종 비용(COM_W)** 추가 + 크롤순서 q.legs 정합(WALK_QLEG)로 평지 walk falls=0 성립. CoM 쏠린 walk는 레퍼런스로도 무의미하니 이 수정이 DTC-walk 직결.
- **오프라인 캐시라 솔버 iteration 넉넉 → 발판 품질↑.** online RTI 5-iter 제약이 solve_fast 발판 품질을 낮췄으나, DTC 오프라인 캐시는 수십 iter 가능 → 더 나은 발판.
- **TO의 값어치 = 발판+접촉스케줄**이지 실행가능 base 궤적이 아님. 배포 험지는 A+footScore(selectFoot)로 가고(계단 실측 A 완주 vs solve_fast 0), TO는 정밀 험지 DTC 레퍼런스로 재사용.

**요약: DTC TO 정식화 = walk(CoM-centering+크롤순서 수정) + footScore 발판, 오프라인 캐시(iter 넉넉). base-z/실행=RL. 평지·관용 gap엔 DTC off(순수 Raibert-RL, P2.7).**

### P2.9 — stepping-stone: walk 게이트(10×)가 핵심·발판(TO)=초기 샘플효율만 (★2026-08-05~06, gap/계단과 일관)
§P2.8대로 **균형-aware DTC는 walk(3발지지=GIAC 유효)라야 검증 가능** → 정밀지형(2.5D 이산 stepping-stone) + walk 게이트 RL로 직접 검증. (사용자 질문서 도출: "gap이 발판의 진짜 이유"·GIAC 마찰-sway·2발 GIAC 퇴화.)

**셋업**(worker): stepping-stone 지형 커리큘럼(돌 0.55m/gap0.02 → **0.14m/gap0.30m**, level0~9)·2.5D 이산 발판 snap-selection(최근접 돌)·**발판 4색 마커 시각화**(play default). **walk(정적크롤) 게이트 모드**(4-phase FL→HL→FR→HR·duty0.25·**항상 3발지지**·period0.7, trot byte불변 gated). 지표=terrain_level(통과)·foothold_place_err(추종)·epLen. env토글 `QUAD17_{GAIT=walk,TERRAIN_KIND=stepping}`.

**★★핵심결과1 — walk ≫ trot on stepping (~10×)**:
| | 최종 terrain_level (stepping) |
|---|---|
| trot RL (heightmap) | **0.32 정체**(≈평지, 큰돌만) |
| **walk RL (heightmap)** | **3.37**(돌 0.40m·gap0.12m 실제 통과) |

→ walk(3발지지·GIAC 유효)가 정밀지형 실행기, **trot(2발·GIAC 퇴화)은 부적합** = GIAC 논의 실측 확증. walk RL 잘 학습(reward67·epLen820·낙상1.6). 짤=`walk_vs_trot_stepping.{mp4,gif}`.

**★★핵심결과2 — walk 게이트 품질(자세)이 발판 가치를 좌우: 크라우치=중립 → 직립=도움 (★2차 정정)**:
- (1차) 저품질 walk(크라우치)에선 발판 delay-then-catchup=**중립**(walk_fh 1.97≈walk_off 1.99 @iter7000). 앞선 "발판이 해"(iter2000 조기중단)는 부트스트랩 조기판단 오류(사용자 지적 맞음).
- ★**(2차, 결정적) walk가 저품질이었음**: 자세보상 약해(base_height −10·orient −1) RL이 **저CoM 웅크림-셔플로 gaming**(terrain_level은 traverse만 재고 자세는 안 잼). base_height_target=0.82(지형보정 정상)인데도 −10이 약해 웅크림. **재균형(base_height −10→−25·orient −1→−3·swing 0.08→0.11, biped 재균형#4 동일 처방)** → 직립 크롤 회복(base 목표서 3cm·`walk_rebalanced.mp4`).
- ★★**깨끗한(직립) walk 발판 = 초기 샘플효율만, 최종은 baseline이 더 높음(발판이 점근 제약)**: 재균형 walk+발판(walkrb_fh) vs baseline(walkrb_off) — **iter5000서 1.58 vs 0.89(발판 ~1.8× 앞섬, 샘플효율)** 이나 **iter8000부터 발판 정체(2.82→2.98) BUT baseline 계속 상승 → 최종 3.72 vs 2.98 = baseline 추월/우위**. 게다가 재균형 baseline(3.72) > 크라우치 walk(3.37) = 직립이 셔플보다 통과↑.
- → **정밀지형서도 발판(TO) 가치 = "초기 샘플효율 레버일 뿐, 최종능력 개선 못 함(오히려 제약)"** = **gap(emergent 이김)·계단(P2.7 샘플효율만·점근동률)과 완전 일관.** RL 창발 발배치가 점근서 동급~우위. **정밀지형 근본=게이트(walk, 10×)이지 발판 아님.**
- ★★방법론 교훈(중요): **중반(iter5000-7000)서 "발판 도움" 판단은 성급**했고 최종(iter10000)서 뒤집힘(baseline 추월). **점근은 끝까지 학습해야 판정**. 또 **terrain_level은 traverse만 재고 게이트 품질(자세)은 안 잼** → 크라우치 셔플을 수치상 "성공"으로 오판 → 반드시 렌더로 자세 확인. (본 세션서 이 두 함정에 각각 걸렸다가 최종·렌더로 교정.)

**★교훈**:
- **정밀지형의 근본 = 올바른 게이트(walk/정적안정)이지 발판(TO)이 아님.** 게이트=10×, 발판=~중립. DTC 발판의 진짜 가치는 **RL이 혼자 못 푸는 지형서만**(우리 stepping은 walk RL이 스스로 품) → Kim2025 샘플링 플래너=진짜 이산지형용.
- **GIAC 한계 3종(사용자 도출)**: ①2발(trot)=지지 선분붕괴로 퇴화(roll 미최적화) ②마찰콘이 lateral sway 허용→CoM 안 당기면 드리프트(COM_W 필요, walk 0.017 vs trot 0.064 sway) ③gradient-TAMOLS 2.5D 이산발판 약함(~70% 돌위)이나 **발-on-stone은 RL 추종 몫이지 planner 실패 아님**(사용자 정정).
- **방법론**: **walk 부트스트랩(느림) 구간서 조기판단 금지** — walk_off·walk_fh 둘 다 iter4000까진 terrain_level 0였다가 급가속. 서버 WiFi 단절에도 tmux 학습 생존(uptime 무영향).

**자산**: stepping 지형·walk gait mode(env, worker·gated)·`test_stepping.cpp`(발판 feasibility)·`cmp_gait.cpp`(walk/trot 발판 정량)·`walk_vs_trot_stepping.{mp4,gif}`. 학습 dir `2026-08-04_17-21-15(walk_off)`·`2026-08-05_09-39(walk_fh)`. 서버 미커밋.

### P3.0 — Go2 4-팔 통제실험: **값은 몸통참조에 있고 발판에는 없다** ★★2026-09-11~17

**설계.** 징검돌(GO2_TERRAIN=stepping, HVAR 0.05, walk 고정 게이트, seed 42, 3000 iter, 4096 env)에서
**한 번에 한 변수만** 바꾼 네 팔. 판정은 강건성 스위트 **n=24 × L7/L8/L9**(terrain_level 미사용 — 속도를
점수로 세고 순위가 뒤집혀 이 라인에서 네 번 깨졌다).

| 팔 | 발판 참조원 | 몸통 참조 | 게이트 |
|---|---|---|---|
| G | 기하 Raibert + 최근접-돌 스냅 | TAMOLS 캐시 | (기준선) |
| T | TAMOLS, **스냅 없음** | TAMOLS | `GO2_STEP_TAMOLS_FH=1` |
| TS | TAMOLS + **동일 스냅** | TAMOLS | `+ GO2_STEP_TAMOLS_FH_SNAP=1` |
| N | 기하 스냅 | **없음** | `unset GO2_STEP_TAMOLS*` |
| S | 기하 스냅 | **해석해**(등속·등고·위상동기 sway 8cm p-p) | `GO2_SREF_BASE_ANALYTIC=1` (진행중) |

**결과 (n=24).**

| 팔 | 몸통참조 | L7 CROSS | L8 CROSS | L9 CROSS | L9 진입 | L9 reset |
|---|---|---|---|---|---|---|
| **G** | TAMOLS(지형 조건화) | 23/24 | **22/24** | **7/24** | **19/24** | 288 |
| TS | TAMOLS(발판도 TAMOLS) | 23/24 | 22/24 | 1/24 | 14/24 | 427 |
| **S'** | 캐시 **위상평균**(게이트 고정만) | 22/24 | 17/24 | **0/24** | **5/24** | 576 |
| **N** | 없음 | 23/24 | 19/24 | **0/24** | **2/24** | 639 |

- **발판 효과**(G vs TS): L9 CROSS 7/24 vs 1/24 **p = 0.048 — TAMOLS 가 지는 방향**
- **몸통참조 전체**(G vs N): L9 진입 19/24 vs 2/24 **p = 1e-6**
- **★지형 조건화**(G vs S'): L9 진입 19/24 vs 5/24 **p = 0.00012** · CROSS 7/24 vs 0/24 p = 0.0094
- **참조 채널 존재만**(S' vs N): L9 진입 5/24 vs 2/24 **p = 0.42** · L8 17 vs 19 p = 0.74 → **무의미**
- L7 은 전 팔 동률, L8 은 유의차 없음

**결론 3줄.**
1. **TAMOLS 발판은 이득이 없고 난도 지형에서는 해롭다.** P2.7(gap)·P4(Go2 gradient-TO)에 이은 3번째
   replication이며 통제가 가장 깨끗하다. 이유: 징검돌·gap 은 **정답 발판이 지형에 새겨져 있어**
   "최근접 돌 클램프"가 이미 최적에 가깝다 = 지각층이 문제를 다 푼다.
2. **★TAMOLS 몸통참조는 유의미하되, 값은 "채널 존재"가 아니라 "지형 조건화"에 있다.**
   게이트 고정 성분만 남긴 팔 S'(위상평균)는 **참조가 아예 없는 팔 N 과 통계적으로 구분되지 않는다**
   (p = 0.42). 팔 G 만 홀로 떨어진다. 즉 값은 **매 사이클 지형을 보고 다르게 기울이는 것** 자체다.
   실측 정합: sway 가 셀별 9.7cm 인데 위상평균은 4.4cm — 상쇄된 **5.3cm 가 지형 조건화 성분**이고,
   그 5.3cm 를 빼자 L9 진입이 19/24 → 5/24 로 무너졌다. **= GIAC 최적화의 실제 값.**
3. **효과는 L9 에서만 난다** = 지지 다각형이 좁아져 **균형이 실제로 구속될 때만** 몸통계획이 값을 낸다.
   쉬운 지형의 무차이는 TAMOLS 무용이 아니라 **제약 미구속**이었다.

**★함정 — 스냅이 제2변수였다.** 팔 T 는 terrain_level 0.000 으로 완전붕괴했는데 원인은 TAMOLS 가
아니라 **최근접-돌 스냅 누락**이었다(팔 TS 로 스냅만 켜니 8.154 ≈ 팔G 8.100 회복). 배선버그 3가설
(다리순서·타겟래치·힙기하)을 전부 데이터로 기각한 뒤 `(타겟−힙)` 전후성분에서 진범을 찾았다 —
팔G 는 다리별로 크게 다른데(뒷발 +0.25/+0.14) 팔T 는 네 발 균일 +0.08. 기하의 비균일은 **이산 돌격자
스냅의 양자화**에서 나온다. 자세히는 메모리 `foothold-snap-confound`.

**비용 실측(별도).** 정밀지형 20-iter solve = **4.4~4.7 초**(평지 벤치 12.7ms 는 350× 낙관 — 대표성 없음).
학습 인루프 23× · 배포 사이클당 1회 4.5~11× 부족. 발판을 지각층에서 p_init 로 급여해도 **싸지지 않았다**
(5,211ms, feasible 130/192 로 악화). 단 이는 "p 를 결정변수에서 제거"의 대리 지표일 뿐 본 주장은 미검증.
메모리 `tamols-solve-cost-wall`.

**미측정 범위(결론을 여기 밖으로 일반화 금지).** zero-shot 신규 지형 · 제약 보증/인증 · **지면 경사**
(마찰추가 처음으로 구속되고 접촉이 진짜 비평면이 되며 "최근접 돌"이 도달불가가 될 수 있는 조건) ·
스냅 대상이 없는 연속 험지. 특히 **경사가 발판 결론이 뒤집힐 수 있는 유일한 후보**다.

**★팔 S 1차 시도는 무효(방법 기록).** 처음엔 sway 를 **추론으로** 사인파(진폭 8cm, +y 시작 2로브)로
만들었는데 캐시 실측은 **−y 단일 로브**였다 — 부호·로브수 모두 반대. 스윙 다리 쪽으로 몸을 기울이는
꼴이라 정책이 STALL 을 택했다(낙상 아님: base_contact 0.42 vs time_out 3.33, 평가 0/24). 게이트 배너도
빠뜨려 실행 여부를 로그로 증명할 수 없었다. **팔 S'는 추측을 전부 제거** — 같은 캐시를 (phase_x,
phase_y) 로 평균내 지형 조건화만 빼고 (vx, level) 조건은 남겼다. `BASEREF_SRC` 배너 추가.

**결론 — 값의 위치.**
```
발판 최적화        불필요 · 난도지형 유해   (p=0.048, TAMOLS 가 짐)
몸통 참조 "존재"   기여 없음               (S' vs N, p=0.42)
몸통 "지형 조건화" ★유의미                 (G vs S', p=0.00012)
발현 조건          L9 에서만 = 균형이 실제로 구속될 때
```
앞선 "TAMOLS 투자가치 없음"은 **full TAMOLS 기준에서만 유효**했고 최종적으로 틀렸다. 값은 존재하며
위치가 특정됐다. 다음 과제 = **발판을 지각층에 넘기고 몸통만 푸는 TO 를 싸게 만들 수 있는가**
(`tamols_qp.hpp` 에 `fix_p` 옵션 추가해 측정 중).

**★★★결론 철회 — 학습 seed 재현 실패(2026-09-22).** 위 §P3.0 의 핵심 주장들은 **학습 seed 42
단일 런**에서 나온 것이다. seed 7 로 팔 G·팔 S' 를 재학습해 같은 스위트로 재면 **부호가 뒤집힌다**:

| L9 | seed 42 | seed 7 |
|---|---|---|
| 진입 G vs S' | 19/24 vs 5/24 · p=0.00012 (G 승) | 7/24 vs 18/24 · **p=0.0034 (S' 승)** |
| CROSS | 7/24 vs 0/24 · p=0.0094 | 2/24 vs 7/24 · p=0.14 |

두 seed 모두 **유의**하나 **방향이 반대**. 합산 26/48 vs 23/48 **p=0.68 = 효과 없음**.
→ **"몸통 지형 조건화가 유의미" 는 단일 학습seed 산물이었다. 철회.**

**동일하게 의심해야 할 것**: 발판 결론(G vs TS, p=0.048)·팔 N 붕괴(진입 2/24) 도 **전부 seed 42
단일 학습**이다. TS·N 은 seed 7 재학습을 하지 않았다. 게다가 seed 간 흔들림 폭(5/24→18/24 = 13/24)이
팔N-팔G 격차(2/24→19/24 = 17/24)와 **같은 크기**라 팔 N 결과도 seed 잡음과 구분되지 않는다.

**무엇이 남는가**: L7·L8 은 두 seed 에서 안정적이고 어느 쪽도 유의하지 않다(L8: G 22/20 vs S' 17/18,
p=0.14/0.72). **L9 는 모든 정책이 실패 경계에 걸친 구간이라 학습seed 분산이 효과를 지배한다.**
현재 데이터로 말할 수 있는 것은 **"L7·L8 에서 팔들은 구분되지 않는다"** 뿐이다.

**필요한 것**: 팔당 학습 seed 3~5 개. 4팔 × 추가 3seed × 3.5h ≈ 42h GPU. 또는 비교를 2팔로 좁혀
5seed 씩. **평가 24 seed 는 롤아웃 seed 라 학습 분산을 전혀 잡지 못한다** — 이 구분을 놓친 것이
이번 실패의 핵심이다.

**★집계 인프라 주의.** `rs_table.py` 는 d-계열 전용으로 작성돼 새 정책명이 들어올 때마다 **조용히
데이터를 버린다.** 이번 세션에 3번 걸렸다: ① 정책명 정규식 `(d\d+)` 하드코딩 → 48 롤아웃 전수 스킵
② `SEEDS` 고정집합 [42,7,1,3,2,5,11,23] → 우리 1~8 과 교집합 5개만 세어 n=8→5 ③ `arm[A-Z]+` 가
숫자 못 받아 `armS2` 행 누락. 모두 "0 rollouts"·"행 없음"으로 드러나 잡았으나, **n 이 조용히 줄어드는
형태였으면 못 잡았다.** 새 팔 추가 시 반드시 롤아웃 수를 먼저 대조할 것.

---

### P3 — 지형·강건성·CVAE (예정)
지형 heightmap → 발판에 물리 압력(갭=헛디디면 낙상), Kim2025 competitive CVAE 커리큘럼, 실제 TAMOLS 참조(`tamols_02leg.py`)로 절차적 발판 대체 ← **P2.5서 실현(오프라인 캐시)**, height scan(발→목표 직선).

---

## 4. 하이브리드 맥락 — 5패턴 중 DTC=패턴 A (나머지 요약)

DTC(패턴 A) 외 4패턴은 상황별 보조:
- **B (RL 상위/MPC 하위·계층)**: RL이 발디딤·gait 의도, MPC/WBC가 물리 실현. RLOC·GLIDE.
- **C (RL을 MPC 내부 주입)**: C-1 학습 가치함수=terminal cost / C-2 잔차 모델(=액추에이터 네트워크·**TOWR 오프라인 참조**) / C-3 RL로 MPC 하이퍼 튜닝.
- **D (MPC=안전계층)**: predictive safety filter·CBF, 또는 MPC 출력 위 RL 잔차(80/20). 실기 보험.
- **E (샘플링 MPC + 학습 prior)**: MPPI를 학습 정책으로 warm-start. sit/getup의 gather seed가 수동 버전.

**역할 분담 원칙**: "RL이 잘 되면 내 궤적·모델·안전계층 덕이고, 안 되면 내 MPC가 백업." MPC 담당 = 모델·레퍼런스 생성기·보상설계 자문·하위 WBC·safety filter. (상세 용어집·문헌은 `dtc-aptrl-papers`·`rl-backbone-8papers` 메모리.)

---

## 5. MPC 교사 공급 · OCP 실시간화 (요약)

**핵심**: RL에게 주는 것은 "실시간 제어기"가 아니라 "궤적·신호". 교사는 오프라인이라 실시간성 불요 → RL 협업 즉시 가능. A(Convex MPC+WBIC, 실시간)는 학습 루프 내 online 참조(②), aligator full-dynamics는 오프라인 정밀 궤적 라이브러리(①).

- **점프 live-solve OCP = 완료**(커밋 57e4e7b·c5cf034→fd8bc95→6bb3142): C++ in-loop solve(캐싱 464→122ms), crocoddyl→aligator 하드 BoxConstraint 이관, wbic_jump(폐루프 GRF) apex 0.546·falls=0.
- **보행 실시간 OCP = 실험 후 미채택**(2026-07-13): B 호라이즌 다이어트 T50=53Hz(marginal)·T40붕괴. 병목=OCP 안정/수렴(02_Leg 무거운 다리 ill-conditioning). **실시간 보행은 A로 충분** → OCP는 저속 정밀 접촉기동 전용.

---

## 6. 로드맵 (DTC 중심)

- **H0.5 sim2sim 브리지** — RL 정책 아티팩트(관측 스펙·정규화·게인) → MuJoCo(C++ 뷰어) 배포. 함정=quaternion(wxyz vs xyzw)·속도 프레임·관절 순서/부호 매핑. RGA `RobotSW_IsaacLab` 클론 확보(R_Skeleton=R.pet·hind_leg 태스크·ActorCriticRMA·obs 46·ONNX+estimator).
- **H1 sit/getup DTC화** — gather-seeded MPPI 궤적 레퍼런스로 RL tracking(축소판 DTC).
- **H2 보행 DTC화** — A/run 프리셋을 레퍼런스 생성기로, RL tracker 배포. WBIC 하위(패턴 B) vs 완전 증류(패턴 A)는 예산·안전으로 결정. ← **현재 P2가 이 방향의 quad17 실증.**
- **H3 안전계층** — RL 배포 시 MPC predictive safety filter 상시.

---

## 참조
- **Jenelten "DTC"** (Science Robotics 2024, arXiv 2309.15462) — 원전. PDF=`quad/backup/docs/2309.15462v2.pdf`.
- **Kim 2025 (Raibo)** (arXiv 2506.02835) — 목표 아키텍처. PDF=`~/다운로드/2506.02835v1.pdf`.
- 코드: `RobotSW_IsaacLab/.../direct/quad17/`(env·cfg·agents) · `rga.py` QUAD_17DOF_CFG · `tamols/tamols_02leg.py`(TAMOLS 참조).
- 관련 문서: `rl_module_train.html`(모듈 학습법) · `D1_OCS2_개발리포트.md`(모델기반 배포 NMPC) · `TOWR_개발리포트.md`(모델기반 상한→RL).
- 메모리: `dtc-17dof-development`(P0~P3 작업기록) · `dtc-aptrl-papers` · `rl-backbone-8papers` · `perceptive-nav-tamols`.
