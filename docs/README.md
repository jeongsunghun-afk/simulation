# 02_Leg 문서 인덱스 — 여기만 보면 전부 찾음

흩어진 문서·스크립트·메모리를 이 한 곳에서 매핑. **컨트롤러 배포 = 17-DOF (C++/Python)**.

## 📄 문서 (simulation/docs/)

### 전체 파이프라인
| 문서 | 역할 | 형식 |
|---|---|---|
| **[pipeline_fullstack.html](pipeline_fullstack.html)** | **전체 시스템 프레임워크** — 형태→기능(기구 4요소)·4계보(CI-MPC·DTC·APT·AMP) 통합 배포 아키텍처·지각/SLAM/계획/제어 층구조(F1–F3)·**실제 코드 대조·수정사항(F4–F5, DOF R¹⁶·다중 repo 통합)**·**부록: nav 통합·RL 정책 구조(구 pipeline_nav·pipeline_rl 흡수, F6–F7)** | 아티팩트 |
| **[rl_module_train.html](rl_module_train.html)** | **RL 모듈 구축·학습 가이드** — 제어층 정책을 **어떻게 만들고 학습하는가**. 두 참조골격(DTC 계획↔제어·APT-RL 3단계)·학습 3단계(표현/RL+RMA/지각증류)·**모듈 카탈로그(ActorCriticRMA: actor·estimator·history/priv_encoder·critic — 배포 vs 학습전용)**·**보상설계 실증(2점 평발 plateau→shuffle→limp 3재균형)**·예정 Depth CNN+GRU·PACE 정합. fullstack이 개요면 이건 **학습법 심화** | 아티팩트 |

### 제어기별 (pipeline · params)
| 제어기 | pipeline | params | 비고 |
|---|---|---|---|
| **A** (MPC+WBIC, **배포**) | [pipeline_a.html](pipeline_a.html) | [params_a.html](params_a.html) | 솔버 문제정의·I/O·값전체+파리티+튕김조절 |
| **B·C** (OCP, 연구·분석) | [pipeline_bc.html](pipeline_bc.html) | (pipeline 내) | Kinodyn OCP+TSID / FullDyn OCP+Riccati — 각 **구조·입출력** |
| **D1** (OCS2 통합 NMPC, **포팅·진행**) | [pipeline_d1.html](pipeline_d1.html) | (pipeline 내) | OCS2 legged_robot→02_Leg MuJoCo 포팅. 동적 TROT falls=0·16-DOF·perceptive 3a. `pipeline_tamols` 양식(대조용) |
| **제어기 비교** (A·B·C·D1·CI) | [제어기_비교.html](제어기_비교.html) | — | 5종 **구조·솔버·I/O·실시간·지형·성능** 대조 + 지형유형 정직비교 |
| **CI** (Contact-Implicit, **종결**) | [pipeline_ci.html](pipeline_ci.html) · [params_ci.html](params_ci.html) | — | 종결: 로버스트 험지 부적합. 상세=아래 개발리포트 |

### 🔬 모델기반 트랙 개발리포트 (통일 `<트랙>_개발리포트.md`)
각 제어기/트랙의 개발 기록·결정·현재상태. (2026-07-30 이름 통일·최신화·트림. 원본 백업=`.docbackup_20260730/`.)

| 리포트 | 트랙 | 상태 |
|---|---|---|
| **[DTC_개발리포트.md](DTC_개발리포트.md)** | **DTC**(MPC 교사 + RL 추종) — 17-DOF quad 위 P0(자산)·P1(속도워커) 완료·**P2(발판추종 tracker) 진행**·P3(지형·CVAE) 예정. 목표 아키텍처=Kim2025(Raibo). 하이브리드 5패턴·H0~H3 로드맵 포함. (구 MPC_RL 하이브리드 전략 리포트 재초점) | ★활성 |
| **[D1_OCS2_개발리포트.md](D1_OCS2_개발리포트.md)** | **D1**(OCS2 통합 perceptive NMPC → 02_Leg) — Phase 1·2 완료·**Phase 2b 동적 TROT 성공**(0.3m/s·13s+·falls=0, 2bb8dd4). 다음=Phase 3 perceptive. **B(quad_centroidal) perceptive NMPC 승격(보류)** 흡수(§0.1) | ★활성 |
| **[CI-MPC_개발리포트.md](CI-MPC_개발리포트.md)** | **CI-MPC**(Contact-Implicit) — 로버스트 험지 부적합 확정, 수행분(해석그래디언트·서기/눕기/앉기 자세)만 남기고 접음. 이후=TOWR+RL | ★종결 |
| **[TAMOLS_개발리포트.md](TAMOLS_개발리포트.md)** | **TAMOLS** — whole-body 온라인 추종(④W_AM 해결·⑤리앵커링 미해결) + **갭 크로싱·C++ TAMOLS 솔버 완성**(§7, 구 갭크로싱 리포트 통합). full TAMOLS 단발 추종=막다른길 → RL 피벗 | 종결→RL |
| **[TOWR_개발리포트.md](TOWR_개발리포트.md)** | **TOWR** — 오프라인 지형 planning(수렴)·추종/폐루프 조사로 모델기반 상한 확정 → robust 불연속 험지=RL. TOWR·훅=RL 교사/오프라인 참조 자산 | 피벗→RL |

### 실행 · 품질 · 기타
| 문서 | 역할 | 형식 |
|---|---|---|
| **[RUN.md](RUN.md)** | 실행 레시피(빌드·GUI·헤드리스·지형·회귀·데모) | md |
| **[MAINTENANCE.md](MAINTENANCE.md)** | 품질 프로세스(하네스·스킬·파리티·문서동기화 규칙) | md |
| **[datasheet_load_17dof.html](datasheet_load_17dof.html)** | 관절별 τ·ω peak/RMS 부하 데이터시트(trot·walk, 실모터 한계 대비) | 아티팩트 |
| **[sim2real_checklist_17dof.html](sim2real_checklist_17dof.html)** | 실기 이식 갭(액추에이터 물리·미모델·운용) | 아티팩트 |
| **[bringup_sequence.html](bringup_sequence.html)** | **실기 브링업 순서** — 모터 격리(위치·토크·ID: 백래쉬·τ_c·b·Rotor_I·지연)→다리(J·G·M,C→F)→로봇 층별 검증 런북. **F↔모델 순서 정정·백래쉬 우선·F센서 불요**(F=WBC QP 결정변수) | 아티팩트 |
| **[biped.html](biped.html)** | **biped** — 2족 MPC+WBIC. 점발/평발 접촉모드·1점/2점 전환·개발여정(C++ 배포) | 아티팩트 |
| [RECORDING.md](RECORDING.md) · [DEVLOG.md](DEVLOG.md) | 화면 녹화 가이드 · 개발일지(연대기) | md |

문서 원칙: **값은 params에만**, 나머지는 포인터로 위임(중복 금지). params=값 / pipeline=수식·구조 / sim2real=실기 갭 / **개발리포트=트랙별 개발기록·결정**. nav·RL은 fullstack으로 통합(개별 pipeline_nav·pipeline_rl 폐지).

## 🔧 코드·도구 (simulation/)
| 위치 | 내용 |
|---|---|
| `quad/quad_mpc_wbic_17dof.py` + `teleop_gui_17dof.py` | **배포 A′(Python)** |
| `quad/cpp/` (quad_control·mpc·trot_controller·trot_sim·trot_view) | **배포 C++(1kHz)** |
| `quad/run_gui.sh` · `run_gui_py.sh` | **원샷 런처**(뷰어+GUI, 기본=종합코스). C++ / Python |
| `quad/gen_jump.sh` | 점프 궤적 생성(J1 OCP→J2 변환→/tmp/jump_traj.txt) |
| `quad/cpp/verify.sh` | **하네스** — 표준 회귀 배터리(`./verify.sh [--python]`) |
| `quad/PARAMS.md` | 파라미터 원본(md, = params_a.html 소스) |
| `quad/make_terrains.py` + `quad_terrain_*.mjcf` | 테스트 지형(3레인 course·계단·험지·마찰·gap·stepping·soft) |
| `quad/record_demo.sh` | 데모 녹화 |
| `.claude/skills/controller-review/SKILL.md` | **스킬** — 검수 SOP(`/controller-review`) |
| `quad/offline/jump/` · `offline/getup/` | 오프라인 궤적(점프 OCP·기립 gather) 생성 파이프라인 |
| `quad/tools/` | 모델 빌드(build_real_quad_17dof·gen_sphere_17dof·plot_perfoot) |
| `quad/tamols/` · `quad/cpp/tamols/` | TAMOLS Drake 레퍼런스 + C++ 솔버(실시간+cold-start) — 상세=TAMOLS 개발리포트 |
| `quad/towr/` (+`towr_ext/`) | TOWR planning(CasADi+원조 ifopt) — 상세=TOWR 개발리포트 |
| `quad/ocs2_02leg/` · `quad/ocs2_ws/` | D1 OCS2 포팅(우리 소스+3rd-party) — 상세=D1 개발리포트 |
| `quad/ci_mpc/` · `cpp/ci_mpc/` | CI-MPC(종결) — 상세=CI-MPC 개발리포트 |
| `quad/research/quad_mpc_wbic.py` + `teleop_gui.py` | 14-DOF(구 본선, 연구) |
| `biped/` | **뒷다리 2족 보행**(MPC+WBIC event-DCM·base-frame·GUI). Python+**C++ 배포**(cpp/, 파리티 1e-11). 실행 `biped/run_gui_biped.sh`(Py)·`run_gui_cpp.sh`(C++). 상세=`biped/README.md`·메모리 biped-mpc-reimpl |
| `simple_mpc/` | 구조 B/C(FullDynamics·Centroidal, marginal 연구) |
| `gait_sim/` | ★구 gait_sim 연구노트(v13/v14 다수 md) — 히스토리, 아카이브 후보 |

## 🧠 메모리 (~/.claude/.../memory) — 프로젝트 방향·진단
`MEMORY.md`(인덱스) + biped-wbic-mpc-project(메인) · dtc-17dof-development(DTC 트랙) · 02leg-motor-spec · joint-load-trot-walk · quad-abc-io-structure · perceptive-nav-tamols · b-elevation-tamols-towr-track · ci-mpc-track · haunch-sit-posture · getup-trajopt-wip · rbq-sdk-reference · no-unphysical-sim2real-hacks · sim2real-checklist(·17dof).
→ sim2real 체크리스트 **값·표는 위 HTML이 정본**, 메모리는 결정·통찰만(중복 슬림).

## 🗺️ 로드맵 / 향후 기능
| 문서 | 내용 | 상태 |
|---|---|---|
| **[WBC_layer2_standalone_fix.md](WBC_layer2_standalone_fix.md)** | 2층(TAMOLS→WBC) baseline. standalone WBC z침하=QP 힘품질. 논문(계층적 WBC·GM-observer) 해법. "최소 baseline→RL" 전략. | 진행·게이트 |
| **[pipeline_tamols.html](pipeline_tamols.html)** | TAMOLS 계획층(A 실행스택 위 ③④ 주입)·TAM_BASE base-발판 협조 실증·한계·D1포팅/DTC. | 아티팩트 |

## ✅ 유지 워크플로 (요약)
1. 변경 → `cd quad/cpp && ./verify.sh` (회귀 PASS 확인).
2. 파라미터/게인 바꿨으면 `--python`으로 파리티, params 문서 갱신.
3. 스킬 `/controller-review`가 이 절차를 자동 수행(MAINTENANCE.md 기준).
4. 논리단위 커밋(요청 시). 상세=[MAINTENANCE.md](MAINTENANCE.md).

## 🗑️ 정리됨
- (2026-07-30) **개발리포트 6개 → 5개로 통일·최신화·트림** (`<트랙>_개발리포트.md`): B_elevation_perceptive_NMPC → D1에 흡수, MPC_RL_하이브리드_전략_리포트 → DTC로 재초점, `모델기반_갭크로싱_탐색리포트.html` → TAMOLS §7 통합(삭제). 원본 백업=`.docbackup_20260730/`.
- (2026-07-30) **RPET 미래설계/계획서 3종 삭제**(제어기 개발기록이 아님·백업됨): RPET_JUMP(점프는 live-solve로 완료, DTC §5) · RPET_TERRAIN_MAP(perception 인프라 설계, 메모리 `terrainmap-elevation-pointcloud`) · RPET_HEAD_GAZE(head 미래설계). RPET_ALIGATOR 참조행도 제거.
- (2026-07-09) 삭제: `RBQGUI-x86_64.AppImage`(167M)·`squashfs-root/`(448M) = 써드파티. 상위 흩어진 노트(`실행코드`·`obs.md`·`개발일지`·`메모`) → 이 docs/로 통합.
