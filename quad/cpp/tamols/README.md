# C++ 실시간 TAMOLS (모델기반 지형 joint MPC) — 개발

**목표** — D1(OCS2 Perceptive NMPC) 청사진을 **우리 MuJoCo/C++ 스택에 native로** 구현.
base 스플라인 + 발판 + GIAC 안정성 + 지형(footScore/edge)을 **공동최적화**하는 실시간 지형 MPC.
= B 제어기의 실시간화(발판만 greedy → base+발판 joint 온라인 협조). APT-RL 전 모델기반 정공법 1판.

**왜 D1 포팅이 아니라 native 구현** — D1(OCS2/Gazebo)은 MuJoCo 작동 보장 없음(sim2sim+ROS+로봇적응 삼중 리스크).
우리가 원하는 건 코드가 아니라 **알고리즘**(OCS2식 joint 지형 MPC + 빠른 QP). Drake 프로토타입으로 정식화는 검증됨 → C++로 포팅.

## 접근

- **정식화**: Drake `quad/tamols/tamols_02leg.py`(우리가 갭회피·GIAC로 검증) → C++ 포팅.
- **솔버**: SQP-RTI(1 iteration warm-start) + 빠른 QP(**eiquadprog**, 이미 quad/cpp WBIC에 있음). 실시간(20ms) 목표.
  - Drake 오프라인 11.7s(583× 느림)의 원인 = 범용 NLP·전체수렴. 극복 = 짧은 horizon + RTI + 커스텀 QP.
- **지형**: `quad/cpp/src/terrain_map.hpp`(footScore/edgeSDF/slope) 재사용.
- **실행**: `quad/cpp` WBIC(eiquadprog)로 계획 추종 + MuJoCo 폐루프(`trot_sim`·`DISABLE_FLOOR`·`quad_tamols_gap.mjcf`).

## 변수/제약/비용 (Drake 대응)

| 항목 | Drake | C++ |
|---|---|---|
| base 스플라인 | `spline_coeffs[phase]` (base_dims×order) | `TamolsState.a[phase]` MatrixXd(6×4) |
| 발판 | `p` (num_legs×3) | `p` Matrix4x3 |
| GIAC slack | `epsilon` (num_phases) | `epsilon` VectorXd |
| 스플라인 평가 | `helpers.py` pos/vel/acc | `tamols.hpp` pos/vel/acc ✅검증(1e-10) |
| 초기·연속 | `add_initial_constraints` | `constraints.hpp` ✅검증 |
| GIAC 동역학 | `add_dynamics/giac_constraints` (Eq17) | `constraints.hpp` ✅검증(4.6e-6) |
| kinematic reach | `add_kinematic_constraints` (l_min/max) | `constraints.hpp` ✅검증 |
| friction cone | `add_friction_cone_constraints` | `constraints.hpp` ✅검증 |
| 비용 | tracking/foothold/nominal (활성) | `costs.hpp` ✅검증(rel 1e-9) |

## 진행 상태

- ✅ **스플라인 평가 핵심** (`tamols.hpp`) — pos/vel/acc, 초기조건 매핑. `test_spline.cpp` 검증(FD 1e-10·초기조건 정확).
- ✅ **지형 처리** (`terrain_proc.hpp`) — h_s1(gaussian)·h_s2(virtual floor)·∇h/∇h_s1/∇h_s2(5점 FD). `test_terrain.cpp` **Python(Drake) 정합 검증**(gaussian 5e-9·∇ 1e-7). ★부호주의: scipy convolve1d=convolution이라 미분커널 뒤집음([-1,8,0,-8,1]).
- ✅ **제약 5종 완성** (`constraints.hpp`) — 초기·위상연속·friction·kinematic reach·**GIAC(Eq17: 17a 마찰콘·17b 다중접촉·17c/d 이중지지)**. Drake feasible 해 로드해 residual 전부 검증(초기 0·위상연속 8e-9·friction 0·kinematic 0·**GIAC 4.6e-6**). L̇=I·dω+ω×Iω(각운동량), det=a·(b×c). `test_constraints.cpp`.
- ✅ **비용 3종 완성** (`costs.hpp`) — tracking(x속도)·foothold_on_ground(양선형 높이보간)·nominal_kinematic. Drake `EvalBinding` 값과 대조 rel~1e-9(track 0.1131·foot 8e-6·nom 3e-5). base_pose_align·edge·prev·smoothness는 Drake서 비활성(주석). `test_costs.cpp`.
- ✅ 갭회피 발판 구속 = 선형 bound(앞발 ≥gap_hi+margin·뒷발 ≤gap_lo−margin), Drake `add_gap_avoid_footholds`. 솔버 조립 시 부등식 bound로 편입.
- ✅ **SQP-RTI 솔버**(`tamols_qp.hpp`, eiquadprog) — 결정벡터 pack/unpack(nz=137) + 비용 GN 최소제곱(H=2JᵀJ·g=2JᵀR) + 제약 선형화(FD Jacobian) + ℓ1 merit 라인서치 + **적응형 Levenberg-Marquardt(trust-region reg)**. Drake 해=고정점(eq 1e-14·ineq 1.6e-7), 섭동서 feasible 복귀. `test_qp.cpp`.
  - **정확성·수렴 ✓** / **실시간 아직**: FD Jacobian 5.2 ms/iter(반복당 ~822 함수평가), 5-iter RTI 26 ms(>20ms). → **해석 Jacobian 교체가 실시간 경로**(반복당 10–50×↑ 기대). FD는 검증용.
- ⬜ 해석 Jacobian(비용·제약) → 실시간 20ms
- ⬜ WBIC 폐루프 통합 + 갭 MuJoCo 검증(vs Drake 오프라인)
- ✅ **플랜→WBIC 참조 변환기**(`tamols_track.hpp`) — TAMOLS 해(base 스플라인·발판·게이트)를 매 스텝 (com_ref·yaw·contacts·swing 궤적)으로. `test_track.cpp` 검증: base 연속·트롯 스케줄·스윙끝↔발판 6e-4·x 0→0.73(갭전진).
- ⬜ **MuJoCo 폐루프 러너**(`tamols_sim.cpp`) — 변환기로 ctrl.wbic_track 구동 + quad_tamols_gap.mjcf 갭 크로싱 검증(per-gait 검증 역할)

**검증 방식** — 각 조각을 Drake Python 출력과 대조(같은 입력 → 오차 <1e-5). `test_*.cpp` 회귀.

## 빌드/실행

```bash
cd /home/jsh/문서/jsh/simulation/quad/cpp
g++ -O2 -std=c++17 tamols/test_spline.cpp -I/usr/include/eigen3 -o tamols/test_spline && ./tamols/test_spline
```

관련: 모델기반 갭크로싱 리포트(`docs/모델기반_갭크로싱_탐색리포트.html`) · Drake `quad/tamols/` · TAMOLS 논문 `~/다운로드/논문/TAMOLS_2206.14049.pdf`

## 해석 Jacobian (실시간화) — 진행
- ✅ **비용 잔차 해석 Jacobian**(`tamols_jac.hpp`) — tracking(선형)·foothold(양선형 ∂h)·nominal(∂R_B). FD 대조 오차 2.5e-10·**9× 빠름**(26.5µs vs 237µs). `test_jac.cpp`.
- ✅ **등식 제약 해석 Jacobian**(초기·위상연속=선형 상수). FD 대조 1.4e-10. `test_jac.cpp`.
- ✅ **부등식 해석 Jacobian(비GIAC 299행)** — friction·kinematic(2차)·bounds·foot_y·gap·terminal. FD 대조 7e-10. GIAC 179행=mask(FD 대상).
- ✅ **GIAC 블록-sparse FD** — phase-k 행은 a[k]+p+eps(k)만 의존 → 관련변수만 FD. 전체 부등식 Jacobian(해석+GIAC) 1083µs(vs full FD 4315µs=**4×**), 정합 7e-10.
- ✅ **★실시간 달성** (`solve_fast`, tamols_jac.hpp) — 해석 Jacobian 솔버 **2.26 ms/iter**(FD 6.53 대비 2.9×), **5-iter RTI = 11.7 ms < 20ms(50Hz)**. FD와 동일 수렴. `test_fast.cpp`.
- **C++ TAMOLS = 실시간 온라인 TO(DTC) 가능**. 남은: warm-start receding-horizon 완전크로싱·WBIC 온라인 통합.

## ★2026-09 · 능력 검증(오프라인) — 켠 항목과 기본값 변경

`test_capability.cpp` (빌드 방법은 위 빌드/실행 절과 동일, 실행 `./test_capability {jac|t1|t2|t3|t4}`).

**기본값이 바뀐 env (중요)**

| env | 기본 | 의미 |
|---|---|---|
| `GIAC_FIX` | **ON** (`GIAC_FIX=0` 으로만 끔) | GIAC 식에 중력항 `aG = aB − g` 포함. OFF 면 정지자세서 aB≈0 이라 17a/17b 가 항등만족 = GIAC 기전이 사실상 없음 |
| `GIAC_ORDER` | **ON** (GIAC_FIX 종속) | 17b 접촉쌍을 **지지폴리곤 CW 인접쌍**으로. 레거시(Drake)의 인덱스순 전쌍 `(i<j)` 는 지지다각형 조건이 아님(아래) |
| `EPS_MAX` / `BASE_YBND` | 기존과 동일 | 단, 해당 행의 **해석 Jacobian** 을 추가해 FD 폴백 제거(속도·정합 개선) |
| `edge_layers()` (전역 포인터) | `nullptr` = OFF | `edge_avoidance` 비용 게이트. OFF 면 잔차/자코비안에 **0행** 추가 = 기존과 바이트 동일 |

**오프라인 캐시 생성기는 레거시 고정** — `cache_gen*.cpp` 는 main 첫줄에서 `GIAC_FIX=0 · GIAC_ORDER=0` 을 세팅한다
(이미 배포된 캐시 아티팩트의 재현성 보존). 새 동작으로 생성하려면 `TAMOLS_ALLOW_GIAC=1`.

**GIAC 쌍 정렬 근거** — `det(p_ij, p_B−p_i, (0,0,g)) = g · cross_z(p_j−p_i, p_B−p_i)` 이므로
`… ≤ eps` 형태는 **CW 정렬 인접쌍**일 때만 "p_B ∈ 지지다각형"과 동치다.
레거시(인덱스순 전쌍)로 중력항을 켜면, 예컨대 Go2 walk 의 stance {FR,RL,RR} 에서 해집합이
`{p_x ≤ −0.1934, p_y ≥ +0.142}`(뒷발보다 뒤·왼쪽 = 지지삼각형 바깥)이 되어 base 를 폴리곤 밖으로 민다.

**edge_avoidance** — Drake `costs.py::add_edge_avoidance_cost` 정식화 그대로
`Σ_legs [3·|∇h(p_i)|² + |∇h_s1(p_i)|²]`, 최소제곱 잔차 16행(다리당 4) + 해석 Jacobian.

## ★2026-09 · 다중스텝 예측(③) — 스탠스 reach + N-사이클 호라이즌

`test_capability.cpp` 에 `t3s`(스탠스 reach 결합 정량) · `t3m`(N=2 함정 시퀀스) · `cost`(N=1 vs N=2 비용) 추가.

### 새 게이트 (둘 다 **기본 OFF** — OFF 면 기존과 바이트 동일, `t1~t4`·`test_online` 실측 확인)

| env / 필드 | 기본 | 의미 |
|---|---|---|
| `STANCE_REACH` | **OFF** | **구 발판(`p_meas`) 위에 서 있는 다리**에도 reach 제약. 기존 reach 는 `at_des`(이번에 새로 놓는 발판) 다리에만 걸려 있어 지면에 박힌 발이 base 전진을 전혀 막지 못했다. τ 샘플 `s=1..S`(τ=0 은 초기조건 등식으로 고정이라 위반해도 못 고침). 해석 Jacobian 포함(p_meas 는 파라미터라 `a[k]` 열만) |
| `TamolsState::p_ext` (비면 N=1) | **N=1** | 사이클 1..N−1 의 발판 결정변수. `GaitPhase::foot_idx[L]` = 그 phase 에서 다리 L 이 딛는 발판의 사이클(−1=`p_meas`, −2=`at_des` 유도=레거시). `cyc_end` = 사이클별 마지막 phase(nominal 비용 평가점) |

`Packer` 레이아웃 = `[a(24P) | p(사이클 0..N−1, 12N) | eps(P)]`, `nz = 24P + 12N + P`.
색인 헬퍼 `ipn(P,n,L,c)` · `ieps(P,NC,k)` · `nz_of(P,NC)` (`tamols_jac.hpp`).

### 검증
- **해석 Jacobian vs FD**: N=1 · STANCE_REACH ON · **N=2** 전부 `max|Δ|` R=1.4e-08 · G=7.5e-10 · E=2.7e-11 (`./test_capability jac`).
- **N=1 회귀**: `t1~t4` 바이트 동일, `test_online` 수치 동일.

### 결과 (사전등록 판정)

**① 결합은 생긴다 — 다만 원인은 "호라이즌"이 아니라 "빠진 제약"**
기존 정식화는 **접지 중인 발을 최대 2.9 cm 과신장**시키는 base 궤적을 계획하고 있었다(물리적으로 불가능한 계획).
`STANCE_REACH=1` 이면 그 위반이 정확히 0 이 된다. 결합 정량(구 발판이 붙어 있는 동안 도달 가능한 최대 base_x,
호라이즌을 phase 0-1 로 절단해 "마지막 phase 가속 만회" 탈출구 제거):

| | δ=0 | δ=−0.145(뒷발 뒤로) | Δ |
|---|---|---|---|
| OFF | 0.494 m | 0.455 m | −0.038 (그나마 reach 아닌 다른 원인, 0.494 는 다리 0.61 m 를 요구=불가능) |
| **ON** | 0.341 m | 0.223 m | **−0.118 m** (이론 −0.145, 차이는 크라우치·pitch 여유) |

※ **호라이즌 끝 base_x 로는 못 잰다** — 구 발판이 떨어진 뒤엔 자유라 마지막 phase 에서 가속으로 만회한다(속도 상한 없는 정식화).

**② N=2 는 함정 시퀀스를 못 넘는다 — 그리고 이유는 호라이즌이 아니다**
`t3m`(t3 와 같은 지형): N=2 사이클1 앞발판이 **0.740 = VOID**(LONG 끝 0.72 바로 뒤), NEXT 착지 **0/2**. N=1 순차도 0/2.
터치다운 reach 0.362 < l_max 0.42 → **reach 로 막힌 게 아니다**.

근본 원인 = **목적함수에 VOID 페널티가 없다.** `foothold_on_ground = 100·(h(p)−p.z)²` 는 void 에서 `h=0` 이고
`p.z` 도 0 으로 내리면 비용 0 → **돌을 고를 이유가 비용에 없다.** 실제로 높은 요구에서 발판이 **공중부양**한다
(x_target=1.40 진단: `p.z=0.095`인데 정확 지형 `h=0.000`). 지형이 soft cost 라 **reach 사슬이 실지형에 절대 물리지 않는다.**

부작용으로 드러난 것: N=1 receding-horizon 이 상대적으로 지형에 붙어 보이는 이유는 플래너가 아니라 **매 사이클 재측정**
(re-anchor 시 발 z 를 실지형으로 스냅) 덕분이다. 호라이즌을 늘리면 그 접지가 **사라진다**(내부 사슬엔 실지형 강제가 없음).

**③ 비용 — 실시간 상실**

| | phase | nz | eq행 | ineq행 | RTI 5-iter cold | warm | 완전수렴(max 80) |
|---|---|---|---|---|---|---|---|
| N=1 | 4 | 112 | 48 | 381 (ON 453) | **6.0 ms** | 7.5 ms | it=80, 110 ms |
| N=2 | 8 | 224 | 96 | 833 (ON 905) | **34–37 ms** | 39–41 ms | it=22~35, 266–339 ms |

**N=2 = 5.7× 느려 20 ms(50 Hz) 예산 초과 → 오프라인 전용**(DTC/APT-RL 캐시 용도로는 무방).
`STANCE_REACH` 자체는 **사실상 공짜**(N=1 5.8 vs 6.0 ms, +72 행).

### 결론
- **STANCE_REACH 는 값이 있다** — 다중스텝과 무관하게 **정확성 버그 수정**(접지 발 과신장 2.9 cm → 0). 비용 ~0.
- **N=2 호라이즌 자체는 이 정식화에서 값을 못 낸다** — 지형이 soft cost 라 발판이 공중부양하고, 함정 지형에서
  N=1 대비 이득이 없으며, 실시간을 잃는다. 다중스텝이 값을 내려면 **먼저 지형을 hard 하게 만들어야 한다**
  (foothold-on-ground 등식화 / footScore·void 페널티 / 돌 선택 정수변수). 그건 TAMOLS SQP 가 아니라 TOWR 계열(위상·접촉 포함 TO)의 영역.

## ★2026-09 · 지형을 "물리게" 만들기 — 지지 유효성 하드 제약 (`TERRAIN_HARD`)

### 문제(직전 진단)
`foothold_on_ground = 100·(h(p) − p.z)²` 는 **보이드에서도 0**이다(`h=0` 이고 `p.z→0`).
즉 **돌 위에 서는 것을 보상하는 항이 목적함수에 하나도 없다.** 실측: `x_target=1.40` 에서
발판 `p.z=0.095` 인데 정확 지형 `h=0.000` = **공중부양**. N=1 이 접지돼 보였던 건 플래너가 아니라
매 사이클 **재측정(re-anchor)** 이 발 z 를 실지형에 스냅했기 때문.

세 가지 흔한 처방이 전부 안 통한다 — 셋 다 "높이 자체"만 보기 때문:
| 처방 | 왜 안 되나 |
|---|---|
| `h(p)=p.z` 하드 등식 | 보이드 바닥이 **정확히** 만족(h=0, p.z=0) |
| `edge_avoidance(|∇h|)` | 보이드 바닥은 평평 → `|∇h|=0` |
| 발 반경 침식(erosion) | 보이드 바닥은 넓고 평평 → 침식을 통과 |

### 채택한 정식화 — 회색조 형태학적 닫힘(grayscale closing) 기반 "지지 유효성"
`terrain_proc.hpp :: compute_support()`

1. **국소 작업면** `h_close = erode_R(dilate_R(h))` (원판 SE, 반경 R)
   `gap_depth := h_close − h ≥ 0`
   - **단조 지형(계단·경사·단차)에서 closing = 항등** ⇒ `gap_depth ≡ 0`, **오검출이 원리적으로 없다**
     (1D 증명: 단조 h 에 대해 `dilate_R(x)=h(x+R)`, `erode_R∘dilate_R(x)=h(x−R+R)=h(x)`).
   - 폭 < 2R 인 **구덩이/갭/보이드만** 채워짐 ⇒ `gap_depth` = 그 구덩이 깊이.
2. **유효집합** `valid = {gap_depth ≤ dz_max}` (dz_max = 한 스텝 step-down 허용량)
3. **부호거리** `sdf` (음수 = 유효 안쪽). 보이드 **내부(∇h=0)에서도 |∇sdf|=1** 이라
   SQP 가 돌 쪽으로 나갈 방향을 갖는다(하드 제약이어도 국소 정체하지 않음).
   `sdf(p) ≤ −margin` 은 유효집합을 margin 만큼 **침식**한 것과 동일 ⇒ 발 반경을 margin 으로 흡수.

**제약 (비용 아님, 발판당 3행)** — `tamols_qp.hpp :: ineq_constraints` 끝에 추가
```
band+ :  band − (h(p_xy) − p.z) ≥ 0        ┐ |h(p)−p.z| ≤ band  = 지형 표면에 붙는다(부양 금지)
band− :  band + (h(p_xy) − p.z) ≥ 0        ┘
sdf   : −sdf(p_xy) − margin     ≥ 0          = 그 지형점이 국소 작업면에서의 지지다
```
**둘 다 있어야 한다.** band 만이면 보이드 바닥이 `p.z=h=0` 으로 만족(함정 1), sdf 만이면 보이드 위에
`p.z` 를 띄워 xy 를 유지한다. band 가 `p.z` 를 실지형에 묶고, sdf 가 그 xy 를 무효로 판정한다.

### 게이트/파라미터 (전부 **기본 OFF**)
| env / 필드 | 기본 | 의미 |
|---|---|---|
| `support_layers()` (전역 포인터) | `nullptr` = OFF | OFF 면 제약/자코비안에 **0행** = 기존과 바이트 동일 |
| `TERRAIN_HARD` (하네스) | OFF | Scene 이 지지층을 만들고 포인터를 등록 |
| `TERRAIN_R` | 0.25 m | closing 반경 = **"구덩이 vs 낮은 지형 레벨"의 스케일**(채울 구덩이 최대 반폭) |
| `TERRAIN_DZ` | 0.06 m | 국소 작업면 아래 허용 하강량 |
| `TERRAIN_BAND` | 0.01 m | `|h(p)−p.z|` 허용 |
| `TERRAIN_MARGIN` | 0.025 m | 유효영역 안쪽 여유(= 발 반경, env 기하스냅의 `half−0.025` 와 동일) |
| `TERRAIN_SMOOTH` | OFF(=hraw) | 하드 판정이 볼 맵. **기본은 원본 고도맵** — `hsol(=gaussian σ1)` 은 비용/기울기 shaping 용이지 지형 사실이 아니라 돌 가장자리에 **유령 램프**(실제 h=0 인데 hsol≈0.07)를 만들어 잘못된 허가를 낸다(실측) |

`./test_capability th` = 판별식 자체 검증 · `th2` = "부양 정지" 단독 진단.

### 검증
- **해석 Jacobian vs FD**: `terrain hard`(N=1, +12행) `★N=2+지형하드`(+24행) 모두
  `max|Δ| R=1.41e-08 · G=7.48e-10 · E=2.68e-11` — 신규 행이 기존 최대오차를 **키우지 않음**(`./test_capability jac`).
- **OFF 바이트 동일**: `t1·t2·t3·t3s·t3m·t4` 출력 완전 동일. `test_online/test_fast/test_qp/test_jac/test_stairs/test_stepping/cmp_gait` 은
  변경 전 소스로 빌드한 바이너리와 **수치 동일**(※ 저장소에 커밋돼 있던 구 바이너리는 7월판이라 기준선이 아님).
- **판별식**(`th`): stepping 돌 gap=0.000 / 보이드 0.150 · **계단 100% 유효(오검출 0)** · 트렌치 0.150 · 보이드 내부 `|∇sdf|=1`.

### 결과 — 지형은 물었다. 그런데 4대 기계는 안 살아난다
| 사전등록 | 결과 |
|---|---|
| ① **부양 정지** | **PASS**. feasible 요구에서 `|p.z − h_map| ≤ band` 전부, VOID 0/8(t3·t4). 최대도달 x_target 이 N=2 1.59→1.04 로 떨어진 건 **이전 값이 부양으로 만든 허구**였다는 뜻 |
| ② GIAC 발판선택(t1) | **여전히 FAIL**. R_good 초기화도 R_near 로 회귀(두 후보가 **같은 해**로 수렴). 탈출구는 보이드가 아니라 `nominal_kinematic` 이었다 |
| ③ 몸통계획(t2) | PASS 유지(회귀 OK) |
| ④ 돌 선택(t4) | **여전히 FAIL**(w=1 에서 SMALL). 하드 지형은 "돌인가"만 물지 "어느 돌이 나은가"는 못 문다 — 후자는 **점수**가 필요. 층에는 정보가 있다: clearance(−sdf) SMALL 0.040 vs BIG 0.108 (`4-E`), 목적함수가 안 쓸 뿐 |
| ⑤ 다중스텝(t3m) | **여전히 FAIL**. N=2·N=1 모두 M-A 0/2·M-B 0/2. x_target 스윕에선 오히려 **N=1 순차가 N=2 보다 멀리 간다**(0.85/0.95/1.05 에서 N=1 만 NEXT 착지) |
| ⑥ 실시간 | **유지**. N=1 RTI5 cold **6.1 ms**(OFF 5.8–8.0). N=2 는 40 ms(오프라인 전용, 기존과 동일 결론) |

**한계(정직)**
- `TERRAIN_R` 는 "구덩이 vs 낮은 지형 레벨"의 **스케일 파라미터**다. 보이드가 R 보다 넓게 **열려 있으면**
  판별식은 (정의대로) 그것을 정당한 낮은 지형으로 본다 — t2 지형이 그 경우(void 1/4 잔존).
  독립된 돌들로 이루어진 stepping 필드에서는 낮은 영역이 하나로 연결된 열린 평면이라
  대각 방향으로 **좁은 탈출 채널**이 남는다(실측: 돌 사이 y≈0.15 대역). RL 쪽에서도 같은 이유로
  지형 측도가 아니라 **`GO2_VOID_FIX=die`(태스크 선언)** 로 막았다 — 기하만으론 원리적으로 안 되는 부분.
- 발판이 돌 경계에서 **1 셀(2 cm)** 밖으로 나가 `dz_exact` 가 최대 ~0.09 m 어긋나는 경우가 남는다.
  원인은 고도맵의 2 cm 양자화(수직 15 cm 벽을 가로지르는 셀의 양선형 = 유령 램프). `TERRAIN_MARGIN ≥ 발반경+셀` 로 완화.
- **infeasible 요구 + N=2 는 치명적으로 느리다**(100 s / 2–3 SQP iter). 하드 제약으로 선형화 QP 가
  자주 infeasible → elastic 폴백 QP 차원 `nz+2·neq+nineq ≈ 1345` 를 SQP 반복마다 20 회. **N=1 은 무영향**(t3s 40.7 s ON vs 45.2 s OFF).
- 맵 전처리 `compute_support` = 121×121·R=0.25 에서 **20.9 ms**(R=0.15 8.5 ms). SQP 반복이 아니라 맵 갱신당 1회.
  50 Hz 로 맵을 매 사이클 새로 만들면 예산 초과 → 분리형 box SE 나 증분 갱신 필요.

**결론** — "지형이 soft cost 라 발판이 공중부양한다"는 진단은 맞았고 그건 고쳐졌다.
그러나 **부양을 막아도 ①②④⑤ 중 살아나는 건 없다.** 남은 실패는 지형이 아니라 **목적함수**(어느 돌이 나은지에
대한 점수 부재)와 **탐색**(연속 SQP 가 발판 조합을 못 바꿈)에 있다. 이는 TAMOLS SQP 의 한계이고,
발판을 **선택 변수**로 다루는 TOWR/MINLP 계열 또는 RL 이 필요하다는 근거가 된다.
