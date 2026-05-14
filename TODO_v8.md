# gait_sim 구현 로드맵 (v8 → v12)

현재 상태 (v12.4.1): NMPC (crocoddyl FDDP receding horizon) + sim2real cost 보강 + GaitConfig dataclass

완전 운동방정식 목표:
  M(q)·q̈ + C(q,q̇)·q̇ + g(q) = τ + Jᵀ·λ
  v7: g(q) = τ + Jᵀ·λ  ← quasi-static
  v11: 부유 베이스 통합 동역학 + WBIC FB QP
  v12: NMPC (crocoddyl FDDP) — single optimization 으로 trajectory + control 동시 풀이

---

## Phase 1 — MPC QP (Body-level 힘 계획) — **v8 완료**

MPC는 미래 N스텝 body 궤적을 예측하여 각 발의 목표 GRF(λ_des)를 결정한다.
WBIC/WBC는 이 λ_des를 받아 joint torque로 변환한다.

```
센서 → [MPC QP] → λ_des → [WBC/WBIC QP] → τ_cmd → 액추에이터
```

### 1-1. 선형화된 부유 베이스 동역학 모델
- [x] body 상태벡터: x = [roll, pitch, yaw, px, py, pz, ω, v, g] (13dim)
- [x] 연속 동역학: ẋ = A(Ψ)·x + B(r_i, R)·λ + g
      - A(Ψ): body 자세 의존 상태 행렬
      - B(r_i, R): stance foot 위치 의존 입력 행렬
- [x] Euler 이산화: x_{k+1} = Ac·x_k + Bc·u_k  (dt = DT_MPC = 0.02s)

### 1-2. Receding Horizon QP 구성
- [x] 예측 구간: `N_MPC = 10` 스텝
- [x] 상태/입력 스택: X = [x_1, ..., x_N],  U = [λ_1, ..., λ_N]
- [x] 비용함수: min Σ||x_k - x_ref||²_Q + Σ||λ_k||²_R  (`MPC_Q`, `MPC_R`)
- [x] 구속: 동역학 등식 + 마찰 추 + λ_z ≥ 0 + swing λ=0
- [x] solver: `qpsolvers[quadprog]`

### 1-3. contact schedule 연동
- [x] swing_flag → MPC horizon 내 contact pattern 행렬 생성
- [x] horizon 내 gait phase 예측

### 1-4. MPC 출력 → WBC 입력 연동
- [x] λ_des = MPC 첫 번째 스텝 출력 (receding horizon)
- [x] v7의 균등 배분 교체 (`mpc_qp_plan()` 사용)

---

## Phase 2 — QP GRF (단일 스텝 힘 배분, MPC 경량 대안) — **v8 완료**

MPC 없이 현재 스텝에서만 힘 평형 + 마찰 추 만족하도록 QP로 배분.
`USE_MPC = False` 시 fallback 으로 사용 (`qp_grf_distribute()`).

- [x] 비용함수: min Σ||λ_i||²
- [x] 등식: Σλ_i = [0, 0, M·g], Σ(r_i × λ_i) = [0,0,0]
- [x] 부등식: 마찰 추, λ_z ≥ 0
- [x] r_i: body COM 기준 각 발 위치
- [x] Figure 3 GRF subplot 에 Fx/Fy 마찰 추 한계 표시 (fill_between)

---

## Phase 3 — RNEA (관절 공간 완전 동역학) — **v8 완료**

MPC/QP GRF가 λ_des를 주면, 이를 joint torque로 변환할 때
완전 강체 동역학 사용 (`rnea()` 함수, pinocchio 백엔드도 선택 가능).

### 3-1. RNEA Forward Pass
- [x] 각 링크 각속도/각가속도 재귀 전파: ω_i, α_i
- [x] 각 링크 COM 선가속도 전파: a_c_i
- [x] 입력: q, q̇, q̈ (수치 미분), 루트 body 가속도

### 3-2. RNEA Backward Pass
- [x] 링크별 힘/모멘트 역전파 (f_i, n_i)
- [x] τ_i = n_i · z_i (조인트 축 성분)

### 3-3. τ_ff 업데이트
- [x] τ_ff = RNEA(q, q̇, q̈) - Jᵀ·λ_des (full M·q̈ + C·q̇ + g 포함)

### 3-4. 링크 관성 텐서 정의
- [x] 원통 근사 (`_rod_inertia_local()`, `LINK_RADIUS`)
- [x] `LINK_MASS` 파라미터 정의 (현재 GaitConfig.link_mass)

---

## Phase 4 — Optimization-based IK (앞다리 궤적)

현재 Cycloid 해석 궤적 → 최적화 기반 관절 각도 궤적으로 교체.

- [x] 비용함수: λ_q·||q - q_home||²  (위치는 등식 제약으로 분리)
- [x] scipy.optimize.minimize (SLSQP), warm-start = 이전 프레임 q
- [x] 구속: 관절 상하한(FRONT_Q_LIM) ∩ 각속도 한계(동적 bounds), 토크 부등식(toggle)
- [x] 앞다리 swing 구간에만 적용 (stance는 기존 analytical IK 유지)
- [x] 수렴 실패 시 fallback: analytical IK → Q_HOME 순으로 강등
- [x] (선택) Figure 2에 IK 수렴 반복 횟수 subplot 추가 (v8: nit + pos_err + fallback ★)
- [x] 뒷다리 확장: opt_ik_hind 추가 (v8.7, 동일 cost/제약 구조)

---

## Phase 5 — WBIC QP (토크 공간 최적화)  → **v9 baseline 구현 완료**

RNEA로 계산한 τ_ff를 초기값으로, joint limit/마찰 추 제약 하에서 τ 최적화.

- [x] 변수: per-leg [Δq̈; Δτ; Δλ] (v9; 부유 베이스 통합은 추후)
- [x] 등식: M(q)·Δq̈ - Δτ - Jᵀ·Δλ = r (잔차 형태, 동등)
- [x] 부등식: τ_min ≤ τ_ff+Δτ ≤ τ_max, 마찰 추, swing λ=0
- [x] solver: qpsolvers[quadprog]
- [ ] task 우선순위 (body pose > foot position > joint limit) — **다음 단계**
- [x] 부유 베이스 통합 (단일 QP, body acc 변수 포함) — **v11**

---

## Phase 6 — 검증 & pinocchio 연동

- [x] Σλ_z vs M·g 잔차 플롯 (힘 평형 검증, fig3)
- [x] τ 에너지 Σ|τ·q̇| 계산 및 Figure 추가 — **v12.2 fig8 Cost-of-Transport**
- [x] 관절 한계 위반 감지 + Figure 마킹 — **v12.2 fig8 torque saturation**
- [x] pinocchio 연동 (RNEA/Jacobian/M(q) 등) — **v11 USE_PINOCCHIO toggle**
- [x] pinocchio Model 빌드 (build_pin_model.py) — **v12 crocoddyl 통합 필수**

---

## Phase 7 — 모델 충실도 향상 (설계 데이터 신뢰성)

설계 결정(모터 선정, 링크 응력, 배터리 산정 등)에 활용 가능한 데이터를
얻기 위한 추가 모델링 항목들. 우선순위 = 설계 직결성.

### Tier 1 — 설계 결정 직결

- [ ] **액추에이터 모델** (모터 + 감속기) — `actuator_v??`
      현재: τ_max 상수 한계만
      추가:
        · 토크-속도 곡선 (peak / continuous)
        · 감속비 N + reflected inertia (J_motor·N²)
        · 마찰 (Coulomb τ_c·sign(q̇) + viscous c·q̇)
        · 모터 전류/전압 한계 (선택)
      영향: "Hip 90형 vs 100형" 같은 모터 사양 검증 가능
      파라미터: per-joint (τ_max, ω_max, J_motor, gear_ratio, friction_c, viscous_c)

- [x] **Floating-base 통합 동역학** — **v11**
      현재(v10): body 위치 = V·t (kinematic only)
      추가: M·v̇ + C·v + g_world = Σ Jᵀ·λ를 6-DoF 적분 (Lie group)
      영향: 실제 pitch/roll 응답, ZMP, CoM 진동 측정 가능
      구현: pinocchio.aba() 또는 직접 (M_body·a_lin + I·α + ω×Iω = ...)

- [x] **WBIC 부유 베이스 통합** — **v11**
      현재(v10): per-leg 4개 분리 QP
      추가: 단일 QP, [Δv̇_fb (6); Δq̈_legs (20); Δτ (20); Δλ (12)]
      등식: body 6-DoF 운동방정식 + 4개 다리 운동방정식
      영향: 다리 간 토크 재분배 정확화, body acc task 우선순위 가능

### Tier 2 — 신호 정확도 개선

- [~] **Compliant 접촉 모델** (Hunt-Crossley) — **v12 prep 시도 후 보류**
      현재: 강체 접촉 (touchdown 충격력 계단)
      추가: F_z = K·δ + B·δ̇·δ, δ = penetration depth
      영향: 충격력 시계열 현실화 → 베어링/링크 피로 해석 가능
      파라미터: K (지면 강성, ~10^5 N/m), B (감쇠, ~10^3)
      ※ explicit Euler + 6.3kg body 에서 numerical instability — 어떤 K/B 조합도 발산.
        대안: v12 NMPC 의 friction cone + force reg + stance v=0 cost 로 spike 간접 완화.

- [ ] **Self-collision 검사** (capsule-capsule)
      현재: 검사 없음 (다리-다리, 다리-body 충돌 가능)
      추가: 각 링크에 capsule (반경+축) 부여, 거리 검사
      영향: 새 Q_HOME / swing 궤적의 물리적 가능성 자동 검증
      라이브러리: python-fcl 또는 직접 구현

- [ ] **Spline 기반 q̇/q̈** (수치 미분 노이즈 감소)
      현재: np.gradient(q) 직접 미분
      추가: cubic spline 또는 Savitzky-Golay 필터
      영향: τ_dyn 잔떨림 제거 → RNEA 토크 노이즈↓ → 모터 사양 산정 정확화

### Tier 3 — 시스템 검증용

- [ ] **센서 모델** (인코더 양자화 + 노이즈, IMU bias/drift)
      → 제어기가 실 센서 노이즈 견디는지 stress test

- [ ] **Slip detection + handling**
      마찰 추 위반 시 grip 손실 처리, 회복 제어
      → 미끄러운 환경 robustness 평가

- [ ] **배터리/전력 모델**
      I·V·dt → 전력 → 시간당 에너지 → 배터리 용량 산정
      모터 모델과 결합 (Tier 1 액추에이터 선행 필요)

### 권장 순서 (영향 vs 작업량)

```
pinocchio (Phase 6 기반)
   ↓
액추에이터 모델 (Tier 1.1)  ← 가장 빠른 ROI
   ↓
Floating-base 적분 (Tier 1.2) ← v11
   ↓
WBIC FB 통합 (Tier 1.3) ← v11
   ↓
Compliant contact (Tier 2.4)
   ↓
나머지 검증
```

### 설계 데이터 목적별 우선순위

| 설계 데이터 | 필수 항목 |
|-----------|----------|
| 모터/감속기 선정 | Tier 1.1 + 1.2 |
| 링크/베어링 강성/피로 | Tier 1.1 + 2.4 |
| Gait robustness | Tier 1.2/1.3 + Tier 3 |
| 에너지/주행시간 | Tier 3.9 + Tier 1.1 |

---

## Phase 8 — NMPC 통합 (crocoddyl FDDP) — v12

v11 의 "MPC body→λ_des→WBIC→τ" pipeline 을 single optimization 으로 통합.
미래 N step 에 대해 state x, control u, contact force λ 를 동시에 풀이.

### 8-1. NMPC core 통합 — **v12.0**
- [x] crocoddyl + pinocchio 통합 (`build_pin_model.py` — quadruped pin Model)
- [x] `DifferentialActionModelContactFwdDynamics` (Baumgarte stab)
- [x] `SolverFDDP` (one-shot 1-2 cycle 안정)
- [x] Receding horizon 구현 (4+ cycle 안정: `NMPC_RH_N_HORIZON=24`, `_RESOLVE=12`)
- [x] `USE_NMPC` / `USE_NMPC_RECEDING` toggle

### 8-2. NMPC sim2real cost 보강 — **v12.1, v12.3**
- [x] Friction cone barrier (`ResidualModelContactFrictionCone` + `ActivationModelQuadraticBarrier`)
      `inner_appr=True` 채택 → friction usage peak 1.39 → 1.00
- [x] Force regularization (`ResidualModelContactForce` target=0, xy>z weighting)
      HL Fz peak 928 → 443N (-52%)
- [x] Joint torque limit barrier (`|u_i| ≤ JOINT_TORQUE_LIMIT[i]` soft barrier)
      τ peak 142 → 85 Nm (-40%)
- [x] Swing trajectory: cycloid → smoothstep + bell (v=0 at touchdown)
- [x] Pre-touchdown velocity penalty (`ResidualModelFrameVelocity`, 마지막 N step)
- [x] Stance leg foot velocity=0 cost (slip drift 37mm → 3mm, -92%)
- [ ] Body y drift (44mm/2s) — `_HIP_Y_BIAS = +7.5mm` 모델 비대칭 root cause.
      NMPC cost shaping 으로는 해결 불가 (W=1~1000 모두 시도, vx tracking trade-off).
      → 옵션: 모델 변경 / WBIC hybrid 의 CoP-balance constraint

### 8-3. NMPC 후처리 + 시각화 — **v12.1, v12.2**
- [x] tau_grf, tau_dyn, tau_imp 분해 (NMPC 모드에서 fig3/fig4 의미 회복)
- [x] **v13**: tau_dyn 을 정적 중력 보상 → full RNEA(M·q̈+C·q̇+g) 로 교체.
      tau_imp 가 의미 있는 잔차(per-leg 분해 ↔ whole-body NMPC 불일치)가 됨.
      ※ 순수 후처리 분해일 뿐 — 제어 루프는 여전히 NMPC 단독, τ_cmd 값 변경 없음.
- [x] **fig5**: body state vs cmd (pos/vel/orientation/ω + RMS error)
- [x] **fig6**: foot trajectory cmd vs actual (3×4 grid, swing only)
- [x] **fig7**: gait diagram (Hildebrand-style stance chart + Fz timeline)
- [x] **fig8**: diagnostic (friction usage / mechanical power+CoT / slip vel / τ saturation)

### 8-4. Perturbation recovery — **v12.3**
- [x] `USE_PERTURBATION` toggle (receding horizon 도중 body state 외란 주입)
- [x] vy += 0.5 m/s @ t=1s 외란 → vy_end = 0.004 m/s 회복 검증
- [ ] omega 외란, large vy(>1 m/s) 한계 시험
- [ ] perturbation 시계열 figure (fig9 후보)

### 8-5. Configuration 통합 — **v12.4**
- [x] `@dataclass GaitConfig` (45+ user-tunable param 단일 위치)
- [x] 카테고리: gait / robot / motor / friction / IK / MPC / WBIC / NMPC / perturb / viz
- [x] 기존 module-level globals → `CFG.field` alias (v11/v12 코드 100% 호환)

### 8-6. (검토 대기) Real-time NMPC + WBIC hybrid 모드
- [ ] **옵션 A**: Step-by-step real-time NMPC (매 v11 frame 1-step solver 호출)
      장점: 단순, NMPC 가 직접 τ 만듦. 단점: WBIC 안 씀 → friction/torque hard constraint 없음
- [ ] **옵션 B**: NMPC trajectory + WBIC tracking (NMPC offline ref, WBIC 매 frame enforce)
      장점: sim2real safer (WBIC 가 friction/torque limit hard constraint). 단점: NMPC ref 미세 violation 잘림
- [ ] **옵션 C**: Hierarchical-rate hybrid (실무 표준 구조) — **유력안**
      - NMPC: 저주파(~100–400 Hz) 또는 단순화 모델로 receding-horizon 해 → ref 궤적(τ/ddq/λ) 생성
      - WBIC: 고주파 instantaneous QP 로 매 틱(~2 ms) ref 추종 + hard constraint(토크 한계,
        마찰콘) 강제. NMPC 출력을 레퍼런스 생성기로, WBIC 를 trailing tracker 로 사용.
      - 장점: 가장 robust (sim2real). 단점: 가장 복잡 (2-rate 인터페이스 설계 필요)
      - 주의: NMPC 가 이미 full whole-body OC → 순수하게는 중복. 의미는 "rate 분리 +
        hard constraint 보강"에 있음. v13 의 SE(3) hip transform 정리 후 착수.
- [ ] body y drift 해결 (WBIC CoP / lateral balance constraint 명시)

### 8-7. (추후) Walk / multi-gait NMPC 지원
- [ ] 현재 NMPC: trot only (`if GAIT_TYPE != 'trot': return None`)
- [ ] phase pattern 일반화 (walk, amble, pace, canter, gallop — 각 4-leg phase offset)
- [ ] gait switch test (steady trot → walk transition)
- [ ] speed sweep (V=0.3, 0.5, 1.0, 1.5 m/s 안정성 비교)

---

## 구현 순서 (권장)

```
Phase 2 (QP GRF)  →  Phase 1 (MPC QP)  →  Phase 3 (RNEA)
     ↓                      ↓                      ↓
단일 스텝 힘 배분      미래 예측 GRF          완전 동역학 τ
  (2~3일)               (1주)                  (1주)

     → Phase 4 (Opt-IK) → Phase 5 (WBIC QP) → Phase 6 (검증)
```

---

## 라이브러리 설치

```bash
pip install qpsolvers[quadprog]   # Phase 1, 2, 5
pip install osqp                  # Phase 1 대안
pip install scipy                 # Phase 4
pip install pin                   # Phase 6 (pinocchio)
```

---

## 참고 문헌
#ANYmal
- Kim et al., "Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control", IROS 2019 (MIT Cheetah 3)
- Di Carlo et al., "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control", IROS 2018
- Wensing & Orin, "Generation of Dynamic Humanoid Behaviors through Task-Space Control with Conic Optimization", ICRA 2013
