# gait_sim_v8 구현 로드맵

현재 상태 (v7): quasi-static WBC (g(q) + GRF 피드포워드 + Impedance + PD), 균등 GRF 배분

완전 운동방정식 목표:
  M(q)·q̈ + C(q,q̇)·q̇ + g(q) = τ + Jᵀ·λ
  현재(v7):              g(q) = τ + Jᵀ·λ  ← quasi-static

---

## Phase 1 — MPC QP (Body-level 힘 계획)

MPC는 미래 N스텝 body 궤적을 예측하여 각 발의 목표 GRF(λ_des)를 결정한다.
WBIC/WBC는 이 λ_des를 받아 joint torque로 변환한다.

```
센서 → [MPC QP] → λ_des → [WBC/WBIC QP] → τ_cmd → 액추에이터
```

### 1-1. 선형화된 부유 베이스 동역학 모델
- [ ] body 상태벡터: x = [roll, pitch, yaw, px, py, pz, ω, v] (13dim)
- [ ] 연속 동역학: ẋ = A(Ψ)·x + B(r_i, R)·λ + g
      - A(Ψ): body 자세(오일러각) 의존 상태 행렬
      - B(r_i, R): stance foot 위치 의존 입력 행렬
      - λ: 각 발의 GRF [Fx, Fy, Fz]
- [ ] Euler 이산화: x_{k+1} = Ac·x_k + Bc·u_k  (dt = DT)

### 1-2. Receding Horizon QP 구성
- [ ] 예측 구간: N_MPC = 10~20 스텝 설정
- [ ] 상태 스택: X = [x_1, ..., x_N],  U = [λ_1, ..., λ_N]
- [ ] 비용 함수:
      min  Σ ||x_k - x_ref_k||²_Q  +  Σ ||λ_k||²_R
      Q: body pose/vel 추종 가중치
      R: GRF 크기 최소화 가중치
- [ ] 구속 조건:
      X = Aq·x_0 + Bq·U           (동역학 등식)
      |λ_x,k|, |λ_y,k| ≤ μ_f·λ_z,k  (마찰 추)
      λ_z,k ≥ 0                    (법선력 양수)
      λ_z,k = 0  (swing foot)      (공중 발 제거)
- [ ] solver: qpsolvers[quadprog] 또는 osqp 사용

### 1-3. contact schedule 연동
- [ ] swing_flag[fi] → MPC horizon 내 contact pattern 행렬 생성
- [ ] horizon 내 gait phase 예측 (현재 trot phase 기준)

### 1-4. MPC 출력 → WBC 입력 연동
- [ ] λ_des = MPC 첫 번째 스텝 출력 (receding horizon)
- [ ] 현재 v7의 균등 배분 λ_des 교체

---

## Phase 2 — QP GRF (단일 스텝 힘 배분, MPC 경량 대안)

MPC 없이 현재 스텝에서만 힘 평형 + 마찰 추 만족하도록 QP로 배분.
MPC보다 단순하나 미래 예측 없음. Phase 1 전 중간 단계로 구현 가능.

- [ ] 비용함수: min Σ||λ_i||²  (최소 힘)
- [ ] 등식 구속: Σλ_i = [0, 0, M·g],  Σ(r_i × λ_i) = [0,0,0]
- [ ] 부등식 구속: 마찰 추, λ_z ≥ 0
- [ ] r_i: body COM 기준 각 발 위치 (LEG_HIP_OFFSETS + foot_local)
- [ ] Figure 3 GRF subplot에 Fx, Fy 마찰 추 한계 표시 추가

---

## Phase 3 — RNEA (관절 공간 완전 동역학)

MPC/QP GRF가 λ_des를 주면, 이를 joint torque로 변환할 때
현재 quasi-static 대신 완전 강체 동역학 사용.

### 3-1. RNEA Forward Pass
- [ ] 각 링크 각속도/각가속도 재귀 전파: ω_i, α_i
- [ ] 각 링크 COM 선가속도 전파: a_c_i
- [ ] 입력: q, q̇, q̈ (수치 미분), 루트 body 가속도

### 3-2. RNEA Backward Pass
- [ ] 링크별 힘/모멘트 역전파
      f_i = m_i·a_c_i + ω_i × (I_i·ω_i)
      n_i = I_i·α_i + ω_i × (I_i·ω_i) + r_c_i × f_i
- [ ] τ_i = n_i · z_i  (조인트 축 성분)

### 3-3. τ_ff 업데이트
- [ ] 기존: τ_ff = τ_grav - Jᵀ·λ_des
- [ ] 변경: τ_ff = RNEA(q, q̇, q̈) - Jᵀ·λ_des
            (M(q)·q̈ + C(q,q̇)·q̇ + g(q) 전부 포함)

### 3-4. 링크 관성 텐서 정의
- [ ] 각 링크 I_i (원통/막대 근사 또는 CAD 값)
- [ ] LINK_INERTIA 파라미터 섹션 추가

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

- [ ] Σλ_z vs M·g 잔차 플롯 (힘 평형 검증)
- [ ] τ 에너지 Σ|τ·q̇| 계산 및 Figure 추가
- [ ] 관절 한계 위반 감지 + Figure 마킹
- [ ] (장기) pinocchio 연동: URDF 확보 후 RNEA/Jacobian 대체
      pip install pin
      pinocchio.buildModelFromUrdf()

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

- [ ] **Compliant 접촉 모델** (Hunt-Crossley)
      현재: 강체 접촉 (touchdown 충격력 계단)
      추가: F_z = K·δ + B·δ̇·δ, δ = penetration depth
      영향: 충격력 시계열 현실화 → 베어링/링크 피로 해석 가능
      파라미터: K (지면 강성, ~10^5 N/m), B (감쇠, ~10^3)

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

- Kim et al., "Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control", IROS 2019 (MIT Cheetah 3)
- Di Carlo et al., "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control", IROS 2018
- Wensing & Orin, "Generation of Dynamic Humanoid Behaviors through Task-Space Control with Conic Optimization", ICRA 2013
