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
- [ ] (선택) Figure 2에 IK 수렴 반복 횟수 subplot 추가

---

## Phase 5 — WBIC QP (토크 공간 최적화)

RNEA로 계산한 τ_ff를 초기값으로, joint limit/마찰 추 제약 하에서 τ 최적화.

- [ ] 변수: [q̈; τ; λ]  (부유 베이스 포함 시 +6DOF)
- [ ] 등식: M(q)·q̈ + h = Sᵀ·τ + Jᵀ·λ  (완전 동역학)
- [ ] 부등식: τ_min ≤ τ ≤ τ_max, 마찰 추
- [ ] solver: qpOASES (pip install qpoases) 또는 qpsolvers
- [ ] task 우선순위: body pose > foot position > joint limit

---

## Phase 6 — 검증 & pinocchio 연동

- [ ] Σλ_z vs M·g 잔차 플롯 (힘 평형 검증)
- [ ] τ 에너지 Σ|τ·q̇| 계산 및 Figure 추가
- [ ] 관절 한계 위반 감지 + Figure 마킹
- [ ] (장기) pinocchio 연동: URDF 확보 후 RNEA/Jacobian 대체
      pip install pin
      pinocchio.buildModelFromUrdf()

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
