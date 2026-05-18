# HL th2 tau_grf 비대칭 원인 진단 — v14.5.2

**대상**: `gait_sim_v13.py` (MPC+WBIC 모드, `USE_BODY_DYNAMICS=True`, `USE_MPC_CLOSED_LOOP=True`)
**관찰**: Figure 4 (tau decompose) 에서 HL th2 의 tau_grf 가 FR th2 대비 6× 큼 (peak 109.6 vs 17.5 Nm)
**날짜**: 2026-05-18

---

## 0. TL;DR

| 가설 | 결과 |
|---|---|
| MPC 행렬 부호 오류 | ❌ 부호 정확 (Di Carlo 2018 표준) |
| `_skew(r)` 정의 오류 | ❌ 표준 cross product |
| frame mismatch (R_b.T 누락 [:3090]) | ❌ 영향 ≈ 0 (body pitch 0.21°에서 R_b.T≈I) |
| HR vs HL 좌우 비대칭 | ❌ 기구학적으로 100% 거울 대칭, 동역학 차이는 body y drift(+14mm) 부수효과 (5-11%) |
| **gait 중 동적 발 위치 비대칭** | ✅ **주범** — 같은 phase 인 대각 짝 (FR+HL) 의 hip-foot 기하가 극단적 비대칭 |
| HL Jacobian col(th2)_z 변동 | ✅ 큰 Fz 를 큰 τ_th2 로 변환 (2차 증폭) |

**결론**: MPC 수식·부호 모두 정확. 진짜 원인은 **gait planning 단계에서 만들어진 발 착지 위치 비대칭** 을 MPC 가 충실히 반영. 즉 **"HL 만 큰 게 아니라 HIND 가 큰 것"**, 그리고 **gait/Q_HOME 설계 결함을 MPC 가 올바르게 풀고 있는 결과**.

---

## 1. 베이스라인 측정

`MPLBACKEND=Agg` + `plt.show` no-op 으로 sim 실행, steady-state 구간 통계.

```
[부록] FR / HL τ·dq·GRF peak + 좌우 대칭성  (MPC+WBIC)
  FR τ_cmd  peak [N·m]:   th2: 77.43  th3: 92.89  th4: 51.60
  HL τ_cmd  peak [N·m]:   th2: 93.96  th3: 49.89  th4: 57.12
  FR λ(GRF) peak [N]:     Fx= 19.83  Fy= 15.01  Fz=234.00
  HL λ(GRF) peak [N]:     Fx= 22.43  Fy= 49.71  Fz=515.58
  body 정상상태 y offset = +14.15 mm
  WBIC Δτ RMS [Nm] per leg  FR=1.18 FL=1.13 HR=1.78 HL=2.68
```

| 항목 | FR | HL | 비고 |
|---|---|---|---|
| τ_grf th2 peak | 17.5 Nm | **109.6 Nm** | 6.3× |
| τ_grf th2 rms | 6.7 | **33.2** | 5.0× |
| Fz peak (stance) | 234.0 N | **515.6 N** | 2.2× |
| Fz mean (stance) | 148.9 | **260.5** | 1.75× |
| 토크 포화율 max | 0.95 | 0.95 | 양쪽 모두 saturation 임박 |

---

## 2. 가설 1 — R_b.T frame mismatch (반증)

[gait_sim_v13.py:2937](../gait_sim_v13.py#L2937) 주석:
```
# tau_grf = -Jᵀ × R_body^T × λ_world         (body-local GRF feedforward)
```

NMPC 재분해 경로 [:2956](../gait_sim_v13.py#L2956) 는 `lam_local = R_b.T @ lam_world` 적용 ✓
MPC+WBIC 메인 루프 [:3090](../gait_sim_v13.py#L3090) 은 R_b.T **누락**.

**실험**: 누락된 R_b.T 적용 후 재실행
- HL th2 peak: 109.555 → 109.756 Nm (**+0.2%**)
- FR th2 peak: 17.452 → 17.709 Nm (+1.5%)
- body pitch peak: 0.21° (변함 없음)

→ closed-loop MPC 가 body 를 거의 수평 유지 (pitch 0.21°) → `R_b.T ≈ I` → **회전 보정 효과 무시**.
→ **가설 1 기각**.

(다만 fix 자체는 정합성상 옳음 — body 가 더 기우는 시나리오 (사면, sim2real 외란) 에 대비해 patch 권장. 본 코드는 revert 유지.)

---

## 3. 가설 2 — MPC 부호/수식 오류 (반증)

### 3.1 `_skew` 함수
[gait_sim_v13.py:755](../gait_sim_v13.py#L755), [:949](../gait_sim_v13.py#L949):
```python
def _skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])
```
→ `_skew(v) @ w == v × w` 표준 left cross product ✓

### 3.2 `_build_Bc_at`
[gait_sim_v13.py:1478](../gait_sim_v13.py#L1478):
```python
Bc[6:9,  i*3:(i+1)*3] = I_world_inv @ _skew(r)   # angular
Bc[9:12, i*3:(i+1)*3] = np.eye(3) / TOTAL_MASS    # linear
```
→ Di Carlo et al. (IROS 2018) "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control" 표준 형식 일치 ✓

### 3.3 MPC 단독 호출 검증
대칭 발 위치 (Q_HOME) + upright body 입력 시:

```
Case 1: 4-leg 영구 stance
  FR Fz=121.4  FL Fz=121.4  HR Fz=91.0  HL Fz=91.0
  → HIND/FRONT ratio = 0.750  ✓ (정적 이론 일치)

Case 2: trot 대각 stance (FR+HL only)
  FR Fz=257.1  HL Fz=197.1
  → FR/HL ratio = 1.304  ✓ (정적 이론 1.337에 근접)

Case 4: pitch=+5° 입력
  → 모든 Fz 가 FRONT 로 (HR=HL=0)
  → nose-down 토크 정상 생성 ✓
```

→ **MPC 자체는 모든 면에서 정확**. 가설 2 기각.

---

## 4. 가설 3 — 진짜 원인: gait 중 동적 발 위치 비대칭 (확정)

### 4.1 Q_HOME 정적 발 위치 (대칭)

| leg | hip_x [m] | foot_x [m] | foot–hip Δx [m] |
|---|---|---|---|
| FR/FL | +0.250 | +0.193 | -0.057 |
| HR/HL | -0.250 | -0.258 | -0.008 |

→ Q_HOME 자체도 약간 비대칭 (front foot reach 0.193 vs hind 0.258), 그러나 본질은 다음.

### 4.2 gait 진행 중 (frame fi=625, mid-cycle steady state) 발 위치

| leg | Q_HOME foot_x | **gait 중 foot_x** | hip-foot Δx |
|---|---|---|---|
| FR | +0.193 | **+0.398** | **+148mm 전방** |
| FL | +0.193 | +0.148 | -102mm |
| HR | -0.258 | -0.302 | -52mm |
| HL | -0.258 | **-0.052** | **+198mm 전방** |

→ **HL 발이 body x=0 (CoM) 근처**, **FR 발은 CoM 에서 0.4m 떨어짐**.

### 4.3 정적 평형 예측

FR+HL 대각 stance 의 pitch 평형:
```
0.398·F_FR + (-0.052)·F_HL = 0
→ F_HL = 7.65 × F_FR
→ F_FR ≈ 46 N,  F_HL ≈ 348 N  (이론)
```

같은 입력으로 MPC 재호출 결과:
```
FR Fz = 83 N,  HL Fz = 449 N  (비율 5.4)
```
→ 정적 예측 방향·크기 일치 ✓ — **MPC 가 주어진 발위치 기준으로 정확히 풀이 중**.

### 4.4 원인 메커니즘

- robot 이 V=1.0 m/s 로 전진
- stance 중 발은 ground 고정, body 가 발 위로 지나감 → 발이 body-local 에서 후방으로 이동
- trot 대각 짝 (FR+HL) 이 **같은 phase** 에서 stance — 하지만 hip 위치는 좌우 비대칭 (FR hip +0.25, HL hip -0.25)
- → 같은 stance-phase 시점에서 두 발의 body-local x 가 극단적으로 어긋남
- → 대각선이 CoM 안 지남 → MPC 가 pitch 평형 위해 lever-arm 짧은 다리 (HL) 에 더 큰 Fz 분배

---

## 5. HR vs HL 대칭성 (확인)

phase-aligned (stance progress 0~1) bin 평균 비교:

| 항목 | HR | HL | Δ |
|---|---|---|---|
| **foot_x** | -0.058 ~ -0.295 | -0.058 ~ -0.295 | **0.0000 (모든 bin)** ✓ |
| **Jacobian col(th2)_z** | +0.236 → -0.001 | +0.236 → -0.001 | **0.0000 (모든 bin)** ✓ |
| Fz 평균 | 247.9 N | 259.6 N | +4.7% |
| τ_grf th2 peak | 87.3 Nm | 97.1 Nm | +11.3% |

→ **기구학적으로는 100% 거울 대칭**. 5-11% 차이는 body y drift(+14mm) 의 부수효과.

정적 roll 평형 분석:
- HR foot y 거리 from CoM = 0.072 m
- HL foot y 거리 from CoM = 0.043 m
- 정적 예측: F_HL/F_HR = 1.65 (HL이 1.65× 더 받아야 평형)
- 실측 1.05 → MPC 가 roll moment 를 잘 보정 중 (잔차 5%)

→ **사용자가 본 HL 의 "큰" τ_grf 는 HR 도 같이 크다 (HIND 공통)**. 좌우 비대칭은 부차적.

---

## 6. τ_grf vs τ_pd 균형 — MPC+WBIC 설계 관점

stance 중 th2 분해 (RMS):

| leg | τ_dyn | **τ_grf** | **τ_pd** | τ_imp | τ_cmd | **grf/pd** |
|---|---|---|---|---|---|---|
| FR | 1.6 | 9.5 | 10.5 | 6.0 | 24.8 | **0.90** |
| FL | 1.6 | 10.2 | 10.5 | 6.0 | 25.1 | **0.97** |
| HR | 2.2 | 42.7 | 4.2 | 5.3 | 46.3 | **10.2** ✓ |
| HL | 2.2 | **46.9** | 4.2 | 5.3 | 48.9 | **11.2** ✓ |

표준 MPC+WBIC 권장: grf/pd > 5 (feedforward dominant). HIND 는 이상적 영역.
FRONT 는 grf/pd ≈ 1 — **border-line PD-dominant** 영역.

### 6.1 왜 FRONT 의 PD 가 큰가

| leg | q_err rms [°] | dq_err rms [rad/s] | KP·q_err | KD·dq_err |
|---|---|---|---|---|
| FR/FL | 3.79 | **1.11** | 5.3 | **8.8 (← 주범)** |
| HR/HL | 2.39 | 0.17 | 3.3 | 1.3 |

→ FR/FL th2 의 PD 는 **속도 오차 (KD·dq) 가 80%**.

WBIC Δλ_z RMS (feedforward 모델 오차 지표):
- FR/FL: 6 N
- HR/HL: 19-33 N

→ **흥미로운 모순**: feedforward 모델 정확도는 HIND 가 더 나쁘지만 (Δλ 큼), joint tracking 은 HIND 가 더 잘 됨. FRONT 는 feedforward 정확하지만 q tracking 안 좋음.

### 6.2 원인 가설

1. FRONT Q_HOME θ2 = +133° → swing 중 168°까지 35° 변동 → 빠른 dynamics
2. swing → stance 전환 시 impact (큰 dq jump) — FRONT 발이 더 멀리/빠르게 떨어져 충격 큼
3. plan q̇ peak 14-15 rad/s 도달 → actuator bandwidth 한계

### 6.3 사용자 질문 답

> "HR/HL 의 τ_grf > τ_pd 는 자연스러운가?"

✅ **자연스럽고 오히려 설계 의도 부합**. grf/pd > 5 는 MIT Cheetah / 표준 MPC+WBIC 가 추구하는 "feedforward dominant" 영역.

> "FR/FL 의 PD 비율 높음 = feedforward 가 못 잡는 것?"

⚠️ **부분적 yes**. 정확히는 "feedforward 모델 정확 (WBIC Δλ 작음), 그러나 **plan q̇ vs actual q̇ 추적 오차**가 큼". model accuracy 문제 아님.

---

## 7. Real robot impedance 활성화 시 효과

sim 의 MPC+WBIC ≠ real 의 control architecture (real 은 position control + 옵셔널 impedance).

| 시나리오 | impedance (forceS) 효과 |
|---|---|
| 착지 impact 흡수 | ✅ **크게 좋아짐** |
| 지형 불규칙 | ✅ **좋아짐** |
| stance 외력 | ✅ **좋아짐** |
| plan q̇ vs actual q̇ mismatch | ⚠️ 간접적 |
| MPC GRF 분배 비대칭 | ❌ **무관** (impedance 가 GRF 분배 자체는 못 바꿈) |

real 에서 sim 의 dq_err = 1.1 rad/s 이 그대로 나타날 가능성은 낮음 — MPC plan 의 aggressive trajectory 가 real 에서는 들어오지 않음 (RL / user command via /low_cmd 의 200Hz Hermite 보간 사용).

---

## 8. 권장 조치 (우선순위 순)

### A. sim 개선 (gait_sim_v13 / v14 후속)

1. **Q_HOME 재튜닝** — front/hind foot_x 절대값 동일하게. HIND θ2 = -150° → -130~135° 로 덜 접어서 reach 늘리기. Q_HOME_FRONT 도 swing 중간 자세 근처로 이동해 dq 변동 폭 축소
2. **swing trajectory 재설계** — trot 대각 짝이 같은 stance phase 시점에서 body-local x 가 CoM 대칭이 되도록 swing offset 보정
3. **MPC body 모델에 CoM offset 명시** — 다리 mass(18kg) > body mass(15kg). CRBA-based actual CoM (현재 0,0,0 가정) → 실제 위치
4. **R_b.T fix [:3090]** — 정합성 항목. 평지 trot 영향 ≈ 0, 외란·사면 시나리오 대비

### B. real (motorcortex_bridge) 운용

1. **stance 중 forceS=ON** — 착지 부드러움 + 외란 흡수
2. **forceT + forceF** — GRF feedback 으로 stance 강성 보강
3. **KP_PD 감소 가능** — impedance 가 보충하므로 PD 의존도 ↓

### C. sim2real 전 빚

- **HIND τ saturation 0.95** — real 에서 모터 발열·마모 직결. Q_HOME 튜닝 (A.1) 으로 완화
- **body y drift +14mm** — MPC py weight 보강 또는 _HIP_Y_BIAS 미세조정

---

## 9. 검증 산출물 (외부 보존)

| 파일 | 용도 |
|---|---|
| `/tmp/verify_taugrf.py` | static unit test — pin Jacobian + body pitch β |
| `/tmp/run_sim_capture.py` | sim 실행 wrapper (matplotlib Agg + plt.show no-op) |
| `/tmp/diag_hl_th2.py` | DH Jacobian / lam_des 분해 |
| `/tmp/foot_geom.py` | Q_HOME 발 위치 + 정적 평형 이론값 |
| `/tmp/mpc_static_test.py` | mpc_qp_plan 단독 호출 — 대칭 입력 검증 |
| `/tmp/dump_mpc_inputs.py` | sim 의 stance pattern 통계 + mid-stance 분포 |
| `/tmp/hook_mpc.py` | sim 실 frame 의 MPC 입력 재구성 + 재호출 |
| `/tmp/hr_hl_compare.py` | HR vs HL phase-aligned 비교 |
| `/tmp/tau_decompose.py` | tau_cmd 4성분 분해 (dyn / grf / pd / imp) |
| `/tmp/q_track.py` | joint tracking error + WBIC Δλ |
| `/tmp/baseline.json` | sim 통계 dump (baseline) |
| `/tmp/fixed.json` | R_b.T fix 적용 후 통계 (참고용) |
| `/tmp/gait_sim_v13.orig.py` | 분석 시작 시점 백업 |

---

## 10. 종합 결론

1. **MPC 수식·부호 모두 정확** — Di Carlo 표준 형식 완벽 일치
2. **HL th2 가 큰 게 아니라 HIND th2 가 공통으로 큼** — 좌우 거울 대칭, 5-11% 차이는 body y drift 부수효과
3. **진짜 원인은 gait planning** — 같은 phase 의 대각 짝 hip-foot 기하 비대칭이 MPC 에 의해 충실히 GRF 분배로 반영
4. **τ_grf >> τ_pd (HIND) 는 정상이고 좋음** — feedforward dominant, model-based 제어 의도와 일치
5. **τ_grf ≈ τ_pd (FRONT) 는 q̇ tracking error 가 원인** — feedforward 모델 자체는 정확
6. **real 운용에서 impedance 는 착지·외란만 개선** — GRF 분배 비대칭은 sim 자체 (gait/IK) 개선이 답
7. **현재 시나리오 (정상 보행) 에서 큰 문제 없음** — saturation 0.95 와 sim2real 빚으로 v14 이후 정리 필요

본 분석은 **gait_sim_v13.py 의 control accuracy 검증** 측면에서는 **MPC+WBIC 정상 동작** 을 확인했고, 향후 개선은 **plan side (Q_HOME / swing trajectory)** 와 **MPC body 모델 정확도 (CoM offset)** 가 핵심 lever 라는 점을 입증.
