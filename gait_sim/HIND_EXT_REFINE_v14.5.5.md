# HIND ext sweep 후속 — walk + dq + GRF + BLEND 종합 분석 v14.5.5

**대상**: `gait_sim_v13.py` + `gait_sim/` 패키지 (MPC+WBIC, USE_BODY_DYNAMICS=True)
**선행 보고서**:
- [HL_TH2_TAU_GRF_INVESTIGATION_v14.5.2.md](HL_TH2_TAU_GRF_INVESTIGATION_v14.5.2.md) — HL th2 비대칭 원인 진단
- [HIND_EXT_SWEEP_v14.5.4.md](HIND_EXT_SWEEP_v14.5.4.md) — ext+80mm (531mm) 채택

**날짜**: 2026-05-18

---

## 0. TL;DR

v14.5.4 의 531mm 채택 직후 다음 4가지 후속 검증:

| 실험 | 결과 | 영향 |
|---|---|---|
| **walk 안정성 비교** (491 vs 531) | walk 에서 **491mm 가 우세** (pitch -33%, roll -21%) | 491mm 채택 (multi-gait safety) |
| **trot full sweep** (5 variants × τ/dq/GRF) | dq 단조 감소, GRF ratio 0.62→0.89 (균형) | data 보존 |
| **531mm phase split** (swing vs stance) | **FRONT 모든 joint SWING dominant** (th3: 92.9 vs 38.7) | swing trajectory 가 원인 확인 |
| **SWING_BLEND on/off 비교** | **OFF 시 FR th3 dq −57%, τ −56%** with body 안정성 유지 + foot lift 보존 | env 변수 추가 |

**최종 채택 변경**:
- ext 값: **531mm → 491mm** (walk 안정성 우선) — 531mm 는 코드에 주석으로 보존
- **SWING_BLEND env 변수 추가**: `SWING_BLEND=0 python -m gait_sim` 로 토글 가능
- figure 3 라벨: "tau_cmd − tau_grf" → "tau_cmd" (full motor 명령 표시)

---

## 1. WALK 안정성 — 491mm vs 531mm

trot 만 보면 531mm 가 최적이지만, **walk gait 에서 trade-off 발생**.

### 1.1 가설
531mm 는 FR 이륙위치(+68mm) ↔ HL 착지위치(-213mm) 의 gap 이 281mm. 491mm 는 241mm. walk 의 4-beat 비대칭 시퀀스 (FL→HR→FR→HL) 에서 발 footprint overlap 깨질 위험.

### 1.2 측정 결과 (walk, V=0.4 m/s, T=1.0s, D=0.25)

| 지표 | **491mm** | 531mm | Δ |
|---|---|---|---|
| body pitch peak | **0.868°** | 1.154° | **+33% 악화 (531)** |
| body roll peak | **0.991°** | 1.201° | **+21% 악화 (531)** |
| body yaw peak | 0.383° | 0.419° | +9% |
| body y drift | -5.91mm | -5.58mm | 비슷 |
| FR th2 τ peak | **59.8** | 67.0 | +12% |
| FR th3 τ peak | **86.5** | 90.9 | +5% |
| FR th4 τ peak | **51.5** | 54.4 | +6% |
| HL th2 τ peak | 39.4 | 41.6 | +6% |
| HL th3 τ peak | 36.5 | **32.4** | -11% |

→ **walk 에서는 491mm 우월** (FRONT 부담 5~12% 작음, body 안정성 큰 차이).

### 1.3 결론

| gait | 최적 |
|---|---|
| trot 만 | 531mm |
| walk 만 | 491mm |
| **trot + walk** | **491mm** (safety) |

memory 의 [[sim2sim_sim2real_tracks]] 따라 multi-gait 가능성 → **491mm 채택**.

---

## 2. Trot full sweep — τ + dq + GRF (5 spacings)

### 2.1 τ_cmd peak (Nm) — HIND 만 변동

| spacing | FR th2 | FR th3 | FR th4 | HL th2 | HL th3 | HL th4 |
|---|---|---|---|---|---|---|
| 451 | 77.4 | 92.9 | 51.6 | 94.0 | 49.9 | 57.1 |
| 471 | 77.4 | 92.9 | 51.6 | 84.5 | 42.7 | 56.5 |
| **491** | 77.4 | 92.9 | 51.6 | **77.6** | 40.2 | 56.4 |
| 511 | 77.4 | 92.9 | 51.6 | 70.1 | 36.5 | 55.7 |
| **531** | 77.4 | 92.9 | 51.6 | **61.9** | 34.4 | 55.9 |

**FR 완전 일정** — FRONT peak 가 swing phase 에서 발생하므로 HIND_VARIANT 무관.

### 2.2 dq peak (rad/s) — joint 각속도

| spacing | FR th2 | **FR th3** | **FR th4** | HL th2 | HL th3 | HL th4 |
|---|---|---|---|---|---|---|
| 451 | 10.84 | **14.99** | **14.66** | 5.45 | 7.81 | 11.22 |
| 491 | 10.84 | **14.99** | **14.66** | 6.09 | 6.32 | 10.52 |
| 531 | 10.84 | **14.99** | **14.66** | 6.44 | 5.12 | 9.74 |

→ **FR th3/th4 dq 가 한계 15 rad/s 근접** — 모든 spacing 에서 동일.
→ HL 은 spacing 따라 약간 변동 (th3 감소, th2 증가).

### 2.3 GRF Fz mean + ratio

| spacing | FR Fz | FL Fz | HR Fz | HL Fz | **FR/HR** | **front/hind** |
|---|---|---|---|---|---|---|
| 451 | 148.0 | 153.8 | 240.8 | 245.9 | **0.62** | 0.62 |
| 471 | 157.9 | 163.2 | 231.5 | 236.0 | 0.68 | 0.69 |
| 491 | 166.3 | 172.5 | 222.7 | 227.0 | **0.75** | 0.75 |
| 511 | 174.4 | 180.7 | 214.7 | 218.9 | 0.81 | 0.82 |
| 531 | 182.8 | 187.6 | 207.8 | 210.7 | **0.88** | **0.89** |

→ HIND_VARIANT 가 GRF 분배 직접 조절 — 451: hind 가 1.6배 받음, 531: 거의 균형 (0.89).

### 2.4 핵심 모순 해명

> "FR/L GRF 증가 (148→183) 했는데 왜 FR τ_cmd 안 커지나?"

**FR peak τ 는 swing 에서 발생** → stance Fz 와 무관. swing trajectory plan 이 HIND_VARIANT 와 독립적이라 FR τ 도 일정.

---

## 3. 531mm Phase split — swing peak vs stance peak

### 3.1 결과

| leg | joint | swing peak | stance peak | dominant | 차이 |
|---|---|---|---|---|---|
| **FR** | **th2** | **77.4** | 59.1 | **SWING** | +18 |
| **FR** | **th3** | **92.9** | 38.7 | **SWING** | **+54** |
| **FR** | **th4** | **51.6** | 29.7 | **SWING** | +22 |
| HR | th2 | 57.3 | 58.7 | STANCE | +1 |
| HR | th3 | 34.5 | 27.0 | SWING | +7 |
| HR | th4 | 44.7 | 54.8 | STANCE | +10 |
| HL | th2 | 57.3 | **61.9** | STANCE | +5 |
| HL | th3 | 34.5 | 28.7 | SWING | +6 |
| HL | th4 | 44.7 | **55.9** | STANCE | +11 |

### 3.2 통찰

| | swing dominant | stance dominant |
|---|---|---|
| **FRONT (FR/FL)** | **모든 th2/th3/th4** (특히 th3 차이 54 Nm) | th1, th5 (작음) |
| **HIND (HR/HL)** | th3 (작은 차이 ~7 Nm) | th2, th4 |

원인: **FRONT Q_HOME bent posture** (knee +47°) — swing 중 큰 자세 변화 → 큰 dq/ddq → 큰 τ_dyn.
HIND extended posture — stance lever arm 길어 stance τ 큼, swing 자세 변화 작음.

→ "STANCE 가 가장 큰 τ" 라는 사람 직관은 **straight-leg animal (horse)** 에서 성립.
   crouched robot (Spot, Cheetah, 이 robot) 에서는 swing-dominant 가능.

---

## 4. SWING_BLEND on/off 비교 — 핵심 발견

### 4.1 USE_SWING_QREF_BLEND 의미

FRONT 다리만 영향. swing 중 opt_ik 의 q_ref 가:
- **True** (default): `Q_HOME → Q_SW (foot lift 자세) → Q_HOME` 으로 quintic blend → opt_ik 가 swing 자세 추종 → **큰 dq/τ**
- **False**: `q_ref = Q_HOME` 고정 → 자세 변화 작음 → **작은 dq/τ**

### 4.2 측정 결과 (491mm trot)

| FR joint | **ON (현재)** | **OFF** | Δ |
|---|---|---|---|
| **th2 τ_cmd peak** | 77.4 | **61.7** | **-20%** |
| **th2 dq peak** | 10.84 | 8.80 | -19% |
| **th3 τ_cmd peak** | **92.9** | **40.5** | **-56%** |
| **th3 dq peak** | **14.99** (한계!) | **6.40** | **-57%** 큰 여유 |
| **th4 τ_cmd peak** | 51.6 | 27.4 | -47% |
| **th4 dq peak** | **14.66** (한계!) | **6.79** | **-54%** |

### 4.3 Body 안정성 + foot lift (영향 미미)

| | ON | OFF |
|---|---|---|
| body pitch peak | 0.240° | 0.237° (동일) |
| body roll peak | 0.462° | **0.403°** (-13% 개선!) |
| body y drift | +12.03mm | +12.28mm (동일) |
| **FR foot lift** | **80mm** | **80mm** ← **유지** ✓ |

### 4.4 거의 free lunch

- **FR τ/dq 대폭 감소** (한계 여유 회복)
- **body 안정성 유지** (roll 약간 개선)
- **foot lift height 그대로** — 발 들기 효과 안 잃음
- swing 중 자세 변화만 작아짐 (opt_ik 가 home 근처 해 선택)

### 4.5 환경 변수 추가

`gait_sim/config.py` 에 SWING_BLEND env 추가:
```python
# config.py
_sb = os.environ.get('SWING_BLEND')
if _sb is not None:
    CFG.use_swing_qref_blend = _sb not in ('0', 'false', 'False', '')
```

사용:
```bash
python -m gait_sim                     # default (BLEND ON)
SWING_BLEND=0 python -m gait_sim       # BLEND OFF (FR τ/dq 대폭 감소)
```

---

## 5. 변경 사항 정리 (v14.5.5)

### gait_sim/config.py
- `import os` 추가
- `SWING_BLEND` env 변수 처리 (CFG.use_swing_qref_blend override)

### gait_sim/model.py
- ext default 값: **531mm → 491mm** (walk safety)
- 531mm 값은 주석으로 보존 (`Q_HOME_HIND_DEG = [-175.57, -77.73, 94.21, ...]`)
- HIND variant 코멘트: 531 → 491mm 로 갱신

### gait_sim_v13.py
- 동일 (ext 값 491mm, 531mm 주석)
- HIND_VARIANT 코멘트 갱신

### gait_sim/viz/fig_wbc.py
- Figure 3 row 0 라벨: `tau_cmd − tau_grf` → **`tau_cmd`**
- 데이터: `wbc_tau_cmd - wbc_tau_grf` → `wbc_tau_cmd` (full motor 명령)

---

## 6. 검증 산출물 (외부 보존)

| 파일 | 용도 |
|---|---|
| `/tmp/walk_compare.py` + `.json` + `.log` | walk gait 491 vs 531mm |
| `/tmp/run_trot_full.py` + `.json` + `.log` | 5-variant trot full sweep (τ+dq+GRF) |
| `/tmp/phase_split_531.py` + `.log` | 531mm swing/stance peak split |
| `/tmp/blend_compare.py` + `.json` + `.log` | SWING_BLEND on/off |
| `/tmp/peak_phase.py` | peak frame phase 분류 |

---

## 7. 후속 권장 작업

1. **SWING_BLEND=0 영구 채택 검토** — 거의 free lunch (FR τ/dq -50%~-57%, body 유지). walk 에서도 검증 필요.
2. **FRONT Q_HOME 튜닝** — knee +47° 굽힘 줄이면 swing dynamics ↓. 별도 sweep.
3. **NMPC 모드에서 동일 검증** — 본 sweep 은 MPC+WBIC 만.
4. **각종 gait 외란 robustness** — push test, slope.
5. **real robot mechanical clearance** — 491mm θ2=-165° 에서 하드웨어 간섭 없는지.

---

## 8. 종합 결론

> **trot peak 최저는 531mm 이지만, walk 안정성 + multi-gait 대비 491mm 가 안전한 선택.**
> **SWING_BLEND=0 가 거의 free lunch — FR τ/dq -50%~-57% 인데 body 안정성·foot lift 유지.**
> env 변수로 쉽게 토글 가능하게 환경 변수 추가.

v14.5.4 의 531mm 채택 후, walk 안정성 검증 → 491mm 로 회귀 + 531mm 코드에 주석으로 보존.
v14.5.5 에서 BLEND 토글 인프라 + figure 3 라벨 정리.
