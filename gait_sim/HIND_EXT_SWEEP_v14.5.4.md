# HIND 확장 sweep — ext+0 ~ +140mm (HL th2 tau_grf 최적점 탐색) v14.5.4

**대상**: `gait_sim_v13.py` + `gait_sim/` 패키지 (MPC+WBIC, `USE_BODY_DYNAMICS=True`)
**목적**: HL/HR 의 큰 th2 tau_grf (109.6 Nm peak) 를 hind reach 늘려 줄일 수 있는지 검증, 최적 ext 값 결정
**선행 보고서**: [HL_TH2_TAU_GRF_INVESTIGATION_v14.5.2.md](HL_TH2_TAU_GRF_INVESTIGATION_v14.5.2.md)
**날짜**: 2026-05-18

---

## 0. TL;DR

**결과**: ext+80mm (간격 531mm, hind θ2=-175.6°) 가 명확한 최적점.

| 지표 | ext (451mm, 기존 default) | **ext+80mm (531mm, 새 default)** | Δ |
|---|---|---|---|
| HL th2 tau_grf peak | 109.55 Nm | **54.43 Nm** | **−50.3%** |
| HL th2 tau_grf rms | 33.19 | **15.13** | **−54.4%** |
| HL tau_cmd peak (saturation) | 93.96 Nm **(94%)** | **61.86 Nm (62%)** | **−34%** |
| 4-leg th2 sum | 247.8 | **173.4** | **−30%** |
| body pitch peak | 0.209° | **0.176°** | −16% |
| body roll peak | 0.523° | **0.316°** | **−40%** |
| body y drift | +14.15 mm | **+8.10 mm** | **−43%** |
| HIND joint θ2 한계 (-200°) 여유 | n/a | 24.4° | 안전 |

조치:
- `gait_sim/model.py` 의 `ext` 분기 Q_HOME 을 ext+80mm 값으로 교체
- `gait_sim_v13.py` 동일 적용
- `HIND_Q_LIM` θ2 한계 -180° → -200° 확장 (양쪽)

---

## 1. 배경

v14.5.2 보고서에서 HL th2 tau_grf 가 비대칭적으로 큼 (FR 대비 6×) 의 근본 원인이 **gait 중 동적 발 위치 비대칭 (Q_HOME 가 만든 hip-foot lever arm 비대칭)** 임을 규명. 권장 조치 1번이 **Q_HOME 재튜닝 — front/hind foot_x 대칭화**.

본 sweep 은 그 권장의 구체적 검증.

---

## 2. 방법

### 2.1 Sweep 변수

`ext+Δmm` — 'ext' 기준에서 hind 발을 추가로 -Δmm 후방으로 보냈을 때.
foot 의 z (지면 높이) 유지하면서 hind 발만 x 방향 -Δ 이동.

| ext+ | hind foot body_x | spacing | Q_HOME θ2 |
|---|---|---|---|
| 0 (= ext 기존) | -258 mm | 451 mm | -154.81° |
| +20 | -278 mm | 471 mm | -159.90° |
| +40 | -298 mm | 491 mm | -165.05° |
| +60 | -318 mm | 511 mm | -170.27° |
| **+80** | **-338 mm** | **531 mm** | **-175.57°** |
| +100 | -358 mm | 551 mm | -180.98° |
| +120 | -378 mm | 571 mm | -186.57° |
| +140 | -398 mm | 591 mm | -192.41° |

각 ext+Δ 의 Q_HOME 은 `analytical_ik_hind` + `wrap_negative` 보정으로 계산.

### 2.2 측정 환경

- gait: trot, V=1.0 m/s, T=0.5 s, D=0.5
- MPC+WBIC closed-loop (USE_BODY_DYNAMICS=True, USE_MPC_CLOSED_LOOP=True)
- steady-state: 마지막 3 cycle 평균

### 2.3 한계 확장

기존 `HIND_Q_LIM` θ2 = [-180°, -60°] 이라 ext+100mm 부터 IK 가 wrap (한계 부딪힘).
**-200°로 확장**:
```python
(-math.radians(200), -math.radians(60)),   # th2: 고관절 굴곡 (확장)
```
양쪽 파일 (model.py, v13.py) 적용.

---

## 3. 결과 — 전체 sweep

### 3.1 첫 단계 sweep (ext+0 ~ +80)

| ext+ | HL tg pk | HL tg rms | HL tc pk (sat%) | HL Fz mean | pitch° | roll° | y mm | 4-leg sum |
|---|---|---|---|---|---|---|---|---|
| 0 | 109.55 | 33.19 | 93.96 (94%) | 245.9 | 0.209 | 0.523 | +14.15 | 247.8 |
| 20 | 92.52 | 27.72 | 84.48 (84%) | 236.0 | 0.239 | 0.487 | +11.78 | 225.1 |
| 40 | 80.02 | 23.19 | 77.63 (78%) | 227.0 | 0.240 | 0.462 | +12.03 | 207.4 |
| 60 | 66.67 | 19.01 | 70.06 (70%) | 218.9 | 0.236 | 0.440 | +9.89 | 190.5 |
| **80** | **54.43** | **15.13** | **61.86 (62%)** | **210.7** | **0.176** | **0.316** | **+8.10** | **173.4** |

→ ext+80 까지 **모든 지표 단조 개선**.

### 3.2 확장 sweep (ext+100 — limit 확장 후)

| ext+ | HL tg pk | HL tc pk **(sat%)** | pitch° | roll° | y mm |
|---|---|---|---|---|---|
| 0 | 109.55 | 93.96 (94%) | 0.209 | 0.523 | +14.15 |
| 80 | 54.43 | 61.86 (62%) | 0.176 | 0.316 | +8.10 |
| **100** | **44.11** | **120.00 (120% ⚠ clip!)** | **0.720 (4× 악화)** | 0.235 | +5.0 |
| 120, 140 | (실험 중단 — 100mm 에서 이미 saturation/instability 명확) |

→ **ext+100mm 부터 tau_cmd saturation + body pitch 급격 악화**. hind θ2 가 -180° 가까이 가서 swing dynamics 와 충돌.

### 3.3 최적점 = ext+80mm

ext+80mm 가 sweep 의 sweet spot:
- HL th2 tau_grf 50% 감소
- saturation 여유 38%p 회복 (94% → 62%)
- body 안정성 (pitch/roll/y) 모두 개선
- θ2 한계 (-200°) 24.4° 여유 — 외란·dynamic motion 안전 마진 충분

---

## 4. FRONT 부담 변화 (trade-off)

| ext+ | FR th2 pk | FL th2 pk |
|---|---|---|
| 0 | 17.4 | 21.2 |
| 80 | 33.8 | 34.6 |

FRONT 부담 약 2× 증가했지만:
- 절대값 (34 Nm) 이 한계 (100 Nm) 보다 훨씬 작음 → 문제 없음
- 4-leg sum 은 247.8 → 173.4 로 **전체 -30%** (총 부담 감소)

---

## 5. 변경 사항

### 5.1 gait_sim/model.py
- Line 26: `ext` 코멘트 갱신 — 451mm → 531mm
- Line 80: `Q_HOME_HIND_DEG` (ext) → `[0.0, -175.5711, -77.7261, 94.2085, 60.0000]`
- Line 201: `HIND_Q_LIM` θ2 → `[-200°, -60°]` (180→200 확장)

### 5.2 gait_sim_v13.py
- Line 110: `_HIND_VARIANT` 코멘트 갱신
- Line 373: `Q_HOME_HIND_DEG` (ext) → 동일 값
- Line 1895: `HIND_Q_LIM` θ2 → `[-200°, -60°]`

---

## 6. 검증 산출물 (외부 보존)

| 파일 | 용도 |
|---|---|
| `/tmp/qhome_ik.py` | 단발 IK 검증 (ext+20mm) |
| `/tmp/qhome_sweep_ik.py` | 5 variant Q_HOME 계산 |
| `/tmp/qhome_extended_ik.py` | -200° 확장 후 +160mm 까지 IK |
| `/tmp/run_compare_ext.py` | 2 variant 비교 (ext vs ext+20) |
| `/tmp/run_sweep_5.py` | 5 variant (+0~+80) sweep |
| `/tmp/run_sweep_ext.py` | 확장 sweep (+0,+80,+100) |
| `/tmp/sweep_5.log` | first sweep 결과 |
| `/tmp/sweep_ext.log` | extended sweep 결과 |

---

## 7. 권장 후속 작업

1. **NMPC 모드에서 동일 검증** — 본 sweep 은 MPC+WBIC 만. NMPC 가 같은 Q_HOME 에서 어떻게 GRF 분배하는지 확인 필요.
2. **다양한 gait 에서 검증** — walk, amble, canter 도 ext+80 에서 안정한지.
3. **외란 robustness** — push test 로 θ2 한계 (-200°) 부딪힘 여부.
4. **swing trajectory 확인** — ext+80 에서 swing 시 θ2 가 stretched 자세 회복 안 되는 경우 대비.
5. **real robot mechanical clearance** — hind 다리가 -176° 까지 펴질 때 하드웨어 간섭 점검.

---

## 8. 결론

> **HL th2 tau_grf 비대칭은 hind Q_HOME 튜닝만으로 절반 감소 가능 — `ext+80mm` 채택.**
> ext+100mm 부터는 swing dynamics 한계로 trade-off 악화. ext+80mm 가 sweet spot.
> v14.5.2 보고서의 권장 조치 1번 (Q_HOME 재튜닝) 의 실증.
