# HIND_VARIANT 비교 — `orig` vs `ext` (MPC+WBIC, v13.25)

## 배경

`_HIND_VARIANT` env (`gait_sim_v13.py:110`)가 `Q_HOME_HIND_DEG`를 분기 (`gait_sim_v13.py:371-376`):
- **`orig`**: 원본 home angle, 4발 간격 **401 mm**
  `Q_HOME_HIND_DEG = [0, -150.0, -90.0, 90.0, 60.0]`
- **`ext`**: 뒷발 -50 mm 확장, 4발 간격 **451 mm**
  `Q_HOME_HIND_DEG = [0, -154.8138, -92.8840, 88.6091, 60.0000]`

같은 조건(`MPC+WBIC, v=1.0 m/s, T=0.5s, trot, μ=0.6, 4 cycle, steady-state cycle 2~4`)에서 head-to-head.

## 결과 — 전 지표 비교

### [2] 에너지 효율성

| 지표 | orig | **ext** | 차이 |
|---|---|---|---|
| CoT (E_mech/mgd) | 3.967 | **3.708** | ext −6.5% |
| 기계적 일률 mean | 1543 W | **1453 W** | ext −6% |
| 기계적 일률 peak | 2836 W | 2838 W | 동일 |
| W+ / W- | 2095 / −220 J | 2001 / −179 J | ext 일·재흡수 모두 적음 |

### [3] 이동 성능·민첩성

| 지표 | orig | **ext** | 차이 |
|---|---|---|---|
| vx mean (오차%) | 0.988 (1.2%) | **0.995 (0.5%)** | ext 2.4× 정확 |
| 이동거리 x | 1.972 m | 1.985 m | ext 13mm 더 감 |
| y drift | +9.6 mm | **+3.9 mm** | ext **2.5× 작음** |
| Froude | 0.214 | 0.217 | 비슷 |
| stride/cycle | 0.4933 m | 0.4967 m | ext 약간 우수 |

### [4] 안정성·강인성 — **가장 큰 차이**

| 지표 | orig | **ext** | 차이 |
|---|---|---|---|
| roll_max | 0.681° | **0.523°** | ext 23% 안정 |
| pitch_max | 0.495° | **0.209°** | ext **58% 안정** |
| `\|ω\|max` | 0.198 rad/s | **0.128 rad/s** | ext 35% 작음 |
| z 수직 진동 | 14.0 mm | **6.5 mm** | ext **2× 안정** |
| **마찰콘 사용률 max** | **1.414** 🚨 (slip!) | **0.313** ✓ | ext 4.5× 여유 |
| stance foot slip p95 | 0.190 m/s | 0.192 m/s | 동일 |
| **토크 포화율 max** | **1.000** 🚨 (saturation!) | **0.952** | ext 마진 있음 |

⚠️ **`orig`는 마찰콘 1.41(>1=실제 slip), 토크 100% 포화 — 한계 상황**.

### [5] 제어 정밀도

| 지표 | orig | **ext** | 차이 |
|---|---|---|---|
| body pos err RMS x | 15.8 mm | **7.6 mm** | ext **2.1× 우수** |
| body pos err RMS y | 25.7 mm | **14.8 mm** | ext 1.7× 우수 |
| body pos err RMS z | 3.4 mm | **2.7 mm** | ext 우수 |
| body vel err RMS x | 0.017 m/s | **0.012 m/s** | ext 우수 |
| orientation RMS roll | 0.268° | 0.279° | 동일 |
| orientation RMS pitch | 0.161° | **0.068°** | ext 2.4× 우수 |
| orientation RMS yaw | 0.034° | 0.033° | 동일 |
| swing foot RMS aggregate | 131.1 mm | **128.2 mm** | ext 약간 우수 |
| swing foot 축별 x/y/z | 128.4 / 25.5 / 6.4 | 127.2 / 15.0 / **5.4** | ext y·z 우수 |

## 종합 판정

| | orig | ext |
|---|---|---|
| 에너지 | ★★★ | ★★★★ (CoT −6.5%) |
| 추종 정밀도 | ★★★ | ★★★★★ (body x 2×) |
| 자세 안정성 | ★★★ | ★★★★★ (pitch 58%↓) |
| **물리 한계 마진** | 🚨 slip + 포화 | ✓ 충분 |

**ext가 strict 개선 — trade-off 없음.** 모든 카테고리에서 동등 이상.

## 해석

뒷발을 50mm 더 뒤로 확장하면:
1. **Support polygon 면적 ↑** → 정적/동적 안정성 ↑ → roll/pitch ↓
2. **GRF 수평 분담 쉬워짐** → 마찰콘 사용률 1.41 → 0.31 (실제 slip 사라짐)
3. **모멘트 팔 길이 ↑** → 토크 분담 균등화 → 포화율 1.00 → 0.95
4. **결과적으로 body 추종 정확도 ↑** (자세 안정 → IK 정확 → body x 2배 우수)
5. **에너지 효율 ↑** (PD 보정량 감소 → 일률 −6%, CoT −6.5%)

## 권장 사항

- **기본값은 `ext` 유지** (현재 `gait_sim_v13.py:110` default가 `ext`). `orig`는 사실상 운용 불가.
- 비교 실험 외엔 `HIND_VARIANT=orig` 사용 금지 권장.
- 향후 hardware 빌드 시 뒷발 위치를 `ext` 기준(간격 451mm)으로 반영.

## 재현 명령

```bash
MPLBACKEND=Agg USE_NMPC=0 HIND_VARIANT=orig python3 gait_sim_v13.py
MPLBACKEND=Agg USE_NMPC=0 HIND_VARIANT=ext  python3 gait_sim_v13.py
```
