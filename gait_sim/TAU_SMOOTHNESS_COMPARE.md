# τ smoothness 패치 효과 검증

**비교 대상**:
- pre  = `a1073cf` (v12.7.3) — DT_MPC=0.02, np.gradient, no Δτ penalty
- post = `3c9ab36` (v12.8)   — DT_MPC=0.01, CubicSpline 미분, w_dtau=0.001

**메소드**: 각 scenario subprocess 격리, `wbc_tau_cmd` 시계열 분석
- `tau_smoothness_compare.py` 로 5종 metric 계측 (peak, std, jerk peak, jerk rms, jerk p95)

## 결과

| Scenario | metric | pre | post | Δ |
|---|---|---:|---:|---:|
| **NMPC trot** | τ peak | 62.1 | 92.6 | **+49%** ❌ |
| | τ std | 12.63 | 14.14 | +12% ❌ |
| | jerk peak | 15673 | 17829 | +14% ❌ |
| | jerk rms | 1115 | 1036 | −7% ✅ |
| | jerk p95 | 1127 | 1039 | −8% ✅ |
| **NMPC walk** | τ peak | 41.7 | 78.2 | **+88%** ❌ |
| | τ std | 8.66 | 8.74 | +1% |
| | jerk peak | 9876 | 15185 | **+54%** ❌ |
| | jerk rms | 435 | 552 | +27% ❌ |
| | jerk p95 | 435 | 369 | −15% ✅ |
| **v11 trot** | τ peak | 93.1 | 93.0 | 0% |
| | τ std | 25.45 | 25.53 | 0% |
| | jerk peak | 8024 | 8907 | +11% ❌ |
| | jerk rms | 789 | 820 | +4% |
| | jerk p95 | 1650 | 1666 | +1% |
| **v11 walk** | τ peak | 86.5 | 86.6 | 0% |
| | τ std | 15.26 | 15.26 | 0% |
| | jerk peak | 5676 | 8028 | **+41%** ❌ |
| | jerk rms | 430 | 442 | +3% |
| | jerk p95 | 996 | 1014 | +2% |

## 결론

**패치 3종 (DT_MPC↓ + CubicSpline + Δτ penalty) 은 τ smoothness 개선에 실패.**

- NMPC: τ peak 49~88% 악화, jerk peak 14~54% 악화
- v11: τ peak/std 거의 동일, jerk peak 11~41% 악화
- 일부 NMPC jerk rms/p95 (-7~-15%) 만 개선, peak 폭주가 상쇄

## 원인 가설

1. **DT_MPC 0.02→0.01**: linear MPC refresh 빨라짐 → reference 더 급변 → τ jerk peak 상승
2. **w_dtau=0.001**: 너무 약함 (실효 없음). 0.01~0.05 필요했을 듯, 단 tracking 손상 위험
3. **CubicSpline q̇/q̈**: spline derivative endpoint ringing → jerk peak 악화 가능

## 후속 action 옵션

- **A. revert** `3c9ab36` (a1073cf 상태 복귀)
- **B. 부분 revert**: CubicSpline 만 revert, DT_MPC 유지 + reference interp 추가, w_dtau↑
- **C. 보류**: τ smoothness 는 actuator 모델 (Phase 7 Tier 1) 도입 시 재검토. 분할 우선.

사용자 결정 대기.
