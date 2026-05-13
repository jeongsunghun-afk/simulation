# QP Solver Benchmark (v13.0a)

`quadprog` / `osqp` / `proxqp` 를 동일 시뮬레이션 (1 cycle trot, 250 frames) 에 적용 비교.

## 결과

| Solver | WBIC loop time | Relative speed | x_final | τ peak | Fz peak | wbic success |
|---|---:|---:|---:|---:|---:|---:|
| **quadprog** (baseline) | 9201 ms | **1.00×** | 0.492m | 120.0 Nm | 660 N | 100% |
| osqp | 9712 ms | 0.95× | 0.482m | 118.3 Nm | 803 N | 100% |
| proxqp | 10179 ms | 0.90× | 0.492m | 118.4 Nm | 654 N | 100% |

precompute (opt-IK SLSQP, solver 독립): 17.2s (공유)

## 분석

- **quadprog 가 가장 빠름** — 우리 QP 가 *작은 dense problem* (8~50 vars) 이므로 Goldfarb-Idnani algorithm 에 최적.
- **osqp**: sparse 변환 overhead + dual/primal infeasible warnings 다수 발생 (작은 dense formulation 부적합).
- **proxqp**: 정확한 결과 (x_final 일치), 약간 느림.
- 모두 wbic success 100%, divergence 없음.

## 결론

| 단계 | 권장 solver |
|---|---|
| **현재 Python** | **quadprog** (default 유지) |
| C++ 포팅 (v14) | osqp / qpoases C-API + **sparse formulation 직접 구성** + warm-start |
| Isaac Gym batched GPU | CUDA-based QP solver (예: pdipm batched) |

**Python 단계에서 solver 교체로 추가 향상 없음.** 진정한 향상은 C++ 포팅 + sparse 구성 필요.

## 향후 작업 (Tier 2 C++ 포팅)

1. **sparse formulation 직접 구성** — P, G, A 를 csc_matrix 로 (sparse conversion overhead 제거)
2. **warm-start** — 이전 frame 의 primal/dual 을 다음 solve 의 initial guess
3. **inequality 통합** — 마찰추 5× per stance leg → block-sparse pattern 활용
4. **MPC condensed form** — Aq/Bq decomposition 후 H 직접 sparse 구성

예상 추가 향상:
- sparse construction: 2~3× (현재 dense → sparse 변환 overhead 제거)
- warm-start: 2~5× (특히 receding horizon NMPC)
- C++ + dense (qpoases): 3~5× (interpreter overhead 제거)
- 종합: **10~20× 향상 가능** → Python RTF 0.018 → C++ 0.2~0.4 (real-time 가능)
