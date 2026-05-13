# Timing Benchmark (v13.9, gait_sim.runner)

Stage 별 walltime + per-call QP latency 측정 — Isaac Lab 통합 실시간성 평가용.

**측정 조건**: trot, V=1m/s, T=0.5s, N_FRAMES=250 (1 cycle), DT=2ms (500Hz sim)
**Hardware**: CPU only (Python + numpy + scipy SLSQP + qpsolvers/quadprog)

## MPC+WBIC mode

| Stage | Walltime | Per-frame |
|---|---:|---:|
| precompute_trajectories (opt-IK SLSQP) | **17.7 s** | **71 ms/frame** |
| compute_derivatives                     | 0 ms | — |
| run_wbic_loop                            | **9.6 s** | **38 ms/frame** |
| postprocess_foot_world                  | 3 ms | — |
| **Total** | **27.3 s** | **109 ms/frame** |

**Per-call QP latency** (n=50 sample):

| Call | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| MPC QP (N=10) (quadprog) | 1.33 ms | 1.11 ms | 2.52 ms | 2.88 ms |
| WBIC per-leg QP (8 vars + bounds) | 168 μs | 134 μs | 354 μs | 531 μs |
| WBIC full body 6-DoF QP (50 vars) | 704 μs | 590 μs | 1.37 ms | 1.55 ms |

**Real-time factor**: `500 ms sim / 27.3 s wall = 0.018x` → **55× 느림**

### 병목 분석

| 부분 | 시간 점유 | 비고 |
|---|---|---|
| **opt-IK SLSQP** (precompute) | **65%** | per-frame 4 legs × ~18ms SLSQP scipy.optimize |
| **WBIC main loop** | 35% | RNEA + M·h + per-leg QP + body integrate |
| MPC QP | ~3% | N=10 horizon, quadprog C backend (이미 빠름) |
| WBIC QP | ~5% | leg/full QP 모두 sub-ms |

→ opt-IK SLSQP 가 가장 큰 단일 비용. C++ + analytical IK 사용 시 ~10× 향상 가능.

## NMPC mode

| Stage | Walltime |
|---|---:|
| precompute_trajectories (opt-IK 같음) | 17.3 s |
| solve_nmpc_receding (FDDP, 3 solves) | **845 ms** |
| ┗ per-solve | **280 ms** |
| populate_simstate                      | 387 ms |
| **Total** | **18.6 s** |

**Real-time factor**: 0.027x

### NMPC 특성

- per-solve 280 ms → 효율 frequency ~3.5 Hz
- DT_NMPC = 25 ms 시 1 solve 가 11 step 가속 필요 → **real-time 통합 불가능**
- 용도: **offline NMPC trajectory dataset 생성** → RL imitation reward 학습용 (v14.y)

## Isaac Lab 통합 함의

### Tier 1 (Isaac Sim sim2sim, single env)

| Component | 필요 freq | 현재 Python | C++ + osqp/qpoases | Isaac 통합 |
|---|---|---|---|---|
| MPC | 50 Hz | 1.3 ms (753 Hz 가능) | 0.2 ms | ✅ |
| WBIC full QP | 500 Hz | 0.7 ms (1428 Hz) | 0.2 ms | ✅ |
| opt-IK | per swing | 18 ms/leg | 1-2 ms (analytic IK + Newton refine) | ✅ |
| RNEA + M·h | 500 Hz | (in WBIC budget) | 0.1 ms | ✅ |

→ **Tier 1 sim2sim 단일환경**: C++ 포팅 + osqp 시 real-time 가능

### Tier 2 (Isaac Gym/Lab, 4096 envs GPU)

- WBIC/MPC C++ kernel 의 batched CUDA 구현 필요 (난이도 ↑)
- 또는: RL policy 직접 학습 (MPC/WBIC 없이), NMPC trajectory imitation reward 추가
- → **Tier 2 권장 경로**: legged_gym + RL policy + NMPC offline dataset

### NMPC standalone (real-time)

- 280 ms/solve = **불가능** for real-time
- Offline trajectory generation 용도만 (v14.y NMPC dataset)

## 다음 우선순위

1. **v13.0a**: qpOASES/proxqp + warm-start benchmark — Python 자체 ~2-5× 추가 향상 가능
2. **v14**: Isaac Sim sim2sim Tier 1 — MPC/WBIC C++ 포팅 (~3개월)
3. **v14.y**: NMPC trajectory dataset 생성 (현재 코드 그대로, 4 Hz 충분)
4. **v14.z**: RL policy + NMPC imitation reward (Isaac Gym)
