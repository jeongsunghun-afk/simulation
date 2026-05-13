# Baseline Metrics (v14 = gait_sim.runner 기반, src: gait_sim.runner (v13.6+))

각 scenario subprocess 격리 실행. v13.1 BASELINE_v13.md 와 비교 → 회귀 없음 검증.

| Scenario | use_nmpc | x_final/target | y_final | z_range | vx mean/tgt | roll/pitch max | τ peak | Fz peak max | diverged |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| NMPC trot | True | 1.577 / 2.000 | +12 mm | 465~526 mm | 0.819 / 1.00 | 1.32° / 1.54° | 76.9 Nm | 568 N | ✅ |
| NMPC walk | True | 1.068 / 1.600 | -6 mm | 444~488 mm | 0.266 / 0.40 | 2.64° / 8.16° | 42.0 Nm | 339 N | ✅ |
| MPC+WBIC trot | False | 1.974 / 2.000 | +10 mm | 454~471 mm | 0.987 / 1.00 | 0.75° / 0.56° | 120.0 Nm | 660 N | ✅ |
| MPC+WBIC walk | False | 1.560 / 1.600 | +7 mm | 456~469 mm | 0.390 / 0.40 | 0.98° / 0.72° | 90.5 Nm | 447 N | ✅ |
