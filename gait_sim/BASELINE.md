# Baseline Metrics (v12.8, pre v13 분할)

각 scenario subprocess 격리 실행. Phase 1+ 후 재실행 → 동일값 = no regression.

| Scenario | mode | x_final/target | y_final | z_range | vx mean/tgt | roll/pitch max | τ peak | Fz peak max | diverged |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| NMPC trot | _USE_NMPC_ACTIVE=True | 1.683 / 2.000 | +39 mm | 465~498 mm | 0.842 / 1.00 | 2.69° / 1.13° | 92.6 Nm | 457 N | ✅ |
| NMPC walk | _USE_NMPC_ACTIVE=True | 0.803 / 1.600 | -18 mm | 390~471 mm | 0.200 / 0.40 | 3.71° / 6.96° | 78.2 Nm | 342 N | ✅ |
| v11 trot | _USE_NMPC_ACTIVE=False | 1.991 / 2.000 | +61 mm | 455~465 mm | 0.995 / 1.00 | 0.06° / 0.11° | 93.0 Nm | 455 N | ✅ |
| v11 walk | _USE_NMPC_ACTIVE=False | 1.613 / 1.600 | -565 mm | 461~466 mm | 0.403 / 0.40 | 0.19° / 0.36° | 86.6 Nm | 233 N | ✅ |
