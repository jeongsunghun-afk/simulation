# Baseline Metrics (v12.8, pre v13 분할)

각 scenario subprocess 격리 실행. Phase 1+ 후 재실행 → 동일값 = no regression.

| Scenario | mode | x_final/target | y_final | z_range | vx mean/tgt | roll/pitch max | τ peak | Fz peak max | diverged |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| NMPC trot | _USE_NMPC_ACTIVE=True | 1.581 / 2.000 | +8 mm | 465~526 mm | 0.821 / 1.00 | 1.63° / 1.63° | 62.1 Nm | 502 N | ✅ |
| NMPC walk | _USE_NMPC_ACTIVE=True | 1.034 / 1.600 | -8 mm | 435~485 mm | 0.258 / 0.40 | 2.41° / 7.11° | 41.7 Nm | 340 N | ✅ |
| v11 trot | _USE_NMPC_ACTIVE=False | 2.006 / 2.000 | +67 mm | 462~469 mm | 1.003 / 1.00 | 0.32° / 0.08° | 93.1 Nm | 481 N | ✅ |
| v11 walk | _USE_NMPC_ACTIVE=False | 1.609 / 1.600 | -559 mm | 463~467 mm | 0.402 / 0.40 | 0.29° / 0.38° | 86.5 Nm | 247 N | ✅ |
