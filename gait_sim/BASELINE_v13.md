# Baseline Metrics (v12.8, pre v13 분할)

각 scenario subprocess 격리 실행. Phase 1+ 후 재실행 → 동일값 = no regression.

| Scenario | mode | x_final/target | y_final | z_range | vx mean/tgt | roll/pitch max | τ peak | Fz peak max | diverged |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| NMPC trot | _USE_NMPC_ACTIVE=True | 1.581 / 2.000 | +8 mm | 465~526 mm | 0.821 / 1.00 | 1.63° / 1.63° | 62.1 Nm | 502 N | ✅ |
| NMPC walk | _USE_NMPC_ACTIVE=True | 1.034 / 1.600 | -8 mm | 435~485 mm | 0.258 / 0.40 | 2.41° / 7.11° | 41.7 Nm | 340 N | ✅ |
| v11 trot | _USE_NMPC_ACTIVE=False | 1.987 / 2.000 | +4 mm | 462~470 mm | 0.994 / 1.00 | 0.52° / 0.21° | 98.0 Nm | 530 N | ✅ |
| v11 walk | _USE_NMPC_ACTIVE=False | 1.575 / 1.600 | +9 mm | 456~470 mm | 0.394 / 0.40 | 0.94° / 0.82° | 86.5 Nm | 427 N | ✅ |
