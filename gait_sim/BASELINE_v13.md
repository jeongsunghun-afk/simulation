# Baseline Metrics (v12.8, pre v13 분할)

각 scenario subprocess 격리 실행. Phase 1+ 후 재실행 → 동일값 = no regression.

| Scenario | mode | x_final/target | y_final | z_range | vx mean/tgt | roll/pitch max | τ peak | Fz peak max | diverged |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| NMPC trot | _USE_NMPC_ACTIVE=False | 1.988 / 2.000 | +6 mm | 462~470 mm | 0.994 / 1.00 | 0.44° / 0.18° | 98.0 Nm | 528 N | ✅ |
| NMPC walk | _USE_NMPC_ACTIVE=False | 1.573 / 1.600 | +10 mm | 456~470 mm | 0.393 / 0.40 | 1.09° / 1.09° | 86.5 Nm | 415 N | ✅ |
| v11 trot | _USE_NMPC_ACTIVE=False | 1.988 / 2.000 | +6 mm | 462~470 mm | 0.994 / 1.00 | 0.44° / 0.18° | 98.0 Nm | 528 N | ✅ |
| v11 walk | _USE_NMPC_ACTIVE=False | 1.573 / 1.600 | +10 mm | 456~470 mm | 0.393 / 0.40 | 1.09° / 1.09° | 86.5 Nm | 415 N | ✅ |
