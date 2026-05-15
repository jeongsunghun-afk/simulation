# NMPC Swing Foot Tracking — 가중치 Sweep 리포트 (v13.24)

## 배경

fig6(Foot Trajectory Cmd vs Actual)에서 NMPC 모드의 swing foot 추종이 불량.
특히 발 z가 cmd 0.08m 대비 peak ~0.05m로 under-shoot. 어떤 NMPC 가중치를
건드려야 하는지 확인하기 위한 sweep.

- 조건: NMPC (crocoddyl FDDP receding), v=1.0 m/s, T=0.5s, trot, 4 cycle
- 평가: steady-state 구간만 (cycle 2~4, 첫 사이클 startup 제외)
- 지표: `[5] 제어 정밀도`의 swing foot tracking RMS (축별 x/y/z 분해 추가)

## Sweep 결과

| config | foot x | foot y | **foot z** | foot agg | body x err | roll/pitch max | τ포화 | fails |
|---|---|---|---|---|---|---|---|---|
| `w_track_z=1000`            | 172.7 | 48.0 | **31.9** | 182.0 | —     | —          | —    | 0 |
| `w_track_z=3000`            | 178.2 | 49.0 | **30.6** | 187.4 | 137.9 | 1.12/1.50° | 0.71 | 0 |
| **baseline** (z=10k,xy=100) | 190.0 | 51.0 | **29.4** | 198.9 | 151.1 | 1.24/1.49° | 0.76 | 0 |
| `w_track_z=30000`           | 206.8 | 54.3 | **28.7** | 215.8 | —     | —          | —    | 0 |
| `w_track_z=100000`          | 226.8 | 66.1 | **30.6** | 238.2 | —     | —          | —    | 0 |
| `w_track_xy=500`            | 139.9 | 33.4 | **30.2** | 147.0 | 126.6 | 2.28/2.03° | 0.81 | 0 |
| `w_track_xy=2000`           | 114.2 | 21.9 | **29.5** | 120.0 |  95.5 | 3.35/3.35° | 0.94 | 0 |
| `w_touchdown_v=0`           | 184.3 | 49.7 | **28.7** | 193.0 | —     | —          | —    | 0 |
| `w_touchdown_v=1`           | —     | —    | —        | 193.8 | —     | —          | —    | 0 |
| **`xy=2000 + z=3000` (v13.25 선택)** | 113.1 | 20.1 | **30.7** | 118.9 |  86.4 | 3.03/4.23° | 0.92 | 0 |
| `xy=500 + z=3000` (균형 — env override) | 135.2 | 32.2 | **31.3** | 142.4 | 114.3 | 1.96/1.95° | 0.76 | 0 |

*단위: 위치 RMS [mm], 각도 [°]. foot agg = 3D norm aggregate.*

## 핵심 발견

### 1. 발 z under-shoot는 구조적 — 가중치로 못 고침

`nmpc_w_track_z`를 **1000 → 100000 (100배 범위)** 휘저어도 발 z 오차는
**28.7 ~ 31.9mm 안에서만** 움직임. z weight와 z 추종 사이에 사실상 상관 없음.

→ z under-shoot(peak ~0.05 vs cmd 0.08m)는 NMPC 가중치 무관한 **구조적 한계**.
추정 원인:
- body lag(~150mm)가 다리 DOF 예산을 x 추종에 소진 → z 들 여유 없음
- swing 시간(T_SW=0.25s)이 80mm 들었다 내리기엔 짧음
- joint torque / 기구학적 한계

고치려면 가중치가 아니라 **구조 변경** 필요: swing 시간↑, step height↓,
또는 body lag 자체 해결([B2 body track] 참고).

### 2. `w_track_z`↑는 순손해

z weight를 올리면 z는 안 좋아지면서 **x만 악화** (190→227mm).
optimizer effort를 z로 끌어가는데 z는 구조적으로 막혀 있어 x만 손해.
기존 기본값 `10000`은 과도값.

### 3. 실제 lever는 `w_track_xy` — 단 posture tradeoff

`w_track_xy` 100→2000: 발·몸통 x·y 추종을 단조 개선하나 roll/pitch도
1.2°/1.5° → 3.3°/3.3°로 악화. 강한 발 추종이 다리를 공격적 자세로 몰아
몸통이 더 흔들림.

### 4. `w_touchdown_v`는 영향 없음

10 → 1 → 0: foot agg 198.9 → 193.8 → 193.0. 무시 가능.

## 기본값 변경 (v13.24 → v13.25)

| 파라미터 | 기존 | 변경 | 근거 |
|---|---|---|---|
| `nmpc_w_track_z`  | 10000 | **3000** | z 오차 동일, x·posture·τ 모두 소폭 개선 (free win) |
| `nmpc_w_track_xy` | 100   | **2000** | 추종 최우선 — foot/body 추종 가장 개선 (v13.25 채택) |
| `nmpc_w_touchdown_v` | 1e1 | 유지 | 영향 없음 |

**효과 (`xy=2000 + z=3000`)**: swing foot RMS 199→119mm (−40%),
body x err 151→86mm (−43%), τ포화 0.92(+0.16), roll/pitch 3.03°/4.23°(+1.8°/+2.7°), 0 fails.

자세 안정성 vs 추종 정확도 tradeoff에서 **추종 정확도 우선** 채택.
posture 우선 시 `NMPC_W_TRACK_XY=500` env override → 균형(roll/pitch ~2°, foot agg 142mm).
원래값 동작 필요 시 `NMPC_W_TRACK_XY=100`.

## 인프라 추가

- env override: `NMPC_W_TRACK_XY`, `NMPC_W_TRACK_Z`, `NMPC_W_TOUCHDOWN_V`
- `[5] 제어 정밀도` 로그에 swing foot tracking 축별(x/y/z) RMS 분해 출력

## 남은 과제

- 발 z under-shoot: 구조적 — swing 궤적/시간 재설계 또는 body lag 해결 필요
- 발 x 오차: body lag(`nmpc_w_body_track`)에 종속 — 별도 트랙
