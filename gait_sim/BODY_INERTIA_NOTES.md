# BODY_INERTIA — 역할, 발산 버그, CRBA 측정, CAD 워크플로우

## 1. BODY_INERTIA 의 역할

`BODY_INERTIA` 는 **MPC body 6-DoF + WBIC FB QP 의 reduced-order body 회전 관성 텐서**.

- gait_sim 은 robot 을 `single rigid floating-base + 4 leg` 로 모델링
- MPC body 6-DoF QP 가 body 를 하나의 rigid body 로 보고 angular dynamics 계산
- 이때 사용하는 회전 관성 = `BODY_INERTIA` (3×3 tensor)

**중요**: `BODY_INERTIA` 는 "body 단독" 이 아니라 **body + 4 다리가 home pose 에 붙어있을 때의 composite (합성) 관성**.

## 2. 발산 버그 (v13.14 에서 수정)

### 증상
v12/v13 MPC+WBIC trot 4-cycle 에서 body z 발산:
- cycle 1~2 안정 (~465mm hover)
- cycle 3+ 폭주 (1.4m → 3.7m)
- Fz peak 4498N, z drift +7334mm

### Root cause
`GaitConfig.body_inertia` default = `diag(0.07, 0.26, 0.26)` = **base-link (chassis box) only**.
- 4개 다리 mass 기여분 누락 → 회전 관성 ~10× 과소평가
- MPC 가 body 를 "다리 없는 가벼운 몸통" 으로 오인
- angular dynamics under-damped → roll/pitch oscillation 누적 → 3+ cycle 발산

### v12.7 의 의도된 fix 가 작동 안 한 이유
v12.7 commit (5635a78) 이 CRBA composite upgrade 를 추가했으나, 조건이
`if _CROCODDYL_AVAILABLE:` 로 묶여 있어 crocoddyl import 실패 시
(`liburdfdom_sensor.so.4.0` 결핍) **영구 skip** → base-link only 로 동작.

## 3. v13.14 Fix

`body_inertia` default → **CRBA composite hardcode**:

```python
body_inertia = np.array([
    [ 0.8044,  0.0,    -0.2547],
    [ 0.0,     2.1571,  0.0   ],
    [-0.2547,  0.0,     1.5599],
])
```

- crocoddyl/pinocchio 의존성 없이 어떤 환경에서도 안정
- 적용: `gait_sim/config.py`, `gait_sim_v11.py`, `gait_sim_v12.py`, `gait_sim_v13.py`
- `pin_model.py` double-counting 버그도 수정 (이미-composite 값을 base link 으로 쓰면 2번 누적)

### 검증 (v13.py 4-cycle MPC+WBIC trot)
| metric | BEFORE (base-link) | AFTER (composite) |
|---|---:|---:|
| z drift | +7334 mm 🚨 | +4.7 mm ✅ |
| Fz peak | 4498 N 🚨 | 528 N ✅ |
| z range | 462~3724 mm | 462~470 mm ✅ |

## 4. CRBA 측정 방법

**CRBA (Composite Rigid Body Algorithm)** 는 **pinocchio** 의 알고리즘 (`pin.crba()`).
crocoddyl 은 CRBA 를 제공하지 않음 — crocoddyl 은 pinocchio 를 backend 로 쓰는 OCP solver.

```python
import pinocchio as pin
from gait_sim.pin_model import build_model     # v13 DH-mirror model
from gait_sim.model import Q_HOME_FRONT, Q_HOME_HIND

m = build_model()
d = m.createData()
q0 = pin.neutral(m)
# leg joints 를 home pose 로 set
for leg, qh in {'FR':Q_HOME_FRONT,'FL':Q_HOME_FRONT,'HR':Q_HOME_HIND,'HL':Q_HOME_HIND}.items():
    for i, qi in enumerate(qh):
        q0[m.idx_qs[m.getJointId(f'leg_{leg}_j{i+1}')]] = qi

M = pin.crba(m, d, q0)        # joint-space mass matrix
BODY_INERTIA = M[3:6, 3:6]    # floating base 의 angular block = composite 회전 관성
```

- `M[3:6, 3:6]` = floating base 의 angular block = 다리 포함 전체 회전 관성
- **home pose 1회 측정 → 고정값 (frozen)**. 다리가 움직이면 실제 composite 는 시시각각 변하지만,
  MPC body 6-DoF model 은 reduced-order rigid 가정이라 정적 근사로 충분.
- time-varying CRBA (매 frame 재계산) 는 과도한 정밀도 — 불필요.

## 5. 동역학 역할 분담

| 항목 | 계산 방법 | 사용 파라미터 |
|---|---|---|
| **per-leg 관절 동역학** | RNEA 매 frame 정확 계산 (`dynamics.py`) | `LINK_MASS [3,2,1,0.2,0.1]` |
| **body 6-DoF 회전 관성** (MPC/WBIC) | CRBA 1회 측정 → hardcode | `body_inertia` composite |

→ 다리 link mass 자체는 RNEA 가 매 frame 정확히 사용. CRBA hardcode 는
   body 6-DoF reduced model 의 회전 관성만 담당.

## 6. CAD 모델 도입 워크플로우 (v14.6)

현재 v13.14 의 hardcode `[0.804, 2.157, 1.560]` 은 **cylinder 근사 link inertia 기반 임시값**.
CAD 모델 도착 시 정확값으로 교체:

```
CAD 모델 (SolidWorks / Fusion 360)
  → 각 link 의 정확 mass / CoM / inertia tensor 추출 (CAD 소프트웨어 자동 계산)
  → URDF <inertial> tag 에 반영
  → pinocchio.buildModelFromUrdf() 로 load
  → pin.crba(model, data, q_home) → M[3:6,3:6]
  → GaitConfig.body_inertia 갱신 (현재 hardcode 대체)
  → 동시에 LINK_MASS / LINK_RADIUS 도 CAD 정확값으로 갱신
  → baseline 전체 재측정
```

### Isaac Sim 관점

- **Isaac Sim 자체**: 각 link 가 개별 rigid body 로 simulate → PhysX 가 composite inertia 자동 정확 계산.
  "BODY_INERTIA 발산" 문제 구조적으로 발생 안 함.
- **하지만 controller bridge 시**: MPC+WBIC controller 내부는 여전히 `BODY_INERTIA` reduced model 사용.
  → controller 의 `BODY_INERTIA` 가 정확해야 올바른 GRF/torque 명령 생성.
- **sim2sim validation 신뢰성**: `BODY_INERTIA` 부정확하면 sim2sim gap 이
  controller bug 인지 물리 차이인지 구분 불가 → fix 필수.

→ **결론**: CAD 도착 전까지는 v13.14 hardcode 로 안정 동작 보장. CAD 도착 후 v14.6 에서 정확값 교체.

---

## 7. crocoddyl / pinocchio 환경 setup (NMPC + CRBA upgrade 용)

NMPC (crocoddyl FDDP) 와 BODY_INERTIA 자동 CRBA upgrade 는 pinocchio + crocoddyl 필요.

### 올바른 설치

```bash
pip install --user pin       # ⚠ 패키지명 'pin' (NOT 'pinocchio' — 그건 다른 CLI tool)
                             #   → pin-3.9.0 + cmeel-urdfdom-4.0.1 + eigenpy-3.12.0 + numpy 2.x
# crocoddyl 은 이미 system pip 에 (crocoddyl 3.2.0 + libcrocoddyl 3.2.0 + libpinocchio 3.9.0)
```

### 주의사항

- **`pip install pinocchio` 는 잘못된 패키지** (`pinocchio-0.4.3` = 무관한 CLI tool). 반드시 `pin`.
- `pin` 이 `cmeel-urdfdom-4.0.1` 을 끌어옴 → `liburdfdom_sensor.so.4.0` 제공
  (crocoddyl 3.2.0 이 urdfdom 4.0 요구. urdfdom 3.x 만 있으면 crocoddyl import 실패)
- `pin` 설치 시 numpy 가 2.x 로 올라감 — matplotlib 3.10 / scipy 1.15 / gait_sim 모두 호환 확인됨.

### 검증

```bash
python3 -c "import pinocchio; print(pinocchio.__version__)"   # 3.9.0
python3 -c "import crocoddyl; print('OK')"                     # OK
```

→ crocoddyl import 성공 시 `_CROCODDYL_AVAILABLE=True` → NMPC 활성 + BODY_INERTIA CRBA 자동 upgrade.
   (단 v13.14 hardcode default 가 이미 안전값이라 crocoddyl 없어도 발산 안 함.)
