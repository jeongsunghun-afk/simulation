# biped — 2-leg robot (HL/HR) MJCF wrapper

CM_HL 의 뒷다리 2개 (HL + HR) + base 로 구성된 biped 의 MuJoCo 시뮬레이션
asset. 원본 URDF (SolidWorks export) 를 wrapper MJCF 로 감싸 freejoint base + floor + actuator 추가.

## 디렉토리

```
biped/
├── urdf/biped.urdf            # 원본 URDF (mesh path '../meshes/' 로 변환)
├── meshes/*.STL               # 11개 binary STL (thigh 2개는 100k face 로 decimate)
├── biped_raw.mjcf             # URDF→MJCF 자동 변환 결과
├── biped_wrapper.mjcf         # 사용 mjcf — freejoint base + floor + 8 motor
└── README.md
```

## 구조

- 8 actuated joint (HL/HR × 4): hip_roll → thigh_pitch → calf_pitch → foot_pitch
- joint axis 통일: hip = (1,0,0), thigh/calf/foot = (0,1,0)
- 좌우 mirror: base→hip y offset (HL=+0.0225, HR=-0.0225), hip→thigh y offset (HL=+0.115, HR=-0.115)
- foot collision = box (90×25×8mm) at foot_contact_link 위치
- total mass ≈ 7.5 kg

## URDF angle convention

- URDF `q = 0` (모두 0) = robot 의 **home/standing 자세** (사용자 정의)
- robot encoder/controller angle: q_robot = q_urdf + θ_home
  - θ_home (deg) = [0, 160, 50, -50] for [hip_roll, hip_pitch, knee, ankle]
- joint range (URDF rad):
  - hip_roll: ±0.611
  - hip_pitch: -1.396 ~ +2.094
  - knee: -0.524 ~ +1.484
  - ankle: -1.571 ~ +0.524

## DH 파라미터 (reference, base만 공유)

| i | α_i  | a_i   | d_i   | θ_home (deg) | joint            |
|---|------|-------|-------|--------------|------------------|
| 1 | π/2  | 0.000 | 0.115 | 0            | HL_hip_joint     |
| 2 | 0    | 0.230 | 0.000 | 160          | HL_thigh_joint   |
| 3 | 0    | 0.245 | 0.000 | 50           | HL_calf_joint    |
| 4 | 0    | 0.180 | 0.000 | -50          | HL_foot_joint    |
| 5 | 0    | 0.000 | 0.000 | -160         | HL_foot_contact  |

Base→Leg transform:
- `{Base}→{HL}`: translate(-0.225, +0.0225, 0) · Rz(180°) · Ry(-90°)
- `{Base}→{HR}`: translate(-0.225, -0.0225, 0) · Rz(180°) · Ry(-90°)

**주의**: DH frame 들과 URDF frame 들은 base 만 공유, 나머지 좌표계 다름.
WBIC controller 는 mujoco mj_jac 직접 사용 (URDF 100% 일치 보장). DH 는
robot↔URDF angle 변환 + reference 로만.

## Motor spec

| Joint | URDF rated (Nm) | max (Nm, ×3) |
|-------|-----------------|--------------|
| hip_roll  | 28  | 84  |
| hip_pitch | 28  | 84  |
| knee      | 42  | 126 |
| ankle     | 56  | 168 |

## 빠른 검증 (mujoco viewer)

WSL 의 OpenGL 한계로 Windows native 권장:

```bat
:: C:\Users\jsh\run_biped_viewer.bat 더블클릭
:: → URDF zero pose (robot home) 시각 확인
```

WSL 직접 (WSLg, 검은 화면 가능):
```bash
cd /home/jsh/simulation/biped && python3 -c "
import mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path('biped_wrapper.mjcf')
d = mujoco.MjData(m); d.qpos[2] = 0.55; d.qpos[3] = 1.0
mujoco.viewer.launch(m, d)"
```

## ROS2 controller 연동

`/home/jsh/ros2_ws/src/biped_sim/` 패키지 참조.
