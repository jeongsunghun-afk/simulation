"""
DH → Pinocchio Model 직접 빌드 (URDF 우회).

Standard DH transform: T_i = Rz(θ_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)

Pinocchio joint model:
    joint i 의 placement P_i = parent joint의 post-rotation frame → joint i pre-rotation frame
    joint i의 rotation = R_axis(θ_i)
    그 후 link i body가 부착됨

표준 DH 매핑:
    P_1 = identity   (joint 1은 leg base에서 시작, axis는 leg base z축)
    P_i = M_fixed_{i-1} = Tz(d_{i-1}) Tx(a_{i-1}) Rx(α_{i-1})   for i ≥ 2

마지막 link 5의 foot tip은 추가 fixed transform = M_fixed_5 = Tz(d_5) Tx(a_5) Rx(α_5).
이는 frame으로 추가.
"""
import math
import numpy as np
import pinocchio as pin

# v11 파라미터 (gait_sim_v11.py와 일치)
BODY_FWD_F =  0.250
BODY_FWD_H = -0.250
BODY_LAT   =  0.050
BODY_Z_H   = -0.050
_HIP_Y_BIAS = 0.0075

LEG_HIP_OFFSETS = [
    (+BODY_FWD_F, -BODY_LAT + _HIP_Y_BIAS, 0.0     ),  # FR
    (+BODY_FWD_F, +BODY_LAT + _HIP_Y_BIAS, 0.0     ),  # FL
    (+BODY_FWD_H, -BODY_LAT + _HIP_Y_BIAS, BODY_Z_H),  # HR
    (+BODY_FWD_H, +BODY_LAT + _HIP_Y_BIAS, BODY_Z_H),  # HL
]
LEG_NAMES = ['FR', 'FL', 'HR', 'HL']

DH_FRONT = [
    (+math.pi/2, 0.0,   0.0   ),
    (0.0,        0.21,  0.0075),
    (0.0,        0.235, 0.0   ),
    (0.0,        0.1,   0.0   ),
    (0.0,        0.045, 0.0   ),
]
DH_HIND = [
    (-math.pi/2, 0.0,   0.0   ),
    (0.0,        0.21,  0.0075),
    (0.0,        0.21,  0.0   ),
    (0.0,        0.148, 0.0   ),
    (0.0,        0.045, 0.0   ),
]
LEG_DH = [DH_FRONT, DH_FRONT, DH_HIND, DH_HIND]

BODY_MASS = 15.0
BODY_INERTIA_DIAG = (0.07, 0.26, 0.26)
LINK_MASS = [3.0, 2.0, 1.0, 0.2, 0.1]
LINK_RADIUS = 0.015


def _dh_fixed_se3(alpha, a, d):
    """DH fixed transform Tz(d) · Tx(a) · Rx(α) as SE3."""
    cosa, sina = math.cos(alpha), math.sin(alpha)
    R = np.array([
        [1.0,    0.0,    0.0],
        [0.0,   cosa,  -sina],
        [0.0,   sina,   cosa],
    ])
    t = np.array([a, 0.0, d])
    return pin.SE3(R, t)


def _cyl_inertia_pin(mass, length, radius):
    """원통 (축 = local x) 관성 → pin.Inertia."""
    # 원통 축이 x인 경우: ixx=(1/2)mr², iyy=izz=(1/12)m(3r²+L²)
    ixx = 0.5 * mass * radius**2
    iyy = (1.0/12.0) * mass * (3*radius**2 + length**2)
    izz = iyy
    com = np.array([length/2.0, 0.0, 0.0])
    I = np.diag([ixx, iyy, izz])
    return pin.Inertia(mass, com, I)


def _box_inertia_pin(mass, dx, dy, dz):
    ixx = (mass / 12.0) * (dy**2 + dz**2)
    iyy = (mass / 12.0) * (dx**2 + dz**2)
    izz = (mass / 12.0) * (dx**2 + dy**2)
    com = np.zeros(3)
    return pin.Inertia(mass, com, np.diag([ixx, iyy, izz]))


def build_model():
    model = pin.Model()
    model.name = "quadruped_v11_dh"

    # Floating base (universe → root_joint)
    root_jid = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root_joint")
    body_inertia = pin.Inertia(BODY_MASS, np.zeros(3), np.diag(BODY_INERTIA_DIAG))
    model.appendBodyToJoint(root_jid, body_inertia, pin.SE3.Identity())
    # base 프레임 추가 (parent frame = universe = 0)
    base_frame_id = model.addBodyFrame("base_link", root_jid, pin.SE3.Identity(), 0)

    # 각 다리 추가
    for leg_idx, (name, dh) in enumerate(zip(LEG_NAMES, LEG_DH)):
        hip = LEG_HIP_OFFSETS[leg_idx]
        # leg_base frame: 부착 지점, fixed
        # joint 1은 leg_base의 z축에 대해 rotate
        # P_1 = leg attachment translation (no rotation, axis = parent z)
        leg_attach = pin.SE3(np.eye(3), np.array(hip))

        parent_jid = root_jid
        prev_dh_fixed = leg_attach  # joint i의 placement = 이전 fixed (i-1)

        for i, (alpha, a, d) in enumerate(dh, start=1):
            # joint i: placement = prev_dh_fixed (이전 joint의 post-rotation fixed)
            jid = model.addJoint(parent_jid, pin.JointModelRZ(),
                                 prev_dh_fixed,
                                 f"leg_{name}_j{i}")

            # link i: body inertia (cylinder along x with length max(a, d))
            L = max(a, d, 1e-6)
            inertia_i = _cyl_inertia_pin(LINK_MASS[i-1], L, LINK_RADIUS)
            # body_placement: link i의 frame = joint i의 post-rotation frame
            # COM은 link 중간 (a/2 along x)
            model.appendBodyToJoint(jid, inertia_i, pin.SE3.Identity())
            link_frame_id = model.addBodyFrame(f"leg_{name}_l{i}", jid, pin.SE3.Identity(), base_frame_id)

            # 다음 joint를 위한 fixed transform 준비 (이번 joint의 post-rotation 부분)
            prev_dh_fixed = _dh_fixed_se3(alpha, a, d)
            parent_jid = jid

        # foot tip frame (last joint's post-rotation + DH last fixed)
        last_alpha, last_a, last_d = dh[-1]
        foot_frame_se3 = _dh_fixed_se3(last_alpha, last_a, last_d)
        model.addBodyFrame(f"leg_{name}_foot", parent_jid, foot_frame_se3, base_frame_id)

    return model


if __name__ == '__main__':
    model = build_model()
    data = model.createData()
    print(f'Model: {model.name}')
    print(f'  njoints = {model.njoints}')
    print(f'  nq = {model.nq}, nv = {model.nv}')
    print(f'  joint names: {[model.names[i] for i in range(model.njoints)]}')
    q0 = pin.neutral(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    com = pin.centerOfMass(model, data, q0)
    M  = pin.computeTotalMass(model)
    print(f'  total mass = {M:.3f} kg')
    print(f'  CoM @ neutral = {com}')

    # FR foot 위치 검증
    for name in LEG_NAMES:
        foot_id = model.getFrameId(f'leg_{name}_foot')
        base_id = model.getFrameId(f'leg_{name}_l1')   # joint 1 frame
        oMf = data.oMf[foot_id]
        oMb = data.oMf[base_id]
        # Compare: foot world position
        print(f'  {name} foot world @ neutral q: {oMf.translation}')
