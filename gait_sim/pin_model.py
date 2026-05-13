"""gait_sim.pin_model — DH → pinocchio Model 직접 빌드 (URDF 우회, v13 sync).

v13.y-a: 기존 /home/jsh/simulation/build_pin_model.py 의 v13 sync 버전.
        gait_sim.model 의 LEG_DH/LEG_HIP_OFFSETS 자동 참조 →
        v13 의 DH d2 좌우 mirror + hip_y_bias=0 정합.

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

# v13 sync: gait_sim.model 에서 직접 import → DH 4벌 (R/L mirror) 자동 적용
from gait_sim.model import (
    LEG_DH, LEG_HIP_OFFSETS, LEG_NAMES,
    BODY_MASS, BODY_INERTIA, LINK_MASS, LINK_RADIUS,
)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════
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


# Leg attach SE3: DH→sim frame 변환 흡수 (v11/v13 sim frame 정합)
# Front: sim_x=-dh_z, sim_y=dh_y, sim_z=dh_x   → R_front = [[0,0,-1],[0,1,0],[1,0,0]]
# Hind:  sim_x= dh_z, sim_y=-dh_y, sim_z=dh_x  → R_hind  = [[0,0, 1],[0,-1,0],[1,0,0]]
_R_LEG_ATTACH_FRONT = np.array([
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
    [1.0, 0.0,  0.0],
])
_R_LEG_ATTACH_HIND = np.array([
    [0.0, 0.0,  1.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0,  0.0],
])


def _cyl_inertia_pin(mass, length, radius):
    """원통 (축 = local x) 관성 → pin.Inertia."""
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


# ══════════════════════════════════════════════════════════════
# Model builder (v13 sync)
# ══════════════════════════════════════════════════════════════
def build_model() -> pin.Model:
    """v13 sync: gait_sim.model LEG_DH (4벌, R/L mirror) 기반 pinocchio Model 빌드.

    Returns:
        pin.Model — FreeFlyer base + 4 leg × 5 joint chain
                    Frame names: 'base_link', 'leg_FR_foot' (FR/FL/HR/HL),
                                 'leg_FR_l{1..5}' per leg
    """
    model = pin.Model()
    model.name = "quadruped_v13_dh"

    # Floating base (universe → root_joint)
    root_jid = model.addJoint(0, pin.JointModelFreeFlyer(),
                                pin.SE3.Identity(), "root_joint")
    body_inertia = pin.Inertia(BODY_MASS, np.zeros(3),
                                  np.array(BODY_INERTIA, copy=True))
    model.appendBodyToJoint(root_jid, body_inertia, pin.SE3.Identity())
    base_frame_id = model.addBodyFrame("base_link", root_jid,
                                          pin.SE3.Identity(), 0)

    # 각 다리 추가 (v13: LEG_DH[i] 가 R/L mirror 반영된 4벌)
    for leg_idx, name in enumerate(LEG_NAMES):
        dh = LEG_DH[leg_idx]
        hip = LEG_HIP_OFFSETS[leg_idx]
        is_front = (leg_idx < 2)
        R_attach = _R_LEG_ATTACH_FRONT if is_front else _R_LEG_ATTACH_HIND
        leg_attach = pin.SE3(R_attach, np.array(hip))

        parent_jid = root_jid
        prev_dh_fixed = leg_attach   # joint i 의 placement = 이전 fixed (i-1)

        for i, (alpha, a, d) in enumerate(dh, start=1):
            jid = model.addJoint(parent_jid, pin.JointModelRZ(),
                                   prev_dh_fixed,
                                   f"leg_{name}_j{i}")
            # link i: cylinder along local x (length = max(a, d))
            L = max(a, d, 1e-6)
            inertia_i = _cyl_inertia_pin(LINK_MASS[i-1], L, LINK_RADIUS)
            model.appendBodyToJoint(jid, inertia_i, pin.SE3.Identity())
            model.addBodyFrame(f"leg_{name}_l{i}", jid,
                                  pin.SE3.Identity(), base_frame_id)
            prev_dh_fixed = _dh_fixed_se3(alpha, a, d)
            parent_jid = jid

        # foot tip frame
        last_alpha, last_a, last_d = dh[-1]
        foot_frame_se3 = _dh_fixed_se3(last_alpha, last_a, last_d)
        model.addBodyFrame(f"leg_{name}_foot", parent_jid,
                              foot_frame_se3, base_frame_id)

    return model


# ══════════════════════════════════════════════════════════════
# URDF Export (v13.y-b)
# ══════════════════════════════════════════════════════════════
def export_urdf(model: pin.Model, urdf_path: str,
                  pkg_name: str = 'gait_sim_v13') -> None:
    """pinocchio Model → URDF XML 파일 저장.

    pinocchio 의 buildSampleModelHumanoid 와 같은 in-memory Model 은 직접 URDF 로
    serialize 어려움. 대신 본 함수는 DH parameters 를 직접 URDF format 으로 export.

    Args:
        model:      pinocchio Model (build_model() 결과)
        urdf_path:  output URDF 파일 경로
        pkg_name:   ROS package name for mesh reference (현재 mesh 없음 — 무시 가능)

    URDF schema:
        <robot name="...">
          <link name="base_link"> ... </link>
          <joint name="..."> ... <origin .../> <axis xyz="0 0 1"/> </joint>
          <link name="leg_FR_l1"> ... </link>
          ...
        </robot>
    """
    lines = ['<?xml version="1.0"?>',
             f'<robot name="{model.name}">',
             '']
    # base_link
    lines.append('  <link name="base_link">')
    lines.append(f'    <inertial>')
    lines.append(f'      <mass value="{BODY_MASS}"/>')
    Ib = np.array(BODY_INERTIA, copy=True)
    lines.append(f'      <inertia ixx="{Ib[0,0]:.6f}" iyy="{Ib[1,1]:.6f}" izz="{Ib[2,2]:.6f}"')
    lines.append(f'               ixy="{Ib[0,1]:.6f}" ixz="{Ib[0,2]:.6f}" iyz="{Ib[1,2]:.6f}"/>')
    lines.append('    </inertial>')
    lines.append('    <visual><geometry><box size="0.5 0.1 0.05"/></geometry></visual>')
    lines.append('  </link>')
    lines.append('')

    # Helper: SE3 → URDF origin (xyz + rpy)
    def _se3_to_origin(T: pin.SE3) -> str:
        from scipy.spatial.transform import Rotation as Rot
        rpy = Rot.from_matrix(T.rotation).as_euler('xyz', degrees=False)
        return (f'<origin xyz="{T.translation[0]:.6f} {T.translation[1]:.6f} '
                f'{T.translation[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>')

    # Walk through model joints (skip universe=0 and root_joint=1)
    for jid in range(2, model.njoints):
        jname = model.names[jid]
        parent_jid = model.parents[jid]
        parent_name = ('base_link' if parent_jid == 1
                        else model.names[parent_jid].replace('leg_', 'leg_').replace('_j', '_l'))
        # joint placement = jointPlacements[jid] (parent post-rotation → joint pre-rotation)
        T = model.jointPlacements[jid]
        # axis: pinocchio JointModelRZ → rotation about local z axis
        axis_xyz = "0 0 1"

        # corresponding link name: leg_{name}_l{i} matching joint leg_{name}_j{i}
        link_name = jname.replace('_j', '_l')
        # joint
        lines.append(f'  <joint name="{jname}" type="continuous">')
        lines.append(f'    <parent link="{parent_name}"/>')
        lines.append(f'    <child  link="{link_name}"/>')
        lines.append(f'    {_se3_to_origin(T)}')
        lines.append(f'    <axis xyz="{axis_xyz}"/>')
        lines.append(f'  </joint>')
        # link inertia (from model.inertias[jid])
        I_link = model.inertias[jid]
        mass = float(I_link.mass)
        com  = np.array(I_link.lever)
        Im   = np.array(I_link.inertia)
        lines.append(f'  <link name="{link_name}">')
        lines.append(f'    <inertial>')
        lines.append(f'      <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}"/>')
        lines.append(f'      <mass value="{mass}"/>')
        lines.append(f'      <inertia ixx="{Im[0,0]:.6f}" iyy="{Im[1,1]:.6f}" '
                      f'izz="{Im[2,2]:.6f}"')
        lines.append(f'               ixy="{Im[0,1]:.6f}" ixz="{Im[0,2]:.6f}" '
                      f'iyz="{Im[1,2]:.6f}"/>')
        lines.append('    </inertial>')
        lines.append('    <visual><geometry><cylinder length="0.1" radius="0.01"/>'
                      '</geometry></visual>')
        lines.append('  </link>')
        lines.append('')

    lines.append('</robot>')
    with open(urdf_path, 'w') as f:
        f.write('\n'.join(lines))


# ══════════════════════════════════════════════════════════════
# Diagnostic main
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    model = build_model()
    data = model.createData()
    print(f'Model: {model.name}')
    print(f'  njoints = {model.njoints}')
    print(f'  nq = {model.nq}, nv = {model.nv}')

    q0 = pin.neutral(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    com = pin.centerOfMass(model, data, q0)
    M  = pin.computeTotalMass(model)
    print(f'  total mass     = {M:.3f} kg')
    print(f'  CoM @ neutral  = {com}')

    for name in LEG_NAMES:
        foot_id = model.getFrameId(f'leg_{name}_foot')
        oMf = data.oMf[foot_id]
        print(f'  {name} foot world @ neutral: {oMf.translation}')

    print('\nv13 DH mirror 검증:')
    print('  expect: FR/FL y opposite, HR/HL y opposite (좌우 ±0.0575)')
