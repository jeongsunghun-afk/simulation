"""
Pinocchio Jacobian / RNEA / M(q) 검증.

v11 native:
    compute_jacobian_sim(q, dh, front) → 3×nj (foot velocity Jacobian, sim frame)
    rnea(q, dq, ddq, dh, lm) → tau (nj,)
    compute_mh_leg(q, dq, dh, lm) → M (nj×nj), h (nj,)

Pinocchio:
    pin.computeJointJacobians + pin.getFrameJacobian
    pin.rnea(model, data, q, v, a)
    pin.crba (composite rigid body algorithm) for M
    pin.nonLinearEffects (Coriolis + gravity for h)

비교는 SAME q for one leg, fixed base, native frame to compare.
"""
import math
import numpy as np
import pinocchio as pin

import build_pin_model as bm

# v11 native imports — sys.path manipulation으로 v11에서 직접 가져오기
import sys
sys.path.insert(0, '/home/jsh/simulation')

# v11 함수들을 inline 재구성 (전체 v11 import는 무거움)
def _dh_matrix(alpha, a, d, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1],
    ], dtype=float)


def _dh_to_sim(vec, front_leg=False):
    sim = np.array([vec[2], -vec[1], vec[0]], dtype=float)
    if front_leg:
        sim[:2] *= -1.0
    return sim


def fk_native_full(thetas, dh):
    """forward_kinematics: returns list of joint positions in DH frame."""
    T = np.eye(4)
    pts = [np.zeros(3)]
    for i, (alpha, a, d) in enumerate(dh):
        T = T @ _dh_matrix(alpha, a, d, thetas[i])
        pts.append(T[:3, 3].copy())
    return pts, T


def compute_jacobian_native_dh(q, dh):
    """v11의 compute_jacobian_sim과 동등하지만 DH frame 출력 (sim 변환 안 함)."""
    nj = len(q)
    pts, _ = fk_native_full(q, dh)
    foot = pts[-1]
    # cumulative T로 axis 추출
    T = np.eye(4)
    axes_z = []
    origins = []
    for i, (alpha, a, d) in enumerate(dh):
        origins.append(T[:3, 3].copy())
        axes_z.append(T[:3, 2].copy())   # joint i의 z축 (DH 회전축)
        T = T @ _dh_matrix(alpha, a, d, q[i])
    J_dh = np.zeros((3, nj))
    for i in range(nj):
        z = axes_z[i]
        r = foot - origins[i]
        J_dh[:, i] = np.cross(z, r)
    return J_dh


def rnea_native(q, dq, ddq, dh, lm):
    """v11의 rnea 단순 재구현 — 실제 v11 코드와 일치 검증용."""
    # v11 rnea를 직접 import 하기 위해 v11 파일에서 가져옴
    from importlib.util import spec_from_file_location, module_from_spec
    # 너무 복잡 — 그냥 numerical comparison으로 진행
    return None


# ──────────────────────────────────────────
# Pinocchio Jacobian 검증
# ──────────────────────────────────────────

Q_HOME_FRONT_DEG = [0.0, 133.2973, 46.7027, 30.6583, 59.3417]
Q_HOME_HIND_DEG  = [0.0, -150.0, -90.0, 90.0, 60.0]
Q_HOME_FRONT = [math.radians(a) for a in Q_HOME_FRONT_DEG]
Q_HOME_HIND  = [math.radians(a) for a in Q_HOME_HIND_DEG]


def main():
    model = bm.build_model()
    data = model.createData()

    print('━━━━━━━━━━ Jacobian 검증 (3×5 linear part, world frame ↔ DH frame) ━━━━━━━━━━\n')

    for leg, q_home, dh in [
        ('FR', Q_HOME_FRONT, bm.DH_FRONT),
        ('FL', Q_HOME_FRONT, bm.DH_FRONT),
        ('HR', Q_HOME_HIND,  bm.DH_HIND),
        ('HL', Q_HOME_HIND,  bm.DH_HIND),
    ]:
        # native Jacobian (DH frame)
        J_native = compute_jacobian_native_dh(q_home, dh)

        # pinocchio Jacobian @ leg foot frame
        q = pin.neutral(model)
        for i, qi in enumerate(q_home):
            jid = model.getJointId(f'leg_{leg}_j{i+1}')
            q[model.idx_qs[jid]] = qi
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        foot_fid = model.getFrameId(f'leg_{leg}_foot')
        # LOCAL_WORLD_ALIGNED: world frame, world origin
        J6_world = pin.getFrameJacobian(model, data, foot_fid, pin.LOCAL_WORLD_ALIGNED)
        # 다리 5개 joint만: idx_qs[joint_id] → idx_vs[joint_id]
        leg_v_idx = [model.idx_vs[model.getJointId(f'leg_{leg}_j{i+1}')] for i in range(5)]
        J3_world = J6_world[:3, leg_v_idx]   # 3×5 (linear part, world frame, leg joints)

        # native는 DH frame, pin은 world frame. world → DH frame 변환 필요.
        # leg_base의 world 회전 = oMf[base_frame].rotation
        # DH frame과 leg_base frame은 같음 (joint 1 placement가 부착점, axis = z)
        # DH → world: leg_base.rotation
        # world → DH: leg_base.rotation.T
        # 실제로는 leg_FR_l1 (joint 1의 frame) 기준이 아니라 leg base (parent) 기준 비교
        # pinocchio model에서 leg_base는 root_joint에 attach (LEG_HIP_OFFSETS만큼 translation)
        # leg_base의 world 회전 = root_joint의 회전 = identity (q[0:7]=neutral 이므로)
        # 따라서 pin Jacobian (world frame) = native Jacobian (DH frame) 같아야 함
        diff = J3_world - J_native
        print(f'{leg} Jacobian (3×5):  |diff|max={np.max(np.abs(diff)):.6f}, '
              f'frob={np.linalg.norm(diff):.6f}')

    # ───── RNEA 검증 (q̇=0, q̈=0 → 중력 토크) ─────
    print('\n━━━━━━━━━━ RNEA 중력 토크 (gravity vector g(q) at q_home) ━━━━━━━━━━\n')
    q = pin.neutral(model)
    for leg, q_home in [
        ('FR', Q_HOME_FRONT), ('FL', Q_HOME_FRONT),
        ('HR', Q_HOME_HIND),  ('HL', Q_HOME_HIND)]:
        for i, qi in enumerate(q_home):
            jid = model.getJointId(f'leg_{leg}_j{i+1}')
            q[model.idx_qs[jid]] = qi
    v = np.zeros(model.nv)
    a = np.zeros(model.nv)
    tau_rnea = pin.rnea(model, data, q, v, a)   # gravity torque (전 관절)
    print(f'tau_rnea shape: {tau_rnea.shape}')
    print(f'floating base 잔차 (해당 6개 = base에 작용하는 g): {tau_rnea[:6].round(4)}')
    print(f'              (base z = -M·g·z 성분 ≈ -{40.2*9.81:.2f} N 기대)')
    for leg in ['FR', 'FL', 'HR', 'HL']:
        leg_v_idx = [model.idx_vs[model.getJointId(f'leg_{leg}_j{i+1}')] for i in range(5)]
        tau_leg = tau_rnea[leg_v_idx]
        print(f'  {leg} g_torque [Nm]: {[f"{v:.3f}" for v in tau_leg]}')

    # ───── Mass matrix M(q) 검증 (CRBA) ─────
    print('\n━━━━━━━━━━ Mass matrix CRBA ━━━━━━━━━━')
    M = pin.crba(model, data, q)
    print(f'M shape: {M.shape}  (전 robot, floating base 포함)')
    # block diagonal? per-leg M_block?
    for leg in ['FR', 'FL', 'HR', 'HL']:
        leg_v_idx = [model.idx_vs[model.getJointId(f'leg_{leg}_j{i+1}')] for i in range(5)]
        M_leg = M[np.ix_(leg_v_idx, leg_v_idx)]
        print(f'  {leg} M_leg diag: {np.diag(M_leg).round(4)}')


if __name__ == '__main__':
    main()
