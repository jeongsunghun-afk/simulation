"""
Pinocchio FK가 v11 native forward_kinematics와 일치하는지 검증.

v11 native FK는 DH frame에서 직접 계산.
Pinocchio model: build_pin_model.py에서 생성. joint i의 post-rotation frame은
DH의 frame i와 동일해야 함.

비교 방법:
    1. q_home으로 두 모델의 foot tip을 leg base 기준으로 계산
    2. xyz 차이를 mm로 출력 (0이어야 정합)
"""
import math
import numpy as np
import pinocchio as pin

import build_pin_model as bm

Q_HOME_FRONT_DEG = [0.0, 133.2973, 46.7027, 30.6583, 59.3417]
Q_HOME_HIND_DEG  = [0.0, -150.0, -90.0, 90.0, 60.0]
Q_HOME_FRONT = [math.radians(a) for a in Q_HOME_FRONT_DEG]
Q_HOME_HIND  = [math.radians(a) for a in Q_HOME_HIND_DEG]


def _dh_matrix(alpha, a, d, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1],
    ], dtype=float)


def fk_native(thetas, dh):
    T = np.eye(4)
    for i, (alpha, a, d) in enumerate(dh):
        T = T @ _dh_matrix(alpha, a, d, thetas[i])
    return T


def main():
    model = bm.build_model()
    data = model.createData()

    # joint id 매핑
    print(f'joint 순서: {[model.names[i] for i in range(model.njoints)]}\n')

    def _dh_to_sim(vec, front_leg):
        sim = np.array([vec[2], -vec[1], vec[0]], dtype=float)
        if front_leg:
            sim[:2] *= -1.0
        return sim

    for leg, q_home, dh, hip, is_front in [
        ('FR', Q_HOME_FRONT, bm.DH_FRONT, bm.LEG_HIP_OFFSETS[0], True),
        ('FL', Q_HOME_FRONT, bm.DH_FRONT, bm.LEG_HIP_OFFSETS[1], True),
        ('HR', Q_HOME_HIND,  bm.DH_HIND,  bm.LEG_HIP_OFFSETS[2], False),
        ('HL', Q_HOME_HIND,  bm.DH_HIND,  bm.LEG_HIP_OFFSETS[3], False),
    ]:
        # native FK: leg_base 기준 foot tip → v11 sim frame으로 변환
        T_native = fk_native(q_home, dh)
        foot_native_dh = T_native[:3, 3]
        foot_native = _dh_to_sim(foot_native_dh, front_leg=is_front)   # v11 sim frame

        # pinocchio FK
        q = pin.neutral(model)   # base = identity, 모든 다리 0
        # 해당 다리 home 자세 설정
        for i, qi in enumerate(q_home):
            jid = model.getJointId(f'leg_{leg}_j{i+1}')
            q[model.idx_qs[jid]] = qi
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # foot tip world position
        fid_foot = model.getFrameId(f'leg_{leg}_foot')
        oMf = data.oMf[fid_foot]
        # leg_base는 root_joint에 부착, world에서 hip 위치
        # native와 비교하려면 hip offset 빼야함
        hip_arr = np.array(hip)
        foot_pin_relative = oMf.translation - hip_arr

        diff = foot_pin_relative - foot_native
        nf  = ', '.join(f'{v*1e3:+8.3f}' for v in foot_native)
        pf  = ', '.join(f'{v*1e3:+8.3f}' for v in foot_pin_relative)
        ok  = '✓' if np.linalg.norm(diff) < 1e-6 else '✗'
        print(f'  {leg}:  native=[{nf}]  pin_rel=[{pf}]  '
              f'|diff|={np.linalg.norm(diff)*1e3:.4f}mm  {ok}')


if __name__ == '__main__':
    main()
