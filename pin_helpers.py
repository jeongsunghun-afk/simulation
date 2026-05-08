"""
Pinocchio-based per-leg dynamics helpers (v11 인터페이스 호환).

v11의 per-leg compute_mh_leg, rnea, compute_jacobian_sim를 pinocchio로 대체.
인터페이스는 동일하게 유지하여 USE_PINOCCHIO 토글로 비교 가능.

내부적으로 pinocchio model 한 번 빌드 후 재사용.
"""
import math
import numpy as np
import pinocchio as pin

import build_pin_model as _bm


# ── 모듈 전역: 모델 1회 빌드 ─────────────────────────────────
_MODEL = None
_DATA  = None
_LEG_V_IDX = None   # leg → joint v index list (length 5)
_LEG_Q_IDX = None
_LEG_FOOT_FID = None


def _ensure_model():
    global _MODEL, _DATA, _LEG_V_IDX, _LEG_Q_IDX, _LEG_FOOT_FID
    if _MODEL is not None:
        return
    _MODEL = _bm.build_model()
    _DATA  = _MODEL.createData()
    _LEG_V_IDX, _LEG_Q_IDX, _LEG_FOOT_FID = {}, {}, {}
    for leg in ['FR', 'FL', 'HR', 'HL']:
        _LEG_V_IDX[leg] = [_MODEL.idx_vs[_MODEL.getJointId(f'leg_{leg}_j{i+1}')] for i in range(5)]
        _LEG_Q_IDX[leg] = [_MODEL.idx_qs[_MODEL.getJointId(f'leg_{leg}_j{i+1}')] for i in range(5)]
        _LEG_FOOT_FID[leg] = _MODEL.getFrameId(f'leg_{leg}_foot')


_LEG_NAMES = ['FR', 'FL', 'HR', 'HL']


def _build_full_q(leg_q_dict, base_pos=None, base_quat=None):
    """모든 다리 q + 부유 베이스로 full q (27,) 구성.
    base_pos/base_quat=None이면 identity (q[0:7] = neutral).
    leg_q_dict: {leg_name: [q1..q5]} — 누락된 다리는 0.
    """
    _ensure_model()
    q = pin.neutral(_MODEL)
    if base_pos is not None:
        q[0:3] = base_pos
    if base_quat is not None:
        q[3:7] = base_quat
    for leg in _LEG_NAMES:
        if leg in leg_q_dict:
            for i, qi in enumerate(leg_q_dict[leg]):
                q[_LEG_Q_IDX[leg][i]] = qi
    return q


def _build_full_v(leg_dq_dict, base_v=None, base_w=None):
    """모든 다리 dq + 부유 베이스로 full v (26,) 구성."""
    _ensure_model()
    v = np.zeros(_MODEL.nv)
    if base_v is not None:
        v[0:3] = base_v
    if base_w is not None:
        v[3:6] = base_w
    for leg in _LEG_NAMES:
        if leg in leg_dq_dict:
            for i, vi in enumerate(leg_dq_dict[leg]):
                v[_LEG_V_IDX[leg][i]] = vi
    return v


# ── per-leg 인터페이스 (v11 호환) ──────────────────────────

def rnea_pin_per_leg(q_leg, dq_leg, ddq_leg, leg_name):
    """v11 rnea(q, dq, ddq, dh, lm) 동등 — 단일 다리 RNEA torque (5,)."""
    _ensure_model()
    q  = _build_full_q({leg_name: q_leg})
    v  = _build_full_v({leg_name: dq_leg})
    a  = _build_full_v({leg_name: ddq_leg})
    tau_full = pin.rnea(_MODEL, _DATA, q, v, a)   # (26,)
    return tau_full[_LEG_V_IDX[leg_name]]


def compute_mh_leg_pin(q_leg, dq_leg, leg_name):
    """v11 compute_mh_leg → 다리만 분리한 5×5 M, 5 h.
    h = pin.nonLinearEffects (Coriolis + gravity for full vector).
    """
    _ensure_model()
    q  = _build_full_q({leg_name: q_leg})
    v  = _build_full_v({leg_name: dq_leg})
    M_full = pin.crba(_MODEL, _DATA, q)
    h_full = pin.nonLinearEffects(_MODEL, _DATA, q, v)   # C·v + g
    idx = _LEG_V_IDX[leg_name]
    M_leg = M_full[np.ix_(idx, idx)]
    h_leg = h_full[idx]
    return M_leg, h_leg


def compute_jacobian_sim_pin(q_leg, leg_name):
    """v11 compute_jacobian_sim 동등 — 3×5 linear part of foot Jacobian (world == DH at neutral base)."""
    _ensure_model()
    q = _build_full_q({leg_name: q_leg})
    pin.computeJointJacobians(_MODEL, _DATA, q)
    pin.updateFramePlacements(_MODEL, _DATA)
    J6 = pin.getFrameJacobian(_MODEL, _DATA, _LEG_FOOT_FID[leg_name],
                               pin.LOCAL_WORLD_ALIGNED)
    return J6[:3, _LEG_V_IDX[leg_name]]


# ── full system (WBIC FB 정밀화용) ──────────────────────────

def compute_full_M_h(leg_q_dict, leg_dq_dict, base_pos=None, base_quat=None,
                      base_v=None, base_w=None):
    """전체 robot M (26×26), h (26,). FB coupling 포함."""
    _ensure_model()
    q = _build_full_q(leg_q_dict, base_pos, base_quat)
    v = _build_full_v(leg_dq_dict, base_v, base_w)
    M = pin.crba(_MODEL, _DATA, q)
    h = pin.nonLinearEffects(_MODEL, _DATA, q, v)
    return M, h, q, v


def get_indices():
    _ensure_model()
    return {
        'leg_v_idx': dict(_LEG_V_IDX),
        'leg_q_idx': dict(_LEG_Q_IDX),
        'leg_foot_fid': dict(_LEG_FOOT_FID),
        'fb_v_slice': slice(0, 6),       # floating base velocity 6 (lin + ang)
    }


if __name__ == '__main__':
    # 빠른 자가 검증
    Q_HOME_FRONT = [0.0, math.radians(133.2973), math.radians(46.7027),
                    math.radians(30.6583), math.radians(59.3417)]
    M_leg, h_leg = compute_mh_leg_pin(Q_HOME_FRONT, [0.0]*5, 'FR')
    print(f'FR M_leg diag: {np.diag(M_leg).round(4)}')
    print(f'FR h_leg     : {h_leg.round(4)}  (gravity torque at q_home)')
    J = compute_jacobian_sim_pin(Q_HOME_FRONT, 'FR')
    print(f'FR Jacobian shape: {J.shape}')
    tau = rnea_pin_per_leg(Q_HOME_FRONT, [0.0]*5, [0.0]*5, 'FR')
    print(f'FR RNEA tau (gravity): {tau.round(4)}')
