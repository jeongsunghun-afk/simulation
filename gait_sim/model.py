"""gait_sim.model — Robot model 정의 (DH, geometry, mass, inertia, joint limits).

v13.0 Phase 2: gait_sim_v12.py 의 model 영역 (line 252~390 + 1635~1751) 추출.

포함:
  · DH parameters (front / hind leg)
  · Q_HOME / Q_SWING joint angles
  · LEG_HIP_OFFSETS (body 기준 hip 위치)
  · LINK_MASS, BODY_MASS, TOTAL_MASS, LINK_RADIUS
  · BODY_INERTIA (pinocchio CRBA composite — 다리 mass 포함)
  · KP_PD / KD_PD / KP_IMP / KD_IMP (joint PD)
  · MU_FRICTION, JOINT_VEL_LIMIT, JOINT_TORQUE_LIMIT, FRONT/HIND_Q_LIM
"""
import math
import os

import numpy as np

from gait_sim.config import (
    CFG, BODY_FWD_F, BODY_FWD_H, BODY_LAT, BODY_Z_H,
)

# ══════════════════════════════════════════════════════════════
# HIND variant (env var: HIND_VARIANT=orig|ext)
#   orig : 원본 (간격 401mm)
#   ext  : 뒷발 -50mm 확장 (간격 451mm)
# ══════════════════════════════════════════════════════════════
_HIND_VARIANT = os.environ.get('HIND_VARIANT', 'orig')
assert _HIND_VARIANT in ('orig', 'ext'), f"HIND_VARIANT={_HIND_VARIANT} (expected 'orig'/'ext')"


# ══════════════════════════════════════════════════════════════
# DH 파라미터 (v13: 좌우 d2 부호 분리 — 거울 대칭 다리)
# Right legs (FR, HR): d2 = +0.0075   (외측으로 +y abduction)
# Left  legs (FL, HL): d2 = -0.0075   (외측으로 -y abduction → body frame +y 발 위치)
# 같은 Q_HOME 으로 FK 시 leg-local sim foot y 부호만 반전 → 좌우 거울 대칭.
# ══════════════════════════════════════════════════════════════
DH_FRONT_R = [
    (+math.pi/2, 0.0,   0.0,    ),
    (0.0,        0.21,  +0.0075,),    # ← d2 = +0.0075
    (0.0,        0.235, 0.0,    ),
    (0.0,        0.1,   0.0,    ),
    (0.0,        0.045, 0.0,    ),
]
DH_FRONT_L = [
    (+math.pi/2, 0.0,   0.0,    ),
    (0.0,        0.21,  -0.0075,),    # ← d2 = -0.0075  (좌측 미러)
    (0.0,        0.235, 0.0,    ),
    (0.0,        0.1,   0.0,    ),
    (0.0,        0.045, 0.0,    ),
]
DH_HIND_R = [
    (-math.pi/2, 0.0,   0.0,    ),
    (0.0,        0.21,  +0.0075,),
    (0.0,        0.21,  0.0,    ),
    (0.0,        0.148, 0.0,    ),
    (0.0,        0.045, 0.0,    ),
]
DH_HIND_L = [
    (-math.pi/2, 0.0,   0.0,    ),
    (0.0,        0.21,  -0.0075,),    # ← d2 = -0.0075  (좌측 미러)
    (0.0,        0.21,  0.0,    ),
    (0.0,        0.148, 0.0,    ),
    (0.0,        0.045, 0.0,    ),
]

# 호환용 alias — v12 코드에서 DH_FRONT/DH_HIND 단일 참조하던 곳은 우측 (R) 로 매핑.
# 좌측 IK 호출 시 LEG_DH[leg] 로 적절한 dh 전달.
DH_FRONT = DH_FRONT_R
DH_HIND  = DH_HIND_R

_A2_F = 0.21; _A3_F = 0.235; _A4_F = 0.1; _A5_F = 0.045; _D2_F = 0.0075


# ══════════════════════════════════════════════════════════════
# Home / Swing joint angles
# ══════════════════════════════════════════════════════════════
Q_HOME_FRONT_DEG = [0.0, 133.2973, 46.7027, 30.6583, 59.3417]    # 원본 (BODY_Z_H=-0.05)
if _HIND_VARIANT == 'ext':
    Q_HOME_HIND_DEG = [0.0, -154.8138, -92.8840, 88.6091, 60.0000]   # 뒷발 -50mm 확장
else:
    Q_HOME_HIND_DEG = [0.0, -150.0, -90.0, 90.0, 60.0]               # 원본
Q_HOME_FRONT = [math.radians(a) for a in Q_HOME_FRONT_DEG]
Q_HOME_HIND  = [math.radians(a) for a in Q_HOME_HIND_DEG]

# swing 중 opt_ik 비용함수 참조 자세 (th4 음수 유도, 나머지는 home 과 동일)
Q_SWING_FRONT_DEG = [0.0, 118.2973, 96.7027, -25.0, 59.3417]
Q_SWING_FRONT = [math.radians(a) for a in Q_SWING_FRONT_DEG]
Q_SWING_HIND_DEG = list(Q_HOME_HIND_DEG)
Q_SWING_HIND = [math.radians(a) for a in Q_SWING_HIND_DEG]

PHI_FRONT    = Q_HOME_FRONT[1] + Q_HOME_FRONT[2] + Q_HOME_FRONT[3]
PHI_HIND     = Q_HOME_HIND[1]  + Q_HOME_HIND[2]  + Q_HOME_HIND[3]
THETA5_FRONT = PHI_FRONT + Q_HOME_FRONT[4]
THETA5_HIND  = PHI_HIND  + Q_HOME_HIND[4]

Q_HOME_PER_LEG      = [Q_HOME_FRONT, Q_HOME_FRONT, Q_HOME_HIND, Q_HOME_HIND]
PHI_PER_LEG         = [PHI_FRONT, PHI_FRONT, PHI_HIND, PHI_HIND]
TRAJ_PT_IDX_PER_LEG = [4, 4, 4, 4]


# ══════════════════════════════════════════════════════════════
# Leg meta (이름, 색, DH, 관절 수)
# ══════════════════════════════════════════════════════════════
LEG_NAMES        = ['FR', 'FL', 'HR', 'HL']
LEG_COLORS       = ['#00d4ff', '#ff6b35', '#00ff99', '#c264ff']
LEG_DH           = [DH_FRONT_R, DH_FRONT_L, DH_HIND_R, DH_HIND_L]   # v13: 좌우 d2 미러
N_JOINTS_PER_LEG = [5, 5, 5, 5]
N_JOINTS_MAX     = 5


# ══════════════════════════════════════════════════════════════
# Hip 위치 (body 기준)
# DH 의 D2 오프셋(=0.0075)으로 인한 좌우 비대칭 보정:
# foot_local_y = -0.0075 → hip_y 에 +0.0075 적용 시 foot_world_y = ±BODY_LAT 정확히 대칭
# ══════════════════════════════════════════════════════════════
_HIP_Y_BIAS = CFG.hip_y_bias   # ≡ DH dh[1][2] (= D2_F = D2_H)
LEG_HIP_OFFSETS = np.array([
    [+BODY_FWD_F, -BODY_LAT + _HIP_Y_BIAS, 0.0     ],
    [+BODY_FWD_F, +BODY_LAT + _HIP_Y_BIAS, 0.0     ],
    [+BODY_FWD_H, -BODY_LAT + _HIP_Y_BIAS, BODY_Z_H],
    [+BODY_FWD_H, +BODY_LAT + _HIP_Y_BIAS, BODY_Z_H],
])


# ══════════════════════════════════════════════════════════════
# Mass / inertia
# ══════════════════════════════════════════════════════════════
BODY_MASS         = CFG.body_mass
LINK_MASS         = CFG.link_mass
LINK_MASS_PER_LEG = [LINK_MASS] * 4
TOTAL_MASS        = BODY_MASS + float(np.sum(LINK_MASS)) * 4.0
LINK_RADIUS       = CFG.link_radius

# Body inertia — default 는 base link 만 (CFG 의 0.07, 0.26, 0.26)
BODY_INERTIA = CFG.body_inertia

# v12.7: pinocchio CRBA 로 *다리 포함* composite body inertia (home pose 근사).
# base-link only 면 v11 standalone trot 발산 → CRBA 로 보강.
# crocoddyl / build_pin_model 모듈이 import 가능하면 자동 upgrade.
try:
    import pinocchio as _pin_init
    import build_pin_model as _bm_init
    _m_init = _bm_init.build_model()
    _d_init = _m_init.createData()
    _q0_init = _pin_init.neutral(_m_init)
    _Q_HOME_LEGS = {'FR': Q_HOME_FRONT, 'FL': Q_HOME_FRONT,
                    'HR': Q_HOME_HIND,  'HL': Q_HOME_HIND}
    for _leg_name, _qh in _Q_HOME_LEGS.items():
        for _i, _qi in enumerate(_qh):
            _q0_init[_m_init.idx_qs[_m_init.getJointId(f'leg_{_leg_name}_j{_i+1}')]] = _qi
    _M_full = _pin_init.crba(_m_init, _d_init, _q0_init)
    BODY_INERTIA = np.array(_M_full[3:6, 3:6])
    print(f"  [model] BODY_INERTIA upgraded (CRBA composite, home pose): "
          f"diag=[{BODY_INERTIA[0,0]:.3f}, {BODY_INERTIA[1,1]:.3f}, {BODY_INERTIA[2,2]:.3f}] "
          f"(vs base-link only [0.07, 0.26, 0.26])")
    del _m_init, _d_init, _q0_init, _M_full
except ImportError:
    print("  [model] pinocchio/build_pin_model 미설치 — BODY_INERTIA base-link only (다리 mass 미포함)")
except Exception as _e:
    print(f"  [model] BODY_INERTIA CRBA upgrade failed: {_e}")


# ══════════════════════════════════════════════════════════════
# Joint PD / Impedance / Actuator dynamics
# ══════════════════════════════════════════════════════════════
KP_PD  = CFG.kp_pd
KD_PD  = CFG.kd_pd
KP_IMP = CFG.kp_imp
KD_IMP = CFG.kd_imp

MU_DAMP      = CFG.mu_damp
TAU_LAG      = CFG.tau_lag
INIT_ERR_RAD = CFG.init_err_rad


# ══════════════════════════════════════════════════════════════
# Friction (contact)
# ══════════════════════════════════════════════════════════════
MU_FRICTION = CFG.mu_friction


# ══════════════════════════════════════════════════════════════
# Joint limits (속도 / 토크 / 각도)
# ══════════════════════════════════════════════════════════════
JOINT_VEL_LIMIT_RAD_S = CFG.joint_vel_limit_rad_s.astype(float)
JOINT_TORQUE_LIMIT    = CFG.joint_torque_limit
VEL_LIMIT_MARGIN      = CFG.vel_limit_margin

# 앞다리 관절 위치 한계 [rad]  — home: [0, 133.30, 46.70, 30.66, 59.34] deg
FRONT_Q_LIM = [
    (-math.radians(45),  math.radians(45)),    # th1: 어깨 벌림
    ( math.radians(45),  math.radians(210)),   # th2: 어깨 굴곡 (swing 고각 여유)
    (-math.radians(45),  math.radians(135)),   # th3: 팔꿈치
    (-math.radians(120), math.radians(60)),    # th4: 손목 (home +30.66 대비 마진 ~30°)
    (-math.radians(90),  math.radians(120)),   # th5: 발끝
]
# 뒷다리 관절 위치 한계 [rad]  — home: [0, -150, -90, 90, 60] deg
HIND_Q_LIM = [
    (-math.radians(45),  math.radians(45)),    # th1: 고관절 벌림
    (-math.radians(180), -math.radians(60)),   # th2: 고관절 굴곡
    (-math.radians(120),  math.radians(30)),   # th3: 무릎
    (-math.radians(30),   math.radians(150)),  # th4: 발목
    (-math.radians(90),   math.radians(120)),  # th5: 발끝
]
