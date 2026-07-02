"""quad_mpc_wbic — 실제 4족(02_Leg_UFDF_260610_7) MuJoCo 통합 테스트/제어.

biped wbic_balance.py 와 동일한 '단일 파일 + --mode' 관리 방식.
모델: quad_real.mjcf  (build_real_quad.py 로 생성).

  python3 quad_mpc_wbic.py --mode view    # 정적 기립 (physics 정지, 자세 확인)
  python3 quad_mpc_wbic.py --mode stand   # 중력 하 PD(+중력보상) 기립
  (향후) stance(WBIC) → lipm → march → mpc → nmpc
  ※ check(모델/동역학 정합)는 stance(WBIC) 동작 시 자동 검증되므로 별도 단계 없음

시각화: 구조3(quad_fulldynamics) 스타일 — footstep타겟·지지다각형·CoM투영·명령화살표·base/발궤적·마찰콘+GRF·텍스트. 항상표시.
"""
import os
import json
import sys
import math
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from qpsolvers import solve_qp

KP, KD, TAU_MAX = 80.0, 4.0, 80.0
MU, LAMZ_MIN = 0.6, 1.0          # 마찰계수(물리), 최소 수직지지력
# 제어기 마찰 안전마진: box pyramid(|λx|,|λy|≤μλz)는 대각서 원뿔보다 √2배 큼 →
# μ_ctrl = μ/√2 로 elliptic cone 안쪽에 내접시켜 보수화. (추후 SOCP 로 정식 elliptic 대체)
MU_MARGIN = float(os.environ.get('MU_MARGIN', '0.707'))
# 통합 WBC(legged_control식) swing 발끝 작업공간 PD 게인/가중
SW_KP = float(os.environ.get('SW_KP', '2400.0'))   # ★스윙 발끝 추종게인↑(800→2400): 1.0m/s 추종오차 124→46mm. 약하면 발끝이 plan 못쫓아 lurching(calf 가짜 스파이크)
SW_KD = float(os.environ.get('SW_KD', '110.0'))    # ↑(80→110): SW_KP 상향에 맞춘 임계감쇠(2√2400≈98)
W_SW = float(os.environ.get('W_SW', '90.0'))        # ★스윙 task 가중치↑(30→90): QP서 발끝추종 우선순위↑
_NOLIMIT = False                 # --nolimit: 관절 한계 해제 (가동범위 테스트용)

# 로봇 설정 — 우리 모델 / Go2 (--robot 으로 선택). 다리 인덱스 0,3 / 1,2 = 대각쌍(둘 다 동일)
_HERE = os.path.dirname(os.path.abspath(__file__))
ROBOTS = {
    'ours': dict(mjcf=os.path.join(_HERE, 'quad_real.mjcf'),
                 legs=['HL', 'HR', 'FL', 'FR'], dof=4,
                 foot_body='{L}_foot_contact_link', hip_body='{L}_hip_link',
                 foot_kind='mesh', base_z0=0.52, foot_z0=0.02, mu=0.6),   # _9 고정앞발목: 0.52서 무한안정(0.42=앞피칭 전복)
    'go2':  dict(mjcf=os.path.join(_HERE, '..', 'mujoco_menagerie', 'unitree_go2', 'scene.xml'),
                 legs=['FL', 'FR', 'RL', 'RR'], dof=3,
                 foot_geom='{L}', hip_body='{L}_hip',
                 foot_kind='sphere', base_z0=0.30, foot_z0=0.02, mu=0.6),
    # 02_Leg 발을 sphere 충돌로 교체(발목 자세 무관 점접촉) — box 모서리 rocking 회피 검증용
    'ours_sphere': dict(mjcf=os.path.join(_HERE, 'quad_real_sphere.mjcf'),
                        legs=['HL', 'HR', 'FL', 'FR'], dof=4,
                        foot_geom='{L}_sphere', hip_body='{L}_hip_link',
                        foot_kind='sphere', base_z0=0.52, mu=0.6),   # _9 고정앞발목: 0.52서 무한안정(0.42=앞피칭 전복)
    # ★260701 17-DOF(허리 fixed=16관절 전발목 4족). build_real_quad_17dof.py 생성. 앞/뒤 thigh·calf 축부호 반대
    'ours_17dof': dict(mjcf=os.path.join(_HERE, 'quad_real_17dof.mjcf'),
                       legs=['HL', 'HR', 'FL', 'FR'], dof=4,
                       foot_body='{L}_foot_contact_link', hip_body='{L}_hip_link',
                       foot_kind='mesh', base_z0=0.527, foot_z0=0.02, mu=0.6),
    'ours_17dof_sphere': dict(mjcf=os.path.join(_HERE, 'quad_real_17dof_sphere.mjcf'),
                       legs=['HL', 'HR', 'FL', 'FR'], dof=4,
                       foot_geom='{L}_sphere', hip_body='{L}_hip_link',
                       foot_kind='sphere', base_z0=0.5234, mu=0.6),
    # ★허리(FB_waist) 능동 17-DOF(nu=17). 허리=index8, 다리매핑 이름기반이라 무관. WAIST_KP로 요각제어
    'ours_17dof_waist_sphere': dict(mjcf=os.path.join(_HERE, 'quad_real_17dof_waist_sphere.mjcf'),
                       legs=['HL', 'HR', 'FL', 'FR'], dof=4,
                       foot_geom='{L}_sphere', hip_body='{L}_hip_link',
                       foot_kind='sphere', base_z0=0.5234, mu=0.6),
}
_ROBOT = 'ours'
MJCF = ROBOTS['ours']['mjcf']    # (하위호환)


class QuadSim:
    def __init__(self, mjcf=None):
        cfg = ROBOTS[_ROBOT]
        self.cfg = cfg
        self.legs = cfg['legs']; self.dof = cfg['dof']
        self.foot_kind = cfg['foot_kind']; self.base_z0 = cfg['base_z0']
        self.foot_z0 = cfg.get('foot_z0', 0.0)   # nominal 자세 발바닥 목표 z(접지=0). 과거 0.02→20mm 부양 버그
        global MU; MU = cfg['mu']
        _xmlp = mjcf or cfg['mjcf']
        self._stair = None                        # 계단 파라미터(H,D,N,X0) — 시각화/디버그용
        self._elev = os.environ.get('ELEV') is not None  # ★elevation 쿼리 강제 ON(실 depth/험지용, STAIRS 없이도 raycast)
        # ★GUI 라이브 컨트롤(CMDFILE로 갱신): 배속·모니터표시·지형적응
        self._rate = float(os.environ.get('RATE', '1.0'))            # 뷰어 배속(0=최대)
        self._viz = os.environ.get('VIZ', '1') != '0'               # 모니터 overlay(GRF/CoM/궤적/elevation) 표시
        self._terrain_on = False                                    # 지형적응(perception) — STAIRS/ELEV 확정 후 아래서 설정
        self._body_terr = 0.0                                       # 틱당 캐시: 4hip 평균 지형높이(a_z+MPC 공유)
        self._elev_cache = None; self._elev_t = -1.0                # elevation 시각화 격자 캐시(100ms마다 raycast)
        self._linefoot = float(os.environ.get('LINEFOOT', '0'))   # ★뒷발 선-접촉: 2번째 접촉구 간격[m](0=off, 점발). 발목이 pitch모멘트 받음
        if os.environ.get('STAIRS') or self._linefoot > 0:
            with open(_xmlp) as _f: _xml = _f.read()
            if os.environ.get('STAIRS'):          # ★계단: STAIR_H(단높이)·STAIR_D(단깊이)·STAIR_N(단수)·STAIR_X0(시작x)
                H = float(os.environ.get('STAIR_H', '0.05')); D = float(os.environ.get('STAIR_D', '0.25'))
                N = int(os.environ.get('STAIR_N', '6')); X0 = float(os.environ.get('STAIR_X0', '0.7'))
                self._stair = (H, D, N, X0)
                _box = '<geom type="box" pos="%.4f 0 %.4f" size="%.4f 1.0 %.4f" rgba="0.55 0.55 0.62 1" friction="1.3 0.02 0.001" condim="3"/>\n'
                _g = ''.join(_box % (X0 + i*D + D/2, (i+1)*H/2, D/2, (i+1)*H/2) for i in range(N))   # 오름 N단
                if os.environ.get('STAIR_UPDOWN'):    # ★올라갔다 내려오기: 정상 평지(2칸) + 내림 N단
                    _g += _box % (X0 + N*D + D/2, N*H/2, D/2, N*H/2)                                  # 정상 발판(1칸)
                    _g += ''.join(_box % (X0 + (N+1)*D + j*D + D/2, (N-1-j)*H/2, D/2, (N-1-j)*H/2)
                                  for j in range(N-1))                                                # 내림(N-1단, 마지막=지면)
                _xml = _xml.replace('</worldbody>', _g + '</worldbody>')
            if self._linefoot > 0:                # ★선-발: HL/HR에 2번째 접촉구(원점에서 후방 _linefoot)
                for _L in ('HL', 'HR'):
                    _orig = '<geom name="%s_sphere"' % _L
                    _s2 = ('<geom name="%s_sphere2" type="sphere" size="0.018" pos="%.6f 0.000000e+00 -0.056668" '
                           'rgba="0.3 0.4 0.95 1" friction="1.3 0.02 0.001" condim="3" />\n                ' % (_L, 0.024546 - self._linefoot)) + _orig
                    _xml = _xml.replace(_orig, _s2, 1)
            _tmp = os.path.join(os.path.dirname(_xmlp), '_inject_tmp.mjcf')   # 메시 상대경로 위해 같은 폴더
            with open(_tmp, 'w') as _f: _f.write(_xml)
            self.m = mujoco.MjModel.from_xml_path(_tmp); os.remove(_tmp)
        else:
            self.m = mujoco.MjModel.from_xml_path(_xmlp)
        self._terrain_on = bool(self._stair) or self._elev          # ★_stair 확정 후 지형적응 기본값 설정
        if _NOLIMIT:
            self.m.jnt_limited[:] = 0    # 관절 한계 해제 (테스트용)
        else:
            # 관절 한계를 단단하게: 기본 soft 제약(solreflimit 시정수 0.02s)은 동적 충격에
            # 한계를 뚫음(최대 0.3rad). 시정수↓+impedance↑ 로 실제 기계 스톱처럼 강화.
            lim = self.m.jnt_limited.astype(bool)
            self.m.jnt_solref[lim] = [0.004, 1.0]              # 2*timestep, 빠른 복원
            self.m.jnt_solimp[lim] = [0.95, 0.99, 0.001, 0.5, 2.0]  # 높은 impedance
        # ★다리무게 가설 검증: 다리 링크 질량/관성만 스케일 (LEG_MASS_SCALE, 기본1.0=원본)
        _lms = float(os.environ.get('LEG_MASS_SCALE', '1.0'))
        if _lms != 1.0:
            for _b in range(self.m.nbody):
                _bn = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, _b) or ''
                if any(_s in _bn for _s in ('hip', 'thigh', 'calf', 'foot')):
                    self.m.body_mass[_b] *= _lms; self.m.body_inertia[_b] *= _lms
        # ★바디무게 추가(BODY_ADD kg): 다리질량 그대로, 비율만 낮춤(다리 79%→낮게). base 바디에 추가+관성 비례
        _bad = float(os.environ.get('BODY_ADD', '0'))
        if _bad != 0.0:
            _bb = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'base')
            _m0 = self.m.body_mass[_bb]; _mn = _m0 + _bad
            self.m.body_inertia[_bb] *= (_mn / _m0); self.m.body_mass[_bb] = _mn
        self.d = mujoco.MjData(self.m)
        if _lms != 1.0 or _bad != 0.0:
            mujoco.mj_setConst(self.m, self.d)
            print('[MASS] 다리×%.2f 바디+%.1fkg → base %.2fkg 총 %.1fkg (다리비율 %.0f%%)'
                  % (_lms, _bad, self.m.body_mass[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'base')],
                     self.m.body_mass.sum(), 100*(1 - self.m.body_mass[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'base')]/self.m.body_mass.sum())), flush=True)
        self.nv = self.m.nv
        self.nu = self.m.nu          # = 4*dof
        N = lambda kind, nm: mujoco.mj_name2id(self.m, kind, nm)
        self.hip_bid = [N(mujoco.mjtObj.mjOBJ_BODY, cfg['hip_body'].format(L=L))
                        for L in self.legs]
        # 다리별 qpos/qvel 인덱스 — 관절 이름으로 조회(비균일 DOF 대응: 앞발목 fixed면 3, 뒤 4)
        _JT = ['hip', 'thigh', 'calf', 'foot']
        self.legqp = []; self.legqv = []
        for L in self.legs:
            jids = [N(mujoco.mjtObj.mjOBJ_JOINT, '%s_%s_joint' % (L, jt)) for jt in _JT]
            jids = [j for j in jids if j >= 0]
            self.legqp.append([int(self.m.jnt_qposadr[j]) for j in jids])
            self.legqv.append([int(self.m.jnt_dofadr[j]) for j in jids])
        self.leg_dof = [len(qp) for qp in self.legqp]   # 다리별 DOF(비균일)
        # 발 접촉 geom/body + 접촉점 정의 (mesh=최저정점 / sphere=중심−반지름)
        self.foot_bid = [0] * 4; self.foot_gid = [0] * 4
        self.sole_off = [None] * 4; self.foot_r = [0.0] * 4
        for i, L in enumerate(self.legs):
            if self.foot_kind == 'mesh':
                fb = N(mujoco.mjtObj.mjOBJ_BODY, cfg['foot_body'].format(L=L))
                gid = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == fb
                       and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                       and self.m.geom_contype[g] != 0][0]
                mid = self.m.geom_dataid[gid]
                adr = self.m.mesh_vertadr[mid]; nvt = self.m.mesh_vertnum[mid]
                Vv = self.m.mesh_vert[adr:adr + nvt].reshape(-1, 3)
                vlow = Vv[np.argmin(Vv[:, 2])]
                Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, self.m.geom_quat[gid])
                self.sole_off[i] = self.m.geom_pos[gid] + Rg.reshape(3, 3) @ vlow
            else:  # sphere
                gid = N(mujoco.mjtObj.mjOBJ_GEOM, cfg['foot_geom'].format(L=L))
                fb = self.m.geom_bodyid[gid]; self.foot_r[i] = float(self.m.geom_size[gid][0])
            self.foot_bid[i] = fb; self.foot_gid[i] = gid
        # ★선-발: 뒷발 2번째 접촉구(sphere2) 감지 → foot_point2/jac2·WBIC 2점 접촉용
        self.foot_gid2 = [-1] * 4; self.foot_r2 = [0.0] * 4
        if self._linefoot > 0:
            for i, L in enumerate(self.legs):
                g2 = N(mujoco.mjtObj.mjOBJ_GEOM, L + '_sphere2')
                if g2 >= 0:
                    self.foot_gid2[i] = g2; self.foot_r2[i] = float(self.m.geom_size[g2][0])
        # ★물리환경을 구조3(quad_fulldynamics.py line54-58)과 동일하게 ──
        #   timestep(1kHz)·CONE(마찰콘)·STIFF(접촉강성 전 geom solref)·FRIC(마찰). 발 침투(soft 접촉) 해결 포함.
        self.m.opt.timestep = float(os.environ.get('TIMESTEP', '0.001'))    # 구조3 dt_simu=1kHz
        if os.environ.get('CONE'):
            self.m.opt.cone = int(os.environ['CONE'])                       # 0=pyramidal 1=elliptic
        _stiff = float(os.environ.get('STIFF', '0.005'))                    # 기본 단단(발 침투 방지). 0=끄기
        if _stiff > 0:
            self.m.geom_solref[:, 0] = _stiff; self.m.geom_solref[:, 1] = 1.0   # 전 geom(구조3 동일)
        if os.environ.get('FRIC'):
            self.m.geom_friction[:, 0] = float(os.environ['FRIC'])
        self.q_home = None
        self.com_ref = None
        self.last_lam = None
        self.foot_targets = [None, None, None, None]   # 다음 착지 목표 (시각화용)
        self.contact_state = [False, False, False, False]   # detect_contact 결과 (색칠용)
        # 발 접촉 geom (접촉 시 빨강 색칠용) + 원래 색 저장
        self.foot_geoms = [[self.foot_gid[i]] for i in range(4)]
        self._foot_rgba0 = {g: self.m.geom_rgba[g].copy()
                            for gs in self.foot_geoms for g in gs}
        # ── 시각화 궤적(구조3 quad_fulldynamics 스타일 통일) ──
        from collections import deque as _deque
        _tn = int(os.environ.get('TRAIL_N', '300'))
        self.cmd_v = np.zeros(6)                          # 명령속도(화살표·텍스트용; mode가 설정)
        self.cmd_mode = 'move'                            # 현재 모드(GUI; move/stand_up/stand_down)
        # per-joint Peak토크(QP 한계+클립용): jnt_actfrcrange = hip/thigh84·calf126·foot168, qpos/ctrl 순서 일치
        self._tau_peak = np.array([self.m.jnt_actfrcrange[j, 1] if self.m.jnt_actfrcrange[j, 1] > 0 else 1e8
                                   for j in range(self.m.njnt) if self.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE])
        # ★실모터 모델(sim2real): 관절별 각속도한계(no-load ω) — hip/thigh29.6·calf19.7·foot14.8 rad/s
        _jnames = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
                   for j in range(self.m.njnt) if self.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        _wmap = {'hip': 29.6, 'thigh': 29.6, 'calf': 19.7, 'foot': 14.8}
        self._w_limit = np.array([next((v for k, v in _wmap.items() if k in n), 29.6) for n in _jnames])
        self._motor_curve = os.environ.get('MOTOR_CURVE') is not None   # 토크-속도 곡선(고속서 가용토크↓)
        self._qdd_prev = np.zeros(self.nu)                              # JERK_LIM용 직전 q̈(관절)
        self._tau_prev = np.zeros(self.nu)                              # TAU_RATE용 직전 τ(관절)
        self._tau_filt = np.zeros(self.nu)                              # 출력 토크 LPF 상태
        self._tau_lpf = float(os.environ.get('TAU_LPF', '0'))          # ★출력 토크 LPF 차단주파수[Hz](0=off, QP밖 표준필터)
        # per-joint 위치 한계(WBIC QP, 가속도경계용): jnt_range
        _jl = [(self.m.jnt_range[j, 0], self.m.jnt_range[j, 1]) if self.m.jnt_limited[j] else (-1e9, 1e9)
               for j in range(self.m.njnt) if self.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        self._qmin = np.array([a for a, b in _jl]); self._qmax = np.array([b for a, b in _jl])
        # 뒷발목(4DoF 여자유도) nu-order 인덱스 + 핀 가중치: posture로 REAR_ANKLE에 고정→흔들림↓·좌우대칭
        self._ankle_idx = set(int(self.legqv[i][3]) - 6 for i in range(4) if self.leg_dof[i] == 4)
        self._ankle_w = float(os.environ.get('ANKLE_W', '20'))   # 0이면 핀 안함(기존 여자유도)
        _sw0 = os.environ.get('SWING_W', '2.0')                  # ★스윙 여유도 규제(whip 억제) 공통 기본
        self._swing_w_r = float(os.environ.get('SWING_W_R', _sw0))  # 뒷다리 whip(GUI 슬라이더 live)
        self._swing_w_f = float(os.environ.get('SWING_W_F', _sw0))  # 앞다리 whip(별도)
        self._front_idx = set(int(self.legqv[i][t]) - 6 for i in range(4)
                              if self.legs[i] in ('FL', 'FR') for t in range(self.leg_dof[i]))
        # ★허리(FB_waist yaw) 관절 — 큰 몸통 DOF라 강한 전용 홀드 필요(약한 posture론 앞몸통 못잡음)
        _wj = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, 'FB_waist_joint')
        self._waist_idx = int(self.m.jnt_dofadr[_wj]) - 6 if _wj >= 0 else None   # nu-index(없으면 None=16DOF)
        self._waist_ref = 0.0                                    # 허리 요각 목표[rad](조향시 갱신)
        self._waist_w = float(os.environ.get('WAIST_W', '80'))   # 홀드 가중(강하게)
        self._waist_kp = float(os.environ.get('WAIST_KP', '150')); self._waist_kd = float(os.environ.get('WAIST_KD', '20'))
        self._waist_steer = float(os.environ.get('WAIST_STEER', '0'))   # ★허리 조향 게인(0=중립홀드/>0=선회시 앞몸통 굽힘=조향스파인)
        self._wbt = bool(os.environ.get('WBIC_TIMING')); self._qpt = []   # ★WBIC QP solve 시간 계측(1kHz 실현 확인용)
        # ★기어비 재배분(설계검토): GEAR_xxx<1=저기어(속도↑·토크↓), >1=고기어(토크↑·속도↓). 같은 베이스모터 토크↔속도 맞교환.
        #   보행분석: thigh=토크병목·속도여유→GEAR_THIGH>1 이득 / calf·foot=속도병목·토크여유→GEAR<1 이득
        _gdef = {'hip': '1.0', 'thigh': '1.0', 'calf': '0.7619', 'foot': '1.0'}   # ★calf 기본 8:1(10.5→8): 뒷calf 속도병목 해소(19.7→25.9, 토크126→96). foot은 flail이라 유지
        for _grp in ('hip', 'thigh', 'calf', 'foot'):
            _g = float(os.environ.get('GEAR_' + _grp.upper(), _gdef[_grp]))
            if _g != 1.0:
                for k in range(self.nu):
                    if _grp in _jnames[k]:
                        self._w_limit[k] /= _g; self._tau_peak[k] *= _g
        # ★기어박스 물리 모델(sim2real, MJCF엔 0 → flail 과장 보정): 반사관성 I_rotor·N² + 점성감쇠 + Coulomb마찰
        #   GEARBOX=1로 켜고, 값은 env로 조정. 대략값 기본(실측 스펙 들어오면 교체).
        _gearmap = {'hip': 7.0, 'thigh': 7.0, 'calf': 10.5, 'foot': 14.0}   # 관절별 감속비
        if os.environ.get('GEARBOX') == '1':
            _Irot = float(os.environ.get('ROTOR_I', '1e-4'))   # 모터 로터관성[kg·m²] (대략)
            _jdmp = float(os.environ.get('JDAMP', '0.1'))      # 관절 점성감쇠[N·m·s/rad] (대략)
            _jfrc = float(os.environ.get('JFRIC', '0.5'))      # 관절 Coulomb 마찰[N·m] (대략)
            for k in range(self.nu):
                _grp = next((kk for kk in _gearmap if kk in _jnames[k]), 'hip')
                _N = _gearmap[_grp] * float(os.environ.get('GEAR_' + _grp.upper(), '1.0'))   # ★GEAR_* 반영 실효 기어
                _dof = 6 + k                                   # base free=0~5, 능동관절=6+k (qvel 규약 일치)
                self.m.dof_armature[_dof] = _Irot * _N * _N    # ★반사관성: 발목14²=196배 → 유효관성↑로 flail 억제
                self.m.dof_damping[_dof] = _jdmp
                self.m.dof_frictionloss[_dof] = _jfrc
        self.base_trail = _deque(maxlen=_tn)              # base 궤적(마젠타)
        self.foot_trail = [_deque(maxlen=_tn) for _ in range(4)]   # 발별 궤적

    # ── 운동학/동역학 헬퍼 ────────────────────────────
    def fullM(self):
        M = np.zeros((self.nv, self.nv)); mujoco.mj_fullM(self.m, M, self.d.qM); return M

    def terrain_height(self, x, y, z_high=3.0):
        """지형 표면 높이 @ (x,y) [m] — ★표준 perceptive locomotion의 elevation-map 인터페이스.
        현재 백엔드: sim 지오메트리에 하향 raycast(=depth센서가 측정하는 것과 동일, 로봇 geom은 건너뜀).
        ★추후: 실제 depth센서 elevation map으로 이 메서드 백엔드만 교체(인터페이스 동일)."""
        if not self._terrain_on:                     # 지형적응 off(GUI 토글)/평지: raycast 생략(성능, 평지=0)
            return 0.0
        vec = np.array([0.0, 0.0, -1.0]); gid = np.zeros(1, np.int32); z = z_high
        for _ in range(8):                            # 로봇 geom 맞으면 그 아래서 재캐스트 → 지형(worldbody)만
            dist = mujoco.mj_ray(self.m, self.d, np.array([float(x), float(y), z]), vec, None, 1, -1, gid)
            if dist < 0.0:
                return 0.0
            if int(self.m.geom_bodyid[gid[0]]) == 0:  # worldbody = 지형(floor/stairs)
                return z - dist
            z = (z - dist) - 0.01                      # 로봇이면 그 바로 아래서 재시작
        return 0.0                                     # 8회 내 지형 못맞춤 → 평지로

    def foot_point(self, i):
        if self.foot_kind == 'mesh':
            return self.d.xpos[self.foot_bid[i]] + \
                self.d.xmat[self.foot_bid[i]].reshape(3, 3) @ self.sole_off[i]
        return self.d.geom_xpos[self.foot_gid[i]] - np.array([0, 0, self.foot_r[i]])  # sphere 바닥

    def foot_jac(self, i):
        jp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, self.d, jp, None, self.foot_point(i), self.foot_bid[i])
        return jp

    def foot_point2(self, i):                            # ★선-발: 뒷발 2번째 접촉구 바닥점
        return self.d.geom_xpos[self.foot_gid2[i]] - np.array([0, 0, self.foot_r2[i]])

    def foot_jac2(self, i):
        jp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, self.d, jp, None, self.foot_point2(i), self.foot_bid[i])
        return jp

    def _foot_pt(self, i, dat):
        if self.foot_kind == 'mesh':
            return dat.xpos[self.foot_bid[i]] + \
                dat.xmat[self.foot_bid[i]].reshape(3, 3) @ self.sole_off[i]
        return dat.geom_xpos[self.foot_gid[i]] - np.array([0, 0, self.foot_r[i]])

    def ik_leg(self, i, p_tgt, seed):
        """다리 i 의 발끝을 p_tgt(world)로 보내는 관절각 (damped-LS, scratch data)."""
        if not hasattr(self, '_scratch'):
            self._scratch = mujoco.MjData(self.m)
        s = self._scratch; s.qpos[:] = self.d.qpos[:]; qj = seed.copy()
        for _ in range(10):
            s.qpos[self.legqp[i]] = qj
            mujoco.mj_kinematics(self.m, s)
            mujoco.mj_comPos(self.m, s)        # ★ mj_jac 전제: comPos(cdof) 없으면 J=0
            e = p_tgt - self._foot_pt(i, s)
            jp = np.zeros((3, self.nv))
            mujoco.mj_jac(self.m, s, jp, None, self._foot_pt(i, s), self.foot_bid[i])
            J = jp[:, self.legqv[i]]
            qj = qj + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
        return qj

    def crouch_home(self, base_z=None):
        """q=0(또는 keyframe) 넓은 발 위치 유지한 채 base 낮춰 무릎굽힘 → q_home/com_ref."""
        d, m = self.d, self.m
        base_z = base_z if base_z is not None else float(os.environ.get('BASE_Z0', self.base_z0))
        if m.nkey > 0:
            mujoco.mj_resetDataKeyframe(m, d, 0)   # 모델 keyframe(home) 있으면 사용
        else:
            d.qpos[:] = 0; d.qpos[3] = 1
        d.qpos[2] = 0.60; mujoco.mj_forward(m, d)
        foot_xy = [self.foot_point(i)[:2].copy() for i in range(4)]
        d.qpos[2] = base_z
        # ★뒷다리(4DoF) 발목을 REAR_ANKLE로 두고 hip/thigh/calf로 같은 발 위치 IK
        #   → 같은 발 궤적, 더 웅크린 자세(발목 꺾고 thigh 펴짐). 4-DoF 여유(redundancy) 해소.
        _ra = float(os.environ.get('REAR_ANKLE', '-0.3'))   # ★뒷발목(2026-07-02 튜닝): -0.5→-0.3 뒷다리 신전(thigh -0.35→-0.18, 앞다리에 근접)+비대칭 완화로 tilt_max V1.0 2.0→1.0. sphere발이라 접촉無영향
        _fa = float(os.environ.get('FRONT_ANKLE', '-0.5'))  # ★앞발목 기본 -0.5(뒷발과 별도). 앞다리 축부호 반대라 -0.5=앞발 자세. 비대칭(앞-0.5/뒤-0.3)이 대칭보다 tilt 낮음
        for i in range(4):
            if self.leg_dof[i] == 4:
                _ang = _fa if self.legs[i] in ('FL', 'FR') else _ra
                if _ang != 0.0:
                    d.qpos[self.legqp[i][3]] = _ang
        for _ in range(300):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            for i in range(4):
                tgt = np.array([foot_xy[i][0], foot_xy[i][1], self.foot_z0])
                e = tgt - self.foot_point(i)
                vi = self.legqv[i][:3]; qi = self.legqp[i][:3]   # hip/thigh/calf만 — 뒷발 foot관절은 0(직선) 유지해 앞발(용접)과 대칭
                J = self.foot_jac(i)[:, vi]
                d.qpos[qi] += 0.5 * (J.T @ np.linalg.solve(
                    J @ J.T + 1e-4 * np.eye(3), e))
        mujoco.mj_forward(m, d)
        self.q_home = d.qpos[7:7 + self.nu].copy()
        fc = np.mean([self.foot_point(i)[:2] for i in range(4)], axis=0)
        self.com_ref = np.array([fc[0], fc[1], d.subtree_com[0][2]])
        # 명목(기본자세) 발 위치 — hip 기준 오프셋(body frame) + 발 z. Ready 호밍 목표.
        self.foot_hip_off = [self.foot_point(i)[:2] - d.xpos[self.hip_bid[i]][:2] for i in range(4)]
        self.foot_gz0 = [float(self.foot_point(i)[2]) for i in range(4)]
        return self.q_home

    def update_stand_qhome(self, base_z):
        """target base_z용 q_home/com_ref 재계산(IK) — 라이브 d는 복원(텔레포트X).
        WBIC posture+CoM task가 새 q_home으로 부드럽게 구동 → 서기 높이변경/눕기."""
        _q = self.d.qpos.copy(); _v = self.d.qvel.copy(); _t = self.d.time
        self.crouch_home(base_z)                       # q_home/com_ref 갱신(d 텔레포트됨)
        self.d.qpos[:] = _q; self.d.qvel[:] = _v; self.d.time = _t   # 라이브 d 복원
        mujoco.mj_forward(self.m, self.d)

    # ── WBIC stance QP (4발 접촉정합 균형) ─────────────
    def wbic_stance(self, contacts=(0, 1, 2, 3)):
        d, m, nv = self.d, self.m, self.nv
        K = len(contacts); nz = nv + 3 * K
        sl = lambda k: slice(nv + 3 * k, nv + 3 * k + 3)
        M = self.fullM(); h = d.qfrc_bias.copy(); qv = d.qvel.copy()
        P = np.zeros((nz, nz)); g = np.zeros(nz)
        # CoM xyz task
        Jc = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
        a_com = np.array([120, 120, 200]) * (self.com_ref - d.subtree_com[0]) \
            - np.array([20, 20, 25]) * (Jc @ qv)
        for r in range(3):
            P[:nv, :nv] += np.outer(Jc[r], Jc[r]); g[:nv] -= a_com[r] * Jc[r]
        # base orientation (upright)
        oerr = np.zeros(3); mujoco.mju_quat2Vel(oerr, d.qpos[3:7], 1.0)
        a_ori = 150 * (-oerr) - 20 * qv[3:6]
        for j in range(3):
            P[3 + j, 3 + j] += 5.0; g[3 + j] -= 5.0 * a_ori[j]
        # posture (crouch 유지)
        a_post = 60 * (self.q_home - d.qpos[7:7 + self.nu]) - 5 * qv[6:6 + self.nu]
        for j in range(self.nu):
            P[6 + j, 6 + j] += 1.0; g[6 + j] -= a_post[j]
        P[:nv, :nv] += 1e-4 * np.eye(nv)
        for k in range(K):
            P[sl(k), sl(k)] += 1e-3 * np.eye(3)
        # equality: floating-base 6 + 디딘발 무가속
        Js = [self.foot_jac(c) for c in contacts]
        A = np.zeros((6, nz)); b = -h[0:6]; A[:, :nv] = M[0:6, :]
        for k, J in enumerate(Js):
            A[:, sl(k)] = -J[:, 0:6].T
        for J in Js:
            Ac = np.zeros((3, nz)); Ac[:, :nv] = J
            A = np.vstack([A, Ac]); b = np.concatenate([b, np.zeros(3)])
        # ineq: 마찰추 + λz≥min
        lb = np.full(nz, -1e8); ub = np.full(nz, 1e8); Gl = []; hl = []
        for k in range(K):
            o = nv + 3 * k; lb[o + 2] = LAMZ_MIN
            for sx, sy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                row = np.zeros(nz); row[o] = sx; row[o + 1] = sy; row[o + 2] = -MU * MU_MARGIN
                Gl.append(row); hl.append(0.0)
        P = 0.5 * (P + P.T) + 1e-8 * np.eye(nz)
        z = solve_qp(P, g, np.vstack(Gl), np.array(hl), A, b, lb, ub, solver='quadprog')
        if z is None:
            self.last_lam = None
            return None, False
        qdd = z[:nv]; lam = [z[sl(k)] for k in range(K)]
        tau = M[6:6 + self.nu, :] @ qdd + h[6:6 + self.nu]
        for k, J in enumerate(Js):
            tau -= J[:, 6:6 + self.nu].T @ lam[k]
        self.last_lam = lam
        self._qdd_prev = qdd[6:6 + self.nu].copy(); self._tau_prev = tau.copy()   # JERK_LIM/TAU_RATE 상태
        # ★토크 한계: 기본=Peak 상수클립. MOTOR_CURVE면 토크-속도 곡선(고속서 가용토크↓=실모터)
        if self._motor_curve:
            _w = d.qvel[6:6 + self.nu]
            _avail = self._tau_peak * np.maximum(0.0, 1.0 - np.abs(_w) / self._w_limit)
            d.ctrl[:] = np.clip(tau, -_avail, _avail)
        else:
            d.ctrl[:] = np.clip(tau, -self._tau_peak, self._tau_peak)   # per-joint Peak 클립(QP가 이미 존중)
        if self._tau_lpf > 0:                               # ★출력단 토크 LPF(QP밖,1차): 1kHz 계단성분 평활(공진억제). 위상지연=안정성 트레이드오프
            _al = 2 * np.pi * self._tau_lpf * self.m.opt.timestep; _al /= (1 + _al)
            self._tau_filt = _al * d.ctrl[:self.nu] + (1 - _al) * self._tau_filt
            d.ctrl[:self.nu] = self._tau_filt
        return tau, True

    # ── MPC (Linear Convex, gait_sim.controllers.mpc 재사용) + WBIC 추종 ──
    def compute_Icom(self):
        """현재 자세의 CoM 기준 복합 관성 (world)."""
        com = self.d.subtree_com[0]; I = np.zeros((3, 3))
        for b in range(1, self.m.nbody):
            ms = self.m.body_mass[b]
            if ms <= 0:
                continue
            r = self.d.xipos[b] - com
            Rb = self.d.ximat[b].reshape(3, 3)
            Ib = Rb @ np.diag(self.m.body_inertia[b]) @ Rb.T
            I += Ib + ms * (r @ r * np.eye(3) - np.outer(r, r))
        return I

    def setup_mpc(self):
        """gait_sim 4족 SRBD MPC 를 실제 4족용으로 연결 (모듈상수 override)."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _argv = sys.argv; sys.argv = [_argv[0]]
        try:
            import gait_sim.controllers.mpc as MPC
        finally:
            sys.argv = _argv
        mujoco.mj_forward(self.m, self.d)
        TOT = float(self.m.body_subtreemass[0]); Ic = self.compute_Icom()
        MPC.TOTAL_MASS = TOT; MPC.BODY_INERTIA = Ic; MPC._I_BODY = Ic
        MPC.MU_FRICTION = MU * MU_MARGIN   # elliptic cone 내접 마진(A안)
        MPC.MPC_Q = np.diag([100., 100., 1., 1., 1., 50., 0., 0., 1., 1., 1., 1., 0.])
        MPC.MPC_R = 1e-6 * np.eye(3)
        MPC.LAMZ_MIN = 1.0
        MPC.LAMZ_MAX = float(os.environ.get('LAMZ_MAX_K', '2.0')) * TOT * 9.81  # 발당 수직력 상한(동적 trot)
        MPC.N_MPC = int(os.environ.get('N_MPC', '14'))   # 호라이즌 0.28s — 선형MPC 최적(>0.4s면 고정R 가정 깨져 발산)
        self.MPC = MPC; self.N_MPC = MPC.N_MPC

    def body_x0(self):
        """MuJoCo 상태 → MPC body state(13). [r,p,y, px,py,pz, ωxyz, vxyz, g]."""
        d, m = self.d, self.m
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, d.qpos[3:7]); Rm = Rm.reshape(3, 3)
        pitch = math.asin(max(-1, min(1, -Rm[2, 0])))
        roll = math.atan2(Rm[2, 1], Rm[2, 2]); yaw = math.atan2(Rm[1, 0], Rm[0, 0])
        com = d.subtree_com[0]
        Jc = np.zeros((3, m.nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
        vcom = Jc @ d.qvel; omega_w = Rm @ d.qvel[3:6]
        return np.array([roll, pitch, yaw, com[0], com[1], com[2],
                         omega_w[0], omega_w[1], omega_w[2],
                         vcom[0], vcom[1], vcom[2], -9.81])

    def mpc_grf(self, x_ref, contact_sched=None):
        """MPC 호출 → lam_des (4,3). contact_sched=None 이면 전부 stance."""
        com = self.d.subtree_com[0].copy(); N = self.N_MPC
        cs = np.ones((N, 4), dtype=bool) if contact_sched is None else contact_sched
        fp = np.zeros((N, 4, 3))
        foot_rel = [self.foot_point(i) - com for i in range(4)]
        for k in range(N):
            for i in range(4):
                fp[k, i] = foot_rel[i]
        return self.MPC.mpc_qp_plan(self.body_x0(), cs, fp, x_ref_step=x_ref, ltv=True)

    def wbic_track(self, lam_des, contacts=(0, 1, 2, 3), w_lam=10.0, swing=None):
        """통합 WBC (legged_control 식): 단일 QP z=[q̈; λ_stance] → τ(전 관절).

        stance: floating-base EOM(등식) + 디딘발 무가속(등식) + 마찰추(부등식)
                + λ 가 lam_des(MPC) 추종(soft).
        swing : swing={leg:(p_des, v_des)} 주면 발끝 작업공간 PD
                accel = SW_KP(p*−p)+SW_KD(v*−v) 를 J q̈ ≈ accel 로 같은 QP에 soft task 통합.
                → swing 다리도 stance 와 일관된 토크로 풀려 별도 IK-PD 루프 불필요."""
        d, m, nv = self.d, self.m, self.nv
        swing = swing or {}
        # ★선-발: 뒷발 stance면 2번째 접촉점 추가(=선접촉). cjac/cpos/clam=확장 접촉점들. MPC힘은 점들이 분할추종(차등=pitch모멘트→발목이 받음)
        cjac = []; cpos = []; clam = []
        for c in contacts:
            _pts = [(self.foot_jac(c), self.foot_point(c))]
            if self._linefoot > 0 and self.foot_gid2[c] >= 0:
                _pts.append((self.foot_jac2(c), self.foot_point2(c)))
            for _J, _P in _pts:
                cjac.append(_J); cpos.append(_P); clam.append(lam_des[c] / len(_pts))
        K = len(cjac); nz = nv + 3 * K
        sl = lambda k: slice(nv + 3 * k, nv + 3 * k + 3)
        M = self.fullM(); h = d.qfrc_bias.copy(); qv = d.qvel.copy()
        P = np.zeros((nz, nz)); g = np.zeros(nz)
        # swing 발끝 작업공간 PD task (legged_control formulateSwingLegTask)
        sw_vidx = set()
        for leg, (p_des, v_des, *_) in swing.items():
            J = self.foot_jac(leg)
            accel = SW_KP * (p_des - self.foot_point(leg)) + SW_KD * (v_des - J @ qv)
            P[:nv, :nv] += W_SW * (J.T @ J); g[:nv] -= W_SW * (J.T @ accel)
            sw_vidx.update(self.legqv[leg])
        # base 안정화 task (legged_control BaseAccelTask 해당): 자세(upright)+높이(z)
        #   — swing 반력에 의한 pitch/sink 억제. xy 병진은 MPC λ 에 위임(전진 보행 유지).
        Jc = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
        oerr = np.zeros(3); mujoco.mju_quat2Vel(oerr, d.qpos[3:7], 1.0)
        a_ori = 150 * (-oerr) - 20 * qv[3:6]
        w_ori = float(os.environ.get('W_ORI', '20'))   # ★17dof 튜닝(2026-07-02): 5→20 자세추종↑ → 고속 tilt_max −37%(V1.5 5.2→3.3°). 벤치 falls=0 전속도
        for j in range(3):
            P[3 + j, 3 + j] += w_ori; g[3 + j] -= w_ori * a_ori[j]
        _th = self._body_terr                                 # ★틱당 1회 캐시된 4hip 평균 지형높이(mode_trot서 갱신, raycast 절약)
        _zref = self.com_ref[2] + _th                         # 몸통 높이 기준을 지형따라 부드럽게 올림(평지=+0)
        self._dbg_terr = _th; self._dbg_clr = d.subtree_com[0][2] - _th   # 진단: 지형높이·clearance(몸-지면)
        a_z = 200 * (_zref - d.subtree_com[0][2]) - 25 * (Jc @ qv)[2]
        w_z = float(os.environ.get('W_Z', '150.0'))   # 높이 유지 강화(nose-dive 방지 핵심)
        P[:nv, :nv] += w_z * np.outer(Jc[2], Jc[2]); g[:nv] -= w_z * a_z * Jc[2]
        # posture — swing 관절엔 약한 규제(w=0.1)만: 발끝 3D task가 다리 DOF<4 면 여유도
        #   (4-DOF 발목)를 남기므로 null-space 규제 없으면 발목이 발산(flail). 약규제로 안정화.
        # ★능동 발목 flick(생체모방 paw-flick): 스윙 발 수직속도로 발목궤적 구동(상승→배굴/하강→저굴).
        #   반사관성(GEARBOX)·속도한계(VEL_LIM) 안에서 능동 whip = 실물 재현가능한 동물형 발끝 채찍.
        _flick = float(os.environ.get('ANKLE_FLICK', '0'))   # rad, 스윙 발목 능동 flick 진폭(0=off). ★위상기반: 스윙중 배굴→저굴 whip(walk·저속서도 확실)
        _fpow = float(os.environ.get('FLICK_POW', '1'))      # 프로파일 뾰족함(>1=착지쪽 저굴 snap 집중)
        q_ref = self.q_home if _flick == 0.0 else self.q_home.copy()
        if _flick != 0.0:
            for _lg, _sw in swing.items():
                if self.leg_dof[_lg] == 4:
                    _ph = _sw[2] if len(_sw) > 2 else 0.0     # 스윙 위상(0→1)
                    _aj = int(self.legqv[_lg][3]) - 6
                    # ★배굴(전반)→저굴(후반) whip. sin(π·ph) 윈도우로 양끝(liftoff·착지)서 위치·속도 0 → 착지 교란↓
                    _prof = np.sin(2*np.pi*_ph) * (np.sin(np.pi*_ph)**_fpow)
                    q_ref[_aj] = self.q_home[_aj] + _flick * float(_prof)
        a_post = 60 * (q_ref - d.qpos[7:7 + self.nu]) - 5 * qv[6:6 + self.nu]
        # ★발목 컴플라이언트 PD(path A 힘줄모방): kp↓=소프트스프링(뒤로끌렸다 튕김)·kd=감쇠. 기본 60/5(=강한 핀)
        _akp = float(os.environ.get('ANKLE_KP', '60')); _akd = float(os.environ.get('ANKLE_KD', '5'))
        if _akp != 60.0 or _akd != 5.0:
            _qe = q_ref - d.qpos[7:7 + self.nu]; _dqj = qv[6:6 + self.nu]
            for _aj in self._ankle_idx:
                a_post[_aj] = _akp * _qe[_aj] - _akd * _dqj[_aj]
        _pw = float(os.environ.get('POSTURE_W', '1.0'))      # ★stance 다리 posture 가중(계단서 ↓하면 다리 신장 자유=몸 상승)
        for j in range(self.nu):
            _sww = self._swing_w_f if j in self._front_idx else self._swing_w_r  # ★앞/뒤 whip 별도(GUI 슬라이더 live)
            if self._waist_idx is not None and j == self._waist_idx:   # ★허리: 강한 전용 홀드(요각목표=_waist_ref, 조향시 갱신)
                w_post = self._waist_w
                a_post[j] = self._waist_kp * (self._waist_ref - d.qpos[7 + j]) - self._waist_kd * qv[6 + j]
            elif j in self._ankle_idx and self._ankle_w > 0:   # 발목: REAR_ANKLE에 강하게 핀(여자유도 고정→대칭)
                w_post = self._ankle_w
            else:
                w_post = _sww if (6 + j) in sw_vidx else _pw
            P[6 + j, 6 + j] += w_post; g[6 + j] -= w_post * a_post[j]
        P[:nv, :nv] += 1e-3 * np.eye(nv)
        for k in range(K):           # λ tracking (확장 접촉점별, 선-발은 분할된 clam)
            P[sl(k), sl(k)] += w_lam * np.eye(3); g[sl(k)] -= w_lam * clam[k]
        # ★각운동량 보상(leg-heavy 고속): 총 centroidal 각운동량 h_ω 를 GRF 모멘트로 감쇠.
        #   Σ rᵢ×λᵢ ≈ −Kd·h_ω  (SRBD MPC가 무시하는 다리 swing 각운동량을 WBIC가 보상 → 고속 yaw/pitch 드리프트↓)
        _w_am = float(os.environ.get('W_AM', '12'))   # ★17dof 튜닝(2026-07-02): 5→12(각운동량 보상↑). 37.9kg 전발목 고속 yaw발산·외란tilt 억제 → push_tilt −16%. 구14dof는 0(평지)
        if _w_am > 0 and K > 0:
            mujoco.mj_subtreeVel(m, d)
            h_ang = d.subtree_angmom[0].copy()           # 총 각운동량 about CoM (world)
            hdes = -float(os.environ.get('KD_AM', '24')) * h_ang   # ★17dof 튜닝(2026-07-02): 8→24 각운동량 감쇠↑ → tilt_max·z_std 추가개선
            com = d.subtree_com[0]
            A_am = np.zeros((3, nz))
            for k in range(K):
                r = cpos[k] - com                        # 접촉점 위치 − CoM
                A_am[0, sl(k)] = [0.0, -r[2], r[1]]      # skew(r)
                A_am[1, sl(k)] = [r[2], 0.0, -r[0]]
                A_am[2, sl(k)] = [-r[1], r[0], 0.0]
            P += _w_am * (A_am.T @ A_am); g -= _w_am * (A_am.T @ hdes)
        Js = cjac                                        # 확장 접촉점 Jacobian들
        A = np.zeros((6, nz)); b = -h[0:6]; A[:, :nv] = M[0:6, :]
        for k, J in enumerate(Js):
            A[:, sl(k)] = -J[:, 0:6].T
        for J in Js:
            Ac = np.zeros((3, nz)); Ac[:, :nv] = J
            A = np.vstack([A, Ac]); b = np.concatenate([b, np.zeros(3)])
        # ★v13 ②식 발목 하드락(REAR_LOCK): 뒷발목 q̈를 home으로 임계감쇠 servo하는 등식제약.
        #   소프트 ANKLE_W(스프링,동역학과싸워 악화)와 달리 여유 발목을 null-space에서 제거→flail 차단.
        if os.environ.get('REAR_LOCK') and self._ankle_idx:
            _lkp = float(os.environ.get('LOCK_KP', '400')); _lkd = float(os.environ.get('LOCK_KD', '40'))
            for aj in self._ankle_idx:
                row = np.zeros(nz); row[6 + aj] = 1.0
                a_hold = _lkp * (self.q_home[aj] - d.qpos[7 + aj]) - _lkd * qv[6 + aj]
                A = np.vstack([A, row]); b = np.concatenate([b, [a_hold]])
        lb = np.full(nz, -1e8); ub = np.full(nz, 1e8); Gl = []; hl = []
        # ★관절 위치 한계(구조3 kinematics_limits 차용): q(T_la)=q+dq·T_la+½q̈·T_la² ∈ [qmin,qmax]
        #   → q̈ 경계. lookahead T_la로 한계 전 부드럽게 감속. 모순(이미 한계초과)이면 relax.
        if os.environ.get('POS_LIM', '1') != '0':
            _tla = float(os.environ.get('POS_TLA', '0.05'))   # 위치한계 lookahead[s]
            qj = d.qpos[7:7 + self.nu]; dqj = qv[6:6 + self.nu]; _c = 0.5 * _tla * _tla
            ub_p = (self._qmax - qj - dqj * _tla) / _c; lb_p = (self._qmin - qj - dqj * _tla) / _c
            for j in range(self.nu):
                _u = min(ub[6 + j], ub_p[j]); _l = max(lb[6 + j], lb_p[j])
                if _l <= _u: ub[6 + j] = _u; lb[6 + j] = _l   # 일관시만 적용(infeasible 방지)
        # ★각속도 한계 QP제약(controller-aware, sim2real): ω+q̈·T_la ∈ [−ωlim,ωlim] → q̈ 경계. POS_LIM과 동일구조.
        if os.environ.get('VEL_LIM'):
            _tlav = float(os.environ.get('VEL_TLA', '0.02'))   # 속도한계 lookahead[s]
            dqv = qv[6:6 + self.nu]
            ub_v = (self._w_limit - dqv) / _tlav; lb_v = (-self._w_limit - dqv) / _tlav
            for j in range(self.nu):
                _u = min(ub[6 + j], ub_v[j]); _l = max(lb[6 + j], lb_v[j])
                if _l <= _u: ub[6 + j] = _u; lb[6 + j] = _l   # 일관시만(infeasible 방지)
        # ★저크 제한 QP제약(controller-aware): |q̈ − q̈_prev| ≤ Δmax → q̈∈[q̈_prev±Δ]. jerk≈Δq̈/dt 직접 제한.
        if os.environ.get('JERK_LIM'):
            _dmax = float(os.environ.get('JERK_DDQ', '300'))   # 스텝당 q̈ 변화 한계[rad/s²](jerk_max·dt_ctrl)
            qp_ = self._qdd_prev
            for j in range(self.nu):
                _u = min(ub[6 + j], qp_[j] + _dmax); _l = max(lb[6 + j], qp_[j] - _dmax)
                if _l <= _u: ub[6 + j] = _u; lb[6 + j] = _l   # 일관시만(infeasible 방지)
        for k in range(K):
            o = nv + 3 * k; lb[o + 2] = LAMZ_MIN
            for sx, sy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                row = np.zeros(nz); row[o] = sx; row[o + 1] = sy; row[o + 2] = -MU * MU_MARGIN
                Gl.append(row); hl.append(0.0)
        P = 0.5 * (P + P.T) + 1e-8 * np.eye(nz)
        # ★토크 = M_act·q̈ + h_act − Σ Jᵀλ (z의 선형식 τ=T_mat·z+h_act). 한계(제약) + effort최소화(비용)
        _tau_lim = os.environ.get('TAU_LIM', '1') != '0'
        _w_tau = float(os.environ.get('W_TAU', '0'))        # 토크 effort 최소화 가중(기본0=off; 0.001~0.01 권장)
        _tau_rate = os.environ.get('TAU_RATE')              # 토크 변화율 한계[Nm/step](기어박스 응력; 저크 직접)
        if _tau_lim or _w_tau > 0 or _tau_rate:
            h_act = h[6:6 + self.nu]
            T_mat = np.zeros((self.nu, nz)); T_mat[:, :nv] = M[6:6 + self.nu, :]
            for k, J in enumerate(Js):
                T_mat[:, sl(k)] = -J[:, 6:6 + self.nu].T
            if _w_tau > 0:                                  # min ||τ||²: 여러 해 중 토크 작은 해 선택
                P += _w_tau * (T_mat.T @ T_mat); g += _w_tau * (T_mat.T @ h_act)
            if _tau_lim:                                    # per-joint 토크 한계 −τ_peak ≤ τ ≤ τ_peak
                Gl.extend(list(T_mat));  hl.extend(list(self._tau_peak - h_act))
                Gl.extend(list(-T_mat)); hl.extend(list(self._tau_peak + h_act))
            if _tau_rate:                                   # |τ−τ_prev|≤Δτ : T_mat·z ∈ [τ_prev−Δ−h, τ_prev+Δ−h]
                _dtau = float(_tau_rate)
                Gl.extend(list(T_mat));  hl.extend(list(self._tau_prev + _dtau - h_act))
                Gl.extend(list(-T_mat)); hl.extend(list(-(self._tau_prev - _dtau) + h_act))
        G = np.vstack(Gl) if Gl else None; hh = np.array(hl) if hl else None
        if self._wbt: import time as _tm9; _tq9 = _tm9.perf_counter()
        z = solve_qp(P, g, G, hh, A, b, lb, ub, solver='quadprog')
        if self._wbt:
            self._qpt.append(_tm9.perf_counter() - _tq9)
            if len(self._qpt) % 1000 == 0:
                _a9 = np.array(self._qpt[-1000:]); print('[WBIC_QP] solve만 평균%.2fms 최대%.2fms (%.0fHz 가능)' % (_a9.mean()*1000, _a9.max()*1000, 1/_a9.mean()), flush=True)
        if z is None:
            self.last_lam = None; return None, False
        qdd = z[:nv]; lam = [z[sl(k)] for k in range(K)]
        tau = M[6:6 + self.nu, :] @ qdd + h[6:6 + self.nu]
        for k, J in enumerate(Js):
            tau -= J[:, 6:6 + self.nu].T @ lam[k]
        self.last_lam = lam
        self._qdd_prev = qdd[6:6 + self.nu].copy(); self._tau_prev = tau.copy()   # JERK_LIM/TAU_RATE 상태
        # ★토크 한계: 기본=Peak 상수클립. MOTOR_CURVE면 토크-속도 곡선(고속서 가용토크↓=실모터)
        if self._motor_curve:
            _w = d.qvel[6:6 + self.nu]
            _avail = self._tau_peak * np.maximum(0.0, 1.0 - np.abs(_w) / self._w_limit)
            d.ctrl[:] = np.clip(tau, -_avail, _avail)
        else:
            d.ctrl[:] = np.clip(tau, -self._tau_peak, self._tau_peak)   # per-joint Peak 클립(QP가 이미 존중)
        if self._tau_lpf > 0:                               # ★출력단 토크 LPF(QP밖,1차): 1kHz 계단성분 평활(공진억제). 위상지연=안정성 트레이드오프
            _al = 2 * np.pi * self._tau_lpf * self.m.opt.timestep; _al /= (1 + _al)
            self._tau_filt = _al * d.ctrl[:self.nu] + (1 - _al) * self._tau_filt
            d.ctrl[:self.nu] = self._tau_filt
        return tau, True

    def wbic_jump(self, com_ref, comv_ref, acom_ref, q_ref, dq_ref, contacts,
                  kp_lin=120.0, kd_lin=22.0, kp_j=160.0, kd_j=12.0,
                  w_lin=120.0, w_ori=8.0, w_j=2.0, w_lam=0.1):
        """표준 trajectory-tracking WBC — offline 점프궤적을 GRF로 닫아 추종.
           단일 QP z=[q̈; λ]: ①CoM 3-DOF(피드포워드 acom_ref + PD → GRF가 폭발푸시
           실현 + 수평표류 억제) ②base 자세 upright ③관절추종(q_ref/dq_ref).
           contacts=현 위상 디딘발(load/push/land/recover=4, flight=()).
           open-loop(피드포워드+관절PD)와 달리 base를 닫아 표류 제거 = MIT-Cheetah류."""
        d, m, nv, nu = self.d, self.m, self.nv, self.nu
        M = self.fullM(); h = d.qfrc_bias.copy(); qv = d.qvel.copy()
        K = len(contacts); nz = nv + 3 * K
        sl = lambda k: slice(nv + 3 * k, nv + 3 * k + 3)
        P = np.zeros((nz, nz)); g = np.zeros(nz)
        # ① CoM 3-DOF 추종 (접촉 있을 때만; flight=탄도라 제어 불가)
        if K > 0:
            Jc = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
            com = d.subtree_com[0]; comv = Jc @ qv
            a_lin = acom_ref + kp_lin * (com_ref - com) + kd_lin * (comv_ref - comv)
            P[:nv, :nv] += w_lin * (Jc.T @ Jc); g[:nv] -= w_lin * (Jc.T @ a_lin)
        # ② base 자세 upright (점프는 수평 유지)
        oerr = np.zeros(3); mujoco.mju_quat2Vel(oerr, d.qpos[3:7], 1.0)
        a_ori = 150 * (-oerr) - 20 * qv[3:6]
        for j in range(3):
            P[3 + j, 3 + j] += w_ori; g[3 + j] -= w_ori * a_ori[j]
        # ③ 관절 추종 (q_ref/dq_ref — 발목 포함 전관절 → 여자유도 flail 차단)
        a_j = kp_j * (q_ref - d.qpos[7:7 + nu]) + kd_j * (dq_ref - qv[6:6 + nu])
        for j in range(nu):
            P[6 + j, 6 + j] += w_j; g[6 + j] -= w_j * a_j[j]
        P[:nv, :nv] += 1e-3 * np.eye(nv)
        for k in range(K):
            P[sl(k), sl(k)] += w_lam * np.eye(3)
        # EOM(floating-base 6) + 접촉 무가속(등식)
        Js = [self.foot_jac(c) for c in contacts]
        A = np.zeros((6, nz)); b = -h[0:6]; A[:, :nv] = M[0:6, :]
        for k, J in enumerate(Js):
            A[:, sl(k)] = -J[:, 0:6].T
        for J in Js:
            Ac = np.zeros((3, nz)); Ac[:, :nv] = J
            A = np.vstack([A, Ac]); b = np.concatenate([b, np.zeros(3)])
        # 마찰추(부등식) + λz 하한
        lb = np.full(nz, -1e8); ub = np.full(nz, 1e8); Gl = []; hl = []
        for k in range(K):
            o = nv + 3 * k; lb[o + 2] = LAMZ_MIN
            for sx, sy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                row = np.zeros(nz); row[o] = sx; row[o + 1] = sy; row[o + 2] = -MU * MU_MARGIN
                Gl.append(row); hl.append(0.0)
        P = 0.5 * (P + P.T) + 1e-8 * np.eye(nz)
        # per-joint 토크 한계(부등식): τ=M_act·q̈+h_act−ΣJᵀλ ∈ [±τ_peak]
        h_act = h[6:6 + nu]
        T_mat = np.zeros((nu, nz)); T_mat[:, :nv] = M[6:6 + nu, :]
        for k, J in enumerate(Js):
            T_mat[:, sl(k)] = -J[:, 6:6 + nu].T
        Gl.extend(list(T_mat));  hl.extend(list(self._tau_peak - h_act))
        Gl.extend(list(-T_mat)); hl.extend(list(self._tau_peak + h_act))
        G = np.vstack(Gl) if Gl else None; hh = np.array(hl) if hl else None
        z = solve_qp(P, g, G, hh, A, b, lb, ub, solver='quadprog')
        if z is None:
            return None, False
        qdd = z[:nv]; lam = [z[sl(k)] for k in range(K)]
        tau = M[6:6 + nu, :] @ qdd + h[6:6 + nu]
        for k, J in enumerate(Js):
            tau -= J[:, 6:6 + nu].T @ lam[k]
        if self._motor_curve:
            _w = qv[6:6 + nu]; _avail = self._tau_peak * np.maximum(0.0, 1.0 - np.abs(_w) / self._w_limit)
            d.ctrl[:] = np.clip(tau, -_avail, _avail)
        else:
            d.ctrl[:] = np.clip(tau, -self._tau_peak, self._tau_peak)
        return tau, True

    # ── 제어 ──────────────────────────────────────────
    def set_home(self):
        mujoco.mj_forward(self.m, self.d)
        self.q_home = self.d.qpos[7:7 + self.nu].copy()

    def pd_grav(self):
        """중력보상 + home 자세 PD → ctrl."""
        q = self.d.qpos[7:7 + self.nu]
        qd = self.d.qvel[6:6 + self.nu]
        grav = self.d.qfrc_bias[6:6 + self.nu]
        tau = grav + KP * (self.q_home - q) - KD * qd
        self.d.ctrl[:] = np.clip(tau, -TAU_MAX, TAU_MAX)

    # ── 발 GRF (MuJoCo 접촉에서) ───────────────────────
    def foot_grf(self):
        """각 발의 world GRF (3,) — 시각화용. 반환 list[4] of (3,) or None."""
        d, m = self.d, self.m
        grf = [np.zeros(3) for _ in range(4)]
        hit = [False] * 4
        for c in range(d.ncon):
            con = d.contact[c]
            g1, g2 = con.geom1, con.geom2
            bod = None
            for gid in (g1, g2):
                bb = m.geom_bodyid[gid]
                if bb in self.foot_bid:
                    bod = self.foot_bid.index(bb)
            if bod is None:
                continue
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, c, f6)
            # contact frame → world (con.frame 는 3x3 row-major, x축=법선)
            R = con.frame.reshape(3, 3)
            fw = R.T @ f6[:3]
            grf[bod] += fw
            hit[bod] = True
        return [grf[i] if hit[i] else None for i in range(4)]

    # ── 발끝 접촉 감지 (Di Carlo 2018 MIT Cheetah 3 식) ──────────
    def foot_fz(self):
        """각 발 추정 수직 접촉력 fz (force-sensor proxy = MuJoCo 접촉력).
        실하드웨어: 발 힘센서 또는 관절토크 기반 추정기(τ→Jᵀ⁻¹)로 대체."""
        grf = self.foot_grf()
        return np.array([g[2] if g is not None else 0.0 for g in grf])

    def detect_contact(self, scheduled, fz_on=10.0, fz_off=4.0, z_thr=0.03, alpha=0.6):
        """발끝 접촉 감지 — 주기 gait 스케줄(Di Carlo) + 측정 융합 + 히스테리시스.

        scheduled[i] : gait 가 예측한 stance 여부 (primary, Di Carlo periodic schedule).
        측정 융합     : 수직 GRF fz(저역통과) + 발끝 높이 z 로 조기/지연 착지 보정.
        히스테리시스 : 접촉중이면 낮은 임계(fz_off)로 유지, 비접촉이면 높은 임계(fz_on)로
                      진입 → 임계 근처 채터링(출렁임) 방지.
        반환: contact(4,) bool, fz_filt(4,)."""
        if not hasattr(self, '_fz_filt'):
            self._fz_filt = np.zeros(4)
        self._fz_filt = alpha * self._fz_filt + (1 - alpha) * self.foot_fz()
        contact = []
        for i in range(4):
            near = self.foot_point(i)[2] < z_thr
            thr = fz_off if self.contact_state[i] else fz_on   # 히스테리시스
            loaded = self._fz_filt[i] > thr
            c = (loaded or near) if scheduled[i] else (loaded and near)
            contact.append(bool(c))
        self.contact_state = contact
        return contact, self._fz_filt.copy()

    # ── 뷰어 오버레이 ─────────────────────────────────
    def _key_callback(self, keycode):
        pass                                            # 시각화는 항상 표시(구조3 스타일). 토글 없음.

    def draw_overlay(self, v):
        # ★구조3(quad_fulldynamics.py) 스타일로 시각화 통일:
        #   빨강구=타겟footstep(swing) / 청록선=지지다각형 / 노랑구+선=CoM지면투영 / 노랑화살표=명령방향
        #   마젠타=base궤적 / 파란콘+초록화살표=마찰콘+GRF / 발별색선=발궤적
        d = self.d; m = self.m
        scn = v.user_scn; scn.ngeom = 0; eye = np.eye(3).flatten()
        if not self._viz:                                    # ★모니터 표시 OFF(GUI 토글): overlay 전부 끔, 발색 복원
            for i in range(4):
                for g in self.foot_geoms[i]:
                    self.m.geom_rgba[g] = self._foot_rgba0[g]
            return
        def _sph(p, r, c):
            if scn.ngeom >= scn.maxgeom: return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([r, 0, 0]), np.asarray(p, float), eye, np.asarray(c, np.float32))
            scn.ngeom += 1
        def _ln(a, b, w, c, typ=mujoco.mjtGeom.mjGEOM_LINE):
            if scn.ngeom >= scn.maxgeom: return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, typ, np.zeros(3), np.zeros(3), eye, np.asarray(c, np.float32))
            mujoco.mjv_connector(g, typ, w, np.asarray(a, float), np.asarray(b, float)); scn.ngeom += 1
        # ── ★elevation map 시각화(모니터 토글에 포함): 로봇중심 격자를 높이별 색(파랑낮음→빨강높음). 100ms캐시 ──
        if self._terrain_on:
            if self._elev_cache is None or d.time - self._elev_t > 0.1 or d.time < self._elev_t:
                _bx, _by = d.qpos[0], d.qpos[1]; _cc = []
                for _gx in np.arange(_bx - 0.3, _bx + 1.05, 0.12):
                    for _gy in np.arange(_by - 0.36, _by + 0.37, 0.12):
                        _cc.append((float(_gx), float(_gy), self.terrain_height(_gx, _gy)))
                self._elev_cache = _cc; self._elev_t = d.time
            for _gx, _gy, _h in self._elev_cache:
                _t = min(1.0, max(0.0, _h / 0.30))
                _sph([_gx, _gy, _h + 0.012], 0.022, [_t, 0.25, 1 - _t, 0.7])
        ZC = 0.02                                            # 발 접지판별(발끝 z<ZC=접지)
        fp = [self.foot_point(i).copy() for i in range(4)]; fz = [p[2] for p in fp]
        # 발 접촉링: 접지면 빨강
        RED = np.array([0.9, 0.1, 0.1, 1.0], np.float32)
        for i in range(4):
            for g in self.foot_geoms[i]:
                self.m.geom_rgba[g] = RED if fz[i] < ZC else self._foot_rgba0[g]
        # ── 타겟 footstep(swing 발만, 빨강구) ──
        for i in range(4):
            ft = self.foot_targets[i]
            if ft is not None and fz[i] > ZC:
                _sph([ft[0], ft[1], 0.008], 0.012, [1, 0.1, 0.1, 0.9])
        # ── 지지다각형(접지 발 연결, 청록 지면선) ──
        _ord = [2, 3, 1, 0]                                  # FL,FR,HR,HL 둘레순(quad_mpc_wbic legs=HL,HR,FL,FR)
        sp = [fp[i] for i in _ord if fz[i] < ZC]
        for k in range(len(sp)):
            if len(sp) < 2: break
            a = sp[k].copy(); a[2] = 0.003; b = sp[(k + 1) % len(sp)].copy(); b[2] = 0.003
            _ln(a, b, 0.006, [0.1, 0.9, 0.9, 1])
        # ── CoM 지면투영(노랑구+수직선) ──
        com = d.subtree_com[0].copy()
        _sph([com[0], com[1], 0.004], 0.020, [1, 0.9, 0.1, 0.95]); _ln([com[0], com[1], 0.0], com, 0.003, [1, 0.9, 0.1, 0.6])
        # ── 명령방향 화살표(로봇 위 노랑) ──
        cv = self.cmd_v
        if float(np.hypot(cv[0], cv[1])) > 1e-3:
            frm = d.qpos[0:3].copy() + np.array([0, 0, 0.20]); to = frm + np.array([cv[0], cv[1], 0.0]) * 0.4
            _ln(frm, to, 0.015, [1, 0.85, 0.1, 1], mujoco.mjtGeom.mjGEOM_ARROW)
        # ── base 궤적(마젠타·굵게) ──
        self.base_trail.append(d.qpos[0:3].copy()); bp = self.base_trail
        for k in range(1, len(bp)):
            if np.linalg.norm(bp[k] - bp[k - 1]) < 1e-4: continue
            _ln(bp[k - 1], bp[k], 0.010, [1, 0.15, 0.9, 1])
        # ── 마찰콘(파랑) + GRF(초록 화살표): GRF가 콘 벗어나면 슬립 ──
        mu = float(os.environ.get('CONE_MU', str(m.geom_friction[self.foot_gid[0]][0])))
        hh = 0.10; Ncn = 8
        if not os.environ.get('NOCONE'):
            for i in range(d.ncon):
                c = d.contact[i]
                if c.geom1 not in self.foot_gid and c.geom2 not in self.foot_gid: continue
                p = c.pos.copy()
                for k in range(Ncn):
                    a1 = 2 * np.pi * k / Ncn; a2 = 2 * np.pi * (k + 1) / Ncn
                    r1 = np.array([np.cos(a1), np.sin(a1), 0]) * hh * mu + np.array([0, 0, hh])
                    r2 = np.array([np.cos(a2), np.sin(a2), 0]) * hh * mu + np.array([0, 0, hh])
                    _ln(p, p + r1, 0.0015, [0.3, 0.5, 1, 0.5]); _ln(p + r1, p + r2, 0.0015, [0.3, 0.5, 1, 0.5])
                f6 = np.zeros(6); mujoco.mj_contactForce(m, d, i, f6)
                fw = c.frame.reshape(3, 3).T @ f6[:3]
                if fw[2] < 0: fw = -fw
                mag = np.linalg.norm(fw)
                if mag > 1.0: _ln(p, p + fw / mag * min(mag / 250.0, 0.15), 0.008, [0.1, 1, 0.2, 1], mujoco.mjtGeom.mjGEOM_ARROW)
        # ── 발 궤적(발별 색선) ──
        _tc = [[0.2, 0.6, 1, 1], [0.2, 1, 0.4, 1], [1, 0.6, 0.2, 1], [1, 0.3, 0.85, 1]]
        for i in range(4):
            self.foot_trail[i].append(fp[i].copy()); pts = self.foot_trail[i]
            for k in range(1, len(pts)):
                if np.linalg.norm(pts[k] - pts[k - 1]) < 1e-4: continue
                _ln(pts[k - 1], pts[k], 0.004, _tc[i % 4])

    def publish_state(self, path):
        """구조3와 동일 스키마로 상태 발행(원자적) → teleop_gui 모니터 패널(IMU/Actuator)."""
        d, m = self.d, self.m
        if not hasattr(self, '_jnames'):                  # q/dq 순서와 일치(MuJoCo 관절순)
            self._jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).replace('_joint', '')
                            for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, d.qpos[3:7]); Rm = Rm.reshape(3, 3)
        rpy = [math.degrees(math.atan2(Rm[2, 1], Rm[2, 2])),
               math.degrees(math.asin(max(-1, min(1, -Rm[2, 0])))),
               math.degrees(math.atan2(Rm[1, 0], Rm[0, 0]))]
        # ★전진 실제속도(heading 방향, EMA 필터) vs 명령속도 — cmd/actual 차이 모니터
        _yaw = math.radians(rpy[2])
        _vf = float(d.qvel[0] * math.cos(_yaw) + d.qvel[1] * math.sin(_yaw))
        self._vf_filt = 0.97 * getattr(self, '_vf_filt', _vf) + 0.03 * _vf
        st = {'mode': self.cmd_mode, 'base_z': float(d.qpos[2]), 't': float(d.time),
              'rpy': rpy, 'gyro': [float(x) for x in d.qvel[3:6]], 'names': self._jnames,
              'q': [float(x) for x in d.qpos[7:7 + self.nu]],
              'dq': [float(x) for x in d.qvel[6:6 + self.nu]],
              'tau': [float(x) for x in d.ctrl[:self.nu]],
              'v_cmd': float(self.cmd_v[0]), 'v_act': self._vf_filt}   # 명령/실제 전진속도[m/s]
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(st, f)
        os.replace(tmp, path)

    def run_viewer(self, control_fn, reset_on_fall=True, reset_fn=None):
        m, d = self.m, self.d
        reset_fn = reset_fn or self.set_home
        _sp = os.environ.get('STATE_PUB'); _pc = 0   # 상태발행 채널 + 프레임카운터
        if os.environ.get('HEADLESS'):                       # 헤드리스 정량테스트(뷰어 없이 N스텝)
            nsteps = int(os.environ.get('STEPS', '1500'))
            pe = int(os.environ.get('PRINT_EVERY', '100'))   # 출력 간격(스텝). 첫사이클 관찰은 20 등
            falls = 0
            _logj = os.environ.get('LOG_JOINTS')             # ★관절 각속도·토크 로깅 → npz(그래프용)
            _Lt, _Ldq, _Ltau, _Lq, _Lquat, _Lse, _Lfho = [], [], [], [], [], [], []
            _push = os.environ.get('PUSH')                    # ★외란 테스트: 지정시각에 base에 측방 임펄스
            _pf = float(os.environ.get('PUSH_F', '150')); _pt = float(os.environ.get('PUSH_T', '8'))
            _pdur = float(os.environ.get('PUSH_DUR', '0.1')); _maxtilt = 0.0
            for s in range(nsteps):
                if _push:                                     # 측방(y) 힘 주입 후 해제
                    d.xfrc_applied[1, 1] = _pf if _pt <= d.time < _pt + _pdur else 0.0
                control_fn(); mujoco.mj_step(m, d)
                if _push and d.time > _pt:
                    _x, _y = d.qpos[4], d.qpos[5]; _maxtilt = max(_maxtilt, np.degrees(np.arccos(max(-1, min(1, 1 - 2 * (_x*_x + _y*_y))))))
                if _logj:
                    _Lt.append(d.time); _Ldq.append(d.qvel[6:6+self.nu].copy()); _Ltau.append(d.ctrl[:self.nu].copy())
                    _Lq.append(d.qpos[7:7+self.nu].copy()); _Lquat.append(d.qpos[3:7].copy())
                    _Lse.append([getattr(self, '_swing_err', 0.0), getattr(self, '_swing_errh', 0.0)])  # swing 추종오차(전체·수평)
                    _Lfho.append(getattr(self, '_fho', [np.nan]*4))   # foothold-hip x offset(전진+) 4발
                if reset_on_fall and d.qpos[2] < 0.2:
                    falls += 1; mujoco.mj_resetData(m, d); reset_fn()
                if _sp and s % 30 == 0: self.publish_state(_sp)   # GUI 모니터 패널
                if s % pe == 0:
                    w, x, y, z = d.qpos[3:7]                  # base quat [w,x,y,z] → tilt(수직과의 각)
                    tilt = np.degrees(np.arccos(max(-1, min(1, 1 - 2 * (x * x + y * y)))))
                    pen = [0.0] * 4                           # 발별 접촉침투(mm, 음수=파고듦)
                    for ci in range(d.ncon):
                        c = d.contact[ci]
                        for fi, g in enumerate(self.foot_gid):
                            if c.geom1 == g or c.geom2 == g:
                                pen[fi] = min(pen[fi], c.dist)
                    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                    pr = min(pen[0], pen[1]); pf = min(pen[2], pen[3])  # 뒤/앞 최대침투
                    _ext = ''
                    if self._stair is not None:               # 계단: 지형높이·clearance(몸-지면 간격, 일정해야 정상 등반)
                        _ext = ' terr=%.2f clr=%.2f' % (getattr(self, '_dbg_terr', 0.0), getattr(self, '_dbg_clr', 0.0))
                    print('[hl] s=%d t=%.2f z=%.3f x=%+.3f y=%+.3f yaw=%+.0f tilt=%.1f 침투뒤/앞=%.1f/%.1fmm falls=%d%s'
                          % (s, d.time, d.qpos[2], d.qpos[0], d.qpos[1], yaw, tilt, pr*1000, pf*1000, falls, _ext), flush=True)
            print('[hl] 종료: %d스텝 falls=%d 최종 x=%+.3f%s' % (nsteps, falls, d.qpos[0],
                  (' push후최대tilt=%.1f°' % _maxtilt) if _push else ''), flush=True)
            if _logj:
                _names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ('act%d'%i) for i in range(self.nu)]
                np.savez(_logj, t=np.array(_Lt), dq=np.array(_Ldq), tau=np.array(_Ltau), names=np.array(_names),
                         q=np.array(_Lq), quat=np.array(_Lquat), swerr=np.array(_Lse), fho=np.array(_Lfho))
                print('[hl] 관절로그 저장: %s (%d스텝 %d관절)' % (_logj, len(_Lt), self.nu), flush=True)
            return
        with mujoco.viewer.launch_passive(m, d, key_callback=self._key_callback) as v:
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 1     # 외란 박스 ON
            _re = int(os.environ.get('RENDER_EVERY', '10'))     # 렌더당 물리스텝(1kHz라 매스텝 렌더는 과함→10=100Hz)
            _rate = float(os.environ.get('RATE', '1.0'))        # ★재생 배속(2=2배빠르게, 0=최대속도/sleep없음)
            print('viewer open — 창 닫으면 종료. (더블클릭+Ctrl+우드래그=외란) | RATE=%.1f배 RENDER_EVERY=%d' % (_rate, _re))
            while v.is_running():
                t0 = time.time()
                for _ in range(_re):                            # 물리 _re스텝 후 1회 렌더(빠르고 부드럽게)
                    control_fn()
                    mujoco.mjv_applyPerturbForce(m, d, v.perturb)
                    mujoco.mj_step(m, d)
                    if reset_on_fall and d.qpos[2] < 0.2:
                        mujoco.mj_resetData(m, d); reset_fn()
                _pc += 1
                if _sp and _pc % 3 == 0: self.publish_state(_sp)    # GUI 모니터 패널(~30Hz)
                self.draw_overlay(v)                                # (내부서 self._viz 체크해 overlay on/off)
                # 좌상단: 시뮬 시간 / 우상단: 외란 힘 N
                fext = max((float(np.linalg.norm(d.xfrc_applied[b, :3]))
                            for b in range(1, m.nbody)), default=0.0)
                cv = self.cmd_v
                _yw = math.atan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]),
                                 1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
                _vact = float(d.qvel[0] * math.cos(_yw) + d.qvel[1] * math.sin(_yw))  # 실제 전진속도(heading투영)
                v.set_texts([                                    # 구조3와 동일: 좌상=시간 우상=외력 좌하=명령/실제
                    (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     'sim time', '%.2f s' % d.time),
                    (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                     'ext force', '%.0f N' % fext),
                    (mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     'cmd vx/vy/wz\nactual vx', '%+.2f %+.2f %+.2f\n%+.2f m/s' % (cv[0], cv[1], cv[5], _vact))])
                v.sync()
                if self._rate > 0:                              # ★live 배속(GUI). RATE=0이면 sleep없이 최대속도
                    dt = _re * m.opt.timestep / self._rate - (time.time() - t0)
                    if dt > 0:
                        time.sleep(dt)


# ══════════════════════════════════════════════════════════════
# 모드  (check 단계는 stance/WBIC가 동작하면 자동 검증되므로 별도 두지 않음)
# ══════════════════════════════════════════════════════════════
def mode_view():
    """정적 기립 (physics 정지) — 자세/모델 시각 확인."""
    q = QuadSim(); q.set_home()
    q.run_viewer(lambda: mujoco.mj_forward(q.m, q.d), reset_on_fall=False)


def mode_inspect():
    """관절 가동범위 확인 — 중력 OFF + 제어 OFF + 베이스 고정.
    뷰어에서 발/링크를 **Ctrl+드래그**하면 관절이 움직여 한계(joint limit)가 보임."""
    q = QuadSim(); q.crouch_home()
    q.m.opt.gravity[:] = 0.0                      # 중력 끔 (안 떨어짐)
    base0 = q.d.qpos[0:7].copy()
    # 관절 가동범위 출력 (구동 hinge 관절 전체)
    print('=== 관절 가동범위 ===')
    for j in range(q.m.njnt):
        if q.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        nm = mujoco.mj_id2name(q.m, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = q.m.jnt_range[j]
        print('  %-18s : [%+.2f, %+.2f] (%.0f~%.0f°)' %
              (nm, lo, hi, np.degrees(lo), np.degrees(hi)))

    def ctrl():
        q.d.ctrl[:] = 0                           # 관절 토크 0 (자유, 드래그로 가동)
        q.d.qpos[0:7] = base0; q.d.qvel[0:6] = 0  # 베이스 고정 (드리프트 방지)
    q.run_viewer(ctrl, reset_on_fall=False)


def mode_stand():
    """중력 하 PD(+중력보상) 기립 (제어 sanity)."""
    q = QuadSim(); q.set_home()
    q.run_viewer(q.pd_grav)


def mode_stance():
    """1단계 — WBIC 단독 정지 균형 (크라우치 자세, 4발 접촉정합 제어)."""
    q = QuadSim(); q.crouch_home()
    print('크라우치 base_z=%.3f  com_ref=%s' % (q.d.qpos[2], np.round(q.com_ref, 3)))
    q.run_viewer(q.wbic_stance, reset_fn=q.crouch_home)


def mode_mpc():
    """3단계 — MPC(Linear Convex) + WBIC 연동. 현재: 정지(전부 stance) GRF 계획→추종."""
    q = QuadSim(); q.crouch_home(); q.setup_mpc()
    x_ref = q.body_x0().copy()        # 현재 자세를 목표로 (정지 유지)
    lam0 = q.mpc_grf(x_ref)
    print('MPC Σλz=%.1f N (무게 %.1f)' %
          (lam0[:, 2].sum(), q.m.body_subtreemass[0] * 9.81))

    def ctrl():
        q.wbic_track(q.mpc_grf(x_ref))

    def reset():
        q.crouch_home()
    q.run_viewer(ctrl, reset_fn=reset)


def mode_walk():
    """3.b단계 — MPC+WBIC 정적 크롤 보행 (제자리). [개발중 — 뷰어 판단용]
    각 발 슬롯: 전반 CoM를 나머지3발 중심으로 이동(4발지지) → 후반 그 발 swing."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        from gait_sim.gait import swing_foot_pos
    finally:
        sys.argv = _a
    q = QuadSim(); q.crouch_home(); q.setup_mpc()
    T_GAIT, ORDER, SHIFT_FRAC, STEP_H = 2.4, [3, 0, 2, 1], 0.55, 0.05
    KP_SW, KD_SW = 40.0, 2.0
    state = {'nominal': [q.foot_point(i).copy() for i in range(4)],
             'liftoff': [q.foot_point(i).copy() for i in range(4)],
             'x0': q.body_x0().copy()}

    def schedule(t):
        ph = (t % T_GAIT) / T_GAIT; slot = int(ph * 4) % 4; local = (ph * 4) % 1.0
        leg = ORDER[slot]
        if local < SHIFT_FRAC:
            return leg, local / SHIFT_FRAC, [0, 1, 2, 3], []
        return leg, (local - SHIFT_FRAC) / (1 - SHIFT_FRAC), \
            [j for j in range(4) if j != leg], [leg]

    def csched(t):
        cs = np.zeros((q.N_MPC, 4), dtype=bool)
        for k in range(q.N_MPC):
            _, _, st, _ = schedule(t + k * q.MPC.DT_MPC)
            for i in st:
                cs[k, i] = True
        return cs

    def ctrl():
        t = q.d.time
        leg, s, st, sw = schedule(t)
        if sw and s < 0.02:
            state['liftoff'][leg] = q.foot_point(leg).copy()
        tri = [j for j in range(4) if j != leg]
        cen = np.mean([q.foot_point(j)[:2] for j in tri], axis=0)
        x_ref = state['x0'].copy(); x_ref[3] = cen[0]; x_ref[4] = cen[1]
        q.wbic_track(q.mpc_grf(x_ref, csched(t)), contacts=tuple(st))
        for i in sw:
            p_tgt = swing_foot_pos(s, state['liftoff'][i], state['nominal'][i],
                                   np.zeros(3), step_height=STEP_H, tau_land=1.0)
            qt = q.ik_leg(i, p_tgt, q.d.qpos[q.legqp[i]])
            q.d.ctrl[i * q.dof:(i + 1) * q.dof] = np.clip(
                KP_SW * (qt - q.d.qpos[q.legqp[i]]) - KD_SW * q.d.qvel[q.legqv[i]], -TAU_MAX, TAU_MAX)

    q.run_viewer(ctrl, reset_fn=q.crouch_home)


def mode_trot():
    """3.b단계 — MPC+WBIC trot 보행 (제자리). [개발중 — 뷰어 판단용]
    대각쌍(HL+FR / HR+FL) 교대. baseline 방식: x_ref=속도/자세/높이 추종(위치 안박음),
    MPC_Q 자세가중치↑. 동적 게이트라 roll 균형이 난관(튜닝 필요)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _a = sys.argv; sys.argv = [_a[0]]
    try:
        from gait_sim.gait import swing_foot_pos
    finally:
        sys.argv = _a
    q = QuadSim()
    _tz = os.environ.get('TROT_Z')           # trot crouch 높이 override(낮추면 CoM↓ roll안정↑)
    q.crouch_home(float(_tz) if _tz else None); q.setup_mpc()
    # ★py(world y위치) 가중치=0 — x와 대칭으로 자유. (과거 100=직진 y앵커였으나 측방/대각·헤딩추종을 막음;
    #   위치 드리프트는 속도추종 vx/vy=10이 잡음). global y앵커 제거 → 선회 후 전진이 헤딩 따라감.
    TROT_Q = np.diag([200., 200., 100., 0., float(os.environ.get('WPY', '0.')), 200., 0., 0., 1., 10., 10., 1., 0.])
    # ★게이트 프리셋(gait_sim_v13 참조). 다리순서 [HL,HR,FL,FR]. OFFSET=위상오프셋(swing타이밍),
    #   T=주기[s], SWF=swing비율(D), STEPH=발높이, V=기본속도.
    #   trot=대각쌍(HL+FR/HR+FL) 동적·2지지 / walk=순차(FR→HL→FL→HR) 정적안정·75%stance(3~4지지)
    GAIT = os.environ.get('GAIT', 'trot')
    GAITS = {       # ★게이트 프리셋(gait_sim_v13 참조). 다리순서[HL,HR,FL,FR]. GUI gait 토글로 라이브 전환
        # LOCK=평지 foothold lock 시점(swing 위상). 1.0=항상 reactive(고속 강건성) / <1=late-swing commit(저속 터치다운 매끄러움)
        'trot': dict(OFFSET={0: 0.0, 1: 0.5, 2: 0.5, 3: 0.0}, T=0.50, SWF=0.50, STEPH=0.10, V=0.30, LOCK=1.0),
        'walk': dict(OFFSET={0: 0.25, 1: 0.75, 2: 0.50, 3: 0.0}, T=1.00, SWF=0.25, STEPH=0.05, V=0.25, LOCK=0.35),
    }
    _GP = GAITS[GAIT]   # trot=대각 A(HL,FR)=0·B(HR,FL)=0.5 / walk=순차 FR0→HL.25→FL.5→HR.75(정적안정·75%stance)
    # ★라이브 게이트 holder(GUI 토글이 갱신→재arm으로 위상 재앵커, 불연속 방지). gait()가 GP를 읽음
    _FLENV = os.environ.get('FOOT_LOCK_S')   # 설정 시 게이트 무관 강제(없으면 게이트별 LOCK)
    GP = {'OFFSET': _GP['OFFSET'], 'T': float(os.environ.get('TROT_T', str(_GP['T']))),
          'SWF': float(os.environ.get('TROT_SWF', str(_GP['SWF']))),
          'LOCK': float(_FLENV) if _FLENV else _GP['LOCK']}
    # SWING_FRAC<0.5 → double-support 겹침 → 공중(flight) 방지. walk(0.25)=항상 ≥3발 지지=정적안정
    SETTLE = 0.5
    OFFSET = GP['OFFSET']; T_TROT = GP['T']; SWING_FRAC = GP['SWF']  # T_ST/T_SW(SWING_VREF 해석)용 초기값; 라이브는 gait()가 GP 사용
    STEP_H = float(os.environ.get('TROT_STEPH', str(_GP['STEPH'])))
    V = float(os.environ.get('TROT_V', str(_GP['V'])))     # 전진속도[m/s] 초기/기본
    VY = float(os.environ.get('TROT_VY', '0.0'))    # ★측방속도[m/s] (+좌 −우)
    WZ = float(os.environ.get('TROT_WZ', '0.0'))    # 선회각속도[rad/s] (+좌선회)
    ACC = float(os.environ.get('TROT_ACC', '0.6'))  # 명령 가속도제한[m/s²]: 시작램프+GUI 급조작 완화
    WARMUP = float(os.environ.get('TROT_WARMUP', '0.6'))  # 시작 제자리trot 시간[s]: 첫 사이클 리듬확립 후 이동(시작 lurch 완화)
    CMDFILE = os.environ.get('CMDFILE')             # ★GUI 연동: JSON(/tmp/quad_cmd.json) 폴링(v/vy/w/mode/body_h)
    if CMDFILE: q.cmd_mode = 'stand_up'             # ★GUI 모드: 시작=Ready(stand). Walk 버튼 눌러야 gait 시작 (standalone은 move=즉시보행)
    GROUND_Z = float(os.environ.get('GROUND_Z', '0.18'))   # Ground(눕기) 목표 높이[m] — 낮춰 prone(recovery 대상)
    # ★Recovery getup: 누운 상태(base 낮음)서 Ready=서기 요청 → 중력보상 q_home PD로 일어남(wbic_stance는 발 몸아래 가정→prone실패)
    GETUP_TRIG = float(os.environ.get('GETUP_TRIG', '0.32'))   # 이 높이 미만서 서기요청=getup(누움 감지)
    GETUP_DONE = float(os.environ.get('GETUP_DONE', '0.40'))   # 이 높이 넘으면 정상 wbic_stance로 핸드오프
    GETUP_KP = float(os.environ.get('GETUP_KP', '90')); GETUP_KD = float(os.environ.get('GETUP_KD', '3'))
    GETUP_RATE = float(os.environ.get('GETUP_RATE', '0.18'))   # getup 높이 램프[m/s](느리게=안정)
    REST_KD = float(os.environ.get('REST_KD', '3.0'))          # ★눕기 완료 후 damping(kd-only) 게인 — 능동 hold 제거=모터off 등가(실로봇 power-off 안전)
    JOINT_SLEW = float(os.environ.get('JOINT_SLEW', '1.5'))    # ★fold 관절목표 slew속도[rad/s] — 벌어진 다리 정리 시 q_home으로 점진이동(launch/튀어오름 방지)
    HRATE = float(os.environ.get('HEIGHT_RATE', '0.3'))    # 높이 변경 속도[m/s] (body_h·Ground 부드럽게)
    KP_SW = float(os.environ.get('TROT_KPSW', '40.0')); KD_SW = 2.0
    KCAP = float(os.environ.get('TROT_KCAP', '0.16'))   # capture 게인 ≈√(z/g) (LIPM)
    RAIBERT_K = float(os.environ.get('RAIBERT_K', '0.8'))   # ★전방 reach 게인 기본 0.8(시원한 reach + 고속안정 1.74m/s, 중간속도 손실11%뿐). 1.2는 과제동(명령1.0→0.48). ↑=발앞→제동↑=느림+안정. GUI 슬라이더 live
    RAI_CLIP = float(os.environ.get('RAI_CLIP', '0.25'))    # 최대 발배치[m] (reach 게인 올려도 클립 안 걸리게 여유)
    # ★평지 foothold lock = GP['LOCK'](게이트별: trot1.0=reactive고속강건 / walk0.5=저속 터치다운 매끄러움). env FOOT_LOCK_S로 강제 가능.
    USE_DETECT = os.environ.get('DETECT', '1') == '1'   # detect_contact 조기착지 보정 on/off
    # ── 점프: GUI Jump 트리거 시 offline 궤적(quad_hop.solve_jump) 재생(피드포워드 u* + 관절 PD) ──
    JUMP_NPZ = os.environ.get('JUMP_NPZ', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jump_stand.npz'))
    JUMP = None
    if os.path.exists(JUMP_NPZ):
        _jd = np.load(JUMP_NPZ, allow_pickle=True)
        _aq = q.m.actuator_trnid[:, 0]                  # 액추에이터→조인트
        _perm = q.m.jnt_qposadr[_aq] - 7                # 액추에이터 i → qpos 관절슬롯
        _qp = np.zeros_like(_jd['q']); _qp[:, _perm] = _jd['q']     # qpos순서 q_ref(WBC용)
        _dp = np.zeros_like(_jd['dq']); _dp[:, _perm] = _jd['dq']
        _has_wbc = 'com_ref' in _jd                     # 신 npz(CoM기준궤적)=WBC추종 / 구=open-loop
        JUMP = {'q': _jd['q'], 'dq': _jd['dq'], 'tau': _jd['tau'], 'base_z': _jd['base_z'],
                'q_qp': _qp, 'dq_qp': _dp,
                'sub': max(1, round(float(_jd['dt']) / q.m.opt.timestep)), 'N': len(_jd['tau']),
                'qadr': q.m.jnt_qposadr[_aq], 'vadr': q.m.jnt_dofadr[_aq],
                'wbc': _has_wbc and os.environ.get('JUMP_OPENLOOP') != '1',
                'com_ref': _jd['com_ref'] if _has_wbc else None,
                'comv_ref': _jd['comv_ref'] if _has_wbc else None,
                'acom_ref': _jd['acom_ref'] if _has_wbc else None,
                'sched': [str(s) for s in _jd['sched']] if 'sched' in _jd else None}
        print('[trot] 점프 궤적 로드: %s (knots=%d sub=%d 정점z=%.2f 추종=%s)'
              % (JUMP_NPZ, JUMP['N'], JUMP['sub'], JUMP['base_z'].max(),
                 'WBC(base닫음)' if JUMP['wbc'] else 'open-loop'), flush=True)
    JKP = float(os.environ.get('JUMP_KP', '120')); JKD = float(os.environ.get('JUMP_KD', '3'))
    # ── Ready 호밍: 누르면 발을 명목 위치로 다시 디뎌 기본자세 복귀(대각쌍 2스텝) ──
    HOME_ON_READY = os.environ.get('HOME_ON_READY', '0') == '1'   # 기본 OFF(사용자 선택): Ready=그자리 stance 유지. =1이면 발구름 호밍
    HOME_T = float(os.environ.get('HOME_T', '1.8'))         # 호밍 지속[s]: 제자리 trot로 발을 기본명목 재정렬
    HOME_TOL = float(os.environ.get('HOME_TOL', '0.03'))    # 발편차 이하면 호밍 생략(이미 기본자세)
    T_ST = T_TROT * (1 - SWING_FRAC)
    T_SW = T_TROT * SWING_FRAC                               # swing 지속[s] (해석 v_des 위상율용)
    SWING_VREF = os.environ.get('SWING_VREF') == '1'        # v13식 스플라인 해석 v_des. ★측정결과 net-negative(전체제한시 falls=1, jerk 가속악화) → OFF default (v13 use_spline_diff=False와 동일결론)
    S = {'armed': False, 't0': 0.0, 'nominal': None, 'liftoff': None, 'x_ref': None,
         'ptgt_prev': [None, None, None, None], 'foothold': [None, None, None, None],   # foothold=지형서 스윙중 고정된 착지목표
         'lam_des': None, 'mpc_t': -1.0, 'bx': 0.0,
         'settle_until': SETTLE,
         'Vt': V, 'Vyt': VY, 'Wt': WZ,                      # 목표명령(GUI가 갱신)
         'Vs': 0.0, 'Vys': 0.0, 'Ws': 0.0, 'cmd_t': -1.0,   # 스무딩 적용명령(0서 시작)
         'yaw_ref': 0.0, 'last_t': -1.0,                     # 선회 yaw각 참조(적분) · 직전 시각(reset 감지용)
         'body_h': q.base_z0, 'ht_cur': q.base_z0, 'qhome_h': q.base_z0,   # body_h슬라이더 · 보간높이 · q_home 계산높이
         'step_h': STEP_H,                                                # ★GUI step height(live 갱신)
         'raibert_k': RAIBERT_K,                                          # ★전방 reach 게인(GUI 슬라이더 live)
         'gait': GAIT,                                                     # ★현 게이트(GUI walk/trot 토글 live)
         'pos_hold': None,                                                 # ★정지 시 래치한 x,y(드리프트 보정 기준)
         'pos_hold_on': os.environ.get('POS_HOLD', '1') != '0',           # ★정지 위치홀드 on/off (GUI live 격리용)
         'foot_lock_on': os.environ.get('FOOT_LOCK_ON', '1') != '0',      # ★터치다운 foothold lock on/off (GUI live 격리용)
         'foot_lock_s': GP['LOCK'], 'fl_seen': None,                      # ★lock 시점(낮을수록 강함). 게이트전환=프리셋 / 슬라이더=오버라이드(엣지)
         'q_ref': None,                                                   # ★fold 관절목표(slew용): 진입 시 현자세서 q_home으로 점진
         'yaw_hold': None,                                                 # 선회정지 시 유지할 헤딩(드리프트 보정)
         'jseq': None, 'jact': False, 'jk': 0, 'jsub': 0,                  # 점프: 마지막seq · 재생중 · 현knot · sub카운터
         'prev_mode': q.cmd_mode, 'homing': False, 'home_t0': 0.0,        # Ready 호밍: 직전모드 · 진행중 · 시작시각
         'home_phase': -1, 'home_lift': None,                             #   현 스텝위상 · liftoff 캡처
         'hseq': None, 'home_req': False, 'rseq': None}                   # home_seq · 호밍요청 · ★reset_seq 마지막값(RESET 버튼)

    def gait(i, tg):
        ph = (tg / GP['T'] + GP['OFFSET'][i]) % 1.0     # ★라이브 게이트 파라미터(GP) — GUI 토글 반영
        return (ph >= GP['SWF'], 0.0) if ph >= GP['SWF'] else (False, ph / GP['SWF'])

    def csched(tg):
        cs = np.zeros((q.N_MPC, 4), dtype=bool)
        for k in range(q.N_MPC):
            for i in range(4):
                cs[k, i] = gait(i, tg + k * q.MPC.DT_MPC)[0]
        return cs

    def home_feet_dev():
        """현 발이 명목(hip 기준 기본자세) 위치서 얼마나 벗어났나(최대, m)."""
        _qq = q.d.qpos[3:7]
        yaw = float(np.arctan2(2 * (_qq[0]*_qq[3] + _qq[1]*_qq[2]), 1 - 2 * (_qq[2]**2 + _qq[3]**2)))
        cy, sy = np.cos(yaw), np.sin(yaw); Rw = np.array([[cy, -sy], [sy, cy]])
        dev = 0.0
        for i in range(4):
            tgt = q.d.xpos[q.hip_bid[i]][:2] + Rw @ q.foot_hip_off[i]
            dev = max(dev, float(np.linalg.norm(q.foot_point(i)[:2] - tgt)))
        return dev

    def ctrl():
        t = q.d.time
        # ★ 뷰어 reset(Backspace)/낙상 감지 — ★mode 무관 최우선(stand 모드·armed 전에도 잡힘). 시간역행 OR 큰 위치점프.
        if t < S['last_t'] - 1e-6 or abs(q.d.qpos[0] - S['bx']) > 0.5:
            q.crouch_home()                                  # 깨끗한 crouch 복원
            S['armed'] = False; S['lam_des'] = None; S['ptgt_prev'] = [None, None, None, None]
            S['t0'] = 0.0; S['settle_until'] = q.d.time + SETTLE; S['yaw_ref'] = 0.0
            S['Vs'] = S['Vys'] = S['Ws'] = 0.0; S['cmd_t'] = -1.0   # ★cmd_t 리셋: 시간역행 후 CMDFILE 폴링 재개(안하면 GUI 먹통)
            S['bx'] = float(q.d.qpos[0]); S['last_t'] = q.d.time
            print('[trot] reset 감지 → crouch 복원 후 재정착(%s 모드)' % q.cmd_mode, flush=True)
            return
        S['last_t'] = t; S['bx'] = float(q.d.qpos[0])        # 매틱 갱신(reset 감지 기준)
        # ── GUI 명령 폴링(CMDFILE, ~20Hz): v/vy/w + mode ──
        if CMDFILE and (S['cmd_t'] < 0 or t - S['cmd_t'] > 0.05):
            S['cmd_t'] = t
            try:
                with open(CMDFILE) as _f: _c = json.load(_f)
                S['Vt'] = float(_c.get('v', S['Vt'])); S['Vyt'] = float(_c.get('vy', S['Vyt'])); S['Wt'] = float(_c.get('w', S['Wt']))
                q.cmd_mode = _c.get('mode', 'move')
                S['body_h'] = float(_c.get('body_h', S['body_h']))   # 서기 높이 슬라이더
                _ph = S['step_h']; S['step_h'] = float(_c.get('step_h', S['step_h']))   # ★step height 슬라이더(live)
                if os.environ.get('SHDBG') and abs(S['step_h'] - _ph) > 1e-4:
                    print('[step_h] %.3f → %.3f (GUI live)' % (_ph, S['step_h']), flush=True)
                _g = _c.get('gait', S['gait'])                       # ★게이트 토글(walk/trot 라이브 전환)
                if _g != S['gait'] and _g in GAITS:
                    S['gait'] = _g; GP['OFFSET'] = GAITS[_g]['OFFSET']
                    GP['T'] = GAITS[_g]['T']; GP['SWF'] = GAITS[_g]['SWF']
                    GP['LOCK'] = float(_FLENV) if _FLENV else GAITS[_g]['LOCK']   # 게이트별 lock(trot=reactive/walk=commit)
                    S['foot_lock_s'] = GP['LOCK']                    # 게이트전환=프리셋 lock으로 리셋(trot 고속 reactive 보장)
                    if S['armed']: S['armed'] = False                # 재arm=위상클럭 재앵커(현 stance서 새 게이트 리듬 재확립, 불연속 방지)
                    print('[trot] 게이트 전환 → %s (재정렬)' % _g, flush=True)
                S['raibert_k'] = float(_c.get('raibert_k', S['raibert_k']))       # ★전방 reach 게인 슬라이더(live)
                if 'swing_w' in _c: q._swing_w_r = q._swing_w_f = float(_c['swing_w'])  # 통합(하위호환)
                q._swing_w_f = float(_c.get('swing_w_f', q._swing_w_f))            # ★앞다리 whip 슬라이더(live)
                q._swing_w_r = float(_c.get('swing_w_r', q._swing_w_r))            # ★뒷다리 whip 슬라이더(live)
                S['pos_hold_on'] = bool(_c.get('pos_hold', S['pos_hold_on']))     # ★정지 위치홀드 토글(격리용)
                S['foot_lock_on'] = bool(_c.get('foot_lock', S['foot_lock_on']))  # ★터치다운 lock 토글(격리용)
                _fl = _c.get('foot_lock_s')                          # ★lock 강도 슬라이더(엣지 오버라이드 → 게이트전환 리셋과 공존)
                if _fl is not None and _fl != S['fl_seen']:
                    S['foot_lock_s'] = float(_fl); S['fl_seen'] = _fl
                q._rate = float(_c.get('rate', q._rate))             # ★뷰어 배속 슬라이더(live)
                q._viz = bool(_c.get('viz', q._viz))                 # ★모니터 표시 토글(live)
                _tn = bool(_c.get('terrain', q._terrain_on))         # ★지형적응 토글: launch 기본(STAIRS) 보존 위해 edge-trigger
                if 'terr_seen' not in S: S['terr_seen'] = _tn        # 첫폴링=GUI기본값 기록만(적용X)
                elif _tn != S['terr_seen']: q._terrain_on = _tn; S['terr_seen'] = _tn   # 사용자 토글만 적용
                if JUMP is not None:                                  # 점프 트리거(jump_seq 상승엣지)
                    _js = int(_c.get('jump_seq', 0))
                    if S['jseq'] is not None and _js > S['jseq'] and not S['jact']:  # 첫폴링=동기화(시작점프 방지)
                        S['jact'] = True; S['jk'] = 0; S['jsub'] = 0
                        print('[trot] 점프 트리거(seq=%d) → offline 궤적 재생' % _js, flush=True)
                    S['jseq'] = _js
                _hs = int(_c.get('home_seq', 0))                     # Ready 버튼 → 기본자세 호밍 요청(모드 무관)
                if S['hseq'] is not None and _hs > S['hseq']:
                    S['home_req'] = True
                S['hseq'] = _hs
                _rs = int(_c.get('reset_seq', 0))                    # ★RESET 버튼 → 시뮬 리셋(넘어짐 복구)
                if S['rseq'] is not None and _rs > S['rseq']:
                    mujoco.mj_resetData(q.m, q.d); q.crouch_home()   # 깨끗한 상태 + 기립자세
                    S['armed'] = False; S['lam_des'] = None; S['ptgt_prev'] = [None, None, None, None]
                    S['settle_until'] = q.d.time + SETTLE; S['yaw_ref'] = 0.0
                    S['Vs'] = S['Vys'] = S['Ws'] = 0.0; S['bx'] = float(q.d.qpos[0]); S['last_t'] = q.d.time
                    print('[trot] RESET 버튼 → 시뮬 리셋', flush=True)
                S['rseq'] = _rs
            except Exception: pass
        _prev_mode = S['prev_mode']; S['prev_mode'] = q.cmd_mode    # 매틱 직전모드(호밍 진입엣지 감지)
        if q.cmd_mode == 'stand_up' and _prev_mode != 'stand_up':   # 다른모드→Ready 진입도 호밍 요청
            S['home_req'] = True
        # ── 점프 재생: WBC추종(base를 GRF로 닫음, 표준) 또는 open-loop(피드포워드+관절PD) ──
        if S['jact'] and JUMP is not None:
            k = min(S['jk'], JUMP['N'] - 1)
            if JUMP['wbc']:                                          # ★표준 WBC: CoM 3-DOF 피드백 + 관절추종
                _ph = JUMP['sched'][k] if JUMP['sched'] is not None else 'push'
                _cts = () if _ph == 'flight' else (0, 1, 2, 3)       # flight=탄도(접촉없음)
                _r = q.wbic_jump(JUMP['com_ref'][k], JUMP['comv_ref'][k], JUMP['acom_ref'][k],
                                 JUMP['q_qp'][k], JUMP['dq_qp'][k], _cts)
                if _r[0] is None:                                    # QP 실패 시 open-loop 폴백
                    qcur = q.d.qpos[JUMP['qadr']]; dqcur = q.d.qvel[JUMP['vadr']]
                    tau = JUMP['tau'][k] + JKP * (JUMP['q'][k] - qcur) + JKD * (JUMP['dq'][k] - dqcur)
                    q.d.ctrl[:] = np.clip(tau, -q._tau_peak, q._tau_peak)
            else:                                                    # open-loop(구 npz/JUMP_OPENLOOP=1)
                qcur = q.d.qpos[JUMP['qadr']]; dqcur = q.d.qvel[JUMP['vadr']]
                tau = JUMP['tau'][k] + JKP * (JUMP['q'][k] - qcur) + JKD * (JUMP['dq'][k] - dqcur)
                q.d.ctrl[:] = np.clip(tau, -q._tau_peak, q._tau_peak)
            q.cmd_v[:] = 0.0
            S['jsub'] += 1
            if S['jsub'] >= JUMP['sub']:
                S['jsub'] = 0; S['jk'] += 1
                if S['jk'] >= JUMP['N']:                              # 착지·복귀 완료 → 정착(자동호밍 없음; 발 재정렬은 Ready 수동)
                    S['jact'] = False; S['armed'] = False
                    S['settle_until'] = t + SETTLE
                    q.update_stand_qhome(q.base_z0); S['ht_cur'] = q.base_z0; S['qhome_h'] = q.base_z0
                    print('[trot] 점프 완료 → 정착(base_z=%.3f)' % q.d.qpos[2], flush=True)
            return
        # ── 모드: move 외(Ready 서기/Ground 눕기/STOP)는 제자리 WBIC stance + 높이제어 ──
        if q.cmd_mode != 'move':
            # ★전원 off = 능동제어 없이 damping만 → 로봇 쓰러짐(recovery 데모: off→ground→ready).
            if q.cmd_mode == 'off':
                q.d.ctrl[:] = np.clip(-REST_KD * q.d.qvel[6:6 + q.nu], -q._tau_peak, q._tau_peak)
                S['armed'] = False; q.cmd_v[:] = 0.0; S['q_ref'] = None
                return
            # ★낮은 자세(Ground 눕기 / Ready getup) = 중력보상 PD로 "수평" q_home 추종.
            #   wbic_stance는 ~0.29 미만 못 내려가고, 무제어 collapse는 비대칭으로 뒤집힘 →
            #   수평 q_home PD가 양방향(눕기↓·일어서기↑) 모두 몸을 수평 유지하며 fold. ≥GETUP_DONE이면 wbic_stance(균형/외란대응).
            _bz = float(q.d.qpos[2])
            if _bz < GETUP_TRIG and S['ht_cur'] > GETUP_DONE:               # ★쓰러짐/off로 실제 낮은데 ht_cur 높음 → 동기화(현 높이서 fold 시작)
                S['ht_cur'] = max(0.12, _bz)
                print('[trot] recovery (%s, base_z=%.2f)' % (q.cmd_mode, _bz), flush=True)
            _tgt = GROUND_Z if q.cmd_mode == 'stand_down' else S['body_h']   # 눕기=낮게, 서기=슬라이더 높이
            _low = (S['ht_cur'] < GETUP_DONE) or (_tgt < GETUP_DONE)
            _rate = GETUP_RATE if _low else HRATE                            # 낮은 자세 fold는 느리게(안정)
            S['ht_cur'] += float(np.clip(_tgt - S['ht_cur'], -_rate * q.m.opt.timestep, _rate * q.m.opt.timestep))
            if abs(S['ht_cur'] - S['qhome_h']) > 6e-3:        # 높이 램프 q_home/com_ref IK 재계산
                q.update_stand_qhome(S['ht_cur']); S['qhome_h'] = S['ht_cur']
            _jerr = float(np.mean(np.abs(q.q_home - q.d.qpos[7:7 + q.nu])))   # 다리가 q_home(정리자세)에 얼마나 가까운지
            if q.cmd_mode == 'stand_down' and abs(S['ht_cur'] - GROUND_Z) <= 0.02 and _jerr < 0.3:   # ★다리 정리(fold) 완료 후에만 damp(벌어진채 damp 방지)
                # ★눕기 완료(trunk 지면 접지) → damping(kd-only, 능동 hold 제거) = 모터 off 등가.
                #   실로봇서 여기서 전원 차단해도 지면이 받쳐 추가 처짐 없음(Go2 'damping' 자세와 동일).
                q.d.ctrl[:] = np.clip(-REST_KD * q.d.qvel[6:6 + q.nu], -q._tau_peak, q._tau_peak)
                S['armed'] = False; q.cmd_v[:] = 0.0; S['q_ref'] = None      # damp=fold 종료 → q_ref 리셋
                return
            if S['ht_cur'] < GETUP_DONE and not S['homing']:  # ★낮은 자세 = 수평 PD fold(눕기/getup 공통, 안 뒤집힘)
                if S['q_ref'] is None:                                       # fold 진입=현 자세서 시작(벌어진 다리도 그자리서)
                    S['q_ref'] = q.d.qpos[7:7 + q.nu].copy()
                S['q_ref'] += np.clip(q.q_home - S['q_ref'],                 # ★관절목표 점진 slew → q_home (launch 방지)
                                      -JOINT_SLEW * q.m.opt.timestep, JOINT_SLEW * q.m.opt.timestep)
                tau = q.d.qfrc_bias[6:6 + q.nu] + GETUP_KP * (S['q_ref'] - q.d.qpos[7:7 + q.nu]) \
                    - GETUP_KD * q.d.qvel[6:6 + q.nu]                        # 중력보상 + 점진목표 PD
                q.d.ctrl[:] = np.clip(tau, -q._tau_peak, q._tau_peak)
                S['armed'] = False; q.cmd_v[:] = 0.0
                return
            S['q_ref'] = None                                               # fold 벗어남(wbic/damp/off/move) → 다음 진입 시 현자세서 재시작
            # ★ 기본자세 호밍 요청(Ready 버튼·Ready 진입·점프완료) → 정착 후 제자리 trot로 발 재정렬 트리거
            if (HOME_ON_READY and S['home_req'] and not S['homing']
                    and q.cmd_mode == 'stand_up' and t >= S['settle_until']):
                S['home_req'] = False
                _dev = home_feet_dev()
                if _dev > HOME_TOL:                           # 이미 기본자세면(편차 작음) 생략
                    S['homing'] = True; S['home_t0'] = t; S['armed'] = False   # 트로트 재arm 강제(아래 경로로 진행)
                    print('[trot] 기본자세 호밍 시작(발편차 %.0fmm, 제자리 trot)' % (_dev * 1000), flush=True)
            if not S['homing']:                               # 호밍 아니면 일반 stance 유지
                q.wbic_stance(); S['armed'] = False; q.cmd_v[:] = 0.0
                return
            # 호밍 중이면 아래 trot 경로로 진행(제자리 v=0, 발=기본명목 — MPC 균형 재활용)
        else:
            S['homing'] = False; S['q_ref'] = None            # move(보행) 명령 = 호밍 취소 + fold q_ref 리셋
        # ── trot 경로 (move 보행 또는 homing 제자리 재정렬) ──
        if S['homing'] and (t - S['home_t0'] > HOME_T):       # 호밍 종료 → stance 복귀
            S['homing'] = False; S['armed'] = False; S['settle_until'] = t + SETTLE
            q.wbic_stance(); q.cmd_v[:] = 0.0
            print('[trot] 호밍 완료 → stance(base_z=%.3f 발편차%.0fmm)'
                  % (q.d.qpos[2], home_feet_dev() * 1000), flush=True)
            return
        if t < S['settle_until']:            # 1) WBIC stance 정착 (초기/리셋)
            q.wbic_stance(); return
        if not S['armed']:                   # 2) 정착/리셋 자세를 trot 기준으로 캡처
            S['armed'] = True; S['t0'] = t; S['yaw_ref'] = 0.0
            S['nominal'] = [q.foot_point(i).copy() for i in range(4)]
            S['liftoff'] = [q.foot_point(i).copy() for i in range(4)]
            xr = np.zeros(13); xr[5] = float(q.d.subtree_com[0][2]); xr[12] = -9.81  # vx/vy/wz는 매틱 갱신
            S['x_ref'] = xr
            # ★ 발 목표 = 몸(hip) 기준 절대 명목위치(foot_hip_off) — 현재 발위치 무관.
            #   어긋난 발(점프·외란·호밍 직후)도 기본 stance로 수렴. (호밍·보행 공통)
            S['hip_off'] = [q.foot_hip_off[i].copy() for i in range(4)]
            S['gz'] = [float(z) for z in q.foot_gz0]
            q.MPC.MPC_Q = TROT_Q
            print('[trot] %s 시작 v=%.2f vy=%.2f w=%.2f%s (base_z=%.3f)'
                  % ('호밍(제자리)' if S['homing'] else '정착 완료 → trot', S['Vt'], S['Vyt'], S['Wt'],
                     ' [GUI]' if CMDFILE else '', q.d.qpos[2]))
        tg = t - S['t0']                     # 3) trot
        # ── 가속도제한 스무딩 + 시작 warmup(첫 WARMUP초 제자리trot로 리듬확립 후 이동) ──
        dts = q.m.opt.timestep
        _go = tg > WARMUP                                   # warmup 지난 뒤에만 목표속도 적용
        _vt, _vyt, _wt = (0.0, 0.0, 0.0) if S['homing'] else \
            ((S['Vt'], S['Vyt'], S['Wt']) if _go else (0.0, 0.0, 0.0))   # 호밍=제자리(발만 재정렬)
        S['Vs']  += float(np.clip(_vt  - S['Vs'],  -ACC * dts, ACC * dts))
        S['Vys'] += float(np.clip(_vyt - S['Vys'], -ACC * dts, ACC * dts))
        S['Ws']  += float(np.clip(_wt  - S['Ws'],  -2.0 * dts, 2.0 * dts))
        V_eff, Vy_eff, W_eff = S['Vs'], S['Vys'], S['Ws']
        if q._waist_idx is not None:                                            # ★허리 조향: 선회명령에 앞몸통 굽힘 연동(WAIST_STEER=0이면 중립홀드)
            q._waist_ref = float(np.clip(q._waist_steer * W_eff, -0.75, 0.75))
        # 선회: yaw각 참조 + 명령(body)→world 회전 (SRBD 상태는 world frame)
        _qq = q.d.qpos[3:7]                                                     # quat [w,x,y,z]
        yaw_m = float(np.arctan2(2 * (_qq[0]*_qq[3] + _qq[1]*_qq[2]), 1 - 2 * (_qq[2]**2 + _qq[3]**2)))
        if abs(W_eff) > 0.02:                                                   # 선회중: yaw_ref 적분(측정 0.3rad 이내 클램프=windup방지)
            S['yaw_ref'] = float(np.clip(S['yaw_ref'] + W_eff * dts, yaw_m - 0.3, yaw_m + 0.3)); S['yaw_hold'] = None
        else:                                                                  # ★선회 멈추면 현재 헤딩 래치·유지 → MPC가 드리프트 보정(전진이 헤딩 따라감)
            if S['yaw_hold'] is None: S['yaw_hold'] = yaw_m
            S['yaw_ref'] = S['yaw_hold']
        cy, sy = np.cos(yaw_m), np.sin(yaw_m)
        vx_w = V_eff * cy - Vy_eff * sy; vy_w = V_eff * sy + Vy_eff * cy        # body→world 속도
        # ★정지 시 위치 홀드(yaw_hold의 병진판): 게이트 체계바이어스(walk 후방드리프트) 보정.
        #   정지명령서 현 x,y 래치 → 표류를 작은 보정속도로 되돌림(MPC vx추종 + Raibert 발배치 반영). 이동중 해제.
        #   상태추정(odometry) 기반 외부루프 위치제어 = 실로봇도 동일(비물리 아님).
        if S['pos_hold_on'] and abs(V_eff) < 0.03 and abs(Vy_eff) < 0.03 and abs(W_eff) < 0.05:
            if S['pos_hold'] is None:
                S['pos_hold'] = (float(q.d.qpos[0]), float(q.d.qpos[1]))
            _phk = float(os.environ.get('POS_HOLD_K', '0.6'))              # 위치오차→보정속도 게인
            vx_w += float(np.clip(-_phk * (q.d.qpos[0] - S['pos_hold'][0]), -0.15, 0.15))
            vy_w += float(np.clip(-_phk * (q.d.qpos[1] - S['pos_hold'][1]), -0.15, 0.15))
        else:
            S['pos_hold'] = None
        S['x_ref'][2] = S['yaw_ref']; S['x_ref'][8] = W_eff                     # yaw각·yaw rate 참조
        S['x_ref'][9] = vx_w; S['x_ref'][10] = vy_w                            # world vx,vy
        if q._terrain_on:                                                      # ★perceptive: MPC 높이 기준=평지값+지형(상승 GRF 계획). 정석=planner가 참조생성
            S.setdefault('z_ref0', S['x_ref'][5])                             #   arming 평지 높이 저장(최초 1회)
            q._body_terr = float(np.mean([q.terrain_height(q.d.xpos[q.hip_bid[i]][0], q.d.xpos[q.hip_bid[i]][1]) for i in range(4)]))  # ★틱당1회(a_z 재사용)
            S['x_ref'][5] = S['z_ref0'] + q._body_terr
        else:
            q._body_terr = 0.0
        q.cmd_v[0] = V_eff; q.cmd_v[1] = Vy_eff; q.cmd_v[5] = W_eff             # 시각화(body명령)
        # ★ Di Carlo 식: gait 스케줄=primary, detect_contact=조기/지연 착지 보정.
        #   스케줄 stance → 힘제어. 스케줄 swing 후반(>0.7)+접촉감지 → 조기착지로 stance 승격.
        #   그 외 swing → 들어올림. (USE_DETECT=False 면 순수 스케줄, A/B 비교용)
        contact, _ = q.detect_contact([gait(i, tg)[0] for i in range(4)])
        st, sw = [], []
        for i in range(4):
            sch_stance, s_prog = gait(i, tg)
            early = USE_DETECT and (not sch_stance) and contact[i] and s_prog > 0.7
            if sch_stance or early:                         # 스케줄 stance OR 조기착지 → 힘제어
                st.append(i); S['ptgt_prev'][i] = None      # 다음 swing 첫 프레임 v_des=0
                S['foothold'][i] = None                     # ★stance=foothold 잠금해제(다음 스윙서 재선정)
            else:                                           # 스케줄 swing → 위치제어(들어올림)
                sw.append(i)
                if s_prog < 0.03:                           # swing 시작 시 liftoff 캡처
                    S['liftoff'][i] = q.foot_point(i).copy()
        # 발 배치 = hip 기준 default + Raibert(전진 0.5·T_st·V + 피드백 KCAP·(v−v_des))
        Jc = np.zeros((3, q.nv)); mujoco.mj_jacSubtreeCom(q.m, q.d, Jc, 0)
        vcom = Jc @ q.d.qvel
        v_des = np.array([vx_w, vy_w])       # Raibert 발배치 = world 속도(전진+측방, 선회 회전반영)
        v_fb = vcom[:2].copy()               # 발배치 피드백 속도 (기본=CoM 속도)
        # ★ALIP: CoM속도 대신 각운동량 반영 속도 v_alip = vcom + [L_y,−L_x]/(m·H).
        #   centroidal 각운동량 L(다리 swing momentum 포함)을 발배치에 녹여 leg-heavy 고속 안정화.
        if os.environ.get('ALIP', '1') != '0':       # 기본 ON (정상속도 무해·고속 도움; ALIP=0로 끔)
            mujoco.mj_subtreeVel(q.m, q.d)
            L = q.d.subtree_angmom[0]                       # centroidal 각운동량 (world, 다리 포함)
            H = max(0.1, float(q.d.subtree_com[0][2]))      # CoM 높이
            mtot = float(q.m.body_mass.sum())
            _ag = float(os.environ.get('ALIP_G', '1.0'))    # ALIP 항 가중(튜닝)
            v_fb = v_fb + _ag * np.array([L[1], -L[0]]) / (mtot * H)
        rai = np.clip(S['raibert_k'] * T_ST * v_des + KCAP * (v_fb - v_des), -RAI_CLIP, RAI_CLIP)   # ★전방 reach 게인(GUI 슬라이더 live) / 최대 발배치 클립
        q.foot_targets = [None, None, None, None]
        dt = q.m.opt.timestep; swing = {}
        Rw = np.array([[cy, -sy], [sy, cy]])                 # body→world(뒷몸통/base yaw)
        # ★앞다리(FL/FR)는 앞몸통에 붙음 → 명목 발offset을 앞몸통 방향(base+허리 yaw)으로 회전. 허리 꺾이면 앞발배치도 따라감(기하 반영)
        _wa = float(q.d.qpos[7 + q._waist_idx]) if q._waist_idx is not None else 0.0
        _cyf, _syf = np.cos(yaw_m + _wa), np.sin(yaw_m + _wa)
        Rw_front = np.array([[_cyf, -_syf], [_syf, _cyf]])
        _STH = S['step_h']                                  # ★GUI live step height
        _sh = _STH if S['homing'] else (                    # 호밍=풀 step height(발 어긋남 클리어), 보행=시작 ramp
            _STH * (0.2 + 0.8 * min(1.0, tg / WARMUP)) if WARMUP > 1e-6 else _STH)
        for i in sw:                                        # swing 발끝 작업공간 목표(p,v)
            s_ = gait(i, tg)[1]
            if S['foothold'][i] is not None:                # ★잠긴 foothold 사용(지형=스윙시작 / 평지=late-swing) → 착지목표 고정=매끄러운 터치다운
                p_end = S['foothold'][i]
            else:                                           # 평지 초반=매틱 reactive Raibert(적응) / 지형=스윙첫프레임 1회
                hip_xy = q.d.xpos[q.hip_bid[i]][:2]
                r_xy = hip_xy - q.d.qpos[:2]                # 몸중심→hip
                tw = W_eff * T_ST * np.array([-r_xy[1], r_xy[0]])  # 선회 접선 발배치(yaw)
                _Rleg = Rw_front if q.legs[i] in ('FL', 'FR') else Rw   # ★앞다리=앞몸통방향(허리반영)
                pe_xy = hip_xy + _Rleg @ S['hip_off'][i] + rai + tw  # nominal도 몸따라 회전 + Raibert + 선회
                _ex, _ey = float(pe_xy[0]), float(pe_xy[1])
                if q._stair is not None:                    # ★foothold를 tread 중앙쪽으로 snap(riser 모서리 6cm 회피)
                    _H, _D, _N, _X0 = q._stair
                    if _ex >= _X0:
                        _ti = min(int((_ex - _X0) // _D), _N - 1)
                        _ex = min(max(_ex, _X0 + _ti * _D + 0.06), _X0 + (_ti + 1) * _D - 0.06)
                p_end = np.array([_ex, _ey, S['gz'][i] + q.terrain_height(_ex, _ey)])
                _locks = S['foot_lock_s'] if S['foot_lock_on'] else 1.0   # ★lock 강도(슬라이더/게이트프리셋) · off=항상 reactive
                if q._terrain_on or s_ >= _locks:           # ★지형=스윙시작 lock / 평지=게이트별 late-swing lock(walk0.5/trot1.0=reactive)
                    S['foothold'][i] = p_end                # 초반 reactive 적응 + 후반 commit(착지목표 고정)
            q.foot_targets[i] = p_end                       # 착지 목표 시각화(고정된 빨강구)
            # ★계단: swing_foot_pos의 Z는 liftoff 높이로 되돌아옴(p_end[2] 무시) → 계단서 착지면 아래로 내리꽂음.
            #   Z baseline을 liftoff→landing 높이로 s5 보간해 보정(평지=Δz0 → 무변화). _dzland 클로저로 재사용.
            _liftz = S['liftoff'][i][2]; _dzl = p_end[2] - _liftz
            _zfix = lambda ss: _dzl * (10 * ss**3 - 15 * ss**4 + 6 * ss**5)
            p_tgt = swing_foot_pos(s_, S['liftoff'][i], p_end,
                                   np.array([vcom[0], vcom[1], 0]), step_height=_sh, tau_land=1.0)
            p_tgt[2] += _zfix(s_)
            if SWING_VREF and s_ < 1.0:                     # ★v13식 일관 레퍼런스: 한 스텝 앞 위상 샘플=스플라인 해석속도
                _ph = min(s_ + dt / T_SW, 0.999)            #   (차분+±1.0clip의 불일치 제거 → SW_KD 피드포워드 정확 → 저크↓)
                _pa = swing_foot_pos(_ph, S['liftoff'][i], p_end,
                                     np.array([vcom[0], vcom[1], 0]), step_height=_sh, tau_land=1.0)
                _pa[2] += _zfix(_ph)
                v_tgt = (_pa - p_tgt) / dt
            else:
                pv = S['ptgt_prev'][i]                      # (기존) 차분+±1.0 clip
                v_tgt = np.clip((p_tgt - pv) / dt, -1.0, 1.0) if pv is not None else np.zeros(3)
            S['ptgt_prev'][i] = p_tgt.copy(); swing[i] = (p_tgt, v_tgt, s_)   # ★s_=스윙위상(발목 능동flick용)
        # ★swing 추종오차 계측(LOG_JOINTS시): 발끝이 target plan을 얼마나 못 쫓는지(수평/수직)
        if sw:
            _e=[q.foot_point(i)-swing[i][0] for i in sw]
            q._swing_err=max(np.linalg.norm(e) for e in _e); q._swing_errh=max(np.hypot(e[0],e[1]) for e in _e)
        else: q._swing_err=0.0; q._swing_errh=0.0
        # ★foothold가 hip 기준 앞/뒤 어디 생기는지(전진+): 앞다리 뻗음 진단
        q._fho=[float(q.foot_targets[i][0]-q.d.xpos[q.hip_bid[i]][0]) if q.foot_targets[i] is not None else np.nan for i in range(4)]
        # ★ 표준 구조: MPC 저주파 재계획(DT_MPC=0.02s=50Hz), WBIC 풀주파(500Hz).
        #   MPC는 무겁고(긴 호라이즌 QP) 느리게 변하는 GRF 계획 → 매 스텝 풀 필요 없음.
        #   WBIC는 빠른 외란/접촉 반응 위해 매 스텝.  → 긴 호라이즌도 실시간 감당.
        # 재계획: 현재 MPC 윈도우[mpc_t, mpc_t+DT_MPC) 밖이면. t<mpc_t(시간역행/리셋)도 즉시 재계획.
        dmpc = t - S['mpc_t']
        if st and (S['lam_des'] is None or dmpc < 0 or dmpc >= q.MPC.DT_MPC):
            S['lam_des'] = q.mpc_grf(S['x_ref'], csched(tg)); S['mpc_t'] = t
        lam_des = S['lam_des'] if (st and S['lam_des'] is not None) else np.zeros((4, 3))
        q.wbic_track(lam_des, contacts=tuple(st), swing=swing)

    def reset():
        q.crouch_home(); S['armed'] = False

    # ★RESET_ON_FALL=0(GUI/복구용): 쓰러져도 자동 reset 안 함 → off→ground→ready 복구 가능. 기본 1(헤드리스 falls 측정).
    # ★GUI 연동(CMDFILE) 시 낙상 자동리셋 OFF → RESET 버튼으로만 복구(C++와 일관). 헤드리스/standalone은 기존 동작.
    q.run_viewer(ctrl, reset_fn=reset,
                 reset_on_fall=(os.environ.get('RESET_ON_FALL', '1') != '0') and not CMDFILE)


def main():
    ap = argparse.ArgumentParser(description='실제 4족 MuJoCo 통합 테스트')
    # 단계: stance(1) → mpc(3.a 정지) → walk(3.b 보행) → nmpc(4).  view/stand 는 sanity.
    ap.add_argument('--mode', default='stance',
                    choices=['view', 'inspect', 'stand', 'stance', 'mpc', 'walk', 'trot'])
    ap.add_argument('--nolimit', action='store_true', help='관절 한계 해제 (가동범위 테스트용)')
    ap.add_argument('--robot', default='ours', choices=list(ROBOTS), help='로봇 모델 선택')
    a = ap.parse_args()
    global _NOLIMIT, _ROBOT; _NOLIMIT = a.nolimit; _ROBOT = a.robot
    os.environ.setdefault('DISPLAY', ':0')
    {'view': mode_view, 'inspect': mode_inspect, 'stand': mode_stand, 'stance': mode_stance,
     'mpc': mode_mpc, 'walk': mode_walk, 'trot': mode_trot}[a.mode]()
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
