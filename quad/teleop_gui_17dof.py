"""teleop_gui — 02_Leg 제어 GUI (RBQ 스타일: dual 조이스틱 + 모션 버튼).

Rainbow Robotics RBQ GUI 참고(JoystickThumbPad + motionStaticReady/Ground/DynamicWalk).
명령을 JSON 채널(/tmp/quad_cmd.json)로 발행 → 컨트롤러는 CMDFILE 로 소비(sim/실 동일).
배포 시 SportClient._pub 백엔드만 ROS2/DDS(또는 RBQ setPosRef/setTorqueRef 상위)로 교체.

사용:
  ① GUI:    python teleop_gui.py
  ② 컨트롤러: cd /home/jsh/simple-mpc && VIEW=1 CMDFILE=/tmp/quad_cmd.json ~/.pixi/bin/pixi run python examples/quad_fulldynamics.py
좌스틱=전후/측방, 우스틱=선회. 버튼: Ready(서기)/Ground(눕기)/Walk(보행)/STOP.
"""
import os
import json
import math
import dearpygui.dearpygui as dpg

CMD_PATH = os.environ.get('QUAD_CMD', '/tmp/quad_cmd.json')
VMAX = float(os.environ.get('VMAX', '0.4'))
WMAX = float(os.environ.get('WMAX', '0.3'))


class SportClient:
    """명령 발행기(JSON 원자적 쓰기). RBQ 모션명 별칭 포함(motionStaticReady/Ground/DynamicWalk)."""

    def __init__(self, path=CMD_PATH):
        self.path = path
        self.vmax = VMAX; self.wmax = WMAX          # 조이스틱 풀스케일(보행속도 게이지로 live 조절)
        self.cmd = {'v': 0.0, 'vy': 0.0, 'w': 0.0, 'mode': 'stand_up',   # 시작=Ready(서기). Walk 눌러야 보행
                    'body_h': 0.52, 'step_h': 0.10, 'euler': [0.0, 0.0, 0.0], 'gait': 'trot',   # ★초기 서기 높이=0.52(Ready·slider와 일치, 보행은 gait 버튼서 0.50)
                    'vmax': VMAX, 'jump_seq': 0, 'home_seq': 0, 'reset_seq': 0,
                    'rate': 1.0, 'viz': True, 'terrain': True,   # ★rate=뷰어배속 viz=모니터표시+추정 base 고스트(개루프·보행 GT제어) terrain=지형적응
                    'foot_nudge': False,                         # ★③ footScore 발판선택(갭·스테핑 crossing). 체크 시 도달반경 내 지형 발판으로 스윙발 유도. 연속지형(험지·평지)엔 불필요
                    'pos_hold': True,                            # ★정지 위치홀드(드리프트 보정, 효용확인됨)
                    'raibert_k': 0.5,
                    'swing_w_f': 0.1, 'swing_w_r': 0.6, 'auto_whip': True,  # ★앞/뒤 whip 목표(고속) · 속도연동 자동whip
                    'steer': 0.0,                                # ★허리 핸들=자동차식 조향각[rad] (컨트롤러 부호 +좌/−우, GUI 슬라이더는 우스틱과 통일해 오른쪽=우선회로 반전발행). Ackermann 반경 R=축거/tanδ, 다리선회가 실행 + 허리 안쪽 lean
                    'g_lie_z': 0.226, 'g_rear_foot': -1.15, 'g_front_thigh': -0.24, 'g_front_calf': -0.4}  # ★눕기(ground) 자세 실시간 조각 — 기본값=뷰어 라이브튜닝 확정
        self._pub()

    def SetGround(self, k, v):                      # ★눕기 자세 파라미터 실시간 조절(뷰어 보며 CoM균형·수평·무슬라이드)
        self.cmd[k] = float(v); self._pub()

    def SimRate(self, r):                           # 뷰어 배속(0.25~4, 0=최대) — live
        self.cmd['rate'] = float(r); self._pub()

    def SetViz(self, on):                           # 모니터 overlay(GRF/CoM/궤적/elevation) 표시 on/off — live
        self.cmd['viz'] = bool(on); self._pub()

    def SetTerrain(self, on):                       # 지형적응(perception: 발/몸높이+elevation) on/off — live
        self.cmd['terrain'] = bool(on); self._pub()

    def SetFootNudge(self, on):                     # ③ footScore 발판선택(갭·스테핑 crossing) on/off — live
        self.cmd['foot_nudge'] = bool(on); self._pub()

    def SetGait(self, g):                           # 게이트 walk/trot 라이브 전환(컨트롤러가 재arm으로 위상 재앵커)
        self.cmd['gait'] = str(g); self._pub()

    def SetPosHold(self, on):                        # 정지 위치홀드 on/off (드리프트 보정)
        self.cmd['pos_hold'] = bool(on); self._pub()

    def SetRaibertK(self, k):                        # 전방 reach 게인(↑=앞으로 더 시원하게 뻗음) — live
        self.cmd['raibert_k'] = float(k); self._pub()

    def SetSwingWF(self, w):                         # ★앞다리 whip 억제(↑=억제, ↓=whip 심함) — live
        self.cmd['swing_w_f'] = float(w); self._pub()

    def SetSwingWR(self, w):                         # ★뒷다리 whip 억제 — live
        self.cmd['swing_w_r'] = float(w); self._pub()

    def SetAutoWhip(self, on):                       # ★속도연동 자동 whip(on=고속서 자동 채찍질, off=수동 슬라이더)
        self.cmd['auto_whip'] = bool(on); self._pub()

    def SetSteer(self, s):                           # ★허리 핸들=자동차식 조향각. 다리선회(A)가 반경 실행 + 허리 안쪽 lean
        self.cmd['steer'] = float(s); self._pub()

    def BodyHeight(self, h):                        # ★몸통 높이[m] — 서기+보행 공통(보행 중에도 추종)
        self.cmd['body_h'] = float(h); self._pub()

    def WalkSpeed(self, v):                         # 보행속도 게이지[m/s] — 조이스틱 풀스케일 (양 컨트롤러 공통)
        self.vmax = float(v); self.cmd['vmax'] = float(v); self._pub()

    def Jump(self):                                 # 점프 트리거(상승엣지로 컨트롤러가 offline 호핑 실행)
        self.cmd['jump_seq'] = int(self.cmd.get('jump_seq', 0)) + 1; self._pub()

    def Ready(self):                                # 서기 + 기본자세 복귀(home_seq 상승엣지 → 발 명목위치 호밍)
        self.cmd['mode'] = 'stand_up'; self.cmd.update(v=0.0, vy=0.0, w=0.0)
        self.cmd['home_seq'] = int(self.cmd.get('home_seq', 0)) + 1; self._pub()

    def StepHeight(self, h):                        # 발 들림[m] (launch 적용)
        self.cmd['step_h'] = float(h); self._pub()

    def _pub(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.cmd, f)
        os.replace(tmp, self.path)

    def Move(self, vx, vy, vyaw):
        self.cmd.update(v=vx, vy=vy, w=vyaw); self._pub()

    def StopMove(self):
        self.cmd.update(v=0.0, vy=0.0, w=0.0); self._pub()

    def Reset(self):                                 # ★시뮬 리셋(넘어짐 복구): reset_seq 상승엣지 → 컨트롤러가 mj_resetData+crouch_home
        self.cmd['reset_seq'] = int(self.cmd.get('reset_seq', 0)) + 1
        self.cmd.update(v=0.0, vy=0.0, w=0.0, mode='stand_up'); self._pub()

    def SetMode(self, m):
        self.cmd['mode'] = m
        if m not in ('move', 'tamols'):                # ★tamols(full-TAMOLS injection)도 보행모드=속도 허용
            self.cmd.update(v=0.0, vy=0.0, w=0.0)      # 자세전환 = 속도 0
        self._pub()

    # ── RBQ 모션 API 별칭(이름 그대로) ──
    def motionDynamicWalk(self):  self.SetMode('move')
    def motionTamols(self):       self.SetMode('tamols')   # ★full-TAMOLS injection(TAMOLS 발판+MPC 예측GRF+strict HQP)
    def motionStaticReady(self):  self.SetMode('stand_up')     # 서기
    def motionStaticGround(self): self.SetMode('stand_down')   # 눕기


sc = SportClient()


class JoyPad:
    """가상 조이스틱(RBQ JoystickThumbPad 참고): 좌드래그=축[-1,1], 놓으면 center 복귀.
       ★우클릭=고정(latch): 켜두면 놓아도 마지막 값 유지 → 마우스 한개로 전진 유지하며 조향 슬라이더 조작.
         사용법: 조이스틱 위에서 우클릭(고정ON, 손잡이 초록) → 좌드래그로 속도맞춤 → 놓아도 유지. 다시 우클릭 또는 STOP(X)=해제."""

    def __init__(self, tag, size, on_change, x_only=False):
        self.tag = tag; self.sz = size; self.R = size * 0.5 - 16; self.c = size / 2
        self.on_change = on_change; self.x_only = x_only; self.active = False
        self.latched = False

    def build(self):
        with dpg.drawlist(width=self.sz, height=self.sz, tag=self.tag):
            dpg.draw_circle([self.c, self.c], self.R, color=(80, 90, 120), fill=(28, 30, 42), thickness=2)
            dpg.draw_line([self.c - self.R, self.c], [self.c + self.R, self.c], color=(58, 62, 84))
            dpg.draw_line([self.c, self.c - self.R], [self.c, self.c + self.R], color=(58, 62, 84))
            dpg.draw_circle([self.c, self.c], self.R * 0.5, color=(52, 56, 76))
            dpg.draw_circle([self.c, self.c], self.sz * 0.15, color=(250, 195, 75),
                            fill=(238, 178, 58), tag=self.tag + '_k')

    def _loc(self):
        m = dpg.get_mouse_pos(local=False); r = dpg.get_item_rect_min(self.tag)
        return m[0] - r[0], m[1] - r[1]

    def press(self):
        if dpg.is_item_hovered(self.tag):
            self.active = True; self.move()

    def move(self):
        if not self.active:
            return
        lx, ly = self._loc(); dx = lx - self.c; dy = ly - self.c; d = math.hypot(dx, dy)
        if d > self.R and d > 0:
            dx *= self.R / d; dy *= self.R / d
        if self.x_only:
            dy = 0
        dpg.configure_item(self.tag + '_k', center=[self.c + dx, self.c + dy])
        self.on_change(dx / self.R, -dy / self.R)      # ax 우=+, ay 위=+

    def release(self):
        if self.active:
            self.active = False
            if not self.latched:                          # ★고정 아니면 center 복귀·0. 고정이면 현 값 유지(마우스 놔도 전진)
                dpg.configure_item(self.tag + '_k', center=[self.c, self.c])
                self.on_change(0.0, 0.0)

    def toggle_latch(self):                               # ★우클릭: 조이스틱 위에서만 고정/해제 토글
        if not dpg.is_item_hovered(self.tag):
            return
        self.latched = not self.latched
        dpg.configure_item(self.tag + '_k', fill=(95, 210, 120) if self.latched else (238, 178, 58))
        if not self.latched:                              # 해제 = center 복귀·정지
            dpg.configure_item(self.tag + '_k', center=[self.c, self.c]); self.on_change(0.0, 0.0)

    def clear_latch(self):                                # STOP 등 외부 정지 시 고정 해제
        self.latched = False; self.active = False
        dpg.configure_item(self.tag + '_k', center=[self.c, self.c], fill=(238, 178, 58))
        self.on_change(0.0, 0.0)


def _status():
    dpg.set_value('status', 'v=%+.2f  vy=%+.2f  w=%+.2f   |  mode=%s'
                  % (sc.cmd['v'], sc.cmd['vy'], sc.cmd['w'], sc.cmd['mode']))


def _whip_ui(gait):
    """★whip 컨트롤을 gait에 연동. gait 버튼 누르면 그 gait 기본 세팅으로 자동전환.
    체크박스는 두 gait 다 활성 — trot=속도연동 자동whip on/off, walk=whip 사용 on/off(끄면 매끈)."""
    walk = (gait == 'walk')
    # ★기본: 체크박스 두 gait 다 ON. 슬라이더=gait 전용 최적 whip.
    #   trot=앞0.1/뒤0.6(공격적 paw-tuck) · walk=앞1.0/뒤1.5(완만 whip, 직진4.1°·선회8.1° 최적)
    _auto = True
    _wf, _wr = (1.0, 1.5) if walk else (0.1, 0.6)
    if dpg.does_item_exist('swing_w_f'):
        dpg.set_value('swing_w_f', _wf); sc.SetSwingWF(_wf)
        dpg.configure_item('swing_w_f', label=(
            'walk 앞다리 whip 강도  (0.1=강 / 2.0=약; 체크박스 켜야 적용)' if walk
            else '앞다리 whip 목표  (0.1=강한 paw-tuck / 2.0=매끈)   auto on:고속목표'))
    if dpg.does_item_exist('swing_w_r'):
        dpg.set_value('swing_w_r', _wr); sc.SetSwingWR(_wr)
        dpg.configure_item('swing_w_r', label=(
            'walk 뒷다리 whip 강도  (0.6=완만 / 2.0=약)' if walk
            else '뒷다리 whip 목표  (0.6=완만 채찍질 / 2.0=매끈)'))
    if dpg.does_item_exist('auto_whip'):
        dpg.set_value('auto_whip', _auto); sc.SetAutoWhip(_auto)
        dpg.configure_item('auto_whip', label=(
            'whip 사용 (walk: 체크=슬라이더 whip 적용 / 해제=매끈)' if walk
            else '속도연동 자동 whip (on=고속서 슬라이더값까지 선형↑ / off=슬라이더 상수)'))


_last_left = [0.0, 0.0]                                 # 슬라이더 변경 시 live 재적용용 마지막 축
_last_right = [0.0, 0.0]


def _left(ax, ay):                                     # 좌스틱: 전후(ay)/측방(ax) — 풀스케일=보행속도 게이지
    _last_left[:] = [ax, ay]
    sc.Move(ay * sc.vmax, -ax * sc.vmax, sc.cmd['w']); _status()


def _right(ax, ay):                                    # 우스틱: 선회(ax)
    _last_right[:] = [ax, ay]
    sc.Move(sc.cmd['v'], sc.cmd['vy'], -ax * sc.wmax); _status()


def _set_walk_speed(v):                                # 보행속도 게이지 콜백 — vmax 갱신 + 현 입력 live 재적용
    sc.WalkSpeed(v)
    _left(_last_left[0], _last_left[1])


def _gait_preset(gait):                                # ★gait 버튼 → Walk Speed 게이지·Step Height를 그 gait 최적값으로 자동 세팅
    vmax = {'walk': 0.6, 'trot': 1.4, 'run': 2.0, 'stairs': 0.3}.get(gait, 1.4)   # 조이스틱 풀스케일(gait 상한). ★계단=느리게 0.3
    sh   = {'walk': 0.10, 'trot': 0.10, 'run': 0.08, 'stairs': 0.18}.get(gait, 0.10)  # 발 들림(고속 run=낮게·계단=높게 0.18 라이저 클리어). ★walk 0.05→0.10 복원(발 아치 자연스럽게·앞다리 뻣뻣함 해소)
    bz   = {'walk': 0.50, 'trot': 0.50, 'run': 0.48, 'stairs': 0.50}.get(gait, 0.50)  # ★gait별 최적 base height(축별 worst-util): run=0.48(발목ω 여유)·나머지 0.50. 컨트롤러 set_gait와 동일값
    if dpg.does_item_exist('ws'): dpg.set_value('ws', vmax)
    _set_walk_speed(vmax)
    if dpg.does_item_exist('sh'): dpg.set_value('sh', sh)
    sc.StepHeight(sh)
    if dpg.does_item_exist('bh'): dpg.set_value('bh', bz)   # ★base height 슬라이더도 gait 최적값 연동
    sc.BodyHeight(bz)


def _ready_preset():                                   # ★Ready(서기) → base height 슬라이더를 기본 서기 높이로 연동
    bz = 0.52                                          # 서기 명목높이 0.52(더 곧게 선 자세). 보행 최적(0.50)은 gait 버튼서 적용
    if dpg.does_item_exist('bh'): dpg.set_value('bh', bz)
    sc.BodyHeight(bz)


left = JoyPad('joyL', 200, _left)
right = JoyPad('joyR', 200, _right, x_only=True)


def _mode_btn(m):
    sc.SetMode(m); _status()


def _key(sender, app_data):                            # 키보드 백업: 화살표=이동, ,/. =선회, X=STOP
    k = app_data; s = 0.05
    if k == dpg.mvKey_Up:      sc.Move(min(sc.vmax, sc.cmd['v'] + s), sc.cmd['vy'], sc.cmd['w'])
    elif k == dpg.mvKey_Down:  sc.Move(max(-sc.vmax, sc.cmd['v'] - s), sc.cmd['vy'], sc.cmd['w'])
    elif k == dpg.mvKey_Left:  sc.Move(sc.cmd['v'], min(sc.vmax, sc.cmd['vy'] + s), sc.cmd['w'])
    elif k == dpg.mvKey_Right: sc.Move(sc.cmd['v'], max(-sc.vmax, sc.cmd['vy'] - s), sc.cmd['w'])
    elif k == dpg.mvKey_Comma:  sc.Move(sc.cmd['v'], sc.cmd['vy'], min(sc.wmax, sc.cmd['w'] + s))
    elif k == dpg.mvKey_Period: sc.Move(sc.cmd['v'], sc.cmd['vy'], max(-sc.wmax, sc.cmd['w'] - s))
    elif k in (dpg.mvKey_X, dpg.mvKey_Spacebar): sc.StopMove(); left.clear_latch(); right.clear_latch()   # STOP=고정도 해제
    elif k == dpg.mvKey_J: sc.Jump()
    else: return
    _status()


# ── 모니터링 패널 (plugin 구조: Raion/RaiSim 스타일 IMU·Actuator) ──
STATE_PATH = os.environ.get('QUAD_STATE', '/tmp/quad_state.json')


def read_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return None


class Panel:                                        # plugin 베이스(build=위젯생성 / update=상태반영)
    title = 'Panel'
    def build(self): pass
    def update(self, st): pass


class StatusPanel(Panel):
    title = 'Robot'
    def build(self):
        with dpg.collapsing_header(label='Robot State', default_open=True):
            dpg.add_text('-', tag='rs_text')
            dpg.add_text('-', tag='rs_spd', color=(150, 220, 150))   # ★cmd vs 실제 전진속도
    def update(self, st):
        dpg.set_value('rs_text', 'mode = %s     base_z = %.3f m     t = %.1f s'
                      % (st.get('mode', '?'), st.get('base_z', 0.0), st.get('t', 0.0)))
        _vc = st.get('v_cmd', 0.0); _va = st.get('v_act', 0.0)
        _gap = (_va - _vc) / _vc * 100 if abs(_vc) > 0.05 else 0.0
        dpg.set_value('rs_spd', '전진속도: 명령 %+.2f  →  실제 %+.2f m/s   (차이 %+.0f%%)' % (_vc, _va, _gap))


class IMUPanel(Panel):
    title = 'IMU'
    def build(self):
        with dpg.collapsing_header(label='IMU', default_open=True):
            dpg.add_text('Orientation  Roll / Pitch / Yaw [deg]', color=(150, 160, 190))
            dpg.add_text('-', tag='imu_rpy')
            dpg.add_text('Angular velocity  Gyro [rad/s]', color=(150, 160, 190))
            dpg.add_text('-', tag='imu_gyro')
    def update(self, st):
        r = st.get('rpy', [0, 0, 0]); g = st.get('gyro', [0, 0, 0])
        dpg.set_value('imu_rpy', 'R %+7.1f    P %+7.1f    Y %+7.1f' % (r[0], r[1], r[2]))
        dpg.set_value('imu_gyro', '%+6.2f    %+6.2f    %+6.2f' % (g[0], g[1], g[2]))


class ActuatorPanel(Panel):
    title = 'Actuators'
    def build(self):
        with dpg.collapsing_header(label='Actuators', default_open=True, tag='act_hdr'):   # ★관절 수는 상태(names)로 표시(16/17-DOF 공용)
            with dpg.table(header_row=True, row_background=True, borders_innerH=True,
                           borders_outerV=True, scrollY=True, height=270):
                for c in ('Joint', 'q [rad]', 'dq [rad/s]', 'tau [Nm]'):
                    dpg.add_table_column(label=c)
                for i in range(17):   # 최대 17칸(구 허리 모델) — 16-DOF 면 마지막 칸 숨김
                    with dpg.table_row(tag='act_row%d' % i):
                        for col in ('n', 'q', 'd', 't'):
                            dpg.add_text('-', tag='act_%s%d' % (col, i))
    def update(self, st):
        n = st.get('names', []); q = st.get('q', []); d = st.get('dq', []); t = st.get('tau', [])
        dpg.configure_item('act_hdr', label='Actuators  (%d DOF)' % len(n))
        for i in range(17): dpg.configure_item('act_row%d' % i, show=i < len(n))
        for i in range(min(17, len(n))):
            dpg.set_value('act_n%d' % i, n[i]); dpg.set_value('act_q%d' % i, '%+.2f' % q[i])
            dpg.set_value('act_d%d' % i, '%+.2f' % d[i]); dpg.set_value('act_t%d' % i, '%+.1f' % t[i])


PANELS = [StatusPanel(), IMUPanel(), ActuatorPanel()]   # ★plugin 등록: PANELS.append(MyPanel())로 패널 추가


dpg.create_context()

# 한글 폰트(없으면 ??로 표기)
_FONT = os.environ.get('GUI_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
_kf = None
if os.path.exists(_FONT):
    with dpg.font_registry():
        _kf = dpg.add_font(_FONT, 18)   # 한글 범위는 이 dpg 버전서 자동(range_hint는 deprecated no-op)

# 다크 테마(RBQ 느낌)
with dpg.theme() as _dark:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (22, 24, 32))
        dpg.add_theme_color(dpg.mvThemeCol_Button, (46, 52, 74))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (66, 74, 104))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 224, 235))
with dpg.theme() as _stop_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (170, 45, 45))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (210, 65, 65))
with dpg.theme() as _walk_theme:          # 보행=초록
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 120, 70))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 160, 95))
with dpg.theme() as _jump_theme:          # 점프=파랑
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (45, 90, 170))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (65, 120, 210))

with dpg.window(tag='main'):
    dpg.add_text('02_Leg Teleop   (RBQ 스타일 · SportClient)')
    dpg.add_separator()
    with dpg.group(horizontal=True):
        with dpg.group():
            dpg.add_text('이동  (좌스틱: ↕전후 / ↔측방)')
            left.build()
        dpg.add_spacer(width=20)
        with dpg.group():
            dpg.add_text('선회  (우스틱: ↔좌우)')
            right.build()
    dpg.add_text('★우클릭=조이스틱 고정(초록). 전진 고정 후 마우스로 아래 "허리 핸들"로 조향. 재우클릭/X=해제',
                 color=(250, 195, 75))
    dpg.add_separator()
    dpg.add_text('모션', color=(170, 175, 195))
    with dpg.group(horizontal=True):   # ★버튼 배치: 리셋 → 전원 → 눕기 → 서기 → 앉기 → 이동(Run) → 점프
        _b = dpg.add_button(label='RESET', width=80, callback=lambda: (sc.Reset(), _status()))
        dpg.bind_item_theme(_b, _stop_theme)
        _ob = dpg.add_button(label='Off 전원', width=100, callback=lambda: (_mode_btn('off'), _status()))
        dpg.bind_item_theme(_ob, _stop_theme)
        dpg.add_button(label='Ground 눕기', width=110, callback=lambda: _mode_btn('stand_down'))
        dpg.add_button(label='Ready 서기', width=110, callback=lambda: (sc.Ready(), _ready_preset(), _status()))
        dpg.add_button(label='Sit 앉기', width=100, callback=lambda: _mode_btn('sit'))   # ★뒷다리 접고 앞다리 편 앉기
        # ★초록 '이동' 버튼 = 이동(move)모드 진입(게이트는 아래 trot/run/walk 선택). walk 게이트와 혼동 방지로 'Run 이동' 표기
        _wb = dpg.add_button(label='Run 이동', width=110, callback=lambda: _mode_btn('move'))
        dpg.bind_item_theme(_wb, _walk_theme)
        _tb = dpg.add_button(label='TAMOLS 인젝션', width=130, callback=lambda: (_mode_btn('tamols'), _gait_preset('trot'), _status()))  # ★full-TAMOLS injection(발판+MPC예측GRF+HQP). 평지 V≤0.4
        dpg.bind_item_theme(_tb, _walk_theme)
        _jb = dpg.add_button(label='Jump 점프', width=100, callback=lambda: (sc.Jump(), _status()))
        dpg.bind_item_theme(_jb, _jump_theme)
    with dpg.group(horizontal=True):
        dpg.add_text('복구 순서: 전원(Off) → 눕기 → 앉기 → 서기  (단계적 기립이 넘어짐서 바로 서기보다 안전)', color=(150, 155, 175))
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text('게이트:', color=(170, 175, 195))
        dpg.add_button(label='trot 대각', width=95,
                       callback=lambda: (sc.SetGait('trot'), _whip_ui('trot'), _gait_preset('trot'), _status()))
        dpg.add_button(label='run 고속', width=90,
                       callback=lambda: (sc.SetGait('run'), _whip_ui('run'), _gait_preset('run'), _status()))
        dpg.add_button(label='walk 순차', width=95,
                       callback=lambda: (sc.SetGait('walk'), _whip_ui('walk'), _gait_preset('walk'), _status()))
        dpg.add_button(label='stairs 계단', width=95,       # ★계단 등반: 느림0.3·높은발0.18·정밀배치. 완만한 계단용
                       callback=lambda: (sc.SetGait('stairs'), _whip_ui('walk'), _gait_preset('stairs'), _status()))
        dpg.add_text('(버튼=속도게이지·발높이 자동세팅. trot 1.4 / run 2.0·발낮음 / walk 0.6)', color=(120, 125, 145))
    with dpg.group(horizontal=True):
        dpg.add_text('보행개선:', color=(170, 175, 195))
        dpg.add_checkbox(label='정지 위치홀드 (드리프트 보정)', tag='pos_hold', default_value=True,
                         callback=lambda s, a: (sc.SetPosHold(a), _status()))
    dpg.add_slider_float(label='전방 reach 게인  (0.5=표준중립·GRF균형 / ↑=발 앞으로·제동↑·뒤하중↑)', tag='raibert_k',
                         min_value=0.3, max_value=1.2, default_value=0.5,
                         callback=lambda s, a: sc.SetRaibertK(a))
    dpg.add_checkbox(label='속도연동 자동 whip (on=고속서 슬라이더값까지 선형↑ paw-tuck / off=슬라이더값 상수)', tag='auto_whip', default_value=True,
                     callback=lambda s, a: (sc.SetAutoWhip(a), _status()))
    dpg.add_slider_float(label='앞다리 whip 목표  (0.1=강한 paw-tuck / 2.0=매끈)   auto on:고속목표', tag='swing_w_f',
                         min_value=0.1, max_value=4.0, default_value=0.1,
                         callback=lambda s, a: sc.SetSwingWF(a))
    dpg.add_slider_float(label='뒷다리 whip 목표  (0.6=완만 채찍질 / 2.0=매끈)', tag='swing_w_r',
                         min_value=0.1, max_value=4.0, default_value=0.6,
                         callback=lambda s, a: sc.SetSwingWR(a))
    dpg.add_slider_float(label='허리 핸들  (자동차식 조향각 ° · 우스틱과 동일: 오른쪽=우선회 · 전진해야 돎 · 저속=tight R0.24m / 고속=자동제한)', tag='steer',
                         min_value=-68.0, max_value=68.0, default_value=0.0, format='%.0f°',
                         callback=lambda s, a: sc.SetSteer(-a * math.pi / 180.0))  # ★-부호=우스틱(-ax)과 통일: 슬라이더 오른쪽(+)=우선회
    dpg.add_separator()
    with dpg.collapsing_header(label='눕기(Ground) 자세 조각  — 뷰어 보며 CoM균형·수평·무슬라이드 튜닝', default_open=False):
        dpg.add_text('Ground(눕기) 눌러 자세 확인하며 조절. 뒷발목 접기+앞다리 fold로 수평·무슬라이드 찾기', color=(150, 155, 175))
        dpg.add_slider_float(label='눕기 높이 (낮을수록 낮게 눕음)', tag='g_lie_z',
                             min_value=0.16, max_value=0.29, default_value=0.226, format='%.3f',
                             callback=lambda s, a: sc.SetGround('g_lie_z', a))
        dpg.add_slider_float(label='뒷발목 접기 (−클수록 접힘 · 너무 접으면 뒤 들려 nose-down)', tag='g_rear_foot',
                             min_value=-1.4, max_value=-0.3, default_value=-1.15, format='%.2f',
                             callback=lambda s, a: sc.SetGround('g_rear_foot', a))
        dpg.add_slider_float(label='앞다리 thigh 오프셋 (+앞 세움=slide↓ / −앞 낮춤=CoM앞)', tag='g_front_thigh',
                             min_value=-0.4, max_value=0.4, default_value=-0.24, format='%.2f',
                             callback=lambda s, a: sc.SetGround('g_front_thigh', a))
        dpg.add_slider_float(label='앞다리 calf(무릎) 오프셋 (앞발 tuck 정도)', tag='g_front_calf',
                             min_value=-0.4, max_value=0.4, default_value=-0.4, format='%.2f',
                             callback=lambda s, a: sc.SetGround('g_front_calf', a))
    dpg.add_separator()
    dpg.add_text('속도/높이 (Walk=보행속도 게이지·live / Body=서기 높이·live / Step=발 들림)', color=(170, 175, 195))
    dpg.add_slider_float(label='Walk Speed [m/s]  (조이스틱 풀스케일 · 양 컨트롤러 공통)', tag='ws',
                         min_value=0.0, max_value=2.0, default_value=VMAX,
                         callback=lambda s, a: _set_walk_speed(a))
    dpg.add_slider_float(label='Body Height [m]  (★서기+보행 공통 · 0.42~0.52 안정범위 · 보행최적 0.50)', tag='bh',
                         min_value=0.42, max_value=0.52, default_value=0.52,
                         callback=lambda s, a: sc.BodyHeight(a))
    dpg.add_slider_float(label='Step Height [m]  (보행중 live 적용)', tag='sh',
                         min_value=0.05, max_value=0.20, default_value=0.10,
                         callback=lambda s, a: sc.StepHeight(a))
    dpg.add_separator()
    dpg.add_text('뷰어/지형/모니터 (live)', color=(170, 175, 195))
    dpg.add_slider_float(label='Sim Speed (배속)  (1=실시간, 0.25=느리게, 4=빠르게, 0=최대)', tag='rate',
                         min_value=0.0, max_value=4.0, default_value=1.0,
                         callback=lambda s, a: sc.SimRate(a))
    with dpg.group(horizontal=True):
        dpg.add_checkbox(label='지형적응 (Terrain perception)', tag='terr', default_value=True,
                         callback=lambda s, a: sc.SetTerrain(a))
        dpg.add_spacer(width=20)
        dpg.add_checkbox(label='모니터 표시 + 추정 base 고스트 (초록=GT·주황=추정 · 보행은 GT제어 유지)', tag='viz', default_value=True,
                         callback=lambda s, a: sc.SetViz(a))
        dpg.add_spacer(width=20)
        dpg.add_checkbox(label='발 배치 (③ footScore 발판선택 · 갭/스테핑 crossing)', tag='foot_nudge', default_value=False,
                         callback=lambda s, a: sc.SetFootNudge(a))
    dpg.add_separator()
    dpg.add_text('키: ↑↓=전후  ←→=측방  ,/.=선회  J=점프  X/Space=STOP', color=(140, 140, 155))
    dpg.add_text('', tag='status')
    dpg.add_text('채널: ' + CMD_PATH, color=(115, 118, 130))
    dpg.add_separator()
    dpg.add_text('모니터 (plugin: ' + ', '.join(p.title for p in PANELS) + ')', color=(170, 175, 195))
    for _p in PANELS:                                  # plugin 패널 빌드
        _p.build()
    dpg.add_text('상태채널: ' + STATE_PATH, color=(115, 118, 130))

with dpg.handler_registry():
    dpg.add_key_press_handler(callback=_key)
    dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Left, callback=lambda s, a: (left.press(), right.press()))
    dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=lambda s, a: (left.move(), right.move()))
    dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=lambda s, a: (left.release(), right.release()))
    dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=lambda s, a: (left.toggle_latch(), right.toggle_latch()))  # ★우클릭=고정/해제

_status()
dpg.create_viewport(title='02_Leg Teleop (RBQ style) + Monitor', width=540, height=820)
dpg.setup_dearpygui()
if _kf is not None:
    dpg.bind_font(_kf)
dpg.bind_theme(_dark)
dpg.show_viewport()
dpg.set_primary_window('main', True)

# ── 렌더 루프: ~20Hz 상태채널 폴링 → 패널 갱신 ──
_fc = 0
while dpg.is_dearpygui_running():
    _fc += 1
    if _fc % 3 == 0:
        _st = read_state()
        if _st:
            for _p in PANELS:
                try:
                    _p.update(_st)
                except Exception:
                    pass
    dpg.render_dearpygui_frame()
dpg.destroy_context()
