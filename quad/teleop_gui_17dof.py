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
                    'body_h': 0.52, 'step_h': 0.10, 'euler': [0.0, 0.0, 0.0], 'gait': 'trot',
                    'vmax': VMAX, 'jump_seq': 0, 'home_seq': 0, 'reset_seq': 0,
                    'rate': 1.0, 'viz': True, 'terrain': True,   # ★rate=뷰어배속 viz=모니터표시 terrain=지형적응
                    'foot_lock': True, 'pos_hold': True,         # ★터치다운 foothold lock · 정지 위치홀드 (격리 비교용)
                    'foot_lock_s': 0.35, 'raibert_k': 0.8,
                    'swing_w_f': 2.0, 'swing_w_r': 2.0}  # ★앞/뒤 whip 억제(스윙여유도, 기본2.0·최대4.0)
        self._pub()

    def SimRate(self, r):                           # 뷰어 배속(0.25~4, 0=최대) — live
        self.cmd['rate'] = float(r); self._pub()

    def SetViz(self, on):                           # 모니터 overlay(GRF/CoM/궤적/elevation) 표시 on/off — live
        self.cmd['viz'] = bool(on); self._pub()

    def SetTerrain(self, on):                       # 지형적응(perception: 발/몸높이+elevation) on/off — live
        self.cmd['terrain'] = bool(on); self._pub()

    def SetGait(self, g):                           # 게이트 walk/trot 라이브 전환(컨트롤러가 재arm으로 위상 재앵커)
        self.cmd['gait'] = str(g); self._pub()

    def SetFootLock(self, on):                       # 터치다운 foothold lock on/off (off=항상 reactive) — 격리 비교
        self.cmd['foot_lock'] = bool(on); self._pub()

    def SetPosHold(self, on):                        # 정지 위치홀드 on/off — 격리 비교
        self.cmd['pos_hold'] = bool(on); self._pub()

    def SetFootLockS(self, s):                       # 터치다운 lock 시점(낮을수록 일찍 얼림=강함)
        self.cmd['foot_lock_s'] = float(s); self._pub()

    def SetRaibertK(self, k):                        # 전방 reach 게인(↑=앞으로 더 시원하게 뻗음) — live
        self.cmd['raibert_k'] = float(k); self._pub()

    def SetSwingWF(self, w):                         # ★앞다리 whip 억제(↑=억제, ↓=whip 심함) — live
        self.cmd['swing_w_f'] = float(w); self._pub()

    def SetSwingWR(self, w):                         # ★뒷다리 whip 억제 — live
        self.cmd['swing_w_r'] = float(w); self._pub()

    def BodyHeight(self, h):                        # 서기 높이[m] (★보행중 무시 — 자세모드서만)
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
        if m != 'move':
            self.cmd.update(v=0.0, vy=0.0, w=0.0)      # 자세전환 = 속도 0
        self._pub()

    # ── RBQ 모션 API 별칭(이름 그대로) ──
    def motionDynamicWalk(self):  self.SetMode('move')
    def motionStaticReady(self):  self.SetMode('stand_up')     # 서기
    def motionStaticGround(self): self.SetMode('stand_down')   # 눕기


sc = SportClient()


class JoyPad:
    """가상 조이스틱(RBQ JoystickThumbPad 참고): 드래그=축[-1,1], 놓으면 center 복귀."""

    def __init__(self, tag, size, on_change, x_only=False):
        self.tag = tag; self.sz = size; self.R = size * 0.5 - 16; self.c = size / 2
        self.on_change = on_change; self.x_only = x_only; self.active = False

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
            dpg.configure_item(self.tag + '_k', center=[self.c, self.c])
            self.on_change(0.0, 0.0)


def _status():
    dpg.set_value('status', 'v=%+.2f  vy=%+.2f  w=%+.2f   |  mode=%s'
                  % (sc.cmd['v'], sc.cmd['vy'], sc.cmd['w'], sc.cmd['mode']))


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
    elif k in (dpg.mvKey_X, dpg.mvKey_Spacebar): sc.StopMove()
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
        with dpg.collapsing_header(label='Actuators  (14 DOF)', default_open=True):
            with dpg.table(header_row=True, row_background=True, borders_innerH=True,
                           borders_outerV=True, scrollY=True, height=270):
                for c in ('Joint', 'q [rad]', 'dq [rad/s]', 'tau [Nm]'):
                    dpg.add_table_column(label=c)
                for i in range(14):
                    with dpg.table_row():
                        for col in ('n', 'q', 'd', 't'):
                            dpg.add_text('-', tag='act_%s%d' % (col, i))
    def update(self, st):
        n = st.get('names', []); q = st.get('q', []); d = st.get('dq', []); t = st.get('tau', [])
        for i in range(min(14, len(n))):
            dpg.set_value('act_n%d' % i, n[i]); dpg.set_value('act_q%d' % i, '%+.2f' % q[i])
            dpg.set_value('act_d%d' % i, '%+.2f' % d[i]); dpg.set_value('act_t%d' % i, '%+.1f' % t[i])


PANELS = [StatusPanel(), IMUPanel(), ActuatorPanel()]   # ★plugin 등록: PANELS.append(MyPanel())로 패널 추가


dpg.create_context()

# 한글 폰트(없으면 ??로 표기)
_FONT = os.environ.get('GUI_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
_kf = None
if os.path.exists(_FONT):
    with dpg.font_registry():
        _kf = dpg.add_font(_FONT, 18)

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
with dpg.theme() as _jump_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 120, 70))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 160, 95))

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
    dpg.add_separator()
    dpg.add_text('모션', color=(170, 175, 195))
    with dpg.group(horizontal=True):
        dpg.add_button(label='Ready 서기', width=120, callback=lambda: (sc.Ready(), _status()))
        dpg.add_button(label='Ground 눕기', width=120, callback=lambda: _mode_btn('stand_down'))
        dpg.add_button(label='Walk 보행', width=120, callback=lambda: _mode_btn('move'))
        _jb = dpg.add_button(label='Jump 점프', width=110, callback=lambda: (sc.Jump(), _status()))
        dpg.bind_item_theme(_jb, _jump_theme)
        _b = dpg.add_button(label='RESET', width=90, callback=lambda: (sc.Reset(), _status()))
        dpg.bind_item_theme(_b, _stop_theme)
    with dpg.group(horizontal=True):
        _ob = dpg.add_button(label='Off 전원(쓰러짐)', width=130, callback=lambda: (_mode_btn('off'), _status()))
        dpg.bind_item_theme(_ob, _stop_theme)
        dpg.add_text('recovery 순서: Off(쓰러짐) → Ground(눕혀 정리) → Ready(일어서기)', color=(150, 155, 175))
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text('게이트:', color=(170, 175, 195))
        dpg.add_button(label='trot 대각', width=100,
                       callback=lambda: (sc.SetGait('trot'), _status()))
        dpg.add_button(label='walk 순차', width=100,
                       callback=lambda: (sc.SetGait('walk'), _status()))
        dpg.add_text('(trot=빠름 / walk=정적안정·저속)', color=(120, 125, 145))
    with dpg.group(horizontal=True):
        dpg.add_text('보행개선:', color=(170, 175, 195))
        dpg.add_checkbox(label='터치다운 lock', tag='foot_lock', default_value=True,
                         callback=lambda s, a: (sc.SetFootLock(a), _status()))
        dpg.add_checkbox(label='정지 위치홀드', tag='pos_hold', default_value=True,
                         callback=lambda s, a: (sc.SetPosHold(a), _status()))
        dpg.add_text('(각각 끄고 비교)', color=(120, 125, 145))
    dpg.add_slider_float(label='터치다운 lock 강도  (낮을수록 강함: 0=스윙시작부터 고정 / 1=reactive)', tag='foot_lock_s',
                         min_value=0.0, max_value=1.0, default_value=0.35,
                         callback=lambda s, a: sc.SetFootLockS(a))
    dpg.add_slider_float(label='전방 reach 게인  (0.8=기본, 시원한 reach / ↑=제동↑ 느림+안정 / 1.2=과제동)', tag='raibert_k',
                         min_value=0.3, max_value=1.2, default_value=0.8,
                         callback=lambda s, a: sc.SetRaibertK(a))
    dpg.add_slider_float(label='앞다리 whip 억제  (0.1=whip심함 / 2.0=기본 / ~8=거의whip제거·포화)  ↓낮추면 앞발 채찍질(paw-tuck)', tag='swing_w_f',
                         min_value=0.1, max_value=10.0, default_value=2.0,
                         callback=lambda s, a: sc.SetSwingWF(a))
    dpg.add_slider_float(label='뒷다리 whip 억제  (0.1=whip심함 / 2.0=기본 / ~8=거의whip제거·포화)', tag='swing_w_r',
                         min_value=0.1, max_value=10.0, default_value=2.0,
                         callback=lambda s, a: sc.SetSwingWR(a))
    dpg.add_separator()
    dpg.add_text('속도/높이 (Walk=보행속도 게이지·live / Body=서기 높이·live / Step=발 들림)', color=(170, 175, 195))
    dpg.add_slider_float(label='Walk Speed [m/s]  (조이스틱 풀스케일 · 양 컨트롤러 공통)', tag='ws',
                         min_value=0.0, max_value=2.0, default_value=VMAX,
                         callback=lambda s, a: _set_walk_speed(a))
    dpg.add_slider_float(label='Body Height [m]  (서기만, 보행중 무시)', tag='bh',
                         min_value=0.34, max_value=0.52, default_value=0.52,
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
        dpg.add_checkbox(label='모니터 표시 (GRF/CoM/궤적/elevation)', tag='viz', default_value=True,
                         callback=lambda s, a: sc.SetViz(a))
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
    dpg.add_mouse_down_handler(callback=lambda s, a: (left.press(), right.press()))
    dpg.add_mouse_drag_handler(callback=lambda s, a: (left.move(), right.move()))
    dpg.add_mouse_release_handler(callback=lambda s, a: (left.release(), right.release()))

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
