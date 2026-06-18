"""teleop_gui — 02_Leg 제어 GUI (RBQ 스타일: dual 조이스틱 + 모션 버튼).

Rainbow Robotics RBQ GUI 참고(JoystickThumbPad + motionStaticReady/Ground/DynamicWalk).
명령을 JSON 채널(/tmp/quad_cmd.json)로 발행 → 컨트롤러는 CMDFILE 로 소비(sim/실 동일).
배포 시 SportClient._pub 백엔드만 ROS2/DDS(또는 RBQ setPosRef/setTorqueRef 상위)로 교체.

사용:
  ① GUI:    python teleop_gui.py
  ② 컨트롤러: cd /home/jsh/simple-mpc && VIEW=1 CMDFILE=/tmp/quad_cmd.json ~/.pixi/bin/pixi run python examples/02leg9_fulldynamics_mujoco.py
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
        self.cmd = {'v': 0.0, 'vy': 0.0, 'w': 0.0, 'mode': 'move',
                    'body_height': 0.0, 'euler': [0.0, 0.0, 0.0], 'gait': 0}
        self._pub()

    def _pub(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.cmd, f)
        os.replace(tmp, self.path)

    def Move(self, vx, vy, vyaw):
        self.cmd.update(v=vx, vy=vy, w=vyaw); self._pub()

    def StopMove(self):
        self.cmd.update(v=0.0, vy=0.0, w=0.0); self._pub()

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


def _left(ax, ay):                                     # 좌스틱: 전후(ay)/측방(ax)
    sc.Move(ay * VMAX, -ax * VMAX, sc.cmd['w']); _status()


def _right(ax, ay):                                    # 우스틱: 선회(ax)
    sc.Move(sc.cmd['v'], sc.cmd['vy'], -ax * WMAX); _status()


left = JoyPad('joyL', 200, _left)
right = JoyPad('joyR', 200, _right, x_only=True)


def _mode_btn(m):
    sc.SetMode(m); _status()


def _key(sender, app_data):                            # 키보드 백업: 화살표=이동, ,/. =선회, X=STOP
    k = app_data; s = 0.05
    if k == dpg.mvKey_Up:      sc.Move(min(VMAX, sc.cmd['v'] + s), sc.cmd['vy'], sc.cmd['w'])
    elif k == dpg.mvKey_Down:  sc.Move(max(-VMAX, sc.cmd['v'] - s), sc.cmd['vy'], sc.cmd['w'])
    elif k == dpg.mvKey_Left:  sc.Move(sc.cmd['v'], min(VMAX, sc.cmd['vy'] + s), sc.cmd['w'])
    elif k == dpg.mvKey_Right: sc.Move(sc.cmd['v'], max(-VMAX, sc.cmd['vy'] - s), sc.cmd['w'])
    elif k == dpg.mvKey_Comma:  sc.Move(sc.cmd['v'], sc.cmd['vy'], min(WMAX, sc.cmd['w'] + s))
    elif k == dpg.mvKey_Period: sc.Move(sc.cmd['v'], sc.cmd['vy'], max(-WMAX, sc.cmd['w'] - s))
    elif k in (dpg.mvKey_X, dpg.mvKey_Spacebar): sc.StopMove()
    else: return
    _status()


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
        dpg.add_button(label='Ready 서기', width=120, callback=lambda: _mode_btn('stand_up'))
        dpg.add_button(label='Ground 눕기', width=120, callback=lambda: _mode_btn('stand_down'))
        dpg.add_button(label='Walk 보행', width=120, callback=lambda: _mode_btn('move'))
        _b = dpg.add_button(label='STOP', width=90, callback=lambda: (sc.StopMove(), _status()))
        dpg.bind_item_theme(_b, _stop_theme)
    dpg.add_separator()
    dpg.add_text('키: ↑↓=전후  ←→=측방  ,/.=선회  X/Space=STOP', color=(140, 140, 155))
    dpg.add_text('', tag='status')
    dpg.add_text('채널: ' + CMD_PATH, color=(115, 118, 130))

with dpg.handler_registry():
    dpg.add_key_press_handler(callback=_key)
    dpg.add_mouse_down_handler(callback=lambda s, a: (left.press(), right.press()))
    dpg.add_mouse_drag_handler(callback=lambda s, a: (left.move(), right.move()))
    dpg.add_mouse_release_handler(callback=lambda s, a: (left.release(), right.release()))

_status()
dpg.create_viewport(title='02_Leg Teleop (RBQ style)', width=500, height=510)
dpg.setup_dearpygui()
if _kf is not None:
    dpg.bind_font(_kf)
dpg.bind_theme(_dark)
dpg.show_viewport()
dpg.set_primary_window('main', True)
dpg.start_dearpygui()
dpg.destroy_context()
