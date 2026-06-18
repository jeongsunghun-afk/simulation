"""teleop_gui — 02_Leg/go2 제어 GUI 패널 (Dear PyGui).

GUI는 컨트롤러와 분리된 '명령 발행측'. **Unitree SportClient 시그니처 그대로** 따른다
(Move/StopMove/BalanceStand/StandUp/StandDown/BodyHeight/Euler/SwitchGait).
명령을 채널(현재 JSON 파일)로 발행 → 컨트롤러는 `--cmd json` 으로 소비(sim/실 동일).
배포 시 SportClient._publish 백엔드만 ROS2/DDS(LowCmd 상위) 로 교체하면 됨.

사용:
  ① 컨트롤러: python quad_proxddp.py --test loop --robot ours --exec-robot ours_sphere --cmd json --gait-T 0.35 --step-h 0.05
  ② GUI:      python teleop_gui.py
키: ↑↓=vx  ←→=vy  ,/. =w  X(또는 Space)=STOP
"""
import os
import json
import dearpygui.dearpygui as dpg

CMD_PATH = os.environ.get('QUAD_CMD', '/tmp/quad_cmd.json')
VMAX = float(os.environ.get('VMAX', '0.5'))      # 컨트롤러 JsonCmd vmax 와 동일
WMAX = float(os.environ.get('WMAX', '0.25'))
STEP_V, STEP_W = 0.05, 0.05
JR = 95; JCX = 110; JCY = 110           # 조이스틱 반경·중심(드로잉 좌표)
_jdrag = {'on': False}


class SportClient:
    """Unitree SportClient 시그니처 그대로의 명령 발행기.
       백엔드=JSON 원자적쓰기(배포시 ROS2/DDS LowCmd 상위로 교체).
       컨트롤러(JsonCmd)는 v/vy/w 소비. mode/body_height/euler 는 프로토콜 준비(추후 컨트롤러 연동)."""

    def __init__(self, path=CMD_PATH):
        self.path = path
        self.cmd = {'v': 0.0, 'vy': 0.0, 'w': 0.0, 'mode': 'move',
                    'body_height': 0.0, 'euler': [0.0, 0.0, 0.0], 'gait': 0}
        self._publish()

    def _publish(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.cmd, f)
        os.replace(tmp, self.path)              # 원자적 교체(컨트롤러가 부분쓰기 안 읽게)

    # ── Unitree SportClient API (이름 그대로) ──
    def Move(self, vx, vy, vyaw):
        self.cmd.update(v=vx, vy=vy, w=vyaw, mode='move'); self._publish()

    def StopMove(self):
        self.cmd.update(v=0.0, vy=0.0, w=0.0); self._publish()

    def BalanceStand(self):
        self.cmd.update(v=0.0, vy=0.0, w=0.0, mode='balance_stand'); self._publish()

    def StandUp(self):
        self.cmd.update(mode='stand_up'); self._publish()

    def StandDown(self):
        self.cmd.update(mode='stand_down'); self._publish()

    def RecoveryStand(self):
        self.cmd.update(mode='recovery_stand'); self._publish()

    def BodyHeight(self, h):
        self.cmd.update(body_height=float(h)); self._publish()

    def Euler(self, roll, pitch, yaw):
        self.cmd.update(euler=[float(roll), float(pitch), float(yaw)]); self._publish()

    def SwitchGait(self, d):
        self.cmd.update(gait=int(d)); self._publish()


sc = SportClient()


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _send_move():
    vx = dpg.get_value('vx'); vy = dpg.get_value('vy'); w = dpg.get_value('vyaw')
    sc.Move(vx, vy, w)
    dpg.set_value('status', 'Move   vx=%+.2f  vy=%+.2f  w=%+.2f   | mode=%s' %
                  (vx, vy, w, sc.cmd['mode']))


def _set_vel(vx=None, vy=None, w=None):
    if vx is not None:
        dpg.set_value('vx', _clamp(vx, -VMAX, VMAX))
    if vy is not None:
        dpg.set_value('vy', _clamp(vy, -VMAX, VMAX))
    if w is not None:
        dpg.set_value('vyaw', _clamp(w, -WMAX, WMAX))
    _send_move()


def _stop():
    dpg.set_value('vx', 0.0); dpg.set_value('vy', 0.0); dpg.set_value('vyaw', 0.0)
    sc.StopMove()
    dpg.set_value('status', 'STOP   (모든 속도 0)')


def _mode(fn, label):
    fn()
    dpg.set_value('status', label + '   | mode=%s' % sc.cmd['mode'])


# ── 키보드(↑↓ vx, ←→ vy, ,/. w, X/Space STOP) ──
def _key(sender, app_data):
    k = app_data
    if k == dpg.mvKey_Up:      _set_vel(vx=dpg.get_value('vx') + STEP_V)
    elif k == dpg.mvKey_Down:  _set_vel(vx=dpg.get_value('vx') - STEP_V)
    elif k == dpg.mvKey_Left:  _set_vel(vy=dpg.get_value('vy') + STEP_V)
    elif k == dpg.mvKey_Right: _set_vel(vy=dpg.get_value('vy') - STEP_V)
    elif k == dpg.mvKey_Comma:  _set_vel(w=dpg.get_value('vyaw') + STEP_W)
    elif k == dpg.mvKey_Period: _set_vel(w=dpg.get_value('vyaw') - STEP_W)
    elif k in (dpg.mvKey_X, dpg.mvKey_Spacebar): _stop()


# ── 가상 조이스틱(드래그=전후/측방, 놓으면 정지) ──
def _joy_apply(dx, dy):
    import math
    d = math.hypot(dx, dy)
    if d > JR: dx *= JR / d; dy *= JR / d          # 패드 안으로 클램프
    dpg.configure_item('knob', center=[JCX + dx, JCY + dy])
    _set_vel(vx=-dy / JR * VMAX, vy=-dx / JR * VMAX)   # 위=전진, 좌=+측방

def _joy_press(sender, app_data):
    if dpg.is_item_hovered('joydraw'):
        _jdrag['on'] = True
        mp = dpg.get_drawing_mouse_pos(); _joy_apply(mp[0] - JCX, mp[1] - JCY)

def _joy_move(sender, app_data):
    if _jdrag['on']:
        mp = dpg.get_drawing_mouse_pos(); _joy_apply(mp[0] - JCX, mp[1] - JCY)

def _joy_release(sender, app_data):
    if _jdrag['on']:
        _jdrag['on'] = False
        dpg.configure_item('knob', center=[JCX, JCY])
        _set_vel(vx=0.0, vy=0.0)                    # 놓으면 정지(전후/측방만; 선회는 슬라이더 유지)


dpg.create_context()

# STOP 버튼용 빨강 테마
with dpg.theme() as stop_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (170, 40, 40))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (210, 60, 60))

with dpg.window(tag='main'):
    dpg.add_text('가상 조이스틱 — 드래그=전후(↕)/측방(↔), 놓으면 정지')
    with dpg.drawlist(width=2*JR+30, height=2*JR+30, tag='joydraw'):
        dpg.draw_circle([JCX, JCY], JR, color=(90, 90, 120), fill=(38, 38, 52), thickness=2)   # 패드
        dpg.draw_line([JCX-JR, JCY], [JCX+JR, JCY], color=(70, 70, 95))                          # 십자
        dpg.draw_line([JCX, JCY-JR], [JCX, JCY+JR], color=(70, 70, 95))
        dpg.draw_circle([JCX, JCY], JR*0.5, color=(60, 60, 85))
        dpg.draw_circle([JCX, JCY], 24, color=(245, 190, 70), fill=(235, 175, 55), tag='knob')   # 노브
    dpg.add_separator()
    dpg.add_text('속도 명령 (슬라이더 드래그 = 실시간 발행)')
    dpg.add_slider_float(label='vx 전진 [m/s]', tag='vx', min_value=-VMAX, max_value=VMAX,
                         default_value=0.0, callback=lambda s, a: _send_move())
    dpg.add_slider_float(label='vy 측방 [m/s]  (+좌/−우)', tag='vy', min_value=-VMAX, max_value=VMAX,
                         default_value=0.0, callback=lambda s, a: _send_move())
    dpg.add_slider_float(label='w 선회 [rad/s]', tag='vyaw', min_value=-WMAX, max_value=WMAX,
                         default_value=0.0, callback=lambda s, a: _send_move())
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_button(label='  Move  ', callback=_send_move)
        b = dpg.add_button(label='  STOP  ', callback=_stop); dpg.bind_item_theme(b, stop_theme)
        dpg.add_button(label='BalanceStand', callback=lambda: _mode(sc.BalanceStand, 'BalanceStand'))
    with dpg.group(horizontal=True):
        dpg.add_button(label='StandUp', callback=lambda: _mode(sc.StandUp, 'StandUp'))
        dpg.add_button(label='StandDown', callback=lambda: _mode(sc.StandDown, 'StandDown'))
        dpg.add_button(label='RecoveryStand', callback=lambda: _mode(sc.RecoveryStand, 'RecoveryStand'))
    dpg.add_slider_float(label='BodyHeight [m]', tag='bh', min_value=-0.1, max_value=0.1,
                         default_value=0.0, callback=lambda s, a: sc.BodyHeight(a))
    dpg.add_separator()
    dpg.add_text('키: ↑↓=vx  ←→=vy  ,/.=w  X/Space=STOP', color=(150, 150, 150))
    dpg.add_text('', tag='status')
    dpg.add_text('채널: ' + CMD_PATH + '  (컨트롤러는 --cmd json 으로 소비)', color=(130, 130, 130))

with dpg.handler_registry():
    dpg.add_key_press_handler(callback=_key)
    dpg.add_mouse_down_handler(callback=_joy_press)      # 조이스틱 드래그
    dpg.add_mouse_drag_handler(callback=_joy_move)
    dpg.add_mouse_release_handler(callback=_joy_release)

dpg.set_value('status', 'Move   vx=+0.00  vy=+0.00  w=+0.00   | mode=move')
dpg.create_viewport(title='02_Leg Teleop  (SportClient)', width=500, height=470)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window('main', True)
dpg.start_dearpygui()
dpg.destroy_context()
