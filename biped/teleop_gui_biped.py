"""biped 전용 슬림 GUI (17-DOF GUI 규약) — dearpygui.

명령을 JSON 채널(/tmp/biped_cmd.json)로 발행 → biped_run.py 소비.
★듀얼 조이스틱: 좌스틱=전후(vx)/측방(vy) · 우스틱=선회(wz). 버튼: 정지·현자세 · 2점 평발 stand · 점발 보행.
실행(proxddp env): /home/jsh/miniforge3/envs/proxddp/bin/python teleop_gui_biped.py
  ① 컨트롤러: python biped_run.py   ② GUI: 위 명령
"""
import os, json, math, socket, time, subprocess, threading
import dearpygui.dearpygui as dpg

CMD    = os.environ.get('QUAD_CMD',   '/tmp/biped_cmd.json')
STATE  = os.environ.get('QUAD_STATE', '/tmp/biped_state.json')
# ★Isaac Sim 배선: TELEOP_UDP="host:port" 설정 시 명령을 UDP로도 발행(원격 play.py 수신).
_UDP_TGT  = os.environ.get('TELEOP_UDP')          # 예: 192.168.1.205:9999
_udp_sock = None; _udp_addr = None
if _UDP_TGT:
    _uh, _up = _UDP_TGT.split(':'); _udp_addr = (_uh, int(_up))
    _udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# ★범위 env 오버라이드(hind_leg는 vx~2.0·yaw~0.5라 기본보다 넓게 쓸 수 있음)
VMAX   = float(os.environ.get('VMAX',   '0.15'))  # 전진 상한[m/s]
VY_MAX = float(os.environ.get('VY_MAX', '0.10'))  # 좌우 상한[m/s]
WZ_MAX = float(os.environ.get('WZ_MAX', '0.30'))  # 선회 상한[rad/s]
H_MIN, H_MAX, H_DEF = 0.36, 0.54, 0.38  # 슬라이더 전체범위·시작(2점)기본
H_DEF_1PT, H_DEF_2PT = 0.48, 0.38       # ★접촉모드별 기본 몸통높이. 1점 0.50→0.48(08-28 스윕: 강건성 — biped_mpc_wbic.py 주석)

# ── 각축(JOG) 검증용 관절 정의 — emb/config/biped_emb.yaml 있으면 로드, 없으면 기본값 ──
#   실기(app/biped_emb.py) 배포 시 축별 목표각·통신 LED로 각 모터 확인. sim에선 inert(무해).
JOG_NAMES = ['HL_hip', 'HL_thigh', 'HL_calf', 'HL_foot', 'HR_hip', 'HR_thigh', 'HR_calf', 'HR_foot']
JOG_LIM   = [(-17, 17), (-67, 32), (-27, 32), (-40, 20)] * 2   # jog 안전범위(deg)=mjcf range×0.5
# ★경로는 try 밖에 둔다 — yaml 이 없는 venv 에서도 영점 표는 이 경로를 써야 한다.
#   (run_hw.sh 는 GUI 를 ~/.venvs/gui 로 띄운다. 거기 PyYAML 이 없을 수 있다.)
_cfgp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emb', 'config', 'biped_emb.yaml')
try:
    import yaml
    _cfg = yaml.safe_load(open(_cfgp))
    _frac = float(_cfg.get('jog', {}).get('range_frac', 0.5))
    JOG_NAMES = [j['name'] for j in _cfg['joints']]
    # ★축별 예외(jog_min_deg/jog_max_deg)를 반드시 반영한다 — 2026-08-10.
    #   여기는 emb/interface/joint_map.py 의 jog 한계 규칙을 **복제**한 코드다(GUI 는 별도
    #   venv 라 joint_map 을 import 하지 않는다). 그래서 그쪽만 고치면 여기가 조용히 어긋난다.
    #   실제로 어긋났었다: calf 를 [−55, 32.5] 로 넓혔는데 GUI 는 min_deg×0.5 = −27.5 를 써서,
    #   구조적 한계(−55°)에 서 있는 무릎이 [JOG 검증] 진입 순간 **+27.5° 명령**을 받았다.
    #   ⚠joint_map 의 규칙을 바꾸면 여기도 같이 고칠 것.
    JOG_LIM = [(float(j.get('jog_min_deg', j['min_deg'] * _frac)),
                float(j.get('jog_max_deg', j['max_deg'] * _frac))) for j in _cfg['joints']]
except Exception:
    pass
NJ = len(JOG_NAMES)

# ── ★위치모드 강성 배율 (2026-08-21) ────────────────────────────────────────
#   home/hold/jog 의 kp 에 곱하는 배율. **stand/walk 는 안 쓴다**(WBIC 와 싸우면 안 된다).
#   왜 GUI 로 빼는가: 접지시켜 하중을 걸며 "이 자세를 지키려면 얼마나 세야 하나" 를 찾는
#   작업이라, 매번 제어기를 껐다 켜며 env 를 바꾸는 게 실무상 불가능하다.
#
#   ⚠올릴수록 **토크트립까지의 각도가 줄어든다.** τ_trip ÷ (kp_ch·gear_k·π/180·배율) 이다
#     — **gear_k 는 1승**이다(트립이 채널토크로 걸리므로. emb/README "트립각은 1승이다").
#     그래서 버튼마다 **가장 예민한 축의 트립각**을 같이 찍는다 — 숫자만 보고 올리면
#     접지 순간 그 축이 먼저 트립한다(calf 는 7.16°/배율이라 ×5 에서 **1.43°** 다).
#   ⚠kd 는 제어기가 **√배율**로 같이 올린다(ζ ∝ kd/√kp 보존). GUI 가 따로 안 보낸다.
# ★2026-08-21 ×5 까지로는 부족했다(사용자: "5배는 해야 잘될 때가 있다") → ×10 까지.
#   ⚠deploy 의 POS_KP_SCALE_MAX 도 같이 올려야 한다 — 거기서 클램프한다.
KP_STEPS = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
# ★kd 배율 (2026-08-21). None = **자동(√kp)** — ζ ∝ kd/√kp 보존이 원칙이다.
#   그런데 kd 는 **속도잡음을 그대로 토크로 증폭**한다(τ_ripple = kd_ch × dq_noise).
#     정지 중 잡음 ±7dps 면 kp×10(kd×3.16)에서 hip 이 2.3Nm — 트립 15Nm 의 15%.
#   ⇒ 잡음이 지배해 틱틱거리면 ζ 를 좀 포기하고 낮춘다. 진동이 나면 올린다.
KD_STEPS = [None, 1.0, 1.5, 2.0, 3.0]

# ── ★무중력(중력보상) 배율 (2026-08-24) ──────────────────────────────────────
#   `float` 모드에서 τ_ff = GRAV_SCALE · G_model(q) 로 나간다.
#   ★쓰는 법 — **중립점을 브래킷한다**: 올리며 다리가 뜨기 시작하는 g⁺,
#     내리며 지기 시작하는 g⁻ 를 잡으면 마찰이 소거된다. g* = (g⁺+g⁻)/2.
#     그 g* 가 곧 "지금 중력보상이 몇 % 모자라나" 다 — stand 처짐의 크기와 같다.
GRAV_STEPS = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30]
try:
    _tt = float(_cfg.get('safety', {}).get('tau_trip_nm', 15.0))
    # ★★트립은 **채널토크**로 걸린다(biped_deploy 가 hs.tau_nm 을 그대로 비교한다).
    #   ⇒ 트립이 발생하는 **관절**토크는 tau_trip × gear_k 다. 종전엔 tau_trip 을 관절토크로
    #     취급해 calf 를 1.5배·foot 을 1.2배 **보수적으로** 찍었다(실제보다 좁게 표시).
    #       calf 예: 표시 4.77°/배율 → 실제 7.16°/배율
    #   kp_raw = kp_ch·gear_k²  (emb/README "게인 이름 규칙"). ★관절 게인이 아니다 —
    #   스칼라 관절게인은 발목 커플링 때문에 존재하지 않는다.
    #   ⇒ 트립각 = (tau_trip·gear_k) / (kp_ch·gear_k²·배율) = tau_trip / (kp_ch·gear_k·배율)
    _e1 = [(j['name'], _tt / (float(j['kp']) * float(j.get('gear_k', 1.0))) * 180.0 / math.pi)
           for j in _cfg['joints']]
    _tight = min(_e1, key=lambda t: t[1])       # 트립각이 가장 작은 축 = 가장 예민하다
    KP_TRIP = [(_tight[0], _tight[1] / s) for s in KP_STEPS]
except Exception:
    KP_TRIP = [('?', float('nan'))] * len(KP_STEPS)


class Pub:
    def __init__(self, path=CMD):
        self.path = path
        # ★시작 모드 — 'stand' 는 절대 금지. Pub() 이 생성 시점에 곧바로 _pub() 하므로
        #   (뷰포트 생성 전!) **기동 4초 내 자동 무장**되고 모델기반 제어가 돌아버린다.
        #
        # ★2026-08-07: 'off' → 'hold'. 실기에서 Emb 가 4.5초 램프로 잡아둔 자세를
        #   GUI 가 off 를 쏘는 순간 놓아버려 **다리가 떨어졌다**(hip 중력토크 4.96 Nm).
        #   (2026-08-26 정정: 그 램프가 가는 곳은 0° 가 아니라 **측정각**이다 — 제자리를
        #    잡고 있을 뿐이다. 그래도 "잡고 있던 것을 놓으면 떨어진다" 는 인과는 그대로다.)
        #   hold 는 "지금 그 자리를 유지" 라 인계 시 움직임이 0 이다.
        #   ⚠stand/walk 와 달리 hold 는 모델기반 제어가 아니다 — 측정각 임피던스 유지뿐.
        #   sim 에서는 컨트롤러가 hold 를 모르면 무시하므로 무해하다.
        # ★pos_kp_scale=1.0 으로 시작한다 — GUI 를 띄우는 것만으로 강성이 바뀌면 안 된다.
        #   (제어기 쪽 env POS_KP_SCALE 은 **이 값이 도착하는 순간 덮인다.** GUI 를 쓸 거면
        #    env 로 주지 말고 여기 버튼으로 줄 것.)
        self.cmd = {'v': 0.0, 'vy': 0.0, 'w': 0.0, 'body_h': H_DEF, 'mode': 'hold', 'contact': '2pt',
                    'jog_deg': [0.0] * NJ, 'pos_kp_scale': 1.0, 'grav_scale': 1.0, 'seq': 0}
        # ★발행 잠금 (2026-09-03) — 하트비트가 별도 스레드로 옮겨가면서(아래) 콜백(메인
        #   스레드)과 동시에 cmd 를 만질 수 있다. dict 순회 중 변경은 예외로 터진다.
        self._lk = threading.RLock()
        self._pub()

    def set_jog(self, i, val):
        self.cmd['jog_deg'][i] = float(val); self._pub()

    def set(self, **kw):
        with self._lk:
            self.cmd.update(kw)
        # ★스틱/슬라이더로 非零 속도가 들어오면 자동 walk 전환.
        #   (안 그러면 sim이 mode=stand를 보고 속도를 0으로 무시 → "명령이 안 먹는" 증상)
        # ★'off' 를 자동승격 대상에서 뺐다. off 는 "아직 무장 안 함" 상태이므로
        #   조이스틱 한 번에 stand 를 건너뛰고 walk 로 뛰어드는 것을 막는다.
        if self.cmd.get('mode') in ('stand',) and (
            abs(self.cmd.get('v', 0.0)) > 1e-3 or abs(self.cmd.get('vy', 0.0)) > 1e-3
            or abs(self.cmd.get('w', 0.0)) > 1e-3):
            self.cmd['mode'] = 'walk'
        self._pub()

    def _pub(self):
        # ★seq 를 증가시킨다. emb 앱의 워치독은 "파일이 읽히는가" 가 아니라
        #   "명령 내용이 바뀌는가" 로 살아있음을 판정하므로(biped_emb.read_cmd_fresh),
        #   정적 파일은 통신두절과 구분되지 않는다. seq 가 그 구분을 만든다.
        # ★2026-09-04 경합 수정: 하트비트 데몬스레드와 콜백스레드가 **같은 .tmp 를 공유**해
        #   A 가 쓴 tmp 를 B 가 replace 로 채가면 A 의 os.replace 가 FileNotFoundError 로
        #   스레드를 죽였다(실기 GUI DEAD). ⇒ ①write+replace 를 락 안에서 원자적으로
        #   ②스레드별 고유 tmp(pid+tid) ③replace 예외를 삼켜 어떤 경합도 GUI 를 안 죽인다.
        try:
            with self._lk:
                self.cmd['seq'] = int(self.cmd.get('seq', 0)) + 1
                body = json.dumps(self.cmd)
                tmp = '%s.%d.tmp' % (self.path, threading.get_ident())
                with open(tmp, 'w') as f:
                    f.write(body)
                os.replace(tmp, self.path)
        except Exception:
            pass
        if _udp_sock is not None:                      # ★Isaac Sim으로 UDP 발행
            try: _udp_sock.sendto(json.dumps(self.cmd).encode(), _udp_addr)
            except Exception: pass


class JoyPad:
    """가상 조이스틱(17-DOF GUI 참조). 좌드래그=축[-1,1]·놓으면 복귀·우클릭=고정."""
    def __init__(self, tag, size, on_change, x_only=False, cross_only=False):
        self.tag = tag; self.sz = size; self.R = size * 0.5 - 16; self.c = size / 2
        self.on_change = on_change; self.x_only = x_only; self.cross_only = cross_only
        self.active = False; self.latched = False

    def build(self, label=''):
        with dpg.drawlist(width=self.sz, height=self.sz, tag=self.tag):
            dpg.draw_circle([self.c, self.c], self.R, color=(80, 90, 120), fill=(28, 30, 42), thickness=2)
            dpg.draw_line([self.c - self.R, self.c], [self.c + self.R, self.c], color=(58, 62, 84))
            dpg.draw_line([self.c, self.c - self.R], [self.c, self.c + self.R], color=(58, 62, 84))
            if label:
                dpg.draw_text([8, 6], label, color=(120, 140, 170), size=14)
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
        lx, ly = self._loc(); dx = lx - self.c; dy = ly - self.c
        dx = max(-self.R, min(self.R, dx)); dy = max(-self.R, min(self.R, dy))
        if self.x_only:
            dy = 0
        if self.cross_only:                                 # ★십자(4-way): 우세 축만 = vx·vy 동시 금지(대각 marginal 방지)
            if abs(dx) >= abs(dy): dy = 0
            else: dx = 0
        dpg.configure_item(self.tag + '_k', center=[self.c + dx, self.c + dy])
        self.on_change(dx / self.R, -dy / self.R)           # ax 우=+, ay 위=+

    def release(self):
        if self.active:
            self.active = False
            if not self.latched:
                dpg.configure_item(self.tag + '_k', center=[self.c, self.c]); self.on_change(0.0, 0.0)

    def toggle_latch(self):
        if not dpg.is_item_hovered(self.tag):
            return
        self.latched = not self.latched
        dpg.configure_item(self.tag + '_k', fill=(95, 210, 120) if self.latched else (238, 178, 58))
        if not self.latched:
            dpg.configure_item(self.tag + '_k', center=[self.c, self.c]); self.on_change(0.0, 0.0)

    def clear(self):
        self.latched = False; self.active = False
        dpg.configure_item(self.tag + '_k', center=[self.c, self.c], fill=(238, 178, 58))
        self.on_change(0.0, 0.0)


pub = Pub()


# ★★하트비트 스레드 (2026-09-03 — 자립 중 낙하 사고 방지).
#   종전엔 파일 하트비트(20Hz)가 **렌더 루프 안**에 있었다. 실기에서 렌더가 1.16초
#   멈추자(창 조작/스톨) 발행이 같이 멈췄고, 배포기 워치독(0.5s)이 **크레인 없이
#   자립 중이던 로봇을 limp 로 떨궜다.** 발행 생존이 화면 프레임에 묶여 있으면 안 된다.
#   ⇒ 데몬 스레드가 렌더와 무관하게 20Hz 로 발행한다. GUI 가 완전히 죽으면(프로세스
#     종료) 스레드도 죽고 워치독이 잡는다 — 그건 의도된 동작이다.
def _hb_thread():
    while True:
        try:
            pub._pub()
        except Exception:
            pass
        time.sleep(0.05)


threading.Thread(target=_hb_thread, daemon=True).start()
_expo = lambda a: a * abs(a)          # 중앙 미세·끝 최대


def on_left(ax, ay):                  # 좌스틱: 전후(ay=vx) · 측방(ax=vy)
    v = round(_expo(ay) * VMAX, 3)
    vy = round(_expo(-ax) * VY_MAX, 3)   # 스틱 좌 = +vy(좌)
    pub.set(v=v, vy=vy)
    dpg.set_value('spd_sl', v); dpg.set_value('vy_sl', vy)


def on_right(ax, _ay):                # 우스틱: 선회(ax=wz)
    w = round(_expo(-ax) * WZ_MAX, 3)    # 스틱 우 = 우선회(−wz)
    pub.set(w=w); dpg.set_value('turn_sl', w)


def on_vx(_, val): pub.set(v=round(val, 3))
def on_vy(_, val): pub.set(vy=round(val, 3))
def on_turn(_, val): pub.set(w=round(val, 3))
def on_height(_, val): pub.set(body_h=round(val, 3))


def on_jog(sender, val, i):           # 각축 슬라이더 → 목표각(deg) 발행
    pub.set_jog(i, val)


_kd_sel = [None]                      # 현재 고른 kd 배율(None = 자동 √kp)


def _kp_now():
    v = pub.cmd.get('pos_kp_scale', 1.0)
    return float(v) if v else 1.0


def _refresh_gain_lbl():
    s = _kp_now()
    kd = math.sqrt(s) if _kd_sel[0] is None else _kd_sel[0]
    who, trip = KP_TRIP[KP_STEPS.index(s)] if s in KP_STEPS else ('?', float('nan'))
    # ζ 는 kd/√kp 에 비례한다 — 자동이면 1.00 유지, 낮추면 그만큼 저감쇠가 된다.
    zeta = kd / math.sqrt(s) if s > 0 else 1.0
    tag = '자동(√kp)' if _kd_sel[0] is None else '수동'
    dpg.set_value('kp_lbl', f'kp×{s:g}  ·  kd×{kd:.2f} {tag}  ·  ζ {zeta:.2f}배  ·  '
                            f'트립 예민축 {who} {trip:.2f}°')
    dpg.configure_item('kp_lbl', color=(210, 120, 100) if (trip < 2.0 or zeta < 0.7)
                                 else (150, 155, 175))


def set_kp_scale(s):
    """위치모드(home/hold/jog) 강성 배율. 제어기가 1초에 걸쳐 램프한다."""
    pub.set(pos_kp_scale=float(s))
    for k, v in enumerate(KP_STEPS):
        dpg.bind_item_theme(f'kpbtn_{k}', _kp_on if abs(v - s) < 1e-6 else _kp_off)
    _refresh_gain_lbl()


def set_grav_scale(g):
    """무중력(float) 모드의 중력보상 배율. **중립점 탐색용**이다.

    ★판정: 배율을 올려 다리가 **뜨기 시작**하면 g⁺, 내려 **지기 시작**하면 g⁻.
      마찰(관절 0.6~0.9Nm)이 데드밴드를 만들므로 **양방향 평균**이 참값이다.
      g* > 1 이면 제어기의 중력보상이 그만큼 모자라다는 뜻이다.
    """
    pub.set(grav_scale=float(g))
    for k, v in enumerate(GRAV_STEPS):
        dpg.bind_item_theme(f'gvbtn_{k}', _kp_on if abs(v - g) < 1e-6 else _kp_off)
    dpg.set_value('gv_lbl', f'×{g:.2f}   (중립 g* 를 찾는다 — 뜨면 내리고 지면 올린다)')


def set_kd_scale(d):
    """kd 배율. None = 자동(√kp, ζ 보존). 숫자로 주면 제어기가 그 값을 쓴다.

    ★낮추면 토크 리플이 줄지만 ζ 가 같이 떨어진다 — 틱틱거림(잡음)과 진동(저감쇠)은
      **다른 증상**이다. 정지 중 떨면 낮추고, 움직임 끝에 출렁이면 올린다.
    """
    _kd_sel[0] = d
    pub.set(pos_kd_scale=(-1.0 if d is None else float(d)))
    for k, v in enumerate(KD_STEPS):
        on = (v is None and d is None) or (v is not None and d is not None and abs(v - d) < 1e-6)
        dpg.bind_item_theme(f'kdbtn_{k}', _kp_on if on else _kp_off)
    _refresh_gain_lbl()


# ── ★발밀기(push) — z-힘 제어 (2026-08-25) ──────────────────────────────
#   발밑 저울로 α 를 **외부 기준**으로 재는 모드. τ = g*·G + Jᵀ(0,0,−F).
#   ★한 점 값은 관절마찰(축당 2~3 N) 때문에 흐리다 — **0→50→0 램프 왕복의 평균**으로
#     읽는다(g* 브래킷과 같은 소거). 제어기가 5 N/s 로 램프하므로 버튼은 목표만 정한다.
PUSH_STEPS = [0, 10, 20, 30, 40, 50]
_push = [0.0, 0]                      # [목표 N, 다리(0=HL·1=HR)]


def set_push_fz(f):
    if f is None:                          # ★user_data 미전달 등 — 조용한 사망 방지
        print('[gui] set_push_fz(None) — user_data 누락?'); return
    _push[0] = float(f)
    pub.set(push_fz=float(f), push_leg=int(_push[1]))
    # ★강조는 여기서 하지 않는다 — 실제 적용값(state.push_fz) 기준으로
    #   _refresh_mode_led 가 켠다. 클릭 반응은 라벨의 '목표' 표시가 준다.
    dpg.set_value('push_lbl', f'{"HR" if _push[1] else "HL"} · 목표 {f:g} N 으로 램프 중…')


def set_push_leg(l):
    _push[1] = int(l)
    pub.set(push_leg=int(l), push_fz=float(_push[0]))
    for k in (0, 1):
        dpg.bind_item_theme(f'plbtn_{k}', _kp_on if k == l else _kp_off)
    dpg.set_value('push_lbl', f'{"HR" if l else "HL"} · 목표 {_push[0]:g} N (램프 5 N/s)')


# ── ★hold 중력지지율 (2026-08-28) ────────────────────────────────────────────
#   hold 는 순수 위치 PD 라 중력을 **오차로만** 이긴다 → 처짐(실측 foot 8.2°).
#   강성을 올려 처짐을 줄이려다 ×3·×5 에서 HL_calf 가 주저앉았으므로, 게인 대신
#   **필요한 토크를 전방보상으로 직접 준다.** 100% = 한 다리가 mg/2 를 떠받침.
#   총 명령토크는 그대로고(중력 요구량은 자세가 정한다) 오차만 0 으로 간다 —
#   그래서 트립 여유가 나빠지지 않는다. 이게 강성 상향과 결정적으로 다른 점이다.
HOLD_FF_STEPS = [0, 25, 50, 75, 100]
# ★좌우 배분 (2026-09-02 실기 1차): 50:50 고정이었더니 HR_foot 이 −3.3° 과보상
#   = 실제 하중이 HL 쏠림. 양발 잔차가 대칭이 되도록 운전자가 트림한다(2%/s 램프).
HOLD_SPLIT_STEPS = [40, 45, 50, 55, 60]
_holdff = [0.0, 50.0]                 # [지지%, HL배분%]


def set_hold_ff(p):
    if p is None:
        print('[gui] set_hold_ff(None) — user_data 누락?'); return
    _holdff[0] = float(p)
    pub.set(hold_ff_pct=float(p), hold_ff_split=float(_holdff[1]))
    dpg.set_value('hff_lbl', f'목표 {p:g} % 로 램프 중…')


def set_hold_split(sp):
    if sp is None:
        print('[gui] set_hold_split(None) — user_data 누락?'); return
    _holdff[1] = float(sp)
    pub.set(hold_ff_split=float(sp), hold_ff_pct=float(_holdff[0]))
    dpg.set_value('hff_lbl', f'배분 목표 HL {sp:g} % 로 램프 중…')


def jog_zero():                       # 전체 0(home)
    for i in range(NJ):
        dpg.set_value(f'jog_{i}', 0.0)
    pub.set(jog_deg=[0.0] * NJ)


# ── ★영점 (2026-08-24) ───────────────────────────────────────────────────
#   버튼 **한 번**에 셋을 순서대로 한다:
#       ① calib_zero.py                → 계산. 표를 창에 찍는다(무엇이 박히는지 보인다)
#       ② calib_zero.py --apply        → config 의 offset_deg
#       ③ gen_grav_table.py --apply    → spec 의 중력표
#
#   ★③이 이 버튼의 존재 이유다. 중력표는 **채널각으로 색인**돼 있어서, offset 만 바꾸고
#     표를 두면 8축 전부 자기 offset 만큼 밀린 중력보상을 쓴다. 실제로 그렇게 됐었다
#     (HR_calf offset 13.05°). 손으로 두 명령을 치는 한 언제든 또 빠뜨린다 — 버튼이 묶는다.
#   ★그래서 ③의 인터프리터를 **①보다 먼저** 확인하고, 없으면 아예 시작하지 않는다.
#     반쪽 상태(offset 새것 + 표 낡은것)를 만드느니 아무것도 안 하는 게 낫다.
#
#   ★`--force` 는 **절대 안 넣는다.** 2026-08-21(cc321fc) 에 그걸 강행해 HR_calf 가
#     모델각 −8.70° 인 채로 박혔고, 그 커밋은 "이상하면 여기를 볼 것" 이라 적어 뒀다.
#     지금 thigh 좌우 비대칭이 그 후과로 의심된다. ⇒ 3° 문턱은 살려 둔다.
#
#   ⚠**삭제된 '제어기 재시작' 버튼과 다르다.** 그건 `emb_ctl.sh` 의 가드(중복기동·로그
#     필터·신선도)를 GUI 가 **우회**해서 지웠다. 이건 우회할 가드가 없다 —
#     calib_zero.py 의 게이트(상태 신선도 3s · off 모드 · 정지 8s · 재현성 · 변화량 3°)는
#     전부 **도구 안**에 있고 GUI 는 그 도구를 그대로 실행할 뿐이다.
#     ⇒ 그래서 여기서 게이트를 **다시 구현하지 않는다.** 도구가 거부하면 그 말을 그대로 띄우고
#       거기서 멈춘다(①이 막히면 ②③은 아예 안 돈다).
#       (게이트를 GUI 에도 복사하면 도구가 바뀔 때 조용히 갈라진다 — calib_zero.py 가
#        환산식 복사본을 갖고 있다가 stale 이 됐던 것과 같은 실수다.)
#
#   ★`python3` 로 띄운다 — sys.executable 이 아니다. GUI 를 다른 인터프리터로 띄웠어도
#     문서에 적힌 실행법과 **같은 것**이 돌아야 한다(yaml·numpy 는 python3 에 있다).
#     단 ③은 mujoco 가 필요해 따로 찾는다(Pi 시스템 python 에는 없다 — 도구 주석).
_CALIB_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emb', 'diag', 'calib_zero.py')
_GRAV_PY  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'gen_grav_table.py')
_EMB_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emb')
_calib_busy = [False]
_calib_buf  = ['']          # 리스트 = 클로저 없이 가변(위 _last_file_hb 와 같은 관용)


# ── ★CPU·온도 (2026-08-24) — 계측은 sysload.py 가 한다 ──────────────────
#   ★여기에 복사하지 않는다. 텍스트 모니터(monitor_state.py)도 같은 모듈을 쓴다 —
#     양쪽에 복사하면 한쪽만 고쳐지고 조용히 갈라진다(이 파일의 JOG_LIM 주석 참조).
_sysload = [None]
try:
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sysload as _sysload_mod
    _sysload[0] = _sysload_mod
except Exception:
    pass
_last_sys = [0.0]


# ── ★모드/힘 LED — **제어기가 보고하는 실제 상태** 기준 (2026-08-25) ─────
#   버튼 클릭이 아니라 state 의 mode·push_fz 로 켠다. 그래야 "명령이 안 먹었는데
#   불만 들어오는" 거짓 표시가 없다 — 클릭해도 제어기가 거부(가드·옛 바이너리)하면
#   불이 안 들어와서 **그 자체가 진단**이 된다.
_MODE_BTNS = {'hold': 'mbtn_hold', 'off': 'mbtn_off', 'jog': 'mbtn_jog',
              'soft_off': 'mbtn_softoff',
              'float': 'mbtn_float', 'push': 'mbtn_push', 'home': 'mbtn_home',
              'stand': 'mbtn_stand', 'walk': 'mbtn_walk'}


def _refresh_mode_led(st):
    md = st.get('mode')
    for m, t in _MODE_BTNS.items():
        try: dpg.bind_item_theme(t, _kp_on if m == md else _kp_off)
        except Exception: pass
    # 힘 버튼 — **실제 램프된 값**이 그 단계에 도달했을 때만 켠다(램프 중엔 아무것도 안 켜짐)
    pf = st.get('push_fz')
    if pf is not None:
        for k, v in enumerate(PUSH_STEPS):
            try: dpg.bind_item_theme(f'pfbtn_{k}', _kp_on if abs(pf - v) < 0.5 else _kp_off)
            except Exception: pass
        try:
            tgt = pub.cmd.get('push_fz', 0.0)
            dpg.set_value('push_lbl', f'{"HR" if _push[1] else "HL"} · 적용 {pf:.1f} N'
                                      + (f' → 목표 {tgt:g}' if abs(pf - tgt) > 0.5 else ''))
        except Exception: pass
    # 중력지지 버튼 — 힘 버튼과 같은 규칙(**실제 램프된 값**이 도달했을 때만 켠다)
    hp = st.get('hold_ff_pct')
    if hp is not None:
        for k, v in enumerate(HOLD_FF_STEPS):
            try: dpg.bind_item_theme(f'hffbtn_{k}', _kp_on if abs(hp - v) < 1.0 else _kp_off)
            except Exception: pass
        hsp = st.get('hold_ff_split', 50.0)
        for k, v in enumerate(HOLD_SPLIT_STEPS):
            try: dpg.bind_item_theme(f'hspbtn_{k}', _kp_on if abs(hsp - v) < 0.5 else _kp_off)
            except Exception: pass
        try:
            tgt = pub.cmd.get('hold_ff_pct', 0.0)
            hn, hf = st.get('hold_ff_n', 0.0), st.get('hold_ff_full_n', 0.0)
            dpg.set_value('hff_lbl', f'적용 {hp:.0f} % ({hn:.1f}/{hf:.1f} N·다리 · HL{hsp:.0f})'
                                     + (f' → 목표 {tgt:g} %' if abs(hp - tgt) > 1.0 else ''))
        except Exception: pass


def _refresh_sysload():
    """CPU·온도 — 계측은 sysload 모듈이 한다(텍스트 모니터와 **같은 코드**)."""
    if _sysload[0] is None:
        return
    txt, sev = _sysload[0].line()
    dpg.set_value('sysload', txt)
    dpg.configure_item('sysload', color=((150, 220, 150), (240, 170, 90), (235, 110, 110))[sev])


# ── ★영점 표 (2026-08-24) ────────────────────────────────────────────────
#   두 줄을 나란히 놓는다:
#     config  — 파일(emb/config/biped_emb.yaml)의 **지금** 값
#     제어기  — 제어기가 **기동 시 읽은** 값(state 의 `offset_deg`)
#   ★둘이 다르면 = config 는 바뀌었는데 제어기가 아직 옛 영점을 쓴다 = **재시작 필요**.
#     이게 cc321fc 가 "⚠재시작해야 반영된다" 로 경고했던 그 함정이다. 이제 눈에 보인다.
#   ★제어기 값은 **제어기가 직접 발행**한다. GUI 가 채널각↔모델각 역산식을 복사하지
#     않는다 — 그 복사본이 stale 이 되는 게 이 저장소가 반복해서 당한 버그다
#     (바로 위 JOG_LIM 주석이 같은 실수를 기록해 뒀다).
_off_cache = [0.0, None]     # (mtime, 값) — yaml 을 매 프레임 파싱하지 않는다
_off_live  = [None]          # 제어기가 발행한 값(렌더 루프가 채운다)
_off_base  = [None]          # [영점] 을 누른 시각의 config 값 — Δ 의 기준


def _off_cfg_read():
    """config 파일의 offset_deg. 파일이 바뀔 때만 다시 읽는다.

    ★GUI 는 별도 venv(~/.venvs/gui)로 뜨고 거기 PyYAML 이 없을 수 있다.
      그때는 **system python3 에 물어본다** — calib_zero.py 를 돌리는 그 인터프리터라
      yaml 이 반드시 있다. mtime 캐시가 있어 파일이 바뀔 때만 도니 사실상 공짜다.
    """
    try:
        mt = os.path.getmtime(_cfgp)
    except Exception:
        _off_cache[1] = None
        return None
    if mt == _off_cache[0]:
        return _off_cache[1]
    _off_cache[0] = mt
    _off_cache[1] = None
    try:                                        # ①같은 프로세스 안에 yaml 이 있으면 그대로
        import yaml as _y
        _off_cache[1] = [float(j['offset_deg']) for j in _y.safe_load(open(_cfgp))['joints']]
        return _off_cache[1]
    except Exception:
        pass
    try:                                        # ②없으면 system python3 에 물어본다
        r = subprocess.run(
            ['python3', '-c',
             'import sys,json,yaml;'
             'print(json.dumps([float(j["offset_deg"]) for j in '
             'yaml.safe_load(open(sys.argv[1]))["joints"]]))', _cfgp],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            _off_cache[1] = [float(x) for x in json.loads(r.stdout)]
    except Exception:
        pass
    return _off_cache[1]


def _refresh_offsets():
    cf, lv, diff = _off_cfg_read(), _off_live[0], False
    for i in range(NJ):
        c = cf[i] if (cf and i < len(cf)) else None
        l = lv[i] if (lv and i < len(lv)) else None
        dpg.set_value('offc_%d' % i, '—' if c is None else '%+.2f' % c)
        dpg.set_value('offl_%d' % i, '—' if l is None else '%+.2f' % l)
        bad = (c is not None and l is not None and abs(c - l) > 0.005)
        dpg.configure_item('offl_%d' % i, color=(240, 170, 90) if bad else (130, 200, 140))
        diff = diff or bad
        b = _off_base[0][i] if (_off_base[0] and i < len(_off_base[0])) else None
        d = (c - b) if (c is not None and b is not None) else None
        dpg.set_value('offd_%d' % i, '' if (d is None or abs(d) < 0.005) else '%+.2f' % d)
    dpg.set_value('off_msg',
                  '⚠제어기가 옛 영점을 쓴다 — **제어기**를 재시작해야 반영된다  '
                  '(RobotEmbedded 는 그대로 둘 것 — emb_ctl.sh 는 그쪽이라 여기선 소용없다)'
                  if diff else '')


# ── ★드라이버 알람 로그 (2026-09-11) ──────────────────────────────────────
#   영점세팅(calib_zero 러너)은 제거했다 — 실기 영점은 ZeroSet_RobotEmbedded(0xE2 하드웨어)
#   로만 잡는다. 이 로그는 그 자리를 이어받아 **드라이버 에러/두절/estop 을 시각별로 남긴다.**
#   ★모드 LED·모터 LED 는 '지금 상태'만 보여줘 순간 폴트(5× 강성 중 off 등)를 놓친다 —
#     로그는 edge(전이) 를 잡으므로 자기클리어된 폴트도 기록에 남는다.
#   ★데이터원 = deploy state JSON 의 err(=ucStatus)·health·n_dead·estop (별도 .so 불요).
_ALARM_BITS = ['과전류', '과전압', '저전압', '모터과온', 'MOSFET과온', 'ADC오프셋']  # ucStatus bit0..5


def _alarm_decode(e):
    e = int(e) & 0xFF
    if e == 0:
        return ''
    w = [_ALARM_BITS[b] for b in range(6) if e & (1 << b)]
    hi = [b for b in (6, 7) if e & (1 << b)]
    if hi:
        w.append('보호셧다운(bit%s·전원사이클로만 클리어)' % ','.join(map(str, hi)))
    return ('·'.join(w) if w else '정의밖') + ' [0x%02X]' % e


def _alarm_log(msg):
    ts = time.strftime('%H:%M:%S')
    _calib_buf[0] = (_calib_buf[0] + '[%s] %s\n' % (ts, msg))[-6000:]   # 최근분만 유지
    try:
        dpg.set_value('calib_out', _calib_buf[0])
    except Exception:
        pass


_alarm_prev = [None]      # 직전 관측(health·err·n_dead·estop·motors_on) — edge 검출용


def _check_driver_alarms(st):
    """state 의 health/err/두절/estop 변화를 edge 로 잡아 로그에 남긴다."""
    health = st.get('health') or []
    err = st.get('err') or []
    nd, nf = st.get('n_dead', 0), st.get('n_fault', 0)
    estop = bool(st.get('estop') or st.get('estop_latched'))
    mon = bool(st.get('motors_on'))
    mode = st.get('mode')
    cur = {'h': tuple(health), 'e': tuple(int(x) for x in err),
           'nd': nd, 'nf': nf, 'estop': estop, 'mon': mon, 'mode': mode}
    prev = _alarm_prev[0]
    if prev is None:
        _alarm_prev[0] = cur
        _alarm_log('▶ 알람 감시 시작 — mode=%s 모터on=%s (정상%s/에러%s/두절%s)'
                   % (mode, mon, st.get('n_ok', '?'), nf, nd))
        return
    for i in range(min(len(JOG_NAMES), max(len(cur['h']), len(cur['e'])))):
        nm = JOG_NAMES[i]
        ph = prev['h'][i] if i < len(prev['h']) else None
        ch = cur['h'][i] if i < len(cur['h']) else None
        if ch != ph and ch is not None:
            if ch in ('fault', 'dead'):
                _alarm_log('⚠ %s → %s' % (nm, {'fault': '에러(FAULT)', 'dead': '두절(DEAD)'}.get(ch, ch)))
            elif ch == 'ok' and ph in ('fault', 'dead'):
                _alarm_log('✓ %s 정상 복귀' % nm)
        pe = prev['e'][i] if i < len(prev['e']) else 0
        ce = cur['e'][i] if i < len(cur['e']) else 0
        if ce != pe and ce != 0:
            _alarm_log('⚠ %s 드라이버에러: %s' % (nm, _alarm_decode(ce)))
    if cur['estop'] and not prev['estop']:
        _alarm_log('⛔ E-STOP 래치 (reason=%s)' % st.get('estop_reason'))
    if (not cur['mon']) and prev['mon'] and mode not in ('off', 'soft_off'):
        _alarm_log('⚠ 모터 OFF 전환 (mode=%s 인데 motors_on=False — 드라이버 드롭 의심)' % mode)
    _alarm_prev[0] = cur




# ── ★제어기 재기동 (통신 두절 복구) ──────────────────────────────────────
#   왜 필요한가: 실기 브링업 중 SHM/제어기가 이따금 끊긴다. 그때마다 터미널로 가서
#   프로세스를 죽이고 다시 띄우는 게 번거롭다. GUI 에서 한 번에 복구한다.
#
#   ⚠**RESET 버튼을 덮어쓰지 않았다.** RESET 은 이미 `jogger.reset(q_leg)` → HOLD 라는
#     안전 동작을 한다(jog 램프를 실측각으로 재시드해 점프를 막는다). 그걸 잃으면 안 된다.
#   ★RobotEmbedded 도 띄운다(2026-08-11) — setcap 이 적용돼 일반 사용자로 실행된다:
#       cap_net_admin,cap_net_raw=eip   (getcap 으로 확인)
#     SHM 세그먼트는 root 소유지만 perms 666 이라 비루트 attach 에 문제가 없다.
#     죽어 있을 때만 띄우고, 살아 있으면 건드리지 않는다.
#   ⚠두 번 눌러야 실행된다(오조작 방지) — 프로세스를 띄우고 모터를 물리는 동작이다.
# _EMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emb')
# _EMB_BIN = os.path.expanduser('~/ZSource/RobotEmbedded/build/src/RobotEmbedded')
# _restart_armed = [0.0]


# def _proc_pids(prefix):
#     """명령줄이 prefix 로 **시작**하는 PID. bash 래퍼·자기 자신이 안 걸리게 한다."""
#     try:
#         out = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True).stdout
#     except Exception:
#         return []
#     pids = []
#     for line in out.splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         pid, _, args = line.partition(' ')
#         if args.strip().startswith(prefix):
#             pids.append(int(pid))
#     return pids
#
#
# def _pgrep_x(name):
#     """프로세스 **이름**이 정확히 일치하는 PID (명령줄 prefix 로는 sudo 래퍼를 못 잡는다)."""
#     try:
#         r = subprocess.run(['pgrep', '-x', name], capture_output=True, text=True)
#         return [int(x) for x in r.stdout.split()]
#     except Exception:
#         return []
#
#
# def _restart_worker():
#     def say(m):
#         print(f'[gui] {m}', flush=True)
#         try:
#             dpg.set_value('restart_msg', m)
#         except Exception:
#             pass
#
#     # ★RobotEmbedded 검출은 프로세스 **이름**으로 한다 — 명령줄 prefix 로는 `sudo ./src/...`
#     #   형태를 못 잡는다(처음에 그렇게 짰다가 실기에서 빈 배열이 나와 발견).
#     if not _pgrep_x('RobotEmbedded'):
#         # ★setcap 이 돼 있어 **일반 사용자로 띄울 수 있다**(2026-08-11 확인):
#         #     cap_net_admin,cap_net_raw=eip  ·  SHM perms 666 이라 비루트 attach 도 된다.
#         #   그 전에는 sudo 가 필요해 GUI 에서 못 띄웠다.
#         if not os.path.exists(_EMB_BIN):
#             say(f'✗ RobotEmbedded 바이너리가 없다: {_EMB_BIN}'); return
#         say('RobotEmbedded 기동 중 …')
#         try:
#             subprocess.Popen(['./src/RobotEmbedded'], cwd=os.path.dirname(os.path.dirname(_EMB_BIN)),
#                              stdout=open('/tmp/robotembedded.log', 'w'),
#                              stderr=subprocess.STDOUT, start_new_session=True)
#         except Exception as e:
#             say(f'✗ RobotEmbedded 기동 실패: {e} — setcap 확인'); return
#         # 초기화 게이트: halGait 는 수신 100틱 + 램프 4500틱 @1kHz ≈ 4.6초 동안 SHM 명령을
#         # 무시한다. 그 전에 제어기를 붙이면 첫 명령이 버려진다 ⇒ 충분히 기다린다.
#         for i in range(80):
#             time.sleep(0.1)
#             if not _pgrep_x('RobotEmbedded'):
#                 say('✗ RobotEmbedded 가 곧바로 죽었다 — /tmp/robotembedded.log 확인'); return
#             if i * 0.1 > 6.0:
#                 break
#         say('RobotEmbedded 준비됨(초기화 게이트 6s 대기 완료)')
#
#     old = _proc_pids('python3 app/biped_emb.py')
#     if old:
#         say(f'제어기 종료 {old} …')
#         for pid in old:
#             try:
#                 os.kill(pid, 15)
#             except Exception:
#                 pass
#         for _ in range(40):                       # 최대 4초 대기
#             time.sleep(0.1)
#             if not _proc_pids('python3 app/biped_emb.py'):
#                 break
#         else:
#             say('✗ 이전 제어기가 안 죽는다 — 수동 확인 필요'); return
#
#     say('제어기 기동 중 …')
#     try:
#         # ★hold 로 시작한다. off 로 띄우면 Emb 게이트가 열리는 순간 무여자 명령이 먹혀
#         #   다리가 풀린다(2026-08-10 실기에서 겪음).
#         subprocess.Popen(['python3', 'app/biped_emb.py', '--start-mode', 'hold'],
#                          cwd=_EMB_DIR, stdout=open('/tmp/biped_emb.log', 'w'),
#                          stderr=subprocess.STDOUT, start_new_session=True)
#     except Exception as e:
#         say(f'✗ 기동 실패: {e}'); return
#
#     for _ in range(100):                          # state 가 신선해질 때까지 최대 10초
#         time.sleep(0.1)
#         try:
#             if time.time() - os.path.getmtime(STATE) < 0.5:
#                 st = json.load(open(STATE))
#                 say(f"✅ 복구됨 — mode={st.get('mode')} n_ok={st.get('n_ok')}/8 "
#                     f"{st.get('loop_hz', 0):.0f}Hz")
#                 return
#         except Exception:
#             pass
#     say('✗ 기동했으나 state 가 갱신되지 않는다 — /tmp/biped_emb.log 확인')
#
#
# def on_restart():
#     now = time.time()
#     if now - _restart_armed[0] > 3.0:             # 1차 클릭 = 무장
#         _restart_armed[0] = now
#         dpg.set_value('restart_msg', '⚠ 3초 안에 한 번 더 누르면 제어기를 재기동한다')
#         return
#     _restart_armed[0] = 0.0
#     threading.Thread(target=_restart_worker, daemon=True).start()
#
#
def set_mode(mode):
    if mode == 'reset':
        left.clear(); right.clear()
        pub.set(mode='reset', v=0.0, vy=0.0, w=0.0)
        dpg.set_value('spd_sl', 0); dpg.set_value('vy_sl', 0); dpg.set_value('turn_sl', 0)
        return
    if mode == 'walk':          # ★걸으려면 1점 점발로 자동 전환(평발=발목토크0라 걷기 부적합)
        pub.set(contact='1pt', body_h=H_DEF_1PT, mode='walk')
        dpg.set_value('h_sl', H_DEF_1PT)
        return
    if mode == 'stand':         # ★서려면 2점 평발로 자동 전환(밑창ZMP=안정 서기)
        left.clear(); right.clear()
        pub.set(contact='2pt', body_h=H_DEF_2PT, mode='stand', v=0.0, vy=0.0, w=0.0)
        dpg.set_value('h_sl', H_DEF_2PT)
        dpg.set_value('spd_sl', 0); dpg.set_value('vy_sl', 0); dpg.set_value('turn_sl', 0)
        return
    if mode == 'jog':           # ★각축 검증 진입: 슬라이더를 현재 실측각으로 정렬(명령 점프 방지)
        # ★실패하면 **jog 로 들어가지 않는다**(2026-08-10). 종전엔 except: pass 라
        #   상태를 못 읽어도 그대로 jog 로 진입했고, 그때 jog_deg 는 초기값 0 이다.
        #   영점을 잡은 지금 그건 "전 축 모델 0° 로 가라" 는 뜻이라 무릎이 55° 펴진다.
        #   시드에 실패했다는 건 명령 점프를 막을 근거가 없다는 뜻이므로 fail-closed 가 맞다.
        try:
            q = json.load(open(STATE))['q_leg_deg']
            if len(q) < NJ:
                raise ValueError(f'q_leg_deg 길이 {len(q)} < {NJ}')
            for i in range(NJ):
                v = float(max(JOG_LIM[i][0], min(JOG_LIM[i][1], q[i])))
                if abs(v - float(q[i])) > 0.5:
                    raise ValueError(
                        f'{JOG_NAMES[i]} 실측 {q[i]:+.1f}° 가 jog 한계 '
                        f'[{JOG_LIM[i][0]:+.1f},{JOG_LIM[i][1]:+.1f}] 밖 — 진입 시 '
                        f'{abs(v-float(q[i])):.1f}° 움직인다. 한계를 넓히거나 자세를 먼저 옮길 것')
                dpg.set_value(f'jog_{i}', v); pub.cmd['jog_deg'][i] = v
        except Exception as e:
            msg = f'JOG 진입 취소 — {e}'
            print(f'[gui] {msg}', flush=True)          # /tmp/teleop_gui_biped.log
            try:
                dpg.set_value('state', msg)            # 갱신 루프가 곧 덮어쓸 수 있다
            except Exception:
                pass
            return                                     # ★mode 를 안 보낸다 = 현재 모드 유지
    pub.set(mode=mode)          # off/jog/hold 등
    left.clear(); right.clear(); pub.set(v=0.0, vy=0.0, w=0.0)
    dpg.set_value('spd_sl', 0); dpg.set_value('vy_sl', 0); dpg.set_value('turn_sl', 0)


left  = JoyPad('joyL', 190, on_left, cross_only=True)   # ★십자만(전후 XOR 측방, 대각 금지)
right = JoyPad('joyR', 190, on_right, x_only=True)

dpg.create_context()
# ★한글 폰트 — GUI(dearpygui) 는 TTF 를 로드하므로 한글이 정상이다.
#   ⚠깨져 보였던 것은 **MuJoCo 뷰어** 쪽이다(mjr_overlay = ASCII 전용 비트맵 폰트).
#     둘은 별개다 — 뷰어 HUD 는 영문으로 바꿨고(biped_monitor.cpp), 여기는 TTF 를 쓴다.
#   폰트 파일이 없으면 라벨이 깨져 JOG 검증에서 축 이름을 못 읽으므로 기동 시 경고한다.
#   ★고정 경로 목록만으로는 못 찾는다 — 사용자가 `~/.local/share/fonts` 에 깔아둔 폰트를
#     놓친다(2026-08-21 실측: 이 기기는 시스템 경로 셋이 다 없고 NanumGothic 이 홈에 있었다).
#     그래서 고정 목록 뒤에 **fontconfig 질의**를 붙인다. 그러면 어디에 깔았든 잡힌다.
def _fc_korean():
    """fc-list 로 한글 글리프가 있는 TTF/TTC 를 찾는다. fontconfig 이 없으면 조용히 포기."""
    import shutil, subprocess
    if not shutil.which('fc-list'):
        return []
    try:
        out = subprocess.run(['fc-list', ':lang=ko', 'file'],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    paths = []
    for line in out.splitlines():
        p = line.split(':')[0].strip()
        if p.lower().endswith(('.ttf', '.ttc', '.otf')):
            paths.append(p)
    return paths

_FONT_CANDS = [os.environ.get('GUI_FONT', ''),
               '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
               '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
               '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
               os.path.expanduser('~/.local/share/fonts/NanumGothic-Regular.ttf')]
_FONT = next((f for f in _FONT_CANDS if f and os.path.exists(f)), None)
if _FONT is None:                                   # 마지막 수단 — 시스템에 뭐가 있든 찾는다
    _FONT = next((f for f in _fc_korean() if os.path.exists(f)), None)
_kf = None
if _FONT:
    with dpg.font_registry():
        _kf = dpg.add_font(_FONT, 18)
    # ⚠add_font_range_hint 는 쓰지 않는다 — dearpygui 2.x 에서 **deprecated no-op** 이다
    #   ("character ranges are now automatic"). 부르면 경고만 나오고 효과가 없다.
    print(f'[gui] 한글 폰트: {_FONT}')
else:
    print('[gui] ⚠ CJK 폰트를 못 찾았다 — 한글 라벨이 깨져 보인다.\n'
          '      설치: sudo apt install fonts-noto-cjk   (또는 GUI_FONT=/경로/폰트.ttf)')
with dpg.theme() as _dark:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (22, 24, 32))
        dpg.add_theme_color(dpg.mvThemeCol_Button, (46, 52, 74))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (66, 74, 104))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 224, 235))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
with dpg.theme() as _stop:                 # RESET·Off 전원 = 빨강(17-DOF 규약)
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (170, 45, 45))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (210, 65, 65))
with dpg.theme() as _walk:                  # Walk 이동 = 초록
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 120, 70))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 160, 95))
with dpg.theme() as _home:                  # Home 복귀 = 파랑(이동은 하지만 자율보행은 아님)
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 85, 140))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 115, 185))
with dpg.theme() as _kp_on:                 # 강성 배율 — 선택된 단계(주황)
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (185, 110, 40))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (215, 140, 60))
with dpg.theme() as _kp_off:                # 강성 배율 — 나머지(어둡게)
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (44, 48, 62))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 76, 96))

with dpg.window(tag='main'):
    dpg.add_text('biped teleop  —  MPC + WBIC (event-DCM)', color=(150, 200, 255))
    dpg.add_separator()
    with dpg.group(horizontal=True):
        with dpg.group():
            left.build('전후/측방')
            dpg.add_text('좌: 위아래=전후 · 좌우=측방 (십자=하나씩)', color=(120, 130, 150))
        with dpg.group():
            right.build('선회')
            dpg.add_text('우: 좌우=선회 (★점발 한계로 매우 약함 ~1-2°/s·측방도 60%)', color=(120, 130, 150))
        with dpg.group():
            dpg.add_text('vx [m/s]')
            dpg.add_slider_float(tag='spd_sl', default_value=0.0, min_value=-VMAX, max_value=VMAX, width=180, callback=on_vx)
            dpg.add_text('vy [m/s] (측방)')
            dpg.add_slider_float(tag='vy_sl', default_value=0.0, min_value=-VY_MAX, max_value=VY_MAX, width=180, callback=on_vy)
            dpg.add_text('wz [rad/s] (선회)')
            dpg.add_slider_float(tag='turn_sl', default_value=0.0, min_value=-WZ_MAX, max_value=WZ_MAX, width=180, callback=on_turn)
            dpg.add_text('몸통 높이 [m]')
            dpg.add_slider_float(tag='h_sl', default_value=H_DEF, min_value=H_MIN, max_value=H_MAX, width=180, callback=on_height)
    dpg.add_spacer(height=8)
    dpg.add_text('모션', color=(170, 175, 195))
    # ★2026-08-14 라벨 정리 — 이름이 동작을 오해시키고 있었다.
    #   · 'RESET' → **'정지·현자세'**: 프로세스와 아무 상관이 없다. 하는 일은
    #     `jogger.reset(q_leg)` → HOLD, 즉 "jog 램프를 **현재 측정각**으로 재시드하고
    #     그 자리에 선다". 램프를 재시드하는 게 핵심이라 다음 JOG 에서 계단이 안 생긴다.
    #     'RESET' 이라 써 두니 '제어기 재기동'과 헷갈렸다 — 둘은 전혀 다른 동작이다.
    #   · '제어기 재기동' → **'제어기 재시작(프로세스)'**: 진짜로 프로세스를 죽였다 띄운다.
    #     SHM/통신 두절 복구용. 오조작 방지로 3초 안에 두 번 눌러야 실행된다.
    #   · 'Stand 서기' → **'2점 평발 stand'** · 'Walk 이동' → **'점발 보행'**
    #     접촉모드가 동작 이름 안에 들어가야 한다. 평발/점발은 발 자세가 53° 다르고
    #     (밑창 수평 vs toe-down) 전환은 매달아서 해야 하는 별개 작업이다.
    with dpg.group(horizontal=True):   # 정지 → Off전원 → JOG → 재시작
        _rb = dpg.add_button(label='정지·현자세', width=110, tag='mbtn_hold', callback=lambda: set_mode('reset'))
        dpg.bind_item_theme(_rb, _stop)
        _ob = dpg.add_button(label='Off 전원', width=100, tag='mbtn_off', callback=lambda: set_mode('off'))
        dpg.bind_item_theme(_ob, _stop)
        # ★안전 종료 (2026-08-28) — 현재 자세에서 **기동 시 자세**로 20dps 서행 후 무여자.
        #   ⚠[Off 전원] 은 **비상 탈출구로 그대로 둔다**(즉시 무여자). 넘어지는 중에
        #     "천천히" 내려가면 그게 더 위험하다. 평시 종료만 이 버튼을 쓴다.
        #   ⚠하중을 받고 있으면 이 버튼도 낙차를 줄일 뿐이다 — 크레인으로 먼저 받칠 것.
        _sb = dpg.add_button(label='안전 종료', width=100, tag='mbtn_softoff',
                             callback=lambda: set_mode('soft_off'))
        dpg.bind_item_theme(_sb, _stop)
        dpg.add_button(label='JOG 검증', width=90, tag='mbtn_jog', callback=lambda: set_mode('jog'))   # ★각축 검증(실기)
        with dpg.group():              # ★무중력 = 중력보상만. 매달린 상태 전용
            _fb = dpg.add_button(label='무중력', width=100, tag='mbtn_float', callback=lambda: set_mode('float'))
            dpg.add_text('(매달린 채만)', color=(120, 130, 150))
        with dpg.group():              # ★발밀기 = 무중력 + 발끝 z-힘 (저울 α 측정)
            _pb = dpg.add_button(label='발밀기', width=100, tag='mbtn_push', callback=lambda: set_mode('push'))
            dpg.add_text('(발밑에 저울)', color=(120, 130, 150))
        with dpg.tooltip(_pb):
            dpg.add_text('τ = g*·G(q) + Jᵀ(0,0,−F). 발밑 저울로 α 를 외부기준 측정.\n'
                         '⚠F 는 항상 0 에서 시작해 5 N/s 로 램프한다(계단 없음).\n'
                         '⚠크레인 줄 팽팽하게 — 미는 반작용을 줄이 받는다.\n'
                         '읽는 법: 0→50→0 왕복, 올림/내림 저울값 평균(마찰 소거).\n'
                         '기울기 = α (예상: 5 kg 명령 → 저울 ≈ 4.7 kg @ α 0.93)')
        with dpg.tooltip(_fb):
            dpg.add_text('중력보상(zero-g). τ_ff = 배율 × G_model(q) · kp=0 · kd 소량.\n'
                         '손으로 밀어 자세를 잡을 수 있다.\n'
                         '⚠접지 중이면 제어기가 **거부**한다 — 지면 반력을 모르는 채 밀어 올린다.\n'
                         '⚠마찰(관절 0.6~0.9Nm)은 안 지워진다 — 뻑뻑한 게 정상이다.')
        # ★'제어기 재시작' 버튼 **삭제** (2026-08-14). 코드는 아래에 주석으로 남긴다.
        #   ⚠이 버튼은 `emb_ctl.sh` 를 **우회하고 RobotEmbedded 를 직접 띄웠다**:
        #       subprocess.Popen(['./src/RobotEmbedded'], stdout=open('/tmp/robotembedded.log','w'))
        #     ⇒ ① awk 로그필터 없음 — 시간당 1.7GB 로 자란다. 게다가 파일명이 달라
        #          (`robotembedded.log`) **아무도 안 보는 자리**에 쌓인다.
        #       ② 중복기동 가드 없음 — emb_ctl 은 "중복 기동은 EtherCAT 버스를 깬다" 며 막는다.
        #       ③ 신선도(MotorStatus16) 확인 없음 · stdbuf 없음(버퍼링으로 로그가 안 보임).
        #   ⇒ 기동/재기동은 **`emb/diag/emb_ctl.sh` 한 곳으로** 모은다. GUI 는 모드만 바꾼다.
        #     복구:  cd ~/simulation/biped/emb && diag/emb_ctl.sh stop && diag/emb_ctl.sh start
        with dpg.group():              # ★Home=기하 진리 고정(2026-08-28): 1점 Qhome8 · 2점 Qflat8
            #   배포기가 yaml home.q_deg 를 더 읽지 않는다 — 측정용 임시 자세는 HOME_DEG env.
            _hb = dpg.add_button(label='Home 복귀', width=100, tag='mbtn_home', callback=lambda: set_mode('home'))
            dpg.add_text('(Qhome8/Qflat8)', color=(120, 130, 150))
        dpg.bind_item_theme(_hb, _home)
        # ★Hold 버튼 제거 (2026-08-21, 사용자: "안 쓸 것").
        #   ⚠**모드 자체는 지우지 않았다** — 제어기가 내부 폴백으로 쓴다:
        #     reset→hold · 접지거부→hold · 자세거부→hold · --start-mode hold.
        #     지우면 갈 곳이 off(=limp=낙하)나 home(하중 실린 채 큰 이동)뿐이라 더 위험하다.
        #   ⇒ 조작 표면에서만 뺀다. 필요하면 '정지·현자세'(reset)가 hold 로 들어간다.
        with dpg.group():              # ★2점 평발 = 정적 자세유지(보행 안 함)
            dpg.add_button(label='2점 평발 stand', width=130, tag='mbtn_stand', callback=lambda: set_mode('stand'))
            dpg.add_text('(밑창 접지·정적)', color=(120, 130, 150))
        with dpg.group():              # ★1점 점발 = stepping 보행
            _wb = dpg.add_button(label='점발 보행', width=110, tag='mbtn_walk', callback=lambda: set_mode('walk'))
            dpg.add_text('(발끝 1점·동적)', color=(120, 130, 150))
        dpg.bind_item_theme(_wb, _walk)
    dpg.add_text('복구 순서: Off 전원 → Home 복귀 → (접지·하중전달) → 2점 평발 stand'
                 '   · Off=명령토크 0 (Kp=Kd=τ=0)', color=(150, 155, 175))
    dpg.add_text('⚠매달린 채로 stand/보행을 켜지 말 것 — GRF 를 전제한 QP 라 해가 안 나오고 '
                 '중력보상 폴백으로 떨어진다(겉보기엔 안정돼 보인다). 매달려서 되는 건 off/jog/home 뿐.',
                 color=(210, 150, 90))
    dpg.add_text('Home=제어기가 정한 자세로 전축 동시 S-curve 이동 · Hold=지금 그 자리를 잡기\n'
                 '  목표자세: 1점 점발 = config home.q_deg(전축 0, biped_emb.py 와 동일) · '
                 '2점 평발 = Qflat8(발목 채널 100.4°, 밑창 수평)',
                 color=(150, 155, 175))
    # ── ★위치모드 강성 5단계 (2026-08-21) ────────────────────────────────────
    #   home/hold/jog 의 kp 배율. **stand/walk 에는 안 걸린다**(WBIC 와 싸우면 안 된다).
    #   쓰는 자리: home 으로 자세를 잡고 hold 로 굳힌 뒤 **크레인을 내려 하중을 실을 때**,
    #   그 자세를 지키려면 얼마나 세야 하는지 여기서 올려 가며 찾는다.
    #   ⚠제어기가 1초에 걸쳐 램프한다 — 계단으로 바꾸면 벌어져 있던 축의 토크가
    #     그 자리에서 배율만큼 튀어 접지 중에 τ_trip 이 바로 걸린다.
    dpg.add_text('● 위치모드 강성 (home·hold·jog 의 kp 배율 — stand/walk 무관)',
                 color=(255, 205, 120))
    with dpg.group(horizontal=True):
        for _k, _s in enumerate(KP_STEPS):
            _who, _tr = KP_TRIP[_k]
            dpg.add_button(label=f'×{_s:g}', width=52, tag=f'kpbtn_{_k}', user_data=_s,
                           callback=lambda _s_, _a_, _u_: set_kp_scale(_u_))
            with dpg.tooltip(f'kpbtn_{_k}'):
                dpg.add_text(f'kp×{_s:g} · kd×{math.sqrt(_s):.2f}\n'
                             f'가장 예민한 축 {_who} — {_tr:.2f}° 에서 토크트립')
        dpg.add_text('', tag='kp_lbl', color=(150, 155, 175))
    with dpg.group(horizontal=True):
        dpg.add_text('kd 배율', color=(255, 205, 120))
        for _k, _d in enumerate(KD_STEPS):
            _lb = '자동' if _d is None else f'×{_d:g}'
            dpg.add_button(label=_lb, width=52, tag=f'kdbtn_{_k}', user_data=_d,
                           callback=lambda _s_, _a_, _u_: set_kd_scale(_u_))
            with dpg.tooltip(f'kdbtn_{_k}'):
                if _d is None:
                    dpg.add_text('kd = √kp — ζ 보존(기본).\nkp 만 올리면 ζ 가 √배율만큼 떨어져 저감쇠 진동이 된다.')
                else:
                    dpg.add_text(f'kd 를 ×{_d:g} 로 고정.\n낮출수록 토크 리플(kd x 속도잡음)이 줄지만 ζ 도 같이 떨어진다.')
    # ── ★무중력 중력보상 배율 (2026-08-24) ──────────────────────────────────
    dpg.add_text('● 무중력 중력보상 배율 (float 모드 전용 — 중립점 g* 를 찾는다)',
                 color=(255, 205, 120))
    with dpg.group(horizontal=True):
        for _k, _g in enumerate(GRAV_STEPS):
            dpg.add_button(label=f'×{_g:.2f}', width=58, tag=f'gvbtn_{_k}', user_data=_g,
                           callback=lambda _s_, _a_, _u_: set_grav_scale(_u_))
            with dpg.tooltip(f'gvbtn_{_k}'):
                dpg.add_text(f'τ_ff = {_g:.2f} × G_model(q)\n'
                             + ('모델대로 — 여기서 시작한다' if abs(_g-1.0) < 1e-9 else
                                (f'모델보다 {(_g-1)*100:+.0f}% 세게' if _g > 1 else
                                 f'모델보다 {(1-_g)*100:.0f}% 약하게')))
        dpg.add_text('×1.00', tag='gv_lbl', color=(150, 155, 175))
    dpg.add_text('찾는 법: 올려서 다리가 **뜨기 시작**하는 g⁺, 내려서 **지기 시작**하는 g⁻ → '
                 'g* = (g⁺+g⁻)/2 가 참값(마찰 소거).\n'
                 '  g* > 1 이면 중력보상이 그만큼 모자라다 — 그게 stand 처짐의 크기다. '
                 'hip·thigh 로 볼 것(중력 5Nm 대라 마찰 0.8 에 안 묻힌다).',
                 color=(150, 155, 175))
    # ── ★발밀기 z-힘 (2026-08-25) ──────────────────────────────────────────
    dpg.add_text('● 발밀기 목표힘 (push 모드 전용 — 발밑 저울, 0→최대→0 왕복 평균으로 읽는다)',
                 color=(255, 205, 120))
    with dpg.group(horizontal=True):
        for _k, _n in enumerate(('HL', 'HR')):
            dpg.add_button(label=_n, width=44, tag=f'plbtn_{_k}', user_data=_k,
                           callback=lambda _s_, _a_, _u_: set_push_leg(_u_))
        dpg.add_text('│', color=(90, 95, 105))
        for _k, _f in enumerate(PUSH_STEPS):
            dpg.add_button(label=f'{_f:g}N', width=48, tag=f'pfbtn_{_k}', user_data=_f,
                           callback=lambda _s_, _a_, _u_: set_push_fz(_u_))
        dpg.add_text('', tag='push_lbl', color=(150, 155, 175))
    dpg.add_separator()
    # ── ★hold 중력지지 (2026-08-28) ──────────────────────────────────────
    dpg.add_text('● hold 중력지지 (hold 모드 전용 — 100% = 한 다리가 몸무게 절반을 떠받침)',
                 color=(255, 205, 120))
    with dpg.group(horizontal=True):
        for _k, _p in enumerate(HOLD_FF_STEPS):
            dpg.add_button(label=f'{_p:g}%', width=48, tag=f'hffbtn_{_k}', user_data=_p,
                           callback=lambda _s_, _a_, _u_: set_hold_ff(_u_))
        dpg.add_text('│ 배분(HL%)', color=(90, 95, 105))
        for _k, _sp in enumerate(HOLD_SPLIT_STEPS):
            dpg.add_button(label=f'{_sp:g}', width=36, tag=f'hspbtn_{_k}', user_data=_sp,
                           callback=lambda _s_, _a_, _u_: set_hold_split(_u_))
        dpg.add_text('', tag='hff_lbl', color=(150, 155, 175))
    dpg.add_text('강성을 올리는 것과 다르다 — **총 명령토크는 그대로**고 kp·오차에 있던 몫이 '
                 'τ_ff 로 옮겨갈 뿐이라 처짐만 줄고 토크트립 여유는 안 나빠진다. '
                 '0%면 종전 hold(순수 위치 PD)와 완전히 같다. 진입할 때마다 0에서 다시 램프한다.',
                 color=(150, 155, 175), wrap=760)
    dpg.add_separator()
    dpg.add_text('⚠올릴수록 자세는 잘 지키지만 **토크트립까지의 각도가 줄어든다** '
                 '(τ_trip ÷ (kp_ch·gear_k)). 접지시키며 하중이 실릴 때 여기 걸리기 쉽다 — ×3 부터 시작할 것.',
                 color=(210, 150, 90))
    dpg.add_separator()
    # ── ★영점 (2026-08-24) ────────────────────────────────────────────────
    dpg.add_text('● 영점 offset_deg 대조 (config vs 제어기 · 하드웨어영점=ZeroSet_RobotEmbedded)', color=(255, 205, 120))
    with dpg.table(header_row=True, policy=dpg.mvTable_SizingFixedFit,
                   borders_innerV=True, borders_outerH=True, borders_outerV=True):
        dpg.add_table_column(label='')
        for _n in JOG_NAMES:
            dpg.add_table_column(label=_n)
        with dpg.table_row():                      # 파일의 지금 값
            dpg.add_text('config')
            for _i in range(NJ):
                dpg.add_text('—', tag='offc_%d' % _i)
        with dpg.table_row():                      # 제어기가 기동 시 읽은 값
            dpg.add_text('제어기')
            for _i in range(NJ):
                dpg.add_text('—', tag='offl_%d' % _i, color=(130, 200, 140))
        with dpg.table_row():                      # [영점] 누르기 전 대비 변화
            dpg.add_text('Δ')
            for _i in range(NJ):
                dpg.add_text('', tag='offd_%d' % _i, color=(255, 180, 90))
    dpg.add_text('', tag='off_msg', color=(240, 170, 90))
    dpg.add_text('(영점세팅 제거 — 하드웨어영점 ZeroSet_RobotEmbedded 사용. 위 표는 config↔제어기 영점 대조 진단용)',
                 color=(120, 130, 150))
    dpg.add_separator()
    # ── ★각축(JOG) 패널: 8관절 슬라이더(모터 1:1) + 실측 + 통신 상태 LED ──
    dpg.add_text('● 각축 JOG 검증 (슬라이더=목표각° · 실측° · ●=상태LED)', color=(255, 205, 120))
    with dpg.group(horizontal=True):
        # ★라벨에서 'home' 을 뺐다 — 위 [Home 복귀] 버튼과 전혀 다른 동작이다.
        #   이건 JOG 슬라이더를 0 으로 놓는 것(등속 램프, 축마다 도착시각 제각각)이고,
        #   [Home 복귀] 는 home 모드의 S-curve 동시도착 궤적이다.
        dpg.add_button(label='슬라이더 모두 0', width=130, callback=jog_zero)
        dpg.add_text('LED 초록=정상·노랑=에러·빨강=두절(배선O)·어두움=미장착 · 실기(app/biped_emb.py)서 각 모터 확인',
                     color=(120, 130, 150))
    _LED_R = 7
    for i, nm in enumerate(JOG_NAMES):
        with dpg.group(horizontal=True):
            with dpg.drawlist(width=2 * _LED_R + 6, height=2 * _LED_R + 6, tag=f'leddl_{i}'):
                dpg.draw_circle([_LED_R + 3, _LED_R + 3], _LED_R, fill=(70, 70, 78),
                                color=(30, 30, 36), tag=f'led_{i}')
            dpg.add_text(f'{nm:9s}', color=(190, 195, 210))
            # ★슬라이더 양끝에 jog 한계를 숫자로 박아 둔다. 이 한계는 축마다 다르고
            #   (config 의 jog_min_deg/jog_max_deg 예외), 관절한계와도 다르다 —
            #   화면에 안 쓰여 있으면 "왜 여기서 안 넘어가지" 를 매번 config 를 열어 확인해야 한다.
            dpg.add_text(f'{JOG_LIM[i][0]:>6.1f}', color=(120, 130, 150))
            dpg.add_slider_float(tag=f'jog_{i}', default_value=0.0,
                                 min_value=JOG_LIM[i][0], max_value=JOG_LIM[i][1],
                                 width=240, format='%.1f', user_data=i,
                                 callback=lambda s, v, u: on_jog(s, v, u))
            dpg.add_text(f'{JOG_LIM[i][1]:<6.1f}', color=(120, 130, 150))
            dpg.add_text('--.-', tag=f'meas_{i}', color=(150, 220, 150))
    dpg.add_separator()
    dpg.add_text('-', tag='state', color=(150, 220, 150))
    dpg.add_text('-', tag='sysload', color=(150, 220, 150))   # ★CPU·온도(500Hz 루프가 여기 물려 있다)

with dpg.handler_registry():
    dpg.add_mouse_down_handler(callback=lambda: (left.press(), right.press()))
    dpg.add_mouse_drag_handler(callback=lambda: (left.move(), right.move()))
    dpg.add_mouse_release_handler(callback=lambda: (left.release(), right.release()))
    dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=lambda: (left.toggle_latch(), right.toggle_latch()))

# ★드라이버 알람 로그창 — 'main' **밖**에 만든다(안에 넣으면 자식 위젯이 돼 창이 안 뜬다).
#   상시 표시. 모드/모터 LED 는 '지금'만 보여줘 순간 폴트를 놓치므로, 여기 시각별로 남긴다.
with dpg.window(label='드라이버 알람 로그', tag='calib_win',
                width=660, height=330, pos=(30, 430), show=True):
    with dpg.group(horizontal=True):
        dpg.add_text('드라이버 에러·두절·estop·모터OFF 를 시각별 기록 (state err=ucStatus 기반)',
                     color=(150, 155, 175))
        dpg.add_button(label='지우기', width=70,
                       callback=lambda: (_calib_buf.__setitem__(0, ''), dpg.set_value('calib_out', '')))
    dpg.add_separator()
    dpg.add_input_text(tag='calib_out', multiline=True, readonly=True,
                       width=-1, height=270, default_value='(감시 대기중… 컨트롤러 연결되면 시작)')

dpg.bind_theme(_dark)
if _kf is not None:
    dpg.bind_font(_kf)
dpg.create_viewport(title='biped teleop', width=700, height=800)
dpg.setup_dearpygui(); dpg.show_viewport(); dpg.set_primary_window('main', True)
set_kp_scale(1.0)      # ★강성 버튼 초기 선택 표시(×1). Pub 기본값과 반드시 일치시킬 것.
set_push_leg(0)        # ★발밀기 다리 초기 선택 표시(HL)

# ★absent(미장착)를 dead(배선됐는데 두절)와 다른 색으로 둔다 — 전자는 정상, 후자는 고장이다.
#   같은 회색으로 뭉뚱그리면 "원래 없는 축" 과 "죽은 축" 이 구분되지 않는다.
_LED = {'ok': (60, 210, 90), 'fault': (235, 200, 60),
        'dead': (150, 90, 90), 'absent': (48, 50, 58)}
_last_file_hb = [0.0]          # ★파일 하트비트 타이머(리스트=클로저 없이 가변)
_last_off_ref = [0.0]          # 영점 표 갱신 타이머
while dpg.is_dearpygui_running():
    try:
        with open(STATE) as f:
            st = json.load(f)
        try: _check_driver_alarms(st)            # ★드라이버 에러/두절/estop edge → 알람 로그
        except Exception: pass
        _off_live[0] = st.get('offset_deg')      # ★제어기가 **기동 시 읽은** 영점
        _refresh_mode_led(st)                    # ★모드/힘 LED — 실제 상태 기준
        if 'health' in st or 'q_leg_deg' in st:          # ── emb(app/biped_emb) 상태: LED+실측 ──
            q = st.get('q_leg_deg', [0.0] * NJ)
            health = st.get('health', ['dead'] * NJ)
            for i in range(min(NJ, len(q))):
                dpg.set_value(f'meas_{i}', f'{q[i]:+6.1f}')
            for i in range(min(NJ, len(health))):
                dpg.configure_item(f'led_{i}', fill=_LED.get(health[i], (70, 70, 78)))
            # ★분모는 **실장축 수**. 미장착을 분모에 넣으면 정상인데도 "8중 2" 로 보인다.
            n_inst = st.get('n_installed', NJ)
            line = ('[emb] mode=%s  backend=%s  정상%d/에러%d/두절%d/%d  tilt%.1f°  loop%.0fHz'
                    % (st.get('mode', '-'), st.get('backend', '-'), st.get('n_ok', 0),
                       st.get('n_fault', 0), st.get('n_dead', n_inst), n_inst,
                       st.get('tilt_deg', 0), st.get('loop_hz', 0)))
            if st.get('n_absent'):
                line += '  · 미장착%d' % st['n_absent']
            if 'home_progress' in st:            # home 모드 진행률 + 실제 도달 여부
                # ★진행률(명령 기준)과 도달(측정 기준)을 따로 보여준다 — 궤적이 끝나도
                #   부하·마찰로 실제로는 안 들어와 있을 수 있고, 그게 중요한 정보다.
                line += '\nHome 궤적 %3.0f%%%s  %s' % (
                    st['home_progress'] * 100,
                    ' (완료)' if st.get('home_done') else '',
                    '✓ 홈 도달' if st.get('home_at_goal') else '… 미도달')
        else:                                            # ── sim(biped_run/view) 상태 ──
            line = ('mode=%s  높이%.2f  vx%+.2f vy%+.2f wz%+.2f  yaw%+.0f°  tilt%.1f°  (%+.1f,%+.1f)'
                    % (st.get('mode', '-'), st.get('base_z', 0), st.get('vx_cmd', 0), pub.cmd['vy'],
                       st.get('wz_cmd', 0), st.get('yaw', 0), st.get('tilt', 0),
                       st.get('x', 0), st.get('y', 0)))
            if 'est_perr' in st:                         # biped_deploy = leg-odometry 추정오차(GT 대비)
                line += '\n추정(leg-odom) 오차: pos %.1fcm  vel %.3fm/s   EST(%+.2f,%+.2f)' % (
                    st['est_perr']*100, st['est_verr'], st.get('est_x', 0), st.get('est_y', 0))
        dpg.set_value('state', line)
    except Exception:
        _off_live[0] = None
        dpg.set_value('state', '(컨트롤러 대기중…)')
    # ★영점 표 갱신 2Hz — yaml 은 mtime 이 바뀔 때만 다시 읽으니 사실상 공짜다.
    if time.time() - _last_off_ref[0] > 0.5:
        _last_off_ref[0] = time.time()
        try: _refresh_offsets()
        except Exception: pass
    if time.time() - _last_sys[0] > 1.0:            # ★CPU·온도 1Hz
        _last_sys[0] = time.time()
        try: _refresh_sysload()
        except Exception: pass
    # ★연속 발행: 스틱을 가만히 눌러 유지해도(=drag 이벤트 없음) 명령이 계속 전송되게.
    #   dpg drag 핸들러는 마우스가 움직일 때만 발화 → 정지 유지 시 패킷 끊김 → sim이 옛 명령 유지/누락.
    #   매 프레임 현재 pub.cmd를 UDP로 재전송(≈60Hz)해 이벤트 타이밍 의존 제거.
    if _udp_sock is not None:
        try: _udp_sock.sendto(json.dumps(pub.cmd).encode(), _udp_addr)
        except Exception: pass
    # (파일 하트비트는 2026-09-03 에 **데몬 스레드로 이동** — Pub() 아래 _hb_thread 참조.
    #  렌더 루프에 두면 프레임 스톨 = 발행 중단 = 워치독 limp = 자립 중 낙하였다.)
    dpg.render_dearpygui_frame()

dpg.destroy_context()
