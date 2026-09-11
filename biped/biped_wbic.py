"""biped WBIC 균형 (B1) — 성숙 quad_control.hpp wbic_stance 를 8-DOF biped(HL/HR)로 이식.

정식화 (quad_control.hpp:314 wbic_stance 동일):
  변수 z = [ q̈(nv=14) ; λ(3·K), K=2 ]
  min  1·‖Jc q̈ − a_com‖²  +  5·‖자세 roll/pitch/yaw 레벨링‖²  +  Σ w·‖posture‖²  + reg
  s.t. 부동베이스 6행:  M[0:6] q̈ − Σ Jsᵀ λ = −h[0:6]
       접촉 3K:         Js q̈ = −STANCE_KD·(Js q̇)      (baumgarte, 터치다운 잔류속도→0)
       마찰추(피라미드) |λx|,|λy| ≤ μλz ,  λz ≥ LAMZ_MIN
  τ = M[6:] q̈ + h[6:] − Σ Jsᵀλ  → clip(±τ_peak)

옛 wbic_balance.py 대비 개선(quad 교훈): 현재-yaw 프레임 roll/pitch 레벨링(yaw 안 되당김)·STANCE_KD·발목 posture 가중·Peak토크.
헤드리스: python biped_wbic.py  (3s 균형, 드리프트 리포트) · 뷰어: VIEW=1 python biped_wbic.py
"""
from __future__ import annotations
import os, time, numpy as np, mujoco, mujoco.viewer
from qpsolvers import solve_qp

MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'biped_from_quad.mjcf')

# ── 게인/파라미터 (quad_control.hpp 기본값) ──
STANCE_KD = 20.0
W_ORI     = 5.0          # 자세 레벨링 task 가중
W_POST    = 1.0          # 관절 posture 가중(기본)
W_ANKLE   = 20.0         # 발목(foot) posture 가중 ↑ (whip 억제)
MU, MU_MARGIN = 0.8, 0.707     # μ_eff = 0.566 (덜 보수적, 뷰어 피드백. MJCF 물리 1.6)
LAMZ_MIN  = 1.0
# ⚠ 하위호환 폴백일 뿐이다. **실제로 쓰이는 값은 self.drv_peak(MJCF 에서 파생)** 이다.
#   MJCF 를 고치면 자동으로 따라가므로 여기를 손댈 일은 없다. (foot 100.8 = 12Nm × 8.4)
# ★2026-08-13 개명 TAU_PEAK → DRV_PEAK. 관절토크 한계가 아니라 **드라이브(모터)** 한계다.
#   발목 액추에이터를 tendon 으로 옮기면서 둘이 갈라졌다 — calf **관절**은 무릎·발목 두
#   드라이브를 합쳐 226.8 을 받지만, 무릎 **드라이브** 상한은 여전히 126 이다.
DRV_PEAK  = np.array([84, 84, 126, 100.8, 84, 84, 126, 100.8])   # HL/HR × (hip,thigh,calf,foot)
# ── 액추에이터 물리 — ★2026-08-06 C++ 기준으로 통일 (cpp/src/biped_control.hpp:51-61) ──
#   출처: emb/pace/RESULTS.md — HL_hip·HR_hip 을 PACE 처프로 실측 식별.
#   ⚠ 이 값들은 **C++ 가 기준**이다. 여기서 재유도하거나 되돌리지 말 것.
#     동기화 방향은 C++ → Python (emb/NEXT_HW.md §8). 종전 Python 값
#     (ROTOR_I 1e-4 / JDAMP 0.1 / JFRIC 0.5 / foot 8.0)은 **실측·안정성 스윕 이전** 값이다.
#
#   ROTOR_I : 전 관절이 동일 모터 + 공통 7:1 이고 관절별 추가 감속단만 붙으므로
#             ROTOR_I(모터축 관성)는 **전 관절 공통 상수**다. armature = ROTOR_I·N².
#
# ★★2026-08-14 **PACE 식별 최종값으로 전면 교체** (emb/pace/RESULTS.md §8).
#   종전 값은 `JDAMP 0.09 / JFRIC 0.38` **스칼라 하나를 8축 전부에** 쓰고 있었다 —
#   그 근거가 **hip 2축·다리 미장착** 실측이라, 감속단이 다른 calf/foot 에는 맞을 이유가
#   없었다(그 사실이 종전 주석에도 "장착 후 재측정" 으로 적혀 있었다).
#   지금은 8축을 다리 장착 상태에서 재고 kind 별로 넣는다.
#
#   ROTOR_I  7.4e-4 → **7.327e-4**  (foot τ_ff 경로 7.327e-4 · calf 공통속도법 7.340e-4,
#            두 축·두 방법이 **0.17%** 로 만났다 — 순환 없는 경로의 독립 검증)
#   JDAMP    0.09 스칼라 → kind 별 [0.0900, 0.1696, 0.0092, 0.1100]
#   JFRIC    0.38 스칼라 → kind 별 [0.8270, 0.5064, 0.5717, 0.2517]
#            ⚠**전 축이 종전보다 크다**(0.66~2.2배). 보행 거동이 바뀐다 — 회귀로 확인할 것.
#
#   ⚠신뢰도가 축마다 다르다. 아래 표의 표식을 그대로 옮긴다:
#     ⚠hip 의 damping·frictionloss 는 **식별된 게 아니라 고정한 값**이다 —
#       hip 자극이 비용의 4% 뿐이라 적합이 아무 값이나 고른다. 관성도 미측정.
#     ★calf damping(0.0092)·foot frictionloss(0.2517)는 **탐색범위 끝에 붙은 값**이라
#       그 방향으로 더 낮을 수 있다.
#   ⚠foot 의 dof_armature 는 **0** 이다 — tendon 으로 옮겨 갔다(_foot_rotor_to_tendon).
GEAR    = np.array([7.0, 7.0, 10.5, 8.4])   # hip,thigh,calf,foot
                                            # ★foot 8.0→8.4 (총 8.4 = 7×1.2 추가단, 2026-08-05 확인)
ROTOR_I = 6.11e-4                           # ★α-보정 물리값(2026-09-12): 옛 7.327e-4 는 I/α
                                            #   (PACE 적합에 토크스케일 항 없음 → α 가 관성에 흡수).
                                            #   벤치 측정 α=0.834 로 ×α = 6.11e-4. datasheet 5.29e-4 근접.
                                            #   ★actuator_gear ×α(아래)와 별개 항 — 중복 아님(토크 vs 관성).
#          hip     thigh    calf     foot   ← kind 순. j%4 로 색인한다
# ★2026-08-14 fit_v2 → **fit_v6**. C++ biped_control.hpp 와 **같은 값**이어야 한다.
#   JDAMP.calf 0.0092★→0 확정 · JDAMP.thigh 0.1696→0.022 · JFRIC.thigh 0.5064→0.592
#   JFRIC.foot 0.2517★→0.241.  v6 은 탐색범위 경고 0건(★가 전부 해소된 첫 판).
# ⚠thigh 의 두 값은 **짝으로만** 의미가 있다 — b 가 8.1배 흔들려도 b·q̇+τ_c 는 ±4.9%.
# ★2026-08-14(2차) 최종 문서 반영: JDAMP.thigh 0.022→0 · JFRIC.thigh 0.592→0.603
#   JFRIC.calf 0.572→0.537.  JFRIC.foot 만 옛 자료(fit_v6) 0.241 유지(§1 ◆).
# ⚠JDAMP 는 **지연과 같은 것을 본다**(적합은 b + kp·T_d 의 합만 본다). 지연을 실측
#   8.39ms 에 못박았기에 이 b 가 의미를 갖는다 — 자유로 두면 아무 값이나 된다.
# ★★2026-08-19 **손실공간 변경** — foot 의 세 항 전부 tendon(모터축). RESULTS.md §1-b.
#   모터 마찰·점성도 관절각이 아니라 raw각(q_foot+q_calf)에서 작용한다. 통제실험에서
#   초기RMS −32% · 적합 −32.9% · 게인2배 검증 −25.5% 로 네 지표 전부 이겼고,
#   무엇보다 **독립 실측(등속 스윕)과 −10~−14% 로 모였다**(종전 산포 −10~−68%).
#   ⇒ 아래 배열의 **foot 칸은 관절이 아니라 tendon 으로 간다**.
JDAMP = np.array([0.0900, 0.0000, 0.0000, 0.1100])   # hip~calf=관절축 · foot=tendon
JFRIC = np.array([0.8270, 0.6040, 0.8710, 0.6390])   # [Nm] 쿨롱마찰 (foot 0.639=tendon)

# home posture — ★2026-08-12 새 CAD(몸통 placeholder→실측)로 **재산출**. 구값 (0.05,−0.2) 폐기.
#
#   왜 바뀌어야 했나: 새 CAD 는 몸통이 3.0kg placeholder(CoM x=0)에서 실측 2.8kg(CoM x=−0.0727)
#   으로 바뀌고 고관절 부착점이 x +0.035 → −0.225 로 **26cm** 이동했다. 구 자세를 그대로 쓰면
#   HOME 에서 CoM 이 지지중심보다 **6cm 앞**에 놓인다(구 1.6cm).
#   ★그 오차가 biped_step.py:69 `nominal_off` 에 **스폰 시점에 굳어** 매 스텝 반복된다
#     ⇒ 발이 늘 CoM 뒤에 놓여 전방 폭주(명령 0.15 인데 실측 0.5~0.6 m/s) → 1초 내 낙상.
#
#   재산출 기준 — 두 불변량을 목표로 (thigh, calf) 를 풀었다(hip·foot 은 0 유지):
#       nominal_off_x = 발x − CoMx = **+0.02**   (발을 CoM 바로 아래보다 살짝 앞)
#       다리높이(CoMz − 발z)        = 0.4651     (구 모델과 동일 = ω=√(g/z) 보존)
#   ⚠구 모델의 off_x(−0.0159)를 그대로 맞추면 **안 된다** — 실측 스윕에서 −0.0159 는
#     T_STEP 0.30 이라는 폭 ±0.005 짜리 칼날 위에서만 살았고, 정지·후진이 깨졌다.
#     +0.02 는 T_STEP 을 **배포값 0.38 그대로 두고** 8조건 전부 통과한다.
#
#   검증(2026-08-12, 15s × 8조건, biped_from_quad.mjcf 점발):
#       정지 · 전진 0.05/0.10/0.15/0.20 · 후진 −0.10 · 측방 0.05 · 선회 0.2  → **8/8 무낙상**
#       tilt 3.0~4.1°(구 모델 5~6°보다 낫다) · 측방드리프트 |y| ≤ 0.03 m (VX 0.20 만 0.55)
Q_HOME = np.array([0.0, 0.203054, -0.671148, 0.0,  0.0, 0.203054, -0.671148, 0.0])
# ★평발(flat-foot) home: 발목을 눕혀 heel(foot_joint)+toe 밑창을 지면과 수평 + CoM 을 밑창중심에.
# ★2026-08-13 **새 CAD 로 재산출**. 구값 {0.25, −0.50, −1.14626} 은 구 CAD 유래이고,
#   2026-08-12 Q_HOME 재산출(커밋 37a517e) 때 **여기가 빠졌다** — 그 검증이 점발 8조건뿐이었다.
#   새 CAD 에서 구값은 CoM 이 밑창중심보다 **4.5cm 앞**(밑창 반길이 7.3cm → 전방여유 37.9%)
#   이라 전방 폭주로 1.29s 낙상했다. ⚠밑창 기울기는 구값도 0 이었다 — 눈으로는 멀쩡해 보이고
#   깨진 건 **전후 정렬**뿐이다. 그래서 오래 안 들켰다.
#   재산출 도구: cpp/src/flat_home.cpp (밑창수평 · CoM=밑창중심 · CoM높이 유지 3조건 Newton).
#   결과: CoM−밑창중심 0.00000 m(여유 100%) · CoM z 0.3649 유지 · base z 0.4451.
#   ⚠C++ Qflat8(cpp/src/biped_control.hpp)과 **같이** 고칠 것. 한쪽만 고치면 파리티가 깨진다.
Q_HOME_FLAT = np.array([0.0, 0.064256, -0.416657, -1.043858,  0.0, 0.064256, -0.416657, -1.043858])
ANKLE_IDX = [3, 7]       # HL_foot, HR_foot


# ★관절토크 τ ↔ 드라이브토크 u (2026-08-13). 발목이 링키지 구동이라 두 좌표가 다르다.
#     u_calf = τ_calf − τ_foot        u_foot = τ_foot        (hip·thigh 는 그대로)
#   발목 드라이브 좌표가 raw각(q_calf+q_foot)이므로 일률보존에서 **전치**로 들어간다.
#   축순서 = (hip,thigh,calf,foot) × 다리  ⇒  [2::4] 가 calf, [3::4] 가 foot.
#   ⚠MJCF 에서 발목 액추에이터가 tendon(coef 1,1)에 물려 있어야 이 규약이 성립한다.
#   ⚠**같은 전단이 세 곳에 있다** — 여기 · C++ cpp/src/biped_wbic.hpp ·
#     실기 emb/interface/joint_map.py:tau_ctrl_to_ch. 하나만 고치면 시뮬과 실기가 갈린다.
def tau_to_drive(tau):
    t = np.asarray(tau, float)
    u = t.copy()
    u[2::4] = t[2::4] - t[3::4]
    return u


def drive_to_tau(u):
    v = np.asarray(u, float)
    tau = v.copy()
    tau[2::4] = v[2::4] + v[3::4]
    return tau


class BipedWBIC:
    def __init__(self, mjcf=MJCF):
        self.m = mujoco.MjModel.from_xml_path(mjcf)
        self.d = mujoco.MjData(self.m)
        self.nv, self.nu = self.m.nv, self.m.nu          # 14, 8
        self.K = 2
        # ★드라이브 토크한계를 **MJCF actuator ctrlrange 에서 읽는다** (2026-08-13).
        #   하드코딩하면 감속비를 바꿀 때 따라가지 않는다 — 실제로 GEAR foot 8→8.4 로
        #   고쳤을 때 96(=12Nm×8)이 그대로 남아 peak÷gear 가 11.43 이 됐었다.
        #   ⚠종전 출처(jnt_actfrcrange, hinge 순회)는 이제 틀리다. 발목이 tendon
        #     액추에이터가 되면서 calf **관절** 한계가 226.8(=두 드라이브 합)이 됐기 때문이다.
        #     ctrlrange 는 액추에이터당 하나라 인덱스가 제어벡터와 그대로 맞는다 —
        #     "hinge 를 순서대로 세면 액추에이터와 맞는다" 는 가정 자체가 사라진다.
        #   ⚠C++ cpp/src/biped_control.hpp 와 **같은 출처**여야 한다.
        self.drv_peak = np.array([self.m.actuator_ctrlrange[i, 1]
                                  if (self.m.actuator_ctrllimited[i]
                                      and self.m.actuator_ctrlrange[i, 1] > 0) else 1e8
                                  for i in range(self.nu)])
        # ★마찰 전방보상 게인 — C++ 기본값과 **같아야** 파리티가 유지된다(env 이름도 동일).
        self.FRIC_COMP = float(os.environ.get('FRIC_COMP', 1.0))
        self.FRIC_V0 = float(os.environ.get('FRIC_V0', 0.20))
        # ★foot 상수결손 보상(08-27) — 토크부호 기반 k·tanh(τ/τ0). C++ FOOT_COMP_NM 파리티.
        #   속도기반 FRIC_COMP 와 달리 준정적/저속 힘제어에서도 작동. 기본 0(꺼짐).
        self.FOOT_COMP = float(os.environ.get('FOOT_COMP_NM', '0'))
        self.FOOT_COMP_T0 = max(1e-3, float(os.environ.get('FOOT_COMP_T0', '0.30')))  # 0 나눗셈 가드
        self.sph = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f) for f in ['HL_sphere', 'HR_sphere']]
        self.fbody = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
                      for b in ['HL_foot_contact_link', 'HR_foot_contact_link']]
        # ★다접촉(line/flat foot): 발당 접촉구 리스트 = tip + 추가구(sphere2/3/4…). 없으면 1점(하위호환).
        self.foot_spheres = [[self.sph[k]] for k in range(2)]
        for suf in ['2', '3', '4']:
            ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f'{L}_sphere{suf}') for L in ['HL', 'HR']]
            if all(i >= 0 for i in ids):
                for k in range(2): self.foot_spheres[k].append(ids[k])
        self.two_contact = len(self.foot_spheres[0]) > 1
        self.sph2 = [fs[1] for fs in self.foot_spheres] if self.two_contact else None
        # ★heel 구 보유 = 통합 모델(발목 세우면 toe만=점발, 눕히면 heel+toe=평발). 접촉모드는 런타임 자세로 결정.
        self.has_heel = self.two_contact and any(
            self.m.geom_bodyid[fs[1]] != self.fbody[k] for k, fs in enumerate(self.foot_spheres))
        # 런타임 접촉모드: 통합 모델은 평발(정적 rest) 기본, 아니면 점발. q_home/높이 목표를 결정.
        self.contact_mode = '2pt' if self.has_heel else '1pt'
        self.q_home = Q_HOME_FLAT if self.contact_mode == '2pt' else Q_HOME   # ★자세 task 기준(램프됨)
        self.qmin = self.m.jnt_range[1:, 0].copy()       # 관절 하한(freejoint 제외)
        self.qmax = self.m.jnt_range[1:, 1].copy()
        self.com_ref = None
        self.setup_gearbox()

    def _foot_comp(self, drv):
        """foot 드라이브(3·7)에 상수결손 보상 k·tanh(τ/τ0) — C++ foot_comp 와 동일."""
        if self.FOOT_COMP > 0:
            for i in (3, 7):
                drv[i] += self.FOOT_COMP * np.tanh(drv[i] / self.FOOT_COMP_T0)
        return drv

    def setup_gearbox(self):
        """반사관성(armature=Irot·N²) + 점성감쇠 + 마찰. mature와 동일(GEARBOX ON). 다리 flail 억제."""
        m = self.m
        for j in range(self.nu):                          # 액추에이터 관절 = dof 6+j (freejoint 뒤 hinge)
            N = GEAR[j % 4]
            dof = 6 + j
            m.dof_armature[dof] = ROTOR_I * N * N
            m.dof_damping[dof] = JDAMP[j % 4]          # ★kind 별(2026-08-14 PACE 최종)
            m.dof_frictionloss[dof] = JFRIC[j % 4]
        self._foot_rotor_to_tendon()
        # ★α(토크스케일) 주입 — **벤치 실측 확정 α=0.834**(2026-09-12, 이전 저울추정 0.80).
        #   명령의 α 만 실현된다. 자리는 actuator_gear (적용토크 = gear·ctrl) — 제어기(WBIC·
        #   마찰보상)는 α 를 모른 채 두는 것이 핵심: 실기와 같은 "약한 로봇" 을 재현해야
        #   보상(deploy 의 1/α FF보정)을 sim 에서 검증할 수 있다.
        #   ★기본 0.834 로 켬(측정 반영). 끄려면 ALPHA_AXIS=1. kind별 4개/축별 8개도 가능.
        #   ⚠ROTOR_I 를 물리값(×α)으로 이미 내렸으니 여기 α 와 **중복 아님**(토크 vs 관성).
        #   ⚠C++ 파리티: biped_deploy.cpp 가 1/α FF보정으로 대응(2026-09-12 반영).
        a = [float(x) for x in os.environ.get('ALPHA_AXIS', '0.834').split(',')]
        if len(a) == 1: a = a * 4
        self.ALPHA = np.array(a * 2 if len(a) == 4 else a)
        assert len(self.ALPHA) == self.nu, "ALPHA_AXIS 는 1·4·8개"
        if np.any(self.ALPHA != 1.0):
            for j in range(self.nu):
                m.actuator_gear[j, 0] *= self.ALPHA[j]
            print(f"  ★α 주입: actuator_gear ×{a}  (실기 재현 모드)")

    def _foot_rotor_to_tendon(self):
        """★foot 로터 반사관성을 dof_armature 에서 **tendon 으로 옮긴다**(calf→foot 커플링).

        foot 로터는 관절각이 아니라 raw 각으로 돈다(실기 coef=+1, biped_emb.yaml):
            raw_foot = q_foot + coef·q_calf
        ⇒ 로터 KE = ½·I_rot·N²·(q̇_foot + coef·q̇_calf)² 이라 반사관성이
          (calf, foot) **비대각**으로 걸린다:  M += a·[[coef², coef], [coef, 1]]
        ⚠`dof_armature` 는 M 의 **대각뿐**이라 이 항을 표현할 수 없다. fixed tendon 의
          `armature` 가 정확히 위 형태를 만든다(MuJoCo 3.9.0·3.11.0 지원 확인).
        ⚠**옮기는** 것이지 더하는 게 아니다 — dof_armature[foot] 을 0 으로 두지 않으면
          이중 계상된다.
        ⚠축별 측정에서는 이 항이 죽어 있었다(타축 고정 ⇒ q̇_calf=0). 전축 동시 가진
          (PACE 다축 처프)에서만 살아난다.

        검증(2026-08-12, HOME 자세):
            M[foot,foot] 0.05434 → 0.05434 (불변)   M[calf,calf] 0.11258 → 0.16480 (+46%)
            M[calf,foot] 0.00448 → 0.05669           hip·thigh 블록 변화 0
        """
        m = self.m
        tid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, f'{s}_foot_rotor')
               for s in ('HL', 'HR')]
        if any(t < 0 for t in tid):
            # 구 MJCF(tendon 없음) 호환 — 대각 armature 를 그대로 둔다. 커플링은 누락된 채다.
            print('  ⚠MJCF 에 *_foot_rotor tendon 이 없다 — calf↔foot 커플 반사관성 누락 상태로 돈다')
            return
        # ★★2026-08-19 반사관성뿐 아니라 **점성·마찰도** 옮긴다(RESULTS.md §1-b).
        #   모터의 마찰·점성도 관절각이 아니라 raw각에서 작용한다 — 논거가 armature 와 같다.
        #   관절에 두면 무릎만 돌 때 발목 모터 마찰이 calf 에 반력을 못 준다(실측 0 이었다).
        #   ⚠C++ biped_control.hpp:foot_rotor_to_tendon 과 **같이** 고칠 것.
        for j in range(self.nu):
            if j % 4 == 3:                                # foot 축 — 셋 다 대각에서 뺀다
                m.dof_armature[6 + j] = 0.0
                m.dof_damping[6 + j] = 0.0
                m.dof_frictionloss[6 + j] = 0.0
        for t in tid:
            m.tendon_armature[t] = ROTOR_I * GEAR[3] ** 2   # 0.0517
            m.tendon_damping[t] = JDAMP[3]                   # 0.110
            m.tendon_frictionloss[t] = JFRIC[3]              # 0.639
        # ★플랜트 결손 주입(08-27 무게추 브래킷): 실물 foot 은 벤치 마찰(0.639) 외에
        #   상수 ~0.36 Nm 을 더 먹는다(r_foot(G)=α−k/G). 실물 재현 모드:
        #   FOOT_FRIC_EXTRA=0.36 → tendon 마찰 가산. 제어기는 모른다(ALPHA_AXIS 와 동일 철학).
        extra = float(os.environ.get('FOOT_FRIC_EXTRA', '0'))
        if extra > 0:
            for t in tid:
                m.tendon_frictionloss[t] += extra
            print(f"  ★플랜트 foot 결손 주입: tendon frictionloss +{extra} → {m.tendon_frictionloss[tid[0]]:.3f} Nm")

    # ── 초기화: home pose 스폰 + 발 착지 높이 + com_ref = 지지중심 ──
    def reset_stand(self):
        d, m = self.d, self.m
        d.qpos[:] = 0; d.qpos[3:7] = [1, 0, 0, 0]
        d.qvel[:] = 0; d.qacc[:] = 0                  # ★런타임 재정착(모드전환) 시 잔류속도 제거
        d.qpos[7:] = Q_HOME_FLAT if self.contact_mode == '2pt' else Q_HOME   # ★2점=눕힌 발목 home
        d.qpos[2] = 0.7
        mujoco.mj_forward(m, d)
        _allsph = [g for fs in self.foot_spheres for g in fs]
        zmin = min(d.geom_xpos[s][2] - m.geom_size[s][0] for s in _allsph)   # 모든 접촉구 바닥 z=0
        d.qpos[2] -= zmin
        mujoco.mj_forward(m, d)
        fp = np.array([self.foot_center(k) for k in range(2)])               # 발 중심(2구 평균)
        self.com_ref = np.array([fp[:, 0].mean(), fp[:, 1].mean(), d.subtree_com[0][2]])  # 지지중심 xy + 현 CoM z
        # ★평발 swing 발 수평 목표 = home(밑창 수평)에서의 발 world quat (yaw=0 기준). swing 중 이 자세로 유지.
        self.foot_home_quat = [d.xquat[self.fbody[k]].copy() for k in range(2)]
        self.q_home = (Q_HOME_FLAT if self.contact_mode == '2pt' else Q_HOME).copy()   # 스폰 자세 = q_home

    def foot_jac(self, k):
        jacp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, self.d, jacp, None, self.d.geom_xpos[self.sph[k]], self.fbody[k])
        return jacp

    # ── 2점 접촉 헬퍼 ──
    def foot_jac_at(self, geom, body):
        jacp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, self.d, jacp, None, self.d.geom_xpos[geom], body)
        return jacp

    def contact_pts(self, stance):
        """stance 발 → 접촉점 [(geom, body, foot), ...]. ★실제 지면 근처 구만(적응: 발 기울면 뜬 구 제외).
        둘 다 닿음=2점(line-foot 이득) / 하나만 닿음=1점(점발처럼, 과구속 회피)."""
        pts = []
        for f in stance:
            insph = [g for g in self.foot_spheres[f]
                     if self.d.geom_xpos[g][2] < self.m.geom_size[g][0] + 0.006]   # 바닥 6mm 이내=접촉
            if not insph:
                insph = [self.foot_spheres[f][0]]   # 다 뜸(이례적)=tip 폴백
            for g in insph:
                pts.append((g, int(self.m.geom_bodyid[g]), f))   # ★구별 실제 body(heel=foot_link, toe=contact_link)
        return pts

    def _ref_sph(self, leg):
        """발 기준 구 = 접촉모드별. 점발=tip 하나(heel은 공중), 평발=heel+toe 전부(밑창중점)."""
        return self.foot_spheres[leg] if getattr(self, 'contact_mode', '1pt') == '2pt' else [self.foot_spheres[leg][0]]

    def foot_center(self, leg):
        """발 기준점(world) = 모드별 기준구 평균(점발=tip 위치, 평발=밑창중점)."""
        return np.mean([self.d.geom_xpos[g] for g in self._ref_sph(leg)], axis=0)

    def foot_jac_center(self, leg):
        """발 중심 자코비안(swing task용) = 모드별 기준구 자코비안 평균. ★구별 실제 body."""
        return np.mean([self.foot_jac_at(g, int(self.m.geom_bodyid[g])) for g in self._ref_sph(leg)], axis=0)

    # ── WBIC stance QP (1틱) ──
    def wbic_stance(self):
        d, m, nv, nu = self.d, self.m, self.nv, self.nu
        cpts = self.contact_pts([0, 1])            # ★양발 stance 접촉점(2접촉=4점)
        K = len(cpts); nz = nv + 3 * K
        M = np.zeros((nv, nv)); mujoco.mj_fullM(m, M, d.qM)
        h = d.qfrc_bias.copy(); qv = d.qvel.copy()
        Js = [self.foot_jac_at(g, b) for (g, b, f) in cpts]
        Jc = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
        com = d.subtree_com[0].copy()

        P = np.zeros((nz, nz)); g = np.zeros(nz)
        # CoM task (weight 1)
        a_com = np.array([120, 120, 200]) * (self.com_ref - com) - np.array([20, 20, 25]) * (Jc @ qv)
        P[:nv, :nv] += Jc.T @ Jc; g[:nv] -= Jc.T @ a_com
        # 자세 레벨링: 현재-yaw 프레임서 roll/pitch/yaw (yaw 0으로 안 되당김)
        qc = d.qpos[3:7]
        yaw = np.arctan2(2 * (qc[0]*qc[3] + qc[1]*qc[2]), 1 - 2 * (qc[2]**2 + qc[3]**2))
        qlev = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        oerr = np.zeros(3); mujoco.mju_subQuat(oerr, qc, qlev)
        for j in range(3):
            a = 150 * (-oerr[j]) - 20 * qv[3 + j]
            P[3 + j, 3 + j] += W_ORI; g[3 + j] -= W_ORI * a
        # 관절 posture (nullspace)
        for j in range(nu):
            a = 60 * (self.q_home[j] - d.qpos[7 + j]) - 5 * qv[6 + j]
            w = W_ANKLE if j in ANKLE_IDX else W_POST
            P[6 + j, 6 + j] += w; g[6 + j] -= w * a
        # 정칙화
        P[:nv, :nv] += 1e-4 * np.eye(nv)
        for k in range(K):
            P[nv + 3*k:nv + 3*k + 3, nv + 3*k:nv + 3*k + 3] += 1e-3 * np.eye(3)

        # 등식: 부동베이스 6 + 접촉 3K
        A = np.zeros((6 + 3 * K, nz)); b = np.zeros(6 + 3 * K)
        A[:6, :nv] = M[:6, :]; b[:6] = -h[:6]
        for k in range(K):
            A[:6, nv + 3*k:nv + 3*k + 3] = -Js[k][:, :6].T
            A[6 + 3*k:6 + 3*k + 3, :nv] = Js[k]
            b[6 + 3*k:6 + 3*k + 3] = -STANCE_KD * (Js[k] @ qv)
        # 부등식: 마찰추 4 + λz≥min 1, per foot
        G = np.zeros((5 * K, nz)); hh = np.zeros(5 * K)
        mu = MU * MU_MARGIN; sgn = [(1, 0), (-1, 0), (0, 1), (0, -1)]; r = 0
        for k in range(K):
            o = nv + 3 * k
            for sx, sy in sgn:
                G[r, o] = sx; G[r, o + 1] = sy; G[r, o + 2] = -mu; r += 1
            G[r, o + 2] = -1.0; hh[r] = -LAMZ_MIN; r += 1

        P = 0.5 * (P + P.T) + 1e-8 * np.eye(nz)
        # ★실제 발당 2점 접지 시만 rank-deficient → proxqp. 점발(발당1점)=quadprog(강건·결정적).
        _fc = {}
        for (_g, _b, _f) in cpts: _fc[_f] = _fc.get(_f, 0) + 1
        x = solve_qp(P, g, G, hh, A, b, solver='proxqp' if any(n > 1 for n in _fc.values()) else 'quadprog')
        if x is None:
            return False
        qdd = x[:nv]
        tau = M[6:, :] @ qdd + h[6:]
        for k in range(K):
            tau -= Js[k][:, 6:].T @ x[nv + 3*k:nv + 3*k + 3]
        # ★d.ctrl 은 이제 **드라이브 토크**다 (2026-08-13, MJCF 발목 액추에이터 tendon 이전).
        #   관절토크를 그대로 쓰면 발목 모터가 tendon(coef 1,1)이라 무릎에 발목토크가 덤으로
        #   실린다. 클립도 드라이브 한계로 — 실기 한계는 모터에 걸리지 관절에 걸리지 않는다.
        # ★마찰 전방보상 — C++ biped_control.hpp:set_ctrl_from_tau 와 **같은 식**이어야 한다.
        #   WBIC 는 관절 쿨롱마찰을 모른다(JFRIC 은 모델에만 들어간다). 축별 실측 마찰
        #   (hip 0.827)에선 그 미보상 외란 때문에 2점 stand 가 20.7s 에 넘어졌다.
        #   ⚠**2점 평발(cmode='2pt')에서만** 켠다 — 보행에선 음의 감쇠로 작용해 해롭다
        #     (배포경로 vx0.20 이 2회 낙상). 근거는 C++ 쪽 주석에 정리.
        # ★2026-08-19 보상도 **손실이 있는 좌표**에서. foot 마찰은 tendon(raw각)에 있으므로
        #   판단속도는 L̇ = q̇_calf + q̇_foot 이고, 결과는 coefᵀ=(1,1) 로 두 관절에 걸린다.
        if self.FRIC_COMP > 0 and self.contact_mode == '2pt':
            dq = d.qvel[6:6 + self.nu]
            comp = np.zeros(self.nu)
            for leg in range(2):
                b = 4 * leg
                for k in range(3):                                   # hip·thigh·calf = 관절축
                    comp[b + k] = JFRIC[k] * np.tanh(dq[b + k] / self.FRIC_V0)
                f = JFRIC[3] * np.tanh((dq[b + 2] + dq[b + 3]) / self.FRIC_V0)
                comp[b + 2] += f; comp[b + 3] += f                   # tendon → coefᵀ
            tau = tau + self.FRIC_COMP * comp
        d.ctrl[:] = np.clip(self._foot_comp(tau_to_drive(tau)), -self.drv_peak, self.drv_peak)
        return True


def base_rpy(qc):
    r = np.arctan2(2*(qc[0]*qc[1] + qc[2]*qc[3]), 1 - 2*(qc[1]**2 + qc[2]**2))
    p = np.arcsin(np.clip(2*(qc[0]*qc[2] - qc[3]*qc[1]), -1, 1))
    y = np.arctan2(2*(qc[0]*qc[3] + qc[1]*qc[2]), 1 - 2*(qc[2]**2 + qc[3]**2))
    return np.degrees([r, p, y])


def main():
    c = BipedWBIC()
    c.reset_stand()
    m, d = c.m, c.d
    print(f"모델 nv={c.nv} nu={c.nu} · com_ref={np.round(c.com_ref,3)} · 초기 base z={d.qpos[2]:.3f}")
    T = float(os.environ.get('T', 3.0))
    steps = int(T / m.opt.timestep)
    fails = 0
    view = os.environ.get('VIEW', '0') == '1'
    viewer = mujoco.viewer.launch_passive(m, d) if view else None
    z0 = d.qpos[2]
    for i in range(steps):
        if not c.wbic_stance():
            fails += 1
        mujoco.mj_step(m, d)
        if viewer is not None and i % 10 == 0:
            viewer.sync()
        if d.qpos[2] < 0.2:                     # 낙상
            print(f"❌ 낙상 @ t={i*m.opt.timestep:.2f}s (base z={d.qpos[2]:.3f})")
            break
    rpy = base_rpy(d.qpos[3:7])
    print(f"t={min(i*m.opt.timestep, T):.2f}s 종료 · QP실패 {fails}회")
    print(f"base pos={np.round(d.qpos[:3],4)}  (z 드리프트 {d.qpos[2]-z0:+.4f})")
    print(f"base rpy(deg)={np.round(rpy,2)}  · tilt={np.hypot(rpy[0],rpy[1]):.2f}°")
    print("✅ 균형 유지" if d.qpos[2] > 0.2 and np.hypot(rpy[0], rpy[1]) < 15 else "⚠️ 불안정")
    if viewer is not None:
        while viewer.is_running():
            c.wbic_stance(); mujoco.mj_step(m, d); viewer.sync(); time.sleep(m.opt.timestep)


if __name__ == '__main__':
    main()
