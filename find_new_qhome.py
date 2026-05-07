"""
옵션 2 helper — 발 시작점 거리를 늘리기 위한 새 Q_HOME 계산.

scipy.optimize.minimize로 FK(q) = 목표_발위치 만족하는 q 탐색.
v10의 analytical_ik 부호 규약과 무관하게 동작 (FK만 사용).
"""
import math
import numpy as np
from scipy.optimize import minimize

# v10에서 그대로 가져옴
DH_FRONT = [(+math.pi/2, 0, 0), (0, 0.21, 0.0075), (0, 0.235, 0), (0, 0.1, 0), (0, 0.045, 0)]
DH_HIND  = [(-math.pi/2, 0, 0), (0, 0.21, 0.0075), (0, 0.21, 0),  (0, 0.148, 0), (0, 0.045, 0)]
Q_HOME_FRONT_DEG = [0.0, 133.2973, 46.7027, 30.6583, 59.3417]
Q_HOME_HIND_DEG  = [0.0, -150.0, -90.0, 90.0, 60.0]
Q_HOME_FRONT = [math.radians(a) for a in Q_HOME_FRONT_DEG]
Q_HOME_HIND  = [math.radians(a) for a in Q_HOME_HIND_DEG]
BODY_FWD_F =  0.250
BODY_FWD_H = -0.250

def _dh_matrix(alpha, a, d, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [ 0,     sa,     ca,    d],
                     [ 0,      0,      0,    1]], dtype=float)

def fk(thetas, dh):
    T = np.eye(4)
    for i, (alpha, a, d) in enumerate(dh):
        T = T @ _dh_matrix(alpha, a, d, thetas[i])
    return T[:3, 3]

def _dh_to_sim(vec, front_leg=False):
    sim = np.array([vec[2], -vec[1], vec[0]], dtype=float)
    if front_leg:
        sim[:2] *= -1.0
    return sim

def _sim_to_dh(vec, front_leg=False):
    sim = np.array(vec, dtype=float)
    if front_leg:
        sim[:2] *= -1.0
    return np.array([sim[2], -sim[1], sim[0]], dtype=float)

def find_qhome(q_init, dh, target_dh, fix_q5=True):
    """FK(q) = target_dh 만족하는 q 탐색. q_init과 가까운 해 선호.
    fix_q5=True면 q[4]는 q_init[4]로 고정."""
    q_init = np.array(q_init)
    nj = len(q_init)
    def cost(q):
        # 위치 오차 + home에서 멀어지지 않도록 약한 정규화
        err = fk(q, dh) - target_dh
        reg = 1e-3 * np.sum((q - q_init)**2)
        return float(np.sum(err**2) + reg)
    cons = []
    if fix_q5:
        cons.append({'type': 'eq', 'fun': lambda q: q[4] - q_init[4]})
    res = minimize(cost, q_init, method='SLSQP', constraints=cons,
                   options={'ftol': 1e-12, 'maxiter': 200})
    return res.x, fk(res.x, dh)

def report(name, q_init, dh, dx_sim, front_leg):
    """sim x 방향으로 dx_sim만큼 발을 이동시키는 새 q 탐색."""
    p_curr_dh = fk(q_init, dh)
    p_curr_sim = _dh_to_sim(p_curr_dh, front_leg=front_leg)
    p_new_sim  = p_curr_sim + np.array([dx_sim, 0.0, 0.0])
    p_new_dh   = _sim_to_dh(p_new_sim, front_leg=front_leg)

    q_new, p_check_dh = find_qhome(q_init, dh, p_new_dh, fix_q5=True)
    p_check_sim = _dh_to_sim(p_check_dh, front_leg=front_leg)

    print(f'━━━ {name} (sim Δx={dx_sim*1e3:+.0f}mm) ━━━')
    print(f'  현재 sim foot: x={p_curr_sim[0]*1e3:+7.2f}  y={p_curr_sim[1]*1e3:+7.2f}  z={p_curr_sim[2]*1e3:+7.2f} mm')
    print(f'  목표 sim foot: x={p_new_sim[0]*1e3:+7.2f}  y={p_new_sim[1]*1e3:+7.2f}  z={p_new_sim[2]*1e3:+7.2f} mm')
    print(f'  실제 sim foot: x={p_check_sim[0]*1e3:+7.2f}  y={p_check_sim[1]*1e3:+7.2f}  z={p_check_sim[2]*1e3:+7.2f} mm  '
          f'(오차={np.linalg.norm(p_check_sim-p_new_sim)*1e3:.3f}mm)')
    print(f'  새 Q_HOME [deg]: [{", ".join(f"{math.degrees(v):.4f}" for v in q_new)}]')
    new_phi = sum(q_new[1:4])
    new_th5 = new_phi + q_new[4]
    print(f'  새 PHI = {math.degrees(new_phi):.4f}°,  THETA5_TARGET = {math.degrees(new_th5):.4f}°')
    print()
    return q_new

if __name__ == '__main__':
    # ─── 사용자 설정 ───────────────────────────────────────────
    DELTA_FRONT_MM = +50  # 앞발을 sim +x로 이동 (양수 = 더 앞)
    DELTA_HIND_MM  = -50  # 뒷발을 sim -x로 이동 (음수 = 더 뒤)
    # ───────────────────────────────────────────────────────────

    print(f'설정: 앞발 {DELTA_FRONT_MM:+d}mm, 뒷발 {DELTA_HIND_MM:+d}mm  '
          f'→ 발 간격 {DELTA_FRONT_MM - DELTA_HIND_MM:+d}mm 변화\n')
    qf = report('FRONT', Q_HOME_FRONT, DH_FRONT, DELTA_FRONT_MM*1e-3, front_leg=True)
    qh = report('HIND',  Q_HOME_HIND,  DH_HIND,  DELTA_HIND_MM*1e-3,  front_leg=False)

    # 새 발 간격 계산
    pf = _dh_to_sim(fk(qf, DH_FRONT), front_leg=True)
    ph = _dh_to_sim(fk(qh, DH_HIND),  front_leg=False)
    new_gap = (BODY_FWD_F + pf[0]) - (BODY_FWD_H + ph[0])
    print(f'━━━━━ 결과 요약 ━━━━━')
    print(f'새 발 간격 = {new_gap*1e3:.2f}mm  (현재 401.02mm)')
    print()
    print('v10에 적용하려면 다음을 수정:')
    print(f'  Q_HOME_FRONT_DEG = [{", ".join(f"{math.degrees(v):.4f}" for v in qf)}]')
    print(f'  Q_HOME_HIND_DEG  = [{", ".join(f"{math.degrees(v):.4f}" for v in qh)}]')
    print('  (PHI_FRONT/HIND, THETA5_FRONT/HIND는 자동으로 재계산됨 — sum으로 정의)')
