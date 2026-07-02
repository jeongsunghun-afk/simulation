"""wbic_track 기준값 덤프 → C++ 포팅 검증용.
   실제 trot 루프를 헤드리스로 굴리다 '스윙이 활성인' wbic_track 호출을 낚아채
   그 순간의 전체상태(qpos/qvel)+입력(lam_des·contacts·swing)+출력(raw tau)을 저장.
   실행: env HEADLESS=1 STEPS=400 <proxddp python> dump_track.py
"""
import os, sys
sys.argv = [sys.argv[0], '--robot', 'ours_sphere', '--mode', 'trot']
os.environ.setdefault('HEADLESS', '1')
os.environ.setdefault('STEPS', '600')
os.environ['RESET_ON_FALL'] = '0'
sys.path.insert(0, '/home/jsh/문서/jsh/simulation/quad')
import numpy as np, quad_mpc_wbic as Q

TARGET_CALL = int(os.environ.get('TARGET_CALL', '5'))   # 스윙활성 N번째 호출 채택(초반 warmup 지나서)
_orig = Q.QuadSim.wbic_track
_hits = {'n': 0, 'z': None}
# solve_qp 래핑: 마지막 QP 해 z(=[q̈;λ]) 포착 → C++ qdd 직접비교용
_orig_sq = Q.solve_qp
def _sq(*a, **k):
    z = _orig_sq(*a, **k)
    _hits['z'] = z
    return z
Q.solve_qp = _sq

def _spy(self, lam_des, contacts=(0, 1, 2, 3), w_lam=10.0, swing=None):
    swing = swing or {}
    tau, ok = _orig(self, lam_des, contacts=contacts, w_lam=w_lam, swing=swing)
    # 스윙 다리가 실제 있고(=trot 비행상 존재) 유효해야 채택
    if len(swing) > 0 and len(contacts) < 4 and ok and tau is not None:
        _hits['n'] += 1
        if _hits['n'] == TARGET_CALL:
            _dump(self, lam_des, contacts, w_lam, swing, tau)
            print('덤프 완료 → /tmp/wbic_track_ref.txt (call#%d)' % TARGET_CALL)
            raise SystemExit(0)
    return tau, ok

def _dump(q, lam_des, contacts, w_lam, swing, tau):
    d = q.d
    with open('/tmp/wbic_track_ref.txt', 'w') as f:
        w = lambda name, arr: f.write(name + ' ' + ' '.join('%.12g' % x for x in np.asarray(arr).ravel()) + '\n')
        f.write('dims %d %d %d\n' % (q.m.nq, q.nv, q.nu))
        f.write('MU %g %g %g\n' % (Q.MU, Q.MU_MARGIN, Q.LAMZ_MIN))
        f.write('SW %g %g %g\n' % (Q.SW_KP, Q.SW_KD, Q.W_SW))
        f.write('w_lam %g\n' % w_lam)
        f.write('body_terr %.12g\n' % q._body_terr)
        f.write('contacts ' + ' '.join('%d' % c for c in contacts) + '\n')
        w('qpos', d.qpos); w('qvel', d.qvel)
        w('q_home', q.q_home); w('com_ref', q.com_ref)
        w('lam_des', np.asarray(lam_des))   # 4x3 → 12
        # swing: 각 다리 "swingleg <leg> px py pz vx vy vz"
        for leg, (p_des, v_des) in swing.items():
            f.write('swingleg %d ' % leg + ' '.join('%.12g' % x for x in np.concatenate([p_des, v_des])) + '\n')
        w('tau', tau)
        if _hits['z'] is not None:
            w('zsol', _hits['z'])   # 전체 QP해 [q̈(nv); λ(3K)]

Q.QuadSim.wbic_track = _spy
Q.main()
print('경고: 목표 스윙호출(%d)에 도달못함 — STEPS/TARGET_CALL 조정' % TARGET_CALL)
