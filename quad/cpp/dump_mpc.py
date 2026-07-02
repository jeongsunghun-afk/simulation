"""MPC(mpc_qp_plan) 기준값 덤프 → C++ 포팅 검증용.
   실제 trot 루프를 헤드리스로 굴리다 armed 상태의 mpc_qp_plan 호출을 낚아채
   입력(x0·contact_schedule·foot_positions·x_ref)+출력(lam_des)+MPC상수를 저장.
   실행: env HEADLESS=1 STEPS=600 <proxddp python> dump_mpc.py
"""
import os, sys
sys.argv = [sys.argv[0], '--robot', 'ours_sphere', '--mode', 'trot']
os.environ.setdefault('HEADLESS', '1'); os.environ.setdefault('STEPS', '600')
os.environ['RESET_ON_FALL'] = '0'; os.environ['DETECT'] = '0'   # 순수 스케줄(C++ 첫 포팅과 정합)
sys.path.insert(0, '/home/jsh/문서/jsh/simulation/quad')
sys.path.insert(0, '/home/jsh/문서/jsh/simulation')
import numpy as np, quad_mpc_wbic as Q

TARGET_CALL = int(os.environ.get('TARGET_CALL', '4'))
_hits = {'n': 0}

def _dump(MPC, x0, cs, fp, x_ref, lam):
    with open('/tmp/mpc_ref.txt', 'w') as f:
        w = lambda nm, a: f.write(nm + ' ' + ' '.join('%.12g' % x for x in np.asarray(a).ravel()) + '\n')
        f.write('N_MPC %d\n' % MPC.N_MPC)
        f.write('DT_MPC %.12g\n' % MPC.DT_MPC)
        f.write('TOTAL_MASS %.12g\n' % MPC.TOTAL_MASS)
        f.write('G_ACC %.12g\n' % 9.81)
        f.write('MU_FRICTION %.12g\n' % MPC.MU_FRICTION)
        f.write('LAMZ %.12g %.12g\n' % (MPC.LAMZ_MIN, MPC.LAMZ_MAX))
        w('BODY_INERTIA', MPC._I_BODY)       # 3x3
        w('MPC_Q', np.diag(MPC.MPC_Q))       # 13
        w('MPC_R', np.diag(MPC.MPC_R))       # 3
        w('x0', x0)                          # 13
        w('x_ref', x_ref)                    # 13
        # contact_schedule: N x 4  (0/1)
        for k in range(MPC.N_MPC):
            f.write('cs %d ' % k + ' '.join('%d' % int(cs[k, i]) for i in range(4)) + '\n')
        # foot_positions: N x 4 x 3 — 매 k 동일(foot_rel)이므로 k=0만 있어도 되나 전량 저장
        w('fp0', fp[0])                      # 4x3 (k=0; C++서 전 k 복제)
        w('lam_des', lam)                    # 4x3
    print('MPC 덤프 완료 → /tmp/mpc_ref.txt (call#%d, N=%d)' % (TARGET_CALL, MPC.N_MPC))

_orig_plan = None
def _spy_plan(x0, contact_schedule, foot_positions, x_ref_step=None, ltv=False):
    lam = _orig_plan(x0, contact_schedule, foot_positions, x_ref_step=x_ref_step, ltv=ltv)
    _hits['n'] += 1
    if _hits['n'] == TARGET_CALL:
        _dump(_MPC, x0, contact_schedule, foot_positions, x_ref_step, lam)
        raise SystemExit(0)
    return lam

# setup_mpc 이후 MPC 모듈이 로드되므로, QuadSim.setup_mpc를 감싸 plan을 래핑
_MPC = None
_orig_setup = Q.QuadSim.setup_mpc
def _setup(self):
    global _MPC, _orig_plan
    _orig_setup(self)
    _MPC = self.MPC
    _orig_plan = _MPC.mpc_qp_plan
    _MPC.mpc_qp_plan = _spy_plan
Q.QuadSim.setup_mpc = _setup

Q.main()
print('경고: 목표 MPC호출(%d) 미도달' % TARGET_CALL)
