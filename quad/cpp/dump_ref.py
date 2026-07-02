"""Python wbic_stance 기준값 덤프 → C++ 포팅 검증용. 표준 standing 자세서 tau 계산·저장."""
import sys; sys.argv = [sys.argv[0]]
sys.path.insert(0, '/home/jsh/문서/jsh/simulation/quad')
import quad_mpc_wbic as Q
Q._ROBOT = 'ours_sphere'
import mujoco, numpy as np, json
q = Q.QuadSim()
q.crouch_home()                       # q_home·com_ref 설정 + d.qpos=standing
q.d.qvel[:] = 0.0
mujoco.mj_forward(q.m, q.d)
tau, ok = q.wbic_stance()             # 기준 토크(clip 전 raw 반환값)
out = {
    'mjcf': Q.ROBOTS['ours_sphere']['mjcf'],
    'nq': int(q.m.nq), 'nv': int(q.nv), 'nu': int(q.nu),
    'qpos': [float(x) for x in q.d.qpos], 'qvel': [float(x) for x in q.d.qvel],
    'q_home': [float(x) for x in q.q_home], 'com_ref': [float(x) for x in q.com_ref],
    'MU': float(Q.MU), 'MU_MARGIN': float(Q.MU_MARGIN), 'LAMZ_MIN': float(Q.LAMZ_MIN),
    'foot_bid': [int(x) for x in q.foot_bid], 'foot_gid': [int(x) for x in q.foot_gid],
    'foot_r': [float(x) for x in q.foot_r],
    'tau': [float(x) for x in tau],
}
json.dump(out, open('/tmp/wbic_ref.json', 'w'), indent=1)
# ★C++ 파싱용 플랫 텍스트: 각 줄 = 이름 값들...
with open('/tmp/wbic_ref.txt', 'w') as f:
    f.write('dims %d %d %d\n' % (q.m.nq, q.nv, q.nu))
    f.write('MU %g %g %g\n' % (Q.MU, Q.MU_MARGIN, Q.LAMZ_MIN))
    f.write('qpos ' + ' '.join('%.10g' % x for x in q.d.qpos) + '\n')
    f.write('qvel ' + ' '.join('%.10g' % x for x in q.d.qvel) + '\n')
    f.write('q_home ' + ' '.join('%.10g' % x for x in q.q_home) + '\n')
    f.write('com_ref ' + ' '.join('%.10g' % x for x in q.com_ref) + '\n')
    f.write('tau ' + ' '.join('%.10g' % x for x in tau) + '\n')
print('ok=%s' % ok)
print('기준 tau =', np.round(tau, 3))
print('덤프 → /tmp/wbic_ref.{json,txt} (nq=%d nv=%d nu=%d)' % (q.m.nq, q.nv, q.nu))
