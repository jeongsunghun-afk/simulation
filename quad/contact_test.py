"""접촉 모델 격리: FDDP closed-loop을 MuJoCo 박스발(ours) vs 점발(ours_sphere)로 비교.
   NMPC 모델은 동일 URDF(점접촉 sole). 점발이 훨씬 안정적이면 접촉 mismatch가 주원인."""
import sys, os; sys.argv=[sys.argv[0]]
sys.path.insert(0,'/home/jsh/문서/jsh/simulation/quad')
import quad_nmpc as Q, numpy as np, crocoddyl, mujoco, quad_sim
ROBOT=os.environ.get('MJROBOT','ours')
M=Q.NMPCModel()                              # NMPC 모델(URDF, 점접촉 sole) — 항상 동일
quad_sim._ROBOT=ROBOT; q=quad_sim.QuadSim(); q.crouch_home()  # MuJoCo 물리만 교체
m,d=q.m,q.d; pin2mj,mj_to_pin=Q._bridge(M,q); dt=0.02; sim_dt=m.opt.timestep
N=int(os.environ.get('N','24')); MI=int(os.environ.get('MI','20'))
xs_ws=[M.x0]*(N+1); us_ws=[np.zeros(M.nu)]*N; tmax=0; tfall=None; drift=0
for it in range(int(os.environ.get("STEPS","2000"))):
    x=mj_to_pin(d); t=it*sim_dt
    acts=[Q.build_action(M,*(lambda r:(r[0],))(Q._trot_schedule(t+k*dt,dt,0.5,0.5,0.06,M.foot_home,0.0)),M.x0,dt,swing_targets=Q._trot_schedule(t+k*dt,dt,0.5,0.5,0.06,M.foot_home,0.0)[1]) for k in range(N)]
    prob=crocoddyl.ShootingProblem(x.copy(),acts,Q.build_terminal(M,M.x0))
    s=crocoddyl.SolverFDDP(prob); s.solve(xs_ws,us_ws,MI,False,1e-9)
    xs_ws=list(s.xs); us_ws=list(s.us)
    # 계획 vs 실제 추적오차(접촉 mismatch 지표): 1스텝 예측 base_z 와 실제 비교
    umj=np.zeros(M.nu); umj[pin2mj]=np.array(s.us[0]); d.ctrl[:]=np.clip(umj,-80,80); mujoco.mj_step(m,d)
    drift=max(drift, abs(d.qpos[2]-s.xs[1][2]))   # 실제 vs 계획 base_z 편차
    xx,yy=d.qpos[4],d.qpos[5]; ti=np.degrees(np.arccos(np.clip(1-2*(xx*xx+yy*yy),-1,1))); tmax=max(tmax,ti)
    if tfall is None and (ti>40 or d.qpos[2]<0.2): tfall=t
print('[MuJoCo=%s] %s tilt_max=%.0f° 계획추적편차_max=%.3fm'%(ROBOT,'✅4s생존' if tfall is None else '❌@%.2fs'%tfall,tmax,drift))
