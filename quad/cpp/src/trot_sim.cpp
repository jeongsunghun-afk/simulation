// trot_sim — quad_mpc_wbic mode_trot 핵심경로 C++ closed-loop (헤드리스). 제어=TrotCtrl(trot_view와 공유).
// 대상: standalone 평지 trot (DETECT=0 순수스케줄). 검증: falls=0 + 전진거리·tilt를 Python과 비교.
#include "trot_controller.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

int main(int argc,char**argv){
  const char* path=argc>1?argv[1]:"../quad_real_sphere.mjcf";
  int STEPS = (argc>2)?atoi(argv[2]) : (getenv("STEPS")?atoi(getenv("STEPS")):3000);
  QuadControl q; q.load(path); apply_env_gains(q); q.crouch_home(); q.setup_mpc();
  TrotCtrl ctrl(q);
  if(getenv("TROT_V")) ctrl.V=atof(getenv("TROT_V"));
  if(getenv("ALIP") && !strcmp(getenv("ALIP"),"0")) ctrl.ALIP=false;
  if(getenv("POS_HOLD") && !strcmp(getenv("POS_HOLD"),"0")) ctrl.POS_HOLD=false;
  mjModel*m=q.m; mjData*d=q.d; double dt=m->opt.timestep;
  if(getenv("DBG")) std::printf("[dbg] nu=%d leg_dof=[%d %d %d %d] standing_z=%.5f com_ref=[%.5f %.5f %.5f]\n",
      q.nu,q.leg_dof[0],q.leg_dof[1],q.leg_dof[2],q.leg_dof[3],d->qpos[2],q.com_ref[0],q.com_ref[1],q.com_ref[2]);

  int falls=0; double max_tilt=0;
  auto t0=std::chrono::high_resolution_clock::now();
  for(int step=0; step<STEPS; step++){
    ctrl.control(); mj_step(m,d);
    double td=ctrl.tiltdeg(); max_tilt=std::max(max_tilt,td);
    if(td>50||d->qpos[2]<0.2) falls++;
    if(step%500==0) std::printf("[hl] s=%d t=%.2f z=%.3f x=%+.3f y=%+.3f tilt=%.1f Veff=%.2f falls=%d\n",
                                step,d->time,d->qpos[2],d->qpos[0],d->qpos[1],td,ctrl.Veff_dbg,falls);
  }
  double wall=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
  std::printf("\n=== 종료: STEPS=%d(%.1fs) x=%+.3f y=%+.3f z=%.3f max_tilt=%.1f° falls=%d | wall=%.2fs(%.0f steps/s) ===\n",
              STEPS,STEPS*dt,d->qpos[0],d->qpos[1],d->qpos[2],max_tilt,falls,wall,STEPS/wall);
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
