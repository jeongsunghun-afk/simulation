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
  if(getenv("TROT_WZ")) ctrl.WZ=atof(getenv("TROT_WZ"));   // ★선회 각속도 테스트
  if(getenv("GAIT")) ctrl.set_gait(getenv("GAIT"));        // ★게이트 테스트(trot/walk/gallop)
  if(getenv("AUTO_WHIP") && strcmp(getenv("AUTO_WHIP"),"0")) ctrl.auto_whip=true;  // ★속도↑→whip↑
  if(getenv("ALIP") && !strcmp(getenv("ALIP"),"0")) ctrl.ALIP=false;
  if(getenv("POS_HOLD") && !strcmp(getenv("POS_HOLD"),"0")) ctrl.POS_HOLD=false;
  mjModel*m=q.m; mjData*d=q.d; double dt=m->opt.timestep;
  if(getenv("DBG")) std::printf("[dbg] nu=%d leg_dof=[%d %d %d %d] standing_z=%.5f com_ref=[%.5f %.5f %.5f]\n",
      q.nu,q.leg_dof[0],q.leg_dof[1],q.leg_dof[2],q.leg_dof[3],d->qpos[2],q.com_ref[0],q.com_ref[1],q.com_ref[2]);

  int falls=0; double max_tilt=0, penF=0, penR=0, pitchSum=0, tauEff=0, calfTau=0; int pn=0;
  auto t0=std::chrono::high_resolution_clock::now();
  for(int step=0; step<STEPS; step++){
    ctrl.control(); mj_step(m,d);
    double td=ctrl.tiltdeg(); max_tilt=std::max(max_tilt,td);
    if(td>50||d->qpos[2]<0.2) falls++;
    if(d->time>1.5){ // 정착후 앞/뒤 발침투 평균(진단): 스텝별 최소침투를 누적
      double pf=0,pr=0;
      for(int ci=0;ci<d->ncon;ci++){ const auto&c=d->contact[ci];
        for(int fi=0;fi<4;fi++) if(c.geom1==q.fgid[fi]||c.geom2==q.fgid[fi]){
          if(fi>=2) pf=std::min(pf,c.dist); else pr=std::min(pr,c.dist); } }
      penF+=pf; penR+=pr;
      double R[9]; mju_quat2Mat(R,&d->qpos[3]); pitchSum+=std::asin(std::max(-1.0,std::min(1.0,-R[6])))*180/M_PI; pn++;
      for(int j=0;j<q.nu;j++) tauEff+=std::abs(d->ctrl[j]);   // 총 토크 effort(에너지 대리)
      for(int i=0;i<4;i++){ int cj=q.legqv[i][2]-6; if(cj>=0&&cj<q.nu) calfTau+=std::abs(d->ctrl[cj]); } }  // calf(whip 관절) 토크
    if(step%250==0){ double*qq=&d->qpos[3];
      double yaw=std::atan2(2*(qq[0]*qq[3]+qq[1]*qq[2]),1-2*(qq[2]*qq[2]+qq[3]*qq[3]))*180/M_PI;
      std::printf("[hl] s=%d t=%.2f z=%.3f x=%+.3f y=%+.3f yaw=%+.0f° tilt=%.1f falls=%d\n",
                  step,d->time,d->qpos[2],d->qpos[0],d->qpos[1],yaw,td,falls); }
  }
  double wall=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
  std::printf("\n=== 종료: STEPS=%d(%.1fs) x=%+.3f z=%.3f max_tilt=%.1f° falls=%d | ★침투평균 앞=%.1fmm 뒤=%.1fmm pitch=%.1f° | %.0f steps/s ===\n",
              STEPS,STEPS*dt,d->qpos[0],d->qpos[2],max_tilt,falls,pn?penF/pn*1000:0,pn?penR/pn*1000:0,pn?pitchSum/pn:0,STEPS/wall);
  std::printf("    토크effort 평균Σ|τ|=%.1fNm  calf평균Σ|τ|=%.2fNm (whip 관절)\n", pn?tauEff/pn:0, pn?calfTau/pn:0);
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
