// trot_sim — quad_mpc_wbic.py mode_trot 핵심경로 C++ closed-loop.
// 대상: standalone 평지 trot (DETECT=0 순수스케줄, GUI/점프/getup/지형 제외).
// 검증: falls=0 + 전진거리·tilt를 Python(DETECT=0)과 비교.
#include "quad_control.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// ── gait 상수(config + trot 프리셋) ──
static const double GP_T=0.50, GP_SWF=0.50;
static const double GP_OFFSET[4]={0.0,0.5,0.5,0.0};   // HL,HR,FL,FR
static const double T_SW=0.25, T_ST=0.25, STANCE_DELTA=0.005;
static const double WARMUP=0.6, SETTLE=0.5, ACC=0.6;
static const double KCAP=0.16, RAIBERT_K=0.8, RAI_CLIP=0.25;

// gait(i,tg) → (is_stance, s_prog)
static inline void gait(int i,double tg,bool&stance,double&sprog){
  double ph=std::fmod(tg/GP_T+GP_OFFSET[i],1.0); if(ph<0) ph+=1.0;
  if(ph>=GP_SWF){ stance=true; sprog=0.0; } else { stance=false; sprog=ph/GP_SWF; }
}

// swing Z 6차계수 (Zeng 2019): A[c2,c4,c6]=b
static void swing_z_coeffs(double sh,double Th,double Vz,double&c2,double&c4,double&c6){
  Matrix3d A; A<< Th*Th, std::pow(Th,4), std::pow(Th,6),
                  2.0, 12.0*Th*Th, 30.0*std::pow(Th,4),
                  2.0*Th, 4.0*std::pow(Th,3), 6.0*std::pow(Th,5);
  Vector3d b(-sh,0.0,-Vz); Vector3d c=A.colPivHouseholderQr().solve(b);
  c2=c[0]; c4=c[1]; c6=c[2];
}
// swing_foot_pos (Zeng Scheme I): X 5차, Y 5차, Z 6차
static Vector3d swing_foot_pos(double sw_t,const Vector3d&p0,const Vector3d&pe,const Vector3d&bvel,double sh,double tau_land=1.0){
  if(sw_t>=tau_land) return pe;
  double tau=sw_t/tau_land;
  double s5=10*std::pow(tau,3)-15*std::pow(tau,4)+6*std::pow(tau,5);
  double Tl=tau_land*T_SW;
  double DXx=(pe[0]-p0[0])+bvel[0]*Tl;
  Vector3d pos;
  pos[0]=p0[0]-bvel[0]*tau*Tl+DXx*s5;
  pos[1]=(1.0-s5)*p0[1]+s5*pe[1];
  double Th=Tl/2.0, u=tau*Tl-Th, Vz=STANCE_DELTA*M_PI/T_ST;
  double c2,c4,c6; swing_z_coeffs(sh,Th,Vz,c2,c4,c6);
  double zoff=sh+c2*u*u+c4*std::pow(u,4)+c6*std::pow(u,6);
  pos[2]=p0[2]+zoff;
  return pos;
}
static inline double clip(double v,double lo,double hi){ return v<lo?lo:(v>hi?hi:v); }

int main(int argc,char**argv){
  const char* path=argc>1?argv[1]:"../quad_real_sphere.mjcf";
  int STEPS = (argc>2)?atoi(argv[2]) : (getenv("STEPS")?atoi(getenv("STEPS")):3000);
  double V = getenv("TROT_V")?atof(getenv("TROT_V")):0.30;
  bool ALIP = !(getenv("ALIP") && !strcmp(getenv("ALIP"),"0"));
  bool POS_HOLD = !(getenv("POS_HOLD") && !strcmp(getenv("POS_HOLD"),"0"));
  QuadControl q; q.load(path);
  // 모델별 파라미터(14dof 기본 / 17dof는 env로: BASE_Z0=0.5234 REAR_ANKLE=-0.5 W_AM=5)
  if(getenv("BASE_Z0")) q.base_z0=atof(getenv("BASE_Z0"));
  if(getenv("REAR_ANKLE")){ q.REAR_ANKLE=atof(getenv("REAR_ANKLE")); q.FRONT_ANKLE=q.REAR_ANKLE; }
  if(getenv("FRONT_ANKLE")) q.FRONT_ANKLE=atof(getenv("FRONT_ANKLE"));
  if(getenv("W_AM")) q.W_AM=atof(getenv("W_AM"));
  if(getenv("PIN_ANKLE")) q.stance_pin_ankle=true;   // (실험용) stance 발목핀
  q.crouch_home(); q.setup_mpc();
  if(getenv("DBG")) std::printf("[dbg] nu=%d leg_dof=[%d %d %d %d] standing_z=%.5f com_ref=[%.5f %.5f %.5f]\n",
      q.nu,q.leg_dof[0],q.leg_dof[1],q.leg_dof[2],q.leg_dof[3],q.d->qpos[2],q.com_ref[0],q.com_ref[1],q.com_ref[2]);
  mjModel*m=q.m; mjData*d=q.d; int nv=q.nv;
  double dt=m->opt.timestep;
  // 상태
  bool armed=false; double t0=0, settle_until=SETTLE;
  double Vs=0,Vys=0,Ws=0, yaw_ref=0; bool yaw_hold_set=false; double yaw_hold=0;
  bool pos_hold_set=false; double phx=0,phy=0;
  VectorXd x_ref=VectorXd::Zero(13);
  std::array<Vector3d,4> liftoff, nominal; std::array<Vector2d,4> hip_off; std::array<double,4> gz;
  std::array<bool,4> have_prev={false,false,false,false}; std::array<Vector3d,4> ptgt_prev;
  Vector3d lam_des[4]={Vector3d::Zero(),Vector3d::Zero(),Vector3d::Zero(),Vector3d::Zero()};
  double mpc_t=-1.0;
  int falls=0; double max_tilt=0;
  auto quat_yaw=[&](){ double*qq=&d->qpos[3];
    return std::atan2(2*(qq[0]*qq[3]+qq[1]*qq[2]),1-2*(qq[2]*qq[2]+qq[3]*qq[3])); };
  auto tiltdeg=[&](){ double R[9]; mju_quat2Mat(R,&d->qpos[3]); return std::acos(clip(R[8],-1,1))*180/M_PI; };

  auto tstart=std::chrono::high_resolution_clock::now();
  for(int step=0; step<STEPS; step++){
    double t=d->time;
    // 1) settle
    if(t < settle_until){ q.wbic_stance(); mj_step(m,d);
      double td=tiltdeg(); if(td>50||d->qpos[2]<0.2) falls++; max_tilt=std::max(max_tilt,td); continue; }
    // 2) arm
    if(!armed){ armed=true; t0=t; yaw_ref=0;
      for(int i=0;i<4;i++){ nominal[i]=q.foot_point(i); liftoff[i]=q.foot_point(i);
        hip_off[i]=q.foot_hip_off[i]; gz[i]=q.foot_gz0[i]; }
      x_ref.setZero(); x_ref[5]=d->subtree_com[2]; x_ref[12]=-9.81;
    }
    double tg=t-t0;
    // 3) 명령 스무딩 + warmup
    bool go = tg>WARMUP;
    double vt = go?V:0.0, vyt=0.0, wt=0.0;
    Vs  += clip(vt -Vs, -ACC*dt, ACC*dt);
    Vys += clip(vyt-Vys,-ACC*dt, ACC*dt);
    Ws  += clip(wt -Ws, -2.0*dt, 2.0*dt);
    double Veff=Vs, Vyeff=Vys, Weff=Ws;
    double yaw_m=quat_yaw();
    if(std::abs(Weff)>0.02){ yaw_ref=clip(yaw_ref+Weff*dt, yaw_m-0.3, yaw_m+0.3); yaw_hold_set=false; }
    else { if(!yaw_hold_set){ yaw_hold=yaw_m; yaw_hold_set=true; } yaw_ref=yaw_hold; }
    double cy=std::cos(yaw_m), sy=std::sin(yaw_m);
    double vx_w=Veff*cy-Vyeff*sy, vy_w=Veff*sy+Vyeff*cy;
    if(POS_HOLD && std::abs(Veff)<0.03 && std::abs(Vyeff)<0.03 && std::abs(Weff)<0.05){
      if(!pos_hold_set){ phx=d->qpos[0]; phy=d->qpos[1]; pos_hold_set=true; }
      vx_w += clip(-0.6*(d->qpos[0]-phx),-0.15,0.15);
      vy_w += clip(-0.6*(d->qpos[1]-phy),-0.15,0.15);
    } else pos_hold_set=false;
    x_ref[2]=yaw_ref; x_ref[8]=Weff; x_ref[9]=vx_w; x_ref[10]=vy_w;
    q._body_terr=0.0;
    // 4) stance/swing 분할 (DETECT=0 순수스케줄)
    std::vector<int> st; std::map<int,std::pair<Vector3d,Vector3d>> swing;
    for(int i=0;i<4;i++){ bool sch; double sp; gait(i,tg,sch,sp);
      if(sch){ st.push_back(i); have_prev[i]=false; }
      else { if(sp<0.03) liftoff[i]=q.foot_point(i); } }
    // 5) 발배치 (Raibert + ALIP)
    std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
    Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
    Map<VectorXd> qv(d->qvel,nv); Vector3d vcom=Jc*qv;
    Vector2d v_des(vx_w,vy_w), v_fb=vcom.head(2);
    if(ALIP){ mj_subtreeVel(m,d);
      Vector3d L(d->subtree_angmom[0],d->subtree_angmom[1],d->subtree_angmom[2]);
      double H=std::max(0.1,d->subtree_com[2]); double mtot=q.mpc.TOTAL_MASS;
      v_fb += Vector2d(L[1],-L[0])/(mtot*H); }
    Vector2d rai; for(int k=0;k<2;k++) rai[k]=clip(RAIBERT_K*T_ST*v_des[k]+KCAP*(v_fb[k]-v_des[k]),-RAI_CLIP,RAI_CLIP);
    Matrix2d Rw; Rw<<cy,-sy,sy,cy;
    double STH=0.10;
    double sh = STH*(0.2+0.8*std::min(1.0,tg/WARMUP));
    // 6) swing 발끝 목표
    for(int i=0;i<4;i++){ bool sch; double s_; gait(i,tg,sch,s_); if(sch) continue;
      Vector2d hip_xy(d->xpos[q.hip_bid[i]*3],d->xpos[q.hip_bid[i]*3+1]);
      Vector2d pe_xy=hip_xy+Rw*hip_off[i]+rai;   // (선회 tw=0)
      Vector3d p_end(pe_xy[0],pe_xy[1],gz[i]);
      double liftz=liftoff[i][2], dzl=p_end[2]-liftz;
      Vector3d bvel(vcom[0],vcom[1],0.0);
      Vector3d p_tgt=swing_foot_pos(s_,liftoff[i],p_end,bvel,sh,1.0);
      double zf=dzl*(10*std::pow(s_,3)-15*std::pow(s_,4)+6*std::pow(s_,5)); p_tgt[2]+=zf;
      Vector3d v_tgt=Vector3d::Zero();
      if(have_prev[i]){ for(int c=0;c<3;c++) v_tgt[c]=clip((p_tgt[c]-ptgt_prev[i][c])/dt,-1.0,1.0); }
      ptgt_prev[i]=p_tgt; have_prev[i]=true; swing[i]={p_tgt,v_tgt};
    }
    // 7) MPC 재계획(50Hz)
    double dmpc=t-mpc_t;
    if(!st.empty() && (mpc_t<0 || dmpc<0 || dmpc>=q.mpc.DT)){
      std::vector<std::array<int,4>> cs(q.mpc.N);
      for(int k=0;k<q.mpc.N;k++){ for(int i=0;i<4;i++){ bool sch; double sp; gait(i,tg+k*q.mpc.DT,sch,sp); cs[k][i]=sch?1:0; } }
      Matrix<double,4,3> L=q.mpc_grf(x_ref,cs);
      for(int i=0;i<4;i++) lam_des[i]=L.row(i).transpose();
      mpc_t=t;
    }
    Vector3d lam_use[4];
    for(int i=0;i<4;i++) lam_use[i]= st.empty()?Vector3d::Zero():lam_des[i];
    // 8) WBIC
    bool ok=q.wbic_track(st,swing,lam_use);
    if(!ok) q.wbic_stance();
    mj_step(m,d);
    double td=tiltdeg(); max_tilt=std::max(max_tilt,td);
    if(td>50||d->qpos[2]<0.2) falls++;
    if(step%500==0) std::printf("[hl] s=%d t=%.2f z=%.3f x=%+.3f y=%+.3f tilt=%.1f Veff=%.2f falls=%d\n",
                                step,t,d->qpos[2],d->qpos[0],d->qpos[1],td,Veff,falls);
  }
  double wall=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tstart).count();
  std::printf("\n=== 종료: STEPS=%d(%.1fs) x=%+.3f y=%+.3f z=%.3f max_tilt=%.1f° falls=%d | wall=%.2fs(%.0f steps/s) ===\n",
              STEPS,STEPS*dt,d->qpos[0],d->qpos[1],d->qpos[2],max_tilt,falls,wall,STEPS/wall);
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
