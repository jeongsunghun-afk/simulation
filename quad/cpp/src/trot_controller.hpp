// TrotCtrl — mode_trot 핵심경로 1틱 제어(설정/스윙/MPC/WBIC → d->ctrl). trot_sim(헤드리스)·trot_view(뷰어) 공유.
#pragma once
#include "quad_control.hpp"
#include <vector>
#include <array>
#include <map>
#include <cmath>

// gait 상수(config + trot 프리셋)
static const double GP_T=0.50, GP_SWF=0.50;
static const double GP_OFFSET[4]={0.0,0.5,0.5,0.0};   // HL,HR,FL,FR
static const double TC_SW=0.25, TC_ST=0.25, TC_SDELTA=0.005;
static const double TC_WARMUP=0.6, TC_SETTLE=0.5, TC_ACC=0.6;
static const double TC_KCAP=0.16, TC_RAIBERT=0.8, TC_RAICLIP=0.25;

static inline void tc_gait(int i,double tg,bool&stance,double&sprog){
  double ph=std::fmod(tg/GP_T+GP_OFFSET[i],1.0); if(ph<0) ph+=1.0;
  if(ph>=GP_SWF){ stance=true; sprog=0.0; } else { stance=false; sprog=ph/GP_SWF; }
}
static inline void tc_swing_z(double sh,double Th,double Vz,double&c2,double&c4,double&c6){
  Matrix3d A; A<< Th*Th, std::pow(Th,4), std::pow(Th,6),
                  2.0, 12.0*Th*Th, 30.0*std::pow(Th,4),
                  2.0*Th, 4.0*std::pow(Th,3), 6.0*std::pow(Th,5);
  Vector3d b(-sh,0.0,-Vz); Vector3d c=A.colPivHouseholderQr().solve(b); c2=c[0];c4=c[1];c6=c[2];
}
static inline Vector3d tc_swing_foot(double sw_t,const Vector3d&p0,const Vector3d&pe,const Vector3d&bvel,double sh,double Tl,double Tst){
  if(sw_t>=1.0) return pe;
  double tau=sw_t, s5=10*std::pow(tau,3)-15*std::pow(tau,4)+6*std::pow(tau,5);
  double DXx=(pe[0]-p0[0])+bvel[0]*Tl; Vector3d pos;
  pos[0]=p0[0]-bvel[0]*tau*Tl+DXx*s5; pos[1]=(1.0-s5)*p0[1]+s5*pe[1];
  double Th=Tl/2.0, u=tau*Tl-Th, Vz=TC_SDELTA*M_PI/Tst; double c2,c4,c6; tc_swing_z(sh,Th,Vz,c2,c4,c6);
  pos[2]=p0[2]+sh+c2*u*u+c4*std::pow(u,4)+c6*std::pow(u,6); return pos;
}
static inline double tc_clip(double v,double lo,double hi){ return v<lo?lo:(v>hi?hi:v); }

struct TrotCtrl {
  QuadControl& q;
  double V=0.30, VY=0.0, WZ=0.0;   // 명령속도(뷰어 키보드/GUI)
  double step_h=0.10, raibert_k=0.8;   // ★GUI 슬라이더(live): step height·전방 reach
  bool ALIP=true, POS_HOLD=true;
  // ── 모드관리(배포용): move/stand_up(서기)/stand_down(눕기)/off ──
  std::string mode="move";
  double body_h=0.5234, ht_cur=0.5234, qhome_h=0.5234;   // 서기높이 슬라이더·보간높이·q_home 계산높이
  VectorXd q_ref; bool have_qref=false;                  // fold 관절목표 slew
  // 모드관리 상수(Python 17dof와 동일)
  double GROUND_Z=0.18, GETUP_TRIG=0.32, GETUP_DONE=0.40, GETUP_KP=90, GETUP_KD=3, GETUP_RATE=0.18, REST_KD=3.0, JOINT_SLEW=1.5, HRATE=0.3;
  // ── 게이트 프리셋(trot/walk/gallop) ──
  std::string gait_type="trot";
  double gp_T=0.5, gp_SWF=0.5, gp_off[4]={0,0.5,0.5,0}, gp_Tsw=0.25, gp_Tst=0.25;
  // ── ★속도 트리거 자동 whip(고속 trot=동물형 채찍질) ──
  bool auto_whip=false; double whip_v0=0.8, whip_v1=1.6;   // v0~v1 구간서 whip 증가(swing_w 2.0→낮게)
  double whip_lo_f=0.15, whip_lo_r=0.8;                    // 고속 swing_w: 앞=강whip(paw-tuck)·뒤=완만(안정)
  // 상태
  bool armed=false; double t0=0, settle_until=TC_SETTLE;
  double Vs=0,Vys=0,Ws=0, yaw_ref=0; bool yaw_hold_set=false; double yaw_hold=0;
  bool pos_hold_set=false; double phx=0,phy=0;
  VectorXd x_ref=VectorXd::Zero(13);
  std::array<Vector3d,4> liftoff, nominal; std::array<Vector2d,4> hip_off; std::array<double,4> gz;
  std::array<bool,4> have_prev={false,false,false,false}; std::array<Vector3d,4> ptgt_prev;
  Vector3d lam_des[4]={Vector3d::Zero(),Vector3d::Zero(),Vector3d::Zero(),Vector3d::Zero()};
  double mpc_t=-1.0, Veff_dbg=0;

  TrotCtrl(QuadControl& q_):q(q_){ q_ref=VectorXd::Zero(q.nu); body_h=ht_cur=qhome_h=q.base_z0; }

  void set_gait(const std::string& g){        // trot/walk/gallop 프리셋(GUI 토글·속도트리거)
    if(g==gait_type) return; gait_type=g;
    if(g=="walk"){ gp_T=1.0; gp_SWF=0.25; gp_off[0]=0.25; gp_off[1]=0.75; gp_off[2]=0.5; gp_off[3]=0.0; }
    else if(g=="gallop"){ gp_T=0.35; gp_SWF=0.55; gp_off[0]=0.0; gp_off[1]=0.05; gp_off[2]=0.55; gp_off[3]=0.5; } // 회전형 갤럽(비행상 有)
    else         { gp_T=0.5; gp_SWF=0.5;  gp_off[0]=0.0;  gp_off[1]=0.5;  gp_off[2]=0.5; gp_off[3]=0.0; }
    gp_Tsw=gp_T*gp_SWF; gp_Tst=gp_T*(1.0-gp_SWF); armed=false;   // 재arm=위상 재앵커(불연속 방지)
  }
  void gait(int i,double tg,bool&stance,double&sprog){
    double ph=std::fmod(tg/gp_T+gp_off[i],1.0); if(ph<0) ph+=1.0;
    if(ph>=gp_SWF){ stance=true; sprog=0.0; } else { stance=false; sprog=ph/gp_SWF; }
  }
  void reset(){   // ★시뮬 리셋(RESET 버튼): 컨트롤러 상태 초기화(crouch_home 후 호출)
    armed=false; settle_until=q.d->time+TC_SETTLE; have_qref=false;
    Vs=Vys=Ws=0; yaw_hold_set=false; mpc_t=-1.0;
    ht_cur=qhome_h=body_h=q.base_z0; for(int i=0;i<4;i++) have_prev[i]=false;
  }

  // 1틱 제어: d->ctrl 설정(mj_step은 호출자). q.d->time 기준.
  void control(){
    mjModel*m=q.m; mjData*d=q.d; int nv=q.nv; double dt=m->opt.timestep;
    double t=d->time; int nu=q.nu;
    // ── 모드 dispatch(배포용): move 외 = 서기/눕기/getup/off ──
    if(mode!="move"){
      if(mode=="off"){ for(int j=0;j<nu;j++) d->ctrl[j]=tc_clip(-REST_KD*d->qvel[6+j],-q.tau_peak[j],q.tau_peak[j]); armed=false; have_qref=false; return; }
      double bz=d->qpos[2];
      if(bz<GETUP_TRIG && ht_cur>GETUP_DONE) ht_cur=std::max(0.12,bz);      // 쓰러짐/off로 낮음→동기화
      double tgt=(mode=="stand_down")?GROUND_Z:body_h;                      // 눕기=낮게 / 서기=슬라이더
      bool low=(ht_cur<GETUP_DONE)||(tgt<GETUP_DONE); double rate=low?GETUP_RATE:HRATE;
      ht_cur+=tc_clip(tgt-ht_cur,-rate*dt,rate*dt);
      if(std::abs(ht_cur-qhome_h)>6e-3){ q.update_stand_qhome(ht_cur); qhome_h=ht_cur; }
      double jerr=0; for(int j=0;j<nu;j++) jerr+=std::abs(q.q_home[j]-d->qpos[7+j]); jerr/=nu;
      if(mode=="stand_down" && std::abs(ht_cur-GROUND_Z)<=0.02 && jerr<0.3){ // 눕기완료→damp(모터off 등가)
        for(int j=0;j<nu;j++) d->ctrl[j]=tc_clip(-REST_KD*d->qvel[6+j],-q.tau_peak[j],q.tau_peak[j]); armed=false; have_qref=false; return; }
      if(ht_cur<GETUP_DONE){                                                // 낮은자세=수평 PD fold(눕기/getup 공통)
        if(!have_qref){ for(int j=0;j<nu;j++) q_ref[j]=d->qpos[7+j]; have_qref=true; }
        for(int j=0;j<nu;j++) q_ref[j]+=tc_clip(q.q_home[j]-q_ref[j],-JOINT_SLEW*dt,JOINT_SLEW*dt);
        for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+GETUP_KP*(q_ref[j]-d->qpos[7+j])-GETUP_KD*d->qvel[6+j];
          d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
        armed=false; return; }
      have_qref=false; q.wbic_stance(); armed=false; return;                // 서기(높이충분)=wbic_stance
    }
    have_qref=false;   // move → fold 리셋
    auto quat_yaw=[&](){ double*qq=&d->qpos[3]; return std::atan2(2*(qq[0]*qq[3]+qq[1]*qq[2]),1-2*(qq[2]*qq[2]+qq[3]*qq[3])); };
    if(t < settle_until){ q.wbic_stance(); return; }
    if(!armed){ armed=true; t0=t; yaw_ref=0;
      for(int i=0;i<4;i++){ nominal[i]=q.foot_point(i); liftoff[i]=q.foot_point(i); hip_off[i]=q.foot_hip_off[i]; gz[i]=q.foot_gz0[i]; }
      x_ref.setZero(); x_ref[5]=d->subtree_com[2]; x_ref[12]=-9.81; }
    double tg=t-t0; bool go=tg>TC_WARMUP;
    double vt=go?V:0.0, vyt=go?VY:0.0, wt=go?WZ:0.0;
    Vs+=tc_clip(vt-Vs,-TC_ACC*dt,TC_ACC*dt); Vys+=tc_clip(vyt-Vys,-TC_ACC*dt,TC_ACC*dt); Ws+=tc_clip(wt-Ws,-2.0*dt,2.0*dt);
    double Veff=Vs,Vyeff=Vys,Weff=Ws; Veff_dbg=Veff;
    double spd=std::hypot(Veff,Vyeff);
    if(auto_whip){   // ★속도↑ → whip↑(swing_w 낮춤). 앞다리 강하게(paw-tuck)·뒷다리 완만(안정)
      double s=tc_clip((spd-whip_v0)/(whip_v1-whip_v0),0.0,1.0);
      q.swing_w_f=2.0+s*(whip_lo_f-2.0); q.swing_w_r=2.0+s*(whip_lo_r-2.0); }
    double yaw_m=quat_yaw();
    if(std::abs(Weff)>0.02){ yaw_ref=tc_clip(yaw_ref+Weff*dt,yaw_m-0.3,yaw_m+0.3); yaw_hold_set=false; }
    else { if(!yaw_hold_set){ yaw_hold=yaw_m; yaw_hold_set=true; } yaw_ref=yaw_hold; }
    double cy=std::cos(yaw_m), sy=std::sin(yaw_m);
    double vx_w=Veff*cy-Vyeff*sy, vy_w=Veff*sy+Vyeff*cy;
    if(POS_HOLD && std::abs(Veff)<0.03 && std::abs(Vyeff)<0.03 && std::abs(Weff)<0.05){
      if(!pos_hold_set){ phx=d->qpos[0]; phy=d->qpos[1]; pos_hold_set=true; }
      vx_w+=tc_clip(-0.6*(d->qpos[0]-phx),-0.15,0.15); vy_w+=tc_clip(-0.6*(d->qpos[1]-phy),-0.15,0.15);
    } else pos_hold_set=false;
    x_ref[2]=yaw_ref; x_ref[8]=Weff; x_ref[9]=vx_w; x_ref[10]=vy_w; q._body_terr=0.0;
    std::vector<int> st; std::map<int,std::pair<Vector3d,Vector3d>> swing;
    for(int i=0;i<4;i++){ bool sch; double sp; gait(i,tg,sch,sp);
      if(sch){ st.push_back(i); have_prev[i]=false; } else { if(sp<0.03) liftoff[i]=q.foot_point(i); } }
    std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
    Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
    Map<VectorXd> qv(d->qvel,nv); Vector3d vcom=Jc*qv;
    Vector2d v_des(vx_w,vy_w), v_fb=vcom.head(2);
    if(ALIP){ mj_subtreeVel(m,d); Vector3d L(d->subtree_angmom[0],d->subtree_angmom[1],d->subtree_angmom[2]);
      double H=std::max(0.1,d->subtree_com[2]); v_fb+=Vector2d(L[1],-L[0])/(q.mpc.TOTAL_MASS*H); }
    Vector2d rai; for(int k=0;k<2;k++) rai[k]=tc_clip(raibert_k*gp_Tst*v_des[k]+TC_KCAP*(v_fb[k]-v_des[k]),-TC_RAICLIP,TC_RAICLIP);
    Matrix2d Rw; Rw<<cy,-sy,sy,cy; double sh=step_h*(0.2+0.8*std::min(1.0,tg/TC_WARMUP));
    for(int i=0;i<4;i++){ bool sch; double s_; gait(i,tg,sch,s_); if(sch) continue;
      Vector2d hip_xy(d->xpos[q.hip_bid[i]*3],d->xpos[q.hip_bid[i]*3+1]);
      Vector2d r_xy=hip_xy-Vector2d(d->qpos[0],d->qpos[1]);        // 몸중심→hip
      Vector2d tw=Weff*gp_Tst*Vector2d(-r_xy[1],r_xy[0]);          // ★선회 접선 발배치(yaw) — 없으면 회전시 표류·붕괴
      Vector2d pe_xy=hip_xy+Rw*hip_off[i]+rai+tw; Vector3d p_end(pe_xy[0],pe_xy[1],gz[i]);
      double dzl=p_end[2]-liftoff[i][2]; Vector3d bvel(vcom[0],vcom[1],0.0);
      Vector3d p_tgt=tc_swing_foot(s_,liftoff[i],p_end,bvel,sh,gp_Tsw,gp_Tst);
      p_tgt[2]+=dzl*(10*std::pow(s_,3)-15*std::pow(s_,4)+6*std::pow(s_,5));
      Vector3d v_tgt=Vector3d::Zero();
      if(have_prev[i]) for(int c=0;c<3;c++) v_tgt[c]=tc_clip((p_tgt[c]-ptgt_prev[i][c])/dt,-1.0,1.0);
      ptgt_prev[i]=p_tgt; have_prev[i]=true; swing[i]={p_tgt,v_tgt}; }
    double dmpc=t-mpc_t;
    if(!st.empty() && (mpc_t<0||dmpc<0||dmpc>=q.mpc.DT)){
      std::vector<std::array<int,4>> cs(q.mpc.N);
      for(int k=0;k<q.mpc.N;k++) for(int i=0;i<4;i++){ bool sch; double sp; gait(i,tg+k*q.mpc.DT,sch,sp); cs[k][i]=sch?1:0; }
      Matrix<double,4,3> L=q.mpc_grf(x_ref,cs); for(int i=0;i<4;i++) lam_des[i]=L.row(i).transpose(); mpc_t=t; }
    Vector3d lam_use[4]; for(int i=0;i<4;i++) lam_use[i]= st.empty()?Vector3d::Zero():lam_des[i];
    if(!q.wbic_track(st,swing,lam_use)) q.wbic_stance();
  }
  double tiltdeg(){ double R[9]; mju_quat2Mat(R,&q.d->qpos[3]); return std::acos(tc_clip(R[8],-1,1))*180/M_PI; }
};

// 17dof 튜닝 게인 적용(env 우선). 14dof는 기본값 유지.
static inline void apply_env_gains(QuadControl& q){
  if(getenv("BASE_Z0")) q.base_z0=atof(getenv("BASE_Z0"));
  if(getenv("REAR_ANKLE")){ q.REAR_ANKLE=atof(getenv("REAR_ANKLE")); q.FRONT_ANKLE=q.REAR_ANKLE; }
  if(getenv("FRONT_ANKLE")) q.FRONT_ANKLE=atof(getenv("FRONT_ANKLE"));
  if(getenv("W_AM")) q.W_AM=atof(getenv("W_AM"));
  if(getenv("KD_AM")) q.KD_AM=atof(getenv("KD_AM"));
  if(getenv("W_ORI")) q.w_ori=atof(getenv("W_ORI"));
  if(getenv("SWING_W")){ double v=atof(getenv("SWING_W")); q.swing_w_r=v; q.swing_w_f=v; }
  if(getenv("SWING_W_R")) q.swing_w_r=atof(getenv("SWING_W_R"));
  if(getenv("SWING_W_F")) q.swing_w_f=atof(getenv("SWING_W_F"));
  if(getenv("PIN_ANKLE")) q.stance_pin_ankle=true;
}
