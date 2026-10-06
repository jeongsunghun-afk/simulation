// TrotCtrl — mode_trot 핵심경로 1틱 제어(설정/스윙/MPC/WBIC → d->ctrl). trot_sim(헤드리스)·trot_view(뷰어) 공유.
#pragma once
#include "quad_control.hpp"
#include "terrain_map.hpp"   // ★P1: footScore nudge용 로컬 elevation map
#include "../tamols/tamols_online.hpp"   // ★receding-horizon 온라인 TAMOLS 플래너(RSL_ONLINE)
#include <vector>
#include <array>
#include <map>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdio>
#ifdef HAVE_JUMP_SOLVER
#include <future>
#include "jump_solver.hpp"   // ★S3-b Step2: 점프 crouch중 live-solve(trot_view만, crocoddyl 링크 시)
#endif

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
  pos[2]=(1.0-s5)*p0[2]+s5*pe[2]+sh+c2*u*u+c4*std::pow(u,4)+c6*std::pow(u,6); return pos;   // ★z=liftoff→착지목표 보간+lift arc(공중발 영구화·지형착지 오차 수정). 평지 p0=pe=0=무변화
}
static inline double tc_clip(double v,double lo,double hi){ return v<lo?lo:(v>hi?hi:v); }

struct TrotCtrl {
  QuadControl& q;
  double V=0.30, VY=0.0, WZ=0.0;   // 명령속도(뷰어 키보드/GUI)
  double step_h=0.10, raibert_k=0.5;   // ★GUI 슬라이더(live): step height·전방 reach. ★0.5=표준 Raibert 중립점(발을 앞으로 과하게 안던짐→GRF 앞/뒤 균형·뒤thigh 절반). 외란복구는 KCAP+MPC 담당
  double gait_base_z=0.50;   // ★gait별 최적 base height(축별 worst-util 스윕): walk/trot 0.50·run 0.48. 보행 중 추종(set_gait서 설정)
  bool ALIP=false, POS_HOLD=true;   // ★ALIP 기본 off: push복구 이득 미미(300N서만 22vs26°)한데 지속선회 붕괴시킴. ALIP=1로 켬
  bool SPIN_HOLD=false;             // ★제자리선회(V=0,WZ≠0)서도 위치홀드 유지 → 허리조향 표류 상쇄(베이스기준 wz선회). 주행선회엔 영향無
  bool perceptive=true;             // ★perceptive: 스윙 착지 XY 아래 지형높이를 mj_ray로 샘플→착지 z를 지형에 맞춤(계단/험지). off=blind(평지 가정). PERCEPTIVE=0으로 끔
  double PCV_CLR=0.04;              // ★perceptive 상향 스텝 시 추가 스윙 클리어런스(up-step 높이×비율만큼 apex↑, 라이저 헛디딤 방지)
  double com_h0=0.52;               // ★평지 위 CoM 명목높이(arming서 캡처). perceptive 몸통높이 목표=지형높이+com_h0
  // ★P1 footScore nudge: Raibert 발타깃(pe_xy)을 반경내 최고 footScore 셀로 당김. 스윙시작 1회 계산·홀드(채터 방지). env FOOT_NUDGE/GUI.
  MjRayTerrainMap tmap; bool foot_nudge=getenv("FOOT_NUDGE")!=nullptr;
  double nudge_r=getenv("NUDGE_R")?atof(getenv("NUDGE_R")):0.18;    // ★③ 도달반경(P1 0.12→0.18): 갭 너머 돌까지 닿게
  double nudge_wbal=getenv("W_BAL")?atof(getenv("W_BAL")):0.6;      // ★밸런스 가중(footScore vs Raibert타깃 거리 저울질)
  Vector2d nudge_off[4]={Vector2d::Zero(),Vector2d::Zero(),Vector2d::Zero(),Vector2d::Zero()};   // ★오프셋(절대타깃 동결 아님)=Raibert 연속 밸런스 피드백 보존
  double _bterr_s=0.0;              // ★슬루된 지형높이(4hip평균을 부드럽게) → MPC x_ref[5]·WBIC z-task 양쪽 일관 공급
  double _tambz_s=0.5;             // ★TAM_BASE: 슬루된 TAMOLS 계획 base z(주입, 급변 방지)
  // ── 모드관리(배포용): move/stand_up(서기)/stand_down(눕기)/off ──
  std::string mode="move";
  double body_h=0.5234, ht_cur=0.5234, qhome_h=0.5234;   // 서기높이 슬라이더·보간높이·q_home 계산높이
  VectorXd q_ref; bool have_qref=false;                  // fold 관절목표 slew
  double SIT_Z=0.32, SIT_PITCH=0.70, SIT_REAR_FOOT=-1.35, SIT_REAR_CALF=1.10, SIT_REAR_THIGH=-0.6; bool have_qsit=false; // ★앉기=crouch-sit(SIT_Z 저crouch, 4발 planted→기립가능). SIT_PITCH등은 구 haunch-sit 잔여
  double SIT_CPITCH=1.0, SIT_REACH=0.08;   // ★앉기 nose-up 목표(~25° 앞올림=앉은자세, wbic_stance 능동제어→안정+기립가능). SIT_REACH 미사용(잔여)
  // ★개-앉기(haunch sit): 정착 후 crouch→haunch(뒷다리 접어 발링크 바닥밀착)로 fold, 기립 시 높이스케줄 언폴드. q_home/com_ref 블렌드.
  double HAUNCH_Z=0.30, HAUNCH_FOLD_RATE=0.35, HAUNCH_UNFOLD_Z=0.40, SIT_POSTURE_W=40.0, HAUNCH_PITCH=0.50, SIT_KP=90.0;   // ★엉덩이 주저앉기: base 낮춤·fold 완만(0.60→0.35=~2.9s, 서기→앉기 급강하 완화)·nose-up 크게. 기립불요·자세우선
  double SIT_BELOW_SPEED=2.5;   // ★눕기(from_below)→앉기 fold 배수: 엉덩이 이미 낮아 엉덩방아 위험 없음→빠르게(위→앉기 완만착지는 미적용). ~2.9s→~1.2s
  double FRONT_PULL_SPEED=2.2;  // ★앞다리 끌어당김 배수(앞다리는 뒤 엉덩이 착지 tail과 무관하게 더 빨리 접힘): 앞은 haunch_fold*이 배수로 조기 완료, 뒤(엉덩방아)는 원래 pace 유지
  VectorXd q_crouch, q_haunch; Vector3d com_crouch=Vector3d::Zero(), com_haunch=Vector3d::Zero(); double haunch_fold=0; bool haunch_ready=false;
  // ★기립 궤적 추종(offline gather 궤적 /tmp/getup_traj.txt): sit→gather(CoM 전진)→일어서기. phaseA(0/1)=PD추종, phaseB(2)=wbic 상승 인계.
  std::vector<VectorXd> getup_q, getup_dqv; std::vector<int> getup_ph; double getup_dt=0.01; int getup_N=-1, getup_k=-1; double getup_kt=0, GETUP_TRAJ_KP=80.0, GETUP_TRAJ_KD=6.0;   // ★앉기→서기 튕김 힘 완화(120→80)+감쇠↑(4→6): 기립 중 최대tilt 70°→49°(더 약하고 안전하게). 기립 완료는 유지(z≈0.49)
  void load_getup(const char* path){
    std::ifstream f(path); if(!f){ getup_N=0; return; }
    f>>getup_N>>getup_dt; getup_q.clear(); getup_dqv.clear(); getup_ph.clear();
    for(int k=0;k<getup_N;k++){ int ph; f>>ph; getup_ph.push_back(ph);
      VectorXd qv(q.nu),dv(q.nu); for(int j=0;j<q.nu;j++) f>>qv[j]; for(int j=0;j<q.nu;j++) f>>dv[j];
      getup_q.push_back(qv); getup_dqv.push_back(dv); }
    std::printf("[getup] 궤적 로드 N=%d dt=%.3f\n", getup_N, getup_dt);
  }
  std::vector<Vector3d> getup_com, getup_comv, getup_acom;   // ★WBIC-추종용 CoM(기립도 점프처럼 폐루프 추적)
  mjData* gu_d=nullptr;
  static void gu_slerp(const double a[4], const double b[4], double t, double out[4]){  // wxyz
    double bb[4]={b[0],b[1],b[2],b[3]}, dot=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3];
    if(dot<0){ for(int i=0;i<4;i++) bb[i]=-bb[i]; dot=-dot; }
    if(dot>0.9995){ for(int i=0;i<4;i++) out[i]=a[i]+t*(bb[i]-a[i]); }
    else{ double th=std::acos(std::max(-1.0,std::min(1.0,dot))), s=std::sin(th), w0=std::sin((1-t)*th)/s, w1=std::sin(t*th)/s;
      for(int i=0;i<4;i++) out[i]=w0*a[i]+w1*bb[i]; }
    double n=std::sqrt(out[0]*out[0]+out[1]*out[1]+out[2]*out[2]+out[3]*out[3]); for(int i=0;i<4;i++) out[i]/=n;
  }
  // ★[DEAD CODE·미호출] 구 gather 궤적 getup. 앉기→서기가 측방 발산이라 앉기→눕기→서기 라우팅으로 교체(e3461f6). 동적 gather 재도입 시 참조용 보존.
  // ★C++ 자립 기립 궤적 생성(getup_kinematic.py 포팅, 순수 MuJoCo IK). 현재 sit qpos=q_sit, base_z0+q_home=stand.
  //   스케줄 G(gather:전진+낮춤+앞숙임)→A1(HL착지)→A2(HR착지)→B(상승·레벨). CoM ref도 생성(wbic_jump 추적용).
  void gen_getup(){
    mjModel* m=q.m; if(!gu_d) gu_d=mj_makeData(m);
    int nqm=m->nq, num=q.nu;
    auto fpos=[&](int L)->Vector3d{ int gg=q.fgid[L]; return Vector3d(gu_d->geom_xpos[3*gg],gu_d->geom_xpos[3*gg+1],gu_d->geom_xpos[3*gg+2]-q.fr[L]); };
    auto ikfeet=[&](Vector3d* tg){
      for(int it=0;it<200;it++){ mj_kinematics(m,gu_d);
        for(int L=0;L<4;L++){ Vector3d e=tg[L]-fpos(L); if(e.norm()<1e-4) continue;
          int gg=q.fgid[L]; std::vector<double> jb(3*m->nv); mjtNum p[3]={gu_d->geom_xpos[3*gg],gu_d->geom_xpos[3*gg+1],gu_d->geom_xpos[3*gg+2]};
          mj_jac(m,gu_d,jb.data(),nullptr,p,m->geom_bodyid[gg]);
          Matrix3d Jl; for(int r=0;r<3;r++)for(int c=0;c<3;c++) Jl(r,c)=jb[r*m->nv+q.legqv[L][c]];
          Vector3d dqi=0.5*Jl.transpose()*(Jl*Jl.transpose()+1e-4*Matrix3d::Identity()).ldlt().solve(e);
          for(int c=0;c<3;c++) gu_d->qpos[q.legqp[L][c]]+=dqi[c]; } }
    };
    VectorXd qsf(nqm); for(int i=0;i<nqm;i++) qsf[i]=q.d->qpos[i];   // 현재 sit qpos
    // ★서기 관절 config(getup 시점 q.q_home은 sit이라 별도 계산): update_stand_qhome(LUT), 후 복원
    VectorXd qhome_save=q.q_home; Vector3d comref_save=q.com_ref;
    q.update_stand_qhome(q.base_z0); VectorXd qstand_j=q.q_home;
    q.q_home=qhome_save; q.com_ref=comref_save;
    for(int i=0;i<nqm;i++) gu_d->qpos[i]=qsf[i];
    gu_d->qpos[2]=q.base_z0; gu_d->qpos[3]=1; gu_d->qpos[4]=0; gu_d->qpos[5]=0; gu_d->qpos[6]=0;
    for(int j=0;j<num;j++) gu_d->qpos[7+j]=qstand_j[j];
    mj_forward(m,gu_d);
    Vector3d foot_stand[4]; for(int L=0;L<4;L++) foot_stand[L]=fpos(L);
    double base_stand_x=gu_d->qpos[0];
    for(int i=0;i<nqm;i++) gu_d->qpos[i]=qsf[i]; mj_forward(m,gu_d);
    Vector3d foot_sit[4]; for(int L=0;L<4;L++) foot_sit[L]=fpos(L);
    Vector3d base_sit(gu_d->qpos[0],gu_d->qpos[1],gu_d->qpos[2]);
    double ank_sit[4],ank_st[4]; for(int L=0;L<4;L++){ ank_sit[L]=qsf[q.legqp[L][3]]; ank_st[L]=qstand_j[q.legqp[L][3]-7]; }
    int NG=45,NA1=35,NA2=35,NB=90,N=NG+NA1+NA2+NB; getup_dt=0.01;
    double GX=base_stand_x, GZ=0.20, LEAN=0.10;
    double qsit_q[4]={qsf[3],qsf[4],qsf[5],qsf[6]}, q_lean[4]={std::cos(LEAN/2),0,std::sin(LEAN/2),0}, q_id[4]={1,0,0,0};
    auto lp=[&](double a,double b,double t){ return a+(b-a)*std::max(0.0,std::min(1.0,t)); };
    auto bprof=[&](int k,double& bx,double& bz,double* oq){
      if(k<NG){ double t=(k+1.0)/NG; bx=lp(base_sit[0],GX,t); bz=lp(base_sit[2],GZ,t); gu_slerp(qsit_q,q_lean,t,oq); }
      else if(k<NG+NA1+NA2){ bx=GX; bz=GZ; for(int i=0;i<4;i++) oq[i]=q_lean[i]; }
      else{ double t=(k-NG-NA1-NA2)/(double)NB; bx=lp(GX,base_stand_x,t); bz=lp(GZ,q.base_z0,t); gu_slerp(q_lean,q_id,t,oq); } };
    std::vector<VectorXd> tq;
    getup_ph.clear();
    for(int k=0;k<N;k++){
      int ph=(k<NG+NA1)?0:((k<NG+NA1+NA2)?1:2); double bx,bz,oq[4]; bprof(k,bx,bz,oq);
      gu_d->qpos[0]=bx; gu_d->qpos[1]=0; gu_d->qpos[2]=bz; for(int i=0;i<4;i++) gu_d->qpos[3+i]=oq[i];
      Vector3d tg[4];
      for(int L=0;L<4;L++){
        if(L>=2) tg[L]=foot_stand[L];
        else if(L==0){ double s=(k<NG)?0.0:((k<NG+NA1)?std::min(1.0,(k-NG)/(double)NA1):1.0); tg[L]=foot_sit[L]+(foot_stand[L]-foot_sit[L])*s; }
        else { double s=(k<NG+NA1)?0.0:((k<NG+NA1+NA2)?std::min(1.0,(k-NG-NA1)/(double)NA2):1.0); tg[L]=foot_sit[L]+(foot_stand[L]-foot_sit[L])*s; } }
      for(int L=0;L<4;L++) gu_d->qpos[q.legqp[L][3]]=lp(ank_sit[L],ank_st[L],(k+1.0)/N);
      ikfeet(tg);
      VectorXd qj(num); for(int j=0;j<num;j++) qj[j]=gu_d->qpos[7+j]; tq.push_back(qj); getup_ph.push_back(ph);
    }
    getup_q=tq; getup_dqv.assign(N,VectorXd::Zero(num));
    for(int k=0;k<N;k++){ int kp=std::min(k+1,N-1),km=std::max(k-1,0); double hh=(kp-km)*getup_dt; if(hh>0) getup_dqv[k]=(tq[kp]-tq[km])/hh; }
    getup_com.assign(N,Vector3d::Zero());
    for(int k=0;k<N;k++){ double bx,bz,oq[4]; bprof(k,bx,bz,oq);
      gu_d->qpos[0]=bx; gu_d->qpos[1]=0; gu_d->qpos[2]=bz; for(int i=0;i<4;i++) gu_d->qpos[3+i]=oq[i];
      for(int j=0;j<num;j++) gu_d->qpos[7+j]=tq[k][j]; for(int i=0;i<m->nv;i++) gu_d->qvel[i]=0; mj_forward(m,gu_d);
      getup_com[k]=Vector3d(gu_d->subtree_com[0],gu_d->subtree_com[1],gu_d->subtree_com[2]); }
    getup_comv.assign(N,Vector3d::Zero()); getup_acom.assign(N,Vector3d::Zero());
    for(int k=0;k<N;k++){ int kp=std::min(k+1,N-1),km=std::max(k-1,0); double hh=(kp-km)*getup_dt; if(hh>0) getup_comv[k]=(getup_com[kp]-getup_com[km])/hh; }
    for(int k=0;k<N;k++){ int kp=std::min(k+1,N-1),km=std::max(k-1,0); double hh=(kp-km)*getup_dt; if(hh>0) getup_acom[k]=(getup_comv[kp]-getup_comv[km])/hh; }
    getup_N=N;
    std::printf("[getup] C++ 생성 N=%d (sit z=%.3f→stand z=%.3f, CoM x %.3f→%.3f)\n", N, base_sit[2], q.base_z0, getup_com[0][0], getup_com.back()[0]);
  }
  // ★앉기→서기 스크립트 기립(앞다리 굽혀 앞발 들어 폴볼트 차단 + 뒷다리 박차 extend). 앉기에서만 발동.
  bool was_sit=false; double sit_getup_t0=-1;
  bool sit_init=false, sit_from_below=false;   // ★sit 진입 방향 1회 latch: 눕기 등 아래서 진입 시 0.32 crouch 오버슈트 없이 현재 자세서 곧바로 haunch로 morph
  bool from_sit=false;   // ★crouch-sit서 기립: 저crouch(≥0.29)라 저-PD 대신 wbic_stance로 매끈 기립(오버슈트 방지)
  double lie_settle=0;   // ★강건 기립: 앉기→눕기 언폴드 후 저크라우치 정착 대기(dwell) 타이머
  // ★★TAMOLS 재생(perceptive foothold 플래너 추종): base pose/vel 궤적 + 발판 + 접촉스케줄을 로드해
  //   기존 보행스택(MPC→lam→wbic_track)의 참조를 TAMOLS로 교체(속도/z/yaw + 발판 스윙 + 접촉).
  std::vector<std::array<double,12>> tam_s;   // 샘플: [t·pose6(x y z r p yw)·vel6]
  std::vector<std::array<int,4>> tam_c;       // 샘플별 접촉(4)
  Vector3d tam_fh[4];                          // 발판(world, 로봇중심 프레임)
  double tam_dt=0.005, tam_t=-1, tam_t0=0, tam_ax=0, tam_ay=0; int tam_N=0; bool tam_loaded=false;
  double tam_vbx=0, tam_vby=0; bool tam_vbinit=false;   // ★VADV_REF: MPC식 계획 위치참조(측정 re-anchor 대신 V로 전진+leaky 이완 → 위치복원력=속도폭주 방지)
  Vector3d tam_lift[4]; bool tam_sw_prev[4]={false,false,false,false}; double tam_sw_start[4]={0,0,0,0};
  Vector3d tam_swtgt[4];   // ★online swing foot commitment: liftoff 시 착지target 동결(재anchor에도 불변, 표준 receding-horizon)
  tamols::TamolsState tam_ol; bool tam_ol_warm=false; double tam_ol_last=-1; int tam_ol_phase=0; bool tam_inj=false;   // ★온라인 replan 상태(warm-start·위상·주입모드=plan정지 발판만)
  double tam_ent=-1e9;   // ★tamols 진입 절대시각(cold-start settle 계측용, tam_t는 재앵커로 리셋되므로 별도)
  double comy_lp=0;      // ★횡 CoM 저주파(드리프트) 상태: 빠른 sway 제거하고 느린 드리프트만 발배치로 보정
  tamols::Grid tam_ol_h; double tam_ol_cell=0.05; int tam_ol_ms=41; bool tam_ol_mapinit=false;
  // 현재 MuJoCo 상태 → online_replan → tam_s 리필(receding-horizon). RSL_ONLINE=1.
  double tam_z0=0.52;   // ★online_replan에 넘길 base z 기준(지형모드=지형+명목높이)
  bool tam_prm_done=false;   // ★prm(로봇 파라미터) 모델주입 1회 완료 플래그
  // ★TAMOLS 플래너 파라미터를 현재 모델에서 채움(로봇 무관 이식용, TAM_PRM_MODEL=1).
  //   기본 prm은 02_Leg 하드코딩(mass37.9·height0.52·hip_offsets±0.225/±0.14·[FL,FR,RL,RR]순)이라
  //   다른 로봇(Go2) 이식·공정비교 불가 + hip_offsets가 발측정(q.legs=[HL,HR,FL,FR]순)과 leg-order 불일치.
  //   여기서 mass·높이·발반경·관성·hip_offsets(실제 hip위치, q.legs순)를 모델서 계산→양쪽 로봇 동일 정합.
  void tamols_init_prm(){
    mjModel* m=q.m; auto& P=tam_ol.prm;
    P.mass=m->body_subtreemass[0]; P.foot_radius=q.fr[0]; P.g=9.81;
    P.nominal_height=q.base_z0; P.h_des=q.base_z0;
    P.l_max=q.base_z0*1.5; P.l_min=q.base_z0*0.35;                    // 다리 reach(base_z0 비례 근사)
    Matrix3d Ic=q.compute_Icom(); P.inertia_diag=Vector3d(Ic(0,0),Ic(1,1),Ic(2,2));
    Vector3d base(q.d->qpos[0],q.d->qpos[1],q.d->qpos[2]);           // hip_offsets=hip xy(base 상대), q.legs순
    double _swx=getenv("TAM_SWX")?atof(getenv("TAM_SWX")):1.0, _swy=getenv("TAM_SWY")?atof(getenv("TAM_SWY")):1.0;   // ★발판 스탠스 폭 스케일(hip 대비): 앞뒤(x)·좌우(y). 최적폭 탐색용
    for(int i=0;i<4;i++){ int hb=q.hip_bid[i];
      P.hip_offsets(i,0)=(q.d->xpos[hb*3]-base[0])*_swx; P.hip_offsets(i,1)=(q.d->xpos[hb*3+1]-base[1])*_swy; P.hip_offsets(i,2)=0; }
    std::printf("[tamols-prm] mass=%.1f h=%.3f fr=%.3f l_max=%.2f I=(%.3f,%.3f,%.3f) hipx=%.3f hipy=%.3f\n",
      P.mass,P.nominal_height,P.foot_radius,P.l_max,P.inertia_diag[0],P.inertia_diag[1],P.inertia_diag[2],
      P.hip_offsets(0,0),P.hip_offsets(0,1));
  }
  void tamols_online_replan(mjData* d){
    if(getenv("TAM_PRM_MODEL") && !tam_prm_done){ tamols_init_prm(); tam_prm_done=true; }
    mjModel* m=q.m; double bx=d->qpos[0], by=d->qpos[1];
    bool terr = getenv("TAMOLS_TERRAIN")!=nullptr;   // ★지형 heightmap을 솔버에 주입(flat_costmap 대체). off=평지회귀
    if(terr){   // ── mj_ray 실측 지형 → 로컬 heightmap(yaw회전·월드z) ──
      tmap.foot_r=q.fr[0]; tmap.update(m,d,bx,by,(uint64_t)(d->time*1e9));
      double qw=d->qpos[3],qx=d->qpos[4],qy=d->qpos[5],qz=d->qpos[6];
      double yaw=std::atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)), cs=std::cos(yaw), sn=std::sin(yaw);
      int N=tam_ol_ms; double cell=tam_ol_cell, off=cell*N/2.0;
      tam_ol_h=tamols::Grid::Zero(N,N);
      for(int k=0;k<N;k++) for(int l=0;l<N;l++){
        double lx=k*cell-off, ly=l*cell-off, wx=bx+cs*lx-sn*ly, wy=by+sn*lx+cs*ly;
        double z=tmap.z(wx,wy); tam_ol_h(k,l)=(z<-50.0)?0.0:z; }   // 무효셀=0(평지)
      tam_z0=tmap.z(bx,by); if(tam_z0<-50.0) tam_z0=0.0; tam_z0+=getenv("TAM_PRM_MODEL")?q.base_z0:0.50;   // base=로봇밑 지형+명목높이(로봇무관: base_z0)
      tam_ol_mapinit=true;
    } else if(!tam_ol_mapinit){ tamols::flat_costmap(tam_ol_h,tam_ol_cell,tam_ol_ms); tam_z0=getenv("TAM_PRM_MODEL")?q.base_z0:0.52; tam_ol_mapinit=true; }
    Eigen::Matrix<double,4,3> fmeas;
    for(int i=0;i<4;i++){ Vector3d fp=q.foot_point(i); fmeas(i,0)=fp[0]-bx; fmeas(i,1)=fp[1]-by; fmeas(i,2)=fp[2]; }
    int _rti=getenv("RTI_ITER")?atoi(getenv("RTI_ITER")):5;   // ★warm RTI iteration(기본5). ↑=warm-drift↓(발판 명목 회복 여유)
    tamols::OnlineCfg cfg; cfg.vadv=tam_inj?0.0:V; cfg.phase_dur=getenv("PHASE_DUR")?atof(getenv("PHASE_DUR")):0.2; cfg.rti_iter=tam_ol_warm?_rti:60; cfg.warm=tam_ol_warm;   // ★phase_dur 노출: 짧을수록 빠른 cadence(고속 발 따라잡기). 주입=plan 정지
    cfg.walk = getenv("RSL_GAIT") && !std::strcmp(getenv("RSL_GAIT"),"walk");   // ★walk(정적안정) vs trot
    int _Pg = cfg.walk?4:5;
    cfg.phase_off=tam_ol_phase;   // ★horizon-shift: 위상 회전(swing 실행)
    int _gnd=0; for(int i=0;i<4;i++) if(q.foot_point(i)[2]<0.04) _gnd++;   // 접지 발 수
    if(!getenv("GAIT_SYNC") || _gnd>=3) tam_ol_phase=(tam_ol_phase+1)%_Pg;   // ★GAIT_SYNC: swing 착지(≥3지지) 시만 phase 진행=실제 발상태 동기(재anchor desync 방지). 미착지면 phase 유지(발 착지 대기)
    double vx0=d->qvel[0], vy0=d->qvel[1];
    if(getenv("VCLAMP")){ double vc=atof(getenv("VCLAMP"));   // ★plan 초기속도 클램프: 측정 속도스파이크를 plan이 echo/증폭하는 것 차단(vadv 근처로)
      vx0=tc_clip(vx0, V-vc, V+vc); vy0=tc_clip(vy0, -vc, vc); }
    if(!tam_inj && getenv("GAP_X0")){ cfg.gap_x0=atof(getenv("GAP_X0"))-bx; cfg.gap_x1=atof(getenv("GAP_X1"))-bx; }  // ★월드 gap→로컬. 주입모드는 per-foot 회피(전역straddle 끔)
    cfg.z0_terrain = getenv("TAMOLS_TERRAIN")!=nullptr;   // ★지형모드=z밴드를 z0 기준 상대로(계단 base 상승 허용)
    tamols::online_replan(tam_ol,tam_ol_h,tam_ol_cell,tam_ol_ms, tam_z0,0.0,vx0,vy0, fmeas, cfg);
    tam_ol_warm=true;
    // 샘플링 → tam_s/tam_c/tam_fh
    std::vector<std::array<double,12>> smp; std::vector<std::array<int,4>> con; Eigen::Matrix<double,4,3> fh;
    tamols::sample_plan(tam_ol, tam_dt, smp, con, fh);
    tam_s=smp; tam_c=con; tam_N=(int)smp.size();
    for(int i=0;i<4;i++){ tam_fh[i]=Vector3d(fh(i,0),fh(i,1),fh(i,2)); }
    tam_loaded=true;
    if(getenv("TAM_DUMP")){   // ★계획을 load_tamols 포맷으로 덤프(offline 추종용, re-anchor 회피 테스트)
      std::ofstream df(getenv("TAM_DUMP")); df<<tam_N<<" "<<tam_dt<<"\nFOOTHOLDS\n";
      for(int L=0;L<4;L++) df<<tam_fh[L][0]<<" "<<tam_fh[L][1]<<" "<<tam_fh[L][2]<<"\n";
      df<<"SAMPLES\n";
      for(int kk=0;kk<tam_N;kk++){ df<<(kk*tam_dt); for(int j=0;j<12;j++) df<<" "<<tam_s[kk][j]; for(int i=0;i<4;i++) df<<" "<<tam_c[kk][i]; df<<"\n"; }
      std::printf("[TAM_DUMP] 계획 기록 N=%d dt=%.3f → %s (발판 대칭확인: FL_y=%.3f FR_y=%.3f RL_y=%.3f RR_y=%.3f)\n",
        tam_N,tam_dt,getenv("TAM_DUMP"),tam_fh[0][1],tam_fh[1][1],tam_fh[2][1],tam_fh[3][1]); }
  }
  Vector3d tam_ptgt[4]; bool tam_have_ptgt[4]={false,false,false,false};
  void load_tamols(const char* path){
    std::ifstream f(path); if(!f){ tam_N=0; tam_loaded=true; std::printf("[tamols] 파일 없음: %s\n",path); return; }
    tam_s.clear(); tam_c.clear(); std::string tok;
    f>>tam_N>>tam_dt; f>>tok;                 // N dt, "FOOTHOLDS"
    for(int L=0;L<4;L++) f>>tam_fh[L][0]>>tam_fh[L][1]>>tam_fh[L][2];
    f>>tok;                                   // "SAMPLES"
    for(int k=0;k<tam_N;k++){ double t; std::array<double,12> s; std::array<int,4> c;
      f>>t; for(int j=0;j<12;j++) f>>s[j];    // t + pose6(x y z r p yw) + vel6(vx vy vz wr wp wy)
      for(int i=0;i<4;i++) f>>c[i];
      tam_s.push_back(s); tam_c.push_back(c); }
    tam_loaded=true; std::printf("[tamols] 로드 N=%d dt=%.4f 발판FL(%.2f,%.2f)\n",tam_N,tam_dt,tam_fh[0][0],tam_fh[0][1]);
  }
  double SGU_KICK_T=0.5, SGU_FB_THIGH=-0.55, SGU_FB_CALF=1.20, SGU_SLEW=1.5, SGU_KP=120, SGU_GATHER_Z=0.24, SGU_DONE_TILT=22;
  double SGU_WALKOUT_V=0.6, SGU_HANDOFF_Z=0.34;   // ★기립 후 전진 트로트로 인계(walk-out)해 균형회복. bz>HANDOFF_Z면 move로 전환
  double SIT_SLEW=0.6; // ★앉기 하강 슬루(rad/s, 작을수록 천천히·충격↓). 기본 JOINT_SLEW(1.5)보다 느리게→충격 완화
  // ★★점프(J0.2 스크립트 pronk, RPET_JUMP_MPC.md): crouch(wbic)→thrust(강PD 신전)→flight(tuck)→touchdown→흡수→wbic회복. 이벤트 트리거(시간 아님).
  int jphase=-1; double jt0=0, jzpk=0; VectorXd jqc, jqs;
  double JUMP_CROUCH_Z=0.30, JUMP_THRUST_T=0.16, JUMP_KP=380, JUMP_KD=6;   // (구 스크립트 스냅용)
  // ★★J2 통합: OCP 궤적(/tmp/jump_traj.txt, MuJoCo순서 phase·q·dq·tau) 재생 → thrust/flight, 착지는 wbic_stance
  std::vector<VectorXd> jump_q, jump_dq, jump_tau; std::vector<int> jump_ph;
  std::vector<Vector3d> jump_com, jump_comv, jump_acom;   // ★WBIC-추종용 CoM 위치/속도/가속(궤적 자체 프레임: yaw0·원점기준)
  int jump_N=-1, jump_k=-1; double jump_dt=0.01, jump_kt=0, JUMP_TRAJ_KP=120, JUMP_TRAJ_KD=4;
  // ★점프 궤적을 "발사 시점 현재 자세(위치·헤딩)"로 re-anchor: 선회 후 점프해도 바라보는 방향으로 뛰고 스핀 안 함(궤적은 yaw0 프레임이라 그대로 쓰면 world+x로 튐)
  bool jump_anchored=false; Vector3d jump_launch_com=Vector3d::Zero(), jump_com0=Vector3d::Zero(); double jump_launch_yaw=0;
  void load_jump(const std::string& path){
    std::ifstream f(path); if(!f){ jump_N=0; return; }
    f>>jump_N>>jump_dt; jump_q.clear(); jump_dq.clear(); jump_tau.clear(); jump_ph.clear();
    jump_com.clear(); jump_comv.clear(); jump_acom.clear();
    for(int k=0;k<jump_N;k++){ int ph; f>>ph; jump_ph.push_back(ph);
      VectorXd qq(q.nu),dd(q.nu),tt(q.nu);
      for(int j=0;j<q.nu;j++) f>>qq[j]; for(int j=0;j<q.nu;j++) f>>dd[j]; for(int j=0;j<q.nu;j++) f>>tt[j];
      Vector3d cm,cv,ca; for(int c=0;c<3;c++) f>>cm[c]; for(int c=0;c<3;c++) f>>cv[c]; for(int c=0;c<3;c++) f>>ca[c];
      jump_q.push_back(qq); jump_dq.push_back(dd); jump_tau.push_back(tt);
      jump_com.push_back(cm); jump_comv.push_back(cv); jump_acom.push_back(ca); } }
  double JUMP_VX=0.6;   // ★점프 전방 이륙속도(0=수직 제자리). live-solve·gen_jump 공통 의미
#ifdef HAVE_JUMP_SOLVER
  std::string JUMP_URDF="/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf";
  std::string JUMP_MJCF="/home/jsh/문서/jsh/simulation/quad/mjcf/quad_real_16dof_sphere.mjcf";   // ★16-DOF(허리 제거). URDF 는 허리 있는 판을 쓰되 jump_solver 가 허리를 0°로 잠금
  int JUMP_MAXIT=8;     // ★crouch중 live-solve FDDP 반복(S2: iter~8. crouch 예산 450ms 내)
  JumpSolver jsolver; bool jsolver_ready=false;   // ★셋업(모델·MJCF·IK) 캐시 → 점프마다 solve만
  std::future<JumpTraj> jfut; bool jsolving=false, jlaunched=false;   // ★실기대비: solve를 별도 스레드로(1kHz 루프 안 멈춤). crouch 유지하며 백그라운드 계산→완료 시 인계
  void warmup_jump(){   // ★뷰어 시작 시 1회: 무거운 셋업 + 예열 solve → 첫 점프도 빠르게(stall=solve만)
    if(jsolver_ready) return;
    jsolver_ready=jsolver.init(JUMP_URDF,JUMP_MJCF);
    if(jsolver_ready) jsolver.solve(JUMP_VX,JUMP_MAXIT,false); }
  bool solve_jump_live(double vx){   // crouch 정착 시 1회 → 신선 궤적 인메모리(셋업은 재사용, solve만)
    if(!jsolver_ready){ jsolver_ready=jsolver.init(JUMP_URDF,JUMP_MJCF); if(!jsolver_ready) return false; }
    auto _t=std::chrono::steady_clock::now();
    JumpTraj J=jsolver.solve(vx,JUMP_MAXIT,false);
    double _ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-_t).count();
    std::printf("[jump] live-solve %.0f ms (crouch stall) · apex=%.3f\n", _ms, J.apex); std::fflush(stdout);
    if(J.N<=0) return false;
    jump_N=J.N; jump_dt=J.dt; jump_q=J.q; jump_dq=J.dq; jump_tau=J.tau; jump_ph=J.ph;
    jump_com=J.com; jump_comv=J.comv; jump_acom=J.acom; return true; }
#endif
  // 모드관리 상수(Python 17dof와 동일)
  double GROUND_Z=0.18, GETUP_TRIG=0.32, GETUP_DONE=0.40, GETUP_KP=90, GETUP_KD=3, GETUP_RATE=0.18, REST_KD=3.0, JOINT_SLEW=1.5, HRATE=0.3;
  double GROUND_LIE_Z=0.226, GROUND_REAR_FOOT=-1.15, GROUND_FRONT_FOOT=-0.5, GROUND_FRONT_THIGH=-0.24, GROUND_FRONT_CALF=-0.4;   // ★눕기(ground) 저자세: base 낮춤 + 앞뒤 발목/앞다리 fold(GUI 실시간 슬라이더로 CoM균형·수평·무슬라이드 조각). PD-fold 홀드. ★기본값=뷰어 라이브튜닝서 확정(base_z≈0.166 깊은 belly-lie)
  // ── 게이트 프리셋(trot/walk/gallop) ──
  std::string gait_type="trot";
  double gp_T=0.5, gp_SWF=0.5, gp_off[4]={0,0.5,0.5,0}, gp_Tsw=0.25, gp_Tst=0.25;
  // ── ★속도 트리거 자동 whip(고속 trot=동물형 채찍질) ──
  bool auto_whip=true; double whip_v0=0.8, whip_v1=1.6;    // ★기본 ON: v0~v1서 whip 선형증가(swing_w 2.0→낮게)
  double whip_hi=2.0, whip_lo_f=0.1, whip_lo_r=0.6;        // ★앞발 paw-tuck whip(앞0.1강·뒤0.6). yaw-fight 수정 후 선회 최고(15.8°)+원래 의도. 슬라이더로 조절
  double waist_steer=0.4, waist_cap=0.20;                  // ★허리 lean 보조 게인(선회시 앞몸통 안쪽 굽힘=차동차식). 캡±0.20rad(11°, 구0.75=43°는 낙상). 다리(MPC)가 선회수행·허리는 lean
  double steer=0.0, Ss_steer=0.0, wheelbase=0.5;           // ★자동차식 조향각 δ[rad](GUI 허리핸들). Ackermann Weff+=V·tanδ/축거(전진해야 조향). 축거=앞뒤 힙거리
  // 상태
  bool armed=false; double t0=0, settle_until=TC_SETTLE;
  bool stop_settle=false;   // ★달리다 서기: 정지 감속·CoM 재중심 중(뒤 엉덩방아 방지)
  double stand_ax=0, stand_ay=0; bool stand_set=false;   // ★서기 위치 앵커(현재 위치서 서기 — 홈으로 빨려감 방지)
  bool off_settled=false;   // ★전원차단(off): 바닥까지 완만 하강 완료 여부(선 채로 툭 damp=낙하 방지)
  double Vs=0,Vys=0,Ws=0, yaw_ref=0; bool yaw_hold_set=false; double yaw_hold=0;
  bool pos_hold_set=false; double phx=0,phy=0;
  VectorXd x_ref=VectorXd::Zero(13);
  std::array<Vector3d,4> liftoff, nominal; std::array<Vector2d,4> hip_off; std::array<double,4> gz;
  std::array<bool,4> have_prev={false,false,false,false}; std::array<Vector3d,4> ptgt_prev;
  Vector3d lam_des[4]={Vector3d::Zero(),Vector3d::Zero(),Vector3d::Zero(),Vector3d::Zero()};
  double mpc_t=-1.0, Veff_dbg=0;

  TrotCtrl(QuadControl& q_):q(q_){ q_ref=VectorXd::Zero(q.nu); q_crouch=q_haunch=VectorXd::Zero(q.nu); body_h=ht_cur=qhome_h=q.base_z0;
    double xf=(q.d->xpos[q.hip_bid[2]*3]+q.d->xpos[q.hip_bid[3]*3])/2;   // FL,FR 힙 x(초기 yaw0=body frame)
    double xr=(q.d->xpos[q.hip_bid[0]*3]+q.d->xpos[q.hip_bid[1]*3])/2;   // HL,HR 힙 x
    wheelbase=std::max(0.15, xf-xr); }                                   // ★축거 L(≈0.61m). Ackermann 조향 반경 R=L/tanδ

  void set_gait(const std::string& g){        // trot/walk/gallop 프리셋(GUI 토글·속도트리거)
    if(g==gait_type) return; gait_type=g;
    if(g=="walk"){ gp_T=0.7; gp_SWF=0.25; gp_off[0]=0.25; gp_off[1]=0.75; gp_off[2]=0.5; gp_off[3]=0.0; raibert_k=0.5; step_h=0.10; gait_base_z=0.50; PCV_CLR=0.04; }  // ★walk 안정화(T1.0→0.7·RAI0.8→0.5): reach↓ stumble/bounce방지, 상한~0.6m/s. ★발높이 0.10 복원(9fcfe81서 0.05로 반감됐던 것=발 아치 낮아 앞다리 뻣뻣, falls=0 유지)
    else if(g=="run"){ gp_T=0.40; gp_SWF=0.5; gp_off[0]=0.0; gp_off[1]=0.5; gp_off[2]=0.5; gp_off[3]=0.0; raibert_k=0.5; step_h=0.08; gait_base_z=0.48; PCV_CLR=0.04; }  // ★고속 trot(빠른 cadence T0.4·낮은 발높이0.08): 최고속 1.8→~2.0m/s, 발목ω↓. ★base_z 0.48(발목ω 여유)
    else if(g=="stairs"){ gp_T=0.7; gp_SWF=0.25; gp_off[0]=0.25; gp_off[1]=0.75; gp_off[2]=0.5; gp_off[3]=0.0; raibert_k=0.4; step_h=0.18; gait_base_z=0.50; PCV_CLR=0.08; }  // ★계단 등반: walk 시퀀셜(정적안정)+높은 발높이0.18(라이저 클리어)+낮은 reach0.4(정밀 발배치)+2배 up-step 클리어런스. 느린속도(GUI 0.3)와 조합. 완만한 계단(≤0.10) 확실·가파른(0.15+)은 marginal
    else if(g=="gallop"){ gp_T=0.35; gp_SWF=0.55; gp_off[0]=0.0; gp_off[1]=0.05; gp_off[2]=0.55; gp_off[3]=0.5; raibert_k=0.8; step_h=0.10; PCV_CLR=0.04; } // 회전형 갤럽(비행상 有)
    else if(g=="bound"){ gp_T=0.45; gp_SWF=0.5; gp_off[0]=0.5; gp_off[1]=0.5; gp_off[2]=0.0; gp_off[3]=0.0; raibert_k=0.5; step_h=0.10; gait_base_z=0.48; PCV_CLR=0.04; } // ★bound(실험적·불안정): 앞쌍(FL,FR)·뒷쌍(HL,HR) 동위상 교대. ⚠검증(2026-08): 2점 측방축 지지가 [레벨홀드+추진+높이유지] 동시불가→z 0.48→0.25침하·피치8~42°·V≥1.0 붕괴. 반응형 MPC+WBIC 한계(bound=본질적 피치진동=RL/궤적최적화 몫, Raibo2025도 RL). 안정화=MPC 피치-레퍼런스 진동 필요(별도 개발)
    else         { gp_T=0.5; gp_SWF=0.5;  gp_off[0]=0.0;  gp_off[1]=0.5;  gp_off[2]=0.5; gp_off[3]=0.0; raibert_k=0.5; step_h=0.10; gait_base_z=0.50; PCV_CLR=0.04; }  // ★trot 표준 중립(GRF균형·뒤thigh절반·falls=0·push복구↑)
    gp_Tsw=gp_T*gp_SWF; gp_Tst=gp_T*(1.0-gp_SWF); armed=false;   // 재arm=위상 재앵커(불연속 방지)
  }
  void gait(int i,double tg,bool&stance,double&sprog){
    double ph=std::fmod(tg/gp_T+gp_off[i],1.0); if(ph<0) ph+=1.0;
    if(ph>=gp_SWF){ stance=true; sprog=0.0; } else { stance=false; sprog=ph/gp_SWF; }
  }
  void reset(){   // ★시뮬 리셋(RESET 버튼): 컨트롤러 상태 초기화(crouch_home 후 호출)
    armed=false; settle_until=q.d->time+TC_SETTLE; have_qref=false;
    Vs=Vys=Ws=Ss_steer=0; yaw_hold_set=false; mpc_t=-1.0;
    ht_cur=qhome_h=body_h=q.base_z0; for(int i=0;i<4;i++) have_prev[i]=false;
    haunch_ready=false; haunch_fold=0; jphase=-1; jump_anchored=false;
#ifdef HAVE_JUMP_SOLVER
    jlaunched=false; jsolving=false;
#endif
  }

  // 1틱 제어: d->ctrl 설정(mj_step은 호출자). q.d->time 기준.
  void control(){
    mjModel*m=q.m; mjData*d=q.d; int nv=q.nv; double dt=m->opt.timestep;
    double t=d->time; int nu=q.nu;
    if(foot_nudge){ static long _mk=0; if(_mk++ % 50 == 0){ tmap.foot_r=q.fr[0]; tmap.update(m,d,d->qpos[0],d->qpos[1],(uint64_t)(t*1e9)); } }  // ★맵 rate(~20Hz) 갱신, 1kHz 밖
    // ── 모드 dispatch(배포용): move 외 = 서기/눕기/getup/off ──
    if(mode!="sit") q.sit_pitch+=tc_clip(0.0-q.sit_pitch,-1.2*dt,1.2*dt);   // ★nose-up 부드럽게 해제(리셋 아닌 슬루=언폴드 중 뒷다리 펴짐과 함께 nose-up 풀림)
    if(mode!="sit") sit_init=false;                                          // ★sit 이탈=진입방향 latch 리셋(다음 진입서 재판정)
    q.posture_w=1.0; q.sit_hock_contact=false;   // ★자세task 가중·뒤 hock접촉 기본off(서기/보행). 개-앉기 홀드서만 on
    if(mode!="jump" && jphase>=0) jphase=-1;      // 점프 중 다른 모드로 이탈=상태 초기화(재진입 정상화)
    // ★달리다 서기(stand_up): 전진속도 크면 먼저 move(명령0)로 감속→발이 CoM 밑으로 재중심된 뒤 stand. (뒤 엉덩방아 방지)
    if(mode=="stand_up" && armed){ double sp=std::hypot(d->qvel[0],d->qvel[1]);
      if(sp>0.20) stop_settle=true; else if(sp<0.12) stop_settle=false; }
    else stop_settle=false;
    if(mode!="off") off_settled=false;   // ★off 벗어나면 리셋(다음 off서 다시 완만 하강)
    // ★★TAMOLS 재생: 궤적(base pose/vel + 발판 + 접촉)을 로드해 보행스택(MPC→lam→wbic_track) 참조로 주입.
    if(mode=="tamols"){
      bool online=getenv("RSL_ONLINE");
      if(online){ double RDT=getenv("REPLAN_DT")?atof(getenv("REPLAN_DT")):0.2;
        bool first=(tam_ol_last<0);
        if(first || t-tam_ol_last>=RDT){ tamols_online_replan(d); tam_ol_last=t;
          if(first) tam_t=-1;                                    // 첫 진입=아래 full 앵커+swing 리셋
          else { tam_t0=t; tam_ax=d->qpos[0]; tam_ay=0.0; } } }  // ★proper receding-horizon: 후속=위치만 재앵커(swing 연속·저크 제거)
      else if(!tam_loaded) load_tamols(getenv("TAMOLS_TRAJ")?getenv("TAMOLS_TRAJ"):"/tmp/tamols_traj.txt");
      if(tam_N<=0){ q.wbic_stance(); armed=false; return; }             // 파일없음=제자리
      if(tam_t<0){ tam_t0=t; tam_ent=t; tam_ax=d->qpos[0]; tam_ay=online?0.0:d->qpos[1];   // 진입=현재 위치 앵커. ★온라인=y를 0고정(직선 유지, 드리프트 보정). tam_ent=진입 절대시각(settle)
        for(int i=0;i<4;i++){ tam_sw_prev[i]=(tam_c[0][i]!=0); tam_have_ptgt[i]=false; }
        x_ref.setZero(); x_ref[12]=-9.81; com_h0=d->subtree_com[2]; }
      tam_t=t-tam_t0;
      int k=std::min((int)(tam_t/tam_dt),tam_N-1); if(k<0)k=0;
      auto& s=tam_s[k]; auto& cc=tam_c[k];
      double tgt_x=tam_ax+s[0], tgt_y=tam_ay+s[1];                      // base 위치 피드백(드리프트 보정) + TAMOLS 속도
      if(getenv("TAM_WDBG")){ static double _lw=-99; if(d->time-_lw>=0.3){ _lw=d->time;
        int nsw=0; for(int i=0;i<4;i++) if(cc[i]==0) nsw++;
        // ★2-접촉 wrench 랭크결손 진단: 두 지지발을 잇는 대각선 축(u)에 대한 모멘트는 점접촉 힘으로 절대 못 냄(rank-5).
        //   M_null=m·g·d_perp(CoM의 대각선 수직거리)=uncorrectable 전복 모멘트. tilt와 함께 커지면 곧 그 축 벽의 smoking gun.
        double d_perp=0, M_null=0; std::vector<int> _stf;
        for(int i=0;i<4;i++) if(cc[i]!=0) _stf.push_back(i);
        if(_stf.size()==2){ Vector3d p1=q.foot_point(_stf[0]), p2=q.foot_point(_stf[1]);
          double ux=p2[0]-p1[0], uy=p2[1]-p1[1], ul=std::hypot(ux,uy);
          if(ul>1e-6){ ux/=ul; uy/=ul;
            double cx=d->subtree_com[0]-p1[0], cy=d->subtree_com[1]-p1[1];
            d_perp=cx*(-uy)+cy*ux;                       // CoM→대각선 수직거리(부호)
            M_null=q.mpc.TOTAL_MASS*9.81*d_perp; } }     // 대각선 축 uncorrectable 모멘트
        int ng=0; for(int i=0;i<4;i++) if(q.foot_point(i)[2]<0.03) ng++;   // ★실제 접지 발 개수(계획 nsw와 별개=phantom stance 검출)
        double qw=d->qpos[3],qx=d->qpos[4],qy=d->qpos[5],qz=d->qpos[6];     // ★roll(측방)/pitch(전후) 분리=붕괴 축 진단
        double roll=std::atan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy))*57.3;
        double pitch=std::asin(std::max(-1.0,std::min(1.0,2*(qw*qy-qz*qx))))*57.3;
        std::printf("[WDBG] t=%.2f  comz=%.3f planz=%.3f  roll=%+.1f pitch=%+.1f  comy=%+.3f plany=%+.3f  nsw=%d ng=%d\n",
          d->time,d->subtree_com[2],s[2],roll,pitch,d->subtree_com[1],s[1],nsw,ng); } }
      double vx_w=s[6]+tc_clip(-1.0*(d->qpos[0]-tgt_x),-0.3,0.3);
      double vy_w=s[7]+tc_clip(-1.0*(d->qpos[1]-tgt_y),-0.3,0.3);
      if(online && getenv("TAM_CLEANV")){   // ★X(전진)만 clean(solve_fast wild vx 회피). ★Y(sway)·z·자세는 plan 유지=GIAC body-sway 보존(CoM 지지폴리곤 유지=lateral 드리프트 방지). 구버전이 tgt_y=0으로 sway 지운 게 lateral 드리프트 원인
        tgt_x=tam_ax+V*tam_t; vx_w=V; }   // ★(YHOLD 강 lateral 속도홀드 실험=무효 확인, 제거. 횡드리프트는 발배치/gait 타이밍 수준)
      if(online && getenv("VADV_REF")){   // ★MPC식 계획 위치참조: V로 전진 + 측정으로 leaky 이완. leak↓=강복원(속도폭주 억제·z침하위험)·leak↑=re-anchor(폭주). 위치오차가 복원력 제공
        double leak=getenv("VADV_LEAK")?atof(getenv("VADV_LEAK")):1.0;
        if(!tam_vbinit){ tam_vbx=d->qpos[0]; tam_vby=d->qpos[1]; tam_vbinit=true; }
        tam_vbx += (V + leak*(d->qpos[0]-tam_vbx))*dt; tam_vby += (leak*(d->qpos[1]-tam_vby))*dt;
        tgt_x=tam_vbx; tgt_y=tam_vby; vx_w=V; }
      x_ref[3]=tgt_x; x_ref[4]=tgt_y;   // ★base x,y 위치 참조(오버슛/횡드리프트 방지) — Qdiag[3,4]>0 필요
      x_ref[2]=s[5]; x_ref[5]=s[2]; x_ref[8]=s[11]; x_ref[9]=vx_w; x_ref[10]=vy_w; q.yaw_des=s[5];
      if(getenv("YAW0")){ q.yaw_des=0; x_ref[2]=0; }   // ★★option1(재anchor 근본): yaw 참조를 plan(재anchor로 드리프트 추종) 대신 world-frame 절대0으로 고정 → yaw ratchet 드리프트 제거(직진 heading hold)
      bool rsl=(getenv("RSL_TRACK")||online) && !getenv("TAM_MPC");   // ★RSL식 직접추종. ★TAM_MPC=1이면 MPC 기반 추종(z침하 해결: RSL gravity-comp가 MPC 우회→침하)
      if(rsl){ auto& sn=tam_s[std::min(k+1,tam_N-1)];
        q.W_BASE_XY=getenv("W_BASE_XY")?atof(getenv("W_BASE_XY")):80.0;
        q.w_ori=getenv("W_ORI")?atof(getenv("W_ORI")):200.0;   // ★RSL: 자세 강홀드(축소지지서 레벨링 모멘트 우선). 낮으면 첫swing tilt발산
        q.KD_BASE=getenv("KD_BASE")?atof(getenv("KD_BASE")):25.0;   // ★속도 추종 게인(전진 authority)
        q.SW_TRACK_W=getenv("SW_TRACK_W")?atof(getenv("SW_TRACK_W")):90.0;   // ★swing 발 추종 가중(착지 정확)
        q.KP_BASE=getenv("KP_BASE")?atof(getenv("KP_BASE")):150.0;
        q.com_ref[0]=tgt_x; q.com_ref[1]=tgt_y; q.com_ref[2]=s[2];   // 계획 base 위치. ★온라인 위치ref vadv전진은 z침하 유발→plan 추종 유지(안정 우선)
        q.com_vel_ref[0]=online?V:s[6]; q.com_vel_ref[1]=s[7]; q.com_vel_ref[2]=s[8];   // 계획 base 속도(★z속도도 전달=WBC가 base bob 예측→z침하 완화)
        double _accap=getenv("ACC_CAP")?atof(getenv("ACC_CAP")):3.0;   // ★계획 가속 ff 클램프(스파이크→pitch-dive 방지)
        q.com_acc_ref[0]=tc_clip((sn[6]-s[6])/tam_dt,-_accap,_accap); q.com_acc_ref[1]=tc_clip((sn[7]-s[7])/tam_dt,-_accap,_accap);
        q.com_acc_ref[2]=tc_clip((sn[8]-s[8])/tam_dt,-_accap,_accap); }   // ★z가속 ff(접촉전이 예측=z침하 핵심)
      else { double qp=getenv("TAM_QPOS")?atof(getenv("TAM_QPOS")):50.0; q.mpc.Qdiag[3]=qp; q.mpc.Qdiag[4]=qp; }
      std::vector<int> st; std::map<int,std::pair<Vector3d,Vector3d>> swing;
      double SW_DUR=getenv("SW_DUR")?atof(getenv("SW_DUR")):(online?0.14:0.4);   // ★swing<위상: 여유 두고 착지완료→위상전환 시 발이 실제 지면(phantom stance 방지). ★env를 offline도 읽음(계획 swing위상에 맞춰야=phantom 방지)
      double sh=online?(getenv("STEP_H")?atof(getenv("STEP_H")):0.05):0.08;      // ★온라인 발높이 낮춤(빠른 착지)
      // ★capture-point 발배치(A식): raibert FF + 속도오차 피드백=균형(횡드리프트 억제 핵심). leg-독립이라 1회 계산
      double _rai=getenv("TAM_RAI")?atof(getenv("TAM_RAI")):0.5;      // raibert_k(FF), A표준 0.5
      double _kcap=getenv("TAM_KCAP")?atof(getenv("TAM_KCAP")):0.16;  // capture 피드백 게인
      double _tst=getenv("TAM_TST")?atof(getenv("TAM_TST")):0.25;     // stance time(trot T0.5)
      double _capc=getenv("TAM_CAPCLIP")?atof(getenv("TAM_CAPCLIP")):0.25;
      mj_subtreeVel(m,d); double _H=std::max(0.1,d->subtree_com[2]), _M=q.mpc.TOTAL_MASS;
      double vfx=d->subtree_linvel[0]+d->subtree_angmom[1]/(_M*_H);   // CoM 속도(world)+각운동량 capture 보정
      double vfy=d->subtree_linvel[1]-d->subtree_angmom[0]/(_M*_H);
      double capx=tc_clip(_rai*_tst*V+_kcap*(vfx-V),-_capc,_capc);     // 전진 capture
      double _kyp=getenv("YPOS_K")?atof(getenv("YPOS_K")):0.0;        // ★횡 위치 앵커(정적crawl drift 보정): CoM y를 tam_ay(직진선)로 되돌림. capy는 속도만 잡아 위치는 누적드리프트
      double capy;
      if(getenv("YDRIFT_K")){   // ★★저주파 드리프트 보정(정적 crawl 정석): comy를 low-pass해 빠른 sway 제거→느린 드리프트만 발배치로 보정(sway와 안 싸움. DCM은 sway와 충돌해 발산).
        double _tau=getenv("YDRIFT_TAU")?atof(getenv("YDRIFT_TAU")):0.6;   // 시상수>gait주기(0.8s) → sway(1.25Hz) 통과 차단, 드리프트(<0.3Hz)만
        comy_lp += (dt/std::max(0.05,_tau))*((d->subtree_com[1]-tam_ay)-comy_lp);
        capy=tc_clip(atof(getenv("YDRIFT_K"))*comy_lp,-_capc,_capc);
      } else if(getenv("DCM_K")){   // ★★DCM 발배치(정석): K_CAP·v + YPOS_K·pos를 ω-원리적으로 통합. ξ_y=com_y+v_y/ω(=capture point), foot 편차=k·ξ_y.
        double zc=std::max(0.15,d->subtree_com[2]), om=std::sqrt(9.81/zc);   // ω=√(g/z) LIPM
        double xiy=(d->subtree_com[1]-tam_ay)+vfy/om;                        // 횡 DCM 오차(위치+속도/ω, 물리적 비율 1:1/ω)
        capy=tc_clip(atof(getenv("DCM_K"))*xiy,-_capc,_capc);
      } else
        capy=tc_clip(_kcap*(vfy-0.0)+_kyp*(d->subtree_com[1]-tam_ay),-_capc,_capc);   // 구: K_CAP 속도 + YPOS_K 위치(휴리스틱)
      for(int i=0;i<4;i++){ bool sched=(cc[i]!=0);
        bool grounded=(!online)||(q.foot_point(i)[2]<0.03);   // ★온라인=실제 접지 확인(phantom stance 방지)
        if(sched && grounded){ st.push_back(i); tam_have_ptgt[i]=false; tam_sw_prev[i]=true; }   // 실제 접지 stance만
        else if(!sched){ if(tam_sw_prev[i]){ tam_lift[i]=q.foot_point(i); tam_sw_start[i]=t;   // ★liftoff: 절대시간 캡처(재anchor에도 진행 연속)
            double tgx=tam_ax+tam_fh[i][0]+capx, tgy=tam_ay+tam_fh[i][1]+capy, tgz=tam_fh[i][2];
            if(getenv("FOOT_SNAP") && getenv("TAMOLS_TERRAIN") && !tmap.valid(tgx,tgy)){   // ★발이 void 위→안전지형 snap(전방우선=gap 건너 far 플랫폼 도달). capture와 지형제약 reconcile
              double _sr=getenv("SNAP_REACH")?atof(getenv("SNAP_REACH")):0.35;
              for(double dx=0.04; dx<=_sr; dx+=0.04){ if(tmap.valid(tgx+dx,tgy)){ tgx+=dx; break; }
                                                       if(tmap.valid(tgx-dx*0.5,tgy)){ tgx-=dx*0.5; break; } } }
            if(getenv("FOOT_SNAP") && getenv("TAMOLS_TERRAIN")){ double tz=tmap.z(tgx,tgy); if(tz>-50.0) tgz=tz; }  // 발 z=지형 높이
            tam_swtgt[i]=Vector3d(tgx,tgy,tgz); tam_sw_prev[i]=false; }  // ★착지target=계획발판+capture+지형snap. swing foot commitment 동결
          double sprog=tc_clip((t-tam_sw_start[i])/SW_DUR,0.0,1.0);   // ★절대시간 진행(tam_t 리셋 무관)
          Vector3d p_end=tam_swtgt[i];   // ★동결된 target(재anchor에도 불변→발이 완주해 착지)
          Vector3d bvel(s[6],s[7],0.0);
          Vector3d p_tgt=tc_swing_foot(sprog,tam_lift[i],p_end,bvel,sh,SW_DUR,SW_DUR);
          Vector3d v_tgt=Vector3d::Zero();
          if(tam_have_ptgt[i]) for(int c=0;c<3;c++) v_tgt[c]=tc_clip((p_tgt[c]-tam_ptgt[i][c])/dt,-1.0,1.0);
          tam_ptgt[i]=p_tgt; tam_have_ptgt[i]=true; swing[i]={p_tgt,v_tgt}; }
        else { Vector3d fp=q.foot_point(i);   // ★sched stance인데 공중=phantom → 지면 착지 유도(stance 취급X)
          swing[i]={Vector3d(fp[0],fp[1],0.0),Vector3d::Zero()}; tam_sw_prev[i]=true; } }
      double dmpc=t-mpc_t; bool rslmpc=getenv("RSL_MPC");   // ★★injection: rsl(plan base 참조)+MPC 예측GRF 결합 — pure plan은 예측GRF 못내는 벽을 MPC로 우회
      if((!rsl || rslmpc) && !st.empty() && (mpc_t<0||dmpc<0||dmpc>=q.mpc.DT)){
        std::vector<std::array<int,4>> cs(q.mpc.N);
        for(int kk=0;kk<q.mpc.N;kk++){ int kf=std::min((int)((tam_t+kk*q.mpc.DT)/tam_dt),tam_N-1);
          for(int i=0;i<4;i++) cs[kk][i]=tam_c[kf][i]; }
        Matrix<double,4,3> L=q.mpc_grf(x_ref,cs); for(int i=0;i<4;i++) lam_des[i]=L.row(i).transpose(); mpc_t=t; }
      Vector3d lam_use[4]; bool grfff=getenv("GRF_FF");
      if(rsl && rslmpc && !st.empty()){ for(int i=0;i<4;i++) lam_use[i]=lam_des[i]; }   // ★rsl+MPC GRF(예측 피드백력)
      else if(rsl){ int Kc=(int)st.size(); double fz=Kc>0? mj_getTotalmass(m)*9.81/Kc : 0.0;   // ★RSL: λ=중력보상 baseline(WBC가 base task로 실제 분배)
        double M=mj_getTotalmass(m), aGx=grfff?q.com_acc_ref[0]:0.0, aGy=grfff?q.com_acc_ref[1]:0.0;  // ★GRF_FF: 계획 base가속→예측 수평 GRF
        for(int i=0;i<4;i++) lam_use[i]=Vector3d(M*aGx/std::max(1,Kc), M*aGy/std::max(1,Kc), fz); }
      else for(int i=0;i<4;i++) lam_use[i]=st.empty()?Vector3d::Zero():lam_des[i];
      double wl=(rsl&&rslmpc)?(getenv("W_LAM")?atof(getenv("W_LAM")):10.0):(rsl?(getenv("W_LAM")?atof(getenv("W_LAM")):(grfff?5.0:0.1)):10.0);   // ★rsl+MPC: λ강추종(예측GRF)
      if(getenv("TAM_DBG")){ static long _dc=0; if(_dc++%25==0){
        double* qc2=&d->qpos[3]; double yaw_a=std::atan2(2*(qc2[0]*qc2[3]+qc2[1]*qc2[2]),1-2*(qc2[2]*qc2[2]+qc2[3]*qc2[3]))*180/M_PI;
        std::fprintf(stderr,"[dbg] t=%.2f tt=%.2f z=%.3f stance=%d c=%d%d%d%d footz=%.2f/%.2f/%.2f/%.2f yawR=%.1f yaw=%.1f comy=%.3f\n",
          t,tam_t,d->subtree_com[2],(int)st.size(),cc[0],cc[1],cc[2],cc[3],
          q.foot_point(0)[2],q.foot_point(1)[2],q.foot_point(2)[2],q.foot_point(3)[2],
          x_ref[2]*180/M_PI,yaw_a,d->subtree_com[1]); } }
      // ★lateral lead(정적 crawl 핵심): plan s[1]은 재anchor로 현재 y서 출발해 lag → 대신 LEAD_CEN초 앞 위상의
      //   지지발 centroid를 직접 com_ref[1]로 명령. WBC lateral은 빠름(step 0.06→0.3s에 0.051 실측)이라 선행 추종 가능.
      if(getenv("LEAD_CEN")){ double lead=atof(getenv("LEAD_CEN"));
        int kl=std::min((int)((tam_t+lead)/tam_dt),tam_N-1); double cyl=0,cxl=0,cx4=0; int ncl=0;
        for(int i=0;i<4;i++){ cx4+=tam_fh[i][0]; if(tam_c[kl][i]!=0){ cyl+=tam_fh[i][1]; cxl+=tam_fh[i][0]; ncl++; } }
        double ycap=getenv("LEAD_YCAP")?atof(getenv("LEAD_YCAP")):0.06;   // ★과드리프트 방지 clamp
        if(ncl>0){ double _yc=tc_clip(cyl/ncl,-ycap,ycap);
          if(getenv("LEAD_ACAP")) _yc+=capy;   // ★CoM 참조를 실제 지지(nominal+capy)로 정합: capy 누적 시 CoM ref와 실제 support 어긋남 방지(느린 발산 근본)
          q.com_ref[1]=tam_ay+_yc; q.com_vel_ref[1]=0; q.com_acc_ref[1]=0;
          if(getenv("LEAD_CX")) q.com_ref[0]=tgt_x+(cxl/ncl - cx4/4);   // x-centroid 오프셋(전후 pitch 관리): 3발 centroid − 4발 평균
        } }
      // ★★CPLAN(offline 주기 CoM 궤적, 정석 static crawl): 게이트 위상 φ에 동기된 매끈한 주기 CoM ref.
      //   재anchor 온라인plan의 lag 없이 lead가 위상오프셋으로 내장. 스윙발로 φ 동기(drift 없음). WBC가 추종(빠름 확인됨).
      if(getenv("CPLAN")){
        int swf=-1; for(int i=0;i<4;i++) if(tam_c[k][i]==0){ swf=i; break; }
        static const int ord_old[4]={3,1,2,0};   // cs_old q.legs 스윙순서: FR,HR,FL,HL (sub 0..3)
        if(swf>=0){
          int sub=0; for(int j=0;j<4;j++) if(ord_old[j]==swf) sub=j;
          double pdur=0.2, within=tc_clip((t-tam_sw_start[swf])/pdur,0.0,1.0);
          double phi=(sub+within)/4.0;                                   // 게이트 위상 [0,1)
          double Ay=getenv("CP_AY")?atof(getenv("CP_AY")):0.05, dy=getenv("CP_DY")?atof(getenv("CP_DY")):0.12;
          double Ax=getenv("CP_AX")?atof(getenv("CP_AX")):0.04, dx=getenv("CP_DX")?atof(getenv("CP_DX")):0.0;
          double w=2*M_PI;
          double yref=Ay*std::sin(w*(phi+dy));                           // 1×/cycle: +,+,-,-(우발 리프트=+y)
          double vyref=Ay*w*std::cos(w*(phi+dy))/(4*pdur);               // 속도 ff(위상속도 1/(4·pdur))
          double xref=Ax*std::sin(2*w*(phi+dx));                         // 2×/cycle: 앞발 리프트=-x, 뒷발=+x
          q.com_ref[1]=tam_ay+yref; q.com_vel_ref[1]=vyref; q.com_acc_ref[1]=0;
          q.com_ref[0]=tgt_x+xref; q.com_vel_ref[0]=V;
        }
      }
      // ★cold-start settle(정적 crawl): 첫 SETTLE_T초 동안 4접촉 홀드하며 CoM을 첫 지지 centroid로 램프 선이동.
      //   원인=t0 첫 리프트 시 CoM이 (0,0)=지지삼각형 엣지→즉시 tip→단항 불복구. 미리 centroid로 옮겨 리프트 시 이미 삼각형 안.
      double SETTLE_T=getenv("SETTLE_T")?atof(getenv("SETTLE_T")):0.0;
      double sett=t-tam_ent;   // ★절대 진입경과(tam_t는 재앵커로 리셋되므로 사용불가)
      if(SETTLE_T>0 && sett<SETTLE_T){
        st.clear(); swing.clear(); for(int i=0;i<4;i++) st.push_back(i);           // 전부 stance
        double cx=0,cy=0; int nc=0; for(int i=0;i<4;i++) if(tam_c[0][i]!=0){ cx+=q.foot_point(i)[0]; cy+=q.foot_point(i)[1]; nc++; }
        if(nc>0){ cx/=nc; cy/=nc; }
        if(getenv("SETTLE_Y")) cy=atof(getenv("SETTLE_Y"));   // ★고정 목표(WBC lateral authority 검증용)
        double fr=tc_clip(sett/std::max(1e-3,SETTLE_T),0.0,1.0);                     // 램프(0→1)
        double ytgt=getenv("SETTLE_STEP")?cy:fr*cy;                                  // ★STEP=즉시 목표(WBC lateral 최대속도 측정용)
        double _vtau=getenv("SETTLE_VTAU")?atof(getenv("SETTLE_VTAU")):0.0;          // ★속도 feedforward 시상수(0=off). KD 감쇠 상쇄=빠른 lateral 이동
        q.com_ref[1]=ytgt; q.com_acc_ref[1]=0;
        q.com_vel_ref[1]=(_vtau>0)? tc_clip((ytgt-d->subtree_com[1])/_vtau,-0.6,0.6) : 0;   // 남은오차/시상수=필요속도 FF
        q.com_ref[0]=d->subtree_com[0]; q.com_vel_ref[0]=0;                          // x 홀드(제자리 settle)
        q.W_BASE_XY=getenv("SETTLE_WXY")?atof(getenv("SETTLE_WXY")):400.0;           // ★settle 중 lateral CoM 강추종(선이동 확실)
        if(getenv("SETTLE_KD")) q.KD_BASE=atof(getenv("SETTLE_KD"));                 // ★lateral 감쇠 완화(빠른 이동)
        if(getenv("SETTLE_WORI")) q.w_ori=atof(getenv("SETTLE_WORI"));               // ★attitude 완화(body-lean 허용=CoM 빠르게 시프트)
        double Ms=mj_getTotalmass(m); for(int i=0;i<4;i++) lam_use[i]=Vector3d(0,0,Ms*9.81/4);
      }
      if(!q.wbic_track(st,swing,lam_use,wl)) q.wbic_stance();
      armed=false; return;
    }
    bool run_move=(mode=="move")||stop_settle;
    if(!run_move){
      if(mode=="off"){   // ★전원차단: 선 채로 툭 damp(낙하) 대신 바닥(GROUND_Z)까지 완만 PD 하강 후 damp (실로봇=눕고 전원끔)
        double bzo=d->qpos[2];
        was_sit=false; sit_getup_t0=-1; from_sit=false; haunch_ready=false; haunch_fold=0; jphase=-1; armed=false;
        if(off_settled || bzo<=GROUND_Z+0.04){   // 바닥 근처 → damp(전원차단 등가)
          off_settled=true; have_qref=false;
          for(int j=0;j<nu;j++) d->ctrl[j]=tc_clip(-REST_KD*d->qvel[6+j],-q.tau_peak[j],q.tau_peak[j]); q.couple_clamp(d); return; }
        // 아직 높음 → GROUND_Z까지 완만 하강(PD fold): 서 있으면 눕히고, 눕는 중이면 계속
        if(ht_cur>bzo+0.06 || ht_cur<0.06) ht_cur=bzo;   // 진입 시 현재높이 동기화
        ht_cur+=tc_clip(GROUND_Z-ht_cur,-GETUP_RATE*dt,GETUP_RATE*dt);
        if(std::abs(ht_cur-qhome_h)>6e-3){ q.update_stand_qhome(ht_cur); qhome_h=ht_cur; }
        if(!have_qref){ for(int j=0;j<nu;j++) q_ref[j]=d->qpos[7+j]; have_qref=true; }
        for(int j=0;j<nu;j++) q_ref[j]+=tc_clip(q.q_home[j]-q_ref[j],-JOINT_SLEW*dt,JOINT_SLEW*dt);
        for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+GETUP_KP*(q_ref[j]-d->qpos[7+j])-GETUP_KD*d->qvel[6+j];
          d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
        q.couple_clamp(d);
        return; }
      if(mode=="jump"){   // ★★J2 통합 점프: crouch(wbic)→OCP궤적 재생(thrust/flight τ_ff+PD)→touchdown→wbic_stance 착지.
        double bz=d->qpos[2];
        int ncon=0; for(int ci=0;ci<d->ncon;ci++){ int g1=d->contact[ci].geom1,g2=d->contact[ci].geom2;
          for(int fi=0;fi<4;fi++) if(g1==q.fgid[fi]||g2==q.fgid[fi]){ ncon++; break; } }
        if(jphase<0){ jphase=0; jt0=t; jzpk=bz; ht_cur=std::max(JUMP_CROUCH_Z,bz); qhome_h=-1; jump_k=-1; jump_anchored=false;   // ★새 점프=발사 프레임 재캡처
          from_sit=false; haunch_ready=false; haunch_fold=0; have_qref=false;   // ★눕기/앉기서 점프 진입=sit/ground 잔여상태 리셋(wbic_stance 정상 크라우치)
#ifdef HAVE_JUMP_SOLVER
          jlaunched=false; jsolving=false;   // 새 점프=미발사 상태
#endif
        }
        if(jphase==0){   // crouch: wbic_stance로 JUMP_CROUCH_Z 정착 → OCP 궤적 확보 후 jphase1
          ht_cur+=tc_clip(JUMP_CROUCH_Z-ht_cur,-0.4*dt,0.4*dt);
          if(std::abs(ht_cur-qhome_h)>6e-3){ q.update_stand_qhome(ht_cur); qhome_h=ht_cur; }
          q.wbic_stance();
          // ★takeoff 트리거=crouch 명령(ht_cur) 도달 기준(아래=눕기/앉기서 올라와도 트리거. 구 bz≤0.315는 아래서 정착0.32라 갇힘)
          bool jsettled=(std::abs(ht_cur-JUMP_CROUCH_Z)<0.02 && std::abs(d->qvel[2])<0.08 && bz<=JUMP_CROUCH_Z+0.06 && t-jt0>0.45);
#ifdef HAVE_JUMP_SOLVER
          // ★S3-b: solve(aligator, ~331ms)를 별도 스레드로 → 1kHz 루프는 crouch 유지하며 안 멈춤(실기대비).
          //   crouch 정착 시 백그라운드 launch → future ready면 신선 궤적 인계 후 jphase1. 셋업실패=파일 replay.
          if(jsettled && !jlaunched){
            if(jsolver_ready){ double vx=JUMP_VX; jfut=std::async(std::launch::async,[this,vx]{ return jsolver.solve(vx,JUMP_MAXIT,false); }); jsolving=true; jlaunched=true; }
            else { if(jump_N<=0) load_jump("/tmp/jump_traj.txt"); jump_k=0; jump_kt=0; jphase=1; jt0=t; jzpk=bz; jlaunched=true; }
          }
          if(jsolving && jfut.wait_for(std::chrono::seconds(0))==std::future_status::ready){
            JumpTraj J=jfut.get(); jsolving=false;
            if(J.N>0){ jump_N=J.N; jump_dt=J.dt; jump_q=J.q; jump_dq=J.dq; jump_tau=J.tau; jump_ph=J.ph;
              jump_com=J.com; jump_comv=J.comv; jump_acom=J.acom; }
            else if(jump_N<=0) load_jump("/tmp/jump_traj.txt");
            jump_k=0; jump_kt=0; jphase=1; jt0=t; jzpk=bz;
          }
#else
          if(jsettled){ if(jump_N<=0) load_jump("/tmp/jump_traj.txt"); jump_k=0; jump_kt=0; jphase=1; jt0=t; jzpk=bz; }
#endif
          armed=false; return; }
        if(jphase==1){   // ★OCP 궤적 WBIC-추종(push+flight): wbic_jump(CoM ff+접촉GRF로 base 닫음). QP실패 시 τ_ff+PD fallback
          if(jump_N>0 && jump_k<jump_N){
            bool wbc_ok=false;
            if((int)jump_com.size()>jump_k){   // CoM 참조 있으면 WBIC-추종
              if(!jump_anchored){   // ★발사 프레임 캡처(1회): 현재 CoM·yaw를 원점 삼아 궤적을 re-anchor
                jump_launch_com=Vector3d(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]);
                const double* qc=&d->qpos[3];
                jump_launch_yaw=std::atan2(2*(qc[0]*qc[3]+qc[1]*qc[2]),1-2*(qc[2]*qc[2]+qc[3]*qc[3]));
                jump_com0=jump_com[0]; jump_anchored=true;
              }
              double cy=std::cos(jump_launch_yaw), sy=std::sin(jump_launch_yaw);
              auto rotz=[&](const Vector3d& v){ return Vector3d(cy*v[0]-sy*v[1], sy*v[0]+cy*v[1], v[2]); };   // 궤적 프레임→현재 헤딩
              Vector3d com_ref=jump_launch_com+rotz(jump_com[jump_k]-jump_com0);   // 위치=현재 발사점 + 헤딩회전된 궤적변위
              Vector3d comv_ref=rotz(jump_comv[jump_k]), acom_ref=rotz(jump_acom[jump_k]);   // 속도·가속=헤딩회전
              std::vector<int> cts = (jump_ph[jump_k]==1) ? std::vector<int>{} : std::vector<int>{0,1,2,3};  // flight(ph1)=무접촉
              wbc_ok=q.wbic_jump(com_ref,comv_ref,acom_ref,jump_q[jump_k],jump_dq[jump_k],cts);
            }
            if(!wbc_ok){   // WBIC 미가용/실패 → τ_ff + 관절 PD 개루프 재생
              for(int j=0;j<nu;j++){ double tau=jump_tau[jump_k][j]+JUMP_TRAJ_KP*(jump_q[jump_k][j]-d->qpos[7+j])+JUMP_TRAJ_KD*(jump_dq[jump_k][j]-d->qvel[6+j]);
                d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
            }
            jzpk=std::max(jzpk,bz);
            jump_kt+=dt; if(jump_kt>=jump_dt && jump_k<jump_N-1){ jump_kt=0; jump_k++; }
            bool airborne=(jzpk>JUMP_CROUCH_Z+0.10);
            if(jump_ph[jump_k]>=1 && airborne && ncon>=2 && bz<jzpk-0.02){ jphase=3; jt0=t; }   // touchdown → wbic 착지
            return; }
          if(jump_N<=0){   // 궤적 없음: 구 스크립트 스냅(fallback)
            q.update_stand_qhome(0.52);
            for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+JUMP_KP*(q.q_home[j]-d->qpos[7+j])-JUMP_KD*d->qvel[6+j];
              d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
            jzpk=std::max(jzpk,bz);
            if((ncon==0||t-jt0>JUMP_THRUST_T) && jzpk>JUMP_CROUCH_Z+0.10 && ncon>=2 && bz<jzpk-0.02){ jphase=3; jt0=t; }
            return; }
          jphase=3; jt0=t; return; }   // 궤적 끝 → 착지
        // jphase==3: land — 짧은 저-PD 흡수 → wbic_stance 회복 → 서기 인계
        if(t-jt0<0.06){ for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+45*(q.q_home[j]-d->qpos[7+j])-6*d->qvel[6+j];
          d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); } return; }
        ht_cur=std::max(0.30,bz); qhome_h=-1; q.wbic_stance();
        if(t-jt0>0.6){ mode="stand_up"; jphase=-1; }
        return; }
      if(mode=="sit"){   // ★개-앉기(haunch sit): 저crouch로 하강 → 정착 후 crouch→haunch(뒷다리 접어 발링크 바닥밀착) fold + nose-up. 기립=from_sit
        was_sit=false; sit_getup_t0=-1; from_sit=true;
        double bz=d->qpos[2];
        if(!sit_init){ sit_from_below=(bz<SIT_Z-0.02); sit_init=true; }     // ★진입방향 1회 판정: 눕기 등 SIT_Z(0.32)보다 낮게 진입=from_below
        if(sit_from_below){   // ★아래(눕기)서 진입: 0.32 crouch로 안 올림. 현재 자세를 blend 시작점으로 캡처 + haunch 즉시 계산 → settled 대기 없이 직행 morph(base는 다리 접힘 기하로 ~0.25까지만 자연 상승, "일어섰다 다시앉기" 오버슈트 제거)
          if(!haunch_ready){
            for(int j=0;j<nu;j++) q_crouch[j]=d->qpos[7+j];                 // 시작점=현재 눕기 자세
            com_crouch=Vector3d(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]);
            q.haunch_sit_home(HAUNCH_Z,HAUNCH_PITCH); q_haunch=q.q_home; com_haunch=q.com_ref; haunch_ready=true;
            ht_cur=std::max(bz,0.15); qhome_h=-1;                           // ht_cur=현재높이 동기(위로 램프 안 함)
          }
        } else {              // ★위(서기)서 진입: 기존 로직 — 0.32로 하강 후 fold
          if(bz<GETUP_TRIG && ht_cur>GETUP_DONE) ht_cur=std::max(0.12,bz); // 쓰러짐서 낮음→동기화
          ht_cur+=tc_clip(SIT_Z-ht_cur,-GETUP_RATE*dt,GETUP_RATE*dt);      // 천천히 SIT_Z로
          if(std::abs(ht_cur-qhome_h)>6e-3){ q.update_stand_qhome(ht_cur); qhome_h=ht_cur; q_crouch=q.q_home; com_crouch=q.com_ref; }  // crouch 원본(blend용)
          bool settled=(std::abs(ht_cur-SIT_Z)<0.03 && bz>0.29);          // SIT_Z 정착 후에만 fold+nose-up
          if(settled && !haunch_ready){ q.haunch_sit_home(HAUNCH_Z,HAUNCH_PITCH); q_haunch=q.q_home; com_haunch=q.com_ref; haunch_ready=true; }  // ★haunch 1회 계산(nose-up 베이킹→PD홀드가 재현)
        }
        double tf=haunch_ready?1.0:0.0;                                    // ★접힘 시작 후 계속 유지(자세 홀드). settled는 최초 트리거만 담당(앉으면 bz<0.29로 settled 풀려 되펴지던 버그 수정)
        double frate=HAUNCH_FOLD_RATE*std::max(0.18,1.0-haunch_fold);       // ★ease-out: fold 끝(→1)에서 느려져 엉덩이 살포시 착지(엉덩방아 완화). 시작은 빠름
        if(sit_from_below) frate=HAUNCH_FOLD_RATE*SIT_BELOW_SPEED*std::max(0.5,1.0-haunch_fold);  // ★눕기서 진입=엉덩이 이미 낮음→빠른 fold + 얕은 ease-out(tail 완만 불필요, 기어가는 꼬리 제거)
        haunch_fold+=tc_clip(tf-haunch_fold,-frate*dt,frate*dt);
        if(haunch_ready){ double ff=std::min(1.0,haunch_fold*FRONT_PULL_SPEED); int jfront=q.legqp[2][0]-7;  // ★앞다리(FL_hip 이후 관절)=더 빠른 fold로 조기 끌어당김. 뒤(엉덩방아)는 haunch_fold 원래 pace
          for(int j=0;j<nu;j++){ double f=(j>=jfront)?ff:haunch_fold; q.q_home[j]=(1-f)*q_crouch[j]+f*q_haunch[j]; }
                          q.com_ref=(1-haunch_fold)*com_crouch+haunch_fold*com_haunch; }
        if(haunch_ready){ double froll=tc_clip((haunch_fold-0.65)/0.35,0.0,1.0);   // ★뒷발 굴려 착지: 착지중(fold<0.65)=LAND각(발 안 부딪힘) → 바닥 닿은 뒤(fold→1)=CONTACT각(발바닥 밀착)
          double footang=(1.0-froll)*q.HAUNCH_FOOT_LAND+froll*q.HAUNCH_FOOT;
          for(int i=0;i<2;i++) q.q_home[q.legqp[i][3]-7]=footang; }
        q.sit_pitch+=tc_clip(0.0-q.sit_pitch,-1.2*dt,1.2*dt);              // nose-up은 q_home 베이킹(강성 PD홀드)
        double kp=(ht_cur<0.31||bz<0.295)?GETUP_KP:SIT_KP;                 // 하강복구=완만 / 개-앉기 정착=강성(발링크 평평 강제)
        // ★앉기 지연 강건성(2026-07-15): 드리프트는 순전히 지연(≥3ms)·노이즈 무관. PD홀드는 능동 base롤/yaw규제 없어 3ms서 서서히 롤(>4s). wbic는 잡지만 깊은 nose-up 상실 → 사용자결정=깊은 haunch PD 유지·지연예산 ≤2ms(각), 빠른 sit→stand는 3ms도 무해.
        if(ht_cur<0.31 || bz<0.295 || haunch_ready){                       // ★개-앉기 정착(또는 저자세 복구)=중력보상 PD 홀드. 사용자=기립불요·자세우선 → 강성 PD로 접힘+발링크 바닥밀착 강제
          if(!have_qref){ for(int j=0;j<nu;j++) q_ref[j]=d->qpos[7+j]; have_qref=true; }
          for(int j=0;j<nu;j++) q_ref[j]+=tc_clip(q.q_home[j]-q_ref[j],-JOINT_SLEW*dt,JOINT_SLEW*dt);
          for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+kp*(q_ref[j]-d->qpos[7+j])-GETUP_KD*d->qvel[6+j];
            d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
          armed=false; return; }
        have_qref=false; q.wbic_stance(); armed=false; return; }           // 정착 전(하강 crouch)=wbic 균형
      have_qsit=false;
      double bz=d->qpos[2];
      // ★앉기→서기 스크립트 기립: 앞다리 굽혀 앞발 들기(폴볼트 차단) + 뒷다리 박차 extend → 몸 올라오면 정상 stance로 인계
      if(was_sit && mode=="stand_up"){
        // ★정적 front-retraction 기립: Phase1 앞발 들어 올림(폴볼트 차단) → Phase2 앞발 몸밑 재배치(q_home) + 뒷다리 세움
        //   → 수평 저crouch 도달 후 정상 getup(off→서기 검증됨)으로 인계. 뒷발 planted(얕은sit) 전제.
        if(sit_getup_t0<0){ sit_getup_t0=t; for(int j=0;j<nu;j++) q_ref[j]=d->qpos[7+j]; have_qref=true; q.update_stand_qhome(SGU_GATHER_Z); qhome_h=SGU_GATHER_Z; ht_cur=SGU_GATHER_Z; }
        double te=t-sit_getup_t0;
        for(int i=0;i<4;i++){ bool fr=(std::string(q.legs[i])=="FL"||std::string(q.legs[i])=="FR");
          double th,cf,ft,hp;
          if(fr && te<SGU_KICK_T){ hp=0; th=SGU_FB_THIGH; cf=SGU_FB_CALF; ft=-0.70; }   // Phase1: 앞다리 굽혀 발 들기(지면서 뗌)
          else { hp=q.q_home[q.legqp[i][0]-7]; th=q.q_home[q.legqp[i][1]-7];             // Phase2: 앞다리→몸밑 착지, 뒷다리→저crouch
                 cf=q.q_home[q.legqp[i][2]-7]; ft=q.leg_dof[i]==4?q.q_home[q.legqp[i][3]-7]:0; }
          double tar[4]={hp,th,cf,ft};
          for(int cc=0;cc<q.leg_dof[i];cc++){ int j=q.legqp[i][cc]-7;
            q_ref[j]+=tc_clip(tar[cc]-q_ref[j],-SGU_SLEW*dt,SGU_SLEW*dt); } }
        for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+SGU_KP*(q_ref[j]-d->qpos[7+j])-GETUP_KD*d->qvel[6+j];
          d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
        double jerr=0; for(int j=0;j<nu;j++) jerr+=std::abs(q.q_home[j]-d->qpos[7+j]); jerr/=nu;
        if(te>SGU_KICK_T && tiltdeg()<SGU_DONE_TILT && jerr<0.4){ was_sit=false; sit_getup_t0=-1; }  // 수평 저crouch 도달→정상 getup 인계
        armed=false; return;
      }
      // ★★강건 기립(앉기→눕기→서기 라우팅): haunch 직접상승은 측방 발산(불안정)이라, 먼저 대칭 저크라우치(눕기자세)로
      //   언폴드(측방 안정 확인됨)한 뒤 검증된 눕기→서기 상승에 인계. 구 gather(gen_getup) 대체.
      if(mode=="stand_up" && haunch_ready){
        double bzg=d->qpos[2]; double LIEZ=GROUND_LIE_Z;                      // 언폴드 목표=대칭 저크라우치(순수 눕기와 동일 자세)
        if(ht_cur>LIEZ) ht_cur=std::max(LIEZ, ht_cur-GETUP_RATE*dt); else ht_cur=LIEZ;
        q.update_stand_qhome(ht_cur);                                         // 대칭 crouch config(haunch 아님)
        for(int i=0;i<4;i++) if(q.leg_dof[i]==4){                             // 눕기와 동일 발/앞다리 오프셋(안정 저자세)
          q.q_home[q.legqp[i][3]-7]=(i<2)?GROUND_REAR_FOOT:GROUND_FRONT_FOOT;
          if(i>=2){ q.q_home[q.legqp[i][1]-7]+=GROUND_FRONT_THIGH; q.q_home[q.legqp[i][2]-7]+=GROUND_FRONT_CALF; } }
        if(!have_qref){ for(int j=0;j<nu;j++) q_ref[j]=d->qpos[7+j]; have_qref=true; }
        for(int j=0;j<nu;j++) q_ref[j]+=tc_clip(q.q_home[j]-q_ref[j],-JOINT_SLEW*dt,JOINT_SLEW*dt);
        for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+GETUP_KP*(q_ref[j]-d->qpos[7+j])-GETUP_KD*d->qvel[6+j];
          d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
        if(std::abs(ht_cur-LIEZ)<0.02 && tiltdeg()<11) lie_settle+=dt; else lie_settle=0;   // ★정착 대기: 순수 눕기 자세(tilt~9)처럼 완전 정착(dwell)해야 인계(marginal 인계→지연하 전복 방지)
        if(lie_settle>0.5){                                                    // 저크라우치 안정 정착 → haunch 해제, 검증된 눕기→서기 상승 인계
          haunch_ready=false; haunch_fold=0; getup_k=-1; from_sit=false; lie_settle=0; ht_cur=bzg; qhome_h=-1; have_qref=false; }
        else { q.sit_pitch+=tc_clip(0.0-q.sit_pitch,-1.2*dt,1.2*dt); armed=false; return; }
      }
      if(bz<GETUP_TRIG && ht_cur>GETUP_DONE) ht_cur=std::max(0.12,bz);      // 쓰러짐/off로 낮음→동기화
      if(mode=="stand_down" && ht_cur>bz) ht_cur=std::max(GROUND_Z,bz);    // ★눕기=현재높이서 하강(서기 안 거치고 그대로 눕기)
      if(mode=="stand_down"){ from_sit=false; haunch_ready=false; haunch_fold=0; }  // ★눕기=저자세 leg-fold(PD)로 낮게 눕힘. from_sit=false→아래 fold 경로 사용(wbic 0.29 아님)
      if(mode=="stand_up" && bz>0.47) from_sit=false;                        // 서기 완료→해제
      if(!stand_set){ stand_ax=d->subtree_com[0]; stand_ay=d->subtree_com[1]; stand_set=true; }  // ★서기 진입=현재 위치 캡처(홈 x=0으로 안 빨려가게)
      double tgt=(mode=="stand_down")?GROUND_LIE_Z:body_h;                  // ★눕기=저자세(0.22) leg-fold / 서기=슬라이더
      bool low=(ht_cur<GETUP_DONE)||(tgt<GETUP_DONE); double rate=low?GETUP_RATE:HRATE;
      ht_cur+=tc_clip(tgt-ht_cur,-rate*dt,rate*dt);
      if(std::abs(ht_cur-qhome_h)>6e-3 || mode=="stand_down"){ q.update_stand_qhome(ht_cur); qhome_h=ht_cur; q_crouch=q.q_home; com_crouch=q.com_ref; }  // ★눕기=매틱 재계산(LUT, 저비용): 아래 앞다리 오프셋을 깨끗한 base q_home에 얹어 += 누적/슬라이더 무반응 방지(height 고정 시 재계산 스킵되던 버그)
      if(mode=="stand_down") for(int i=0;i<4;i++) if(q.leg_dof[i]==4){   // ★앞뒤 발목 접기 + 앞다리 fold(GUI 슬라이더로 CoM균형·수평·무슬라이드 조각)
        q.q_home[q.legqp[i][3]-7]=(i<2)?GROUND_REAR_FOOT:GROUND_FRONT_FOOT;   // 발목: 뒤/앞
        if(i>=2){ q.q_home[q.legqp[i][1]-7]+=GROUND_FRONT_THIGH; q.q_home[q.legqp[i][2]-7]+=GROUND_FRONT_CALF; } }  // 앞다리 thigh/calf 오프셋(접기)
      if(haunch_ready && mode=="stand_up"){   // ★개-앉기서 기립: 높이-스케줄 언폴드(HAUNCH_Z서 fold=1 → UNFOLD_Z서 0). 몸 오르며 뒷다리 펴짐→q_home 점프 없이 인계
        double target_fold=tc_clip((HAUNCH_UNFOLD_Z-ht_cur)/(HAUNCH_UNFOLD_Z-HAUNCH_Z),0.0,1.0);
        haunch_fold+=tc_clip(target_fold-haunch_fold,-HAUNCH_FOLD_RATE*dt,HAUNCH_FOLD_RATE*dt);
        for(int j=0;j<nu;j++) q.q_home[j]=(1-haunch_fold)*q_crouch[j]+haunch_fold*q_haunch[j];
        q.com_ref=(1-haunch_fold)*com_crouch+haunch_fold*com_haunch;
        if(ht_cur>=HAUNCH_UNFOLD_Z-1e-3 && haunch_fold<0.02){ haunch_ready=false; haunch_fold=0; }   // 언폴드 완료→순수 crouch
      }
      double jerr=0; for(int j=0;j<nu;j++) jerr+=std::abs(q.q_home[j]-d->qpos[7+j]); jerr/=nu;
      // ★눕기=wbic_stance로 저크라우치(0.29) 능동홀드(슬라이드·붕괴 없음). 진짜 belly-flat은 다중접촉문제(haunch-getup과 동일)라 보류.
      // (구 damp 붕괴는 저크라우치서 발 미끄러져 ~0.4m 슬라이드 → 제거)
      double foldZ=GETUP_DONE;                                             // ★눕기·getup 모두 저-PD fold(ht_cur<0.40): 눕기=저자세 leg-fold 홀드, getup=상승 fold. (구 눕기=wbic0.29는 발목으로 못 낮춰 leg-fold로 전환)
      if(ht_cur<foldZ && !from_sit){                                        // 낮은자세=수평 PD fold(눕기/getup). ★from_sit(crouch-sit≥0.29)=wbic_stance로 매끈 기립
        if(!have_qref){ for(int j=0;j<nu;j++) q_ref[j]=d->qpos[7+j]; have_qref=true; }
        for(int j=0;j<nu;j++) q_ref[j]+=tc_clip(q.q_home[j]-q_ref[j],-JOINT_SLEW*dt,JOINT_SLEW*dt);
        for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+GETUP_KP*(q_ref[j]-d->qpos[7+j])-GETUP_KD*d->qvel[6+j];
          d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
        armed=false; return; }
      if(!haunch_ready){
        if(from_sit){ q.com_ref[0]=d->subtree_com[0]; q.com_ref[1]=d->subtree_com[1]; }  // ★기립 rise=현재 CoM 상대추종(절대 xy 비의존→추정 드리프트 무관)
        else { q.com_ref[0]=stand_ax; q.com_ref[1]=stand_ay; }              // 정상 서기=멈춘 위치 절대 홀드
      }
      have_qref=false; q.wbic_stance(); armed=false; return;                // 서기(높이충분)=wbic_stance
    }
    have_qref=false; stand_set=false;   // move → fold·서기앵커 리셋(다음 서기서 새 위치 캡처)
    auto quat_yaw=[&](){ double*qq=&d->qpos[3]; return std::atan2(2*(qq[0]*qq[3]+qq[1]*qq[2]),1-2*(qq[2]*qq[2]+qq[3]*qq[3])); };
    if(t < settle_until){ q.wbic_stance(); return; }
    if(!armed){ armed=true; t0=t; yaw_ref=0;
      for(int i=0;i<4;i++){ nominal[i]=q.foot_point(i); liftoff[i]=q.foot_point(i); hip_off[i]=q.foot_hip_off[i]; gz[i]=q.foot_gz0[i]; }
      x_ref.setZero(); x_ref[5]=d->subtree_com[2]; x_ref[12]=-9.81; com_h0=d->subtree_com[2]; }   // com_h0=평지 위 CoM 명목높이
    double tg=t-t0; bool go=tg>TC_WARMUP;
    double vt=go?V:0.0, vyt=go?VY:0.0, wt=go?WZ:0.0;
    if(stop_settle){ vt=vyt=wt=0.0; }   // ★정지 감속: 명령 0 → MPC/raibert가 속도 죽이며 발을 CoM 밑으로 재중심
    double acc=stop_settle?1.6:TC_ACC;   // ★달리다 서기: 감속을 빠르게(1.6 m/s²)해 코스팅 단축(구 TC_ACC 0.6=3m 코스팅→불안)
    Vs+=tc_clip(vt-Vs,-acc*dt,acc*dt); Vys+=tc_clip(vyt-Vys,-acc*dt,acc*dt); Ws+=tc_clip(wt-Ws,-2.0*dt,2.0*dt);
    double stt=go?steer:0.0; Ss_steer+=tc_clip(stt-Ss_steer,-0.8*dt,0.8*dt);   // 조향각 스무딩[rad/s]
    double Veff=Vs,Vyeff=Vys,Weff=Ws; Veff_dbg=Veff;
    double steer_wz = tc_clip(Veff*std::tan(tc_clip(Ss_steer,-1.2,1.2))/wheelbase, -0.9, 0.9);  // ★자동차식 조향(Ackermann) 성분. yaw rate 캡0.9(understeer). V=0=무효, WZ스핀과 공존
    Weff += steer_wz;
    q.waist_ref=tc_clip(waist_steer*steer_wz,-waist_cap,waist_cap);    // ★허리 lean=조향(핸들)만 구동. 우스틱 WZ 선회는 순수 다리선회(허리 중립홀드). 핸들 0이면 허리 중립
    double spd=std::abs(Veff);   // ★전진속도만 whip 트리거(좌우이동은 whip 유발 안함). 구 hypot은 측방서도 whip 켜짐
    if(gait_type=="walk"){   // ★walk: 체크박스(auto_whip)로 whip on/off. on=슬라이더 강도 직접적용 / off=매끈(whip_hi)
      if(auto_whip){ q.swing_w_f=whip_lo_f; q.swing_w_r=whip_lo_r; } else { q.swing_w_f=whip_hi; q.swing_w_r=whip_hi; } }
    else if(auto_whip){   // ★trot: 속도↑ → whip↑. swing_w 를 whip_hi(저속,매끈)→whip_lo(고속,슬라이더값) 선형보간
      double s=tc_clip((spd-whip_v0)/(whip_v1-whip_v0),0.0,1.0);
      q.swing_w_f=whip_hi+s*(whip_lo_f-whip_hi); q.swing_w_r=whip_hi+s*(whip_lo_r-whip_hi);
    } else { q.swing_w_f=whip_lo_f; q.swing_w_r=whip_lo_r; }   // 수동=슬라이더값 상수
    double lat=tc_clip(std::abs(Vyeff)/0.35,0.0,1.0);   // ★좌우이동 시 whip 억제: 측방 스윙 flail 완화(swing_w→매끈 페이드). walk 측방붕괴는 게이트 고유(whip 무관)
    q.swing_w_f=q.swing_w_f+lat*(whip_hi-q.swing_w_f); q.swing_w_r=q.swing_w_r+lat*(whip_hi-q.swing_w_r);
    double yaw_m=quat_yaw();
    if(std::abs(Weff)>0.02){ yaw_ref=tc_clip(yaw_ref+Weff*dt,yaw_m-0.3,yaw_m+0.3); yaw_hold_set=false; }
    else { if(!yaw_hold_set){ yaw_hold=yaw_m; yaw_hold_set=true; } yaw_ref=yaw_hold; }
    double cy=std::cos(yaw_m), sy=std::sin(yaw_m);
    if(getenv("TAMOLS_INJECT") && getenv("GAP_X0")){ double g0=atof(getenv("GAP_X0")), g1=atof(getenv("GAP_X1")), bx=d->qpos[0];
      if(bx>g0-0.30 && bx<g1+0.15){ double sf=getenv("GAP_SLOW")?atof(getenv("GAP_SLOW")):0.4; Veff*=sf; } }  // ★base 협조: 갭 위서 감속→CoM이 지지폴리곤(짧아진 straddle) 안에 머묾
    bool gap_near_terr=false;   // ★heightmap 갭 근접(감속+CoM shift 공용 게이트)
    if(getenv("TAMOLS_INJECT") && getenv("TAMOLS_TERRAIN")){   // ★heightmap 구동 갭 감속(명시적 GAP_X0 없이): base 전방 void(tmap.valid=false) 감지→준정적. hsteps=void 없어 무영향
      double bx=d->qpos[0], by=d->qpos[1];
      for(double a=0.05; a<=0.50; a+=0.075){ if(!tmap.valid(bx+cy*a,by+sy*a)){ gap_near_terr=true; break; } }   // 전방 0.05~0.50m 스캔
      if(gap_near_terr){ double sf=getenv("GAP_SLOW")?atof(getenv("GAP_SLOW")):0.3; Veff*=sf; } }   // 갭 스텝을 준정적으로(yaw/횡 안정)
    double vx_w=Veff*cy-Vyeff*sy, vy_w=Veff*sy+Vyeff*cy;
    // ★위치홀드: 전진/측방명령 0이면 base 위치 앵커링. SPIN_HOLD=제자리선회(V=0,WZ≠0)서도 유지 → 허리조향 표류 상쇄(베이스 기준 wz 선회)
    bool ph_turn = SPIN_HOLD ? true : (std::abs(Weff)<0.05);
    if(POS_HOLD && std::abs(Veff)<0.03 && std::abs(Vyeff)<0.03 && ph_turn){
      if(!pos_hold_set){ phx=d->qpos[0]; phy=d->qpos[1]; pos_hold_set=true; }
      vx_w+=tc_clip(-0.6*(d->qpos[0]-phx),-0.15,0.15); vy_w+=tc_clip(-0.6*(d->qpos[1]-phy),-0.15,0.15);
    } else pos_hold_set=false;
    x_ref[2]=yaw_ref; x_ref[8]=Weff; x_ref[9]=vx_w; x_ref[10]=vy_w;
    // ★perceptive 몸통높이(Python 동일): 4-hip 평균 지형높이=_body_terr → MPC(x_ref[5])·WBIC z-task(quad_control:354) 양쪽 일관 적용. 평지=0(무변화)
    // ★perceptive 몸통높이: base 1점 지형높이+슬루 → MPC x_ref[5]만(WBIC z-task엔 미공급=_body_terr 0).
    //   실측(course): WBIC 공급시 tilt 3.7→4.3°·4힙평균 6.1° 로 오히려 나빠짐 → 보행중 몸통z는 MPC가 지배, WBIC 지형공급은 진동만 추가.
    // ★body_h(서기+보행 통합 높이): 보행 중에도 body_h를 부드럽게 추종 → update_stand_qhome로 com_ref/q_home/com_h0 갱신(MPC·WBIC 정합). 서기와 동일 슬라이더.
    //   ★★버그수정: body_h와 현 높이 차이로 게이팅(구 hr-qhome_h는 1틱램프량이라 3e-3 못 넘어 영영 미호출). 램프 중 매틱 update.
    if(std::abs(body_h-qhome_h)>2e-3){ qhome_h+=tc_clip(body_h-qhome_h,-0.3*dt,0.3*dt); q.update_stand_qhome(qhome_h); com_h0=q.com_ref[2]; }
    // ★perceptive 몸통높이=base 1점 지형높이+슬루(MPC x_ref[5]만). ★4-hip 평균은 갭서 hip이 발판↔갭 오가며 base 목표 출렁→불안정(전속도 실패)이라 미채택. 갭서 base가 발판만큼 안 올라오는 건 안정적 절충.
    // ★몸통높이=base 1점 mj_ray 지형높이+슬루. ★검증(2026-07-15): tmap support-aware(footprint MAX)·4-hip 평균·슬루감속 전부 갭 크로싱 전복
    //   (base를 미리/과하게 올리거나 지연→발 지지 타이밍과 어긋남). base-1점만 크로싱 성공=base가 발판 위일 때만 point-wise 상승이 지지와 동기.
    //   → 갭서 base z 솟구침/출렁은 크로싱에 필요한 동기화(제거 불가). tmap.supportZ는 보존(다른 용도 참조).
    double bt=0.0; if(perceptive){ double tz=q.terrain_z(d->qpos[0],d->qpos[1]); if(tz>-50.0) bt=tz; }
    _bterr_s+=tc_clip(bt-_bterr_s,-0.5*dt,0.5*dt); q._body_terr=0.0; x_ref[5]=com_h0+_bterr_s;
    std::vector<int> st; std::map<int,std::pair<Vector3d,Vector3d>> swing;
    bool taminj=getenv("TAMOLS_INJECT");   // ★A gait 클럭(연속·swing 확실)에 TAMOLS 발판만 주입=타이밍 유지+계획 협조
    if(taminj){ double RDT=getenv("REPLAN_DT")?atof(getenv("REPLAN_DT")):0.4; tam_inj=true;
      if(tam_ol_last<0||t-tam_ol_last>=RDT){ tamols_online_replan(d); tam_ol_last=t; tam_ax=d->qpos[0]; tam_ay=d->qpos[1]; } }
    if(taminj && getenv("TAM_BASE") && tam_N>0){   // ★TAMOLS 계획 base z 주입=발판+base 완전협조(연속지형 tilt25→1.4)
      double _bz=tam_s[0][2]; _tambz_s+=tc_clip(_bz-_tambz_s,-0.5*dt,0.5*dt);   // 항상 슬루(점프 방지)
      if(!gap_near_terr) x_ref[5]=_tambz_s; }   // ★갭 위선 override 안 함(void로 base z 끌어내림→A perceptive z 유지)
    for(int i=0;i<4;i++){ bool sch; double sp; gait(i,tg,sch,sp);
      if(sch){ st.push_back(i); have_prev[i]=false; } else { if(sp<0.03) liftoff[i]=q.foot_point(i); } }
    // ★★CoM shift(정적 crawl 원리): 갭 크로싱 시 CoM 지면투영을 지지발 centroid로 이동=지지삼각형 안 유지(횡드리프트 방지)
    // ⚠️crude=속도참조 bias(MPC 제약 아님)라 위상 미동기→gait 공진(hsteps 회귀). opt-in(COM_SHIFT 명시 시만). 근본=위상동기 위치참조 or MPC 지지폴리곤 제약
    if(taminj && getenv("TAMOLS_TERRAIN") && getenv("COM_SHIFT") && gap_near_terr && st.size()>=2){
      Vector2d cen(0,0); for(int i: st) cen+=Vector2d(q.foot_point(i)[0],q.foot_point(i)[1]);
      cen/=(double)st.size();
      Vector2d berr=cen-Vector2d(d->qpos[0],d->qpos[1]);
      double kc=getenv("COM_SHIFT")?atof(getenv("COM_SHIFT")):1.2;
      x_ref[10]+=tc_clip(kc*berr[1],-0.25,0.25);        // vy: 횡 centering(주효과=CoM을 지지 안으로)
      x_ref[9] +=tc_clip(0.5*kc*berr[0],-0.10,0.10); }  // vx: 약한 전후 centering(전진 방해 최소)
    // ★★옵션2: MPC 지지폴리곤 hard constraint(CoM px,py∈stance 폴리곤). gap 근처만
    q.mpc_support_con = taminj && getenv("TAMOLS_TERRAIN") && getenv("SUPPORT_CON") && gap_near_terr;
    q.mpc_support_margin = getenv("SUPPORT_MARGIN")?atof(getenv("SUPPORT_MARGIN")):0.0;
    // ★★옵션1: 위상동기 CoM 위치참조 — x_ref[3,4]=지지 centroid(매틱 재계산=위상정합), Qdiag[3,4] 위치가중 on(위치참조=감쇠·비공진, 속도bias와 다름)
    q.mpc.Qdiag[3]=0; q.mpc.Qdiag[4]=0;   // 기본 off(속도제어 유지)
    if(taminj && getenv("TAMOLS_TERRAIN") && getenv("COM_REF") && gap_near_terr && st.size()>=2){
      Vector2d cen(0,0); for(int i: st) cen+=Vector2d(q.foot_point(i)[0],q.foot_point(i)[1]); cen/=(double)st.size();
      double qp=atof(getenv("COM_REF")), lead=getenv("COM_LEAD")?atof(getenv("COM_LEAD")):0.15;
      x_ref[3]=d->qpos[0]+lead*cy; x_ref[4]=d->qpos[1]+lead*sy+ (cen[1]-d->qpos[1]);   // ★X=전방 lead(전진+fore-aft 감쇠), Y=지지 centroid(횡 centering). 위치추종=감쇠·비공진
      q.mpc.Qdiag[3]=qp; q.mpc.Qdiag[4]=qp; }
    std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
    Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
    Map<VectorXd> qv(d->qvel,nv); Vector3d vcom=Jc*qv;
    Vector2d v_des(vx_w,vy_w), v_fb=vcom.head(2);
    if(ALIP){ mj_subtreeVel(m,d); Vector3d L(d->subtree_angmom[0],d->subtree_angmom[1],d->subtree_angmom[2]);
      double H=std::max(0.1,d->subtree_com[2]); v_fb+=Vector2d(L[1],-L[0])/(q.mpc.TOTAL_MASS*H); }
    Vector2d rai; for(int k=0;k<2;k++) rai[k]=tc_clip(raibert_k*gp_Tst*v_des[k]+TC_KCAP*(v_fb[k]-v_des[k]),-TC_RAICLIP,TC_RAICLIP);
    Matrix2d Rw; Rw<<cy,-sy,sy,cy; double sh=step_h*(0.2+0.8*std::min(1.0,tg/TC_WARMUP));
    double wa=(q.waist_idx>=0)?d->qpos[7+q.waist_idx]:0.0;         // ★허리각(앞몸통이 base대비 꺾인 정도)
    double cyf=std::cos(yaw_m+wa), syf=std::sin(yaw_m+wa); Matrix2d Rwf; Rwf<<cyf,-syf,syf,cyf;  // 앞몸통 방향(base+허리 yaw)
    for(int i=0;i<4;i++){ bool sch; double s_; gait(i,tg,sch,s_); if(sch) continue;
      Vector2d hip_xy(d->xpos[q.hip_bid[i]*3],d->xpos[q.hip_bid[i]*3+1]);
      Vector2d r_xy=hip_xy-Vector2d(d->qpos[0],d->qpos[1]);        // 몸중심→hip
      Vector2d tw=Weff*gp_Tst*Vector2d(-r_xy[1],r_xy[0]);          // ★선회 접선 발배치(yaw) — 없으면 회전시 표류·붕괴
      bool frontleg=(std::string(q.legs[i])=="FL"||std::string(q.legs[i])=="FR");  // ★앞다리=앞몸통방향(허리반영)
      Vector2d pe_xy=hip_xy+(frontleg?Rwf:Rw)*hip_off[i]+rai+tw;
      if(taminj && tam_N>0){ int mp[4]={2,3,0,1};   // ★A(HL,HR,FL,FR)↔TAMOLS(FL,FR,RL,RR) 매핑
        double dx=tam_fh[mp[i]][0]-tam_ol.prm.hip_offsets(mp[i],0), dy=tam_fh[mp[i]][1]-tam_ol.prm.hip_offsets(mp[i],1);  // TAMOLS 계획 편차(명목 대비, 평지=0)
        double dyc=getenv("INJ_DYCAP")?atof(getenv("INJ_DYCAP")):0.0; dy=tc_clip(dy,-dyc,dyc);   // ★Y 편차 캡 기본0: TAMOLS는 y-대칭 강제 없어 비대칭 dy→횡/yaw 드리프트(검증: DYCAP0=언덕 falls=0 완주, 0.05=조기전복). 명목대칭 유지, dx(전진)만 주입
        pe_xy+=Vector2d(cy*dx-sy*dy, sy*dx+cy*dy); }   // ★A 발판(검증됨)에 TAMOLS 편차만 더함=A 타이밍·안정 유지
      if(taminj && getenv("GAP_X0")){ double g0=atof(getenv("GAP_X0")), g1=atof(getenv("GAP_X1")), mg=0.05;   // ★per-foot gap 회피(A gait 연속보행 정합)
        if(pe_xy[0]>g0-mg && pe_xy[0]<g1+mg) pe_xy[0]=(pe_xy[0]<0.5*(g0+g1))? g0-mg : g1+mg; }   // 갭에 빠질 발→가까운 안전 edge(앞/뒤)로
      if(foot_nudge){   // ★③ 발판 선택(밸런스-인지): 스윙시작 1회 도달반경내 footScore·밸런스 저울질로 발판 선택→오프셋 홀드(밸런스 피드백 보존)
        if(!have_prev[i]){ double bx,by;
          if(tmap.selectFoot(pe_xy[0],pe_xy[1],nudge_r,nudge_wbal,bx,by)){ Vector2d off(bx-pe_xy[0],by-pe_xy[1]);
            double n=off.norm(); if(n>nudge_r) off*=nudge_r/n; nudge_off[i]=off; }   // 반경 캡
          else nudge_off[i]=Vector2d::Zero(); }                  // 반경내 유효셀 없음(큰 갭)=선택 안 함(폴백)
        pe_xy+=nudge_off[i]; }
      double land_z=gz[i];                                          // ★기본=평지 참조(foot_gz0)
      if(perceptive){ double tz=q.terrain_z(pe_xy[0],pe_xy[1]); if(tz>-50.0) land_z=gz[i]+tz; }  // ★착지 z=평지gz+지형높이(Python 동일). 평지 tz=0=무변화
      Vector3d p_end(pe_xy[0],pe_xy[1],land_z);
      double dzl=p_end[2]-liftoff[i][2]; Vector3d bvel(vcom[0],vcom[1],0.0);
      double sh_i=sh; if(perceptive && dzl>0.005) sh_i=sh+dzl+PCV_CLR;   // ★상향 스텝: apex를 착지높이+여유 위로(라이저 헛디딤 방지)
      Vector3d p_tgt=tc_swing_foot(s_,liftoff[i],p_end,bvel,sh_i,gp_Tsw,gp_Tst);
      p_tgt[2]+=dzl*(10*std::pow(s_,3)-15*std::pow(s_,4)+6*std::pow(s_,5));
      Vector3d v_tgt=Vector3d::Zero();
      if(have_prev[i]) for(int c=0;c<3;c++) v_tgt[c]=tc_clip((p_tgt[c]-ptgt_prev[i][c])/dt,-1.0,1.0);
      ptgt_prev[i]=p_tgt; have_prev[i]=true; swing[i]={p_tgt,v_tgt}; }
    double dmpc=t-mpc_t;
    // ★RSL_TRACK: injection을 RSL-WBC로 추종(SRBD MPC 대체). A gait의 깨끗한 참조(Raibert 발판+측방 capture)라
    //   pure-online의 재앵커 오염/nominal 발판 없음 → RSL 붕괴(yaw/lateral) 회피 기대. injection=속도제어라 xy 위치추종 off·속도 ff로 전진.
    bool rsl_inj=getenv("RSL_TRACK");
    if(rsl_inj){
      q.W_BASE_XY=getenv("W_BASE_XY")?atof(getenv("W_BASE_XY")):40.0;   // ★40=A Raibert 발판이 전진 주도(80은 base task가 발판과 싸워 backward drift)
      q.w_ori   =getenv("W_ORI")?atof(getenv("W_ORI")):200.0;   // ★자세 강홀드(축소지지 레벨링)
      // ★기본=KP_BASE 0(위치추종 off·안정, 전방은 A Raibert 발판 담당). 전방 progression drift는 KP_BASE>0+RSL_LEAD로
      //   전방 lead 위치앵커 실험 가능(자가앵커=현위치+lead·windup無): 평지 강전진(lead0.1·KP60=x2.85)이나 지형선 TAMOLS
      //   발판편차와 충돌해 전복(fragile)=RSL base task가 예측힘(모멘텀률FF) 없어 발판과 강건협조 불가. 근본해결=D1식 aBaseFF.
      q.KP_BASE =getenv("KP_BASE")?atof(getenv("KP_BASE")):0.0;
      q.KD_BASE =getenv("KD_BASE")?atof(getenv("KD_BASE")):25.0;
      q.SW_TRACK_W=getenv("SW_TRACK_W")?atof(getenv("SW_TRACK_W")):90.0;
      double _lead=getenv("RSL_LEAD")?atof(getenv("RSL_LEAD")):0.0;              // 전방 lead(KP_BASE>0일 때만 유효)
      q.com_ref[0]=d->qpos[0]+_lead*cy; q.com_ref[1]=d->qpos[1]+_lead*sy; q.com_ref[2]=x_ref[5];  // xy=현위치(+lead)·z=지형적응
      q.com_vel_ref[0]=vx_w; q.com_vel_ref[1]=vy_w; q.com_vel_ref[2]=0.0;        // 속도 ff=전진
      q.com_acc_ref.setZero();
    }
    else if(!st.empty() && (mpc_t<0||dmpc<0||dmpc>=q.mpc.DT)){
      std::vector<std::array<int,4>> cs(q.mpc.N);
      for(int k=0;k<q.mpc.N;k++) for(int i=0;i<4;i++){ bool sch; double sp; gait(i,tg+k*q.mpc.DT,sch,sp); cs[k][i]=sch?1:0; }
      Matrix<double,4,3> L=q.mpc_grf(x_ref,cs); for(int i=0;i<4;i++) lam_des[i]=L.row(i).transpose(); mpc_t=t; }
    Vector3d lam_use[4];
    bool no_mpc=getenv("NO_MPC");   // ★판별실험: 예측GRF(MPC) 제거→중력보상 baseline만(full-TAMOLS+WBC와 동일 조건). WBIC base task가 분배. 로봇모델이 marginal이면 발산
    if(rsl_inj||no_mpc){ int Kc=(int)st.size(); double fz=Kc>0? mj_getTotalmass(m)*9.81/Kc : 0.0;   // RSL: λ=중력보상 baseline(WBC base task가 실제분배)
      for(int i=0;i<4;i++) lam_use[i]=Vector3d(0,0,fz); }
    else for(int i=0;i<4;i++) lam_use[i]= st.empty()?Vector3d::Zero():lam_des[i];
    q.yaw_des=yaw_ref;                                     // ★자세 task가 명령헤딩 추종(선회시 yaw와 안싸움)
    double wl_inj=rsl_inj?(getenv("W_LAM")?atof(getenv("W_LAM")):0.1):10.0;
    if(!q.wbic_track(st,swing,lam_use,wl_inj)) q.wbic_stance();
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
  if(getenv("W_YAW")) q.w_yaw=atof(getenv("W_YAW"));      // yaw 헤딩홀드 가중(17dof=0권장:선회민감, 14dof=5:직진드리프트방지)
  if(getenv("MU")){ q.MU=atof(getenv("MU")); q.mpc.MU=q.MU*q.MU_MARGIN; }  // ★마찰콘 μ(물리 geom=1.3, 기존 0.6은 과보수→선회 GRF 포화)
  if(getenv("MOTOR_CURVE")) q.motor_curve=true;    // ★토크-속도곡선: 고속서 가용토크↓(속도한계 QP 반영)
  if(getenv("WAIST_W")) q.WAIST_W=atof(getenv("WAIST_W"));    // ★허리 홀드가중
  if(getenv("WAIST_KP")) q.WAIST_KP=atof(getenv("WAIST_KP")); if(getenv("WAIST_KD")) q.WAIST_KD=atof(getenv("WAIST_KD"));
  if(getenv("SWING_W")){ double v=atof(getenv("SWING_W")); q.swing_w_r=v; q.swing_w_f=v; }
  if(getenv("SWING_W_R")) q.swing_w_r=atof(getenv("SWING_W_R"));
  if(getenv("SWING_W_F")) q.swing_w_f=atof(getenv("SWING_W_F"));
  if(getenv("PIN_ANKLE")) q.stance_pin_ankle=true;
}
