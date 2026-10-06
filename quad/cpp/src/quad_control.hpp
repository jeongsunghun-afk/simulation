// QuadControl — quad_mpc_wbic.py 컨트롤러(모델 의존부) C++ 이관.
// crouch_home(IK) · wbic_stance · wbic_track(스윙WBIC) · compute_Icom · body_x0 · mpc_grf.
// main.cpp(검증)와 trot_sim.cpp(closed-loop)가 공유. 검증된 로직 그대로.
#pragma once
#include <mujoco/mujoco.h>
#include <eiquadprog/eiquadprog-fast.hpp>
#include <Eigen/Dense>
#include "mpc.hpp"
#include "quad_wbic_mit.hpp"   // ★KinWBC(하이브리드 드라이버 목표) · MIT 식 WBIC(연구용)
#include <vector>
#include <array>
#include <map>
#include <set>
#include <string>
#include <cmath>
using namespace Eigen;

struct QuadControl {
  mjModel* m=nullptr; mjData* d=nullptr;
  int nq=0,nv=0,nu=0;
  std::vector<std::vector<int>> legqp{4},legqv{4};
  int leg_dof[4]={0}, hip_bid[4]={0}, fgid[4]={0}, fbid[4]={0}; double fr[4]={0};
  int rear_hock_bid[4]={0};   // ★<leg>_foot_link body(발목/hock 원점) — 개-앉기서 발링크 z구속에 사용
  const char* legs[4]={"HL","HR","FL","FR"};
  // 상수
  double MU=0.6, MU_MARGIN=0.707, LAMZ_MIN=1.0;         // wbic 마찰
  double STANCE_KD=20.0;                                 // ★stance 접촉속도 감쇠(baumgarte): cjac·q̈=−KD·(cjac·q̇) → 터치다운 잔류속도→0, 발 slip↓(뒤 7.2→5.9mm). 0=끔
  double base_z0=0.52, REAR_ANKLE=-0.3, FRONT_ANKLE=-0.7;  // 14dof:ours_sphere / 17dof: 0.5234. ★뒷발목 -0.7→-0.3: 축별 τ·ω 스윕 최적(REAR_ANKLE 비교, walk v0.6). 발목 ω가 병목(-0.7서 모터한계 155%=flail)→ -0.3이 ω를 100%로 낮추는 최소확장(calf τ76%·falls=0). 앞발목 -0.7 유지. 14dof(3관절)=무영향
  double HAUNCH_THIGH=-1.0, HAUNCH_CALF=1.2, HAUNCH_FOOT=-0.3, HAUNCH_HOCK_Z=0.019, FRONT_REACH=-0.22;  // ★엉덩이 주저앉기 뒷다리 tuck(femur 위/뒤·발 접힘)+앞발 뒤로(앞다리 곧게 수직으로 펴 상체 받침)
  double HAUNCH_FOOT_LAND=-1.2;  // ★착지 중 뒷발 각도(더 접음=발 curl). 바닥 닿은 뒤 HAUNCH_FOOT(-0.3)로 굴려 발바닥 밀착
  double W_AM=0.0, KD_AM=8.0;                            // 각운동량 보상(14dof평지=0, 17dof튜닝=12/24)
  double w_ori=5.0;                                      // wbic_track 자세 task 가중(14dof=5, 17dof튜닝=20)
  double yaw_des=0.0;                                     // ★자세 task 목표 헤딩(TrotCtrl이 yaw_ref로 설정). 선회시 몸통이 추종
  double sit_pitch=0.0;                                  // ★wbic_stance 자세목표 nose-up(앉기 느낌). 0=수평. CoM은 중앙유지라 안정 nose-up 가능
  double posture_w=1.0;                                  // ★wbic_stance 자세task 가중 스케일(기본1). 개-앉기서 ↑=접힘(q_home) 홀드(CoM균형은 유지). 서기/보행=1
  bool sit_hock_contact=false;                           // ★개-앉기: 뒤 접촉점을 toe sphere→hock(발목)로 전환. CoM을 hock~앞발 지지폴리곤 기준 균형→발링크 평평 유지(toe-stand 방지)
  double w_yaw=0.0;                                        // ★yaw 헤딩홀드 가중(roll/pitch와 분리). euler 표준수정 후 0이 최적(14·17 공통; MPC가 yaw 담당). >0=헤딩홀드
  double W_BASE_XY=0.0, KP_BASE=150.0, KD_BASE=25.0;      // ★RSL식: wbic_track이 base 수평(x,y) 위치를 직접 추종(>0=on). SRBD MPC 대신 계획 base궤적을 WBC가 execute
  Vector3d com_vel_ref=Vector3d::Zero(), com_acc_ref=Vector3d::Zero();  // 수평 base 속도·가속 참조(TAMOLS 계획)
  // ★GM-observer(TAMOLS VI-B): base task 정상상태 잔차를 적분추정→보상(모델오차·접촉전이 잔여힘=z침하 상쇄). GM_KI로 on.
  double gm_zi=0, gm_ri=0, gm_pi=0; void gm_reset(){ gm_zi=gm_ri=gm_pi=0; }
  double swing_w_r=0.1, swing_w_f=0.1;                    // 스윙다리 여유도 posture(앞/뒤 별도, ↑=whip 억제)
  double SW_TRACK_W=90.0;                                 // ★swing 발 task 가중(발이 목표 정확추종·↑=착지정확). horizon-shift 안정화
  std::vector<char> is_front;
  // ★calf-foot 커플 모터공간 사영: 실기 관절토크 τ_calf=τ_km+τ_am·τ_foot=τ_am(발목모터, 이미 ±peak_foot 클램프).
  //   τ_km=clip(τ_calf−τ_am,±peak_calf) 후 재조합 → 독립박스가 허용하던 실기불가 코너(|τ_calf−τ_foot|>126) 제거.
  //   물리적 사영(드라이브가 실제 하는 것)=비물리 캡 아님. 한계: MOTOR_CURVE의 속도의존 감소는 미커플(후속).
  long couple_hits=0, couple_calls=0; double couple_km_pk=0;   // ★binding 계측: 사영이 토크를 실제로 바꾼 횟수
  void couple_clamp(mjData* dd){ if(!couple_on) return;
    for(int a2=0;a2<nu;a2++){ int c=cpl_calf_of[a2]; if(c<0) continue;
      double am=dd->ctrl[a2], km_raw=dd->ctrl[c]-am;
      double km=std::max(-tau_peak[c],std::min(tau_peak[c],km_raw));
      if(std::abs(km_raw)>couple_km_pk) couple_km_pk=std::abs(km_raw);
      if(km!=km_raw) ++couple_hits; ++couple_calls;
      dd->ctrl[c]=km+am; } }
                             // actuator별 앞다리(FL/FR) 여부
  bool stance_pin_ankle=false;                           // 17dof: stance서도 여유발목 핀(전4다리4DOF redundancy 표류차단)
  int waist_idx=-1;                                       // ★허리(FB_waist) nu-index(없으면 -1=16DOF). 큰 몸통DOF라 전용 강홀드
  double waist_ref=0.0, WAIST_W=80.0, WAIST_KP=150.0, WAIST_KD=20.0;  // 요각목표(조향시 갱신)·홀드가중·PD
  VectorXd q_home, q_sit; Vector3d com_ref;   // q_sit=앉기 자세(뒷다리 접고 앞다리 편)
  VectorXd tau_peak, qmin, qmax, w_limit; std::vector<char> is_ankle;   // w_limit=관절속도한계[rad/s]=207/N
  std::vector<int> cpl_calf_of; bool couple_on=true;     // ★calf-foot 기구커플(foot액추에이터→같은다리 calf 인덱스, -1=없음). COUPLE=0로 끔
  bool motor_curve=false;                                 // ★MOTOR_CURVE: 가용토크=tau_peak·max(0,1−|ω|/w_limit) (고속서↓=실모터)
  std::array<Vector2d,4> foot_hip_off; std::array<double,4> foot_gz0;
  MpcCfg mpc; double _body_terr=0.0;
  eiquadprog::solvers::EiquadprogFast _qp_st, _qp_tr, _qp_jm;

  void load(const char* path){
    char err[1000]=""; m=mj_loadXML(path,nullptr,err,1000);
    if(!m){ std::fprintf(stderr,"load fail: %s\n",err); std::exit(1); }
    m->opt.timestep = 0.001;   // ★Python과 동일(1kHz, TIMESTEP env 기본). mjcf 0.002 오버라이드
    // ★관절한계 강화(Python line 119-120): soft 기본이 동적충격에 뚫림 → 시정수↓·impedance↑
    for(int j=0;j<m->njnt;j++) if(m->jnt_limited[j]){
      m->jnt_solref[j*2]=0.004; m->jnt_solref[j*2+1]=1.0;
      m->jnt_solimp[j*5]=0.95; m->jnt_solimp[j*5+1]=0.99; m->jnt_solimp[j*5+2]=0.001;
      m->jnt_solimp[j*5+3]=0.5; m->jnt_solimp[j*5+4]=2.0; }
    // ★접촉 강성(Python STIFF=0.005, 전 geom): 기본 0.02는 물렁→발 35mm 침투. solref 시정수↓로 침투 3mm
    for(int g=0;g<m->ngeom;g++){ m->geom_solref[g*2]=0.005; m->geom_solref[g*2+1]=1.0; }
    d=mj_makeData(m); nq=m->nq; nv=m->nv; nu=m->nu;
    const char* JT[4]={"hip","thigh","calf","foot"};
    for(int i=0;i<4;i++){
      hip_bid[i]=mj_name2id(m,mjOBJ_BODY,(std::string(legs[i])+"_hip_link").c_str());
      legqp[i].clear(); legqv[i].clear();
      for(int t=0;t<4;t++){ int j=mj_name2id(m,mjOBJ_JOINT,(std::string(legs[i])+"_"+JT[t]+"_joint").c_str());
        if(j>=0){ legqp[i].push_back(m->jnt_qposadr[j]); legqv[i].push_back(m->jnt_dofadr[j]); } }
      leg_dof[i]=(int)legqp[i].size();
      int gid=mj_name2id(m,mjOBJ_GEOM,(std::string(legs[i])+"_sphere").c_str());
      fgid[i]=gid; fbid[i]=m->geom_bodyid[gid]; fr[i]=m->geom_size[gid*3];
      rear_hock_bid[i]=mj_name2id(m,mjOBJ_BODY,(std::string(legs[i])+"_foot_link").c_str());   // ★hock(발목 원점) body
    }
    // tau_peak / qmin·qmax / ankle (non-free joint 순서=actuator)
    tau_peak.resize(nu); qmin.resize(nu); qmax.resize(nu); is_ankle.assign(nu,0);
    w_limit.setConstant(nu,1e8);   // 관절속도한계(재기어 블록서 207/N로 설정)
    int a=0; for(int j=0;j<m->njnt;j++){ if(m->jnt_type[j]==mjJNT_FREE) continue;
      double frc=m->jnt_actfrcrange[j*2+1]; tau_peak[a]=frc>0?frc:1e8;
      if(m->jnt_limited[j]){ qmin[a]=m->jnt_range[j*2]; qmax[a]=m->jnt_range[j*2+1]; } else { qmin[a]=-1e9; qmax[a]=1e9; }
      a++; }
    for(int i=0;i<4;i++) if(leg_dof[i]==4) is_ankle[legqv[i][3]-6]=1;
    // ★calf-foot 기구커플 페어맵(biped PACE 실측 c=1: 발목모터가 raw=q_foot+q_calf 좌표 구동)
    cpl_calf_of.assign(nu,-1);
    for(int i=0;i<4;i++) if(leg_dof[i]==4) cpl_calf_of[legqv[i][3]-6]=legqv[i][2]-6;
    couple_on = !(getenv("COUPLE")&&!strcmp(getenv("COUPLE"),"0"));
    is_front.assign(nu,0);   // FL/FR 다리의 actuator = 앞다리
    for(int i=0;i<4;i++){ bool fr=(std::string(legs[i])=="FL"||std::string(legs[i])=="FR");
      for(int t=0;t<leg_dof[i];t++){ int a=legqv[i][t]-6; if(a>=0&&a<nu) is_front[a]=fr; } }
    { int wj=mj_name2id(m,mjOBJ_JOINT,"FB_waist_joint");   // ★허리 조인트(있으면 능동 17-DOF)
      waist_idx = (wj>=0 && m->jnt_type[wj]!=mjJNT_FREE) ? m->jnt_dofadr[wj]-6 : -1; }
    const bool legs4x4 = leg_dof[0]==4&&leg_dof[1]==4&&leg_dof[2]==4&&leg_dof[3]==4;   // ★2026-10-06 실물 허리 제거(16-DOF) — 다리는 17-DOF 와 같으니 같은 튜닝
    if(waist_idx>=0 || legs4x4){   // ★17-DOF(허리모델)·16-DOF(4자유도 다리×4) 자동감지 → Python 17dof 튜닝값을 기본으로(canonical 실행코드와 동일). env로 여전히 override
      w_ori=20.0; W_AM=0.0; KD_AM=24.0; FRONT_ANKLE=-0.5;   // ★W_AM=0(각운동량 보상 제거): 외란복구 이득 측정상 무의미(측방 push서 오히려↑). 14dof 기본(5/0/8/-0.7) 대신 17dof 튜닝
      base_z0=0.50; }   // ★base height 최적(축별 worst-util 스윕): walk 95.8%·trot 67.7%·run 82.5%로 0.50이 3gait Pareto-최적(0.5234보다 발목 ω 여유↑)
    // ★감속비(실값)+기어박스 물리(sim2real). Python line 244-263 일치.
    //   foot 감속비=8.4:1(실값). peak토크=MJCF actuatorfrcrange(foot 100.8Nm=8.4:1)·ω한계=207/8.4=24.6rad/s.
    //   GEAR_* env는 실험용 재기어 배수(기본 1.0). GEARBOX=1: 반사관성 dof_armature=I_rotor·N² + 점성감쇠 + 마찰(MJCF엔 0→발목 flail 과장 보정).
    { const char* GN[4]={"hip","thigh","calf","foot"}; const char* GE[4]={"GEAR_HIP","GEAR_THIGH","GEAR_CALF","GEAR_FOOT"};
      double gear[4]={7.0,7.0,10.5,8.4};                             // 관절별 감속비 실값(hip/thigh7·calf10.5·foot8.4, Python 일치)
      bool gbx = !(getenv("GEARBOX") && !std::strcmp(getenv("GEARBOX"),"0"));   // ★기본 ON(반사관성=실제 물리). GEARBOX=0으로만 끔
      // ★PACE 최종 실측(2026-08-14, biped/emb/pace/RESULTS.md=단일출처): ROTOR_I 7.327e-4·축별 damping/friction.
      //   armature=Irot·N²(hip/thigh0.0359·calf0.0808·foot0.0517). foot 손실=실기선 tendon(모터축) — quad plant는
      //   tendon 미이식이라 foot dof 근사(무릎회전→발목모터 마찰반력 누락, tendon 이식 시 해소). 지연 8.39ms 실측=ACT_LAT용.
      double Irot=getenv("ROTOR_I")?atof(getenv("ROTOR_I")):7.327e-4;
      const double dmpK[4]={0.090,0.0,0.0,0.110}, frcK[4]={0.724,0.604,0.871,0.639};   // hip,thigh,calf,foot(tendon값)
      for(int k=0;k<nu;k++){ int jid=m->actuator_trnid[k*2]; if(jid<0) continue;
        const char* jn=mj_id2name(m,mjOBJ_JOINT,jid); if(!jn) continue;
        int gi=0; for(int g=0;g<4;g++) if(std::strstr(jn,GN[g])) gi=g;   // ★FB_waist는 hip/thigh/calf/foot 어디에도 안 걸려 gi=0(hip)로 fallback → 감속비 7:1. ★실제 허리 감속비=7:1(사용자확인)이라 이 fallback이 정확함(의도됨, 삭제금지). 반사관성·w_limit·tau도 hip과 동일 처리
        double gmul=getenv(GE[gi])?atof(getenv(GE[gi])):1.0;
        w_limit[k]=207.0/(gear[gi]*gmul);                            // ★관절속도한계=motor_noload/N (MOTOR_CURVE용)
        if(gmul!=1.0 && tau_peak[k]<1e7) tau_peak[k]*=gmul;          // ★재기어 토크한계(QP 부등식이 사용)
        if(gbx){ double N=gear[gi]*gmul; int dof=m->jnt_dofadr[jid];
          double jd=getenv("JDAMP")?atof(getenv("JDAMP")):dmpK[gi], jf=getenv("JFRIC")?atof(getenv("JFRIC")):frcK[gi];
          m->dof_armature[dof]=Irot*N*N; m->dof_damping[dof]=jd; m->dof_frictionloss[dof]=jf; } }
    }
    if(getenv("BASE_Z0")) base_z0=atof(getenv("BASE_Z0"));   // ★서기높이 기준 override(다른 로봇 이식 테스트용, 예 Go2=0.30). 02_Leg는 미설정→기본 유지
    q_home.resize(nu);
  }
  Vector3d foot_point(int i){ Vector3d p(d->geom_xpos[fgid[i]*3],d->geom_xpos[fgid[i]*3+1],d->geom_xpos[fgid[i]*3+2]); p[2]-=fr[i]; return p; }
  // ★perceptive: (x,y) 아래 지형 표면 z를 mj_ray 하향캐스트로 샘플. 지형 geom(group2)만 마스킹 → 로봇/바닥 무시.
  //   ★시작 z=2.0(어떤 지형보다 위)서 단일 하향레이: group2 마스크라 로봇 자기충돌 없음 → 0.40 이상 높은 지형도 탐지(구 ray_z0=0.40은 그보다 높으면 미탐지→붕괴).
  //   ★단일레이=고속(재캐스트 루프는 로봇 뚫느라 40ray/step → run 실시간 붕괴). 미검출=−100(폴백=평지).
  double ray_z0=2.0;
  double terrain_z(double x,double y){
    mjtNum pnt[3]={x,y,ray_z0}, vec[3]={0,0,-1}, nrm[3]; int gid=-1;
    static const mjtByte gg[6]={0,0,1,0,0,0};   // group2(지형)만 레이 대상 — 로봇(group0/1)·바닥(plane) 제외
    mjtNum dist=mj_ray(m,d,pnt,vec,gg,1,-1,&gid,nrm);
    if(dist>=0 && gid>=0) return ray_z0-dist;
    return -100.0;
  }
  Matrix<double,3,Dynamic> foot_jac(int i){ std::vector<double> jb(3*nv); Vector3d p=foot_point(i);
    mj_jac(m,d,jb.data(),nullptr,p.data(),fbid[i]);
    Matrix<double,3,Dynamic> J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jb[r*nv+c]; return J; }
  Vector3d hock_point(int i){ return Vector3d(d->xpos[rear_hock_bid[i]*3],d->xpos[rear_hock_bid[i]*3+1],d->xpos[rear_hock_bid[i]*3+2]); }
  Matrix<double,3,Dynamic> hock_jac(int i){ std::vector<double> jb(3*nv); Vector3d p=hock_point(i);   // ★hock(발목원점) 야코비 — 개-앉기서 뒤 접촉=발링크(hock)
    mj_jac(m,d,jb.data(),nullptr,p.data(),rear_hock_bid[i]);
    Matrix<double,3,Dynamic> J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jb[r*nv+c]; return J; }

  // crouch_home: 넓은 발위치 유지 무릎굽힘 → q_home/com_ref/standing. + foot_hip_off/foot_gz0
  void crouch_home(double bz=-1){
    double base_z = (bz>0? bz : base_z0), foot_z0=0.0;
    if(m->nkey>0) mj_resetDataKeyframe(m,d,0); else { for(int i=0;i<nq;i++) d->qpos[i]=0; d->qpos[3]=1; }
    d->qpos[2]=0.60; mj_forward(m,d);
    Vector2d foot_xy[4]; for(int i=0;i<4;i++) foot_xy[i]=foot_point(i).head(2);
    d->qpos[2]=base_z;
    for(int i=0;i<4;i++) if(leg_dof[i]==4){ double ang=(std::string(legs[i])=="FL"||std::string(legs[i])=="FR")?FRONT_ANKLE:REAR_ANKLE;
      if(ang!=0.0) d->qpos[legqp[i][3]]=ang; }
    for(int it=0;it<300;it++){ mj_kinematics(m,d); mj_comPos(m,d);
      for(int i=0;i<4;i++){ Vector3d tgt(foot_xy[i][0],foot_xy[i][1],foot_z0); Vector3d e=tgt-foot_point(i);
        Matrix<double,3,Dynamic> Jf=foot_jac(i); Matrix3d J; for(int r=0;r<3;r++)for(int cc=0;cc<3;cc++) J(r,cc)=Jf(r,legqv[i][cc]);
        Vector3d dq=0.5*(J.transpose()*(J*J.transpose()+1e-4*Matrix3d::Identity()).ldlt().solve(e));
        for(int cc=0;cc<3;cc++) d->qpos[legqp[i][cc]]+=dq[cc]; } }
    mj_forward(m,d);
    for(int i=0;i<nu;i++) q_home[i]=d->qpos[7+i];
    Vector2d fc(0,0); for(int i=0;i<4;i++) fc+=foot_point(i).head(2); fc/=4.0;
    com_ref<<fc[0],fc[1],d->subtree_com[2];
    for(int i=0;i<4;i++){ foot_hip_off[i]=foot_point(i).head(2)-Vector2d(d->xpos[hip_bid[i]*3],d->xpos[hip_bid[i]*3+1]);
      foot_gz0[i]=foot_point(i)[2]; }
    for(int i=0;i<nv;i++) d->qvel[i]=0; mj_forward(m,d);
  }
  // ★크라우치-앉기 자세: 검증된 crouch_home(안정 저crouch, 4발 planted)을 저높이서 계산해 q_sit에 저장.
  //   부작용(q_home·com_ref·foot_hip_off·foot_gz0)·라이브 d 모두 저장·복원 → standing 상태 안 건드림.
  void crouch_sit_home(double base_z, double pitch=0.0, double front_reach=0.0){
    // ★nose-up pitch(앉은느낌)+front_reach(앞발 전방 확장=앞다리 폄). 뒷발은 지면 유지(기립가능). 발 z=0 IK.
    std::vector<double> sq(nq),sv(nv); double st=d->time;
    for(int i=0;i<nq;i++) sq[i]=d->qpos[i]; for(int i=0;i<nv;i++) sv[i]=d->qvel[i];
    for(int i=0;i<nq;i++) d->qpos[i]=0; d->qpos[3]=1; d->qpos[2]=0.60; mj_forward(m,d);
    Vector2d foot_xy[4]; for(int i=0;i<4;i++) foot_xy[i]=foot_point(i).head(2);   // 명목 발 XY
    for(int i=0;i<4;i++){ bool fr=(std::string(legs[i])=="FL"||std::string(legs[i])=="FR");
      if(fr) foot_xy[i][0]+=front_reach; }                                        // ★앞발 전방 확장(앞다리 폄 look)
    d->qpos[2]=base_z; d->qpos[3]=std::cos(pitch/2); d->qpos[4]=0; d->qpos[5]=-std::sin(pitch/2); d->qpos[6]=0;  // nose-up
    for(int i=0;i<4;i++){ bool fr=(std::string(legs[i])=="FL"||std::string(legs[i])=="FR");  // ★안정 가지 초기값(저높이 flip 방지)
      d->qpos[legqp[i][1]]=fr?-0.20:0.68; d->qpos[legqp[i][2]]=fr?0.49:-0.88;
      if(leg_dof[i]==4) d->qpos[legqp[i][3]]=fr?FRONT_ANKLE:REAR_ANKLE; }
    for(int it=0;it<400;it++){ mj_kinematics(m,d); mj_comPos(m,d);   // 발 지면(z=0) IK, 3-DOF
      for(int i=0;i<4;i++){ Vector3d tgt(foot_xy[i][0],foot_xy[i][1],0.0); Vector3d e=tgt-foot_point(i);
        Matrix<double,3,Dynamic> Jf=foot_jac(i); Matrix3d J; for(int r=0;r<3;r++)for(int cc=0;cc<3;cc++) J(r,cc)=Jf(r,legqv[i][cc]);
        Vector3d dq=0.5*(J.transpose()*(J*J.transpose()+1e-4*Matrix3d::Identity()).ldlt().solve(e));
        for(int cc=0;cc<3;cc++) d->qpos[legqp[i][cc]]+=dq[cc]; } }
    mj_forward(m,d); q_sit.resize(nu); for(int i=0;i<nu;i++) q_sit[i]=d->qpos[7+i];
    if(getenv("SITDBG")) for(int i=0;i<4;i++) std::printf("[csit] %s thigh=%.3f calf=%.3f foot=%.3f footZ=%.3f\n",
        legs[i], q_sit[legqp[i][1]-7], q_sit[legqp[i][2]-7], leg_dof[i]==4?q_sit[legqp[i][3]-7]:0.0, foot_point(i)[2]);
    for(int i=0;i<nq;i++) d->qpos[i]=sq[i]; for(int i=0;i<nv;i++) d->qvel[i]=sv[i]; d->time=st; mj_forward(m,d);
  }
  // ★개-앉기(haunch sit): 뒷다리 접어 발 링크(hock→toe 중족골)를 바닥에 평평, 앞다리 편 자세.
  //   뒷다리 3-DOF(thigh/calf/foot, hip=0)로 [toe_x; toe_z=0; hock_z=HAUNCH_HOCK_Z] 구속 → 발링크 양끝 지면=평평(무릎-위 가지=시드).
  //   q_home/com_ref(지지폴리곤 중심)/foot_hip_off/foot_gz0 기록. live d 저장/복원(텔레포트X). base_z=수평(nose-up은 wbic_stance가 담당).
  void haunch_sit_home(double base_z, double pitch=0.0){
    std::vector<double> sq(nq),sv(nv); double st=d->time;
    for(int i=0;i<nq;i++) sq[i]=d->qpos[i]; for(int i=0;i<nv;i++) sv[i]=d->qvel[i];
    if(m->nkey>0) mj_resetDataKeyframe(m,d,0); else { for(int i=0;i<nq;i++) d->qpos[i]=0; d->qpos[3]=1; }
    d->qpos[2]=0.60; mj_forward(m,d);
    Vector2d foot_xy[4]; for(int i=0;i<4;i++) foot_xy[i]=foot_point(i).head(2);   // 명목 발 XY
    d->qpos[2]=base_z;
    d->qpos[3]=std::cos(pitch/2); d->qpos[4]=0; d->qpos[5]=-std::sin(pitch/2); d->qpos[6]=0;  // ★nose-up 베이스(발링크 평평은 월드프레임 z구속이라 pitch 반영). PD홀드 시 이 pitch가 재현됨
    for(int i=0;i<4;i++){ bool fro=(std::string(legs[i])=="FL"||std::string(legs[i])=="FR");
      if(fro){ if(leg_dof[i]==4) d->qpos[legqp[i][3]]=FRONT_ANKLE; }                 // 앞다리 발목 초기값(발 지면 IK로 폄)
      else { d->qpos[legqp[i][0]]=0.0; d->qpos[legqp[i][1]]=HAUNCH_THIGH;            // ★뒷다리 tuck(접힘) 고정: 무릎 위/뒤·종아리·발 접어 엉덩이 밑으로. IK 안 함(발 안 뻗음)
             d->qpos[legqp[i][2]]=HAUNCH_CALF; d->qpos[legqp[i][3]]=HAUNCH_FOOT; } }
    for(int it=0;it<400;it++){ mj_kinematics(m,d);                                    // ★앞다리(FL,FR)만 발 지면 IK로 폄(nose-up 상체 지지) — 뒷다리는 tuck 고정
      for(int i=2;i<4;i++){
        Vector3d tgt(foot_xy[i][0]+FRONT_REACH,foot_xy[i][1],0.0); Vector3d e=tgt-foot_point(i);  // 앞발 약간 전방(앞다리 폄)
        Matrix<double,3,Dynamic> Jf=foot_jac(i); Matrix3d J; for(int r=0;r<3;r++)for(int cc=0;cc<3;cc++) J(r,cc)=Jf(r,legqv[i][cc]);
        Vector3d dq=0.5*(J.transpose()*(J*J.transpose()+1e-4*Matrix3d::Identity()).ldlt().solve(e));
        for(int cc=0;cc<3;cc++) d->qpos[legqp[i][cc]]+=dq[cc]; } }
    mj_forward(m,d);
    for(int i=0;i<nu;i++) q_home[i]=d->qpos[7+i];
    com_ref<<d->subtree_com[0],d->subtree_com[1],d->subtree_com[2];                  // ★com_ref=자연 CoM(엉덩이 주저앉기=rump/뒷몸통 지지, 지지폴리곤 큼)
    Vector2d fc(com_ref[0],com_ref[1]);
    for(int i=0;i<4;i++){ foot_hip_off[i]=foot_point(i).head(2)-Vector2d(d->xpos[hip_bid[i]*3],d->xpos[hip_bid[i]*3+1]);
      foot_gz0[i]=foot_point(i)[2]; }
    if(getenv("SITDBG")) for(int i=0;i<4;i++) std::printf("[haunch] %s thigh=%.3f calf=%.3f foot=%.3f hockZ=%.3f toeZ=%.3f\n",
        legs[i], q_home[legqp[i][1]-7], q_home[legqp[i][2]-7], leg_dof[i]==4?q_home[legqp[i][3]-7]:0.0,
        (i<2)?d->xpos[rear_hock_bid[i]*3+2]:0.0, foot_point(i)[2]);
    for(int i=0;i<nq;i++) d->qpos[i]=sq[i]; for(int i=0;i<nv;i++) d->qvel[i]=sv[i]; d->time=st; mj_forward(m,d);
  }
  // 가변높이 standing q_home/com_ref 재계산(IK) — 라이브 d 복원(텔레포트X). 서기높이변경·눕기용.
  // ── q_home LUT (실로봇 RT-safe: 무거운 IK를 제어루프서 제거 → 오프라인 precompute+선형보간. sim=실배포 동일 아키텍처) ──
  //   높이별 q_home(발목 포함 자세)·com_ref·foot_hip_off·foot_gz0를 시작 시 300회 IK로 표화. RT는 보간만(IK 없음).
  //   ★값은 cold(300회 완전수렴)이라 직접IK와 동일 → walk 회귀 없음. 발목=상수·hip/thigh/calf만 높이변화→보간 정확.
  std::vector<double> lut_h; double lut_step=0.005;
  std::vector<VectorXd> lut_qh; std::vector<Vector3d> lut_com;
  std::vector<std::array<Vector2d,4>> lut_fho; std::vector<std::array<double,4>> lut_fgz;
  void update_stand_qhome_ik(double base_z){   // 직접 IK(오프라인 표생성·self-check 전용, RT 미사용)
    std::vector<double> sq(nq),sv(nv); double st=d->time;
    for(int i=0;i<nq;i++) sq[i]=d->qpos[i]; for(int i=0;i<nv;i++) sv[i]=d->qvel[i];
    crouch_home(base_z);
    for(int i=0;i<nq;i++) d->qpos[i]=sq[i]; for(int i=0;i<nv;i++) d->qvel[i]=sv[i]; d->time=st;
    mj_forward(m,d);
  }
  void build_qhome_lut(double h0=0.18, double h1=0.55, double step=0.005){
    lut_step=step; std::vector<double> sq(nq),sv(nv); double st=d->time;
    for(int i=0;i<nq;i++) sq[i]=d->qpos[i]; for(int i=0;i<nv;i++) sv[i]=d->qvel[i];
    lut_h.clear(); lut_qh.clear(); lut_com.clear(); lut_fho.clear(); lut_fgz.clear();
    for(double h=h0; h<=h1+1e-9; h+=step){
      crouch_home(h);                                        // 300회 cold(시작 1회, RT 아님)
      lut_h.push_back(h); lut_qh.push_back(q_home); lut_com.push_back(com_ref);
      lut_fho.push_back(foot_hip_off); lut_fgz.push_back(foot_gz0);
    }
    for(int i=0;i<nq;i++) d->qpos[i]=sq[i]; for(int i=0;i<nv;i++) d->qvel[i]=sv[i]; d->time=st;
    crouch_home(base_z0);                                    // q_home 등 부작용을 명목높이로 원상복구
    for(int i=0;i<nq;i++) d->qpos[i]=sq[i]; for(int i=0;i<nv;i++) d->qvel[i]=sv[i]; d->time=st; mj_forward(m,d);
  }
  void update_stand_qhome(double base_z){   // ★LUT 보간(RT-safe, IK 없음). 실배포와 동일 경로.
    if(lut_h.empty()) build_qhome_lut();                     // 안전망(보통 init서 미리 빌드)
    double h = std::min(std::max(base_z, lut_h.front()), lut_h.back());
    int i = (int)((h - lut_h.front())/lut_step); i = std::max(0, std::min((int)lut_h.size()-2, i));
    double a = (h - lut_h[i]) / (lut_h[i+1]-lut_h[i]); a = std::max(0.0, std::min(1.0, a));
    q_home = (1-a)*lut_qh[i] + a*lut_qh[i+1];
    com_ref = (1-a)*lut_com[i] + a*lut_com[i+1];
    for(int k=0;k<4;k++){ foot_hip_off[k]=(1-a)*lut_fho[i][k]+a*lut_fho[i+1][k];
      foot_gz0[k]=(1-a)*lut_fgz[i][k]+a*lut_fgz[i+1][k]; }
  }
  // ★앉기 자세 IK: 몸통 pitch↑(nose up)+낮춤 → 앞다리 펴짐·뒷다리 접힘. 뒷다리=calf(+최대)·foot(-최대) 고정,
  //   IK는 앞다리 3DOF(hip/thigh/calf)+뒷다리 2DOF(hip/thigh)만으로 발 지면(z=0) 배치. q_sit 저장(라이브 d 복원).
  void sit_home(double base_z, double pitch, double rear_foot, double rear_calf, double rear_thigh, bool crouch=false){
    bool pin_thigh = rear_thigh > -900;   // ★rear_thigh 지정 시 뒷다리 thigh도 고정→뒷다리는 hip만 IK(발은 접힘형상대로 착지)
    // ★crouch=true(크라우치-앉기): 접힘 없이 모든 발 정상 발목각·전 다리 3-DOF IK → 4발 planted 저crouch(기립 가능). 완만 pitch로 앉은 느낌
    std::vector<double> sq(nq),sv(nv); double st=d->time;
    for(int i=0;i<nq;i++) sq[i]=d->qpos[i]; for(int i=0;i<nv;i++) sv[i]=d->qvel[i];
    if(crouch && m->nkey>0) mj_resetDataKeyframe(m,d,0); else { for(int i=0;i<nq;i++) d->qpos[i]=0; d->qpos[3]=1; }  // ★crouch=keyframe(정상가지 초기값)
    d->qpos[2]=0.60; mj_forward(m,d);
    Vector2d foot_xy[4]; for(int i=0;i<4;i++) foot_xy[i]=foot_point(i).head(2);   // 명목 발 XY
    d->qpos[2]=base_z;
    if(crouch) pitch=0.0;   // ★crouch: 수평(nose-up은 rock-back 유발). crouch_home과 동일 안정 crouch
    d->qpos[3]=std::cos(pitch/2); d->qpos[4]=0; d->qpos[5]=-std::sin(pitch/2); d->qpos[6]=0;  // ★pitch(y축, nose up=앞올림 → 앞다리 펴짐·뒷다리 접힘)
    for(int i=0;i<4;i++){ bool fr=(std::string(legs[i])=="FL"||std::string(legs[i])=="FR");
      if(leg_dof[i]==4) d->qpos[legqp[i][3]]=crouch?(fr?FRONT_ANKLE:REAR_ANKLE):(fr?FRONT_ANKLE:rear_foot);   // crouch=crouch_home과 동일(뒷발=REAR_ANKLE)
      if(!fr && !crouch){ d->qpos[legqp[i][2]]=rear_calf;                          // ★뒷다리 calf=rear_calf(+최대접힘)
               if(pin_thigh) d->qpos[legqp[i][1]]=rear_thigh; } }       // ★뒷다리 thigh=rear_thigh(음수)
    for(int it=0;it<400;it++){ mj_kinematics(m,d); mj_comPos(m,d);   // 발 지면(z=0) IK
      for(int i=0;i<4;i++){ bool fr=(std::string(legs[i])=="FL"||std::string(legs[i])=="FR");
        int nd=(crouch||fr)?3:(pin_thigh?1:2);   // crouch/앞=hip/thigh/calf 3-DOF, 뒤(비crouch)=2 또는 1
        Vector3d tgt(foot_xy[i][0],foot_xy[i][1],0.0); Vector3d e=tgt-foot_point(i);
        Matrix<double,3,Dynamic> Jf=foot_jac(i); Matrix<double,3,Dynamic> J(3,nd);
        for(int r=0;r<3;r++)for(int cc=0;cc<nd;cc++) J(r,cc)=Jf(r,legqv[i][cc]);
        VectorXd dq=0.5*(J.transpose()*(J*J.transpose()+1e-4*Matrix3d::Identity()).ldlt().solve(e));
        for(int cc=0;cc<nd;cc++) d->qpos[legqp[i][cc]]+=dq[cc]; } }
    mj_forward(m,d);
    if(getenv("SITDBG")) for(int i=0;i<4;i++) std::printf("[sitdbg] %s hip=%.3f thigh=%.3f calf=%.3f foot=%.3f footZ=%.3f\n",
        legs[i], d->qpos[legqp[i][0]], d->qpos[legqp[i][1]], d->qpos[legqp[i][2]], leg_dof[i]==4?d->qpos[legqp[i][3]]:0.0, foot_point(i)[2]);
    q_sit.resize(nu); for(int i=0;i<nu;i++) q_sit[i]=d->qpos[7+i];
    for(int i=0;i<nq;i++) d->qpos[i]=sq[i]; for(int i=0;i<nv;i++) d->qvel[i]=sv[i]; d->time=st; mj_forward(m,d);
  }
  Matrix3d compute_Icom(){
    Vector3d com(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]); Matrix3d I=Matrix3d::Zero();
    for(int b=1;b<m->nbody;b++){ double ms=m->body_mass[b]; if(ms<=0) continue;
      Vector3d r(d->xipos[b*3]-com[0],d->xipos[b*3+1]-com[1],d->xipos[b*3+2]-com[2]);
      Matrix<double,3,3,RowMajor> Rb(&d->ximat[b*9]);
      Vector3d bi(m->body_inertia[b*3],m->body_inertia[b*3+1],m->body_inertia[b*3+2]);
      Matrix3d Ib=Rb*bi.asDiagonal()*Rb.transpose();
      I += Ib + ms*(r.dot(r)*Matrix3d::Identity()-r*r.transpose()); }
    return I;
  }
  // setup_mpc: crouch_home 이후 호출. TROT_Q(arming 후 가중) 사용.
  void setup_mpc(){
    mj_forward(m,d);
    mpc.N=14; mpc.DT=0.02; mpc.TOTAL_MASS=m->body_subtreemass[0]; mpc.G_ACC=9.81;
    mpc.MU=MU*MU_MARGIN; mpc.LAMZ_MIN=1.0; mpc.LAMZ_MAX=2.0*mpc.TOTAL_MASS*9.81;
    mpc.I_BODY=compute_Icom();
    mpc.Qdiag.resize(13); mpc.Qdiag<<200,200,100, 0,0,200, 0,0,1, 10,10,1, 0;   // TROT_Q
    mpc.Rdiag=Vector3d(1e-6,1e-6,1e-6);
  }
  VectorXd body_x0(){
    double Rm[9]; mju_quat2Mat(Rm,&d->qpos[3]);
    Matrix<double,3,3,RowMajor> R(Rm);
    double pitch=std::asin(std::max(-1.0,std::min(1.0,-R(2,0))));
    double roll=std::atan2(R(2,1),R(2,2)), yaw=std::atan2(R(1,0),R(0,0));
    std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
    Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int cc=0;cc<nv;cc++) Jc(r,cc)=jcb[r*nv+cc];
    Map<VectorXd> qv(d->qvel,nv); Vector3d vcom=Jc*qv;
    Vector3d wb(d->qvel[3],d->qvel[4],d->qvel[5]); Vector3d omega_w=R*wb;
    VectorXd x(13); x<<roll,pitch,yaw, d->subtree_com[0],d->subtree_com[1],d->subtree_com[2],
      omega_w[0],omega_w[1],omega_w[2], vcom[0],vcom[1],vcom[2], -9.81;
    return x;
  }
  bool mpc_support_con=false; double mpc_support_margin=0.0;   // ★옵션2: 지지폴리곤 제약 on/off·마진(외부 세팅)
  Matrix<double,4,3> mpc_grf(const VectorXd& x_ref, const std::vector<std::array<int,4>>& cs){
    Vector3d com(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]);
    std::array<Vector3d,4> frel; for(int i=0;i<4;i++) frel[i]=foot_point(i)-com;
    std::vector<std::array<Vector3d,4>> fp(mpc.N,frel);
    mpc.support_poly.clear();   // ★옵션2: 현 stance 발 절대 xy로 지지폴리곤 구성(CoM을 이 안으로 hard constraint)
    if(mpc_support_con){ for(int i=0;i<4;i++) if(cs[0][i]){ Vector3d fpi=foot_point(i); mpc.support_poly.push_back(Vector2d(fpi[0],fpi[1])); }
      mpc.support_margin=mpc_support_margin; }
    return mpc_qp_plan(mpc,x0_or(x_ref),cs,fp,x_ref);
  }
  VectorXd x0_or(const VectorXd&){ return body_x0(); }  // (가독성용 wrapper)

  // ── wbic_stance (검증2와 동일) ──
  bool wbic_stance(){
    int K=4, nz=nv+3*K;
    std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
    Map<Matrix<double,Dynamic,Dynamic,RowMajor>> M(Mb.data(),nv,nv);
    Map<VectorXd> h(d->qfrc_bias,nv); Map<VectorXd> qv(d->qvel,nv);
    std::vector<Matrix<double,3,Dynamic>> Js(K); for(int k=0;k<K;k++) Js[k]=(sit_hock_contact&&k<2)?hock_jac(k):foot_jac(k);  // ★개-앉기=뒤 접촉 hock(발링크 평평)
    MatrixXd P=MatrixXd::Zero(nz,nz); VectorXd g=VectorXd::Zero(nz);
    std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
    Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
    Vector3d com(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]);
    Vector3d a_com=Vector3d(120,120,200).cwiseProduct(com_ref-com)-Vector3d(20,20,25).cwiseProduct(Jc*qv);
    P.topLeftCorner(nv,nv)+=Jc.transpose()*Jc; g.head(nv)-=Jc.transpose()*a_com;
    // ★자세 목표 = 현재 yaw 유지 + nose-up(sit_pitch). yaw를 0으로 안 되당김 → 선회 후 서기 삐뚫음/전복 방지(wbic_track과 동일 원리: 현재yaw 프레임서 roll/pitch만 레벨)
    double* qc=&d->qpos[3];
    double yaw_m=std::atan2(2*(qc[0]*qc[3]+qc[1]*qc[2]),1-2*(qc[2]*qc[2]+qc[3]*qc[3]));
    double qz[4]={std::cos(yaw_m/2),0,0,std::sin(yaw_m/2)}, qy[4]={std::cos(sit_pitch/2),0,-std::sin(sit_pitch/2),0}, qt[4];
    mju_mulQuat(qt,qz,qy);   // 목표=현재yaw·nose-up(sit_pitch=0이면 순수 현재yaw 수평)
    double oerr[3]; mju_subQuat(oerr,qc,qt);   // roll/pitch 오차(yaw≈0)
    for(int j=0;j<3;j++){ double a=150*(-oerr[j])-20*qv[3+j]; P(3+j,3+j)+=5.0; g[3+j]-=5.0*a; }
    for(int j=0;j<nu;j++){ double a, w;
      if(j==waist_idx){ a=WAIST_KP*(waist_ref-d->qpos[7+j])-WAIST_KD*qv[6+j]; w=WAIST_W; }  // ★허리 강홀드(서기서도)
      else { a=60*(q_home[j]-d->qpos[7+j])-5*qv[6+j];
        w=(stance_pin_ankle&&is_ankle[j])?20.0:posture_w; }   // ★17dof: 여유발목(4개) stance 핀→nullptr 표류. posture_w↑=개-앉기 접힘 홀드
      P(6+j,6+j)+=w; g[6+j]-=w*a; }
    P.topLeftCorner(nv,nv)+=1e-4*MatrixXd::Identity(nv,nv);
    for(int k=0;k<K;k++) P.block(nv+3*k,nv+3*k,3,3)+=1e-3*Matrix3d::Identity();
    int neq=6+3*K; MatrixXd A=MatrixXd::Zero(neq,nz); VectorXd b=VectorXd::Zero(neq);
    A.block(0,0,6,nv)=M.topRows(6); for(int j=0;j<6;j++) b[j]=-h[j];
    for(int k=0;k<K;k++) A.block(0,nv+3*k,6,3)=-Js[k].leftCols(6).transpose();
    for(int k=0;k<K;k++) A.block(6+3*k,0,3,nv)=Js[k];
    int nineq=5*K; MatrixXd CI=MatrixXd::Zero(nineq,nz); VectorXd ci0=VectorXd::Zero(nineq);
    int sgn[4][2]={{1,0},{-1,0},{0,1},{0,-1}}; int rr=0;
    for(int k=0;k<K;k++){ int o=nv+3*k;
      for(int s=0;s<4;s++){ CI(rr,o)=-sgn[s][0]; CI(rr,o+1)=-sgn[s][1]; CI(rr,o+2)=MU*MU_MARGIN; rr++; }
      CI(rr,o+2)=1.0; ci0[rr]=-LAMZ_MIN; rr++; }
    P=(0.5*(P+P.transpose())).eval()+1e-8*MatrixXd::Identity(nz,nz); VectorXd ce0=-b,x(nz);
    _qp_st.reset(nz,neq,nineq); auto st=_qp_st.solve_quadprog(P,g,A,ce0,CI,ci0,x);
    if(st!=eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) return false;
    VectorXd qdd=x.head(nv); VectorXd tau=M.block(6,0,nu,nv)*qdd+h.segment(6,nu);
    for(int k=0;k<K;k++) tau-=Js[k].block(0,6,3,nu).transpose()*x.segment(nv+3*k,3);
    for(int i=0;i<nu;i++){ double lim=tau_peak[i];
      if(motor_curve && w_limit[i]<1e7) lim=tau_peak[i]*std::max(0.0,1.0-std::abs(d->qvel[6+i])/w_limit[i]);  // ★고속서 가용토크↓(실모터 토크-속도곡선)
      d->ctrl[i]=std::max(-lim,std::min(lim,tau[i])); }
    couple_clamp(d);
    return true;
  }
  // ── wbic_track (검증3과 동일: 기본경로) ──
  // ★★strict null-space HQP-WBC (TAMOLS 논문충실, Bellicoso 우선순위): 접촉/EOM(하드) > swing(L1) > base+모멘텀+joint(L2).
  //   cascade 2-레벨: L1=swing만 풀어 최적값 v1 확보 → L2를 A_sw·z=v1(swing 동결) 등식 하에 풂 = base가 swing에 엄격 양보.
  //   weighted QP(wbic_track)와 차이: swing이 strict 상위라 base task가 swing 추종을 절대 침범 못함. HQP=1 env로 활성.
  bool wbic_track_hqp(const std::vector<int>& contacts, const std::map<int,std::pair<Vector3d,Vector3d>>& swing,
                      const Vector3d lam[4], double w_lam=10.0){
    int Kc=(int)contacts.size(), nzt=nv+3*Kc; auto sl=[&](int k){ return nv+3*k; };
    std::vector<Matrix<double,3,Dynamic>> cjac(Kc); std::vector<Vector3d> cpos(Kc),clam(Kc);
    for(int k=0;k<Kc;k++){ int c=contacts[k]; cjac[k]=foot_jac(c); cpos[k]=foot_point(c); clam[k]=lam[c]; }
    std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
    Map<Matrix<double,Dynamic,Dynamic,RowMajor>> M(Mb.data(),nv,nv);
    Map<VectorXd> h(d->qfrc_bias,nv); VectorXd qv=Map<VectorXd>(d->qvel,nv);
    std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
    Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
    Vector3d Jcqv=Jc*qv;
    // ── HARD (P0-P1): EOM(6)+contact no-motion(3Kc) 등식 · friction/torque/joint-accel/λz 부등식 ──
    int neq0=6+3*Kc; MatrixXd A0=MatrixXd::Zero(neq0,nzt); VectorXd b0=VectorXd::Zero(neq0);
    A0.block(0,0,6,nv)=M.topRows(6); b0.head(6)=-h.head(6);
    for(int k=0;k<Kc;k++) A0.block(0,sl(k),6,3)=-cjac[k].leftCols(6).transpose();
    for(int k=0;k<Kc;k++){ A0.block(6+3*k,0,3,nv)=cjac[k];
      if(STANCE_KD>0) b0.segment(6+3*k,3)=-STANCE_KD*(cjac[k]*qv); }
    std::vector<VectorXd> Gr; std::vector<double> hv; int sgn[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    for(int k=0;k<Kc;k++){ int o=sl(k); for(int s=0;s<4;s++){ VectorXd r=VectorXd::Zero(nzt);
      r[o]=sgn[s][0]; r[o+1]=sgn[s][1]; r[o+2]=-MU*MU_MARGIN; Gr.push_back(r); hv.push_back(0.0); } }
    VectorXd h_act=h.segment(6,nu); MatrixXd T_mat=MatrixXd::Zero(nu,nzt); T_mat.leftCols(nv)=M.block(6,0,nu,nv);
    for(int k=0;k<Kc;k++) T_mat.block(0,sl(k),nu,3)=-cjac[k].block(0,6,3,nu).transpose();
    for(int i=0;i<nu;i++){ Gr.push_back(T_mat.row(i)); hv.push_back(tau_peak[i]-h_act[i]); }
    for(int i=0;i<nu;i++){ Gr.push_back(-T_mat.row(i)); hv.push_back(tau_peak[i]+h_act[i]); }
    { double tla=0.05,c2=0.5*tla*tla;
      for(int j=0;j<nu;j++){ double qj=d->qpos[7+j],dqj=qv[6+j];
        double ubp=(qmax[j]-qj-dqj*tla)/c2, lbp=(qmin[j]-qj-dqj*tla)/c2;
        { VectorXd r=VectorXd::Zero(nzt); r[6+j]=1;  Gr.push_back(r); hv.push_back(ubp);  }   // q̈≤ubp
        { VectorXd r=VectorXd::Zero(nzt); r[6+j]=-1; Gr.push_back(r); hv.push_back(-lbp); } } }// q̈≥lbp
    for(int k=0;k<Kc;k++){ VectorXd r=VectorXd::Zero(nzt); r[sl(k)+2]=-1; Gr.push_back(r); hv.push_back(-LAMZ_MIN); }
    int nci=(int)Gr.size(); MatrixXd CI(nci,nzt); VectorXd ci0(nci);
    for(int i=0;i<nci;i++){ CI.row(i)=-Gr[i]; ci0[i]=hv[i]; }
    // ── 자세(roll/pitch)·z 목표가속 먼저 계산(L1 승격 옵션 대비) ──
    double* qc=&d->qpos[3];
    double yaw_m=std::atan2(2*(qc[0]*qc[3]+qc[1]*qc[2]),1-2*(qc[2]*qc[2]+qc[3]*qc[3]));
    double qlev[4]={std::cos(yaw_m/2),0,0,std::sin(yaw_m/2)};
    double oerr[3]; mju_subQuat(oerr,&d->qpos[3],qlev);
    double _okp=getenv("ORI_KP")?atof(getenv("ORI_KP")):150.0, _okd=getenv("ORI_KD")?atof(getenv("ORI_KD")):20.0;
    double _wori=getenv("W_ORI")?atof(getenv("W_ORI")):w_ori;
    double a_ori[2]; for(int j=0;j<2;j++) a_ori[j]=_okp*(-oerr[j])-_okd*qv[3+j];
    double zref=com_ref[2]+_body_terr;
    if(getenv("ZREF_FIX")) zref=atof(getenv("ZREF_FIX"));   // ★진단/고정 z 참조(plan z침하 우회)
    bool _zvff=!getenv("ZREF_FIX");   // 고정 z면 plan z-속도/가속 ff 무시(상수홀드)
    double _kpz=getenv("KP_Z")?atof(getenv("KP_Z")):200.0, _kdz=getenv("KD_Z")?atof(getenv("KD_Z")):25.0, _wz=getenv("W_Z")?atof(getenv("W_Z")):150.0;
    double a_z=_kpz*(zref-d->subtree_com[2])+_kdz*((_zvff?com_vel_ref[2]:0.0)-Jcqv[2])+(_zvff?com_acc_ref[2]:0.0);
    double yaw_err=std::atan2(std::sin(yaw_des-yaw_m),std::cos(yaw_des-yaw_m));   // ★MODE=tamols=MPC없음→WBC가 yaw 잡아야(방치시 스핀→붕괴)
    double _ykp=getenv("YAW_KP")?atof(getenv("YAW_KP")):150.0, _ykd=getenv("YAW_KD")?atof(getenv("YAW_KD")):20.0;
    double a_yaw=_ykp*yaw_err-_ykd*qv[5];
    // ── L1 (swing [+옵션 base 자세·yaw·z], P2): 엄격 상위 ──
    double _swkp=getenv("SW_KP")?atof(getenv("SW_KP")):2400.0, _swkd=getenv("SW_KD")?atof(getenv("SW_KD")):110.0;
    bool _basel1=getenv("HQP_BASE_L1"); int nbl=_basel1?4:0;   // ★구조옵션: base roll/pitch/yaw/z를 L1 strict로 승격(marginal 로봇=base 고우선, yaw방치=스핀사)
    int nsw=(int)swing.size(); std::set<int> sw_vidx; int nL1=3*nsw+nbl;
    // ★over-determination 가드: strict L2 등식(neq0+nL1)이 변수수(nzt) 이상이면 eiquadprog 내부버퍼 오버런=heap corruption.
    //   적은-DOF 로봇(예 Go2 nv=18, 12관절)서 nsw≥3 위상이면 6+3·nsw+nbl > nv → 초과. base-L1(nbl) 먼저 강등, 그래도 초과면 이 틱 stance 폴백.
    if(neq0+nL1>=nzt && nbl){ nbl=0; nL1=3*nsw; }   // base-L1 강등(swing 등식은 필수라 유지)
    if(neq0+nL1>=nzt) return false;                 // nsw 과다(flight성 위상)=이 틱 wbic_stance 폴백(크래시 방지)
    MatrixXd A1=MatrixXd::Zero(std::max(1,nL1),nzt); VectorXd b1=VectorXd::Zero(std::max(1,nL1));
    { int row=0; for(auto&kv:swing){ int leg=kv.first; Matrix<double,3,Dynamic> J=foot_jac(leg);
        Vector3d accel=_swkp*(kv.second.first-foot_point(leg))+_swkd*(kv.second.second-J*qv);
        A1.block(row,0,3,nv)=J; b1.segment(row,3)=accel; row+=3;
        for(int t=0;t<leg_dof[leg];t++) sw_vidx.insert(legqv[leg][t]); }
      if(nbl){   // ★base를 L1 strict 승격(HQP_BASE_L1). ※2-접촉 위상서 대각선 축 tilt는 점접촉 wrench 랭크결손으로 제어불가(재정식화로 못 고침, docs 참조)—스테핑(capture)이 담당.
        A1(row,3)=1; b1[row]=a_ori[0]; row++;                       // roll 가속(euler)
        A1(row,4)=1; b1[row]=a_ori[1]; row++;                       // pitch 가속(euler)
        A1(row,5)=1; b1[row]=a_yaw; row++;                          // yaw 가속(스핀 방지)
        A1.block(row,0,1,nv)=Jc.row(2); b1[row]=a_z; row++; } }     // z 가속(CoM)
    // ── L2 (base xy + 모멘텀 + joint + λ, P3-P4): 가중 QP (P,g) ──
    MatrixXd P=MatrixXd::Zero(nzt,nzt); VectorXd g=VectorXd::Zero(nzt);
    if(!_basel1){ for(int j=0;j<2;j++){ P(3+j,3+j)+=_wori; g[3+j]-=_wori*a_ori[j]; }   // 자세=L2(미승격시)
      double _wyaw=getenv("W_YAW")?atof(getenv("W_YAW")):w_yaw; P(5,5)+=_wyaw; g[5]-=_wyaw*a_yaw; }
    if(!_basel1){ P.topLeftCorner(nv,nv)+=_wz*(Jc.row(2).transpose()*Jc.row(2)); g.head(nv)-=_wz*a_z*Jc.row(2).transpose(); }  // z=L2(미승격시)
    if(W_BASE_XY>0){ for(int ax=0;ax<2;ax++){
      double a_xy=KP_BASE*(com_ref[ax]-d->subtree_com[ax])+KD_BASE*(com_vel_ref[ax]-Jcqv[ax])+com_acc_ref[ax];
      P.topLeftCorner(nv,nv)+=W_BASE_XY*(Jc.row(ax).transpose()*Jc.row(ax)); g.head(nv)-=W_BASE_XY*a_xy*Jc.row(ax).transpose(); } }
    for(int j=0;j<nu;j++){ double a_post,w_post;
      if(j==waist_idx){ a_post=WAIST_KP*(waist_ref-d->qpos[7+j])-WAIST_KD*qv[6+j]; w_post=WAIST_W; }
      else { a_post=60*(q_home[j]-d->qpos[7+j])-5*qv[6+j]; double sw=(is_front[j]?swing_w_f:swing_w_r);
        w_post=(is_ankle[j])?20.0:(sw_vidx.count(6+j)?sw:1.0);
        if(getenv("W_POST")&&!is_ankle[j]&&!sw_vidx.count(6+j)) w_post=atof(getenv("W_POST")); }
      P(6+j,6+j)+=w_post; g[6+j]-=w_post*a_post; }
    P.topLeftCorner(nv,nv)+=1e-3*MatrixXd::Identity(nv,nv);
    if(getenv("W_LAM")) w_lam=atof(getenv("W_LAM"));
    for(int k=0;k<Kc;k++){ P.block(sl(k),sl(k),3,3)+=w_lam*Matrix3d::Identity(); g.segment(sl(k),3)-=w_lam*clam[k]; }
    if(W_AM>0 && Kc>0){ mj_subtreeVel(m,d);
      Vector3d h_ang(d->subtree_angmom[0],d->subtree_angmom[1],d->subtree_angmom[2]);
      Vector3d hdes=-KD_AM*h_ang; Vector3d com(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]);
      MatrixXd A_am=MatrixXd::Zero(3,nzt);
      for(int k=0;k<Kc;k++){ Vector3d r=cpos[k]-com; int o=sl(k);
        A_am(0,o+1)=-r[2]; A_am(0,o+2)=r[1]; A_am(1,o)=r[2]; A_am(1,o+2)=-r[0]; A_am(2,o)=-r[1]; A_am(2,o+1)=r[0]; }
      P+=W_AM*(A_am.transpose()*A_am); g-=W_AM*(A_am.transpose()*hdes); }
    double _qreg=getenv("QREG")?atof(getenv("QREG")):1e-8;
    P=(0.5*(P+P.transpose())).eval()+_qreg*MatrixXd::Identity(nzt,nzt);
    // ── cascade solve ──
    VectorXd z(nzt);
    double _l1reg=getenv("L1REG")?atof(getenv("L1REG")):1e-6;
    if(nL1>0){ MatrixXd Ps=2.0*(A1.transpose()*A1); Ps.diagonal().array()+=_l1reg; Ps=(0.5*(Ps+Ps.transpose())).eval();
      VectorXd gs=-2.0*(A1.transpose()*b1);
      MatrixXd CE1=A0; VectorXd ce01=-b0;
      _qp_tr.reset(nzt,(int)CE1.rows(),nci);
      double r1=_qp_tr.solve_quadprog(Ps,gs,CE1,ce01,CI,ci0,z);
      if(r1!=eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) return false;
      VectorXd v1=A1*z;                                          // L1(swing[+base]) 최적 달성값(동결)
      MatrixXd CE2(neq0+nL1,nzt); CE2<<A0,A1; VectorXd ce02(neq0+nL1); ce02<<-b0,-v1;
      _qp_tr.reset(nzt,(int)CE2.rows(),nci);
      double r2=_qp_tr.solve_quadprog(P,g,CE2,ce02,CI,ci0,z);
      if(r2!=eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) return false;
    } else {                                                     // 스윙 없음(전지지)=L2만
      MatrixXd CE=A0; VectorXd ce0=-b0; _qp_tr.reset(nzt,(int)CE.rows(),nci);
      double r0=_qp_tr.solve_quadprog(P,g,CE,ce0,CI,ci0,z);
      if(r0!=eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) return false;
    }
    VectorXd qdd=z.head(nv); VectorXd tau=M.block(6,0,nu,nv)*qdd+h.segment(6,nu);
    for(int k=0;k<Kc;k++) tau-=cjac[k].block(0,6,3,nu).transpose()*z.segment(sl(k),3);
    for(int i=0;i<nu;i++){ double lim=tau_peak[i];
      if(motor_curve && w_limit[i]<1e7) lim=tau_peak[i]*std::max(0.0,1.0-std::abs(d->qvel[6+i])/w_limit[i]);
      d->ctrl[i]=std::max(-lim,std::min(lim,tau[i])); }
    couple_clamp(d);
    return true;
  }

  // ── ★2026-09-30 하이브리드 WBC / MIT 구성 (WBIC_MIT: 0=종전 · 1=MIT 전체(연구용) · 2=하이브리드=기본) ──
  //   하이브리드 = 기존 가중 QP(wbic_track) + 접촉·스윙 J̇q̇ + 추적 중 K_D 0(STANCE_KD_TRACK) + KinWBC 계획(mit_qdes/dqdes).
  //   계획 목표는 드라이버 PD 추종(DRV_TRACK=1, trot_sim·sim_bridge·robot_main 공통)에 쓴다. 근거: scratch 4족 sim
  //   실기수준(EST+SENSE2+ACT6ms+노이즈) trot0.3/0.6·walk·선회·제자리 전부 0낙상(MIT 전체는 같은 조건서 붕괴).
  VectorXd mit_qdes, mit_dqdes; bool mit_valid=false; long mit_n=0, mit_fail=0;
  Vector3d jacdot_foot(int i){ std::vector<double> jb(3*nv); Vector3d p=foot_point(i);
    mj_jacDot(m,d,jb.data(),nullptr,p.data(),fbid[i]);
    Vector3d r=Vector3d::Zero(); for(int rr=0;rr<3;rr++){ double s2=0; for(int c=0;c<nv;c++) s2+=jb[rr*nv+c]*d->qvel[c]; r[rr]=s2; } return r; }
  void build_mit_q(const std::vector<int>& contacts, const std::map<int,std::pair<Vector3d,Vector3d>>& swing,
                   const Vector3d lam[4], quadmit::In& in){
    using namespace quadmit; in.nv=nv; in.nu=nu; in.tasks.clear();
    std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
    in.M=Map<Matrix<double,Dynamic,Dynamic,RowMajor>>(Mb.data(),nv,nv);
    in.h=Map<VectorXd>(d->qfrc_bias,nv); VectorXd qv=Map<VectorXd>(d->qvel,nv); in.qv=qv; in.qj=Map<VectorXd>(&d->qpos[7],nu);
    const int Kc=(int)contacts.size(); in.Jc.resize(3*Kc,nv); in.JcDotQdot.resize(3*Kc); in.Fr_des.resize(3*Kc);
    for(int k=0;k<Kc;k++){ int c=contacts[k]; in.Jc.block(3*k,0,3,nv)=foot_jac(c); in.JcDotQdot.segment(3*k,3)=jacdot_foot(c); in.Fr_des.segment(3*k,3)=lam[c]; }
    in.mu=MU*MU_MARGIN; in.fz_min=LAMZ_MIN; in.fz_max=1e4; in.drv_peak=tau_peak;
    static const double WB=getenv("MIT_WB")?atof(getenv("MIT_WB")):0.1, WF=getenv("MIT_WF")?atof(getenv("MIT_WF")):1.0; in.W_b=WB; in.W_f=WF;
    { Task t; t.J=MatrixXd::Zero(3,nv); t.J.block(0,3,3,3).setIdentity();            // T1 몸통 자세 (roll·pitch 수평, yaw 헤딩)
      double* qc=&d->qpos[3]; double yaw_m=std::atan2(2*(qc[0]*qc[3]+qc[1]*qc[2]),1-2*(qc[2]*qc[2]+qc[3]*qc[3]));
      double qlev[4]={std::cos(yaw_m/2),0,0,std::sin(yaw_m/2)}, oe[3]; mju_subQuat(oe,&d->qpos[3],qlev);
      double ye=std::atan2(std::sin(yaw_des-yaw_m),std::cos(yaw_des-yaw_m));
      t.e=Vector3d(-oe[0],-oe[1],ye); t.xd_des=Vector3d::Zero();
      t.xdd_cmd=Vector3d(150*(-oe[0])-20*qv[3],150*(-oe[1])-20*qv[4],150*ye-20*qv[5]); t.JdotQdot=Vector3d::Zero(); in.tasks.push_back(t); }
    { Task t; std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);        // T2 몸통 위치(CoM): xy=계획속도 · z=높이
      MatrixXd Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
      Vector3d v=Jc*qv; double zref=com_ref[2]+_body_terr;
      t.J=Jc; t.e=Vector3d(0,0,zref-d->subtree_com[2]); t.xd_des=com_vel_ref;
      t.xdd_cmd=Vector3d(25*(com_vel_ref[0]-v[0])+com_acc_ref[0], 25*(com_vel_ref[1]-v[1])+com_acc_ref[1],
                         200*(zref-d->subtree_com[2])+25*(com_vel_ref[2]-v[2])+com_acc_ref[2]);
      t.JdotQdot=Vector3d::Zero(); in.tasks.push_back(t); }
    for(auto&kv:swing){ int leg=kv.first; Task t; t.J=foot_jac(leg); Vector3d p=foot_point(leg), vs=t.J*qv;   // T3 스윙 발
      t.e=kv.second.first-p; t.xd_des=kv.second.second; t.xdd_cmd=2400.0*(kv.second.first-p)+110.0*(kv.second.second-vs);
      t.JdotQdot=jacdot_foot(leg); in.tasks.push_back(t); }
    { Task t; t.J=MatrixXd::Zero(nu,nv); t.J.rightCols(nu).setIdentity(); t.e.resize(nu); t.xd_des=VectorXd::Zero(nu);   // T4 관절 자세
      t.xdd_cmd.resize(nu); t.JdotQdot=VectorXd::Zero(nu);
      for(int j=0;j<nu;j++){ double ref=(j==waist_idx)?waist_ref:q_home[j], kp=(j==waist_idx)?WAIST_KP:60.0, kd=(j==waist_idx)?WAIST_KD:5.0;
        t.e[j]=ref-d->qpos[7+j]; t.xdd_cmd[j]=kp*t.e[j]-kd*qv[6+j]; }
      in.tasks.push_back(t); }
  }
  bool wbic_track_mit(const std::vector<int>& contacts, const std::map<int,std::pair<Vector3d,Vector3d>>& swing, const Vector3d lam[4]){
    quadmit::In in; build_mit_q(contacts,swing,lam,in); quadmit::Out o=quadmit::solve(in); mit_n++; if(!o.qp_ok) mit_fail++;
    mit_qdes=o.qdes; mit_dqdes=o.dqdes; mit_valid=true;
    for(int i=0;i<nu;i++){ double lim=tau_peak[i]; d->ctrl[i]=std::max(-lim,std::min(lim,o.tau[i])); }
    couple_clamp(d); return true; }

  bool wbic_track(const std::vector<int>& contacts, const std::map<int,std::pair<Vector3d,Vector3d>>& swing,
                  const Vector3d lam[4], double w_lam=10.0){
    static const int WMIT=getenv("WBIC_MIT")?atoi(getenv("WBIC_MIT")):2;   // ★0=종전 · 1=MIT 전체(연구용) · 2=하이브리드(기본)
    static const double KD_TRK=getenv("STANCE_KD_TRACK")?atof(getenv("STANCE_KD_TRACK")):0.0;   // ★하이브리드 추적 중 접촉 K_D(기본 0=폐기)
    if(WMIT==1 && !contacts.empty()) return wbic_track_mit(contacts, swing, lam);
    if(getenv("HQP")) return wbic_track_hqp(contacts, swing, lam, w_lam);   // ★strict null-space HQP(논문충실)
    int Kc=(int)contacts.size(), nzt=nv+3*Kc; auto sl=[&](int k){ return nv+3*k; };
    std::vector<Matrix<double,3,Dynamic>> cjac(Kc); std::vector<Vector3d> cpos(Kc),clam(Kc);
    for(int k=0;k<Kc;k++){ int c=contacts[k]; cjac[k]=foot_jac(c); cpos[k]=foot_point(c); clam[k]=lam[c]; }
    std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
    Map<Matrix<double,Dynamic,Dynamic,RowMajor>> M(Mb.data(),nv,nv);
    Map<VectorXd> h(d->qfrc_bias,nv); VectorXd qv=Map<VectorXd>(d->qvel,nv);
    MatrixXd P=MatrixXd::Zero(nzt,nzt); VectorXd g=VectorXd::Zero(nzt);
    std::set<int> sw_vidx;
    for(auto&kv:swing){ int leg=kv.first; Matrix<double,3,Dynamic> J=foot_jac(leg);
      Vector3d accel=2400.0*(kv.second.first-foot_point(leg))+110.0*(kv.second.second-J*qv);
      if(WMIT==2) accel-=jacdot_foot(leg);   // ★하이브리드: J q̈ = a − J̇q̇
      P.topLeftCorner(nv,nv)+=SW_TRACK_W*(J.transpose()*J); g.head(nv)-=SW_TRACK_W*(J.transpose()*accel);
      for(int t=0;t<leg_dof[leg];t++) sw_vidx.insert(legqv[leg][t]); }
    std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
    Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
    // ★자세 task 분리: roll/pitch=정확한 현재yaw 프레임서 레벨링(회전 무관 정확) + yaw축=명령헤딩(yaw_des) 약추종.
    //   단위자세 대비로 하면 회전할수록 yaw를 0으로 되당겨 ~120°서 붕괴. yaw_ref 프레임서 레벨링하면 고속선회시 0.3rad 틀어져 붕괴.
    double* qc=&d->qpos[3];
    double yaw_m=std::atan2(2*(qc[0]*qc[3]+qc[1]*qc[2]),1-2*(qc[2]*qc[2]+qc[3]*qc[3]));
    double qlev[4]={std::cos(yaw_m/2),0,0,std::sin(yaw_m/2)};     // 현재 yaw에서 수평(정확 프레임)
    double oerr[3]; mju_subQuat(oerr,&d->qpos[3],qlev);           // roll/pitch 오차(yaw≈0)
    for(int j=0;j<2;j++){ double a=150*(-oerr[j])-20*qv[3+j];
      if(getenv("GM_KI")){ double ki=atof(getenv("GM_KI"))*0.5, dt2=m->opt.timestep; double& gi=(j==0?gm_ri:gm_pi);   // ★GM 적분보상(자세 정상상태 잔차)
        gi+=(-oerr[j])*dt2; gi=std::max(-0.4,std::min(0.4,gi)); a+=ki*gi; }
      P(3+j,3+j)+=w_ori; g[3+j]-=w_ori*a; }
    double yaw_err=std::atan2(std::sin(yaw_des-yaw_m),std::cos(yaw_des-yaw_m));  // 헤딩오차(wrap안전)
    double a_yaw=150*yaw_err-20*qv[5]; P(5,5)+=w_yaw; g[5]-=w_yaw*a_yaw;         // yaw 헤딩홀드(직진 드리프트 방지, 선회시 yaw_des 추종→안싸움)
    double zref=com_ref[2]+_body_terr; Vector3d Jcqv=Jc*qv;
    double _kpz=getenv("KP_Z")?atof(getenv("KP_Z")):200.0, _kdz=getenv("KD_Z")?atof(getenv("KD_Z")):25.0, _wz=getenv("W_Z")?atof(getenv("W_Z")):150.0;
    double a_z=_kpz*(zref-d->subtree_com[2])+_kdz*(com_vel_ref[2]-Jcqv[2])+com_acc_ref[2];   // ★2층 WBC z유지: 계획 z속도·가속 추종(ff=예측 보강, 접촉전이 대비). MPC 예측 대체
    double _gmki=getenv("GM_KI")?atof(getenv("GM_KI")):0.0, _gmdt=m->opt.timestep;   // ★GM-observer 적분보상(정상상태 z침하=지속잔차 상쇄)
    if(_gmki>0){ gm_zi+=(zref-d->subtree_com[2])*_gmdt; gm_zi=std::max(-0.6,std::min(0.6,gm_zi)); a_z+=_gmki*gm_zi; }
    P.topLeftCorner(nv,nv)+=_wz*(Jc.row(2).transpose()*Jc.row(2)); g.head(nv)-=_wz*a_z*Jc.row(2).transpose();
    if(W_BASE_XY>0){ for(int ax=0;ax<2;ax++){   // ★RSL식: base 수평(x,y) 위치를 WBC가 직접 추종(SRBD MPC 대체) — 계획 base궤적 위치레벨 execute
      double a_xy=KP_BASE*(com_ref[ax]-d->subtree_com[ax])+KD_BASE*(com_vel_ref[ax]-Jcqv[ax])+com_acc_ref[ax];
      P.topLeftCorner(nv,nv)+=W_BASE_XY*(Jc.row(ax).transpose()*Jc.row(ax)); g.head(nv)-=W_BASE_XY*a_xy*Jc.row(ax).transpose(); } }
    for(int j=0;j<nu;j++){
      double a_post, w_post;
      if(j==waist_idx){ a_post=WAIST_KP*(waist_ref-d->qpos[7+j])-WAIST_KD*qv[6+j]; w_post=WAIST_W; }  // ★허리: 강한 전용홀드(요각목표=waist_ref)
      else { a_post=60*(q_home[j]-d->qpos[7+j])-5*qv[6+j];
        double sw=(is_front[j]?swing_w_f:swing_w_r);                            // 앞/뒤 스윙 여유도 별도
        w_post = (is_ankle[j])?20.0 : (sw_vidx.count(6+j)?sw:1.0); }            // ↑=calf/thigh whip 억제
      P(6+j,6+j)+=w_post; g[6+j]-=w_post*a_post; }
    P.topLeftCorner(nv,nv)+=1e-3*MatrixXd::Identity(nv,nv);
    for(int k=0;k<Kc;k++){ P.block(sl(k),sl(k),3,3)+=w_lam*Matrix3d::Identity(); g.segment(sl(k),3)-=w_lam*clam[k]; }
    // 각운동량 보상(leg-heavy 고속): Σ rᵢ×λᵢ ≈ −KD_AM·h_ω (17dof=5, 14dof평지=0)
    if(W_AM>0 && Kc>0){ mj_subtreeVel(m,d);
      Vector3d h_ang(d->subtree_angmom[0],d->subtree_angmom[1],d->subtree_angmom[2]);
      Vector3d hdes=-KD_AM*h_ang; Vector3d com(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]);
      MatrixXd A_am=MatrixXd::Zero(3,nzt);
      for(int k=0;k<Kc;k++){ Vector3d r=cpos[k]-com; int o=sl(k);
        A_am(0,o+1)=-r[2]; A_am(0,o+2)=r[1]; A_am(1,o)=r[2]; A_am(1,o+2)=-r[0]; A_am(2,o)=-r[1]; A_am(2,o+1)=r[0]; }
      P+=W_AM*(A_am.transpose()*A_am); g-=W_AM*(A_am.transpose()*hdes); }
    int neq=6+3*Kc; MatrixXd A=MatrixXd::Zero(neq,nzt); VectorXd b=VectorXd::Zero(neq);
    A.block(0,0,6,nv)=M.topRows(6); b.head(6)=-h.head(6);
    for(int k=0;k<Kc;k++) A.block(0,sl(k),6,3)=-cjac[k].leftCols(6).transpose();
    for(int k=0;k<Kc;k++){ A.block(6+3*k,0,3,nv)=cjac[k];
      // ★하이브리드: J q̈ + J̇q̇ = −K_D·J q̇ 에서 K_D=STANCE_KD_TRACK(기본 0). 종전(WMIT=0)은 STANCE_KD(20) 그대로.
      const double kdc=(WMIT==2)?KD_TRK:STANCE_KD;
      Vector3d rc=Vector3d::Zero(); if(kdc>0) rc=-kdc*(cjac[k]*qv); if(WMIT==2) rc-=jacdot_foot(contacts[k]);
      b.segment(6+3*k,3)=rc; }
    VectorXd lb=VectorXd::Constant(nzt,-1e8),ub=VectorXd::Constant(nzt,1e8);
    { double tla=0.05,c2=0.5*tla*tla;
      for(int j=0;j<nu;j++){ double qj=d->qpos[7+j],dqj=qv[6+j];
        double ubp=(qmax[j]-qj-dqj*tla)/c2, lbp=(qmin[j]-qj-dqj*tla)/c2;
        double u=std::min(ub[6+j],ubp), l=std::max(lb[6+j],lbp);
        if(l<=u){ ub[6+j]=u; lb[6+j]=l; } } }
    for(int k=0;k<Kc;k++) lb[sl(k)+2]=LAMZ_MIN;
    std::vector<VectorXd> Gr; std::vector<double> hv;
    int sgn[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    for(int k=0;k<Kc;k++){ int o=sl(k); for(int s=0;s<4;s++){ VectorXd r=VectorXd::Zero(nzt);
      r[o]=sgn[s][0]; r[o+1]=sgn[s][1]; r[o+2]=-MU*MU_MARGIN; Gr.push_back(r); hv.push_back(0.0); } }
    VectorXd h_act=h.segment(6,nu); MatrixXd T_mat=MatrixXd::Zero(nu,nzt); T_mat.leftCols(nv)=M.block(6,0,nu,nv);
    for(int k=0;k<Kc;k++) T_mat.block(0,sl(k),nu,3)=-cjac[k].block(0,6,3,nu).transpose();
    for(int i=0;i<nu;i++){ Gr.push_back(T_mat.row(i)); hv.push_back(tau_peak[i]-h_act[i]); }
    for(int i=0;i<nu;i++){ Gr.push_back(-T_mat.row(i)); hv.push_back(tau_peak[i]+h_act[i]); }
    P=(0.5*(P+P.transpose())).eval()+1e-8*MatrixXd::Identity(nzt,nzt);
    std::vector<VectorXd> CIr; std::vector<double> ci0v;
    for(size_t i=0;i<Gr.size();i++){ CIr.push_back(-Gr[i]); ci0v.push_back(hv[i]); }
    for(int i=0;i<nzt;i++){ if(lb[i]>-1e7){ VectorXd r=VectorXd::Zero(nzt); r[i]=1; CIr.push_back(r); ci0v.push_back(-lb[i]); }
                            if(ub[i]< 1e7){ VectorXd r=VectorXd::Zero(nzt); r[i]=-1; CIr.push_back(r); ci0v.push_back(ub[i]); } }
    int nci=(int)CIr.size(); MatrixXd CI(nci,nzt); VectorXd ci0(nci);
    for(int i=0;i<nci;i++){ CI.row(i)=CIr[i]; ci0[i]=ci0v[i]; }
    MatrixXd CE=A; VectorXd ce0=-b, x(nzt);
    _qp_tr.reset(nzt,neq,nci); auto st=_qp_tr.solve_quadprog(P,g,CE,ce0,CI,ci0,x);
    if(st!=eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) return false;
    VectorXd qdd=x.head(nv); VectorXd tau=M.block(6,0,nu,nv)*qdd+h.segment(6,nu);
    for(int k=0;k<Kc;k++) tau-=cjac[k].block(0,6,3,nu).transpose()*x.segment(sl(k),3);
    for(int i=0;i<nu;i++){ double lim=tau_peak[i];
      if(motor_curve && w_limit[i]<1e7) lim=tau_peak[i]*std::max(0.0,1.0-std::abs(d->qvel[6+i])/w_limit[i]);  // ★고속서 가용토크↓(실모터 토크-속도곡선)
      d->ctrl[i]=std::max(-lim,std::min(lim,tau[i])); }
    couple_clamp(d);
    if(WMIT==2){ quadmit::In kin; build_mit_q(contacts,swing,lam,kin); quadmit::kinwbc(kin,mit_qdes,mit_dqdes); mit_valid=true; }   // ★KinWBC 드라이버 목표
    return true;
  }

  // ── wbic_jump (점프 WBIC-추종: OCP CoM궤적을 가속 피드포워드로 추종+base를 GRF로 닫음. Python wbic_jump 포팅) ──
  //   접촉(push/land)=CoM 3-DOF 추종+접촉/마찰, flight(K=0)=관절·자세만. λ는 자유(reg만). τ=M q̈+h−Jᵀλ.
  bool wbic_jump(const Vector3d& com_ref, const Vector3d& comv_ref, const Vector3d& acom_ref,
                 const VectorXd& q_ref, const VectorXd& dq_ref, const std::vector<int>& contacts,
                 double kp_lin=120, double kd_lin=22, double kp_j=160, double kd_j=12,
                 double w_lin=120, double w_ori=8.0, double w_j=2.0, double w_lam=0.1){
    int Kc=(int)contacts.size(), nzt=nv+3*Kc; auto sl=[&](int k){ return nv+3*k; };
    std::vector<Matrix<double,3,Dynamic>> cjac(Kc);
    for(int k=0;k<Kc;k++) cjac[k]=foot_jac(contacts[k]);
    std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
    Map<Matrix<double,Dynamic,Dynamic,RowMajor>> M(Mb.data(),nv,nv);
    Map<VectorXd> h(d->qfrc_bias,nv); VectorXd qv=Map<VectorXd>(d->qvel,nv);
    MatrixXd P=MatrixXd::Zero(nzt,nzt); VectorXd g=VectorXd::Zero(nzt);
    // ① CoM 3-DOF 추종(가속 피드포워드) — 접촉 시만(flight=탄도, 제어불가)
    if(Kc>0){
      std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
      Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
      Vector3d com(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]);
      Vector3d comv=Jc*qv;
      Vector3d a_lin=acom_ref+kp_lin*(com_ref-com)+kd_lin*(comv_ref-comv);
      P.topLeftCorner(nv,nv)+=w_lin*(Jc.transpose()*Jc); g.head(nv)-=w_lin*(Jc.transpose()*a_lin);
    }
    // ② 자세 레벨링 — ★현재 yaw 프레임서 roll/pitch만 레벨(yaw는 안 되당김). 선회 후 점프해도 스핀 안 함(wbic_track·wbic_stance와 동일 원리).
    //   기존엔 단위자세(yaw=0) 대비 3축이라 몸통 돌린 뒤 점프하면 yaw를 0으로 되당겨 점프 중 회전. yaw는 자유(현재 헤딩 유지).
    const double* qc=&d->qpos[3];
    double yaw_j=std::atan2(2*(qc[0]*qc[3]+qc[1]*qc[2]),1-2*(qc[2]*qc[2]+qc[3]*qc[3]));
    double qlev[4]={std::cos(yaw_j/2),0,0,std::sin(yaw_j/2)};   // 현재 yaw서 수평(정확 프레임)
    double oerr[3]; mju_subQuat(oerr,&d->qpos[3],qlev);          // roll/pitch 오차(yaw≈0)
    for(int j=0;j<2;j++){ double a=150*(-oerr[j])-20*qv[3+j]; P(3+j,3+j)+=w_ori; g[3+j]-=w_ori*a; }   // roll/pitch만(yaw=j2 자유)
    // ③ 관절 posture(OCP q_ref/dq_ref 전관절 추종=발목 flail 억제)
    for(int j=0;j<nu;j++){ double a=kp_j*(q_ref[j]-d->qpos[7+j])+kd_j*(dq_ref[j]-qv[6+j]); P(6+j,6+j)+=w_j; g[6+j]-=w_j*a; }
    P.topLeftCorner(nv,nv)+=1e-3*MatrixXd::Identity(nv,nv);
    for(int k=0;k<Kc;k++) P.block(sl(k),sl(k),3,3)+=w_lam*Matrix3d::Identity();   // λ는 reg만(MPC GRF 추종 없음)
    // 등식: base EOM(6) + 접촉 no-accel baumgarte(3K)
    int neq=6+3*Kc; MatrixXd A=MatrixXd::Zero(neq,nzt); VectorXd b=VectorXd::Zero(neq);
    A.block(0,0,6,nv)=M.topRows(6); b.head(6)=-h.head(6);
    for(int k=0;k<Kc;k++) A.block(0,sl(k),6,3)=-cjac[k].leftCols(6).transpose();
    for(int k=0;k<Kc;k++){ A.block(6+3*k,0,3,nv)=cjac[k];
      if(STANCE_KD>0) b.segment(6+3*k,3)=-STANCE_KD*(cjac[k]*qv); }
    // 관절 위치한계(가속 상하한)
    VectorXd lb=VectorXd::Constant(nzt,-1e8),ub=VectorXd::Constant(nzt,1e8);
    { double tla=0.05,c2=0.5*tla*tla;
      for(int j=0;j<nu;j++){ double qj=d->qpos[7+j],dqj=qv[6+j];
        double ubp=(qmax[j]-qj-dqj*tla)/c2, lbp=(qmin[j]-qj-dqj*tla)/c2;
        double u=std::min(ub[6+j],ubp), l=std::max(lb[6+j],lbp);
        if(l<=u){ ub[6+j]=u; lb[6+j]=l; } } }
    for(int k=0;k<Kc;k++) lb[sl(k)+2]=LAMZ_MIN;
    // 부등식: 마찰추 + 토크한계
    std::vector<VectorXd> Gr; std::vector<double> hv;
    int sgn[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    for(int k=0;k<Kc;k++){ int o=sl(k); for(int s=0;s<4;s++){ VectorXd r=VectorXd::Zero(nzt);
      r[o]=sgn[s][0]; r[o+1]=sgn[s][1]; r[o+2]=-MU*MU_MARGIN; Gr.push_back(r); hv.push_back(0.0); } }
    VectorXd h_act=h.segment(6,nu); MatrixXd T_mat=MatrixXd::Zero(nu,nzt); T_mat.leftCols(nv)=M.block(6,0,nu,nv);
    for(int k=0;k<Kc;k++) T_mat.block(0,sl(k),nu,3)=-cjac[k].block(0,6,3,nu).transpose();
    for(int i=0;i<nu;i++){ Gr.push_back(T_mat.row(i)); hv.push_back(tau_peak[i]-h_act[i]); }
    for(int i=0;i<nu;i++){ Gr.push_back(-T_mat.row(i)); hv.push_back(tau_peak[i]+h_act[i]); }
    P=(0.5*(P+P.transpose())).eval()+1e-8*MatrixXd::Identity(nzt,nzt);
    std::vector<VectorXd> CIr; std::vector<double> ci0v;
    for(size_t i=0;i<Gr.size();i++){ CIr.push_back(-Gr[i]); ci0v.push_back(hv[i]); }
    for(int i=0;i<nzt;i++){ if(lb[i]>-1e7){ VectorXd r=VectorXd::Zero(nzt); r[i]=1; CIr.push_back(r); ci0v.push_back(-lb[i]); }
                            if(ub[i]< 1e7){ VectorXd r=VectorXd::Zero(nzt); r[i]=-1; CIr.push_back(r); ci0v.push_back(ub[i]); } }
    int nci=(int)CIr.size(); MatrixXd CI(nci,nzt); VectorXd ci0(nci);
    for(int i=0;i<nci;i++){ CI.row(i)=CIr[i]; ci0[i]=ci0v[i]; }
    MatrixXd CE=A; VectorXd ce0=-b, x(nzt);
    _qp_jm.reset(nzt,neq,nci); auto st=_qp_jm.solve_quadprog(P,g,CE,ce0,CI,ci0,x);
    if(st!=eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) return false;
    VectorXd qdd=x.head(nv); VectorXd tau=M.block(6,0,nu,nv)*qdd+h.segment(6,nu);
    for(int k=0;k<Kc;k++) tau-=cjac[k].block(0,6,3,nu).transpose()*x.segment(sl(k),3);
    for(int i=0;i<nu;i++){ double lim=tau_peak[i];
      if(motor_curve && w_limit[i]<1e7) lim=tau_peak[i]*std::max(0.0,1.0-std::abs(d->qvel[6+i])/w_limit[i]);
      d->ctrl[i]=std::max(-lim,std::min(lim,tau[i])); }
    couple_clamp(d);
    return true;
  }
};
