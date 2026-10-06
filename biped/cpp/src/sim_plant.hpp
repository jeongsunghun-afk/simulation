#pragma once
// ★2026-10-06 실기 유사 플랜트 — 드라이버 PD 추종 · FF 처리 · 드라이브 한계 · **모터–링크 탄성(벨트·백래시)**.
//
// 왜: 강체 sim 은 실기 떨림(12–22 Hz)을 재현하지 못한다. 10-02 실기 분석 결과 떨림은
//   ① 전달부 탄성(calf·foot 벨트, hip roll 백래시 — 출력축이 모터의 1.5~2배 흔들림)이 공진을 만들고
//   ② 매 걸음 착지 충격·명령 계단이 그것을 두드리고 ③ 8~14 ms 지연된 되먹임이 감쇠 대신 에너지를 넣어 생긴다.
//   ①이 없는 강체 플랜트에서는 같은 지연을 넣어도 떨지 않는다. 이 모듈은 ①을 넣어 떨림을 sim 에서 판정할 수 있게 한다.
//   이족·사족이 같은 다리 모듈(MD80 + 벨트)이라 사족 sim 에도 그대로 쓴다(구동 좌표 = 액추에이터 전달 좌표).
//
// 모델(구동 좌표 a, 액추에이터 i 마다):
//   링크측 구동각  s  = A q            (A: 액추에이터 전달 — 관절 1개 또는 fixed tendon 계수, gear 포함)
//   모터(로터)     J_m ω̇_m = τ_drv − τ_t − b_m ω_m
//   전달 토크      τ_t = k · dz(θ_m − s, bl) + c · (ω_m − ṡ)        (dz = 백래시 사영대)
//   드라이버       τ_drv = sat( s_ff·FF + kp(θ_des − θ_m) + kd(θ̇_des − ω_m) )   ← 드라이버 내부라 **지연 없음**
//   링크에는 d->ctrl = τ_t 가 액추에이터 전달로 걸린다. 제어기 센서 = 모터측(A⁻¹θ_m), 출력축 aux = 링크.
//   J_m 은 setup_gearbox 가 넣은 로터 반사관성(dof_armature / tendon_armature)을 **옮겨** 쓴다(플랜트 모델에선 0).
//   탄성 끔(EL_K 미지정)이면 θ_m≡s 로 두고 드라이버 PD 만 고속으로 돈다(강체 + 드라이버 추종).
//
// env (전부 미지정 = 종전과 같은 동작):
//   WALK_TRACK=1          드라이버 추종 PD 사용(제어기 KinWBC q_des·q̇_des). 미지정이면 순수 토크(종전).
//   TRK_KP / TRK_KD       드라이버 kp/kd 배율(실기 run_deploy_hw.sh 와 같은 이름). 기본 1.0 / 1.0
//   DRV_KP / DRV_KD       구동좌표 기본 게인 4개(hip,thigh,calf,foot) — 기본 100,50,180,43.2 / 6,4,7.9,2.9 (실기 raw)
//   WALK_FF_SCALE         FF 비율(기본 1). WALK_FF_SCALE_CH 로 4개(hip,thigh,calf,foot) 따로.
//   WALK_FF_LPF_HZ        FF 1차 저역통과(기본 0=끔).   WALK_FF_NOTCH_HZ / _Q  FF 노치(기본 0=끔, Q 2)
//   PLANT_DRV_CH          드라이브 채널 토크한계 Nm(기본 0 = MJCF ctrlrange). 구동좌표 한계 = CH × gear_k(1,1,1.5,1.2)
//   EL_K                  탄성 강성 4개 [Nm/rad, 구동좌표] (hip,thigh,calf,foot). 지정 시 탄성 켬.
//   EL_ZETA               전달부 감쇠비(기본 0.05, c = 2ζ√(k·J_m))
//   EL_BL_DEG             백래시 반폭 4개 [deg] (기본 0)
//   PLANT_SUB             물리 서브스텝 수(기본: 탄성 4 · 아니면 1). 제어 2 ms 는 그대로.
//   EL_FRIC_ROTOR         탄성 시 감속기 마찰(점성 damping·쿨롱 frictionloss)을 링크에서 로터로 옮김(기본 1).
//                         실제 감속기 마찰은 모터 쪽에 있어 로터 공진을 누른다 — 링크에 두면 그 감쇠가 빠진다.
#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

struct SimPlant {
  int nu=0, nv=0, NJ=0;
  bool track=false, elastic=false;
  int sub=1;
  double trk_kp=1.0, trk_kd=1.0;
  std::vector<double> kp, kd, ffs, lim, k, cdmp, bl, Jm, bm, fc;
  std::vector<int> grp;                 // 0 hip · 1 thigh · 2 calf · 3 foot (액추에이터 이름으로)
  Eigen::MatrixXd A, Ainv;              // s = A q_leg (nu × NJ), 다리 관절만
  std::vector<double> th, om;           // 로터 상태(구동 좌표)
  // FF 필터
  double lpf_hz=0, notch_hz=0, notch_q=2.0; bool finit=false;
  std::vector<double> lpf_y, nz;        // nz: 4 per actuator (x1,x2,y1,y2)
  std::vector<double> ff_out;           // 마지막으로 낸 FF(로그용, 구동좌표)
  std::vector<double> drv_out;          // 마지막 드라이브 토크(로그용)
  mjModel* mp=nullptr;                  // 플랜트 모델(제어기 모델의 복사본)

  static std::vector<double> envv(const char* e, std::vector<double> def){
    const char* v=getenv(e); if(!v) return def;
    std::vector<double> o; const char* p=v; char* end;
    while(*p){ double x=strtod(p,&end); if(end==p) break; o.push_back(x); p=end; while(*p==','||*p==' ') p++; }
    if(o.size()==1) o.assign(def.size(), o[0]);
    if(o.size()<def.size()) o.resize(def.size(), def.back());
    return o;
  }
  static double envd(const char* e, double def){ const char* v=getenv(e); return v?atof(v):def; }

  // 제어기 모델 m(setup_gearbox 끝난 상태)로부터 플랜트 모델을 만든다.
  void init(mjModel* m, mjData* d){
    nu=m->nu; nv=m->nv; NJ=m->nq-7;
    track = getenv("WALK_TRACK") && atoi(getenv("WALK_TRACK"));
    trk_kp=envd("TRK_KP",1.0); trk_kd=envd("TRK_KD",1.0);
    std::vector<double> KP4=envv("DRV_KP",{100,50,180,43.2}), KD4=envv("DRV_KD",{6,4,7.9,2.9});
    std::vector<double> FF4=envv("WALK_FF_SCALE_CH", std::vector<double>(4, envd("WALK_FF_SCALE",1.0)));
    std::vector<double> K4=envv("EL_K",{0,0,0,0}), BL4=envv("EL_BL_DEG",{0,0,0,0});
    elastic = getenv("EL_K")!=nullptr;
    const double zeta=envd("EL_ZETA",0.05), CH=envd("PLANT_DRV_CH",0.0);
    const double GK[4]={1.0,1.0,1.5,1.2};
    lpf_hz=envd("WALK_FF_LPF_HZ",0.0); notch_hz=envd("WALK_FF_NOTCH_HZ",0.0); notch_q=envd("WALK_FF_NOTCH_Q",2.0);
    sub = (int)envd("PLANT_SUB", elastic?4.0:1.0); if(sub<1) sub=1;
    // 구동좌표 전달행렬 A (다리 관절 NJ 열)
    A=Eigen::MatrixXd::Zero(nu,NJ);
    for(int i=0;i<nu;i++){
      const double g=m->actuator_gear[6*i];
      if(m->actuator_trntype[i]==mjTRN_JOINT){
        int jnt=m->actuator_trnid[2*i]; int qa=m->jnt_qposadr[jnt]-7; if(qa>=0&&qa<NJ) A(i,qa)=g;
      } else if(m->actuator_trntype[i]==mjTRN_TENDON){
        int t=m->actuator_trnid[2*i];
        for(int w=m->tendon_adr[t]; w<m->tendon_adr[t]+m->tendon_num[t]; w++)
          if(m->wrap_type[w]==mjWRAP_JOINT){ int jnt=m->wrap_objid[w]; int qa=m->jnt_qposadr[jnt]-7;
            if(qa>=0&&qa<NJ) A(i,qa)+=g*m->wrap_prm[w]; }
      }
    }
    Ainv = (A.rows()==A.cols()) ? Eigen::MatrixXd(A.inverse()) : Eigen::MatrixXd(A.completeOrthogonalDecomposition().pseudoInverse());
    kp.assign(nu,0); kd.assign(nu,0); ffs.assign(nu,1); lim.assign(nu,1e9); k.assign(nu,0); cdmp.assign(nu,0);
    bl.assign(nu,0); Jm.assign(nu,0); bm.assign(nu,0); fc.assign(nu,0); grp.assign(nu,0);
    const bool frot = elastic && envd("EL_FRIC_ROTOR",1.0)>0.5;
    mp = mj_copyModel(nullptr, m);
    for(int i=0;i<nu;i++){
      const char* nm=mj_id2name(m,mjOBJ_ACTUATOR,i); std::string s=nm?nm:"";
      int gI = s.find("hip")!=std::string::npos?0 : s.find("thigh")!=std::string::npos?1 : s.find("calf")!=std::string::npos?2 : 3;
      grp[i]=gI;
      kp[i]=KP4[gI]*trk_kp; kd[i]=KD4[gI]*trk_kd; ffs[i]=FF4[gI];
      lim[i] = CH>0 ? CH*GK[gI] : std::max(std::fabs(m->actuator_ctrlrange[2*i]), std::fabs(m->actuator_ctrlrange[2*i+1]));
      // 로터 반사관성: 관절 전달이면 dof_armature, tendon 이면 tendon_armature 에 있다 → 플랜트에선 떼어 로터로 옮긴다.
      if(m->actuator_trntype[i]==mjTRN_JOINT){ int jnt=m->actuator_trnid[2*i]; int dof=m->jnt_dofadr[jnt];
        Jm[i]=m->dof_armature[dof]; if(elastic) mp->dof_armature[dof]=0.0;
        if(frot){ bm[i]=m->dof_damping[dof]; fc[i]=m->dof_frictionloss[dof]; mp->dof_damping[dof]=0.0; mp->dof_frictionloss[dof]=0.0; } }
      else if(m->actuator_trntype[i]==mjTRN_TENDON){ int t=m->actuator_trnid[2*i];
        Jm[i]=m->tendon_armature[t]; if(elastic) mp->tendon_armature[t]=0.0;
        if(frot){ bm[i]=m->tendon_damping[t]; fc[i]=m->tendon_frictionloss[t]; mp->tendon_damping[t]=0.0; mp->tendon_frictionloss[t]=0.0; } }
      if(elastic){
        k[i]=K4[gI]; bl[i]=BL4[gI]*M_PI/180.0;
        if(Jm[i]<=0) Jm[i]=1e-3;
        cdmp[i]=2.0*zeta*std::sqrt(std::max(0.0,k[i])*Jm[i]);
      }
    }
    if(elastic) for(int i=0;i<nu;i++) mp->actuator_ctrllimited[i]=0;   // 전달토크는 드라이브 한계와 별개(충격 시 잘리면 안 됨)
    mp->opt.timestep = m->opt.timestep/sub;
    th.assign(nu,0); om.assign(nu,0); ff_out.assign(nu,0); drv_out.assign(nu,0);
    lpf_y.assign(nu,0); nz.assign(4*nu,0); finit=false;
    reset(d);
    std::printf("[plant] 드라이버추종 %s (kp×%.2f kd×%.2f) · FF비율 %.2f/%.2f/%.2f/%.2f · FF LPF %.1fHz · 노치 %.1fHz(Q%.1f) · 서브스텝 %d\n",
      track?"ON":"OFF", trk_kp, trk_kd, FF4[0],FF4[1],FF4[2],FF4[3], lpf_hz, notch_hz, notch_q, sub);
    if(elastic){
      std::printf("[plant] ★탄성 ON — 축(hip,thigh,calf,foot): k=");
      for(int g=0;g<4;g++) std::printf("%s%.0f", g?",":"", K4[g]);
      std::printf(" Nm/rad · 백래시 ±");
      for(int g=0;g<4;g++) std::printf("%s%.2f", g?",":"", BL4[g]);
      std::printf("° · ζ %.2f · 감속기 마찰 %s\n[plant]   로터 J_m(구동좌표):", zeta, frot?"로터 쪽":"링크 쪽");
      for(int i=0;i<nu;i++) std::printf(" %.4f", Jm[i]);
      std::printf(" · 공진(링크 고정 근사) Hz:");
      for(int i=0;i<nu;i++) std::printf(" %.1f", std::sqrt((k[i]+kp[i])/Jm[i])/(2*M_PI));
      std::printf("\n");
    }
  }
  ~SimPlant(){ if(mp) mj_deleteModel(mp); }

  void link_act(const mjData* d, std::vector<double>& s, std::vector<double>& sd) const {
    Eigen::Map<const Eigen::VectorXd> q(d->qpos+7,NJ), qd(d->qvel+6,NJ);
    Eigen::VectorXd a=A*q, ad=A*qd; s.assign(a.data(),a.data()+nu); sd.assign(ad.data(),ad.data()+nu);
  }
  void reset(mjData* d){
    std::vector<double> s,sd; link_act(d,s,sd); th=s; om=sd; finit=false;
  }
  // 제어기 센서(관절좌표, 모터측) — 탄성 끔이면 링크와 같다.
  void sensor_joint(const mjData* d, double* qj, double* dqj) const {
    if(!elastic){ for(int j=0;j<NJ;j++){ qj[j]=d->qpos[7+j]; dqj[j]=d->qvel[6+j]; } return; }
    Eigen::Map<const Eigen::VectorXd> a(th.data(),nu), ad(om.data(),nu);
    Eigen::VectorXd q=Ainv*a, qd=Ainv*ad;
    for(int j=0;j<NJ;j++){ qj[j]=q[j]; dqj[j]=qd[j]; }
  }
  // 관절 목표 → 구동좌표 목표
  void joint_to_act(const double* qj, double* a) const {
    Eigen::Map<const Eigen::VectorXd> q(qj,NJ); Eigen::VectorXd v=A*q; for(int i=0;i<nu;i++) a[i]=v[i];
  }
  double ff_filter(int i, double x, double dt){
    if(lpf_hz>0){ const double al=1.0-std::exp(-2*M_PI*lpf_hz*dt); if(!finit) lpf_y[i]=x; lpf_y[i]+=al*(x-lpf_y[i]); x=lpf_y[i]; }
    if(notch_hz>0){ const double w0=2*M_PI*notch_hz*dt, alp=std::sin(w0)/(2*notch_q), cc=std::cos(w0), a0=1+alp;
      const double b0=1/a0, b1=-2*cc/a0, b2=1/a0, a1=-2*cc/a0, a2=(1-alp)/a0; double* z=&nz[4*i];
      if(!finit){ z[0]=z[1]=z[2]=z[3]=x; }
      const double y=b0*x+b1*z[0]+b2*z[1]-a1*z[2]-a2*z[3]; z[1]=z[0]; z[0]=x; z[3]=z[2]; z[2]=y; x=y; }
    return x;
  }
  static double dz(double x, double b){ return x>b? x-b : (x<-b? x+b : 0.0); }

  // 한 제어주기(dt) 진행. tau_ff = 제어기가 낸(지연된) 구동좌표 토크, qdes/dqdes = (지연된) 관절 목표(없으면 nullptr).
  // tscale = 기존 TAU_SCALE(α) — FF 와 PD 를 모두 깎는다(실기 토크 상수 오차 재현).
  void step(mjData* d, const double* tau_ff, const double* qdes_j, const double* dqdes_j, double dt, double tscale=1.0){
    std::vector<double> ff(nu), qa(nu,0), dqa(nu,0);
    for(int i=0;i<nu;i++){ ff[i]=ff_filter(i, ffs[i]*tau_ff[i], dt); }
    finit=true;
    const bool pd = track && qdes_j && dqdes_j;
    if(pd){ joint_to_act(qdes_j, qa.data()); joint_to_act(dqdes_j, dqa.data()); }
    for(int i=0;i<nu;i++) ff_out[i]=ff[i];
    const double h=dt/sub;
    std::vector<double> s, sd;
    for(int ss=0; ss<sub; ss++){
      link_act(d, s, sd);
      if(!elastic){ th=s; om=sd; }
      for(int i=0;i<nu;i++){
        double u = ff[i] + (pd ? kp[i]*(qa[i]-th[i]) + kd[i]*(dqa[i]-om[i]) : 0.0);
        u = tscale*std::max(-lim[i], std::min(lim[i], u));
        drv_out[i]=u;
        if(!elastic){ d->ctrl[i]=u; continue; }
        const double tt = k[i]*dz(th[i]-s[i], bl[i]) + cdmp[i]*(om[i]-sd[i]);
        const double tf = bm[i]*om[i] + fc[i]*std::tanh(om[i]/0.02);     // 감속기 마찰(로터 쪽)
        om[i] += h*(u - tt - tf)/Jm[i];
        th[i] += h*om[i];
        d->ctrl[i]=tt;
      }
      mj_step(mp, d);
    }
  }
};
