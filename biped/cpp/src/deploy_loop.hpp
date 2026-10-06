#pragma once
// biped 배포 폐루프 — 상태추정 + (선택)센서/구동 지연 + 지연보상(모델 롤아웃). Python biped_deploy 포팅.
//   step(): 센서→(지연)→추정→제어상태(base=추정·관절/자세=센서)→(지연보상 전진예측)→제어→(구동지연)→d.ctrl.
//   물리 상태(d)는 주입 후 복원(제어만 추정상태로, mj_step은 호출자가 GT로). 실HW=지연은 실측, 보상만 사용.
//   env: SENSE_LAT_MS(센서지연)·ACT_LAT_MS(구동지연)·LAT_COMP_MS(보상 예측). 실HW는 SENSE/ACT=0·LAT_COMP=실측지연.
#include <mujoco/mujoco.h>
#include "biped_control.hpp"
#include "state_estimator.hpp"
#include "sim_plant.hpp"
#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <random>

struct DeployLoop {
  BipedEstimator est;
  mjData* dpred=nullptr;
  int sq=-1, sg=-1, SLAT=0, ALAT=0, LCOMP=0, NJ=0, nu=0;
  double NOISE=0;                                // 센서 노이즈 배율(17-DOF GYRON/ENCQN식)
  // ★축별 절대 sigma (실측 주입용). >0 이면 NOISE 배율보다 우선한다.
  //   2026-08-05 실측(HL/HR hip, limp 8초): ENCQ_N=7.6e-5 rad · ENCDQ_N=0.037 rad/s
  double ENCQ_N=0, ENCDQ_N=0, GYRO_N=0;
  // ★재현용 시드. 고정 0 이면 노이즈 실현이 하나뿐이라 "이 시드에서 안 넘어졌다"만 알 수 있다.
  //   노이즈 강건성은 여러 실현에서 봐야 하므로 SEED 로 바꿀 수 있게 한다.
  std::mt19937 rng{0}; std::normal_distribution<double> nd{0.0,1.0};
  std::deque<std::vector<double>> sbuf, abuf;   // 센서 스냅샷 / 구동 토크 링버퍼
  // ★2026-10-06 실기 유사 플랜트(sim_plant.hpp) 연동.
  //   plant 가 있으면 관절 센서를 **모터측**(탄성 앞)에서 읽고, 제어기 KinWBC 목표(q_des·q̇_des)를
  //   토크와 **같은 구동 지연**으로 늦춰 pd_* 로 내보낸다(드라이버 추종 PD 입력). 없으면 종전과 같다.
  const SimPlant* plant=nullptr;
  std::deque<std::vector<double>> pbuf;
  std::vector<double> pd_q, pd_dq; bool pd_valid=false;

  ~DeployLoop(){ if(dpred) mj_deleteData(dpred); }
  void init(mjModel* m, BipedControl& c){
    NJ=m->nq-7; nu=m->nu;
    std::vector<int> fg={c.sph[0],c.sph[1]};
    std::vector<double> fr={m->geom_size[c.sph[0]*3], m->geom_size[c.sph[1]*3]};
    est.init(m,fg,fr);
    sq=mj_name2id(m,mjOBJ_SENSOR,"imu_quat"); sg=mj_name2id(m,mjOBJ_SENSOR,"imu_gyro");
    double dt=m->opt.timestep;
    auto ms=[&](const char*e){ const char*v=getenv(e); return v?atof(v):0.0; };
    SLAT=(int)std::lround(ms("SENSE_LAT_MS")/1000.0/dt);
    ALAT=(int)std::lround(ms("ACT_LAT_MS")/1000.0/dt);
    LCOMP=(int)std::lround(ms("LAT_COMP_MS")/1000.0/dt);
    NOISE=ms("NOISE");
    ENCQ_N=ms("ENCQ_N"); ENCDQ_N=ms("ENCDQ_N"); GYRO_N=ms("GYRO_N");
    if(const char* sd=getenv("SEED")) rng.seed((unsigned)atoi(sd));
    // ★추정기 튜닝 노출. k_anchor 는 BipedEstimator 에 **이미 구현돼 있으나 기본 0(꺼짐)** 이라
    //   지금까지 xy 가 순수 속도적분이었다. quad 는 stance 앵커가 항상 켜져 있다.
    //   점발 보행 발디딤이 K_RETURN*(절대 CoM − com0) 을 쓰므로 xy 추정 품질이 제어에 실제로 전달된다
    //   → 앵커 on/off 를 실험으로 비교할 수 있어야 한다.
    if(getenv("EST_ANCHOR")) est.k_anchor = atof(getenv("EST_ANCHOR"));
    if(getenv("EST_ALPHA"))  est.alpha    = atof(getenv("EST_ALPHA"));
    if(getenv("EST_JAC_CONTACT")) est.jac_at_contact = atoi(getenv("EST_JAC_CONTACT"))!=0;
    if(getenv("EST_DWELL")) est.dwell_steps = atoi(getenv("EST_DWELL"));
    bool kin = getenv("LAT_COMP_KIN") && atoi(getenv("LAT_COMP_KIN"))!=0;
    if(LCOMP>0 && !kin) dpred=mj_makeData(m);   // kin 이면 dpred 없이 외삽 분기로 간다
  }
  void reset(mjModel* m, mjData* d){
    est.reset(Eigen::Vector3d(d->qpos[0],d->qpos[1],d->qpos[2])); sbuf.clear(); abuf.clear(); pbuf.clear(); pd_valid=false;
  }

  // d->ctrl 를 (지연·추정·보상 반영) 세팅. 호출자가 이어서 mj_step. 물리 d는 불변(주입 후 복원).
  void step(mjModel* m, mjData* d, BipedControl& c, double dt){
    // ① 센서 스냅샷(현재)  [q(NJ)|dq(NJ)|quat(4)|gyro(3)]
    std::vector<double> snap(2*NJ+7);
    if(plant) plant->sensor_joint(d, &snap[0], &snap[NJ]);          // 모터측 엔코더(탄성 앞)
    else for(int j=0;j<NJ;j++){ snap[j]=d->qpos[7+j]; snap[NJ+j]=d->qvel[6+j]; }
    double* qs = sq>=0? &d->sensordata[m->sensor_adr[sq]] : &d->qpos[3];
    double* gs = sg>=0? &d->sensordata[m->sensor_adr[sg]] : &d->qvel[3];
    for(int a=0;a<4;a++) snap[2*NJ+a]=qs[a];
    for(int a=0;a<3;a++) snap[2*NJ+4+a]=gs[a];
    // ★센서 노이즈. NOISE 스칼라 하나로는 우리 하드웨어를 표현할 수 없다 —
    //   규약이 sigma_q:sigma_dq = 0.005 rad : 0.05 rad/s (비 10/s) 로 고정인데
    //   2026-08-05 실측은 7.64e-5 rad : 0.0368 rad/s (비 482/s) 로 **48배 다르다**.
    //   위치는 규약보다 훨씬 깨끗하고(NOISE 환산 0.015) 속도는 상대적으로 지저분하다
    //   (0.736). 속도가 위치 차분에서 나오므로 물리적으로 당연한 결과다.
    //   ⇒ quad_ctrl/hal/mujoco_hal.hpp 규약대로 ENCQ_N/ENCDQ_N/GYRO_N 으로 분리한다.
    //     지정하지 않으면 기존 NOISE 배율로 폴백해 하위호환을 유지한다.
    if(NOISE>0 || ENCQ_N>0 || ENCDQ_N>0 || GYRO_N>0){
      const double sq  = ENCQ_N  > 0 ? ENCQ_N  : NOISE*0.005;   // [rad]
      const double sdq = ENCDQ_N > 0 ? ENCDQ_N : NOISE*0.05;    // [rad/s]
      const double sg  = GYRO_N  > 0 ? GYRO_N  : NOISE*0.02;    // [rad/s]
      for(int j=0;j<NJ;j++){ snap[j]+=sq*nd(rng); snap[NJ+j]+=sdq*nd(rng); }
      for(int a=0;a<3;a++) snap[2*NJ+4+a]+=sg*nd(rng);
    }
    // ② 센서 지연(링버퍼)
    sbuf.push_back(snap);
    while((int)sbuf.size()>SLAT+1) sbuf.pop_front();
    std::vector<double>& s=sbuf.front();
    double *sqj=&s[0], *sdq=&s[NJ], *squat=&s[2*NJ], *sgyro=&s[2*NJ+4];
    // ③ 추정(지연 센서로)
    std::vector<bool> cts(2,false);
    for(int ci=0;ci<d->ncon;ci++){ int g1=d->contact[ci].geom1,g2=d->contact[ci].geom2;
      for(int k=0;k<2;k++) if(c.sph[k]==g1||c.sph[k]==g2) cts[k]=true; }
    est.estimate(m, sqj, sdq, squat, sgyro, cts, dt);
    // ④ 제어상태: base=추정 · 자세/관절=지연센서
    std::vector<double> cqp(m->nq), cqv(m->nv);
    cqp[0]=est.p[0]; cqp[1]=est.p[1]; cqp[2]=est.p[2];
    for(int a=0;a<4;a++) cqp[3+a]=squat[a];
    for(int j=0;j<NJ;j++) cqp[7+j]=sqj[j];
    cqv[0]=est.v[0]; cqv[1]=est.v[1]; cqv[2]=est.v[2];
    for(int a=0;a<3;a++) cqv[3+a]=sgyro[a];
    for(int j=0;j<NJ;j++) cqv[6+j]=sdq[j];
    // ⑤ 지연보상: in-flight 토크 유지하며 LCOMP step 모델 롤아웃(Smith predictor)
    if(LCOMP>0 && dpred){
      std::vector<double> th(nu,0.0); if(!abuf.empty()) th=abuf.back();
      for(int i=0;i<m->nq;i++) dpred->qpos[i]=cqp[i];
      for(int i=0;i<m->nv;i++) dpred->qvel[i]=cqv[i];
      for(int l=0;l<LCOMP;l++){ for(int i=0;i<nu;i++) dpred->ctrl[i]=th[i]; mj_step(m,dpred); }
      for(int i=0;i<m->nq;i++) cqp[i]=dpred->qpos[i];
      for(int i=0;i<m->nv;i++) cqv[i]=dpred->qvel[i];
    }
    // ★운동학 외삽(LAT_COMP_KIN=1) — **실기가 실제로 돌릴 경로**라 시뮬에도 둔다.
    //   동역학 롤아웃은 이 Pi 에서 mj_step 604µs × 4 = 2.42ms 라 제어주기 2ms 를 넘는다
    //   (biped_deploy.cpp 주석 참조). 여기서 비교가 안 되면 실기 결과를 예측할 수 없다.
    else if(LCOMP>0){
      const double TT = LCOMP*dt;
      for(int i=0;i<3;i++) cqp[i] += cqv[i]*TT;
      mju_quatIntegrate(cqp.data()+3, cqv.data()+3, TT);
      for(int j=0;j<NJ;j++) cqp[7+j] += cqv[6+j]*TT;
    }
    // ⑥ 주입 → 제어 → 물리 복원
    std::vector<double> gp(m->nq), gv(m->nv);
    for(int i=0;i<m->nq;i++){ gp[i]=d->qpos[i]; d->qpos[i]=cqp[i]; }
    for(int i=0;i<m->nv;i++){ gv[i]=d->qvel[i]; d->qvel[i]=cqv[i]; }
    mj_forward(m,d); c.control(dt);
    // ⑦ 구동 지연(링버퍼) → 지연된 토크를 d->ctrl 에
    std::vector<double> tau(nu); for(int i=0;i<nu;i++) tau[i]=d->ctrl[i];
    abuf.push_back(tau);
    while((int)abuf.size()>ALAT+1) abuf.pop_front();
    std::vector<double> ta=abuf.front();
    { std::vector<double> pv(2*NJ+1,0.0);           // 계획 목표도 같은 구동 지연
      for(int j=0;j<NJ && j<8;j++){ pv[j]=c.mit_qdes[j]; pv[NJ+j]=c.mit_dqdes[j]; } pv[2*NJ]=c.mit_valid?1.0:0.0;
      pbuf.push_back(pv); while((int)pbuf.size()>ALAT+1) pbuf.pop_front();
      const std::vector<double>& pf=pbuf.front();
      pd_q.assign(pf.begin(),pf.begin()+NJ); pd_dq.assign(pf.begin()+NJ,pf.begin()+2*NJ); pd_valid=pf[2*NJ]>0.5; }
    for(int i=0;i<m->nq;i++) d->qpos[i]=gp[i];
    for(int i=0;i<m->nv;i++) d->qvel[i]=gv[i];
    mj_forward(m,d);
    for(int i=0;i<nu;i++) d->ctrl[i]=ta[i];
  }
};
