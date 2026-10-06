#include "sim_plant.hpp"
#include <deque>
// trot_sim — quad_mpc_wbic mode_trot 핵심경로 C++ closed-loop (헤드리스). 제어=TrotCtrl(trot_view와 공유).
// 대상: standalone 평지 trot (DETECT=0 순수스케줄). 검증: falls=0 + 전진거리·tilt를 Python과 비교.
#include "trot_controller.hpp"
#include "state_estimator.hpp"   // ★sim2real: leg-odometry 상태추정기(EST_TEST서 정확도 검증)
#include "terrain_map.hpp"       // ★TAMOLS P0: 로컬 elevation map + footScore (TMAP_DUMP 헤드리스 검증)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>

int main(int argc,char**argv){
  const char* path=argc>1?argv[1]:"../mjcf/quad_real_16dof_sphere.mjcf";
  int STEPS = (argc>2)?atoi(argv[2]) : (getenv("STEPS")?atoi(getenv("STEPS")):3000);
  QuadControl q; q.load(path); apply_env_gains(q); q.crouch_home(); q.build_qhome_lut(); q.setup_mpc();  // ★q_home LUT 시작시 빌드(RT-safe)
  if(getenv("DISABLE_FLOOR")){   // ★갭 코스: 무한바닥 접촉off → 플랫폼 box(group2)만 접촉=갭이 진짜 구멍(발 빠짐)
    int fid=mj_name2id(q.m,mjOBJ_GEOM,"floor");
    if(fid>=0){ q.m->geom_contype[fid]=0; q.m->geom_conaffinity[fid]=0;
      std::printf("[trot_sim] DISABLE_FLOOR: floor 접촉off → 갭=진짜 구멍\n"); } }
  if(getenv("LUT_CHECK")){   // ★LUT 보간 vs 직접IK 정확성 self-check(격자 어긋난 높이=최악)
    double maxdq=0, maxdc=0, wh=0;
    for(double h=0.20; h<=0.53; h+=0.0017){
      q.update_stand_qhome_ik(h); Eigen::VectorXd qi=q.q_home; Eigen::Vector3d ci=q.com_ref;
      q.update_stand_qhome(h);    Eigen::VectorXd ql=q.q_home; Eigen::Vector3d cl=q.com_ref;
      double dq=(qi-ql).cwiseAbs().maxCoeff(), dc=(ci-cl).cwiseAbs().maxCoeff();
      if(dq>maxdq){maxdq=dq;wh=h;} maxdc=std::max(maxdc,dc); }
    std::printf("[LUT_CHECK] 보간 vs 직접IK: q_home 최대오차 %.2e rad(%.4f°, h=%.3f) · com_ref %.2e m — %s\n",
      maxdq, maxdq*57.2958, wh, maxdc, (maxdq<2e-3&&maxdc<2e-3)?"OK(동일수준)":"확인필요");
    q.update_stand_qhome_ik(q.base_z0); return 0; }
  if(getenv("QHDBG")) for(int i=0;i<4;i++) std::printf("[qhome] %s hip=%.3f thigh=%.3f calf=%.3f foot=%.3f\n",
      q.legs[i], q.q_home[q.legqp[i][0]-7], q.q_home[q.legqp[i][1]-7], q.q_home[q.legqp[i][2]-7], q.leg_dof[i]==4?q.q_home[q.legqp[i][3]-7]:0.0);
  TrotCtrl ctrl(q);
#ifdef HAVE_JUMP_SOLVER
  if(!getenv("NO_JUMP_WARMUP")){ std::printf("[trot_sim] 점프 OCP 예열…\n"); std::fflush(stdout); ctrl.warmup_jump(); std::printf("[trot_sim] 예열 완료(live-solve)\n"); }
#endif
  if(getenv("TROT_V")) ctrl.V=atof(getenv("TROT_V"));
  if(getenv("BODY_H")) ctrl.body_h=atof(getenv("BODY_H"));   // ★서기 높이 테스트(슬라이더 범위 검증)
  if(getenv("STANCE_KD")) q.STANCE_KD=atof(getenv("STANCE_KD"));   // ★stance 발 속도감쇠(slip↓)
  if(getenv("TROT_VY")) ctrl.VY=atof(getenv("TROT_VY"));   // ★좌우이동(strafe) 테스트
  if(getenv("WAIST_STEER")) ctrl.waist_steer=atof(getenv("WAIST_STEER"));
  if(getenv("SPIN_HOLD")) ctrl.SPIN_HOLD=true;   // ★허리 조향 게인
  if(getenv("PERCEPTIVE")) ctrl.perceptive=atoi(getenv("PERCEPTIVE"))!=0;   // ★지형인지(mj_ray 착지높이) on/off
  if(getenv("PCV_CLR")) ctrl.PCV_CLR=atof(getenv("PCV_CLR"));
  if(getenv("GROUND_LIE_Z")) ctrl.GROUND_LIE_Z=atof(getenv("GROUND_LIE_Z"));         // ★눕기 저자세 튜닝
  if(getenv("GROUND_REAR_FOOT")) ctrl.GROUND_REAR_FOOT=atof(getenv("GROUND_REAR_FOOT"));
  if(getenv("GROUND_FRONT_FOOT")) ctrl.GROUND_FRONT_FOOT=atof(getenv("GROUND_FRONT_FOOT"));
  if(getenv("GROUND_FRONT_THIGH")) ctrl.GROUND_FRONT_THIGH=atof(getenv("GROUND_FRONT_THIGH"));
  if(getenv("GROUND_FRONT_CALF")) ctrl.GROUND_FRONT_CALF=atof(getenv("GROUND_FRONT_CALF"));
  if(getenv("MODE")) ctrl.mode=getenv("MODE");                              // ★모드 테스트(sit/stand_up/stand_down/off)
  if(getenv("SIT_Z")) ctrl.SIT_Z=atof(getenv("SIT_Z"));
  if(getenv("SIT_PITCH")) ctrl.SIT_PITCH=atof(getenv("SIT_PITCH"));
  if(getenv("SIT_REAR_FOOT")) ctrl.SIT_REAR_FOOT=atof(getenv("SIT_REAR_FOOT"));
  if(getenv("SIT_REAR_CALF")) ctrl.SIT_REAR_CALF=atof(getenv("SIT_REAR_CALF"));
  if(getenv("SIT_REAR_THIGH")) ctrl.SIT_REAR_THIGH=atof(getenv("SIT_REAR_THIGH"));
  if(getenv("SIT_SLEW")) ctrl.SIT_SLEW=atof(getenv("SIT_SLEW"));
  if(getenv("SIT_CPITCH")) ctrl.SIT_CPITCH=atof(getenv("SIT_CPITCH"));
  if(getenv("SIT_REACH")) ctrl.SIT_REACH=atof(getenv("SIT_REACH"));
  if(getenv("HAUNCH_Z")) ctrl.HAUNCH_Z=atof(getenv("HAUNCH_Z"));            // ★개-앉기(haunch sit) 튜닝: 접힘높이·fold속도·언폴드완료높이
  if(getenv("HAUNCH_FOLD_RATE")) ctrl.HAUNCH_FOLD_RATE=atof(getenv("HAUNCH_FOLD_RATE"));
  if(getenv("HAUNCH_UNFOLD_Z")) ctrl.HAUNCH_UNFOLD_Z=atof(getenv("HAUNCH_UNFOLD_Z"));
  if(getenv("SIT_POSTURE_W")) ctrl.SIT_POSTURE_W=atof(getenv("SIT_POSTURE_W"));  // 개-앉기 자세task 가중(접힘 홀드)
  if(getenv("HAUNCH_PITCH")) ctrl.HAUNCH_PITCH=atof(getenv("HAUNCH_PITCH"));  // 개-앉기 nose-up(q_home 베이킹)
  if(getenv("HAUNCH_THIGH")) q.HAUNCH_THIGH=atof(getenv("HAUNCH_THIGH"));   // 뒷다리 개-앉기 시드(무릎-위 가지)
  if(getenv("HAUNCH_CALF")) q.HAUNCH_CALF=atof(getenv("HAUNCH_CALF"));
  if(getenv("HAUNCH_FOOT")) q.HAUNCH_FOOT=atof(getenv("HAUNCH_FOOT"));
  if(getenv("HAUNCH_HOCK_Z")) q.HAUNCH_HOCK_Z=atof(getenv("HAUNCH_HOCK_Z"));  // hock 지면 목표(발링크 평평)
  if(getenv("FRONT_REACH")) q.FRONT_REACH=atof(getenv("FRONT_REACH"));       // 앞발 전방 배치(↑=앞다리 더 폄)
  if(getenv("HAUNCH_FOOT_LAND")) q.HAUNCH_FOOT_LAND=atof(getenv("HAUNCH_FOOT_LAND"));  // 착지 중 뒷발 각도(닿은 뒤 HAUNCH_FOOT로 굴림)
  if(getenv("SGU_KICK_T")) ctrl.SGU_KICK_T=atof(getenv("SGU_KICK_T"));      // ★앉기→서기 스크립트 기립 튜닝
  if(getenv("SGU_FB_THIGH")) ctrl.SGU_FB_THIGH=atof(getenv("SGU_FB_THIGH"));
  if(getenv("SGU_FB_CALF")) ctrl.SGU_FB_CALF=atof(getenv("SGU_FB_CALF"));
  if(getenv("SGU_SLEW")) ctrl.SGU_SLEW=atof(getenv("SGU_SLEW"));
  if(getenv("SGU_KP")) ctrl.SGU_KP=atof(getenv("SGU_KP"));
  if(getenv("GETUP_TRAJ_KP")) ctrl.GETUP_TRAJ_KP=atof(getenv("GETUP_TRAJ_KP"));   // ★개-앉기 기립 궤적추종 강성(=튕김 힘). ↓=부드럽게
  if(getenv("GETUP_TRAJ_KD")) ctrl.GETUP_TRAJ_KD=atof(getenv("GETUP_TRAJ_KD"));   // ↑=튕김 감쇠
  if(getenv("JUMP_KP")) ctrl.JUMP_KP=atof(getenv("JUMP_KP"));                     // ★점프 추진 강성(=점프 높이). ↓=낮게
  if(getenv("JUMP_CROUCH_Z")) ctrl.JUMP_CROUCH_Z=atof(getenv("JUMP_CROUCH_Z"));   // 웅크림 깊이(깊을수록 스트로크↑)
  if(getenv("JUMP_THRUST_T")) ctrl.JUMP_THRUST_T=atof(getenv("JUMP_THRUST_T"));   // 추진 최대시간(이벤트 없을 때 타임아웃)
  if(getenv("SGU_GATHER_Z")) ctrl.SGU_GATHER_Z=atof(getenv("SGU_GATHER_Z"));
  if(getenv("SGU_DONE_TILT")) ctrl.SGU_DONE_TILT=atof(getenv("SGU_DONE_TILT"));
  if(getenv("SGU_WALKOUT_V")) ctrl.SGU_WALKOUT_V=atof(getenv("SGU_WALKOUT_V"));
  if(getenv("SGU_HANDOFF_Z")) ctrl.SGU_HANDOFF_Z=atof(getenv("SGU_HANDOFF_Z"));
  if(getenv("TROT_WZ")) ctrl.WZ=atof(getenv("TROT_WZ"));   // ★선회 각속도(직접 yaw, 제자리 스핀)
  if(getenv("TROT_STEER")) ctrl.steer=atof(getenv("TROT_STEER"));  // ★자동차식 조향각δ(Ackermann R=축거/tanδ)
  if(getenv("GAIT")) ctrl.set_gait(getenv("GAIT"));        // ★게이트 테스트(trot/walk/gallop)
  if(getenv("TROT_T")) ctrl.gp_T=atof(getenv("TROT_T"));           // ★게이트 주기 override(set_gait 뒤)
  if(getenv("TROT_SWF")) ctrl.gp_SWF=atof(getenv("TROT_SWF"));     // ★swing 비율 override
  if(getenv("TROT_T")||getenv("TROT_SWF")){ ctrl.gp_Tsw=ctrl.gp_T*ctrl.gp_SWF; ctrl.gp_Tst=ctrl.gp_T*(1.0-ctrl.gp_SWF); }  // T_sw/T_st 재계산
  if(getenv("TROT_STEPH")) ctrl.step_h=atof(getenv("TROT_STEPH")); // ★발 높이 override
  if(getenv("RAIBERT_K")) ctrl.raibert_k=atof(getenv("RAIBERT_K"));  // set_gait 프리셋 위에 강제 override
  ctrl.auto_whip = !(getenv("AUTO_WHIP") && !strcmp(getenv("AUTO_WHIP"),"0"));  // 기본ON, AUTO_WHIP=0로 끔
  if(getenv("SWING_W")){ double v=atof(getenv("SWING_W")); ctrl.whip_lo_f=v; ctrl.whip_lo_r=v; }  // whip 목표(고속/수동)
  if(getenv("SWING_W_F")) ctrl.whip_lo_f=atof(getenv("SWING_W_F"));
  if(getenv("SWING_W_R")) ctrl.whip_lo_r=atof(getenv("SWING_W_R"));
  if(getenv("ALIP") && !strcmp(getenv("ALIP"),"0")) ctrl.ALIP=false;
  if(getenv("POS_HOLD") && !strcmp(getenv("POS_HOLD"),"0")) ctrl.POS_HOLD=false;
  mjModel*m=q.m; mjData*d=q.d; double dt=m->opt.timestep;
  // ★2026-10-06 PLANT=1 — 실기 유사 플랜트(sim_plant.hpp): 드라이버 PD 추종(구동지연된 KinWBC 목표, 드라이버 내부라 지연 없음)
  //   · FF 비율/저역/노치 · 드라이브 한계 · (EL_K) 모터-링크 탄성·백래시. 이족 실기에서 확정한 구조를 사족 sim 에 같은 규약으로.
  //   미지정이면 종전 경로(DRV_TRACK·mj_step) 그대로 — verify.sh 비트동등 유지. 탄성은 GEARBOX=1(로터 반사관성) 필요.
  const bool PLANT = getenv("PLANT") && atoi(getenv("PLANT"));
  SimPlant plant; if(PLANT) plant.init(m,d);
  MjRayTerrainMap tmap; bool TMAPDUMP=getenv("TMAP_DUMP")!=nullptr;   // ★TAMOLS P0 헤드리스 검증: 로컬 elevation map+footScore 덤프
  if(getenv("DBG")) std::printf("[dbg] nu=%d leg_dof=[%d %d %d %d] standing_z=%.5f com_ref=[%.5f %.5f %.5f]\n",
      q.nu,q.leg_dof[0],q.leg_dof[1],q.leg_dof[2],q.leg_dof[3],d->qpos[2],q.com_ref[0],q.com_ref[1],q.com_ref[2]);

  int falls=0; double max_tilt=0, penF=0, penR=0, pitchSum=0, tauEff=0, calfTau=0, footWmax=0; int pn=0;
  // ★관절(축)별 τ·ω peak/RMS (정착후 t>1.5s) — 기립자세 비교용. JSTAT=1 이면 종료시 관절별 출력
  bool JSTAT=getenv("JSTAT")!=nullptr; int _NU=q.nu;
  std::vector<double> tpk(_NU,0), tsq(_NU,0), wpk(_NU,0), wsq(_NU,0); long jstat_n=0;
  std::vector<std::string> jname; for(int jj=0;jj<m->njnt;jj++) if(m->jnt_type[jj]!=mjJNT_FREE){
    const char* nm=mj_id2name(m,mjOBJ_JOINT,jj); std::string s=nm?nm:""; auto pp=s.find("_joint"); if(pp!=std::string::npos) s.erase(pp); jname.push_back(s); }
  // ★임시 slip 계측: 발이 접촉(dist<1mm)인 동안 anchor 대비 수평 이동 최대치 = slip. 접촉종료 시 누적.
  double f_ax[4]={0},f_ay[4]={0},slip_sum[4]={0},slip_mx[4]={0}; bool f_con[4]={false}; int slip_n[4]={0};
  bool SLIP=getenv("SLIPLOG")!=nullptr;
  double grf_fz[4]={0},grf_fx[4]={0}; int grf_n=0; bool GRF=getenv("GRFLOG")!=nullptr;   // ★발별 수직GRF(앞/뒤 비율)+부호있는 수평력(앞제동/뒤추진)
  auto t0=std::chrono::high_resolution_clock::now();
  double switchT=getenv("SWITCH_T")?atof(getenv("SWITCH_T")):-1;   // ★모드전환 테스트: t>SWITCH_T면 MODE2로(getup 검증)
  bool switched=false;
  // ★외란 PUSH(임펄스): PUSH_F[N] 을 PUSH_T[s]±PUSH_DUR 동안 base에 측방(기본 y)으로 — W_AM 등 외란복구 벤치
  double pF=getenv("PUSH_F")?atof(getenv("PUSH_F")):0, pT=getenv("PUSH_T")?atof(getenv("PUSH_T")):3.0, pDur=getenv("PUSH_DUR")?atof(getenv("PUSH_DUR")):0.1;
  int pAX=getenv("PUSH_AX")?atoi(getenv("PUSH_AX")):1, pbid=m->jnt_bodyid[0];   // free joint의 base body
  // ★sim2real 추정기. EST_TEST=Phase1(정확도만, 미공급) · EST_CTRL=Phase2(컨트롤러가 추정상태로 계산→토크는 실기 적용)
  StateEstimator est; bool ESTTEST=getenv("EST_TEST")!=nullptr; bool ESTCTRL=getenv("EST_CTRL")!=nullptr;
  std::vector<int> _efg={q.fgid[0],q.fgid[1],q.fgid[2],q.fgid[3]};
  std::vector<double> _efr={q.fr[0],q.fr[1],q.fr[2],q.fr[3]};
  est.reset(Eigen::Vector3d(d->qpos[0],d->qpos[1],d->qpos[2]));
  if(getenv("KF_QV")) est.KF_QV=atof(getenv("KF_QV"));   // ★KF 튜닝 스윕(노이즈 강건성): 속도 프로세스노이즈↑=가속도 신뢰↓·접촉측정 의존↑
  if(getenv("KF_QP")) est.KF_QP=atof(getenv("KF_QP"));
  if(getenv("KF_RV")) est.KF_RV=atof(getenv("KF_RV"));   // 정지속도 측정노이즈
  if(getenv("KF_ACC_LP")) est.KF_ACC_LP=atof(getenv("KF_ACC_LP"));   // 가속도 저역통과(0=off, 1=완전홀드)
  double epx=0,epy=0,epz=0,evx=0,evy=0,evz=0; long ecnt=0;
  mjData* d_est=(ESTCTRL)?mj_makeData(m):nullptr;   // 추정상태 mjData(컨트롤러가 여기서 Jacobian·MPC·WBIC 계산)
  // ★Phase3 센서 노이즈(가우시안, 고정시드=재현). IMU gyro[rad/s]·quat[rad]·엔코더 q[rad]·dq[rad/s]
  std::mt19937 _rng(2024); std::normal_distribution<double> _nd(0.0,1.0);
  double GYRON=getenv("GYRO_N")?atof(getenv("GYRO_N")):0.0, QUATN=getenv("QUAT_N")?atof(getenv("QUAT_N")):0.0;
  double ENCQN=getenv("ENCQ_N")?atof(getenv("ENCQ_N")):0.0, ENCDQN=getenv("ENCDQ_N")?atof(getenv("ENCDQ_N")):0.0;
  double ACCN=getenv("ACC_N")?atof(getenv("ACC_N")):0.0; bool ESTKF=getenv("EST_ANCHOR")==nullptr;   // ★표준 접촉KF(IMU 가속도 융합)=기본. EST_ANCHOR=1이면 stance-anchored(비교/폴백)
  // ★sim2real 지연(latency): 센서→추정/제어 지연(SENSE_LAT_MS) + 제어→구동 지연(ACT_LAT_MS). 실기 버스·연산 지연 모델(게인 재튜닝 최대원인).
  //   고정 링버퍼(무한루프 무증가). L=0이면 인덱스 항상 현재=지연 없음(회귀無).
  double SLAT=getenv("SENSE_LAT_MS")?atof(getenv("SENSE_LAT_MS")):0.0, ALAT=getenv("ACT_LAT_MS")?atof(getenv("ACT_LAT_MS")):0.0;
  int Lsense=(int)std::lround(SLAT*1e-3/dt), Lact=(int)std::lround(ALAT*1e-3/dt);
  int _fsz=2*(m->nq-7)+11;   // 센서 프레임: qn[NJ]·dqn[NJ]·quat[4]·gyro[3]·cts[4]
  std::vector<std::vector<double>> sring(Lsense+1, std::vector<double>(_fsz,0.0));   // 센서 지연 링
  std::vector<std::vector<double>> cring(Lact+1, std::vector<double>(m->nu,0.0));    // 구동 지연 링
  for(int step=0; step<STEPS; step++){
    if(TMAPDUMP && step%50==0){ double cx=getenv("TMAP_CX")?atof(getenv("TMAP_CX")):d->qpos[0], cy=getenv("TMAP_CY")?atof(getenv("TMAP_CY")):d->qpos[1];
      tmap.update(m,d,cx,cy,(uint64_t)(d->time*1e9)); }   // 맵 rate 갱신(로봇 위치 또는 TMAP_CX/CY 지정 중심)
    if(switchT>0 && d->time>switchT && getenv("MODE2") && !switched){ ctrl.mode=getenv("MODE2"); switched=true; }  // ★1회성(내부 walk-out 인계 안 덮게)
    if(pF!=0){ for(int k=0;k<6;k++) d->xfrc_applied[pbid*6+k]=0; if(d->time>=pT && d->time<pT+pDur) d->xfrc_applied[pbid*6+pAX]=pF; }
    if(ESTCTRL){
      // ★Phase2/3/4: 실기 센서(+노이즈+지연)→추정→d_est(base=추정 · 자세·gyro·관절=측정)→컨트롤러 계산→토크(지연) 실기 적용
      int NJ=m->nq-7;
      // ── 현재 센서 측정(노이즈) → 지연 링에 기록 ──
      std::vector<bool> cts0(4,false);
      for(int i=0;i<4;i++) for(int ci=0;ci<d->ncon;ci++){ const auto&c=d->contact[ci]; if((c.geom1==q.fgid[i]||c.geom2==q.fgid[i])&&c.dist<0.002){ cts0[i]=true; break; } }
      { auto& fr=sring[step%(int)sring.size()]; int o=0;
        static std::vector<double> _qs,_dqs; _qs.resize(NJ); _dqs.resize(NJ);
        if(PLANT) plant.sensor_joint(d,_qs.data(),_dqs.data());       // 탄성 시 모터측 엔코더
        else for(int j=0;j<NJ;j++){ _qs[j]=d->qpos[7+j]; _dqs[j]=d->qvel[6+j]; }
        for(int j=0;j<NJ;j++) fr[o++]=_qs[j]+ENCQN*_nd(_rng);
        for(int j=0;j<NJ;j++) fr[o++]=_dqs[j]+ENCDQN*_nd(_rng);
        double dqp[4]={1,0.5*QUATN*_nd(_rng),0.5*QUATN*_nd(_rng),0.5*QUATN*_nd(_rng)}; mju_normalize4(dqp);
        double qq[4]; mju_mulQuat(qq,&d->qpos[3],dqp); for(int a=0;a<4;a++) fr[o++]=qq[a];
        for(int a=0;a<3;a++) fr[o++]=d->qvel[3+a]+GYRON*_nd(_rng);
        for(int i=0;i<4;i++) fr[o++]=cts0[i]?1.0:0.0; }
      // ── 지연된 센서 프레임 언팩(SENSE_LAT_MS 전) ──
      auto& df=sring[std::max(0,step-Lsense)%(int)sring.size()];
      static std::vector<double> qn,dqn; qn.resize(NJ); dqn.resize(NJ);
      double quatn[4],gyron[3]; std::vector<bool> cts(4,false);
      { int o=0; for(int j=0;j<NJ;j++) qn[j]=df[o++]; for(int j=0;j<NJ;j++) dqn[j]=df[o++];
        for(int a=0;a<4;a++) quatn[a]=df[o++]; for(int a=0;a<3;a++) gyron[a]=df[o++];
        for(int i=0;i<4;i++) cts[i]=df[o++]>0.5; }
      if(ESTKF){ double aw[3]={d->qacc[0]+ACCN*_nd(_rng),d->qacc[1]+ACCN*_nd(_rng),d->qacc[2]+ACCN*_nd(_rng)};   // ★표준 접촉KF(IMU 가속도 융합)
        est.estimate_kf(m, qn.data(), dqn.data(), quatn, gyron, aw, _efg, _efr, cts, m->opt.timestep); }
      else { est.estimate(m, qn.data(), dqn.data(), quatn, gyron, _efg, _efr, cts, m->opt.timestep);   // ★stance-anchored 위치
        if(!(ctrl.mode=="move"||ctrl.mode=="jump")) est.v.setZero(); }   // 정적/기립=속도 ZUPT(위치는 앵커 유지)
      if(d->time>1.0){ epx+=std::pow(est.p[0]-d->qpos[0],2); epy+=std::pow(est.p[1]-d->qpos[1],2); epz+=std::pow(est.p[2]-d->qpos[2],2);
        evx+=std::pow(est.v[0]-d->qvel[0],2); evy+=std::pow(est.v[1]-d->qvel[1],2); evz+=std::pow(est.v[2]-d->qvel[2],2); ecnt++; }
      for(int c=0;c<3;c++){ d_est->qpos[c]=est.p[c]; d_est->qvel[c]=est.v[c]; }
      for(int c=0;c<4;c++) d_est->qpos[3+c]=quatn[c];
      for(int c=0;c<3;c++) d_est->qvel[3+c]=gyron[c];
      for(int j=0;j<NJ;j++){ d_est->qpos[7+j]=qn[j]; d_est->qvel[6+j]=dqn[j]; }
      d_est->time=d->time; mj_forward(m,d_est);
      q.d=d_est; ctrl.control(); q.d=d;
      // ── 구동 지연: 계산된 토크를 링에 기록 → 지연된 토크(ACT_LAT_MS 전) 적용 ──
      { auto& cf=cring[step%(int)cring.size()]; for(int i=0;i<m->nu;i++) cf[i]=d_est->ctrl[i]; }
      auto& dc=cring[std::max(0,step-Lact)%(int)cring.size()];
      for(int i=0;i<m->nu;i++) d->ctrl[i]=dc[i];
    } else { ctrl.control(); }
    { // ★2026-09-30 DRV_TRACK=1 — 드라이버 PD 추종: τ += kp(q_des−q) + kd(q̇_des−q̇) (관절공간). 목표는 토크와 같은 구동지연으로 도착.
      //   q_des·q̇_des = QuadControl KinWBC 계획(WBIC_MIT=2). 게인 TRK_KP_Q[Nm/rad]·TRK_KD_Q[Nm·s/rad] (sim 권장 20/1).
      //   quad_ctrl MujocoHal·TrotBridge 와 같은 규약 — verify.sh 비트동등 유지.
      static const bool DTRK=getenv("DRV_TRACK")&&atoi(getenv("DRV_TRACK"));
      static const double TKP=getenv("TRK_KP_Q")?atof(getenv("TRK_KP_Q")):20.0, TKD=getenv("TRK_KD_Q")?atof(getenv("TRK_KD_Q")):1.0;
      static std::deque<std::vector<double>> pring;                 // [유효 | q_des(nu) | q̇_des(nu)]
      std::vector<double> cur(1+2*q.nu,0.0);
      if(q.mit_valid && q.mit_qdes.size()==q.nu){ cur[0]=1; for(int i=0;i<q.nu;i++){ cur[1+i]=q.mit_qdes[i]; cur[1+q.nu+i]=q.mit_dqdes[i]; } }
      pring.push_back(cur); const int Ld=ESTCTRL?Lact:0; while((int)pring.size()>Ld+1) pring.pop_front();
      const std::vector<double>& f=pring.front();
      if(PLANT){   // 플랜트가 드라이버 PD(구동좌표·서브스텝)와 물리 진행을 맡는다
        const int NJp=m->nq-7; std::vector<double> qj(NJp), dqj(NJp), tff(d->ctrl, d->ctrl+m->nu);
        for(int j=0;j<NJp;j++){ qj[j]=d->qpos[7+j]; dqj[j]=0.0; }
        const bool pv=f[0]>0.5;
        if(pv) for(int i=0;i<q.nu;i++) if(m->actuator_trntype[i]==mjTRN_JOINT){
          int ja=m->jnt_qposadr[m->actuator_trnid[2*i]]-7; if(ja>=0&&ja<NJp){ qj[ja]=f[1+i]; dqj[ja]=f[1+q.nu+i]; } }
        plant.step(d, tff.data(), pv?qj.data():nullptr, pv?dqj.data():nullptr, dt);
      } else if(DTRK && f[0]>0.5)
        for(int i=0;i<q.nu;i++) d->ctrl[i]+=TKP*(f[1+i]-d->qpos[7+i])+TKD*(f[1+q.nu+i]-d->qvel[6+i]);
      q.mit_valid=false; }
    if(!PLANT) mj_step(m,d);
    if(ESTTEST && !ESTCTRL){
      std::vector<bool> cts(4,false);
      for(int i=0;i<4;i++) for(int ci=0;ci<d->ncon;ci++){ const auto&c=d->contact[ci]; if((c.geom1==q.fgid[i]||c.geom2==q.fgid[i])&&c.dist<0.002){ cts[i]=true; break; } }
      { double gz=0; int gn=0; for(int i=0;i<4;i++) if(cts[i]){ double fx=d->geom_xpos[q.fgid[i]*3],fy=d->geom_xpos[q.fgid[i]*3+1]; double tz=q.terrain_z(fx,fy); if(tz>-50.0){ gz+=tz; gn++; } } if(gn>0) est.ground_z=gz/gn; }  // ★지형서 발밑 높이=접촉발 지형 평균(평지 가정 z오차 제거)
      if(ESTKF){ double aw[3]={d->qacc[0]+ACCN*_nd(_rng),d->qacc[1]+ACCN*_nd(_rng),d->qacc[2]+ACCN*_nd(_rng)};
        est.estimate_kf(m, &d->qpos[7], &d->qvel[6], &d->qpos[3], &d->qvel[3], aw, _efg, _efr, cts, m->opt.timestep); }
      else { est.estimate(m, &d->qpos[7], &d->qvel[6], &d->qpos[3], &d->qvel[3], _efg, _efr, cts, m->opt.timestep);
        if(!(ctrl.mode=="move"||ctrl.mode=="jump")) est.v.setZero(); }
      if(d->time>1.0){ epx+=std::pow(est.p[0]-d->qpos[0],2); epy+=std::pow(est.p[1]-d->qpos[1],2); epz+=std::pow(est.p[2]-d->qpos[2],2);
        evx+=std::pow(est.v[0]-d->qvel[0],2); evy+=std::pow(est.v[1]-d->qvel[1],2); evz+=std::pow(est.v[2]-d->qvel[2],2); ecnt++; } }
    if(SLIP){ for(int i=0;i<4;i++){ bool con=false;
        for(int ci=0;ci<d->ncon;ci++){ const auto&c=d->contact[ci]; if((c.geom1==q.fgid[i]||c.geom2==q.fgid[i])&&c.dist<0.001){con=true;break;} }
        double fx=d->geom_xpos[q.fgid[i]*3], fy=d->geom_xpos[q.fgid[i]*3+1];
        if(con){ if(!f_con[i]){ f_ax[i]=fx; f_ay[i]=fy; slip_mx[i]=0; } slip_mx[i]=std::max(slip_mx[i],std::hypot(fx-f_ax[i],fy-f_ay[i])); }
        else if(f_con[i]&&d->time>1.5){ slip_sum[i]+=slip_mx[i]; slip_n[i]++; }
        f_con[i]=con; } }
    if(GRF && d->time>1.5){ for(int ci=0;ci<d->ncon;ci++){ const auto&c=d->contact[ci];
        double f6[6]; mj_contactForce(m,d,ci,f6); double R[9]; for(int r=0;r<9;r++) R[r]=c.frame[r];
        double fzw=R[2]*f6[0]+R[5]*f6[1]+R[8]*f6[2];   // 접촉프레임→world z성분(각 축의 z성분·힘 내적)
        double fxw=R[0]*f6[0]+R[3]*f6[1]+R[6]*f6[2];   // world x성분(부호: +전진방향)
        for(int fi=0;fi<4;fi++) if(c.geom1==q.fgid[fi]||c.geom2==q.fgid[fi]){ grf_fz[fi]+=std::abs(fzw); grf_fx[fi]+=fxw; } }
      grf_n++; }
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
      for(int i=0;i<4;i++){ int cj=q.legqv[i][2]-6; if(cj>=0&&cj<q.nu) calfTau+=std::abs(d->ctrl[cj]); }   // calf(whip 관절) 토크
      for(int i=0;i<4;i++) if(q.leg_dof[i]==4) footWmax=std::max(footWmax,std::abs(d->qvel[q.legqv[i][3]]));
      if(JSTAT){ for(int j=0;j<_NU;j++){ double t=d->ctrl[j], w=d->qvel[6+j];   // 관절j: 토크=ctrl[j], 각속도=qvel[6+j](1:1)
        tpk[j]=std::max(tpk[j],std::abs(t)); tsq[j]+=t*t; wpk[j]=std::max(wpk[j],std::abs(w)); wsq[j]+=w*w; } jstat_n++; } }  // ★발목 최대각속도(반사관성 효과 확인)
    if(step%250==0){ double*qq=&d->qpos[3];
      double yaw=std::atan2(2*(qq[0]*qq[3]+qq[1]*qq[2]),1-2*(qq[2]*qq[2]+qq[3]*qq[3]))*180/M_PI;
      std::printf("[hl] s=%d t=%.2f z=%.3f x=%+.3f y=%+.3f yaw=%+.0f° tilt=%.1f falls=%d\n",
                  step,d->time,d->qpos[2],d->qpos[0],d->qpos[1],yaw,td,falls); }
  }
  double wall=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
  if(q.mit_n>0) std::printf("[trot_sim] WBIC_MIT=1 QP 실패 %ld/%ld\n", q.mit_fail, q.mit_n);
  std::printf("\n=== 종료: STEPS=%d(%.1fs) x=%+.3f z=%.3f max_tilt=%.1f° falls=%d | ★침투평균 앞=%.1fmm 뒤=%.1fmm pitch=%.1f° | %.0f steps/s ===\n",
              STEPS,STEPS*dt,d->qpos[0],d->qpos[2],max_tilt,falls,pn?penF/pn*1000:0,pn?penR/pn*1000:0,pn?pitchSum/pn:0,STEPS/wall);
  if(q.couple_calls) std::printf("  [COUPLE] 무릎모터(τ_calf−τ_foot) 피크=%.1fNm(한계126) 사영발동=%ld/%ld\n", q.couple_km_pk, q.couple_hits, q.couple_calls);
  std::printf("    토크effort 평균Σ|τ|=%.1fNm  calf평균Σ|τ|=%.2fNm (whip 관절)  발목최대ω=%.1f rad/s\n", pn?tauEff/pn:0, pn?calfTau/pn:0, footWmax);
  if(JSTAT && jstat_n>0){ std::printf("  [JSTAT] 관절별 τ·ω (정착후 %ld스텝)\n", jstat_n);
    std::printf("    %-9s %8s %8s %8s %8s\n","joint","τpeak","τrms","ωpeak","ωrms");
    for(int j=0;j<_NU;j++) std::printf("    %-9s %8.2f %8.2f %8.2f %8.2f\n",
      (j<(int)jname.size()?jname[j].c_str():""), tpk[j], std::sqrt(tsq[j]/jstat_n), wpk[j], std::sqrt(wsq[j]/jstat_n)); }
  std::printf("    ★실제 뒷다리(HL) thigh=%.3f calf=%.3f foot=%.3f | 무릎z=%.3f hockZ=%.3f toeZ=%.3f\n",
      d->qpos[q.legqp[0][1]], d->qpos[q.legqp[0][2]], d->qpos[q.legqp[0][3]], d->xpos[mj_name2id(m,mjOBJ_BODY,"HL_calf_link")*3+2],
      d->xpos[q.rear_hock_bid[0]*3+2], q.foot_point(0)[2]);
  { double fz=d->xpos[q.hip_bid[2]*3+2], rz=d->xpos[q.hip_bid[0]*3+2];   // 앞힙 z vs 뒤힙 z
    std::printf("    ★상체방향: 앞힙z=%.3f 뒤힙z=%.3f → %s\n", fz, rz, fz>rz?"앞이 위=nose-up(엉덩이 주저앉기 ✓)":"뒤가 위=nose-down(✗ 반대)");
    std::printf("    ★앞다리(FL) thigh=%.3f calf=%.3f 무릎z=%.3f (calf≈0=곧게 폄)\n",
      d->qpos[q.legqp[2][1]], d->qpos[q.legqp[2][2]], d->xpos[mj_name2id(m,mjOBJ_BODY,"FL_calf_link")*3+2]); }
  if(GRF && grf_n){ double fr=(grf_fz[0]+grf_fz[1])/grf_n, ff=(grf_fz[2]+grf_fz[3])/grf_n; double tot=fr+ff;
    std::printf("    ★수직GRF 평균[N]: HL=%.0f HR=%.0f FL=%.0f FR=%.0f | 뒤=%.0f(%.0f%%) 앞=%.0f(%.0f%%) 뒤:앞=%.2f\n",
      grf_fz[0]/grf_n,grf_fz[1]/grf_n,grf_fz[2]/grf_n,grf_fz[3]/grf_n, fr,tot>0?fr/tot*100:0, ff,tot>0?ff/tot*100:0, ff>0?fr/ff:0);
    double fxr=(grf_fx[0]+grf_fx[1])/grf_n, fxf=(grf_fx[2]+grf_fx[3])/grf_n;
    std::printf("    ★수평GRF 평균[N](+전진): 뒤=%+.1f 앞=%+.1f 합=%+.1f (뒤+=추진 / 앞−=제동)\n", fxr, fxf, fxr+fxf); }
  if(SLIP){ std::printf("    ★발 slip(접촉중 수평이동 평균, mm): ");
    for(int i=0;i<4;i++) std::printf("%s=%.1f ", q.legs[i], slip_n[i]?slip_sum[i]/slip_n[i]*1000:0);
    std::printf(" | 뒤평균=%.1f 앞평균=%.1f mm\n",
      ((slip_n[0]?slip_sum[0]/slip_n[0]:0)+(slip_n[1]?slip_sum[1]/slip_n[1]:0))/2*1000,
      ((slip_n[2]?slip_sum[2]/slip_n[2]:0)+(slip_n[3]?slip_sum[3]/slip_n[3]:0))/2*1000); }
  if(getenv("DUMP_QPOS")){ FILE*f=fopen(getenv("DUMP_QPOS"),"w");   // ★정착 qpos 덤프(trajopt x0/xf용)
    for(int i=0;i<m->nq;i++) fprintf(f,"%.8f ",d->qpos[i]); fclose(f);
    std::printf("[dump] qpos → %s (nq=%d)\n", getenv("DUMP_QPOS"), m->nq); }
  if(d_est) mj_deleteData(d_est);
  if(ESTCTRL && (Lsense>0||Lact>0))
    std::printf("[LATENCY] 센서지연=%.1fms(%d스텝) · 구동지연=%.1fms(%d스텝)\n", SLAT,Lsense, ALAT,Lact);
  if((ESTTEST||ESTCTRL) && ecnt>0){
    std::printf("[EST%s] leg-odometry 추정오차(RMS, 정착후 %ld스텝): pos xyz=%.3f/%.3f/%.3f m · vel xyz=%.3f/%.3f/%.3f m/s\n", ESTCTRL?"-CTRL":"",
      ecnt, std::sqrt(epx/ecnt),std::sqrt(epy/ecnt),std::sqrt(epz/ecnt), std::sqrt(evx/ecnt),std::sqrt(evy/ecnt),std::sqrt(evz/ecnt));
    std::printf("    ★pos는 적분 드리프트 누적(실기 동일, 절대위치 안 씀) · vel/자세가 제어핵심. 추정 base 최종=[%.3f,%.3f,%.3f] vs true=[%.3f,%.3f,%.3f]\n",
      est.p[0],est.p[1],est.p[2], d->qpos[0],d->qpos[1],d->qpos[2]); }
  if(TMAPDUMP){ const Submap* sm=tmap.map();
    if(sm){ float mn=1e9f,mx=-1e9f; double sum=0; int nvv=0, tot=sm->nx*sm->ny;
      for(int k=0;k<tot;k++){ if(!sm->valid[k]) continue; float fs=sm->footScore.d[k]; mn=std::min(mn,fs); mx=std::max(mx,fs); sum+=fs; nvv++; }
      float ez0=1e9f,ez1=-1e9f; for(int k=0;k<tot;k++){ if(!sm->valid[k]) continue; float e=sm->elevation.d[k]; ez0=std::min(ez0,e); ez1=std::max(ez1,e); }
      std::printf("\n[TMAP] center=(%.2f,%.2f) res=%.3f %dx%d · margin=%.3f(foot_r%.2f+σ%.2f) · elevation %.3f~%.3f(Δ%.3f)\n",
        sm->ox+sm->nx*sm->res*0.5, sm->oy+sm->ny*sm->res*0.5, sm->res, sm->nx, sm->ny, tmap.margin(), tmap.foot_r, tmap.placement_margin, ez0, ez1, ez1-ez0);
      std::printf("[TMAP] footScore valid=%d/%d min=%.2f max=%.2f mean=%.2f (1=발판적합·0=부적합·엣지/경사서↓)\n", nvv,tot,mn,mx, nvv?sum/nvv:0.0);
      std::printf("[TMAP] footScore 맵 (#>0.8 +>0.5 .>0.2 (공백)≤0.2/무효):\n"); int st=std::max(1,sm->ny/28);
      for(int j=sm->ny-1;j>=0;j-=st){ std::printf("    ");
        for(int i=0;i<sm->nx;i+=st){ if(!sm->valid[(size_t)j*sm->nx+i]){ std::printf(" "); continue; }
          float fs=sm->footScore.at(i,j); std::printf("%c", fs>0.8f?'#':(fs>0.5f?'+':(fs>0.2f?'.':' '))); }
        std::printf("\n"); } }
    else std::printf("[TMAP] map 미생성(TMAP_DUMP인데 update 안됨)\n"); }
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
