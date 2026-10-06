// biped C++ 단독 sim — MJCF 로드 → BipedControl 제어루프 → mj_step. 헤드리스 폐루프 검증.
// 실행: ./biped_sim [mjcf] [vx] [T]   (기본 ../biped_from_quad.mjcf 0.15 15)
// ★EST_CTRL=1 : 추정 상태(leg-odom+접촉높이)로 폐루프 제어(물리는 GT). 배포 경로 검증. falls 카운트.
#include <mujoco/mujoco.h>
#include "biped_control.hpp"
#include "deploy_loop.hpp"
#include "sim_plant.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>

static double tilt_deg(const double* q){
  double roll=std::atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]*q[1]+q[2]*q[2]));
  double pitch=std::asin(std::max(-1.0,std::min(1.0,2*(q[0]*q[2]-q[3]*q[1]))));
  return std::hypot(roll,pitch)*180/M_PI;
}

int main(int argc,char**argv){
  const char* mjcf = argc>1?argv[1]:"../biped_from_quad.mjcf";
  double vx = argc>2?atof(argv[2]):0.15;
  double T  = argc>3?atof(argv[3]):15.0;
  char err[1000]={0};
  mjModel* m=mj_loadXML(mjcf,nullptr,err,1000);
  if(!m){ printf("모델 로드 실패: %s\n",err); return 1; }
  // ★2026-08-20 **실기 탑재물 반영**. 저울 실측으로 모델 13.898 kg 은 맞다고 확인됐으나,
  //   MCU·PCB 를 얹은 실기는 약 15 kg 이다(+1.1 kg = **8%**). 중력보상이 8% 모자라면
  //   그것만으로 주저앉는다 ⇒ 실기와 토크를 대조할 땐 **이 차이를 먼저 없애고** 봐야
  //   한다. 안 그러면 전 축에 8% 부족이 깔려 어느 축이 진짜 문제인지 안 보인다.
  //   ⚠질량만 더한다(관성·CoM 은 그대로). 정적 stand 토크 비교가 목적이라 충분하다 —
  //     동적 거동까지 볼 거면 실측 CoM 으로 inertial 을 고쳐야 한다.
  if(const char* e=getenv("TORSO_ADD_KG")){
    int tb=mj_name2id(m,mjOBJ_BODY,"torso");
    if(tb>=0){
      m->body_mass[tb]+=atof(e);
      double tot=0; for(int i=0;i<m->nbody;i++) tot+=m->body_mass[i];
      std::printf("[sim] torso +%.3f kg → 총질량 %.4f kg (%.1f N)\n", atof(e), tot, tot*9.81);
    } else std::printf("[sim] ⚠torso 바디를 못 찾음 — TORSO_ADD_KG 무시\n");
  }
  mjData* d=mj_makeData(m);
  BipedControl c(m,d); c.reset();

  // ★TAU_DBG=<초> — 마지막 N초의 **축별 관절토크**와 **좌우 지면반력**을 낸다.
  //   실기 상태 JSON 의 `tau_leg_nm`(측정)·`tau_cmd_nm`(명령)과 **같은 좌표**다
  //   (deploy 는 ch_to_tau_joint 로 관절토크를 발행한다) ⇒ 변환 없이 그대로 대조된다.
  //   ★`qfrc_actuator` 를 쓰는 이유: 발목이 tendon 에 물려 있어 `ctrl[foot]` 은 모터축
  //     지령이고 calf·foot **두 DOF 에 같이** 걸린다. 축별 실제 관절토크는 일반화력이다.
  //   ★★표준편차는 **발행창(0.05s)** 기준으로도 낸다. 실기는 20Hz 로 발행하면서 각 창의
  //     std 를 500Hz 로 계산해 실어 보낸다(`tau_std_nm`) — 창 길이가 다르면 비교가
  //     성립하지 않는다. 전체창 std 는 자세 변화까지 섞여 훨씬 커진다.
  const double TAU_SCALE = getenv("TAU_SCALE") ? atof(getenv("TAU_SCALE")) : 1.0;
  if(TAU_SCALE != 1.0) std::printf("[sim] ⚠TAU_SCALE=%.3f — 명령 토크의 %.0f%% 만 플랜트에 전달(α 재현)\n",
                                  TAU_SCALE, TAU_SCALE*100);
  const double tau_win = getenv("TAU_DBG") ? atof(getenv("TAU_DBG")) : 0.0;
  const double PUB_DT = 0.05;                 // biped_deploy 의 상태 발행 주기와 같게
  std::vector<double> tsum(m->nu,0.0), tsq(m->nu,0.0);
  std::vector<double> wsum(m->nu,0.0), wsq(m->nu,0.0);      // 발행창 누적
  std::vector<double> sdsum(m->nu,0.0), sdmax(m->nu,0.0);   // 창 std 의 평균·최대
  std::vector<double> tmin(m->nu, 1e300), tmax(m->nu,-1e300);
  long wn=0, nwin=0; double wt0=-1;
  double grf[2]={0,0}; long nacc=0;

  // ★QPOS_LOG=<파일> — **재생용 궤적 덤프**(t + qpos 전체).
  //   이 저장소의 뷰어(`biped_view`)는 GLFW 인데, libglfw3-dev 가 없는 기기에선 아예
  //   빌드되지 않는다. WSLg 에서는 깔아도 인터랙티브 GLFW 가 검은 화면이 되는 이력이 있다.
  //   ⇒ sim 은 헤드리스로 **궤적만** 남기고, 그리기는 EGL 오프스크린이 맡는다
  //     (`tools/render_traj.py`). 이러면 화면이 없는 서버·CI 에서도 그림이 나온다.
  //   QPOS_HZ 로 표본율(기본 50Hz) — 500Hz 를 그대로 쓰면 파일만 커지고 눈으론 같다.
  std::FILE* qlog=nullptr; int qdec=1;
  if(const char* p=getenv("QPOS_LOG")){
    qlog = std::fopen(p,"w");
    if(!qlog) std::printf("[sim] ⚠QPOS_LOG 열기 실패: %s\n", p);
    else {
      const double ts = m->opt.timestep;      // dt 는 아래에서 선언된다 — 여기선 모델에서 직접
      double hz = getenv("QPOS_HZ") ? atof(getenv("QPOS_HZ")) : 50.0;
      qdec = std::max(1, (int)std::lround(1.0/(hz*ts)));
      std::fprintf(qlog, "# nq=%d dt=%.6f dec=%d hz=%.1f\n", m->nq, ts, qdec, 1.0/(qdec*ts));
    }
  }
  if(getenv("CONTACT")) c.set_contact_mode(atoi(getenv("CONTACT")));   // ★0=1점 점발보행·1=2점 평발정적
  if(getenv("STAND_CZ")) c.com_ref_z=atof(getenv("STAND_CZ"));         // 정적 높이 테스트
  c.vx_cmd=vx;
  c.vy_cmd = getenv("VY")?atof(getenv("VY")):0.0;        // 측방/선회 테스트용 env
  c.wz_cmd = getenv("WZ")?atof(getenv("WZ")):0.0;
  double dt=m->opt.timestep; int steps=(int)(T/dt); double fell=-1;

  bool est_ctrl = getenv("EST_CTRL")!=nullptr;
  // ★2026-10-06 실기 유사 플랜트(sim_plant.hpp) — 드라이버 추종·FF 처리·탄성. env 미지정이면 종전과 같다.
  SimPlant plant; plant.init(m,d);
  if(plant.elastic && !est_ctrl){ std::printf("[plant] 탄성은 EST_CTRL 경로에서만 센서를 모터측으로 바꾼다 → EST_CTRL 켬\n"); est_ctrl=true; }
  DeployLoop dl; int falls=0;
  if(est_ctrl){ dl.init(m,c); dl.plant=&plant; dl.reset(m,d); }
  // 밀기(제어기 모름): PUSH_FX/FY [N] · PUSH_T0 [s] · PUSH_DUR [s]
  const double PFX=getenv("PUSH_FX")?atof(getenv("PUSH_FX")):0.0, PFY=getenv("PUSH_FY")?atof(getenv("PUSH_FY")):0.0;
  const double PT0=getenv("PUSH_T0")?atof(getenv("PUSH_T0")):6.0, PDU=getenv("PUSH_DUR")?atof(getenv("PUSH_DUR")):0.2;
  // SIMLOG=<csv>: 500Hz — t,z,roll,pitch,gx,gy,gz(몸통 각속도·몸통좌표),st, 액추에이터별 qm(모터측 관절각°)·ql(링크 관절각°)·
  //   dqm(모터측 관절속도°/s)·dql(링크)·u(드라이브 토크)·ff(FF) — 실기 트레이스(q=모터측, aux=출력축)와 같은 의미.
  std::FILE* slog=nullptr;
  if(const char* p=getenv("SIMLOG")){ slog=std::fopen(p,"w");
    if(slog){ std::fprintf(slog,"t,z,roll,pitch,gx,gy,gz,st");
      for(const char* nm : {"qm","ql","dqm","dql","u","ff"}) for(int j=0;j<m->nu;j++) std::fprintf(slog,",%s%d",nm,j);
      std::fprintf(slog,",fall\n"); } }

  bool do_switch=getenv("SWITCH")!=nullptr;    // ★중간 접촉모드 전환 검증(T/2에 토글)
  for(int i=0;i<steps;i++){
    if(do_switch && i==steps/2){ int nm=c.cmode==1?0:1;
      if(getenv("TRANS")){ c.transition_to(nm); std::printf("  [T/2] 굴림 전환 시작 → 목표 cmode=%d\n",nm); }
      else { c.set_contact_mode(nm); c.vx_cmd=(c.cmode==0?vx:0.0); if(est_ctrl) dl.reset(m,d);
             std::printf("  [T/2] 스냅 전환 → cmode=%d\n",c.cmode); } }
    if(est_ctrl) dl.step(m,d,c,dt);      // 추정+지연+보상 → d->ctrl (물리 d 불변)
    else c.control(dt);
    if(getenv("WALK_DBG") && i%25==0){ double* q=&d->qpos[3];
      double pitch=std::asin(std::max(-1.0,std::min(1.0,2*(q[0]*q[2]-q[3]*q[1]))));
      double roll=std::atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]*q[1]+q[2]*q[2]));
      std::printf("  t%.2f com=(%+.3f,%+.3f,%.3f) v=(%+.2f,%+.2f) pitch%+.1f roll%+.1f sw=%d\n",
        i*dt,d->subtree_com[0],d->subtree_com[1],d->subtree_com[2],d->qvel[0],d->qvel[1],pitch*57.3,roll*57.3,c.swing); }
    // ★★TAU_SCALE — **토크 스케일 α 를 흉내낸다** (2026-08-21).
    //   실기의 α(kt 오차·전류루프·감속비 오설정)는 "명령한 τ 의 α 배만 실제로 나온다" 는
    //   현상이다. 컨트롤러는 100% 를 냈다고 믿는데 플랜트는 α 배만 받는다.
    //   ⚠TORSO_ADD_KG 로는 이걸 재현할 수 없다 — 그건 **컨트롤러가 쓰는 모델도 같이**
    //     무거워져서 불일치가 안 생긴다(biped_sim 은 제어와 물리가 같은 m/d 를 쓴다).
    //     그래서 ctrl 을 직접 깎는다. 이게 모델↔플랜트 불일치를 만드는 유일한 지점이다.
    { const int tb=m->jnt_bodyid[0]; const bool on=(i*dt>=PT0 && i*dt<PT0+PDU);
      d->xfrc_applied[tb*6+0]=on?PFX:0.0; d->xfrc_applied[tb*6+1]=on?PFY:0.0; }
    { std::vector<double> tff(d->ctrl, d->ctrl+m->nu);
      const bool pv = est_ctrl ? dl.pd_valid : c.mit_valid;
      std::vector<double> qd(8,0.0), dqd(8,0.0);
      if(est_ctrl && pv){ qd=dl.pd_q; dqd=dl.pd_dq; }
      else if(pv){ for(int j=0;j<8;j++){ qd[j]=c.mit_qdes[j]; dqd[j]=c.mit_dqdes[j]; } }
      plant.step(d, tff.data(), pv?qd.data():nullptr, pv?dqd.data():nullptr, dt, TAU_SCALE); }   // TAU_SCALE·mj_step 포함
    if(slog){ double* q=&d->qpos[3];
      double pitch=std::asin(std::max(-1.0,std::min(1.0,2*(q[0]*q[2]-q[3]*q[1]))));
      double roll=std::atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]*q[1]+q[2]*q[2]));
      std::fprintf(slog,"%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%d",i*dt,d->qpos[2],roll*57.29578,pitch*57.29578,
                   d->qvel[3]*57.29578,d->qvel[4]*57.29578,d->qvel[5]*57.29578,c.stance);
      std::vector<double> qm(8),dqm(8); plant.sensor_joint(d,qm.data(),dqm.data());
      for(int j=0;j<m->nu;j++) std::fprintf(slog,",%.3f",qm[j]*57.29578);
      for(int j=0;j<m->nu;j++) std::fprintf(slog,",%.3f",d->qpos[7+j]*57.29578);
      for(int j=0;j<m->nu;j++) std::fprintf(slog,",%.2f",dqm[j]*57.29578);
      for(int j=0;j<m->nu;j++) std::fprintf(slog,",%.2f",d->qvel[6+j]*57.29578);
      for(int j=0;j<m->nu;j++) std::fprintf(slog,",%.3f",plant.drv_out[j]);
      for(int j=0;j<m->nu;j++) std::fprintf(slog,",%.3f",plant.ff_out[j]);
      std::fprintf(slog,",%d\n",falls); }
    if(qlog && i%qdec==0){                  // 재생용 궤적(데시메이션)
      std::fprintf(qlog, "%.4f", i*dt);
      for(int j=0;j<m->nq;j++) std::fprintf(qlog, " %.6f", d->qpos[j]);
      std::fputc('\n', qlog);
    }
    if(tau_win>0 && (T - i*dt) <= tau_win){                  // 정상상태 창만 집계
      const double tn = i*dt;
      if(wt0<0) wt0=tn;
      if(tn-wt0 >= PUB_DT && wn>0){                          // 발행창 마감 → 창 std 누적
        for(int j=0;j<m->nu;j++){
          double mu=wsum[j]/wn, sd=std::sqrt(std::max(0.0, wsq[j]/wn - mu*mu));
          sdsum[j]+=sd; if(sd>sdmax[j]) sdmax[j]=sd;
          wsum[j]=0; wsq[j]=0;
        }
        nwin++; wn=0; wt0=tn;
      }
      for(int j=0;j<m->nu;j++){
        double t=d->qfrc_actuator[6+j];
        tsum[j]+=t; tsq[j]+=t*t; wsum[j]+=t; wsq[j]+=t*t;
        if(t<tmin[j]) tmin[j]=t;
        if(t>tmax[j]) tmax[j]=t;
      }
      wn++;
      for(int ci=0;ci<d->ncon;ci++){
        mjtNum f[6]; mj_contactForce(m,d,ci,f);              // f[0]=접촉 법선력
        int g1=d->contact[ci].geom1, g2=d->contact[ci].geom2;
        for(int l=0;l<2;l++)
          if(g1==c.sph[l]||g2==c.sph[l]||(c.has_heel&&(g1==c.sph2[l]||g2==c.sph2[l])))
            grf[l]+=f[0];
      }
      nacc++;
    }
    if(est_ctrl){                                           // 낙상 자동리셋 + 카운트(장시간 통계)
      if(d->qpos[2]<0.2 || tilt_deg(&d->qpos[3])>45){
        c.reset(); c.vx_cmd=vx; dl.reset(m,d); plant.reset(d); falls++;
      }
    } else if(d->qpos[2]<0.15 || tilt_deg(&d->qpos[3])>45){ fell=i*dt; break; }
    // ★tilt 판정 추가(2026-08-05). 기존엔 base 높이만 봐서 **기울어진 채 버티는 것을
    //   성공으로 셌다** — T_STEP=0.24/vx=0.30 이 tilt 81.9° 인데 "무낙상" 으로 집계됐다
    //   (0.28@0.25=50.4°, 0.28@0.30=46.2° 도 동일). 위 EST_CTRL 분기는 이미
    //   `qpos[2]<0.2 || tilt>45` 로 옳게 판정하고 있었으므로 임계를 그쪽에 맞췄다.
    //   ⚠ 45° 도 관대하다. 보행 품질 분석은 tilt 10° 이하만 진짜 성공으로 볼 것.
  }

  if(est_ctrl){
    printf("EST_CTRL vx=%.2f T=%.1fs · falls=%d · 추정 base=(%.2f,%.2f,%.3f) GT=(%.2f,%.2f,%.3f) tilt=%.1f°\n",
           vx, T, falls, dl.est.p[0],dl.est.p[1],dl.est.p[2], d->qpos[0],d->qpos[1],d->qpos[2], tilt_deg(&d->qpos[3]));
  } else {
    printf("vx=%.2f · 생존 %.2fs%s · base=(%.3f,%.3f,%.3f) tilt=%.1f°\n",
           vx, fell<0?T:fell, fell<0?"(무낙상)":"(낙상)",
           d->qpos[0],d->qpos[1],d->qpos[2], tilt_deg(&d->qpos[3]));
  }
  if(tau_win>0 && nacc>0){
    static const char* JN[8]={"HL_hip","HL_thigh","HL_calf","HL_foot",
                              "HR_hip","HR_thigh","HR_calf","HR_foot"};
    std::printf("\n== 정상상태 축별 관절토크 (마지막 %.0fs · %ld 샘플) ==\n", tau_win, nacc);
    std::printf("   실기 상태 JSON 의 tau_leg_nm(측정)·tau_cmd_nm(명령)과 **같은 좌표**다.\n");
    // ★좌우차는 **크기**로 뺀다. hip 은 좌우 부호가 거울이라(−2.59 vs +2.60) 그대로
    //   빼면 5.2 라는 유령이 찍힌다 — 실제 비대칭은 0.013 이다.
    std::printf("   실기 `tau_std_nm` 은 **발행창(0.05s)** std 다 — 그 열과 대조할 것.\n");
    std::printf("  %-9s %11s %9s %9s %9s %9s %9s\n",
                "축","관절토크","σ창평균","σ창최대","min","max","|HR|−|HL|");
    for(int j=0;j<m->nu && j<8;j++){
      double mu=tsum[j]/nacc;
      double sdw = nwin? sdsum[j]/nwin : 0.0;
      char dif[24]="";
      if(j>=4){ double ml=tsum[j-4]/nacc;
        std::snprintf(dif,sizeof dif,"%+.3f", std::fabs(mu)-std::fabs(ml)); }
      std::printf("  %-9s %+11.3f %9.3f %9.3f %+9.3f %+9.3f %9s\n",
                  JN[j], mu, sdw, sdmax[j], tmin[j], tmax[j], dif);
    }
    std::printf("  (발행창 %ld 개 · 창당 %ld 샘플)\n", nwin, nwin? nacc/nwin : 0L);
    double gl=grf[0]/nacc, gr=grf[1]/nacc, gt=gl+gr;
    double W=0; for(int i=0;i<m->nbody;i++) W+=m->body_mass[i]; W*=9.81;
    if(gt>1e-6)
      std::printf("  지면반력 Fz   HL %6.1f N (%4.1f%%) · HR %6.1f N (%4.1f%%)"
                  "   합 %6.1f N / 체중 %6.1f N\n", gl, 100*gl/gt, gr, 100*gr/gt, gt, W);
    else
      std::printf("  ⚠지면반력 0 — 접촉이 없다(공중이거나 낙상). 토크값도 의미 없다.\n");
  }
  if(slog) std::fclose(slog);
  if(qlog){ std::fclose(qlog); std::printf("[sim] 궤적 기록 완료 → %s\n", getenv("QPOS_LOG")); }
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
