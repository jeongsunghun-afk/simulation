// trot_view — C++ closed-loop trot을 MuJoCo GLFW 뷰어로 렌더. trot_sim과 동일 제어(TrotCtrl).
//   마우스: 좌드래그=회전 우드래그=이동 휠=줌.  키보드: ↑↓=전진속도 ←→=선회 space=정지 backspace=리셋.
#include "trot_controller.hpp"
#include "state_estimator.hpp"   // ★sim2real: leg-odometry 추정기(뷰어 추정상태 제어 + GT비교 시각화)
#include "sim_plant.hpp"         // ★2026-10-06 PLANT=1 실기 유사 플랜트(trot_sim 과 같은 규약)
#include "terrain_map.hpp"       // ★TAMOLS P0: 로컬 elevation map(mj_ray→격자) + footScore 시각화
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

// 평면 JSON에서 "key": 숫자 추출(GUI cmd 파일용, 경량)
static double json_get(const std::string& s,const char* key,double def){
  std::string k=std::string("\"")+key+"\""; auto p=s.find(k);
  if(p==std::string::npos) return def; p=s.find(':',p);
  if(p==std::string::npos) return def; return atof(s.c_str()+p+1);
}
// ★"key": true/false(JSON 불리언) 추출 — atof는 "true"→0이라 불리언을 못 읽어(체크박스 stuck) 전용 파서 필요
static bool json_bool(const std::string& s,const char* key,bool def){
  std::string k=std::string("\"")+key+"\""; auto p=s.find(k);
  if(p==std::string::npos) return def; p=s.find(':',p);
  if(p==std::string::npos) return def; auto q=s.find_first_not_of(" \t",p+1);
  if(q==std::string::npos) return def;
  if(s.compare(q,4,"true")==0)  return true;
  if(s.compare(q,5,"false")==0) return false;
  return atof(s.c_str()+q)!=0.0;   // 숫자(1/0) 폴백
}
// "key": "문자열" 추출
static std::string json_str(const std::string& s,const char* key,const std::string& def){
  std::string k=std::string("\"")+key+"\""; auto p=s.find(k);
  if(p==std::string::npos) return def; p=s.find(':',p);
  if(p==std::string::npos) return def; auto q1=s.find('"',p+1);
  if(q1==std::string::npos) return def; auto q2=s.find('"',q1+1);
  if(q2==std::string::npos) return def; return s.substr(q1+1,q2-q1-1);
}

// ★상태 발행(STATE_PUB) — Python publish_state와 동일 스키마로 GUI 모니터(IMU/Actuator) 채움.
//   비-free 관절 전체(17-DOF: HL4·HR4·waist1·FL4·FR4 — 허리 포함) q/dq/tau + base_z·rpy·gyro.
static void publish_state(mjModel* m, mjData* d, TrotCtrl& ctrl, const std::string& path){
  int nu=m->nu;
  static std::vector<std::string> jn; static bool init=false;
  if(!init){ init=true;
    for(int j=0;j<m->njnt;j++) if(m->jnt_type[j]!=mjJNT_FREE){
      const char* nm=mj_id2name(m,mjOBJ_JOINT,j); std::string s=nm?nm:"";
      auto p=s.find("_joint"); if(p!=std::string::npos) s.erase(p);
      jn.push_back(s);
    } }
  double R[9]; mju_quat2Mat(R,d->qpos+3);   // world←body 회전
  double roll=atan2(R[7],R[8]), pitch=asin(std::max(-1.0,std::min(1.0,-R[6]))), yaw=atan2(R[3],R[0]);
  double vf=d->qvel[0]*cos(yaw)+d->qvel[1]*sin(yaw);
  static double vf_filt=0; vf_filt=0.97*vf_filt+0.03*vf;
  std::ostringstream o; o.precision(6); o<<std::fixed;
  o<<"{\"mode\":\""<<ctrl.mode<<"\",\"base_z\":"<<d->qpos[2]<<",\"t\":"<<d->time;
  o<<",\"rpy\":["<<roll*57.29578<<","<<pitch*57.29578<<","<<yaw*57.29578<<"]";
  o<<",\"gyro\":["<<d->qvel[3]<<","<<d->qvel[4]<<","<<d->qvel[5]<<"]";
  o<<",\"names\":["; for(int i=0;i<nu;i++){ o<<(i?",":"")<<"\""<<jn[i]<<"\""; }
  o<<"],\"q\":["; for(int i=0;i<nu;i++){ o<<(i?",":"")<<d->qpos[7+i]; }
  o<<"],\"dq\":["; for(int i=0;i<nu;i++){ o<<(i?",":"")<<d->qvel[6+i]; }
  o<<"],\"tau\":["; for(int i=0;i<nu;i++){ o<<(i?",":"")<<d->ctrl[i]; }
  o<<"],\"v_cmd\":"<<ctrl.V<<",\"v_act\":"<<vf_filt<<"}";
  std::string tmp=path+".tmp"; std::ofstream f(tmp); f<<o.str(); f.close();
  std::rename(tmp.c_str(),path.c_str());   // 원자적 교체
}

static mjvCamera cam; static mjvOption opt; static mjvScene scn; static mjrContext con;
static bool btnL=false, btnR=false, btnM=false; static double lastx=0, lasty=0;
static TrotCtrl* gC=nullptr;

// (키보드 제어 삭제 — GUI(teleop_gui_17dof)로 제어. 마우스 카메라만 유지)
static void mouse_btn(GLFWwindow* w,int b,int act,int mods){
  btnL=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS;
  btnR=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS;
  btnM=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS;
  glfwGetCursorPos(w,&lastx,&lasty);
}
static void mouse_move(GLFWwindow* w,double xp,double yp){
  if(!btnL&&!btnR&&!btnM){ lastx=xp; lasty=yp; return; }
  double dx=xp-lastx, dy=yp-lasty; lastx=xp; lasty=yp;
  int W,H; glfwGetWindowSize(w,&W,&H);
  mjtMouse act = btnR? mjMOUSE_MOVE_H : (btnL? mjMOUSE_ROTATE_H : mjMOUSE_ZOOM);
  mjv_moveCamera(gC?gC->q.m:nullptr,act,dx/H,dy/H,&scn,&cam);
}
static void scroll(GLFWwindow* w,double dx,double dy){ mjv_moveCamera(gC?gC->q.m:nullptr,mjMOUSE_ZOOM,0,-0.05*dy,&scn,&cam); }

int main(int argc,char**argv){
  const char* path=argc>1?argv[1]:"../mjcf/quad_real_sphere.mjcf";
  QuadControl q; q.load(path); apply_env_gains(q);
  q.crouch_home(); q.build_qhome_lut(); q.setup_mpc();   // ★q_home LUT 시작시 빌드(RT-safe: 높이변경 IK를 루프서 제거)
  TrotCtrl ctrl(q); gC=&ctrl;
  ctrl.load_jump("/tmp/jump_traj.txt");   // ★점프 궤적 시작 시 preload(fallback용). 없으면 jump_N=0
#ifdef HAVE_JUMP_SOLVER
  if(!getenv("NO_JUMP_WARMUP")){ std::printf("[trot_view] 점프 OCP 예열(모델·IK 캐시)…\n"); std::fflush(stdout); ctrl.warmup_jump(); std::printf("[trot_view] 예열 완료 — 점프 live-solve 활성\n"); }
#endif
  ctrl.mode = getenv("MODE") ? getenv("MODE") : "stand_up";   // ★뷰어는 Ready(서기)로 시작 — GUI 초기값(stand_up)과 일치(시작 보행→멈춤 방지). 헤드리스 trot_sim은 기본 move 유지
  if(getenv("TROT_V")) ctrl.V=atof(getenv("TROT_V"));
  if(getenv("WAIST_STEER")) ctrl.waist_steer=atof(getenv("WAIST_STEER"));   // 허리 lean 게인(기본0.4)
  if(getenv("TROT_STEER")) ctrl.steer=atof(getenv("TROT_STEER"));           // ★자동차식 조향각
  if(getenv("PERCEPTIVE")) ctrl.perceptive=atoi(getenv("PERCEPTIVE"))!=0;    // ★지형인지(mj_ray 착지높이) on/off
  if(getenv("PCV_CLR")) ctrl.PCV_CLR=atof(getenv("PCV_CLR"));
  // ★눕기(ground) 자세 조각 — GUI 슬라이더가 안 될 때 env로 직접 확인/튜닝(GUI cmd와 동일 파라미터)
  if(getenv("GROUND_LIE_Z"))     ctrl.GROUND_LIE_Z=atof(getenv("GROUND_LIE_Z"));
  if(getenv("GROUND_REAR_FOOT")) ctrl.GROUND_REAR_FOOT=atof(getenv("GROUND_REAR_FOOT"));
  if(getenv("GROUND_FRONT_THIGH")) ctrl.GROUND_FRONT_THIGH=atof(getenv("GROUND_FRONT_THIGH"));
  if(getenv("GROUND_FRONT_CALF"))  ctrl.GROUND_FRONT_CALF=atof(getenv("GROUND_FRONT_CALF"));
  if(getenv("GETUP_TRAJ_KP")) ctrl.GETUP_TRAJ_KP=atof(getenv("GETUP_TRAJ_KP"));   // ★개-앉기 기립 궤적추종 강성(=앉기→서기 튕김 힘). ↓=부드럽게
  if(getenv("GETUP_TRAJ_KD")) ctrl.GETUP_TRAJ_KD=atof(getenv("GETUP_TRAJ_KD"));   // ↑=튕김 감쇠
  if(getenv("SGU_KP")) ctrl.SGU_KP=atof(getenv("SGU_KP"));                        // ★크라우치-앉기 기립 강성(뷰어에도 추가)
  if(getenv("JUMP_KP")) ctrl.JUMP_KP=atof(getenv("JUMP_KP"));                     // ★점프 추진 강성(=높이). ↓=낮게
  if(getenv("JUMP_CROUCH_Z")) ctrl.JUMP_CROUCH_Z=atof(getenv("JUMP_CROUCH_Z"));
  if(getenv("JUMP_THRUST_T")) ctrl.JUMP_THRUST_T=atof(getenv("JUMP_THRUST_T"));
  if(getenv("HAUNCH_Z")) ctrl.HAUNCH_Z=atof(getenv("HAUNCH_Z"));            // ★개-앉기(haunch sit) 튜닝
  if(getenv("HAUNCH_FOLD_RATE")) ctrl.HAUNCH_FOLD_RATE=atof(getenv("HAUNCH_FOLD_RATE"));
  if(getenv("HAUNCH_UNFOLD_Z")) ctrl.HAUNCH_UNFOLD_Z=atof(getenv("HAUNCH_UNFOLD_Z"));
  if(getenv("SIT_POSTURE_W")) ctrl.SIT_POSTURE_W=atof(getenv("SIT_POSTURE_W"));
  if(getenv("HAUNCH_THIGH")) q.HAUNCH_THIGH=atof(getenv("HAUNCH_THIGH"));
  if(getenv("HAUNCH_CALF")) q.HAUNCH_CALF=atof(getenv("HAUNCH_CALF"));
  if(getenv("HAUNCH_FOOT")) q.HAUNCH_FOOT=atof(getenv("HAUNCH_FOOT"));
  if(getenv("HAUNCH_HOCK_Z")) q.HAUNCH_HOCK_Z=atof(getenv("HAUNCH_HOCK_Z"));
  if(getenv("SIT_CPITCH")) ctrl.SIT_CPITCH=atof(getenv("SIT_CPITCH"));      // 앉기 nose-up 목표(뷰어 튜닝)
  mjModel*m=q.m; mjData*d=q.d;
  // ★2026-10-06 PLANT=1 — trot_sim 과 같은 실기 유사 플랜트: 드라이버 PD 추종(구동지연된 KinWBC 목표)·FF 비율/저역/노치·
  //   드라이브 한계·(EL_K) 탄성. 드라이버 추종은 WALK_TRACK=1. 미지정이면 종전 경로(mj_step) 그대로.
  const bool PLANT = getenv("PLANT") && atoi(getenv("PLANT"));
  SimPlant plant; if(PLANT) plant.init(m,d);
  std::deque<std::vector<double>> pring;   // [유효 | q_des(nu) | q̇_des(nu)] — 토크와 같은 구동지연

  // ★sim2real GT↔추정 비교 시각화.
  //   기본(GUI '모니터 표시' viz on): 추정기를 **개루프**로 병행(컨트롤러는 참 상태 GT로 제어=보행 안정 유지) + 추정 base 고스트 오버레이.
  //   env EST_CTRL=1: **폐루프** — 컨트롤러가 추정 base로 계산→토크 참 상태 적용(실기 동일 구조, 스트레스 검증용. 추정오차 폐루프 증폭으로 보행 marginal).
  StateEstimator est; mjData* d_est=(getenv("EST_CTRL")?mj_makeData(m):nullptr);
  std::vector<int> _efg={q.fgid[0],q.fgid[1],q.fgid[2],q.fgid[3]};
  std::vector<double> _efr={q.fr[0],q.fr[1],q.fr[2],q.fr[3]};
  est.reset(Eigen::Vector3d(d->qpos[0],d->qpos[1],d->qpos[2]));
  bool est_ctrl=getenv("EST_CTRL")!=nullptr;   // 폐루프(제어에 추정상태 사용) — env 전용
  bool est_on=est_ctrl;                        // 추정기 실행+고스트 표시(viz 체크박스로 켜짐). 폐루프면 당연히 on
  std::mt19937 _rng(2024); std::normal_distribution<double> _nd(0.0,1.0);
  // 센서노이즈(현실 프리셋 = GUI '센서 노이즈' 체크 시). env로도 덮어씀. 0=완벽센서.
  double GYRON=getenv("GYRO_N")?atof(getenv("GYRO_N")):0.0, QUATN=getenv("QUAT_N")?atof(getenv("QUAT_N")):0.0;
  double ENCQN=getenv("ENCQ_N")?atof(getenv("ENCQ_N")):0.0, ENCDQN=getenv("ENCDQ_N")?atof(getenv("ENCDQ_N")):0.0;
  double ACCN=getenv("ACC_N")?atof(getenv("ACC_N")):0.0; bool ESTKF=getenv("EST_ANCHOR")==nullptr;   // ★표준 접촉KF(IMU가속도융합)=기본(폐루프 기립 안정). EST_ANCHOR=1이면 stance-anchored
  double est_perr=0, est_verr=0;   // 최신 추정오차(HUD 표시)
  double est_quat[4]={1,0,0,0};    // 최신 추정(측정) 방위 — 고스트 헤딩 화살표용
  // ★TAMOLS P0: 로컬 elevation map(mj_ray→격자). GUI '지형맵' 토글(tmap_on) 또는 env TERRAIN_MAP=1. 맵 rate로만 갱신(1kHz 밖).
  MjRayTerrainMap tmap; bool tmap_on=getenv("TERRAIN_MAP")!=nullptr;
  // ★sim2real 지연(latency) — env SENSE_LAT_MS(센서→추정 지연=고스트 lag)·ACT_LAT_MS(제어→구동 지연, 폐루프만). 고정 링버퍼(L=0=지연없음).
  double SLAT=getenv("SENSE_LAT_MS")?atof(getenv("SENSE_LAT_MS")):0.0, ALAT=getenv("ACT_LAT_MS")?atof(getenv("ACT_LAT_MS")):0.0;
  int Lsense=(int)std::lround(SLAT*1e-3/m->opt.timestep), Lact=(int)std::lround(ALAT*1e-3/m->opt.timestep);
  int _fsz=2*(m->nq-7)+11;
  std::vector<std::vector<double>> sring(Lsense+1, std::vector<double>(_fsz,0.0));
  std::vector<std::vector<double>> cring(Lact+1, std::vector<double>(m->nu,0.0));
  long estep=0;

  if(!glfwInit()){ std::fprintf(stderr,"glfw init 실패\n"); return 1; }
  GLFWwindow* win=glfwCreateWindow(getenv("VIEW_W")?atoi(getenv("VIEW_W")):1280, getenv("VIEW_H")?atoi(getenv("VIEW_H")):900,"17-DOF C++ trot (quad_mpc_wbic_17dof)",NULL,NULL);
  if(!win){ std::fprintf(stderr,"창 생성 실패(DISPLAY?)\n"); glfwTerminate(); return 1; }
  glfwMakeContextCurrent(win);
  // ★VIEW_SWAP(기본 1=vsync)·VIEW_MAXSTEP(기본 60=프레임당 따라잡기 상한). WSL 노트북은 화면전송 ~6fps 라
  //   60스텝/프레임이면 sim 이 실시간 0.36배 → VIEW_MAXSTEP 를 올려 실시간 유지(2026-10-06)
  glfwSwapInterval(getenv("VIEW_SWAP")?atoi(getenv("VIEW_SWAP")):1);
  const int VIEW_MAXSTEP=getenv("VIEW_MAXSTEP")?atoi(getenv("VIEW_MAXSTEP")):60;
  mjv_defaultCamera(&cam); mjv_defaultOption(&opt); mjv_defaultScene(&scn); mjr_defaultContext(&con);
  mjv_makeScene(m,&scn,8000); mjr_makeContext(m,&con,mjFONTSCALE_150);   // ★8000: terrain map 격자 오버레이 여유
  cam.distance=2.2; cam.elevation=-20; cam.azimuth=135; cam.lookat[2]=0.35;
  opt.flags[mjVIS_CONTACTFORCE]=1;
  // ★VIEW_LITE=1: 그림자·바닥반사 끔 — WSL d3d12 에서 렌더 38.6→12.4ms(오프스크린 실측, 2026-10-06)
  if(getenv("VIEW_LITE")&&atoi(getenv("VIEW_LITE"))){ scn.flags[mjRND_SHADOW]=0; scn.flags[mjRND_REFLECTION]=0; }
  glfwSetMouseButtonCallback(win,mouse_btn);   // 마우스 카메라만(키보드 제어 삭제, GUI로 조작)
  glfwSetCursorPosCallback(win,mouse_move); glfwSetScrollCallback(win,scroll);

  double RATE = getenv("RATE")?atof(getenv("RATE")):1.0;   // 재생 배속(env, 1=실시간·0.5=슬로모)
  const char* CMDFILE = getenv("CMDFILE");                 // ★GUI 연동: /tmp/quad_cmd.json 폴링(v/vy/w)
  std::string STATE_PUB = getenv("STATE_PUB")?getenv("STATE_PUB"):"/tmp/quad_state.json";  // ★상태 발행(GUI 모니터)
  int falls=0; double max_tilt=0; long frame=0; bool fallen=false; long reset_seen=-1, jump_seen=-1;
  auto wall0=std::chrono::steady_clock::now(); double sim0=d->time;
  while(!glfwWindowShouldClose(win)){
    // ★GUI 명령 폴링(~20Hz): teleop_gui가 쓴 v/vy/w 반영
    if(CMDFILE && (frame++ %3==0)){
      std::ifstream f(CMDFILE);
      if(f){ std::stringstream ss; ss<<f.rdbuf(); std::string c=ss.str();
        ctrl.V=json_get(c,"v",ctrl.V); ctrl.VY=json_get(c,"vy",ctrl.VY); ctrl.WZ=json_get(c,"w",ctrl.WZ);
        { static double sh_seen=-999; double sh=json_get(c,"step_h",sh_seen); if(sh!=sh_seen){ ctrl.step_h=sh; sh_seen=sh; } }  // ★step height 슬라이더=엣지 override(게이트별 프리셋 walk0.05/trot0.10/run0.08과 공존)
        { static double rk_seen=-999; double rk=json_get(c,"raibert_k",rk_seen); if(rk!=rk_seen){ ctrl.raibert_k=rk; rk_seen=rk; } }  // ★reach 슬라이더=엣지 override(게이트별 프리셋 walk0.5/trot0.8과 공존)
        double sw=json_get(c,"swing_w",-1); if(sw>=0){ ctrl.whip_lo_r=sw; ctrl.whip_lo_f=sw; }  // 통합(하위호환)
        double swf=json_get(c,"swing_w_f",-1); if(swf>=0) ctrl.whip_lo_f=swf;   // ★앞다리 whip 목표(고속target·auto off시 상수)
        double swr=json_get(c,"swing_w_r",-1); if(swr>=0) ctrl.whip_lo_r=swr;   // ★뒷다리 whip 목표
        ctrl.auto_whip = json_bool(c,"auto_whip",ctrl.auto_whip);              // ★속도연동 whip 토글(on=속도스케일, off=슬라이더 상수)
        ctrl.POS_HOLD = json_bool(c,"pos_hold",ctrl.POS_HOLD);                 // ★정지 위치홀드(드리프트 보정) 토글
        ctrl.foot_nudge = json_bool(c,"foot_nudge",ctrl.foot_nudge);           // ★③ footScore 발판선택(갭·스테핑) 토글(GUI '발 배치' 체크박스). FOOT_NUDGE env는 초기값
        ctrl.steer = json_get(c,"steer",ctrl.steer);            // ★허리 핸들=자동차식 조향각(GUI 슬라이더). Ackermann 반경으로 선회+허리 lean
        ctrl.GROUND_LIE_Z    = json_get(c,"g_lie_z",   ctrl.GROUND_LIE_Z);      // ★눕기 자세 실시간 조각(GUI 슬라이더)
        ctrl.GROUND_REAR_FOOT= json_get(c,"g_rear_foot",ctrl.GROUND_REAR_FOOT);
        ctrl.GROUND_FRONT_THIGH=json_get(c,"g_front_thigh",ctrl.GROUND_FRONT_THIGH);
        ctrl.GROUND_FRONT_CALF =json_get(c,"g_front_calf", ctrl.GROUND_FRONT_CALF);
        { std::string mc=json_str(c,"mode","move");             // move/stand_up(서기)/stand_down(눕기)/off
          if(ctrl.mode!="jump") ctrl.mode=mc; }                 // ★점프 중엔 GUI mode 무시(안 그러면 매 폴링 덮어써 점프 즉시 취소). 컨트롤러가 완료 후 stand_up 자가전환
        ctrl.set_gait(json_str(c,"gait","trot"));               // trot/walk 게이트 토글
        ctrl.body_h = json_get(c,"body_h",ctrl.body_h);         // 서기 높이 슬라이더
        { bool er=json_bool(c,"viz",est_on) || est_ctrl;        // ★'모니터 표시(viz)' 체크박스 = 상태추정기(EST) 고스트 표시. 개루프(보행 GT제어라 안정). 폐루프(env)면 항상 on
          if(er && !est_on) est.reset(Eigen::Vector3d(d->qpos[0],d->qpos[1],d->qpos[2]));  // 켤 때 현재 참 base로 초기화(발 랜드마크도 재앵커=현재위치 생성)
          est_on=er;
          tmap_on = ctrl.foot_nudge || getenv("TERRAIN_MAP")!=nullptr; }   // ★footScore 격자 오버레이 = '발 배치(③)' 체크박스에 연동(모니터표시=EST / 발배치=perceptive 발판, 시각 피드백 일치)
        double rt=json_get(c,"rate",RATE); if(rt>0) RATE=rt;
        long rseq=(long)json_get(c,"reset_seq",reset_seen);     // ★RESET 버튼(상승엣지): mj_resetData+crouch_home+상태초기화
        if(reset_seen<0) reset_seen=rseq;                       //   첫폴링=동기화(시작리셋 방지)
        else if(rseq>reset_seen){ reset_seen=rseq; mj_resetData(m,d); q.crouch_home(); ctrl.reset(); ctrl.mode="stand_up"; falls=0; fallen=false;
          est.reset(Eigen::Vector3d(d->qpos[0],d->qpos[1],d->qpos[2])); est_perr=est_verr=0;
          if(PLANT){ plant.reset(d); pring.clear(); }   // ★플랜트(로터 상태·FF 필터)·목표 지연링도 리셋   // ★추정기도 리셋(참 base로)
          wall0=std::chrono::steady_clock::now(); sim0=d->time; }   // ★reset=Ready 복귀(mode 미초기화 시 점프 중 reset하면 그대로 재점프하던 버그)
        long jseq=(long)json_get(c,"jump_seq",jump_seen);       // ★Jump 버튼(상승엣지): 스크립트 점프 발동(mode=jump)
        if(jump_seen<0) jump_seen=jseq; else if(jseq>jump_seen){ jump_seen=jseq; ctrl.mode="jump"; } } }
    // ★벽시계 기준 실시간 페이싱: sim_time이 wall_time×RATE 따라가도록(모니터 refresh 무관)
    double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();
    double target=sim0+wall*RATE; int guard=0;
    static const bool VPROF=getenv("VIEW_PROF")&&atoi(getenv("VIEW_PROF")); static double pt[4]={0,0,0,0}; static int pn=0, pst=0;
    auto _t0=std::chrono::steady_clock::now();
    while(d->time < target && guard++ < VIEW_MAXSTEP){    // 따라잡기(최대 60스텝/프레임=버스트 상한↓, 점프 히치 후 몰아치기 완화)
      if(est_on){
        // ★실기 센서(+노이즈+지연)→leg-odometry 추정. 개루프(뷰어 비교)=추정만 갱신·제어는 GT / 폐루프(env)=추정상태로 제어
        int NJ=m->nq-7;
        // 현재 센서(노이즈) → 지연 링 기록
        std::vector<bool> cts0(4,false);
        for(int i=0;i<4;i++) for(int ci=0;ci<d->ncon;ci++){ const auto&cc=d->contact[ci]; if((cc.geom1==q.fgid[i]||cc.geom2==q.fgid[i])&&cc.dist<0.002){ cts0[i]=true; break; } }
        { auto& fr=sring[estep%(int)sring.size()]; int o=0;
          static std::vector<double> _qs,_dqs; _qs.resize(NJ); _dqs.resize(NJ);
          if(PLANT) plant.sensor_joint(d,_qs.data(),_dqs.data());       // 탄성 시 모터측 엔코더
          else for(int j=0;j<NJ;j++){ _qs[j]=d->qpos[7+j]; _dqs[j]=d->qvel[6+j]; }
          for(int j=0;j<NJ;j++) fr[o++]=_qs[j]+ENCQN*_nd(_rng);
          for(int j=0;j<NJ;j++) fr[o++]=_dqs[j]+ENCDQN*_nd(_rng);
          double dqp[4]={1,0.5*QUATN*_nd(_rng),0.5*QUATN*_nd(_rng),0.5*QUATN*_nd(_rng)}; mju_normalize4(dqp);
          double qq[4]; mju_mulQuat(qq,&d->qpos[3],dqp); for(int a=0;a<4;a++) fr[o++]=qq[a];
          for(int a=0;a<3;a++) fr[o++]=d->qvel[3+a]+GYRON*_nd(_rng);
          for(int i=0;i<4;i++) fr[o++]=cts0[i]?1.0:0.0; }
        // 지연 센서 언팩(SENSE_LAT_MS 전)
        auto& df=sring[std::max(0L,estep-Lsense)%(long)sring.size()];
        static std::vector<double> qn,dqn; qn.resize(NJ); dqn.resize(NJ);
        double quatn[4],gyron[3]; std::vector<bool> cts(4,false);
        { int o=0; for(int j=0;j<NJ;j++) qn[j]=df[o++]; for(int j=0;j<NJ;j++) dqn[j]=df[o++];
          for(int a=0;a<4;a++) quatn[a]=df[o++]; for(int a=0;a<3;a++) gyron[a]=df[o++];
          for(int i=0;i<4;i++) cts[i]=df[o++]>0.5; }
        { double gz=0; int gn=0; for(int i=0;i<4;i++) if(cts[i]){ double fx=d->geom_xpos[q.fgid[i]*3],fy=d->geom_xpos[q.fgid[i]*3+1]; double tz=q.terrain_z(fx,fy); if(tz>-50.0){ gz+=tz; gn++; } } if(gn>0) est.ground_z=gz/gn; }  // ★지형서 발밑 높이=접촉발 지형평균(z오차↓). xy 드리프트는 leg-odom 근본한계(exteroception 필요)
        // ★상태추정: EST_KF=표준 접촉KF(IMU가속도 융합, 폐루프 기립 안정) / 기본=stance-anchored(위치 앵커+z기구학, 속도 정적ZUPT)
        if(ESTKF){ double aw[3]={d->qacc[0]+ACCN*_nd(_rng),d->qacc[1]+ACCN*_nd(_rng),d->qacc[2]+ACCN*_nd(_rng)};
          est.estimate_kf(m, qn.data(), dqn.data(), quatn, gyron, aw, _efg, _efr, cts, m->opt.timestep); }
        else { est.estimate(m, qn.data(), dqn.data(), quatn, gyron, _efg, _efr, cts, m->opt.timestep);
          if(!(ctrl.mode=="move"||ctrl.mode=="jump")) est.v.setZero(); }   // 정적/기립=속도 ZUPT(위치는 앵커라 유지)
        for(int a=0;a<4;a++) est_quat[a]=quatn[a];   // 추정(측정) 방위 → 고스트 헤딩 화살표
        est_perr=std::sqrt(std::pow(est.p[0]-d->qpos[0],2)+std::pow(est.p[1]-d->qpos[1],2)+std::pow(est.p[2]-d->qpos[2],2));
        est_verr=std::sqrt(std::pow(est.v[0]-d->qvel[0],2)+std::pow(est.v[1]-d->qvel[1],2)+std::pow(est.v[2]-d->qvel[2],2));
        if(est_ctrl && d_est){       // 폐루프(env EST_CTRL): 컨트롤러=추정상태, 토크(구동지연)=참 상태 적용
          for(int a=0;a<3;a++){ d_est->qpos[a]=est.p[a]; d_est->qvel[a]=est.v[a]; }
          for(int a=0;a<4;a++) d_est->qpos[3+a]=quatn[a];
          for(int a=0;a<3;a++) d_est->qvel[3+a]=gyron[a];
          for(int j=0;j<NJ;j++){ d_est->qpos[7+j]=qn[j]; d_est->qvel[6+j]=dqn[j]; }
          d_est->time=d->time; mj_forward(m,d_est);
          q.d=d_est; ctrl.control(); q.d=d;
          { auto& cf=cring[estep%(int)cring.size()]; for(int i=0;i<m->nu;i++) cf[i]=d_est->ctrl[i]; }
          auto& dc=cring[std::max(0L,estep-Lact)%(long)cring.size()];
          for(int i=0;i<m->nu;i++) d->ctrl[i]=dc[i];
        } else { ctrl.control(); }   // 개루프: 제어는 GT(보행 안정), 추정기는 고스트용으로만 병행
        estep++;
      } else { ctrl.control(); }
      if(PLANT){   // trot_sim 과 같은 규약: KinWBC 목표는 토크와 같은 구동지연, 드라이버 PD·물리 진행은 플랜트
        std::vector<double> cur(1+2*q.nu,0.0);
        if(q.mit_valid && q.mit_qdes.size()==q.nu){ cur[0]=1; for(int i=0;i<q.nu;i++){ cur[1+i]=q.mit_qdes[i]; cur[1+q.nu+i]=q.mit_dqdes[i]; } }
        pring.push_back(cur); const int Ld=(est_on&&est_ctrl)?Lact:0; while((int)pring.size()>Ld+1) pring.pop_front();
        const std::vector<double>& f=pring.front();
        const int NJp=m->nq-7; std::vector<double> qj(NJp), dqj(NJp), tff(d->ctrl, d->ctrl+m->nu);
        for(int j=0;j<NJp;j++){ qj[j]=d->qpos[7+j]; dqj[j]=0.0; }
        const bool pv=f[0]>0.5;
        if(pv) for(int i=0;i<q.nu;i++) if(m->actuator_trntype[i]==mjTRN_JOINT){
          int ja=m->jnt_qposadr[m->actuator_trnid[2*i]]-7; if(ja>=0&&ja<NJp){ qj[ja]=f[1+i]; dqj[ja]=f[1+q.nu+i]; } }
        plant.step(d, tff.data(), pv?qj.data():nullptr, pv?dqj.data():nullptr, m->opt.timestep);
        q.mit_valid=false;
      } else mj_step(m,d);
      double td=ctrl.tiltdeg(); max_tilt=std::max(max_tilt,td);
      // ★낙상 시 자동재시작 안 함(그대로 쓰러진 채 유지 → RESET 버튼으로 복구). 낙상은 엣지로만 카운트
      bool low=(td>50||d->qpos[2]<0.2); if(low && !fallen) falls++; fallen=low;
    }
    // ★페이싱 재동기: 뒤처짐(히치 후 백로그)·앞섬 양쪽 모두 0.2s 초과면 시계 리셋 → 백로그 버림(점프 후 fast-forward 렉 방지)
    if(target-d->time > 0.2 || d->time-sim0 > wall*RATE+0.5){ wall0=std::chrono::steady_clock::now(); sim0=d->time; }
    if(!STATE_PUB.empty() && (frame%3==0)) publish_state(m,d,ctrl,STATE_PUB);   // ★GUI 모니터(IMU/Actuator 17-DOF) 발행

    mjrRect vp={0,0,0,0}; glfwGetFramebufferSize(win,&vp.width,&vp.height);
    cam.lookat[0]=d->qpos[0]; cam.lookat[1]=d->qpos[1];   // 로봇 추적
    auto _t1=std::chrono::steady_clock::now(); pst+=guard;
    mjv_updateScene(m,d,&opt,NULL,&cam,mjCAT_ALL,&scn);
    auto _t2=std::chrono::steady_clock::now();
    if(tmap_on){   // ★TAMOLS P0: 로컬 elevation map 갱신(맵 rate, ~20프레임마다=1kHz 밖) + footScore 격자 오버레이
      { static long _tk=0; if(_tk++ % 20 == 0) tmap.update(m,d,d->qpos[0],d->qpos[1],(uint64_t)(d->time*1e9)); }
      const Submap* sm=tmap.map();
      if(sm){ mjtNum sz[3]={(mjtNum)(sm->res*0.45),(mjtNum)(sm->res*0.45),0.006};   // 셀=얇은 평판(초록=발판적합 / 빨강=부적합, z=지형높이)
        for(int j=0;j<sm->ny && scn.ngeom<scn.maxgeom-8;j++) for(int i=0;i<sm->nx && scn.ngeom<scn.maxgeom-8;i++){
          if(!sm->valid[(size_t)j*sm->nx+i]) continue;
          float fs=sm->footScore.at(i,j);
          float rgba[4]={1.0f-fs, 0.30f+0.55f*fs, 0.12f, 0.5f};
          mjtNum p[3]={sm->ox+i*sm->res, sm->oy+j*sm->res, sm->elevation.at(i,j)+0.004};
          mjvGeom* g=&scn.geoms[scn.ngeom]; mjv_initGeom(g,mjGEOM_BOX,sz,p,NULL,rgba); g->category=mjCAT_DECOR; scn.ngeom++;
        } }
    }
    if(est_on && scn.ngeom+4<=scn.maxgeom){   // ★GT↔추정 비교: 로봇 위로 띄운 마커쌍(초록=참 base / 주황=추정 base) — 정지 시 겹치고, 드리프트하면 수평 이격
      const double H=0.35;                     // 몸통 위 띄움 높이(메시에 안 묻히게)
      mjtNum bz=d->qpos[2];
      mjtNum gtM[3]={d->qpos[0],d->qpos[1],bz+H};      // GT 마커(참 base 위)
      mjtNum esM[3]={est.p[0],est.p[1],bz+H};          // 추정 마커(같은 높이, xy는 추정값 → 수평 이격=xy 드리프트)
      mjtNum gtB[3]={d->qpos[0],d->qpos[1],bz};        // 참 base(테더 앵커)
      float grn[4]={0.20f,0.90f,0.35f,0.95f}, org[4]={1.0f,0.55f,0.10f,0.95f}, orgL[4]={1.0f,0.55f,0.10f,0.65f};
      mjtNum szG[3]={0.05,0,0}, szO[3]={0.06,0,0};
      mjvGeom* g0=&scn.geoms[scn.ngeom]; mjv_initGeom(g0,mjGEOM_SPHERE,szG,gtM,NULL,grn); g0->category=mjCAT_DECOR; scn.ngeom++;   // GT(초록)
      mjvGeom* g1=&scn.geoms[scn.ngeom]; mjv_initGeom(g1,mjGEOM_SPHERE,szO,esM,NULL,org); g1->category=mjCAT_DECOR; scn.ngeom++;   // 추정(주황)
      mjvGeom* g2=&scn.geoms[scn.ngeom]; mjv_initGeom(g2,mjGEOM_LINE,NULL,NULL,NULL,org);  mjv_connector(g2,mjGEOM_LINE,5,gtM,esM); g2->category=mjCAT_DECOR; scn.ngeom++;   // 드리프트 이격선
      mjvGeom* g3=&scn.geoms[scn.ngeom]; mjv_initGeom(g3,mjGEOM_LINE,NULL,NULL,NULL,grn);  mjv_connector(g3,mjGEOM_LINE,2,gtB,gtM); g3->category=mjCAT_DECOR; scn.ngeom++;   // 로봇↔마커 테더
      // ★헤딩 화살표(선회 시각화): 각 마커서 yaw 방향으로 뻗음. 초록=GT yaw / 주황=추정(측정 IMU) yaw. 제자리 선회도 회전이 보임
      if(scn.ngeom+2<=scn.maxgeom){
        double Rg[9]; mju_quat2Mat(Rg,&d->qpos[3]);  double yg=atan2(Rg[3],Rg[0]);   // GT yaw
        double Re[9]; mju_quat2Mat(Re,est_quat);     double ye=atan2(Re[3],Re[0]);   // 추정 yaw
        const double L=0.28;
        mjtNum gtA[3]={gtM[0]+L*cos(yg),gtM[1]+L*sin(yg),gtM[2]}, esA[3]={esM[0]+L*cos(ye),esM[1]+L*sin(ye),esM[2]};
        mjvGeom* g4=&scn.geoms[scn.ngeom]; mjv_initGeom(g4,mjGEOM_LINE,NULL,NULL,NULL,grn); mjv_connector(g4,mjGEOM_LINE,4,gtM,gtA); g4->category=mjCAT_DECOR; scn.ngeom++;
        mjvGeom* g5=&scn.geoms[scn.ngeom]; mjv_initGeom(g5,mjGEOM_LINE,NULL,NULL,NULL,org); mjv_connector(g5,mjGEOM_LINE,4,esM,esA); g5->category=mjCAT_DECOR; scn.ngeom++;
      }
    }
    mjr_render(vp,&scn,&con);
    auto _t3=std::chrono::steady_clock::now();
    char hud[384];
    // ★HUD는 MuJoCo 오버레이(ASCII 전용 비트맵 폰트)라 한글 불가 → 영문 표기
    if(est_on)
      std::snprintf(hud,384,"cmd V=%.2f  WZ=%.2f m/s\nactual z=%.3f  tilt=%.1f deg\nx=%+.2f  falls=%d\n%s  [green=GT  orange=EST]\nest err  pos=%.3f m  vel=%.3f m/s  noise=%s\nlatency sense=%.0fms act=%.0fms",
                    ctrl.V,ctrl.WZ,d->qpos[2],ctrl.tiltdeg(),d->qpos[0],falls,
                    est_ctrl?"EST-CTRL closed-loop (stress)":"estimator open-loop (GT ctrl)",
                    est_perr,est_verr,(GYRON>0||QUATN>0||ENCQN>0||ENCDQN>0)?"ON":"off",SLAT,ALAT);
    else
      std::snprintf(hud,320,"cmd V=%.2f  WZ=%.2f m/s\nactual z=%.3f  tilt=%.1f deg\nx=%+.2f  falls=%d\nstate: GT (ground truth) control\nmouse: rotate/zoom  |  control via GUI",
                    ctrl.V,ctrl.WZ,d->qpos[2],ctrl.tiltdeg(),d->qpos[0],falls);
    mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,vp,est_on?"17-DOF C++ trot (GT vs EST compare)":"17-DOF C++ trot (GUI controlled)",hud,&con);
    glfwSwapBuffers(win); glfwPollEvents();
    if(VPROF){ auto _t4=std::chrono::steady_clock::now(); auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
      pt[0]+=ms(_t0,_t1); pt[1]+=ms(_t1,_t2); pt[2]+=ms(_t2,_t3); pt[3]+=ms(_t3,_t4);
      if(++pn==20){ std::fprintf(stderr,"[VIEW_PROF] 프레임당 ms: 스텝 %.1f(%d개) 장면 %.1f 렌더 %.1f 오버레이+스왑(GPU대기 포함) %.1f · ngeom %d · %dx%d\n",
        pt[0]/pn,pst/pn,pt[1]/pn,pt[2]/pn,pt[3]/pn,scn.ngeom,vp.width,vp.height); pt[0]=pt[1]=pt[2]=pt[3]=0; pn=0; pst=0; } }
  }
  mjv_freeScene(&scn); mjr_freeContext(&con); glfwTerminate();
  if(d_est) mj_deleteData(d_est);
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
