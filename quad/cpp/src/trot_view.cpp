// trot_view — C++ closed-loop trot을 MuJoCo GLFW 뷰어로 렌더. trot_sim과 동일 제어(TrotCtrl).
//   마우스: 좌드래그=회전 우드래그=이동 휠=줌.  키보드: ↑↓=전진속도 ←→=선회 space=정지 backspace=리셋.
#include "trot_controller.hpp"
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <string>

// 평면 JSON에서 "key": 숫자 추출(GUI cmd 파일용, 경량)
static double json_get(const std::string& s,const char* key,double def){
  std::string k=std::string("\"")+key+"\""; auto p=s.find(k);
  if(p==std::string::npos) return def; p=s.find(':',p);
  if(p==std::string::npos) return def; return atof(s.c_str()+p+1);
}
// "key": "문자열" 추출
static std::string json_str(const std::string& s,const char* key,const std::string& def){
  std::string k=std::string("\"")+key+"\""; auto p=s.find(k);
  if(p==std::string::npos) return def; p=s.find(':',p);
  if(p==std::string::npos) return def; auto q1=s.find('"',p+1);
  if(q1==std::string::npos) return def; auto q2=s.find('"',q1+1);
  if(q2==std::string::npos) return def; return s.substr(q1+1,q2-q1-1);
}

static mjvCamera cam; static mjvOption opt; static mjvScene scn; static mjrContext con;
static bool btnL=false, btnR=false, btnM=false; static double lastx=0, lasty=0;
static TrotCtrl* gC=nullptr;

static void kb(GLFWwindow* w,int key,int sc,int act,int mods){
  if(act!=GLFW_PRESS && act!=GLFW_REPEAT) return;
  if(!gC) return;
  if(key==GLFW_KEY_UP)    gC->V = std::min(2.0, gC->V+0.1);
  if(key==GLFW_KEY_DOWN)  gC->V = std::max(-0.5, gC->V-0.1);
  if(key==GLFW_KEY_LEFT)  gC->WZ = std::min(1.0, gC->WZ+0.1);
  if(key==GLFW_KEY_RIGHT) gC->WZ = std::max(-1.0, gC->WZ-0.1);
  if(key==GLFW_KEY_SPACE){ gC->V=0; gC->VY=0; gC->WZ=0; }
}
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
  const char* path=argc>1?argv[1]:"../quad_real_sphere.mjcf";
  QuadControl q; q.load(path); apply_env_gains(q);
  q.crouch_home(); q.setup_mpc();
  TrotCtrl ctrl(q); gC=&ctrl;
  if(getenv("TROT_V")) ctrl.V=atof(getenv("TROT_V"));
  mjModel*m=q.m; mjData*d=q.d;

  if(!glfwInit()){ std::fprintf(stderr,"glfw init 실패\n"); return 1; }
  GLFWwindow* win=glfwCreateWindow(1280,900,"17-DOF C++ trot (quad_mpc_wbic_17dof)",NULL,NULL);
  if(!win){ std::fprintf(stderr,"창 생성 실패(DISPLAY?)\n"); glfwTerminate(); return 1; }
  glfwMakeContextCurrent(win); glfwSwapInterval(1);
  mjv_defaultCamera(&cam); mjv_defaultOption(&opt); mjv_defaultScene(&scn); mjr_defaultContext(&con);
  mjv_makeScene(m,&scn,2000); mjr_makeContext(m,&con,mjFONTSCALE_150);
  cam.distance=2.2; cam.elevation=-20; cam.azimuth=135; cam.lookat[2]=0.35;
  opt.flags[mjVIS_CONTACTFORCE]=1;
  glfwSetKeyCallback(win,kb); glfwSetMouseButtonCallback(win,mouse_btn);
  glfwSetCursorPosCallback(win,mouse_move); glfwSetScrollCallback(win,scroll);

  double RATE = getenv("RATE")?atof(getenv("RATE")):1.0;   // 재생 배속(env, 1=실시간·0.5=슬로모)
  const char* CMDFILE = getenv("CMDFILE");                 // ★GUI 연동: /tmp/quad_cmd.json 폴링(v/vy/w)
  int falls=0; double max_tilt=0; long frame=0; bool fallen=false; long reset_seen=-1;
  auto wall0=std::chrono::steady_clock::now(); double sim0=d->time;
  while(!glfwWindowShouldClose(win)){
    // ★GUI 명령 폴링(~20Hz): teleop_gui가 쓴 v/vy/w 반영
    if(CMDFILE && (frame++ %3==0)){
      std::ifstream f(CMDFILE);
      if(f){ std::stringstream ss; ss<<f.rdbuf(); std::string c=ss.str();
        ctrl.V=json_get(c,"v",ctrl.V); ctrl.VY=json_get(c,"vy",ctrl.VY); ctrl.WZ=json_get(c,"w",ctrl.WZ);
        ctrl.step_h=json_get(c,"step_h",ctrl.step_h);          // step height 슬라이더
        ctrl.raibert_k=json_get(c,"raibert_k",ctrl.raibert_k); // 전방 reach 슬라이더
        double sw=json_get(c,"swing_w",-1); if(sw>=0) q.swing_w=sw;  // ★whip 슬라이더
        ctrl.mode = json_str(c,"mode","move");                  // move/stand_up(서기)/stand_down(눕기)/off
        ctrl.set_gait(json_str(c,"gait","trot"));               // trot/walk 게이트 토글
        ctrl.body_h = json_get(c,"body_h",ctrl.body_h);         // 서기 높이 슬라이더
        double rt=json_get(c,"rate",RATE); if(rt>0) RATE=rt;
        long rseq=(long)json_get(c,"reset_seq",reset_seen);     // ★RESET 버튼(상승엣지): mj_resetData+crouch_home+상태초기화
        if(reset_seen<0) reset_seen=rseq;                       //   첫폴링=동기화(시작리셋 방지)
        else if(rseq>reset_seen){ reset_seen=rseq; mj_resetData(m,d); q.crouch_home(); ctrl.reset(); falls=0; fallen=false;
          wall0=std::chrono::steady_clock::now(); sim0=d->time; } } }
    // ★벽시계 기준 실시간 페이싱: sim_time이 wall_time×RATE 따라가도록(모니터 refresh 무관)
    double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();
    double target=sim0+wall*RATE; int guard=0;
    while(d->time < target && guard++ < 200){   // 따라잡기(최대 200스텝/프레임=폭주 방지)
      ctrl.control(); mj_step(m,d);
      double td=ctrl.tiltdeg(); max_tilt=std::max(max_tilt,td);
      // ★낙상 시 자동재시작 안 함(그대로 쓰러진 채 유지 → RESET 버튼으로 복구). 낙상은 엣지로만 카운트
      bool low=(td>50||d->qpos[2]<0.2); if(low && !fallen) falls++; fallen=low;
    }
    if(d->time-sim0 > wall*RATE+0.5){ wall0=std::chrono::steady_clock::now(); sim0=d->time; }  // 드리프트 리셋
    mjrRect vp={0,0,0,0}; glfwGetFramebufferSize(win,&vp.width,&vp.height);
    cam.lookat[0]=d->qpos[0]; cam.lookat[1]=d->qpos[1];   // 로봇 추적
    mjv_updateScene(m,d,&opt,NULL,&cam,mjCAT_ALL,&scn);
    mjr_render(vp,&scn,&con);
    char hud[256];
    std::snprintf(hud,256,"cmd V=%.2f  WZ=%.2f m/s\nactual z=%.3f  tilt=%.1f deg\nx=%+.2f  falls=%d\n[UP/DOWN]속도 [L/R]선회 [SPACE]정지",
                  ctrl.V,ctrl.WZ,d->qpos[2],ctrl.tiltdeg(),d->qpos[0],falls);
    mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,vp,"17-DOF C++ 최종세팅",hud,&con);
    glfwSwapBuffers(win); glfwPollEvents();
  }
  mjv_freeScene(&scn); mjr_freeContext(&con); glfwTerminate();
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
