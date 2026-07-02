// trot_view — C++ closed-loop trot을 MuJoCo GLFW 뷰어로 렌더. trot_sim과 동일 제어(TrotCtrl).
//   마우스: 좌드래그=회전 우드래그=이동 휠=줌.  키보드: ↑↓=전진속도 ←→=선회 space=정지 backspace=리셋.
#include "trot_controller.hpp"
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

  int falls=0; double max_tilt=0;
  while(!glfwWindowShouldClose(win)){
    double simstart=d->time;
    while(d->time-simstart < 1.0/60.0){    // 실시간(60fps) 만큼 물리 진행
      ctrl.control(); mj_step(m,d);
      double td=ctrl.tiltdeg(); max_tilt=std::max(max_tilt,td);
      if(td>50||d->qpos[2]<0.2){ falls++; ctrl.armed=false; ctrl.settle_until=d->time+TC_SETTLE; q.crouch_home(); }
    }
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
