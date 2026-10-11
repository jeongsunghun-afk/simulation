// diag_eps_stepping.cpp — ★진단: stepping 대표 상태에서 **GIAC slack eps 가 목적함수를 삼키는가**.
//
//   의심(코드 직독):
//     · `eps ≥ 0` 는 항상. 상한 `eps ≤ EPS_MAX` 는 `GIAC_FIX && EPS_MAX>0` 일 때만이고 **기본 EPS_MAX=0 = 꺼짐**.
//     · 비용 `W_EPS·eps²`(기본 50) 도 `giac_fix_on()` 게이트 안 → GIAC_FIX 이전엔 **비용도 상한도 없는 공짜 slack**.
//     · 17b 잔차 스케일 `m·det3(pij, pB−pi, aG)` ≈ 15·0.3·0.3·9.81 ≈ 13 → eps 가 자기가 완화하는 양과 동급이면
//       제약을 사실상 100% 위반하고 페널티로 때우는 것이다.
//   단위(중요): det3(pij[m], (pB−pi)[m], aG[m/s²]) = m³/s² → ×m[kg] ⇒ **N·m²**.
//     `m·g·det` 형태라 `eps = m·g·(2·A_signed)` 이고, 지지다각형 한 변 길이 L_edge 에 대해
//     CoM 이 그 변 밖으로 나간 거리 d 는  **d = eps / (m·g·L_edge)**  — eps 를 물리 마진으로 읽는 환산식.
//   ※ 17d 는 `det3(ez, pij, Mi)` 로 **질량인자가 없다**(m³/s²) → 같은 eps 를 단위가 ×m 다른 행이 공유한다.
//     walk(3발 지지)는 17a·17b 만 걸려 이 불일치가 발화하지 않는다. trot(2발)에서는 문제가 된다.
//
//   비용 잔차 행 배치(N=1, P=4, edge/COM off): tracking 24 | foothold_on_ground 4 | nominal 12 | eps 4 = 44
//
//   빌드: PIX=/home/jsh/simple-mpc/.pixi/envs/default; g++ -O3 -std=c++17 diag_eps_stepping.cpp \
//         -I/usr/include/eigen3 -I$PIX/include -L$PIX/lib -Wl,-rpath,$PIX/lib -leiquadprog -o diag_eps_stepping
//   실행: ./diag_eps_stepping [stones_dir=stepping_ref_go2]
//   env : W_EPS · EPS_MAX · GIAC_NORM(정규화 처방 켜기) · TAMOLS_LEGACY=1
#include "tamols_online.hpp"
#include "terrain_proc.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
using namespace tamols;

struct StoneField { int nx=0, ny=0; double size=0, pitch=0; std::vector<double> cxs, cys, top; };
static bool load_field(const std::string& p, StoneField& f) {
  std::ifstream in(p); if (!in) return false;
  std::string l; std::getline(in,l);
  std::vector<std::array<double,7>> rw; int nx=0, ny=0;
  while (std::getline(in,l)) { std::array<double,7> r;
    if (std::sscanf(l.c_str(),"%lf,%lf,%lf,%lf,%lf,%lf,%lf",&r[0],&r[1],&r[2],&r[3],&r[4],&r[5],&r[6])!=7) continue;
    rw.push_back(r); nx=std::max(nx,(int)r[1]+1); ny=std::max(ny,(int)r[2]+1); }
  if (rw.empty()) return false;
  f.nx=nx; f.ny=ny; f.size=rw[0][5];
  f.cxs.assign(nx,0); f.cys.assign(ny,0); f.top.assign((size_t)nx*ny,0);
  for (auto& r:rw){int ix=(int)r[1],iy=(int)r[2]; f.cxs[ix]=r[3]; f.cys[iy]=r[4]; f.top[(size_t)ix*ny+iy]=r[6];}
  f.pitch = nx>1 ? f.cxs[1]-f.cxs[0] : f.size; return true;
}
static double field_h(const StoneField& f,double x,double y,double tol,int* si){
  if(si)*si=-2;
  if(x>=-0.75-tol&&x<=0.75+tol&&std::fabs(y)<=1.0+tol){if(si)*si=-1;return 0.15;}
  double hs=0.5*f.size;
  int ix=(int)std::lround((x-f.cxs[0])/f.pitch), iy=(int)std::lround((y-f.cys[0])/f.pitch);
  if(ix>=0&&ix<f.nx&&iy>=0&&iy<f.ny&&std::fabs(x-f.cxs[ix])<=hs+tol&&std::fabs(y-f.cys[iy])<=hs+tol){
    if(si)*si=ix*f.ny+iy; return f.top[(size_t)ix*f.ny+iy];}
  return 0.0;
}
static int nearest_stone(const StoneField& f,double x,double y){int b=-1;double bd=1e18;
  for(int ix=0;ix<f.nx;++ix)for(int iy=0;iy<f.ny;++iy){double dx=x-f.cxs[ix],dy=y-f.cys[iy],d=dx*dx+dy*dy;
    if(d<bd){bd=d;b=ix*f.ny+iy;}} return b;}
static void raster(const StoneField& f,double bx,double by,int N,double c,Grid& h){
  double off=c*N/2.0; h.resize(N,N);
  for(int a=0;a<N;++a)for(int b=0;b<N;++b) h(a,b)=field_h(f,bx+a*c-off,by+b*c-off,0.0,nullptr);}
static void set_walk_gait_acc(TamolsState& st,double pd){int P=4;st.gait.resize(P);
  int cs[4][4]={{1,1,1,0},{1,0,1,1},{1,1,0,1},{0,1,1,1}}; int ad[4][4]={{0,0,0,1},{0,1,0,1},{0,1,1,1},{1,1,1,1}};
  for(int k=0;k<P;++k){st.gait[k].duration=pd;for(int i=0;i<4;++i){st.gait[k].contact[i]=cs[k][i];st.gait[k].at_des[i]=ad[k][i];}}}
// ★대조용 trot(2발 대각 지지). 같은 eps 가 **지지가 선분으로 퇴화한** 상황에서 어떻게 되는지 보려고 둔다.
//   walk(3발)는 17a·17b 만 걸리지만 trot(N==2)은 17c·17d 가 걸리고, 17d 는 질량인자가 없어
//   같은 eps 를 단위가 ×m 다른 행이 공유한다.
static void diag_set_trot(TamolsState& st,double pd){int P=2;st.gait.resize(P);
  int cs[2][4]={{1,0,0,1},{0,1,1,0}}; int ad[2][4]={{0,1,1,0},{1,1,1,1}};
  for(int k=0;k<P;++k){st.gait[k].duration=pd;for(int i=0;i<4;++i){st.gait[k].contact[i]=cs[k][i];st.gait[k].at_des[i]=ad[k][i];}}}

int main(int argc,char** argv){
  std::string sdir = argc>1?argv[1]:"stepping_ref_go2";
  bool legacy = getenv("TAMOLS_LEGACY") && getenv("TAMOLS_LEGACY")[0]!='0';
  if (legacy){setenv("GIAC_FIX","0",1);setenv("GIAC_ORDER","0",1);}
  else {setenv("GIAC_FIX","1",1);setenv("GIAC_ORDER","1",1);setenv("STANCE_REACH","1",1);}
  const double we = getenv("W_EPS")?atof(getenv("W_EPS")):50.0;
  const int N=101; const double cell=0.02, base_h=0.34, pd=0.2;
  Params prm; prm.hip_offsets<<0.1934,0.142,0,0.1934,-0.142,0,-0.1934,0.142,0,-0.1934,-0.142,0;
  prm.mass=15.0; prm.h_des=base_h; prm.nominal_height=base_h; prm.foot_radius=0.022;
  prm.l_min=0.10; prm.l_max=0.45;

  std::printf("=== GIAC slack(eps) 목적함수 점유 진단 — stepping %s, %s ===\n",
              getenv("DIAG_TROT")?"TROT(2발 지지=선분 퇴화)":"walk(3발 지지)",
              legacy?"LEGACY(GIAC_FIX=0: eps 비용도 없음)":"수정본(GIAC_FIX=1·ORDER=1·STANCE_REACH=1·TERRAIN_HARD)");
  std::printf("W_EPS=%.1f  EPS_MAX=%s  GIAC_NORM=%s\n", we,
              getenv("EPS_MAX")?getenv("EPS_MAX"):"0(꺼짐)", getenv("GIAC_NORM")?getenv("GIAC_NORM"):"0(꺼짐)");
  std::printf("\n%3s %5s %5s | %9s %9s %9s %9s | %9s | %6s | %8s %8s\n",
              "lvl","phx","phy","track","foot","nominal","**eps**","총비용","eps%","eps_max","d_margin");
  std::printf("%s\n", std::string(104,'-').c_str());

  double share_sum=0; int nstate=0;
  for (int lvl : {2,5,7,9}) {
    StoneField f; if (!load_field(sdir+"/stones_L"+std::to_string(lvl)+".csv",f)) {
      std::printf("L%d csv 없음\n",lvl); continue; }
    int ix0=f.nx/2, iy0=0; {double bd=1e18; for(int i=0;i<f.ny;++i){double d=std::fabs(f.cys[i]); if(d<bd){bd=d;iy0=i;}}}
    for (double phx : {0.15, 0.50, 0.85}) for (double phy : {0.25, 0.75}) {
      double bx=f.cxs[ix0]+phx*f.pitch, by=f.cys[iy0]+phy*f.pitch;
      Grid hraw; raster(f,bx,by,N,cell,hraw);
      Grid hsol=gaussian_filter(hraw,2.0);
      SupportLayer SL=compute_support(hraw,cell,0.06,0.25);
      SupportLayers S; S.h=&hraw; S.sdf=&SL.sdf; S.cell=cell; S.map_size=N; S.band=0.01; S.margin=0.025;
      support_layers() = legacy?nullptr:&S;

      double gsum=0; int gn=0;
      for(int ix=0;ix<f.nx;++ix) if(std::fabs(f.cxs[ix]-bx)<0.45)
        for(int iy=0;iy<f.ny;++iy){gsum+=f.top[(size_t)ix*f.ny+iy];++gn;}
      double ground=gn?gsum/gn:0.15, z0=base_h+ground, vadv=0.35;

      Eigen::Matrix<double,4,3> fm;
      for(int l=0;l<4;++l){double hx=bx+prm.hip_offsets(l,0),hy=by+prm.hip_offsets(l,1);
        int ns=nearest_stone(f,hx,hy); double sx=f.cxs[ns/f.ny],sy=f.cys[ns%f.ny];
        if(hx<0.75){sx=hx;sy=hy;} fm(l,0)=sx-bx;fm(l,1)=sy-by;fm(l,2)=field_h(f,sx,sy,0.02,nullptr);}

      TamolsState st; st.prm=prm;
      if (getenv("DIAG_TROT")) diag_set_trot(st,pd); else set_walk_gait_acc(st,pd);
      int P=st.num_phases(); double T=P*pd, xf=vadv*T;
      st.base_pose<<0,0,z0,0,0,0; st.base_vel<<vadv,0,0,0,0,0;
      st.p_meas=fm; st.ref_vel=Vector3d(vadv,0,0);
      st.a.assign(P,MatrixXd::Zero(6,4));
      double c1=vadv,c3=0,c2=(xf-vadv*T)/(T*T);
      for(int k=0;k<P;++k){st.a[k].col(0)=st.base_pose; st.a[k](2,0)=z0;
        double t0=k*pd; st.a[k](0,0)=c1*t0+c2*t0*t0+c3*t0*t0*t0; st.a[k](0,1)=c1+2*c2*t0;}
      for(int i=0;i<4;++i){st.p(i,0)=prm.hip_offsets(i,0)+0.5*xf; st.p(i,1)=prm.hip_offsets(i,1);
        st.p(i,2)=bilinear_height(hsol,cell,N,st.p(i,0),st.p(i,1));}
      st.epsilon=VectorXd::Zero(P);
      QpOptions o; o.max_iter=20; o.zlo=z0-0.06;o.zhi=z0+0.06;o.rp_max=0.20;o.yaw_max=0.10;
      o.x_target=xf*0.9; o.y_min=0.04;o.y_max=0.34; o.gap=false;
      solve_fast(st,hsol,cell,N,o);
      support_layers()=nullptr;

      // ── 비용 분해 ──
      VectorXd R = cost_residuals(st,hsol,cell,N);
      P = st.num_phases();
      int n_tr=P*6, n_fo=4*st.ncyc(), n_no=12*st.ncyc();
      int n_ep = R.size() - n_tr - n_fo - n_no;                 // GIAC_FIX off 면 0
      double c_tr=R.segment(0,n_tr).squaredNorm();
      double c_fo=R.segment(n_tr,n_fo).squaredNorm();
      double c_no=R.segment(n_tr+n_fo,n_no).squaredNorm();
      double c_ep=(n_ep>0)?R.segment(n_tr+n_fo+n_no,n_ep).squaredNorm():0.0;
      double tot=c_tr+c_fo+c_no+c_ep;
      double share=tot>0?100.0*c_ep/tot:0.0;
      double emax=st.epsilon.size()?st.epsilon.maxCoeff():0.0;
      // eps → 물리 마진: d = eps/(m·g·L_edge). L_edge = 지지다각형 대표 변(hip 간격 ~0.284 m).
      double L_edge = 2.0*prm.hip_offsets(0,1);                  // 0.284 m (좌우 hip 간격)
      double dmar = emax/(prm.mass*9.81*L_edge);
      std::printf("%3d %5.2f %5.2f | %9.3f %9.3f %9.3f %9.3f | %9.3f | %5.1f%% | %8.3f %7.1fcm\n",
                  lvl,phx,phy,c_tr,c_fo,c_no,c_ep,tot,share,emax,dmar*100);
      share_sum+=share; ++nstate;
    }
  }
  std::printf("%s\n", std::string(104,'-').c_str());
  std::printf("평균 eps 점유율 = %.1f%%  (상태 %d개)\n", nstate?share_sum/nstate:0.0, nstate);
  std::printf("\n[읽는 법] eps%% 가 90%% 이상이면 목적함수가 사실상 eps 페널티 하나다 —\n"
              "  tracking/foothold_on_ground/nominal 의 landscape 가 평평해져서 '어느 돌이 나은가'를\n"
              "  고를 근거가 목적함수에 남지 않는다. d_margin 은 그 eps 를 '지지다각형 밖으로 나간 CoM\n"
              "  거리'로 환산한 것 — 수 cm 를 넘으면 GIAC 를 지키는 게 아니라 사서 무력화하는 중이다.\n");
  return 0;
}
