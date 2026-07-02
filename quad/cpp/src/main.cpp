// Phase2b: wbic_stance C++ 포팅 + Python 대비 토크 정합·QP속도 검증
// Python(quad_mpc_wbic.py wbic_stance)와 동일 로직: CoM/자세/posture task + 부동베이스·접촉 등식 + 마찰추 부등식.
#include <mujoco/mujoco.h>
#include <eiquadprog/eiquadprog-fast.hpp>
#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
using namespace Eigen;

static std::map<std::string, std::vector<double>> parse_dump(const char* path) {
  std::map<std::string, std::vector<double>> out; std::ifstream f(path); std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line); std::string key; ss >> key;
    std::vector<double> v; double x; while (ss >> x) v.push_back(x); out[key] = v;
  }
  return out;
}

int main(int argc, char** argv) {
  auto dp = parse_dump("/tmp/wbic_ref.txt");
  const int nq=(int)dp["dims"][0], nv=(int)dp["dims"][1], nu=(int)dp["dims"][2];
  const double MU=dp["MU"][0], MU_MARGIN=dp["MU"][1], LAMZ_MIN=dp["MU"][2];
  const int K=4, nz=nv+3*K;

  const char* path = argc>1 ? argv[1] : "../quad_real_sphere.mjcf";
  char err[1000]=""; mjModel* m = mj_loadXML(path,nullptr,err,1000);
  if(!m){ std::printf("load fail: %s\n",err); return 1; }
  mjData* d = mj_makeData(m);
  for(int i=0;i<nq;i++) d->qpos[i]=dp["qpos"][i];
  for(int i=0;i<nv;i++) d->qvel[i]=dp["qvel"][i];
  mj_forward(m,d);

  const char* legs[4]={"HL","HR","FL","FR"};
  int fgid[4],fbid[4]; double fr[4];
  for(int i=0;i<4;i++){ std::string gn=std::string(legs[i])+"_sphere";
    int gid=mj_name2id(m,mjOBJ_GEOM,gn.c_str()); fgid[i]=gid; fbid[i]=m->geom_bodyid[gid]; fr[i]=m->geom_size[gid*3]; }
  VectorXd q_home(nu),com_ref(3);
  for(int i=0;i<nu;i++) q_home[i]=dp["q_home"][i];
  for(int i=0;i<3;i++) com_ref[i]=dp["com_ref"][i];

  // ── wbic_stance 계산 함수(벤치용 람다) ──
  VectorXd tau(nu); eiquadprog::solvers::EiquadprogFast qp;
  double last_qp_ms=0;
  auto wbic_stance = [&](bool time_qp)->bool {
    std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
    Map<Matrix<double,Dynamic,Dynamic,RowMajor>> M(Mb.data(),nv,nv);
    Map<VectorXd> h(d->qfrc_bias,nv); Map<VectorXd> qv(d->qvel,nv);
    std::vector<Matrix<double,3,Dynamic>> Js(K);
    for(int k=0;k<K;k++){ Vector3d p(d->geom_xpos[fgid[k]*3],d->geom_xpos[fgid[k]*3+1],d->geom_xpos[fgid[k]*3+2]); p[2]-=fr[k];
      std::vector<double> jb(3*nv); mj_jac(m,d,jb.data(),nullptr,p.data(),fbid[k]);
      Matrix<double,3,Dynamic> J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jb[r*nv+c]; Js[k]=J; }
    MatrixXd P=MatrixXd::Zero(nz,nz); VectorXd g=VectorXd::Zero(nz);
    // CoM task
    std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
    Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
    Vector3d com(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]);
    Vector3d kp(120,120,200),kd(20,20,25);
    Vector3d a_com=kp.cwiseProduct(com_ref-com)-kd.cwiseProduct(Jc*qv);
    P.topLeftCorner(nv,nv)+=Jc.transpose()*Jc; g.head(nv)-=Jc.transpose()*a_com;
    // base ori
    double oerr[3]; mju_quat2Vel(oerr,&d->qpos[3],1.0);
    for(int j=0;j<3;j++){ double a=150*(-oerr[j])-20*qv[3+j]; P(3+j,3+j)+=5.0; g[3+j]-=5.0*a; }
    // posture
    for(int j=0;j<nu;j++){ double a=60*(q_home[j]-d->qpos[7+j])-5*qv[6+j]; P(6+j,6+j)+=1.0; g[6+j]-=a; }
    P.topLeftCorner(nv,nv)+=1e-4*MatrixXd::Identity(nv,nv);
    for(int k=0;k<K;k++) P.block(nv+3*k,nv+3*k,3,3)+=1e-3*Matrix3d::Identity();
    // equality: 6(부동베이스) + 3K(접촉 무가속)
    int neq=6+3*K; MatrixXd A=MatrixXd::Zero(neq,nz); VectorXd b=VectorXd::Zero(neq);
    A.block(0,0,6,nv)=M.topRows(6); for(int j=0;j<6;j++) b[j]=-h[j];
    for(int k=0;k<K;k++) A.block(0,nv+3*k,6,3)=-Js[k].leftCols(6).transpose();
    for(int k=0;k<K;k++) A.block(6+3*k,0,3,nv)=Js[k];
    // inequality: 마찰추(4/접촉) + λz>=LAMZ_MIN
    int nineq=4*K+K; MatrixXd CI=MatrixXd::Zero(nineq,nz); VectorXd ci0=VectorXd::Zero(nineq);
    int sgn[4][2]={{1,0},{-1,0},{0,1},{0,-1}}; int rr=0;
    for(int k=0;k<K;k++){ int o=nv+3*k;
      for(int s=0;s<4;s++){ CI(rr,o)=-sgn[s][0]; CI(rr,o+1)=-sgn[s][1]; CI(rr,o+2)=MU*MU_MARGIN; ci0[rr]=0; rr++; }
      CI(rr,o+2)=1.0; ci0[rr]=-LAMZ_MIN; rr++; }
    P=(0.5*(P+P.transpose())).eval()+1e-8*MatrixXd::Identity(nz,nz);
    VectorXd ce0=-b, x(nz);
    if(time_qp){ auto t0=std::chrono::high_resolution_clock::now();
      qp.reset(nz,neq,nineq); qp.solve_quadprog(P,g,A,ce0,CI,ci0,x);
      last_qp_ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
    } else { qp.reset(nz,neq,nineq); qp.solve_quadprog(P,g,A,ce0,CI,ci0,x); }
    VectorXd qdd=x.head(nv);
    tau=M.block(6,0,nu,nv)*qdd+h.segment(6,nu);
    for(int k=0;k<K;k++) tau-=Js[k].block(0,6,3,nu).transpose()*x.segment(nv+3*k,3);
    return true;
  };

  wbic_stance(false);
  // 정합 검증
  double maxerr=0; VectorXd tref(nu);
  std::printf("[검증] C++ tau vs Python tau:\n");
  for(int i=0;i<nu;i++){ tref[i]=dp["tau"][i]; maxerr=std::max(maxerr,std::abs(tau[i]-tref[i])); }
  std::printf("  C++: "); for(int i=0;i<nu;i++) std::printf("%.3f ",tau[i]); std::printf("\n");
  std::printf("  Py : "); for(int i=0;i<nu;i++) std::printf("%.3f ",tref[i]); std::printf("\n");
  std::printf("  ★최대오차 = %.2e Nm  → %s\n", maxerr, maxerr<1e-3?"✅ 정합":"❌ 불일치");
  // 속도 벤치
  int Nb=5000; auto t0=std::chrono::high_resolution_clock::now();
  double qpsum=0; for(int i=0;i<Nb;i++){ wbic_stance(true); qpsum+=last_qp_ms; }
  double full_ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count()/Nb;
  std::printf("[속도] wbic_stance 전체=%.3fms(%.0fHz) / QP solve만=%.3fms  (Python: 전체~3ms QP~0.25ms)\n",
              full_ms,1000/full_ms,qpsum/Nb);
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
