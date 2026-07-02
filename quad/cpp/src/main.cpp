// Phase3a: crouch_home(IK) + wbic_stance C++ 포팅. 자체 생성한 q_home/com_ref/standing으로 Python 정합 검증.
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

static std::map<std::string,std::vector<double>> parse_dump(const char* p){
  std::map<std::string,std::vector<double>> o; std::ifstream f(p); std::string ln;
  while(std::getline(f,ln)){ std::istringstream ss(ln); std::string k; ss>>k;
    std::vector<double> v; double x; while(ss>>x) v.push_back(x); o[k]=v; } return o; }

int main(int argc,char**argv){
  auto dp=parse_dump("/tmp/wbic_ref.txt");
  const int nq=(int)dp["dims"][0], nv=(int)dp["dims"][1], nu=(int)dp["dims"][2];
  const double MU=dp["MU"][0], MU_MARGIN=dp["MU"][1], LAMZ_MIN=dp["MU"][2];
  const int K=4, nz=nv+3*K;
  const double base_z0=0.52, foot_z0=0.0, REAR_ANKLE=-0.7;   // ours_sphere config

  const char* path=argc>1?argv[1]:"../quad_real_sphere.mjcf";
  char err[1000]=""; mjModel* m=mj_loadXML(path,nullptr,err,1000);
  if(!m){ std::printf("load fail: %s\n",err); return 1; }
  mjData* d=mj_makeData(m);

  // ── 다리 셋업(이름 기반): legqp/legqv(관절 qpos/dof 인덱스), leg_dof, hip_bid, foot geom ──
  const char* legs[4]={"HL","HR","FL","FR"};
  const char* JT[4]={"hip","thigh","calf","foot"};
  std::vector<std::vector<int>> legqp(4),legqv(4); int leg_dof[4],hip_bid[4],fgid[4],fbid[4]; double fr[4];
  for(int i=0;i<4;i++){
    hip_bid[i]=mj_name2id(m,mjOBJ_BODY,(std::string(legs[i])+"_hip_link").c_str());
    for(int t=0;t<4;t++){ int j=mj_name2id(m,mjOBJ_JOINT,(std::string(legs[i])+"_"+JT[t]+"_joint").c_str());
      if(j>=0){ legqp[i].push_back(m->jnt_qposadr[j]); legqv[i].push_back(m->jnt_dofadr[j]); } }
    leg_dof[i]=legqp[i].size();
    int gid=mj_name2id(m,mjOBJ_GEOM,(std::string(legs[i])+"_sphere").c_str());
    fgid[i]=gid; fbid[i]=m->geom_bodyid[gid]; fr[i]=m->geom_size[gid*3];
  }
  auto foot_point=[&](int i){ Vector3d p(d->geom_xpos[fgid[i]*3],d->geom_xpos[fgid[i]*3+1],d->geom_xpos[fgid[i]*3+2]); p[2]-=fr[i]; return p; };
  auto foot_jac=[&](int i){ std::vector<double> jb(3*nv); Vector3d p=foot_point(i); mj_jac(m,d,jb.data(),nullptr,p.data(),fbid[i]);
    Matrix<double,3,Dynamic> J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jb[r*nv+c]; return J; };

  // ── crouch_home(IK): 넓은 발위치 유지하며 무릎굽힘 → q_home/com_ref ──
  VectorXd q_home(nu),com_ref(3);
  {
    for(int i=0;i<nq;i++) d->qpos[i]=0; d->qpos[3]=1;   // (keyframe 없음 가정)
    d->qpos[2]=0.60; mj_forward(m,d);
    Vector2d foot_xy[4]; for(int i=0;i<4;i++) foot_xy[i]=foot_point(i).head(2);
    d->qpos[2]=base_z0;
    for(int i=0;i<4;i++) if(leg_dof[i]==4) d->qpos[legqp[i][3]]=REAR_ANKLE;
    for(int it=0;it<300;it++){ mj_kinematics(m,d); mj_comPos(m,d);
      for(int i=0;i<4;i++){ Vector3d tgt(foot_xy[i][0],foot_xy[i][1],foot_z0); Vector3d e=tgt-foot_point(i);
        Matrix<double,3,Dynamic> Jf=foot_jac(i); Matrix3d J; for(int r=0;r<3;r++)for(int c=0;c<3;c++) J(r,c)=Jf(r,legqv[i][c]);
        Vector3d dq=0.5*(J.transpose()*(J*J.transpose()+1e-4*Matrix3d::Identity()).ldlt().solve(e));
        for(int c=0;c<3;c++) d->qpos[legqp[i][c]]+=dq[c]; } }
    mj_forward(m,d);
    for(int i=0;i<nu;i++) q_home[i]=d->qpos[7+i];
    Vector2d fc(0,0); for(int i=0;i<4;i++) fc+=foot_point(i).head(2); fc/=4.0;
    com_ref<<fc[0],fc[1],d->subtree_com[2];
  }
  // 검증: q_home/com_ref vs Python 덤프
  double eqh=0,ecr=0; for(int i=0;i<nu;i++) eqh=std::max(eqh,std::abs(q_home[i]-dp["q_home"][i]));
  for(int i=0;i<3;i++) ecr=std::max(ecr,std::abs(com_ref[i]-dp["com_ref"][i]));
  std::printf("[검증1] crouch_home: q_home 최대오차=%.2e  com_ref 최대오차=%.2e  → %s\n",
              eqh,ecr,(eqh<1e-6&&ecr<1e-6)?"✅ 정합":"❌ 불일치");

  // ── wbic_stance (C++ crouch_home 결과 그대로 사용, 자체완결) ──
  d->qvel[0]=0; for(int i=0;i<nv;i++) d->qvel[i]=0; mj_forward(m,d);
  eiquadprog::solvers::EiquadprogFast qp; VectorXd tau(nu); double last_qp_ms=0;
  auto wbic_stance=[&](bool timed){
    std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
    Map<Matrix<double,Dynamic,Dynamic,RowMajor>> M(Mb.data(),nv,nv);
    Map<VectorXd> h(d->qfrc_bias,nv); Map<VectorXd> qv(d->qvel,nv);
    std::vector<Matrix<double,3,Dynamic>> Js(K); for(int k=0;k<K;k++) Js[k]=foot_jac(k);
    MatrixXd P=MatrixXd::Zero(nz,nz); VectorXd g=VectorXd::Zero(nz);
    std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
    Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
    Vector3d com(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]);
    Vector3d a_com=Vector3d(120,120,200).cwiseProduct(com_ref-com)-Vector3d(20,20,25).cwiseProduct(Jc*qv);
    P.topLeftCorner(nv,nv)+=Jc.transpose()*Jc; g.head(nv)-=Jc.transpose()*a_com;
    double oerr[3]; mju_quat2Vel(oerr,&d->qpos[3],1.0);
    for(int j=0;j<3;j++){ double a=150*(-oerr[j])-20*qv[3+j]; P(3+j,3+j)+=5.0; g[3+j]-=5.0*a; }
    for(int j=0;j<nu;j++){ double a=60*(q_home[j]-d->qpos[7+j])-5*qv[6+j]; P(6+j,6+j)+=1.0; g[6+j]-=a; }
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
    auto t0=std::chrono::high_resolution_clock::now();
    qp.reset(nz,neq,nineq); qp.solve_quadprog(P,g,A,ce0,CI,ci0,x);
    if(timed) last_qp_ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
    VectorXd qdd=x.head(nv); tau=M.block(6,0,nu,nv)*qdd+h.segment(6,nu);
    for(int k=0;k<K;k++) tau-=Js[k].block(0,6,3,nu).transpose()*x.segment(nv+3*k,3);
  };
  wbic_stance(false);
  double et=0; for(int i=0;i<nu;i++) et=std::max(et,std::abs(tau[i]-dp["tau"][i]));
  std::printf("[검증2] wbic_stance tau 최대오차=%.2e Nm → %s\n", et, et<1e-3?"✅ 정합":"❌ 불일치");
  int Nb=5000; auto t0=std::chrono::high_resolution_clock::now(); double qs=0;
  for(int i=0;i<Nb;i++){ wbic_stance(true); qs+=last_qp_ms; }
  double fm=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count()/Nb;
  std::printf("[속도] wbic_stance 전체=%.3fms(%.0fHz) QP=%.3fms\n",fm,1000/fm,qs/Nb);
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
