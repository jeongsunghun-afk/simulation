// Di Carlo(2018) Linear Convex MPC (SRBD) + v11 LTV 확장 — gait_sim.controllers.mpc 포팅.
// 응축(condensed) QP: 변수 u=[λ_0..λ_{N-1}] (각 12=4발×3), 첫스텝 λ 반환.
#pragma once
#include <eiquadprog/eiquadprog-fast.hpp>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cmath>
using namespace Eigen;

struct MpcCfg {
  int N; double DT, TOTAL_MASS, G_ACC, MU, LAMZ_MIN, LAMZ_MAX;
  Matrix3d I_BODY;      // CoM기준 복합관성(body, world 아님 — R로 회전)
  VectorXd Qdiag;       // 13
  Vector3d Rdiag;       // per-foot GRF 가중
};

inline Matrix3d _euler_to_R(double r,double p,double y){
  double cr=std::cos(r),sr=std::sin(r),cp=std::cos(p),sp=std::sin(p),cy=std::cos(y),sy=std::sin(y);
  Matrix3d Rx,Ry,Rz;
  Rx<<1,0,0, 0,cr,-sr, 0,sr,cr;
  Ry<<cp,0,sp, 0,1,0, -sp,0,cp;
  Rz<<cy,-sy,0, sy,cy,0, 0,0,1;
  return Rx*Ry*Rz;
}
inline Matrix3d _euler_rate_T(double r,double p){
  double cr=std::cos(r),sr=std::sin(r),cp=std::cos(p),sp=std::sin(p);
  if(std::abs(cp)<1e-3) cp=1e-3*(cp>=0?1.0:-1.0);
  double tp=sp/cp; Matrix3d T;
  T<<1.0, sr*tp, cr*tp,
     0.0, cr,    -sr,
     0.0, sr/cp, cr/cp;
  return T;
}
inline Matrix3d _skew(const Vector3d&v){
  Matrix3d S; S<<0,-v[2],v[1], v[2],0,-v[0], -v[1],v[0],0; return S;
}

// x0,x_ref: 13 [roll,pitch,yaw, px,py,pz, wx,wy,wz, vx,vy,vz, g]
// cs: N×4 접촉, fp: N×4×3 (발위치, CoM 상대, world)
// 반환: lam_des 4×3 (첫스텝). u_opt 실패 시 fallback(균등 수직력).
inline Matrix<double,4,3> mpc_qp_plan(const MpcCfg&c, const VectorXd&x0,
    const std::vector<std::array<int,4>>&cs, const std::vector<std::array<Vector3d,4>>&fp,
    const VectorXd&x_ref){
  const int nx=13,nu=12,N=c.N;
  // LTV: 현재자세 R로 I_world, A_d
  double roll0=x0[0],pitch0=x0[1],yaw0=x0[2];
  Matrix3d R_now=_euler_to_R(roll0,pitch0,yaw0);
  Matrix3d I_world=R_now*c.I_BODY*R_now.transpose();
  Matrix3d I_world_inv=I_world.inverse();
  Matrix<double,13,13> Ac=Matrix<double,13,13>::Zero();
  Ac.block<3,3>(0,6)=_euler_rate_T(roll0,pitch0);
  Ac.block<3,3>(3,9)=Matrix3d::Identity();
  Ac(9,12)=0; Ac(10,12)=0; Ac(11,12)=1.0;
  Matrix<double,13,13> Ad=Matrix<double,13,13>::Identity()+c.DT*Ac;
  std::vector<Matrix<double,13,13>> Adp(N+1); Adp[0]=Matrix<double,13,13>::Identity();
  for(int k=0;k<N;k++) Adp[k+1]=Ad*Adp[k];
  std::vector<MatrixXd> Bc(N);
  for(int k=0;k<N;k++){ MatrixXd B=MatrixXd::Zero(13,12);
    for(int i=0;i<4;i++) if(cs[k][i]){ const Vector3d&r=fp[k][i];
      B.block<3,3>(6,i*3)=I_world_inv*_skew(r);
      B.block<3,3>(9,i*3)=Matrix3d::Identity()/c.TOTAL_MASS; }
    Bc[k]=c.DT*B; }
  // 응축 Aq(N*13×13), Bq(N*13×N*12)
  MatrixXd Aq(N*nx,nx), Bq=MatrixXd::Zero(N*nx,N*nu);
  for(int i=0;i<N;i++){ Aq.block(i*nx,0,nx,nx)=Adp[i+1];
    for(int j=0;j<=i;j++) Bq.block(i*nx,j*nu,nx,nu)=Adp[i-j]*Bc[j]; }
  // X_ref = tile(x_ref,N); err0=Aq*x0 - X_ref
  VectorXd X_ref(N*nx); for(int i=0;i<N;i++) X_ref.segment(i*nx,nx)=x_ref;
  VectorXd err0=Aq*x0 - X_ref;
  MatrixXd QBq(N*nx,N*nu); VectorXd Qerr(N*nx);
  for(int i=0;i<N;i++){ for(int r=0;r<nx;r++){ QBq.row(i*nx+r)=c.Qdiag[r]*Bq.row(i*nx+r);
      Qerr[i*nx+r]=c.Qdiag[r]*err0[i*nx+r]; } }
  VectorXd Rbar(N*nu); for(int k=0;k<N;k++) for(int i=0;i<4;i++) for(int d=0;d<3;d++) Rbar[k*nu+i*3+d]=c.Rdiag[d];
  MatrixXd H=2.0*(Bq.transpose()*QBq); H.diagonal()+=2.0*Rbar;
  VectorXd f=2.0*(Bq.transpose()*Qerr);
  H=(0.5*(H+H.transpose())).eval();
  H.diagonal().array()+=1e-8;   // eiquadprog PD 보장
  // 부등식 G u <= h (stance 마찰추 + λz경계), 등식 A u = b (swing 힘=0)
  bool has_fmax=std::isfinite(c.LAMZ_MAX);
  std::vector<VectorXd> Gr; std::vector<double> hv, br; std::vector<VectorXd> Ar;
  for(int k=0;k<N;k++) for(int i=0;i<4;i++){ int col=k*nu+i*3;
    if(cs[k][i]){
      { VectorXd g=VectorXd::Zero(N*nu); g[col+2]=-1.0; Gr.push_back(g); hv.push_back(-c.LAMZ_MIN); } // λz≥MIN
      { VectorXd g=VectorXd::Zero(N*nu); g[col]=1.0; g[col+2]=-c.MU; Gr.push_back(g); hv.push_back(0.0);}
      { VectorXd g=VectorXd::Zero(N*nu); g[col]=-1.0;g[col+2]=-c.MU; Gr.push_back(g); hv.push_back(0.0);}
      { VectorXd g=VectorXd::Zero(N*nu); g[col+1]=1.0;g[col+2]=-c.MU; Gr.push_back(g); hv.push_back(0.0);}
      { VectorXd g=VectorXd::Zero(N*nu); g[col+1]=-1.0;g[col+2]=-c.MU;Gr.push_back(g); hv.push_back(0.0);}
      if(has_fmax){ VectorXd g=VectorXd::Zero(N*nu); g[col+2]=1.0; Gr.push_back(g); hv.push_back(c.LAMZ_MAX);} // λz≤MAX
    } else {
      for(int d=0;d<3;d++){ VectorXd a=VectorXd::Zero(N*nu); a[col+d]=1.0; Ar.push_back(a); br.push_back(0.0); }
    }
  }
  int nci=(int)Gr.size(), neq=(int)Ar.size(), nvv=N*nu;
  MatrixXd CE=MatrixXd::Zero(neq,nvv); VectorXd ce0=VectorXd::Zero(neq);
  for(int i=0;i<neq;i++){ CE.row(i)=Ar[i]; ce0[i]=-br[i]; }   // A u = b → CE u + (-b)=0
  MatrixXd CI(nci,nvv); VectorXd ci0(nci);
  for(int i=0;i<nci;i++){ CI.row(i)=-Gr[i]; ci0[i]=hv[i]; }    // G u<=h → -G u + h>=0
  VectorXd u(nvv);
  eiquadprog::solvers::EiquadprogFast qp; qp.reset(nvv,neq,nci);
  auto st=qp.solve_quadprog(H,f,CE,ce0,CI,ci0,u);
  Matrix<double,4,3> lam=Matrix<double,4,3>::Zero();
  if(st==eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL){
    for(int i=0;i<4;i++) lam.row(i)=u.segment(i*3,3).transpose();
  } else { // fallback: 균등 수직력
    int ns=0; for(int i=0;i<4;i++) ns+=cs[0][i];
    if(ns>0){ double fz=c.TOTAL_MASS*c.G_ACC/ns; for(int i=0;i<4;i++) if(cs[0][i]) lam(i,2)=fz; }
  }
  return lam;
}
