// QuadControl — quad_mpc_wbic.py 컨트롤러(모델 의존부) C++ 이관.
// crouch_home(IK) · wbic_stance · wbic_track(스윙WBIC) · compute_Icom · body_x0 · mpc_grf.
// main.cpp(검증)와 trot_sim.cpp(closed-loop)가 공유. 검증된 로직 그대로.
#pragma once
#include <mujoco/mujoco.h>
#include <eiquadprog/eiquadprog-fast.hpp>
#include <Eigen/Dense>
#include "mpc.hpp"
#include <vector>
#include <array>
#include <map>
#include <set>
#include <string>
#include <cmath>
using namespace Eigen;

struct QuadControl {
  mjModel* m=nullptr; mjData* d=nullptr;
  int nq=0,nv=0,nu=0;
  std::vector<std::vector<int>> legqp{4},legqv{4};
  int leg_dof[4]={0}, hip_bid[4]={0}, fgid[4]={0}, fbid[4]={0}; double fr[4]={0};
  const char* legs[4]={"HL","HR","FL","FR"};
  // 상수
  double MU=0.6, MU_MARGIN=0.707, LAMZ_MIN=1.0;         // wbic 마찰
  double base_z0=0.52, REAR_ANKLE=-0.7;                  // ours_sphere
  VectorXd q_home; Vector3d com_ref;
  VectorXd tau_peak, qmin, qmax; std::vector<char> is_ankle;
  std::array<Vector2d,4> foot_hip_off; std::array<double,4> foot_gz0;
  MpcCfg mpc; double _body_terr=0.0;
  eiquadprog::solvers::EiquadprogFast _qp_st, _qp_tr;

  void load(const char* path){
    char err[1000]=""; m=mj_loadXML(path,nullptr,err,1000);
    if(!m){ std::fprintf(stderr,"load fail: %s\n",err); std::exit(1); }
    m->opt.timestep = 0.001;   // ★Python과 동일(1kHz, TIMESTEP env 기본). mjcf 0.002 오버라이드
    d=mj_makeData(m); nq=m->nq; nv=m->nv; nu=m->nu;
    const char* JT[4]={"hip","thigh","calf","foot"};
    for(int i=0;i<4;i++){
      hip_bid[i]=mj_name2id(m,mjOBJ_BODY,(std::string(legs[i])+"_hip_link").c_str());
      legqp[i].clear(); legqv[i].clear();
      for(int t=0;t<4;t++){ int j=mj_name2id(m,mjOBJ_JOINT,(std::string(legs[i])+"_"+JT[t]+"_joint").c_str());
        if(j>=0){ legqp[i].push_back(m->jnt_qposadr[j]); legqv[i].push_back(m->jnt_dofadr[j]); } }
      leg_dof[i]=(int)legqp[i].size();
      int gid=mj_name2id(m,mjOBJ_GEOM,(std::string(legs[i])+"_sphere").c_str());
      fgid[i]=gid; fbid[i]=m->geom_bodyid[gid]; fr[i]=m->geom_size[gid*3];
    }
    // tau_peak / qmin·qmax / ankle (non-free joint 순서=actuator)
    tau_peak.resize(nu); qmin.resize(nu); qmax.resize(nu); is_ankle.assign(nu,0);
    int a=0; for(int j=0;j<m->njnt;j++){ if(m->jnt_type[j]==mjJNT_FREE) continue;
      double frc=m->jnt_actfrcrange[j*2+1]; tau_peak[a]=frc>0?frc:1e8;
      if(m->jnt_limited[j]){ qmin[a]=m->jnt_range[j*2]; qmax[a]=m->jnt_range[j*2+1]; } else { qmin[a]=-1e9; qmax[a]=1e9; }
      a++; }
    for(int i=0;i<4;i++) if(leg_dof[i]==4) is_ankle[legqv[i][3]-6]=1;
    q_home.resize(nu);
  }
  Vector3d foot_point(int i){ Vector3d p(d->geom_xpos[fgid[i]*3],d->geom_xpos[fgid[i]*3+1],d->geom_xpos[fgid[i]*3+2]); p[2]-=fr[i]; return p; }
  Matrix<double,3,Dynamic> foot_jac(int i){ std::vector<double> jb(3*nv); Vector3d p=foot_point(i);
    mj_jac(m,d,jb.data(),nullptr,p.data(),fbid[i]);
    Matrix<double,3,Dynamic> J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jb[r*nv+c]; return J; }

  // crouch_home: 넓은 발위치 유지 무릎굽힘 → q_home/com_ref/standing. + foot_hip_off/foot_gz0
  void crouch_home(double bz=-1){
    double base_z = (bz>0? bz : base_z0), foot_z0=0.0;
    for(int i=0;i<nq;i++) d->qpos[i]=0; d->qpos[3]=1;
    d->qpos[2]=0.60; mj_forward(m,d);
    Vector2d foot_xy[4]; for(int i=0;i<4;i++) foot_xy[i]=foot_point(i).head(2);
    d->qpos[2]=base_z;
    for(int i=0;i<4;i++) if(leg_dof[i]==4) d->qpos[legqp[i][3]]=REAR_ANKLE;
    for(int it=0;it<300;it++){ mj_kinematics(m,d); mj_comPos(m,d);
      for(int i=0;i<4;i++){ Vector3d tgt(foot_xy[i][0],foot_xy[i][1],foot_z0); Vector3d e=tgt-foot_point(i);
        Matrix<double,3,Dynamic> Jf=foot_jac(i); Matrix3d J; for(int r=0;r<3;r++)for(int cc=0;cc<3;cc++) J(r,cc)=Jf(r,legqv[i][cc]);
        Vector3d dq=0.5*(J.transpose()*(J*J.transpose()+1e-4*Matrix3d::Identity()).ldlt().solve(e));
        for(int cc=0;cc<3;cc++) d->qpos[legqp[i][cc]]+=dq[cc]; } }
    mj_forward(m,d);
    for(int i=0;i<nu;i++) q_home[i]=d->qpos[7+i];
    Vector2d fc(0,0); for(int i=0;i<4;i++) fc+=foot_point(i).head(2); fc/=4.0;
    com_ref<<fc[0],fc[1],d->subtree_com[2];
    for(int i=0;i<4;i++){ foot_hip_off[i]=foot_point(i).head(2)-Vector2d(d->xpos[hip_bid[i]*3],d->xpos[hip_bid[i]*3+1]);
      foot_gz0[i]=foot_point(i)[2]; }
    for(int i=0;i<nv;i++) d->qvel[i]=0; mj_forward(m,d);
  }
  Matrix3d compute_Icom(){
    Vector3d com(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]); Matrix3d I=Matrix3d::Zero();
    for(int b=1;b<m->nbody;b++){ double ms=m->body_mass[b]; if(ms<=0) continue;
      Vector3d r(d->xipos[b*3]-com[0],d->xipos[b*3+1]-com[1],d->xipos[b*3+2]-com[2]);
      Matrix<double,3,3,RowMajor> Rb(&d->ximat[b*9]);
      Vector3d bi(m->body_inertia[b*3],m->body_inertia[b*3+1],m->body_inertia[b*3+2]);
      Matrix3d Ib=Rb*bi.asDiagonal()*Rb.transpose();
      I += Ib + ms*(r.dot(r)*Matrix3d::Identity()-r*r.transpose()); }
    return I;
  }
  // setup_mpc: crouch_home 이후 호출. TROT_Q(arming 후 가중) 사용.
  void setup_mpc(){
    mj_forward(m,d);
    mpc.N=14; mpc.DT=0.02; mpc.TOTAL_MASS=m->body_subtreemass[0]; mpc.G_ACC=9.81;
    mpc.MU=MU*MU_MARGIN; mpc.LAMZ_MIN=1.0; mpc.LAMZ_MAX=2.0*mpc.TOTAL_MASS*9.81;
    mpc.I_BODY=compute_Icom();
    mpc.Qdiag.resize(13); mpc.Qdiag<<200,200,100, 0,0,200, 0,0,1, 10,10,1, 0;   // TROT_Q
    mpc.Rdiag=Vector3d(1e-6,1e-6,1e-6);
  }
  VectorXd body_x0(){
    double Rm[9]; mju_quat2Mat(Rm,&d->qpos[3]);
    Matrix<double,3,3,RowMajor> R(Rm);
    double pitch=std::asin(std::max(-1.0,std::min(1.0,-R(2,0))));
    double roll=std::atan2(R(2,1),R(2,2)), yaw=std::atan2(R(1,0),R(0,0));
    std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
    Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int cc=0;cc<nv;cc++) Jc(r,cc)=jcb[r*nv+cc];
    Map<VectorXd> qv(d->qvel,nv); Vector3d vcom=Jc*qv;
    Vector3d wb(d->qvel[3],d->qvel[4],d->qvel[5]); Vector3d omega_w=R*wb;
    VectorXd x(13); x<<roll,pitch,yaw, d->subtree_com[0],d->subtree_com[1],d->subtree_com[2],
      omega_w[0],omega_w[1],omega_w[2], vcom[0],vcom[1],vcom[2], -9.81;
    return x;
  }
  Matrix<double,4,3> mpc_grf(const VectorXd& x_ref, const std::vector<std::array<int,4>>& cs){
    Vector3d com(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]);
    std::array<Vector3d,4> frel; for(int i=0;i<4;i++) frel[i]=foot_point(i)-com;
    std::vector<std::array<Vector3d,4>> fp(mpc.N,frel);
    return mpc_qp_plan(mpc,x0_or(x_ref),cs,fp,x_ref);
  }
  VectorXd x0_or(const VectorXd&){ return body_x0(); }  // (가독성용 wrapper)

  // ── wbic_stance (검증2와 동일) ──
  bool wbic_stance(){
    int K=4, nz=nv+3*K;
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
    _qp_st.reset(nz,neq,nineq); auto st=_qp_st.solve_quadprog(P,g,A,ce0,CI,ci0,x);
    if(st!=eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) return false;
    VectorXd qdd=x.head(nv); VectorXd tau=M.block(6,0,nu,nv)*qdd+h.segment(6,nu);
    for(int k=0;k<K;k++) tau-=Js[k].block(0,6,3,nu).transpose()*x.segment(nv+3*k,3);
    for(int i=0;i<nu;i++) d->ctrl[i]=std::max(-tau_peak[i],std::min(tau_peak[i],tau[i]));
    return true;
  }
  // ── wbic_track (검증3과 동일: 기본경로) ──
  bool wbic_track(const std::vector<int>& contacts, const std::map<int,std::pair<Vector3d,Vector3d>>& swing,
                  const Vector3d lam[4], double w_lam=10.0){
    int Kc=(int)contacts.size(), nzt=nv+3*Kc; auto sl=[&](int k){ return nv+3*k; };
    std::vector<Matrix<double,3,Dynamic>> cjac(Kc); std::vector<Vector3d> cpos(Kc),clam(Kc);
    for(int k=0;k<Kc;k++){ int c=contacts[k]; cjac[k]=foot_jac(c); cpos[k]=foot_point(c); clam[k]=lam[c]; }
    std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
    Map<Matrix<double,Dynamic,Dynamic,RowMajor>> M(Mb.data(),nv,nv);
    Map<VectorXd> h(d->qfrc_bias,nv); VectorXd qv=Map<VectorXd>(d->qvel,nv);
    MatrixXd P=MatrixXd::Zero(nzt,nzt); VectorXd g=VectorXd::Zero(nzt);
    std::set<int> sw_vidx;
    for(auto&kv:swing){ int leg=kv.first; Matrix<double,3,Dynamic> J=foot_jac(leg);
      Vector3d accel=2400.0*(kv.second.first-foot_point(leg))+110.0*(kv.second.second-J*qv);
      P.topLeftCorner(nv,nv)+=90.0*(J.transpose()*J); g.head(nv)-=90.0*(J.transpose()*accel);
      for(int t=0;t<leg_dof[leg];t++) sw_vidx.insert(legqv[leg][t]); }
    std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
    Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
    double oerr[3]; mju_quat2Vel(oerr,&d->qpos[3],1.0);
    for(int j=0;j<3;j++){ double a=150*(-oerr[j])-20*qv[3+j]; P(3+j,3+j)+=5.0; g[3+j]-=5.0*a; }
    double zref=com_ref[2]+_body_terr; Vector3d Jcqv=Jc*qv;
    double a_z=200*(zref-d->subtree_com[2])-25*Jcqv[2];
    P.topLeftCorner(nv,nv)+=150.0*(Jc.row(2).transpose()*Jc.row(2)); g.head(nv)-=150.0*a_z*Jc.row(2).transpose();
    for(int j=0;j<nu;j++){ double a_post=60*(q_home[j]-d->qpos[7+j])-5*qv[6+j];
      double w_post = (is_ankle[j])?20.0 : (sw_vidx.count(6+j)?0.1:1.0);
      P(6+j,6+j)+=w_post; g[6+j]-=w_post*a_post; }
    P.topLeftCorner(nv,nv)+=1e-3*MatrixXd::Identity(nv,nv);
    for(int k=0;k<Kc;k++){ P.block(sl(k),sl(k),3,3)+=w_lam*Matrix3d::Identity(); g.segment(sl(k),3)-=w_lam*clam[k]; }
    int neq=6+3*Kc; MatrixXd A=MatrixXd::Zero(neq,nzt); VectorXd b=VectorXd::Zero(neq);
    A.block(0,0,6,nv)=M.topRows(6); b.head(6)=-h.head(6);
    for(int k=0;k<Kc;k++) A.block(0,sl(k),6,3)=-cjac[k].leftCols(6).transpose();
    for(int k=0;k<Kc;k++) A.block(6+3*k,0,3,nv)=cjac[k];
    VectorXd lb=VectorXd::Constant(nzt,-1e8),ub=VectorXd::Constant(nzt,1e8);
    { double tla=0.05,c2=0.5*tla*tla;
      for(int j=0;j<nu;j++){ double qj=d->qpos[7+j],dqj=qv[6+j];
        double ubp=(qmax[j]-qj-dqj*tla)/c2, lbp=(qmin[j]-qj-dqj*tla)/c2;
        double u=std::min(ub[6+j],ubp), l=std::max(lb[6+j],lbp);
        if(l<=u){ ub[6+j]=u; lb[6+j]=l; } } }
    for(int k=0;k<Kc;k++) lb[sl(k)+2]=LAMZ_MIN;
    std::vector<VectorXd> Gr; std::vector<double> hv;
    int sgn[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    for(int k=0;k<Kc;k++){ int o=sl(k); for(int s=0;s<4;s++){ VectorXd r=VectorXd::Zero(nzt);
      r[o]=sgn[s][0]; r[o+1]=sgn[s][1]; r[o+2]=-MU*MU_MARGIN; Gr.push_back(r); hv.push_back(0.0); } }
    VectorXd h_act=h.segment(6,nu); MatrixXd T_mat=MatrixXd::Zero(nu,nzt); T_mat.leftCols(nv)=M.block(6,0,nu,nv);
    for(int k=0;k<Kc;k++) T_mat.block(0,sl(k),nu,3)=-cjac[k].block(0,6,3,nu).transpose();
    for(int i=0;i<nu;i++){ Gr.push_back(T_mat.row(i)); hv.push_back(tau_peak[i]-h_act[i]); }
    for(int i=0;i<nu;i++){ Gr.push_back(-T_mat.row(i)); hv.push_back(tau_peak[i]+h_act[i]); }
    P=(0.5*(P+P.transpose())).eval()+1e-8*MatrixXd::Identity(nzt,nzt);
    std::vector<VectorXd> CIr; std::vector<double> ci0v;
    for(size_t i=0;i<Gr.size();i++){ CIr.push_back(-Gr[i]); ci0v.push_back(hv[i]); }
    for(int i=0;i<nzt;i++){ if(lb[i]>-1e7){ VectorXd r=VectorXd::Zero(nzt); r[i]=1; CIr.push_back(r); ci0v.push_back(-lb[i]); }
                            if(ub[i]< 1e7){ VectorXd r=VectorXd::Zero(nzt); r[i]=-1; CIr.push_back(r); ci0v.push_back(ub[i]); } }
    int nci=(int)CIr.size(); MatrixXd CI(nci,nzt); VectorXd ci0(nci);
    for(int i=0;i<nci;i++){ CI.row(i)=CIr[i]; ci0[i]=ci0v[i]; }
    MatrixXd CE=A; VectorXd ce0=-b, x(nzt);
    _qp_tr.reset(nzt,neq,nci); auto st=_qp_tr.solve_quadprog(P,g,CE,ce0,CI,ci0,x);
    if(st!=eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) return false;
    VectorXd qdd=x.head(nv); VectorXd tau=M.block(6,0,nu,nv)*qdd+h.segment(6,nu);
    for(int k=0;k<Kc;k++) tau-=cjac[k].block(0,6,3,nu).transpose()*x.segment(sl(k),3);
    for(int i=0;i<nu;i++) d->ctrl[i]=std::max(-tau_peak[i],std::min(tau_peak[i],tau[i]));
    return true;
  }
};
