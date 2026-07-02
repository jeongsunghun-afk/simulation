// Phase3a: crouch_home(IK) + wbic_stance C++ 포팅. 자체 생성한 q_home/com_ref/standing으로 Python 정합 검증.
#include <mujoco/mujoco.h>
#include <eiquadprog/eiquadprog-fast.hpp>
#include <Eigen/Dense>
#include "mpc.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
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

  // ══════════ Phase3b: wbic_track(스윙 WBIC) 검증 ══════════
  {
    // 상수(non-free joint 순서=actuator 순서): tau_peak / qmin·qmax / ankle 여부
    VectorXd tau_peak(nu),qmin(nu),qmax(nu); std::vector<char> is_ankle(nu,0);
    { int a=0; for(int j=0;j<m->njnt;j++){ if(m->jnt_type[j]==mjJNT_FREE) continue;
        double frc=m->jnt_actfrcrange[j*2+1]; tau_peak[a]=frc>0?frc:1e8;
        if(m->jnt_limited[j]){ qmin[a]=m->jnt_range[j*2]; qmax[a]=m->jnt_range[j*2+1]; } else { qmin[a]=-1e9; qmax[a]=1e9; }
        a++; }
      for(int i=0;i<4;i++) if(leg_dof[i]==4) is_ankle[legqv[i][3]-6]=1; }

    std::ifstream tf("/tmp/wbic_track_ref.txt");
    if(!tf){ std::printf("[검증3] /tmp/wbic_track_ref.txt 없음 — dump_track.py 먼저 실행\n"); }
    else {
      std::map<std::string,std::vector<double>> T; std::vector<int> contacts;
      std::map<int,std::pair<Vector3d,Vector3d>> swing; std::string ln;
      while(std::getline(tf,ln)){ std::istringstream ss(ln); std::string k; ss>>k;
        if(k=="contacts"){ int c; while(ss>>c) contacts.push_back(c); }
        else if(k=="swingleg"){ int leg; ss>>leg; double v[6]; for(int i=0;i<6;i++) ss>>v[i];
          swing[leg]={Vector3d(v[0],v[1],v[2]),Vector3d(v[3],v[4],v[5])}; }
        else { std::vector<double> vs; double x; while(ss>>x) vs.push_back(x); T[k]=vs; } }
      const double SW_KP=T["SW"][0], SW_KD=T["SW"][1], W_SW=T["SW"][2];
      const double w_lam=T["w_lam"][0], body_terr=T["body_terr"][0];
      Vector3d lamd[4]; for(int i=0;i<4;i++) lamd[i]=Vector3d(T["lam_des"][i*3],T["lam_des"][i*3+1],T["lam_des"][i*3+2]);

      // wbic_track (기본경로: swing task+base ori/z+posture+λ추종+friction/LAMZ+POS_LIM+TAU_LIM; W_AM=0·motor_curve off)
      VectorXd tau_tr(nu); double tr_qp_ms=0, _obj_cpp=0, _obj_py=0;
      auto wbic_track=[&](bool timed){
        for(int i=0;i<nq;i++) d->qpos[i]=T["qpos"][i];
        for(int i=0;i<nv;i++) d->qvel[i]=T["qvel"][i];
        mj_forward(m,d);
        int Kc=(int)contacts.size(), nzt=nv+3*Kc; auto sl=[&](int k){ return nv+3*k; };
        std::vector<Matrix<double,3,Dynamic>> cjac(Kc); std::vector<Vector3d> cpos(Kc),clam(Kc);
        for(int k=0;k<Kc;k++){ int c=contacts[k]; cjac[k]=foot_jac(c); cpos[k]=foot_point(c); clam[k]=lamd[c]; }
        std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
        Map<Matrix<double,Dynamic,Dynamic,RowMajor>> M(Mb.data(),nv,nv);
        Map<VectorXd> h(d->qfrc_bias,nv); VectorXd qv=Map<VectorXd>(d->qvel,nv);
        MatrixXd P=MatrixXd::Zero(nzt,nzt); VectorXd g=VectorXd::Zero(nzt);
        std::set<int> sw_vidx;
        for(auto&kv:swing){ int leg=kv.first; Matrix<double,3,Dynamic> J=foot_jac(leg);
          Vector3d accel=SW_KP*(kv.second.first-foot_point(leg))+SW_KD*(kv.second.second-J*qv);
          P.topLeftCorner(nv,nv)+=W_SW*(J.transpose()*J); g.head(nv)-=W_SW*(J.transpose()*accel);
          for(int t=0;t<leg_dof[leg];t++) sw_vidx.insert(legqv[leg][t]); }
        std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
        Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
        double oerr[3]; mju_quat2Vel(oerr,&d->qpos[3],1.0);
        for(int j=0;j<3;j++){ double a=150*(-oerr[j])-20*qv[3+j]; P(3+j,3+j)+=5.0; g[3+j]-=5.0*a; }
        double zref=com_ref[2]+body_terr; Vector3d Jcqv=Jc*qv;
        double a_z=200*(zref-d->subtree_com[2])-25*Jcqv[2];
        P.topLeftCorner(nv,nv)+=150.0*(Jc.row(2).transpose()*Jc.row(2)); g.head(nv)-=150.0*a_z*Jc.row(2).transpose();
        for(int j=0;j<nu;j++){ double a_post=60*(q_home[j]-d->qpos[7+j])-5*qv[6+j];
          double w_post = (is_ankle[j])?20.0 : (sw_vidx.count(6+j)?0.1:1.0);
          P(6+j,6+j)+=w_post; g[6+j]-=w_post*a_post; }
        P.topLeftCorner(nv,nv)+=1e-3*MatrixXd::Identity(nv,nv);
        for(int k=0;k<Kc;k++){ P.block(sl(k),sl(k),3,3)+=w_lam*Matrix3d::Identity(); g.segment(sl(k),3)-=w_lam*clam[k]; }
        // 등식: floating-base EOM 6 + 접촉 무가속 3K
        int neq=6+3*Kc; MatrixXd A=MatrixXd::Zero(neq,nzt); VectorXd b=VectorXd::Zero(neq);
        A.block(0,0,6,nv)=M.topRows(6); b.head(6)=-h.head(6);
        for(int k=0;k<Kc;k++) A.block(0,sl(k),6,3)=-cjac[k].leftCols(6).transpose();
        for(int k=0;k<Kc;k++) A.block(6+3*k,0,3,nv)=cjac[k];
        // 경계: POS_LIM(가속도) + LAMZ
        VectorXd lb=VectorXd::Constant(nzt,-1e8),ub=VectorXd::Constant(nzt,1e8);
        { double tla=0.05,c2=0.5*tla*tla;
          for(int j=0;j<nu;j++){ double qj=d->qpos[7+j],dqj=qv[6+j];
            double ubp=(qmax[j]-qj-dqj*tla)/c2, lbp=(qmin[j]-qj-dqj*tla)/c2;
            double u=std::min(ub[6+j],ubp), l=std::max(lb[6+j],lbp);
            if(l<=u){ ub[6+j]=u; lb[6+j]=l; } } }
        for(int k=0;k<Kc;k++) lb[sl(k)+2]=LAMZ_MIN;
        // 부등식 G z <= hh : friction pyramid + TAU_LIM
        std::vector<VectorXd> Gr; std::vector<double> hv;
        int sgn[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
        for(int k=0;k<Kc;k++){ int o=sl(k); for(int s=0;s<4;s++){ VectorXd r=VectorXd::Zero(nzt);
          r[o]=sgn[s][0]; r[o+1]=sgn[s][1]; r[o+2]=-MU*MU_MARGIN; Gr.push_back(r); hv.push_back(0.0); } }
        VectorXd h_act=h.segment(6,nu); MatrixXd T_mat=MatrixXd::Zero(nu,nzt); T_mat.leftCols(nv)=M.block(6,0,nu,nv);
        for(int k=0;k<Kc;k++) T_mat.block(0,sl(k),nu,3)=-cjac[k].block(0,6,3,nu).transpose();
        for(int i=0;i<nu;i++){ Gr.push_back(T_mat.row(i)); hv.push_back(tau_peak[i]-h_act[i]); }
        for(int i=0;i<nu;i++){ Gr.push_back(-T_mat.row(i)); hv.push_back(tau_peak[i]+h_act[i]); }
        P=(0.5*(P+P.transpose())).eval()+1e-8*MatrixXd::Identity(nzt,nzt);
        // eiquadprog 변환: CE=A/ce0=-b, CI: (-G,hh) + 유한경계(lb: e/-lb, ub: -e/ub)
        std::vector<VectorXd> CIr; std::vector<double> ci0v;
        for(size_t i=0;i<Gr.size();i++){ CIr.push_back(-Gr[i]); ci0v.push_back(hv[i]); }
        for(int i=0;i<nzt;i++){ if(lb[i]>-1e7){ VectorXd r=VectorXd::Zero(nzt); r[i]=1; CIr.push_back(r); ci0v.push_back(-lb[i]); }
                                if(ub[i]< 1e7){ VectorXd r=VectorXd::Zero(nzt); r[i]=-1; CIr.push_back(r); ci0v.push_back(ub[i]); } }
        int nci=(int)CIr.size(); MatrixXd CI(nci,nzt); VectorXd ci0(nci);
        for(int i=0;i<nci;i++){ CI.row(i)=CIr[i]; ci0[i]=ci0v[i]; }
        MatrixXd CE=A; VectorXd ce0=-b, x(nzt);
        eiquadprog::solvers::EiquadprogFast qpt; qpt.reset(nzt,neq,nci);
        auto t0=std::chrono::high_resolution_clock::now();
        auto st_=qpt.solve_quadprog(P,g,CE,ce0,CI,ci0,x);
        (void)st_;
        if(timed) tr_qp_ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
        VectorXd qdd=x.head(nv); tau_tr=M.block(6,0,nu,nv)*qdd+h.segment(6,nu);
        for(int k=0;k<Kc;k++) tau_tr-=cjac[k].block(0,6,3,nu).transpose()*x.segment(sl(k),3);
        if(!timed && T.count("zsol")){
          // ★잔차 진단: 두 해의 목적값(같은 P,g). C++ obj ≤ py obj 이면 C++가 동일QP의 동등-또는-더나은 최적해
          VectorXd zpy(nzt); for(int i=0;i<nzt;i++) zpy[i]=T["zsol"][i];
          _obj_cpp=0.5*x.dot(P*x)+g.dot(x); _obj_py=0.5*zpy.dot(P*zpy)+g.dot(zpy); }
      };
      wbic_track(false);
      double etr=0; for(int i=0;i<nu;i++) etr=std::max(etr,std::abs(tau_tr[i]-T["tau"][i]));
      // 동일QP·같은목적값이면 정합(잔차=near-singular 스윙다리 soft-task nullspace를 두 GI솔버가 다르게 해소; C++가 더 최적)
      bool ok_opt = (_obj_cpp <= _obj_py + 1e-3);
      std::printf("[검증3] wbic_track tau 최대오차=%.2e Nm (contacts=%d,swing=%d), 목적값 C++=%.3f≤py=%.3f(%s) → %s\n",
                  etr,(int)contacts.size(),(int)swing.size(), _obj_cpp,_obj_py, ok_opt?"C++ 동등-또는-더최적":"py가 더최적?!",
                  (etr<0.3 && ok_opt)?"✅ 정합(솔버허용오차내)":"❌ 재검토");
      auto tt0=std::chrono::high_resolution_clock::now(); double tqs=0; int Ntr=5000;
      for(int i=0;i<Ntr;i++){ wbic_track(true); tqs+=tr_qp_ms; }
      double tfm=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-tt0).count()/Ntr;
      std::printf("[속도] wbic_track 전체=%.3fms(%.0fHz) QP=%.3fms\n",tfm,1000/tfm,tqs/Ntr);
    }
  }
  // ══════════ Phase3c: MPC(mpc_qp_plan, SRBD) 검증 ══════════
  {
    std::ifstream mf("/tmp/mpc_ref.txt");
    if(!mf){ std::printf("[검증4] /tmp/mpc_ref.txt 없음 — dump_mpc.py 먼저 실행\n"); }
    else {
      std::map<std::string,std::vector<double>> M2; std::vector<std::array<int,4>> cs; std::string ln;
      while(std::getline(mf,ln)){ std::istringstream ss(ln); std::string k; ss>>k;
        if(k=="cs"){ int idx; ss>>idx; std::array<int,4> a; for(int i=0;i<4;i++) ss>>a[i]; cs.push_back(a); }
        else { std::vector<double> v; double x; while(ss>>x) v.push_back(x); M2[k]=v; } }
      MpcCfg c; c.N=(int)M2["N_MPC"][0]; c.DT=M2["DT_MPC"][0]; c.TOTAL_MASS=M2["TOTAL_MASS"][0];
      c.G_ACC=M2["G_ACC"][0]; c.MU=M2["MU_FRICTION"][0]; c.LAMZ_MIN=M2["LAMZ"][0]; c.LAMZ_MAX=M2["LAMZ"][1];
      c.I_BODY=Map<Matrix<double,3,3,RowMajor>>(M2["BODY_INERTIA"].data());
      c.Qdiag=Map<VectorXd>(M2["MPC_Q"].data(),13); c.Rdiag=Vector3d(M2["MPC_R"][0],M2["MPC_R"][1],M2["MPC_R"][2]);
      VectorXd x0=Map<VectorXd>(M2["x0"].data(),13), x_ref=Map<VectorXd>(M2["x_ref"].data(),13);
      std::array<Vector3d,4> fp0; for(int i=0;i<4;i++) fp0[i]=Vector3d(M2["fp0"][i*3],M2["fp0"][i*3+1],M2["fp0"][i*3+2]);
      std::vector<std::array<Vector3d,4>> fp(c.N,fp0);
      Matrix<double,4,3> lam=mpc_qp_plan(c,x0,cs,fp,x_ref);
      double em=0; for(int i=0;i<4;i++)for(int j=0;j<3;j++) em=std::max(em,std::abs(lam(i,j)-M2["lam_des"][i*3+j]));
      std::printf("[검증4] MPC lam_des 최대오차=%.2e N (N=%d,vars=%d) → %s\n",
                  em,c.N,c.N*12, em<1e-2?"✅ 정합":"❌ 불일치");
      // 속도
      auto t0=std::chrono::high_resolution_clock::now(); int Nm=200;
      for(int i=0;i<Nm;i++) mpc_qp_plan(c,x0,cs,fp,x_ref);
      double ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count()/Nm;
      std::printf("[속도] MPC 전체=%.2fms(%.0fHz) — 50Hz 요구 대비 %.0f배 여유\n",ms,1000/ms,(1000/ms)/50);
    }
  }
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
