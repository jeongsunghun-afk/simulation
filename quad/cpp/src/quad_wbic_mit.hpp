// 4족 WBC 보조 — MIT Mini Cheetah 구성 요소 (Kim et al. 2019 · Cheetah-Software KinWBC.cpp / WBIC.cpp) 이식.
//   ★2026-09-30: 4족 기본 = 하이브리드(WBIC_MIT=2): 토크는 기존 가중 QP(wbic_track) + J̇q̇, 드라이버 목표는 kinwbc().
//     solve()(MIT 식 WBIC 전체, WBIC_MIT=1, 기본 MIT 게인)는 드라이버 PD(플랜트)가 있으면 실기조건 전 시나리오 통과,
//     토크만이면 걷기 낙상 → 선택 모드. (09-30 "2+6ms 붕괴"는 A 게인 조건이었음 — 2026-10-06 정정)
//   ① KinWBC: 접촉 영공간 안에서 과제 우선순위 순으로 Δq·q̇ 를 쌓아 **계획 관절 위치/속도(q_des, q̇_des)** 를 만든다.
//   ② WBIC  : 동역학적으로 일관된 역행렬로 q̈_cmd 를 쌓는다. 접촉 = J_c q̈ + J̇_c q̇ = 0 (a_c=0, **K_D 없음**),
//             과제 = J_t q̈ = ẍ_cmd − J̇_t q̇.
//   ③ 작은 QP: z=[δ_b(6); δ_f(3Nc)] — 몸통 가속 보정과 MPC 반력 보정만. 등식=몸통 6행 동역학, 부등식=마찰추·fz·드라이브 토크한계.
//   출력 τ(관절) + q_des·q̇_des(관절). 드라이버는 τ_ff + kp(q_des−q) + kd(q̇_des−q̇) 로 돈다(MIT _UpdateLegCMD).
#pragma once
#include <eiquadprog/eiquadprog-fast.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
using namespace Eigen;

namespace quadmit {   // biped_wbic_mit.hpp 의 4족판 — 발목 전단 없음(관절공간 토크한계)

struct Task {
  MatrixXd J;          // m×nv
  VectorXd e;          // m  위치 오차(목표−현재) — KinWBC
  VectorXd xd_des;     // m  목표 과제속도 — KinWBC
  VectorXd xdd_cmd;    // m  가속 명령(ff + PD) — WBIC
  VectorXd JdotQdot;   // m  J̇ q̇
};

struct In {
  int nv, nu;
  MatrixXd M; VectorXd h, qv;     // nv×nv, nv, nv
  VectorXd qj;                    // nu 현재 관절각
  MatrixXd Jc; VectorXd JcDotQdot;// 3Nc×nv, 3Nc
  VectorXd Fr_des;                // 3Nc  MPC 반력
  std::vector<Task> tasks;        // 우선순위 순
  double mu=0.5, fz_min=1.0, fz_max=400.0;
  VectorXd drv_peak;              // nu 드라이브 토크한계
  double W_b=0.1, W_f=1.0;        // MIT 기본 (_W_floating 0.1 · _W_rf 1.0)
};

struct Out { VectorXd tau, qdes, dqdes, qdd, z; bool qp_ok=false; };

inline MatrixXd pinv(const MatrixXd& A, double tol=1e-4){
  JacobiSVD<MatrixXd> svd(A, ComputeThinU|ComputeThinV);
  VectorXd s=svd.singularValues(); VectorXd si=VectorXd::Zero(s.size());
  const double thr = tol * (s.size()? s[0] : 1.0);
  for(int i=0;i<s.size();i++) if(s[i]>thr) si[i]=1.0/s[i];
  return svd.matrixV()*si.asDiagonal()*svd.matrixU().transpose();
}
// 동역학적으로 일관된 역: J̄ = A⁻¹ Jᵀ (J A⁻¹ Jᵀ)⁺
inline MatrixXd dcinv(const MatrixXd& J, const MatrixXd& Ainv){
  return Ainv*J.transpose()*pinv(J*Ainv*J.transpose());
}

// ① KinWBC (MIT KinWBC::FindConfiguration) — 일반 유사역행렬, 접촉 영공간에서 시작해 과제 우선순위로 Δq·q̇ 누적.
inline void kinwbc(const In& in, VectorXd& qdes, VectorXd& dqdes){
  const int nv=in.nv, nu=in.nu; const MatrixXd I=MatrixXd::Identity(nv,nv);
  MatrixXd Npre = I - pinv(in.Jc)*in.Jc;
  VectorXd dq=VectorXd::Zero(nv), qd=VectorXd::Zero(nv);
  for(const Task& t: in.tasks){
    MatrixXd JtPre=t.J*Npre, JtBar=pinv(JtPre);
    dq += JtBar*(t.e      - t.J*dq);
    qd += JtBar*(t.xd_des - t.J*qd);
    Npre = Npre*(I - JtBar*JtPre);
  }
  qdes = in.qj + dq.tail(nu); dqdes = qd.tail(nu);
}

inline Out solve(const In& in){
  const int nv=in.nv, nu=in.nu, nc=(int)in.Jc.rows();
  const MatrixXd I=MatrixXd::Identity(nv,nv);
  Out o; o.tau=VectorXd::Zero(nu); o.qdes=in.qj; o.dqdes=VectorXd::Zero(nu);

  kinwbc(in, o.qdes, o.dqdes);

  // ② WBIC — 동역학 일관 영공간 투영.  접촉: q̈_pre = J̄_c(−J̇_c q̇)  (a_c = 0, K_D 없음)
  const MatrixXd Ainv = in.M.llt().solve(I);
  MatrixXd JcBar = dcinv(in.Jc, Ainv);
  VectorXd qdd = JcBar*(-in.JcDotQdot);
  MatrixXd Npre = I - JcBar*in.Jc;
  for(const Task& t: in.tasks){
    MatrixXd JtPre=t.J*Npre, JtBar=dcinv(JtPre, Ainv);
    qdd += JtBar*(t.xdd_cmd - t.JdotQdot - t.J*qdd);
    Npre = Npre*(I - JtBar*JtPre);
  }

  // ③ QP: z=[δ_b; δ_f]
  const int nz=6+nc;
  MatrixXd P=MatrixXd::Zero(nz,nz); VectorXd g=VectorXd::Zero(nz);
  P.topLeftCorner(6,6)=in.W_b*MatrixXd::Identity(6,6);
  P.bottomRightCorner(nc,nc)=in.W_f*MatrixXd::Identity(nc,nc);
  P += 1e-9*MatrixXd::Identity(nz,nz);
  // 등식: M_b(q̈_cmd+[δ_b;0]) + h_b = J_cbᵀ(F_des+δ_f)   →  CE z + ce0 = 0
  MatrixXd CE(6,nz); CE.leftCols(6)=in.M.block(0,0,6,6); CE.rightCols(nc)=-in.Jc.leftCols(6).transpose();
  VectorXd ce0 = in.M.topRows(6)*qdd + in.h.head(6) - in.Jc.leftCols(6).transpose()*in.Fr_des;
  // 토크: τ = τ0 + T z
  MatrixXd T(nu,nz); T.leftCols(6)=in.M.block(6,0,nu,6); T.rightCols(nc)=-in.Jc.rightCols(nu).transpose();
  VectorXd tau0 = in.M.bottomRows(nu)*qdd + in.h.tail(nu) - in.Jc.rightCols(nu).transpose()*in.Fr_des;
  // 부등식 CI z + ci0 ≥ 0
  std::vector<VectorXd> R; std::vector<double> c0;
  for(int k=0;k<nc/3;k++){ const int o3=6+3*k; const Vector3d F=in.Fr_des.segment(3*k,3);
    auto row=[&](double ax,double ay,double az,double off){ VectorXd r=VectorXd::Zero(nz); r[o3]=ax; r[o3+1]=ay; r[o3+2]=az; R.push_back(r); c0.push_back(off); };
    row(0,0, 1, F.z()-in.fz_min);                       // fz ≥ fz_min
    row(0,0,-1, in.fz_max-F.z());                       // fz ≤ fz_max
    row(-1,0,in.mu, in.mu*F.z()-F.x()); row( 1,0,in.mu, in.mu*F.z()+F.x());   // |fx| ≤ μ fz
    row(0,-1,in.mu, in.mu*F.z()-F.y()); row(0, 1,in.mu, in.mu*F.z()+F.y());   // |fy| ≤ μ fz
  }
  // 드라이브 토크한계 (발목 링키지: u_calf = τ_calf − τ_foot)
  for(int i=0;i<nu;i++){ VectorXd r=T.row(i); double off=tau0[i];
    R.push_back(-r); c0.push_back(in.drv_peak[i]-off);     // u ≤ peak
    R.push_back( r); c0.push_back(in.drv_peak[i]+off);     // u ≥ −peak
  }
  const int nci=(int)R.size(); MatrixXd CI(nci,nz); VectorXd ci0(nci);
  for(int i=0;i<nci;i++){ CI.row(i)=R[i]; ci0[i]=c0[i]; }
  VectorXd z=VectorXd::Zero(nz);
  eiquadprog::solvers::EiquadprogFast qp; qp.reset(nz,6,nci);
  auto st=qp.solve_quadprog(P,g,CE,ce0,CI,ci0,z);
  o.qp_ok = (st==eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL);
  if(!o.qp_ok) z.setZero();                                // 실패: 보정 없이 q̈_cmd·F_des 그대로
  o.qdd=qdd; o.z=z;
  VectorXd tau = tau0 + T*z;
  for(int i=0;i<nu;i++) tau[i]=std::max(-in.drv_peak[i],std::min(in.drv_peak[i],tau[i]));
  o.tau=tau;
  return o;
}

} // namespace bipedmit
