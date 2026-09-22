// biped WBIC — biped_mpc_wbic.wbic_track 이식(단일 전신 QP). MuJoCo 유래량은 인자로 받음(파리티용).
// 변수 z=[q̈(nv); λ(3·Kc)]. eiquadprog(성숙 quad_control.hpp 변환 규약: CE=A·ce0=−b, CI=−G·ci0=h).
#pragma once
#include <eiquadprog/eiquadprog-fast.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
using namespace Eigen;

namespace bipedwbic {

// ★관절토크 τ ↔ 드라이브토크 u (2026-08-13). 발목이 링키지 구동이라 두 좌표가 다르다.
//     u_calf = τ_calf − τ_foot        u_foot = τ_foot        (hip·thigh 는 그대로)
//   발목 드라이브 좌표가 raw각(q_calf+q_foot)이므로 일률보존에서 **전치**로 들어간다.
//   축순서 = (hip,thigh,calf,foot) × 다리  ⇒  i%4==2 가 calf, 바로 다음이 foot.
//   ⚠MJCF 에서 발목 액추에이터가 tendon(coef 1,1)에 물려 있어야 이 규약이 성립한다.
//   ⚠실기 변환 emb/interface/joint_map.py:tau_ctrl_to_ch 와 **같은 전단**이다.
//     둘이 갈리면 시뮬과 실기가 다른 로봇이 된다 — 한쪽만 고치지 말 것.
inline VectorXd tau_to_drive(const VectorXd& tau){
  VectorXd u=tau;
  for(int i=0;i+1<tau.size();i++) if(i%4==2) u[i]=tau[i]-tau[i+1];
  return u;
}
inline VectorXd drive_to_tau(const VectorXd& u){
  VectorXd tau=u;
  for(int i=0;i+1<u.size();i++) if(i%4==2) tau[i]=u[i]+u[i+1];
  return tau;
}

struct WbicIn {
  int nv, nu, Kc;                 // 14, 8, 접촉수
  MatrixXd M;                     // nv×nv
  VectorXd h, qv;                 // nv
  VectorXd q;                     // nu (관절 qpos)
  Vector4d qc;                    // base quat(wxyz)
  Vector3d com; double zref;      // subtree_com, com_ref[2]
  MatrixXd Jc;                    // 3×nv (CoM jac)
  std::vector<int> contacts;      // stance leg 인덱스
  std::vector<MatrixXd> cjac;     // Kc개 3×nv
  std::vector<Vector3d> lam;      // Kc개 (contacts[k]에 대응 MPC GRF)
  bool has_swing; int swing_leg;
  MatrixXd Jsw;                   // 3×nv (swing jac)
  Vector3d sw_pos, sw_ptgt, sw_vtgt;
  bool has_sw_ori=false;          // ★평발 swing 발 수평 유지
  MatrixXd Jsw_rot;               // 3×nv (swing 회전 jac)
  Vector3d sw_oerr;               // swing 발 방향오차(현재-목표수평)
  bool com_x_track=false; double com_x_ref=0, com_vx_ref=0;   // ★평발 보행 전후 CoM 규제(발목ZMP 활용)
  double lean=0;   // ★본체 forward pitch lean(rad): 뒤로 발라당 상쇄(제어된 전방낙하). base 목표자세 앞으로 기울임
  bool cop_reg=false; double cop_comx=0; std::vector<double> cop_cx; double W_COP=200;  // ★CoP를 CoM 밑에(Σλz(cx−comx)=0)=단일지지 발목ZMP(후방토플 방지)
  bool com_xy_track=false; double com_xr=0,com_yr=0,com_vxr=0,com_vyr=0; double W_COMXY=140;  // ★ZMP프리뷰 CoM xy 추종
  VectorXd Qhome;                 // nu
  VectorXd drv_peak;              // nu — ★**드라이브(모터)** 토크한계다. 관절토크 한계가 아니다
  std::vector<int> ankle_idx;     // 발목 관절
  // 게인
  double SW_KP, SW_KD, W_ORI, W_ANKLE, W_POST, W_LAM, STANCE_KD, MU_EFF, LAMZ_MIN;
  double ANK_KP=60, ANK_KD=5;     // ★점발 발목 posture PD (2026-09-21 whip 튜닝). 기본=종전 60/5(ζ0.32). 파리티/미설정 콜러는 이 기본값.
};

inline VectorXd wbic_track(const WbicIn& in){
  int nv=in.nv, nu=in.nu, Kc=in.Kc, nz=nv+3*Kc;
  auto sl=[&](int k){ return nv+3*k; };
  MatrixXd P=MatrixXd::Zero(nz,nz); VectorXd g=VectorXd::Zero(nz);
  std::vector<int> sw_vidx;
  // 스윙 추종
  if(in.has_swing){
    const MatrixXd& J=in.Jsw;
    Vector3d accel=in.SW_KP*(in.sw_ptgt-in.sw_pos)+in.SW_KD*(in.sw_vtgt-J*in.qv);
    P.topLeftCorner(nv,nv)+=90.0*(J.transpose()*J); g.head(nv)-=90.0*(J.transpose()*accel);
    for(int t=0;t<4;t++) sw_vidx.push_back(6+in.swing_leg*4+t);
    if(in.has_sw_ori){                    // ★평발 swing 발 수평 유지(16cm 발 기울어 착지 교란 억제)
      Vector3d a_rot=120*(-in.sw_oerr)-15*(in.Jsw_rot*in.qv);
      P.topLeftCorner(nv,nv)+=15.0*(in.Jsw_rot.transpose()*in.Jsw_rot); g.head(nv)-=15.0*(in.Jsw_rot.transpose()*a_rot);
    }
  }
  // 자세 레벨링(현재 yaw)
  const Vector4d& qc=in.qc;
  double yaw=std::atan2(2*(qc[0]*qc[3]+qc[1]*qc[2]),1-2*(qc[2]*qc[2]+qc[3]*qc[3]));
  // ★목표자세 = yaw ⊗ pitch(lean): roll=0·pitch=lean(전방기울임)·yaw유지. lean=0이면 기존 수평.
  double cy=std::cos(yaw/2), sy=std::sin(yaw/2), cp=std::cos(in.lean/2), sp=std::sin(in.lean/2);
  double qlev[4]={cy*cp, -sy*sp, cy*sp, sy*cp};
  // subQuat(qc, qlev): oerr = 2*(qlev^-1 * qc)_xyz  (mju_subQuat)
  double ql_conj[4]={qlev[0],-qlev[1],-qlev[2],-qlev[3]};
  double dq[4]={ql_conj[0]*qc[0]-ql_conj[1]*qc[1]-ql_conj[2]*qc[2]-ql_conj[3]*qc[3],
                ql_conj[0]*qc[1]+ql_conj[1]*qc[0]+ql_conj[2]*qc[3]-ql_conj[3]*qc[2],
                ql_conj[0]*qc[2]-ql_conj[1]*qc[3]+ql_conj[2]*qc[0]+ql_conj[3]*qc[1],
                ql_conj[0]*qc[3]+ql_conj[1]*qc[2]-ql_conj[2]*qc[1]+ql_conj[3]*qc[0]};
  Vector3d oerr; { double s=(dq[0]<0?-1:1); Vector3d v(dq[1],dq[2],dq[3]); double n=v.norm();
    oerr = (n<1e-12)? Vector3d(0,0,0) : (2.0*std::atan2(n, std::abs(dq[0])) * s / n) * v; }
  for(int j=0;j<2;j++){ double a=150*(-oerr[j])-20*in.qv[3+j]; P(3+j,3+j)+=in.W_ORI; g[3+j]-=in.W_ORI*a; }
  double a_yaw=150*(-oerr[2])-20*in.qv[5]; P(5,5)+=1.0; g[5]-=1.0*a_yaw;
  // CoM 높이
  Vector3d Jcqv=in.Jc*in.qv; double a_z=300*(in.zref-in.com[2])-30*Jcqv[2];
  P.topLeftCorner(nv,nv)+=400.0*(in.Jc.row(2).transpose()*in.Jc.row(2));
  g.head(nv)-=400.0*a_z*in.Jc.row(2).transpose();
  if(in.com_x_track){            // ★평발 보행 전후 CoM 규제(밑창 발목ZMP로 CoM 속도를 명령에 유지, CoP 앞섬/과속 방지)
    double a_cx=60*(in.com_x_ref-in.com[0])+50*(in.com_vx_ref-Jcqv[0]);   // 속도항↑(과속 제동)
    P.topLeftCorner(nv,nv)+=140.0*(in.Jc.row(0).transpose()*in.Jc.row(0));
    g.head(nv)-=140.0*a_cx*in.Jc.row(0).transpose();
  }
  if(in.com_xy_track){           // ★ZMP 프리뷰 CoM xy 궤적 추종(계획된 CoM ref를 WBIC가 따라감)
    double a_cx=90*(in.com_xr-in.com[0])+30*(in.com_vxr-Jcqv[0]);
    double a_cy=90*(in.com_yr-in.com[1])+30*(in.com_vyr-Jcqv[1]);
    P.topLeftCorner(nv,nv)+=in.W_COMXY*(in.Jc.row(0).transpose()*in.Jc.row(0)+in.Jc.row(1).transpose()*in.Jc.row(1));
    g.head(nv)-=in.W_COMXY*(a_cx*in.Jc.row(0).transpose()+a_cy*in.Jc.row(1).transpose());
  }
  // ★CoP 조절(단일지지 발목ZMP): Σ λk_z·(cx_k−com_x)=0 → CoP를 CoM 밑에(toe로 쏠려 후방토플 방지). soft.
  if(in.cop_reg && (int)in.cop_cx.size()==Kc){
    for(int k=0;k<Kc;k++){ double ak=in.cop_cx[k]-in.cop_comx;
      for(int j=0;j<Kc;j++){ double aj=in.cop_cx[j]-in.cop_comx;
        P(sl(k)+2,sl(j)+2)+=in.W_COP*ak*aj; } }
  }
  // posture
  auto is_ankle=[&](int j){ for(int a:in.ankle_idx) if(a==j) return true; return false; };
  auto is_sw=[&](int vi){ for(int v:sw_vidx) if(v==vi) return true; return false; };
  for(int j=0;j<nu;j++){ bool ank=is_ankle(j);
    double kp_p=ank?in.ANK_KP:60.0, kd_p=ank?in.ANK_KD:5.0;   // ★발목만 튜닝(env)·나머지=종전 60/5
    double a=kp_p*(in.Qhome[j]-in.q[j])-kd_p*in.qv[6+j];
    double w = ank?in.W_ANKLE : (is_sw(6+j)?5.0:in.W_POST);
    P(6+j,6+j)+=w; g[6+j]-=w*a; }
  P.topLeftCorner(nv,nv)+=1e-3*MatrixXd::Identity(nv,nv);
  // MPC GRF 추종
  for(int k=0;k<Kc;k++){ P.block(sl(k),sl(k),3,3)+=in.W_LAM*Matrix3d::Identity();
    g.segment(sl(k),3)-=in.W_LAM*in.lam[k]; }
  // 등식: base6 + 접촉(STANCE_KD)
  int neq=6+3*Kc; MatrixXd A=MatrixXd::Zero(neq,nz); VectorXd bb=VectorXd::Zero(neq);
  A.block(0,0,6,nv)=in.M.topRows(6); bb.head(6)=-in.h.head(6);
  for(int k=0;k<Kc;k++){ A.block(0,sl(k),6,3)=-in.cjac[k].leftCols(6).transpose();
    A.block(6+3*k,0,3,nv)=in.cjac[k]; bb.segment(6+3*k,3)=-in.STANCE_KD*(in.cjac[k]*in.qv); }
  // 부등식 Gx≤h: 마찰추 + λz≥min + 토크한계
  std::vector<VectorXd> Gr; std::vector<double> hv;
  int sgn[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
  for(int k=0;k<Kc;k++){ int o=sl(k);
    for(int s=0;s<4;s++){ VectorXd r=VectorXd::Zero(nz); r[o]=sgn[s][0]; r[o+1]=sgn[s][1]; r[o+2]=-in.MU_EFF; Gr.push_back(r); hv.push_back(0.0); }
    VectorXd r=VectorXd::Zero(nz); r[o+2]=-1; Gr.push_back(r); hv.push_back(-in.LAMZ_MIN); }
  MatrixXd Tm=MatrixXd::Zero(nu,nz); Tm.leftCols(nv)=in.M.block(6,0,nu,nv);
  for(int k=0;k<Kc;k++) Tm.block(0,sl(k),nu,3)=-in.cjac[k].block(0,6,3,nu).transpose();
  // ★토크한계를 **드라이브 공간**에 건다 (2026-08-13). 실기 한계는 관절이 아니라 모터에 걸린다:
  //   무릎 드라이브가 τ_calf−τ_foot 을 짊어지므로, 관절공간 박스는 실기가 못 내는 조합을
  //   허용한다(집합이 박스가 아니라 전단된 평행사변형이다). calf 행만 전단하고 나머지는 그대로.
  for(int i=0;i<nu;i++){
    VectorXd r=Tm.row(i); double off=in.h[6+i];
    if(i%4==2 && i+1<nu){ r-=Tm.row(i+1); off-=in.h[6+i+1]; }
    Gr.push_back(r);  hv.push_back(in.drv_peak[i]-off);
    Gr.push_back(-r); hv.push_back(in.drv_peak[i]+off);
  }
  // eiquadprog: CE x+ce0=0 · CI x+ci0≥0
  //   WBIC_REG: Hessian 대각 가산(기본 1e-8). ⚠**등식 축퇴에는 안 듣는다** — 아래 프루닝이 그 몫.
  static const double WREG = getenv("WBIC_REG") ? atof(getenv("WBIC_REG")) : 1e-8;
  P=(0.5*(P+P.transpose())).eval()+WREG*MatrixXd::Identity(nz,nz);
  MatrixXd CE=A; VectorXd ce0=-bb;
  // ★★등식 제약 프루닝 (2026-08-28 실기에서 확정) ─────────────────────────────
  //   실기 2점 평발 stand 에서 QP 실패율 20~100%, 사유코드 **4 = REDUNDANT_EQUALITIES**.
  //   원인은 수치가 아니라 **구조**다: 한 발에 접촉점이 둘인데 발은 강체라, 두 점의
  //   가속도 구속 6행 중 **두 점을 잇는 선 방향 1행이 종속**이다(강체는 그 거리를 못 바꾼다).
  //   발이 둘이니 정확히 2행이 남는다 → eiquadprog 는 이걸 못 넘기고 통째로 포기하고,
  //   그때마다 중력보상 폴백(τ=h)으로 떨어져 **두 해가 번갈아 나가는 것이 떨림**이었다.
  //   (sim 은 이 경우 solver 를 proxqp 로 갈아타 우회한다 — biped_mpc_wbic.py:243)
  //   ⇒ rank-revealing QR 로 **독립 행만 남긴다.** 버리는 행은 남은 행들의 선형결합이므로
  //     해집합이 바뀌지 않는다(구조적 종속이라 일관성도 보장된다).
  //   끄려면 WBIC_EQ_PRUNE=0.
  // ★기본 OFF 로 되돌림 (2026-08-28) — 실기 미검증. 켜려면 WBIC_EQ_PRUNE=1.
  static const bool EQ_PRUNE = getenv("WBIC_EQ_PRUNE") && atoi(getenv("WBIC_EQ_PRUNE"));
  if(EQ_PRUNE && CE.rows() > 0){
    Eigen::ColPivHouseholderQR<MatrixXd> qr(CE.transpose());
    qr.setThreshold(1e-9);
    const int r = (int)qr.rank();
    { static int last_r=-1, last_n=-1;
      if(r!=last_r || (int)CE.rows()!=last_n){ last_r=r; last_n=(int)CE.rows();
        std::fprintf(stderr,"[wbic] 등식 rank: %d/%ld %s\n", r, (long)CE.rows(),
                     r<CE.rows() ? "→ **프루닝 발동**" : "(축퇴 없음)"); } }
    if(r < CE.rows()){
      const auto& perm = qr.colsPermutation().indices();   // CE^T 의 열 = CE 의 행
      std::vector<int> keep(perm.data(), perm.data()+r);
      std::sort(keep.begin(), keep.end());
      MatrixXd CE2(r, CE.cols()); VectorXd ce02(r);
      for(int i=0;i<r;i++){ CE2.row(i)=CE.row(keep[i]); ce02[i]=ce0[keep[i]]; }
      CE = CE2; ce0 = ce02; neq = r;
    }
  }
  int nci=(int)Gr.size(); MatrixXd CI(nci,nz); VectorXd ci0(nci);
  for(int i=0;i<nci;i++){ CI.row(i)=-Gr[i]; ci0[i]=hv[i]; }
  VectorXd x(nz);
  eiquadprog::solvers::EiquadprogFast qp; qp.reset(nz,neq,nci);
  auto st=qp.solve_quadprog(P,g,CE,ce0,CI,ci0,x);
  VectorXd tau=VectorXd::Zero(nu);
  if(st==eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL){
    VectorXd qdd=x.head(nv); tau=in.M.block(6,0,nu,nv)*qdd+in.h.segment(6,nu);
    for(int k=0;k<Kc;k++) tau-=in.cjac[k].block(0,6,3,nu).transpose()*x.segment(sl(k),3);
    // ★클립도 드라이브 공간에서. 관절공간 클립은 실기가 못 내는 토크를 통과시킨다.
    VectorXd u=tau_to_drive(tau);
    for(int i=0;i<nu;i++) u[i]=std::max(-in.drv_peak[i],std::min(in.drv_peak[i],u[i]));
    tau=drive_to_tau(u);
  }
  return tau;
}

} // namespace bipedwbic
