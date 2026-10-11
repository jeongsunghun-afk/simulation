// tamols_qp.hpp — TAMOLS SQP-RTI 솔버 (eiquadprog QP)
//   정식화(제약5+비용3, Drake 검증)를 결정벡터 z로 묶어 SQP로 풂.
//   비용=가중 비선형최소제곱 → Gauss-Newton(H=2JᵀJ, g=2JᵀR). 제약=선형화(FD Jacobian).
//   QP: min ½Δzᵀ H Δz + gᵀΔz  s.t.  CE·Δz+ce0=0, CI·Δz+ci0≥0  (eiquadprog 규약).
//   ★1차 목표=정확성(FD Jacobian). 실시간(해석 Jacobian)은 수렴검증 후.
#pragma once
#include "costs.hpp"          // bilinear_height, 비용 항, constraints.hpp(R_B·det3) 포함
#include <eiquadprog/eiquadprog-fast.hpp>
#include <functional>

namespace tamols {

// ── 문제 옵션(Drake tamols_02leg 제약과 정합) ──
struct QpOptions {
  double zlo = 0.45, zhi = 0.60, rp_max = 0.20, yaw_max = 0.15;  // base bounds
  int    base_nsamp = 4;
  double x_target = 0.73;                                        // terminal 전진
  double y_min = 0.10, y_max = 0.22;                             // foot_y 대칭
  bool   gap = true; double gap_lo = 0.45, gap_hi = 0.65, gap_margin = 0.03;
  int    max_iter = 50; double tol = 1e-7; double reg = 1.0;     // SQP 시작 reg(적응형 LM, trust-region)
  bool   verbose = false;
  // ★fix_p: 발판 p 를 결정변수에서 사실상 제거한다(Δz 의 p 블록을 0 으로 고정).
  //   지각층이 발판을 이미 안다는 아키텍처에서 "몸통만 푸는 TO" 의 실제 비용을 재기 위한 것.
  //   p 에 붙은 지형/도달/foot_y 제약은 p 가 돌 위에 고정되면 만족 상태가 되어
  //   선형화 QP 의 infeasibility(→ elastic 폴백, 차원 ~1345)가 사라지는지가 관심사.
  bool   fix_p = false;
};

// ── 결정벡터 packing: z = [a[0..P-1] (각 6×4, 열우선) | p(사이클 0..N-1, 각 4×3) | epsilon(P)] ──
//   N=1 이면 레이아웃·크기 전부 기존과 동일(바이트 동일).
struct Packer {
  int P, NC, nz, na;
  Packer(int P_, int NC_ = 1) : P(P_), NC(NC_) { na = 24 * P; nz = na + 12 * NC + P; }
  VectorXd pack(const TamolsState& st) const {
    VectorXd z(nz); int o = 0;
    for (int k = 0; k < P; ++k) for (int i = 0; i < 4; ++i) for (int d = 0; d < 6; ++d) z(o++) = st.a[k](d, i);
    for (int n = 0; n < NC; ++n) for (int L = 0; L < 4; ++L) for (int c = 0; c < 3; ++c)
      z(o++) = (n == 0) ? st.p(L, c) : st.p_ext[n - 1](L, c);
    for (int k = 0; k < P; ++k) z(o++) = (k < st.epsilon.size()) ? st.epsilon(k) : 0.0;
    return z;
  }
  void unpack(const VectorXd& z, TamolsState& st) const {
    int o = 0;
    if ((int)st.a.size() != P) st.a.assign(P, MatrixXd(6, 4));
    for (int k = 0; k < P; ++k) for (int i = 0; i < 4; ++i) for (int d = 0; d < 6; ++d) st.a[k](d, i) = z(o++);
    if ((int)st.p_ext.size() != NC - 1) st.p_ext.assign(NC - 1, Eigen::Matrix<double,4,3>::Zero());
    for (int n = 0; n < NC; ++n) for (int L = 0; L < 4; ++L) for (int c = 0; c < 3; ++c)
      { if (n == 0) st.p(L, c) = z(o++); else st.p_ext[n - 1](L, c) = z(o++); }
    st.epsilon.resize(P); for (int k = 0; k < P; ++k) st.epsilon(k) = z(o++);
  }
};

// ── 비용 최소제곱 잔차 R(z): cost = |R|² = Σ w(...)²,  r_j = √w·(...) ──
inline VectorXd cost_residuals(const TamolsState& st, const Grid& h, double cell, int map_size) {
  std::vector<double> r;
  const int S = 6;
  // tracking: √(2Tk/S)·(vel_x−ref_x)
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    for (int s = 0; s < S; ++s) {
      double tau = Tk * s / (double)S, w = 2.0 * Tk / S;
      r.push_back(std::sqrt(w) * (st.vel_at(k, tau)(0) - st.ref_vel(0)));
    }
  }
  // foothold_on_ground: √100·(h(p_i)−p_i.z)   ★N 사이클 전부
  for (int n = 0; n < st.ncyc(); ++n) for (int i = 0; i < 4; ++i) {
    Vector3d pi = st.fpos(n, i);
    r.push_back(10.0 * (bilinear_height(h, cell, map_size, pi(0), pi(1)) - pi(2)));
  }
  // nominal_kinematic: √20·(p_B+R_B·hip−l_des−p_i)[c], **각 사이클 마지막 phase** τ=T/2
  {
    static double _wnom = getenv("TAM_WNOM") ? atof(getenv("TAM_WNOM")) : 20.0;   // ★명목 발위치(hip 아래) 비용 가중. ↑=발판을 hip 폭으로 벌림(뭉침 방지)
    Vector3d l_des(0, 0, st.prm.h_des);
    for (int n = 0; n < st.ncyc(); ++n) {
      int k = st.cyc_end_phase(n); double tau = st.gait[k].duration / 2.0;
      Vector6d pose = st.pos_at(k, tau); Vector3d pB = pose.head<3>();
      Eigen::Matrix3d R = R_B(pose.tail<3>());
      for (int i = 0; i < 4; ++i) {
        Vector3d e = pB + R * st.prm.hip_offsets.row(i).transpose() - l_des - st.foot_at(k, i);
        for (int c = 0; c < 3; ++c) r.push_back(std::sqrt(_wnom) * e(c));
      }
    }
  }
  // ★GIAC_FIX: eps(GIAC slack) 비용 페널티 → 위반을 slack에 흘리는 대신 base/발판을 움직이게(soft=feasible 유지, 하드바운드 near-infeasible 회피)
  if (giac_fix_on()) { double we = getenv("W_EPS") ? atof(getenv("W_EPS")) : 50.0;
    for (int k = 0; k < st.num_phases(); ++k) r.push_back(std::sqrt(we) * ((k < st.epsilon.size()) ? st.epsilon(k) : 0.0)); }
  // ★edge_avoidance(게이트: edge_layers()==nullptr 이면 0행 = 기존과 바이트 동일)
  edge_residuals(st, r);
  // ★CoM-centering(walk 정적안정): base (x,)y가 각 phase 지지발 centroid를 따라가게 = CoM을 지지삼각형 안으로.
  //   walk 크롤이 CoM을 한쪽으로 0.32m 쏠리게 하던 근본(GIAC slack이 느슨해 방치) 해결. COM_W>0 시 활성(FD jacobian 필요).
  if (getenv("COM_W")) { double cw = atof(getenv("COM_W"));
    bool cx = getenv("COM_WX") != nullptr;   // x도 centering(기본 y만=전진 방해 안 함)
    for (int k = 0; k < st.num_phases(); ++k) {
      double yc = 0, xc = 0; int nst = 0;
      for (int i = 0; i < 4; ++i) if (st.gait[k].contact[i]) { xc += st.p_meas(i, 0); yc += st.p_meas(i, 1); nst++; }
      if (nst > 0) { yc /= nst; xc /= nst;
        double _lead = getenv("COM_LEAD") ? atof(getenv("COM_LEAD")) : 0.0;   // ★리드(위상분수): 0=mid-phase, 0.5=phase-start(리프트 시점에 centroid 도달=선행). CoM sway가 리프트를 앞서게.
        double tau = st.gait[k].duration * std::max(0.0, 0.5 - _lead); Vector6d pk = st.pos_at(k, tau);
        r.push_back(std::sqrt(cw) * (pk(1) - yc));
        if (cx) r.push_back(std::sqrt(cw) * (pk(0) - xc)); } } }
  return Eigen::Map<VectorXd>(r.data(), r.size());
}

// ── 등식 제약 c(z)=0: 초기(12) + 위상연속(12·(P−1)) ──
inline VectorXd eq_constraints(const TamolsState& st) {
  std::vector<double> c;
  for (int d = 0; d < 6; ++d) { c.push_back(st.a[0](d, 0) - st.base_pose(d)); c.push_back(st.a[0](d, 1) - st.base_vel(d)); }
  for (int k = 0; k + 1 < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration; Vector6d pe = st.pos_at(k, Tk), ve = st.vel_at(k, Tk);
    for (int d = 0; d < 6; ++d) { c.push_back(pe(d) - st.a[k + 1](d, 0)); c.push_back(ve(d) - st.a[k + 1](d, 1)); }
  }
  return Eigen::Map<VectorXd>(c.data(), c.size());
}

// ── 부등식 제약 g(z)≥0: friction·kinematic·GIAC·base bounds·foot_y·gap·terminal·eps ──
inline VectorXd ineq_constraints(const TamolsState& st, const QpOptions& o) {
  std::vector<double> g;
  const int S = 6;
  const double lo = st.prm.l_min * st.prm.l_min, hi = st.prm.l_max * st.prm.l_max;
  const double mu = st.prm.mu, m = st.prm.mass;
  const Vector3d ez(0, 0, 1), gvec(0, 0, -9.81);
  // ★GIAC_FIX(2026-08-03): GIAC를 실제로 묶기 — ①aB→(aB−gvec) 중력 포함(정적 CoM-지지 테스트=지배항) ②eps 상한(자유 slack이 제약을 무력화하던 것 차단). off=기존(오프라인/DTC 무영향)
  bool _giacfix=giac_fix_on(); bool _giacord=giac_order_on();
  const bool _stnc = stance_reach_on();   // ★스탠스(구 발판) reach 게이트 — OFF면 0행 추가(바이트 동일)
  double _epsmax=getenv("EPS_MAX")?atof(getenv("EPS_MAX")):0.0;  // 0=소프트페널티만(하드상한 off)
  // ★GIAC_NORM=<L>(기본 0=OFF, 바이트 동일) — GIAC 잔차를 특성 스케일로 나눠 **eps 를 무차원 O(1)** 로.
  //   왜 필요한가(실측): 17b 잔차는 `m·det3(pij, pB−pi, aG)` = N·m² 로 O(10) 이라, 비용 `W_EPS·eps²`(50)
  //   가 tracking/foothold/nominal(합 O(0.1~3))을 압도할 수 있다. 실제로 **2발 지지(trot)** 에서
  //   eps 점유율 평균 55.7%·최대 99.5% 로 목적함수를 삼킨다(지지가 선분으로 퇴화해 GIAC 가 사실상
  //   만족 불가 → 위반을 전부 eps 로 흘린다). **3발 지지(walk)** 에서는 평균 3.8% 로 무해하다.
  //   ※ 단위 불일치도 같이 고친다: 17b/17c 는 질량인자가 있고(N·m²) **17d 는 없다**(m³/s²) —
  //     같은 eps 를 ×m(=15) 다른 눈금의 행이 공유하고 있었다. 각 행을 제 특성 스케일로 나눈다.
  //   정규화 후 의미: eps ≈ (CoM 이 지지다각형 밖으로 나간 거리)/L — 즉 **무차원 마진**이라
  //   EPS_MAX 를 물리 단위에서 유도할 수 있다(예: d≤2cm, L=0.284 ⇒ EPS_MAX≈0.07).
  double _gnormL = getenv("GIAC_NORM") ? atof(getenv("GIAC_NORM")) : 0.0;
  const double _gs_m = (_gnormL > 0) ? (m * 9.81 * _gnormL * _gnormL) : 1.0;   // 17b/17c (질량 포함)
  const double _gs_0 = (_gnormL > 0) ? (    9.81 * _gnormL * _gnormL) : 1.0;   // 17d    (질량 없음)
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    // friction: az(τ)+9.81 ≥ 0
    for (int s = 0; s < S; ++s) { double tau = Tk * s / (double)S; g.push_back(st.acc_at(k, tau)(2) + 9.81); }
    // kinematic reach (at_des = 이미 디딘 발판, 다중 사이클이면 그 사이클 발판): |diff|²−lo ≥ 0, hi−|diff|² ≥ 0
    for (int L = 0; L < 4; ++L) if (st.fsrc(k, L) >= 0)
      for (int s = 0; s < S; ++s) {
        double tau = Tk * s / (double)S; Vector6d pose = st.pos_at(k, tau);
        Vector3d diff = pose.head<3>() + R_B(pose.tail<3>()) * st.prm.hip_offsets.row(L).transpose() - st.foot_at(k, L);
        double t = diff.dot(diff); g.push_back(t - lo); g.push_back(hi - t);
      }
    // ★스탠스 reach(STANCE_REACH): 구 발판 p_meas 위 스탠스 다리도 l_min²≤|·|²≤l_max² (τ 샘플 s=1..S)
    //   p_meas 는 파라미터(결정변수 아님) → 이 행들의 Jacobian 은 base 스플라인 a[k] 열만 채워진다.
    if (_stnc) for (int L = 0; L < 4; ++L) if (is_stance_meas(st, k, L))
      for (int s = 1; s <= S; ++s) {
        double tau = Tk * s / (double)S; Vector6d pose = st.pos_at(k, tau);
        Vector3d diff = pose.head<3>() + R_B(pose.tail<3>()) * st.prm.hip_offsets.row(L).transpose() - st.p_meas.row(L).transpose();
        double t = diff.dot(diff); g.push_back(t - lo); g.push_back(hi - t);
      }
    // GIAC(Eq17)
    std::vector<int> stance; for (int i = 0; i < 4; ++i) if (st.gait[k].contact[i]) stance.push_back(i);
    int N = (int)stance.size(); double eps = (k < st.epsilon.size()) ? st.epsilon(k) : 0.0;
    g.push_back(eps);                                                   // eps ≥ 0
    if (_giacfix && _epsmax > 0) g.push_back(_epsmax - eps);            // ★eps ≤ epsmax(옵션 하드상한, EPS_MAX=0이면 끔=비용페널티만)
    auto foot = [&](int i) { return st.foot_at(k, i); };
    auto _prs = giac_pairs(stance, foot, _giacord);
    for (int s = 0; s < S; ++s) {
      double tau = Tk * s / (double)S;
      Vector3d pB = st.pos_at(k, tau).head<3>(), aB = st.acc_at(k, tau).head<3>(), Ld = st.Ldot_at(k, tau);
      Vector3d aG = _giacfix ? (aB - gvec) : aB;                        // ★gravito-inertial 가속(aB+(0,0,9.81)): 정적 CoM-지지 테스트가 지배
      // ★정적walk(COM_W, 2026-08-04): 17a 마찰콘 z-예산만 gravito-inertial(aB_z+9.81)=GIAC 본래정의.
      //   기존 aG(2)=aB_z≈0(수평정지)이라 (μ·0)²≥aB_x²+aB_y² → 수평 base 가속을 0으로 하드강제 → CoM sway(±0.047m 필요=~9m/s²) 불가능=COM_W 무력.
      //   z-예산을 aB_z+9.81로 하면 수평 예산 0→~5.9m/s². 17b/c/d는 aG 유지(eps 폭주 방지). off=byte-identical(A/DTC 불변).
      double az17a = getenv("COM_W") ? (aB(2) + 9.81) : aG(2);
      if (N > 0) g.push_back((mu * az17a) * (mu * az17a) - aG(0) * aG(0) - aG(1) * aG(1));   // 17a ≥0
      if (N >= 3) for (const auto& pr : _prs) {                                              // ★CW 인접쌍(GIAC_ORDER) 또는 레거시 전쌍
        Vector3d pi = foot(pr.first), pj = foot(pr.second), pij = pj - pi;
        g.push_back(eps - (m * det3(pij, pB - pi, aG) - pij.dot(Ld)) / _gs_m);                // 17b: lhs≤eps
      }
      if (N == 2) {
        Vector3d pi = foot(stance[0]), pj = foot(stance[1]), pij = pj - pi;
        double val = (m * det3(pij, pB - pi, aG) - pij.dot(Ld)) / _gs_m;
        g.push_back(eps - val); g.push_back(eps + val);                                       // 17c |val|≤eps
        Vector3d Mi = (pB - pi).cross(gvec - aB) - Ld / m;
        g.push_back(eps + det3(ez, pij, Mi) / _gs_0);                                         // 17d (질량인자 없음)
      }
    }
    // base bounds (nsamp, τ>0): z·roll·pitch·yaw
    // ★BASE_YBND(2026-08-04): base y(lateral) 위치 제한 — GIAC binding이 sway를 낼 때 x/y 무경계라 폭주(−1.09m)하던 것 방지.
    //   sway는 지지삼각형 centroid(±~0.05m)까지만 필요 → |y|≤ybnd로 묶으면 GIAC가 그 안에서 clean sway. 기본 off(오프라인/DTC 무영향).
    double _ybnd = getenv("BASE_YBND") ? atof(getenv("BASE_YBND")) : 0.0;
    for (int j = 1; j <= o.base_nsamp; ++j) {
      double tau = Tk * j / (double)o.base_nsamp; Vector6d s = st.pos_at(k, tau);
      g.push_back(s(2) - o.zlo); g.push_back(o.zhi - s(2));
      g.push_back(s(3) + o.rp_max); g.push_back(o.rp_max - s(3));
      g.push_back(s(4) + o.rp_max); g.push_back(o.rp_max - s(4));
      g.push_back(s(5) + o.yaw_max); g.push_back(o.yaw_max - s(5));
      if (_ybnd > 0) { g.push_back(s(1) + _ybnd); g.push_back(_ybnd - s(1)); }   // |y|≤ybnd
    }
  }
  // foot_y 대칭: 좌(0,2) y∈[ymin,ymax], 우(1,3) y∈[−ymax,−ymin]   ★N 사이클 전부
  for (int n = 0; n < st.ncyc(); ++n) {
    for (int L : {0, 2}) { double y = st.fpos(n, L)(1); g.push_back(y - o.y_min); g.push_back(o.y_max - y); }
    for (int R : {1, 3}) { double y = st.fpos(n, R)(1); g.push_back(-o.y_min - y); g.push_back(y + o.y_max); }
  }
  // gap 회피: 앞(0,1) x≥gap_hi+m, 뒤(2,3) x≤gap_lo−m  (사이클 0 에만 — 갭은 로컬 1사이클 구속)
  if (o.gap) {
    for (int F : {0, 1}) g.push_back(st.p(F, 0) - (o.gap_hi + o.gap_margin));
    for (int H : {2, 3}) g.push_back((o.gap_lo - o.gap_margin) - st.p(H, 0));
  }
  // terminal 전진: 마지막 phase 끝 x ≥ x_target
  { int k = st.num_phases() - 1; g.push_back(st.pos_at(k, st.gait[k].duration)(0) - o.x_target); }
  // ★지지 유효성 하드 제약(support_layers()==nullptr 이면 0행 = 기존과 바이트 동일)
  //   band: h(p)−p.z ≤ band 와 −(h(p)−p.z) ≤ band  /  sdf: sdf(p_xy) ≤ −margin
  if (const SupportLayers* S = support_layers()) {
    const bool bon = S->band_on && S->h, son = S->sdf_on && S->sdf;
    for (int n = 0; n < st.ncyc(); ++n) for (int L = 0; L < 4; ++L) {
      Vector3d fp = st.fpos(n, L);
      if (bon) { double e = bilinear_height(*S->h, S->cell, S->map_size, fp(0), fp(1)) - fp(2);
                 g.push_back(S->band - e); g.push_back(S->band + e); }
      if (son) { double d = bilinear_height(*S->sdf, S->cell, S->map_size, fp(0), fp(1));
                 g.push_back(-d - S->margin); }
    }
  }
  return Eigen::Map<VectorXd>(g.data(), g.size());
}

// ── 벡터함수 f(z)의 FD Jacobian (중앙차분) ──
inline MatrixXd fd_jacobian(const std::function<VectorXd(const VectorXd&)>& f, const VectorXd& z, double eps = 1e-6) {
  VectorXd f0 = f(z); int m = (int)f0.size(), n = (int)z.size();
  MatrixXd J(m, n); VectorXd zp = z;
  for (int j = 0; j < n; ++j) {
    double h = eps * std::max(1.0, std::fabs(z(j)));
    zp(j) = z(j) + h; VectorXd fp = f(zp);
    zp(j) = z(j) - h; VectorXd fm = f(zp);
    zp(j) = z(j); J.col(j) = (fp - fm) / (2 * h);
  }
  return J;
}

// ── SQP 결과 ──
struct QpResult { bool ok; int iters; double cost; double eq_viol; double ineq_viol; };

// ── ℓ1 위반량 : Σ|eq| + Σ max(0,−ineq) ──
inline double l1_viol(const VectorXd& E, const VectorXd& G) {
  double v = E.cwiseAbs().sum();
  for (int i = 0; i < G.size(); ++i) v += std::max(0.0, -G(i));
  return v;
}

// ── SQP-RTI 솔버: st(초기추정)→수렴. ℓ1 merit 백트래킹 라인서치(비선형 제약서 발산 방지) ──
inline QpResult solve(TamolsState& st, const Grid& h, double cell, int map_size, const QpOptions& o = QpOptions()) {
  Packer pk(st.num_phases(), st.ncyc());
  VectorXd z = pk.pack(st);
  auto Rf = [&](const VectorXd& zz){ TamolsState s = st; pk.unpack(zz, s); return cost_residuals(s, h, cell, map_size); };
  auto Ef = [&](const VectorXd& zz){ TamolsState s = st; pk.unpack(zz, s); return eq_constraints(s); };
  auto Gf = [&](const VectorXd& zz){ TamolsState s = st; pk.unpack(zz, s); return ineq_constraints(s, o); };
  const double mu = 1e3;   // ℓ1 페널티 계수(위반 지배 → feasibility 우선). merit = cost + μ·viol
  auto merit = [&](const VectorXd& zz){ return Rf(zz).squaredNorm() + mu * l1_viol(Ef(zz), Gf(zz)); };

  eiquadprog::solvers::EiquadprogFast qp;
  QpResult res{false, 0, 0, 0, 0};
  double reg = o.reg;                       // 적응형 trust-region 정칙화(LM)
  const int nz = pk.nz;
  for (int it = 0; it < o.max_iter; ++it) {
    VectorXd R = Rf(z), E = Ef(z), G = Gf(z);
    MatrixXd JR = fd_jacobian(Rf, z), JE = fd_jacobian(Ef, z), JG = fd_jacobian(Gf, z);
    int neq = (int)E.size(), nineq = (int)G.size();
    MatrixXd JRtJR = 2.0 * JR.transpose() * JR; VectorXd gg = 2.0 * JR.transpose() * R;
    MatrixXd CE = JE, CI = JG; VectorXd ce0 = E, ci0 = G;                    // JE·Δz+E=0, JG·Δz+G≥0
    if (o.fix_p) {                       // ★Δz_p = 0 (p 를 초기값에 고정)
      const int npin = 12 * pk.NC, na_ = pk.na;
      MatrixXd CE2(neq + npin, nz); CE2.setZero();
      if (neq > 0) CE2.topRows(neq) = CE;
      for (int i = 0; i < npin; ++i) CE2(neq + i, na_ + i) = 1.0;
      VectorXd ce02(neq + npin); ce02.setZero();
      if (neq > 0) ce02.head(neq) = ce0;
      CE = CE2; ce0 = ce02; neq += npin;
    }
    double m0 = merit(z), stepnorm = 0, alpha_used = 0;
    // LM 내부: 큰 α의 merit 감소 스텝 찾을 때까지 reg↑(trust region 축소). α작음=스텝불량 → reg↑
    bool stepped = false; VectorXd best_dz; double best_alpha = 0;
    for (int tr = 0; tr < 20; ++tr) {
      MatrixXd H = JRtJR; H.diagonal().array() += reg;
      VectorXd dz(nz); qp.reset(nz, neq, nineq);
      auto stt = qp.solve_quadprog(H, gg, CE, ce0, CI, ci0, dz);
      if (stt == eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) {
        double alpha = 1.0; int ls;
        for (ls = 0; ls < 25; ++ls) { if (merit(z + alpha * dz) < m0 - 1e-10) break; alpha *= 0.5; }
        if (ls < 25) {                                                       // 감소 스텝 존재 → 폴백 기록
          if (alpha > best_alpha) { best_alpha = alpha; best_dz = alpha * dz; }
          if (alpha >= 0.25) {                                              // 충분히 큰 스텝 → 채택
            z += best_dz; stepnorm = best_dz.norm(); alpha_used = alpha; stepped = true;
            if (alpha >= 0.5) reg = std::max(1e-4, reg * 0.5);              // 좋은 스텝 → trust region 확대
            break;
          }
        }
      }
      reg = std::min(1e6, reg * 4.0);                                        // α작음/QP실패 → trust region 축소
    }
    if (!stepped && best_alpha > 0) { z += best_dz; stepnorm = best_dz.norm(); alpha_used = best_alpha; stepped = true; }
    res.iters = it + 1;
    if (o.verbose) {
      VectorXd E2 = Ef(z), G2 = Gf(z); double iv = 0; for (int i = 0; i < G2.size(); ++i) iv = std::max(iv, std::max(0.0, -G2(i)));
      std::printf("   it%2d  α=%.3f reg=%.1e  cost=%.5f  eq=%.2e  ineq=%.2e  |dz|=%.2e\n",
                  it, alpha_used, reg, Rf(z).squaredNorm(), E2.cwiseAbs().maxCoeff(), iv, stepnorm);
    }
    if (!stepped) break;                                                     // 진척 불가
    if (stepnorm < o.tol) { res.ok = true; break; }
  }
  pk.unpack(z, st);
  VectorXd R = cost_residuals(st, h, cell, map_size), E = eq_constraints(st), G = ineq_constraints(st, o);
  res.cost = R.squaredNorm();
  res.eq_viol = E.size() ? E.cwiseAbs().maxCoeff() : 0;
  res.ineq_viol = 0; for (int i = 0; i < G.size(); ++i) res.ineq_viol = std::max(res.ineq_viol, std::max(0.0, -G(i)));
  if (res.eq_viol < 1e-5 && res.ineq_viol < 1e-5) res.ok = true;
  return res;
}

} // namespace tamols
