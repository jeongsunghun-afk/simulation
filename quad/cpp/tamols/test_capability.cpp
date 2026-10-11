// test_capability.cpp — TAMOLS "네 가지 능력" 오프라인 기전 검증 (2026-09)
//   ① GIAC(균형인지 발판) ② base 궤적 동시계획 ③ 다중스텝 예측 ④ 돌 선택
//   대조군 = RL env의 기하 스냅(Raibert 명목 → 최근접 돌 중심 → 돌 안 클램프 half−0.025).
//   사전등록 판정식은 각 test 함수 헤더에 코드 실행 전에 명시(결과 보고 기준 바꾸지 않음).
//
//   빌드: PIX=/home/jsh/simple-mpc/.pixi/envs/default
//     g++ -O2 -std=c++17 test_capability.cpp -I/usr/include/eigen3 -I$PIX/include \
//         -L$PIX/lib -Wl,-rpath,$PIX/lib -leiquadprog -o test_capability
//   실행: ./test_capability {jac|t1|t2|t3|t4}
#include "tamols_online.hpp"
#include "terrain_proc.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <chrono>
using namespace tamols;

// ─────────────────────────── 돌 필드 ───────────────────────────
struct Stone { double cx, cy, hx, hy, top; const char* tag; };
static const double STONE_TOP = 0.15, VOID_Z = 0.0, BASE_H = 0.34;
static const double HIPX = 0.1934, HIPY = 0.142;          // Go2
static const double T_STANCE = 0.20, FOOT_R = 0.025;      // env 기하스냅 파라미터
static const double L_MAX = 0.42, L_MIN = 0.10;           // Go2 다리 최대신장(≈thigh0.213+calf0.213−α)
static const char* LEGN[4] = {"FL","FR","RL","RR"};

struct Field {
  std::vector<Stone> s;
  double h(double x, double y, int* idx = nullptr) const {         // 정확 지형(래스터 아님)
    if (idx) *idx = -1;
    double best = VOID_Z;
    for (size_t i = 0; i < s.size(); ++i)
      if (std::fabs(x - s[i].cx) <= s[i].hx && std::fabs(y - s[i].cy) <= s[i].hy)
        if (s[i].top >= best) { best = s[i].top; if (idx) *idx = (int)i; }
    return best;
  }
  int nearest(double x, double y) const {                          // env _snap_xy_to_stone: 최근접 중심
    int b = -1; double bd = 1e18;
    for (size_t i = 0; i < s.size(); ++i) {
      double d = (x - s[i].cx) * (x - s[i].cx) + (y - s[i].cy) * (y - s[i].cy);
      if (d < bd) { bd = d; b = (int)i; }
    }
    return b;
  }
  // env 스냅: 최근접 돌 중심 → 그 돌 반경(half − 0.025) 안으로 클램프
  void snap(double x, double y, double& sx, double& sy, double& sz, int& si) const {
    si = nearest(x, y);
    if (si < 0) { sx = x; sy = y; sz = VOID_Z; return; }
    const Stone& S = s[si];
    double ex = std::max(0.0, S.hx - FOOT_R), ey = std::max(0.0, S.hy - FOOT_R);
    sx = std::min(std::max(x, S.cx - ex), S.cx + ex);
    sy = std::min(std::max(y, S.cy - ey), S.cy + ey);
    sz = S.top;
  }
};

// 로컬 창(중심 (xc,0)) 래스터
static void raster(const Field& f, double xc, int N, double cell, Grid& h) {
  double off = cell * N / 2.0;
  h.resize(N, N);
  for (int a = 0; a < N; ++a) for (int b = 0; b < N; ++b)
    h(a, b) = f.h(xc + a * cell - off, b * cell - off);
}

// ─────────────────── walk 게이트(at_des 누적) — cache_gen_go2_stepping 정합 ───────────────────
//   스윙 순서 RR→FR→RL→FL. at_des는 스윙 시점부터 누적(터미널 nominal이 4발 전부 구동).
static void set_walk_acc(TamolsState& st, double dur, int P = 4) {
  st.gait.resize(P);
  int cs[4][4] = {{1,1,1,0},{1,0,1,1},{1,1,0,1},{0,1,1,1}};
  int ad[4][4] = {{0,0,0,1},{0,1,0,1},{0,1,1,1},{1,1,1,1}};
  for (int k = 0; k < P; ++k) { st.gait[k].duration = dur;
    for (int i = 0; i < 4; ++i) { st.gait[k].contact[i] = cs[k][i]; st.gait[k].at_des[i] = ad[k][i]; } }
}
static const int SWING_PHASE[4] = {3, 1, 2, 0};   // leg → 그 다리가 스윙하는 phase

// ★N 사이클 walk 게이트(4N phase). foot_idx[L] = 그 phase 에서 다리 L 이 딛고 있는 발판의 사이클
//   (= 마지막으로 스윙을 마친 사이클, −1 = 아직 한 번도 안 디딤 → p_meas).
//   N=1 이면 set_walk_acc 와 **완전히 같은 스케줄**(at_des 누적)이 된다.
static void set_walk_multi(TamolsState& st, double dur, int N) {
  int P = 4 * N; st.gait.resize(P);
  int cs[4][4] = {{1,1,1,0},{1,0,1,1},{1,1,0,1},{0,1,1,1}};
  st.cyc_end.assign(N, 0);
  for (int n = 0; n < N; ++n) st.cyc_end[n] = 4 * n + 3;
  for (int k = 0; k < P; ++k) {
    int n = k / 4, kk = k % 4; st.gait[k].duration = dur;
    for (int L = 0; L < 4; ++L) {
      st.gait[k].contact[L] = cs[kk][L];
      int src = (kk >= SWING_PHASE[L]) ? n : n - 1;      // 마지막으로 스윙 완료한 사이클
      st.gait[k].foot_idx[L] = src;
      st.gait[k].at_des[L]   = (src >= 0) ? 1 : 0;
    }
  }
  st.p_ext.assign(N - 1, Eigen::Matrix<double,4,3>::Zero());
}

struct Cfg { double vadv = 0.30, dur = 0.20; int iters = 80;
             double y_min = 0.02, y_max = 0.40, l_max = L_MAX;
             int nph = 4; };          // ★호라이즌 phase 수(절단 = 관측점을 terminal 로 올릴 때)

// Hermite 전진 램프 cold init (online_replan / cache_gen 정합)
static void cold_init(TamolsState& st, double z0, double vx0, const Cfg& c) {
  int P = st.num_phases(); double T = P * c.dur, xf = c.vadv * T;
  st.a.assign(P, MatrixXd::Zero(6, 4));
  double c1 = vx0, c3 = (c.vadv - vx0 - 2 * (xf - vx0 * T) / T) / (T * T), c2 = (xf - vx0 * T - c3 * T * T * T) / (T * T);
  auto xg = [&](double t) { return c1 * t + c2 * t * t + c3 * t * t * t; };
  auto vg = [&](double t) { return c1 + 2 * c2 * t + 3 * c3 * t * t; };
  for (int k = 0; k < P; ++k) {
    st.a[k].col(0) = st.base_pose; st.a[k](2, 0) = z0;
    double t0 = k * c.dur, x0 = xg(t0), x1 = xg(t0 + c.dur), v0 = vg(t0), v1 = vg(t0 + c.dur);
    st.a[k](0, 0) = x0; st.a[k](0, 1) = v0;
    st.a[k](0, 2) = (3 * (x1 - x0) / c.dur - 2 * v0 - v1) / c.dur;
    st.a[k](0, 3) = (2 * (x0 - x1) / c.dur + v0 + v1) / (c.dur * c.dur);
  }
}

static QpResult solve_cycle(TamolsState& st, const Grid& hsol, double cell, int N, double z0, double vx0,
                            const Eigen::Matrix<double,4,3>& fm, const Cfg& c,
                            const Eigen::Matrix<double,4,3>* p_init, const double* xtgt = nullptr) {
  set_walk_acc(st, c.dur, c.nph);
  int P = st.num_phases(); double T = P * c.dur, xf = c.vadv * T;
  st.base_pose << 0, 0, z0, 0, 0, 0;
  st.base_vel  << vx0, 0, 0, 0, 0, 0;
  st.p_meas = fm;
  st.ref_vel = Vector3d(c.vadv, 0, 0);
  cold_init(st, z0, vx0, c);
  if (p_init) st.p = *p_init;
  else for (int i = 0; i < 4; ++i) {
    st.p(i,0) = st.prm.hip_offsets(i,0) + 0.5 * xf; st.p(i,1) = st.prm.hip_offsets(i,1);
    st.p(i,2) = bilinear_height(hsol, cell, N, st.p(i,0), st.p(i,1));
  }
  st.epsilon = VectorXd::Zero(P);
  QpOptions o; o.max_iter = c.iters;
  o.zlo = z0 - 0.06; o.zhi = z0 + 0.06; o.rp_max = 0.20; o.yaw_max = 0.10;
  o.x_target = xtgt ? *xtgt : xf * 0.9; o.y_min = c.y_min; o.y_max = c.y_max; o.gap = false;
  return solve_fast(st, hsol, cell, N, o);
}

// ─────────────────── 진단: 지지 폴리곤 마진 / reach ───────────────────
//   margin = base xy 에서 지지다각형 경계까지 부호거리(+ 안쪽). CW 인접변 기준.
static double poly_margin(const std::vector<Eigen::Vector2d>& v, const Eigen::Vector2d& P) {
  int n = (int)v.size(); if (n < 3) return -1e9;
  Eigen::Vector2d c(0, 0); for (auto& q : v) c += q; c /= n;
  std::vector<std::pair<double,int>> ang;
  for (int i = 0; i < n; ++i) ang.emplace_back(std::atan2(v[i](1) - c(1), v[i](0) - c(0)), i);
  std::sort(ang.begin(), ang.end(), [](auto& a, auto& b) { return a.first > b.first; });   // CW
  double m = 1e18;
  for (int k = 0; k < n; ++k) {
    const Eigen::Vector2d& A = v[ang[k].second]; const Eigen::Vector2d& B = v[ang[(k+1)%n].second];
    Eigen::Vector2d u = B - A, d = P - A;
    double cr = u(0) * d(1) - u(1) * d(0);          // ≤0 = 안쪽(CW)
    m = std::min(m, -cr / std::max(1e-12, u.norm()));
  }
  return m;
}
// phase·τ 전체에서 최소 지지마진 (지지발 = at_des ? p : p_meas)
static double min_support_margin(const TamolsState& st, int* wk = nullptr) {
  double worst = 1e18; const int S = 6;
  for (int k = 0; k < st.num_phases(); ++k) {
    std::vector<Eigen::Vector2d> v;
    for (int i = 0; i < 4; ++i) if (st.gait[k].contact[i]) {
      Vector3d f = st.gait[k].at_des[i] ? Vector3d(st.p.row(i).transpose()) : Vector3d(st.p_meas.row(i).transpose());
      v.emplace_back(f(0), f(1));
    }
    for (int s = 0; s < S; ++s) {
      double tau = st.gait[k].duration * s / (double)S;
      Vector6d b = st.pos_at(k, tau);
      double m = poly_margin(v, Eigen::Vector2d(b(0), b(1)));
      if (m < worst) { worst = m; if (wk) *wk = k; }
    }
  }
  return worst;
}
// 마지막 phase(4발 모두 신규 발판=at_des)만의 최소 지지마진 — 신규 발판이 실제로 지배하는 구간
static double last_phase_margin(const TamolsState& st) {
  int k = st.num_phases() - 1; const int S = 6; double worst = 1e18;
  std::vector<Eigen::Vector2d> v;
  for (int i = 0; i < 4; ++i) if (st.gait[k].contact[i]) {
    Vector3d f = st.gait[k].at_des[i] ? Vector3d(st.p.row(i).transpose()) : Vector3d(st.p_meas.row(i).transpose());
    v.emplace_back(f(0), f(1)); }
  for (int q = 0; q <= S; ++q) { double tau = st.gait[k].duration * q / (double)S;
    Vector6d b = st.pos_at(k, tau); worst = std::min(worst, poly_margin(v, Eigen::Vector2d(b(0), b(1)))); }
  return worst;
}
// 다리 L의 터치다운(스윙 phase 끝) 시점 hip→발 거리
static double touchdown_reach(const TamolsState& st, int L, const Eigen::Matrix<double,4,3>& p) {
  int k = std::min(SWING_PHASE[L], st.num_phases() - 1);
  Vector6d pose = st.pos_at(k, st.gait[k].duration);
  Vector3d d = pose.head<3>() + R_B(pose.tail<3>()) * st.prm.hip_offsets.row(L).transpose() - p.row(L).transpose();
  return d.norm();
}
static double max_giac_viol(const TamolsState& st) {   // eps 무시한 순수 GIAC 위반(=필요 eps)
  TamolsState s = st; s.epsilon.setZero();
  return giac_residual(s);
}

// ─────────────────── 기하 스냅 대조군 ───────────────────
struct SnapOut { Eigen::Matrix<double,4,3> p; int stone[4]; double nomx[4], nomy[4]; };
//   nominal = base_xy + hip + 0.5·T_stance·v_cmd (+ lookahead·pitch)  → 최근접 돌 클램프
//   ★env 충실: 각 다리의 명목점은 **그 다리가 착지하는 시점의 base**에서 계산(env 는 매 스텝 live base 사용).
static SnapOut geo_snap(const Field& f, const TamolsState& nom, double xb, double vcmd, double look) {
  SnapOut o;
  for (int L = 0; L < 4; ++L) {
    int k = std::min(SWING_PHASE[L], nom.num_phases() - 1);
    Vector6d bp = nom.pos_at(k, nom.gait[k].duration);
    double base_x = xb + bp(0), base_y = bp(1);
    double hx = (L < 2 ? HIPX : -HIPX), hy = ((L == 0 || L == 2) ? HIPY : -HIPY);
    double nx = base_x + hx + 0.5 * T_STANCE * vcmd + look, ny = base_y + hy;
    double sx, sy, sz; int si; f.snap(nx, ny, sx, sy, sz, si);
    o.p(L,0) = sx; o.p(L,1) = sy; o.p(L,2) = sz; o.stone[L] = si; o.nomx[L] = nx; o.nomy[L] = ny;
  }
  return o;
}

// 명목 base(=cold init, y≡0) 상태를 만든다(대조군 평가용)
static TamolsState nominal_state(const Params& prm, double z0, double vx0, const Cfg& c,
                                 const Eigen::Matrix<double,4,3>& fm, const Eigen::Matrix<double,4,3>& p) {
  TamolsState st; st.prm = prm;
  set_walk_acc(st, c.dur, c.nph);
  st.base_pose << 0, 0, z0, 0, 0, 0; st.base_vel << vx0, 0, 0, 0, 0, 0;
  st.p_meas = fm; st.p = p; st.ref_vel = Vector3d(c.vadv, 0, 0);
  cold_init(st, z0, vx0, c);
  st.epsilon = VectorXd::Zero(st.num_phases());
  return st;
}

// 명목 base 스플라인만 필요한 상태(cold_init 은 p 를 쓰지 않음 → p 아무거나)
static TamolsState nominal_spline(const Params& prm, double z0, const Cfg& c,
                                  const Eigen::Matrix<double,4,3>& fm) {
  return nominal_state(prm, z0, 0.0, c, fm, fm);
}

static Params go2_params(double lmax) {
  Params prm;
  prm.hip_offsets << HIPX, HIPY, 0,  HIPX, -HIPY, 0,  -HIPX, HIPY, 0,  -HIPX, -HIPY, 0;
  prm.mass = 15.0; prm.h_des = BASE_H; prm.nominal_height = BASE_H;
  prm.foot_radius = 0.022; prm.l_min = L_MIN; prm.l_max = lmax; prm.mu = 0.6;
  return prm;
}


// ═══════════════════════ 공통 시나리오 빌더 ═══════════════════════
// ★TERRAIN_HARD 게이트(기본 OFF) — 지지 유효성 하드 제약 파라미터
static bool  terrain_hard_on() { const char* v = getenv("TERRAIN_HARD"); return v && v[0] != '0'; }
static double th_dz()     { return getenv("TERRAIN_DZ")     ? atof(getenv("TERRAIN_DZ"))     : 0.06; }  // 국소 작업면 아래 허용 하강 [m]
static double th_rad()    { return getenv("TERRAIN_R")      ? atof(getenv("TERRAIN_R"))      : 0.25; }  // closing 반경(=채울 구덩이 최대 반폭) [m]
static double th_band()   { return getenv("TERRAIN_BAND")   ? atof(getenv("TERRAIN_BAND"))   : 0.01; }  // |h(p)−p.z| 허용 [m]
static double th_margin() { return getenv("TERRAIN_MARGIN") ? atof(getenv("TERRAIN_MARGIN")) : FOOT_R; }  // 유효영역 안쪽 여유 [m] = 발 반경(접촉면이 지지영역 안이어야)
// 하드 지형 제약이 볼 맵: 기본 = **원본 고도맵 hraw**.
//   hsol(=gaussian σ1) 은 비용/기울기 shaping 용이지 지형 사실이 아니다 — 돌 가장자리에 **유령 램프**
//   (실제 h=0 인데 hsol≈0.07)를 만들어 "그 위에 서도 된다"는 잘못된 허가를 낸다(실측 확인).
//   하드 실현가능성 판정은 고도맵 자체로. TERRAIN_SMOOTH=1 이면 hsol 사용(대조용).
static bool th_smooth() { const char* v = getenv("TERRAIN_SMOOTH"); return v && v[0] != '0'; }

struct Scene {
  Field f; Grid hraw, hsol; TerrainLayers L; EdgeLayers E;
  SupportLayer SL; SupportLayers S;      // ★지지 유효성 층 + 게이트 구조체
  int N = 121; double cell = 0.02;      // 창 ±1.21 m
  double sigma_cells = getenv("TAMCAP_SIGMA") ? atof(getenv("TAMCAP_SIGMA")) : 1.0;   // ★solve map = TAMOLS h_s1 (σ1=1.0 셀, Drake process_height_maps 기본)
  void build(double xc) {
    raster(f, xc, N, cell, hraw);
    hsol = sigma_cells > 0 ? gaussian_filter(hraw, sigma_cells) : hraw;
    L = process_height_maps(hraw, cell, 1.0, 2.0);
    E.gh_x = &L.gh_x; E.gh_y = &L.gh_y; E.gs1_x = &L.gs1_x; E.gs1_y = &L.gs1_y;
    E.cell = cell; E.map_size = N;
    if (terrain_hard_on()) {                       // ★지지 유효성 = 고도맵(hraw) 기준
      const Grid& hh = th_smooth() ? hsol : hraw;
      SL = compute_support(hh, cell, th_dz(), th_rad());
      S.h = th_smooth() ? &hsol : &hraw; S.sdf = &SL.sdf; S.cell = cell; S.map_size = N;
      S.band = th_band(); S.margin = th_margin();
    }
  }
};
// 이 Scene 의 지지층을 전역 게이트에 등록(OFF 면 nullptr = 기존과 바이트 동일)
static void use_scene(const Scene& sc) { support_layers() = terrain_hard_on() ? &sc.S : nullptr; }
// 시작 플랫폼(로봇이 서 있는 넓은 발판)
static Stone platform(double x0, double x1, double hy) {
  return Stone{0.5 * (x0 + x1), 0.0, 0.5 * (x1 - x0), hy, STONE_TOP, "plat"};
}

// ═══════════════════════ jac: 해석 자코비안 정합 ═══════════════════════
static int cmd_jac() {
  std::printf("[0단계 회귀] 해석 Jacobian vs FD — 새 항(edge/eps상한/BASE_YBND/CW쌍) 전부 ON\n");
  setenv("GIAC_FIX", "1", 1); setenv("GIAC_ORDER", "1", 1);
  setenv("EPS_MAX", "3.0", 1); setenv("BASE_YBND", "0.08", 1);
  Scene sc;
  sc.f.s.push_back(platform(-0.45, 0.30, 0.40));
  sc.f.s.push_back(Stone{0.45,  0.15, 0.07, 0.07, 0.15, "a"});
  sc.f.s.push_back(Stone{0.45, -0.15, 0.07, 0.07, 0.15, "b"});
  sc.build(0.0);
  sc.E.w = 1.0;
  Params prm = go2_params(L_MAX);
  Cfg c;
  TamolsState st; st.prm = prm;
  Eigen::Matrix<double,4,3> fm;
  for (int L2 = 0; L2 < 4; ++L2) { fm(L2,0) = prm.hip_offsets(L2,0); fm(L2,1) = prm.hip_offsets(L2,1);
                                   fm(L2,2) = sc.f.h(fm(L2,0), fm(L2,1)); }
  double z0 = BASE_H + STONE_TOP;
  set_walk_acc(st, c.dur);
  st.base_pose << 0,0,z0,0,0,0; st.base_vel << 0.2,0,0,0,0,0; st.p_meas = fm;
  st.ref_vel = Vector3d(c.vadv,0,0); cold_init(st, z0, 0.2, c);
  for (int i = 0; i < 4; ++i) { st.p(i,0)=prm.hip_offsets(i,0)+0.12; st.p(i,1)=prm.hip_offsets(i,1); st.p(i,2)=0.15; }
  st.epsilon = VectorXd::Constant(st.num_phases(), 0.4);
  QpOptions o; o.gap = false; o.y_min = 0.02; o.y_max = 0.40;

  auto check = [&](const char* name, bool edge_on) {
    edge_layers() = edge_on ? &sc.E : nullptr;
    Packer pk(st.num_phases()); VectorXd z = pk.pack(st);
    auto Rf = [&](const VectorXd& zz){ TamolsState s2 = st; pk.unpack(zz, s2); return cost_residuals(s2, sc.hsol, sc.cell, sc.N); };
    auto Gf = [&](const VectorXd& zz){ TamolsState s2 = st; pk.unpack(zz, s2); return ineq_constraints(s2, o); };
    auto Ef = [&](const VectorXd& zz){ TamolsState s2 = st; pk.unpack(zz, s2); return eq_constraints(s2); };
    MatrixXd JRa = cost_jacobian(st, sc.hsol, sc.cell, sc.N), JRf = fd_jacobian(Rf, z);
    MatrixXd JGa = ineq_jacobian_full(st, o, pk),             JGf = fd_jacobian(Gf, z);
    MatrixXd JEa = eq_jacobian(st),                           JEf = fd_jacobian(Ef, z);
    bool dimR = (JRa.rows()==JRf.rows()), dimG = (JGa.rows()==JGf.rows()), dimE = (JEa.rows()==JEf.rows());
    double eR = dimR ? (JRa-JRf).cwiseAbs().maxCoeff() : -1;
    double eG = dimG ? (JGa-JGf).cwiseAbs().maxCoeff() : -1;
    double eE = dimE ? (JEa-JEf).cwiseAbs().maxCoeff() : -1;
    std::printf("  %-14s  rows R %3d/%3d  G %3d/%3d  E %3d/%3d | max|Δ| R=%.2e G=%.2e E=%.2e  %s\n",
                name, (int)JRa.rows(), (int)JRf.rows(), (int)JGa.rows(), (int)JGf.rows(),
                (int)JEa.rows(), (int)JEf.rows(), eR, eG, eE,
                (dimR&&dimG&&dimE&&eR<1e-5&&eG<1e-4&&eE<1e-9) ? "OK" : "★");
    edge_layers() = nullptr;
    return (dimR&&dimG&&dimE&&eR<1e-5&&eG<1e-4&&eE<1e-9);
  };
  bool ok = true;
  ok &= check("edge OFF", false);
  ok &= check("edge ON",  true);
  unsetenv("EPS_MAX"); unsetenv("BASE_YBND");
  ok &= check("bnd/eps OFF", true);
  setenv("STANCE_REACH", "1", 1);            // ★스탠스(구 발판) reach 행 — 해석 Jacobian 신규
  ok &= check("stance reach", true);
  // ★지지 유효성 하드 제약(band 2행 + sdf 1행 per foothold) — 해석 Jacobian 신규
  { setenv("TERRAIN_HARD", "1", 1); sc.build(0.0); sc.E.w = 1.0;
    support_layers() = &sc.S;
    ok &= check("terrain hard", true);
    ok &= check("terrain hard+edgeOFF", false);
    support_layers() = nullptr; unsetenv("TERRAIN_HARD"); }
  unsetenv("STANCE_REACH");
  setenv("GIAC_FIX", "0", 1);
  ok &= check("GIAC_FIX=0", false);
  setenv("GIAC_FIX", "1", 1);
  // ★N=2(다중 사이클 호라이즌) — 발판 12→24 변수, 사이클별 reach/foothold/nominal/foot_y 행
  {
    setenv("EPS_MAX", "3.0", 1); setenv("BASE_YBND", "0.08", 1); setenv("STANCE_REACH", "1", 1);
    TamolsState s2; s2.prm = prm;
    set_walk_multi(s2, c.dur, 2);
    s2.base_pose << 0,0,z0,0,0,0; s2.base_vel << 0.2,0,0,0,0,0; s2.p_meas = fm;
    s2.ref_vel = Vector3d(c.vadv,0,0); cold_init(s2, z0, 0.2, c);
    for (int n = 0; n < 2; ++n) for (int i = 0; i < 4; ++i) {
      double px = prm.hip_offsets(i,0) + 0.12 + 0.24*n, py = prm.hip_offsets(i,1);
      if (n==0) { s2.p(i,0)=px; s2.p(i,1)=py; s2.p(i,2)=0.15; }
      else { s2.p_ext[0](i,0)=px; s2.p_ext[0](i,1)=py; s2.p_ext[0](i,2)=0.15; }
    }
    s2.epsilon = VectorXd::Constant(s2.num_phases(), 0.4);
    QpOptions o2; o2.gap = false; o2.y_min = 0.02; o2.y_max = 0.40;
    for (int thg = 0; thg < 2; ++thg) {
    if (thg) { setenv("TERRAIN_HARD","1",1); sc.build(0.0); sc.E.w = 1.0; support_layers() = &sc.S; }
    else     { support_layers() = nullptr; }
    edge_layers() = &sc.E;
    Packer pk2(s2.num_phases(), s2.ncyc()); VectorXd z2 = pk2.pack(s2);
    auto Rf2 = [&](const VectorXd& zz){ TamolsState t = s2; pk2.unpack(zz, t); return cost_residuals(t, sc.hsol, sc.cell, sc.N); };
    auto Gf2 = [&](const VectorXd& zz){ TamolsState t = s2; pk2.unpack(zz, t); return ineq_constraints(t, o2); };
    auto Ef2 = [&](const VectorXd& zz){ TamolsState t = s2; pk2.unpack(zz, t); return eq_constraints(t); };
    MatrixXd JRa = cost_jacobian(s2, sc.hsol, sc.cell, sc.N), JRf = fd_jacobian(Rf2, z2);
    MatrixXd JGa = ineq_jacobian_full(s2, o2, pk2),           JGf = fd_jacobian(Gf2, z2);
    MatrixXd JEa = eq_jacobian(s2),                           JEf = fd_jacobian(Ef2, z2);
    bool dim = (JRa.rows()==JRf.rows() && JGa.rows()==JGf.rows() && JEa.rows()==JEf.rows()
                && JRa.cols()==JRf.cols() && JGa.cols()==JGf.cols());
    double eR = dim ? (JRa-JRf).cwiseAbs().maxCoeff() : -1, eG = dim ? (JGa-JGf).cwiseAbs().maxCoeff() : -1,
           eE = dim ? (JEa-JEf).cwiseAbs().maxCoeff() : -1;
    bool o2ok = dim && eR<1e-5 && eG<1e-4 && eE<1e-9;
    std::printf("  %-14s nz=%d  rows R %3d/%3d  G %3d/%3d  E %3d/%3d | max|Δ| R=%.2e G=%.2e E=%.2e  %s\n",
                thg ? "★N=2+지형하드" : "★N=2 다중", pk2.nz, (int)JRa.rows(), (int)JRf.rows(), (int)JGa.rows(), (int)JGf.rows(),
                (int)JEa.rows(), (int)JEf.rows(), eR, eG, eE, o2ok ? "OK" : "★");
    ok &= o2ok;
    edge_layers() = nullptr;
    }
    support_layers() = nullptr; unsetenv("TERRAIN_HARD");
    unsetenv("EPS_MAX"); unsetenv("BASE_YBND"); unsetenv("STANCE_REACH");
  }
  std::printf("  → %s\n", ok ? "자코비안 정합 ✓" : "★ 불일치");
  return ok ? 0 : 1;
}


// ═══════════════════════ 공통 리포트 ═══════════════════════
// ═══════════════════════ 공통 리포트/평가 ═══════════════════════
static void report_legs(const char* tag, const Field& f, const TamolsState& st,
                        const Eigen::Matrix<double,4,3>& p, double xb, double lmax) {
  std::printf("   %-9s", tag);
  for (int L = 0; L < 4; ++L) {
    int si; f.h(xb + p(L,0), p(L,1), &si);
    double rr = touchdown_reach(st, L, p);
    std::printf(" %s(%+.3f,%+.3f)%-7s r=%.3f%s", LEGN[L], xb + p(L,0), p(L,1),
                si >= 0 ? f.s[si].tag : "VOID", rr, rr > lmax + 1e-3 ? "!" : " ");
  }
  std::printf("\n");
}
static int n_reach_viol(const TamolsState& st, const Eigen::Matrix<double,4,3>& p, double lmax) {
  int n = 0; for (int L = 0; L < 4; ++L) if (touchdown_reach(st, L, p) > lmax + 1e-3) ++n; return n;
}
static int n_void(const Field& f, const Eigen::Matrix<double,4,3>& p, double xb) {
  int n = 0; for (int L = 0; L < 4; ++L) { int si; f.h(xb + p(L,0), p(L,1), &si); if (si < 0) ++n; } return n;
}
static Eigen::Matrix<double,4,3> to_local(const Eigen::Matrix<double,4,3>& pa, double xb) {
  Eigen::Matrix<double,4,3> q = pa; for (int L = 0; L < 4; ++L) q(L,0) -= xb; return q;
}

static TamolsState nominal_spline(const Params& prm, double z0, const Cfg& c, const Eigen::Matrix<double,4,3>& fm);
// ═══════════ 비용 분해 · 후보 초기화 멀티스타트 ═══════════
struct Br { double total, track, foot, nom, eps, edge; };
static Br breakdown(const TamolsState& st, const Grid& h, double cell, int N) {
  Br b{};
  b.track = tracking_cost(st);
  b.foot  = foothold_on_ground_cost(st, h, cell, N);
  b.nom   = nominal_kinematic_cost(st);
  double we = getenv("W_EPS") ? atof(getenv("W_EPS")) : 50.0;
  b.eps = 0; if (giac_fix_on()) for (int k = 0; k < st.epsilon.size(); ++k) b.eps += we * st.epsilon(k) * st.epsilon(k);
  b.edge = edge_avoidance_cost(st);
  b.total = b.track + b.foot + b.nom + b.eps + b.edge;
  return b;
}
// 후보 배치: 기본 = 돌 **중심**(돌을 "고른다"의 자연스러운 의미 = 가장 안전한 점).
//   TAMCAP_CLAMP=1 이면 env 스냅과 같은 "명목점을 돌 안으로 클램프"(=가장자리로 들어감).
static void clamp_into(const Stone& S, double& x, double& y) {
  if (!getenv("TAMCAP_CLAMP")) { x = S.cx; y = S.cy; return; }
  double ex = std::max(0.0, S.hx - FOOT_R), ey = std::max(0.0, S.hy - FOOT_R);
  x = std::min(std::max(x, S.cx - ex), S.cx + ex);
  y = std::min(std::max(y, S.cy - ey), S.cy + ey);
}
// 한 후보(다리→돌 인덱스, -1=자유/명목)에서 출발해 solve. 결과 + 비용분해 반환
struct CandRes { TamolsState st; QpResult r; Br b; int nvoid; int nviol; double margin; double eps_need; };
static CandRes solve_cand(const Scene& sc, const Params& prm, const Cfg& c, double z0, double xb,
                          const Eigen::Matrix<double,4,3>& fm, const int* assign, const double* xtgt = nullptr) {
  CandRes o; o.st.prm = prm; use_scene(sc);
  Eigen::Matrix<double,4,3> pin;
  TamolsState nsp = nominal_spline(prm, z0, c, fm);
  for (int L = 0; L < 4; ++L) {
    int kk = std::min(SWING_PHASE[L], nsp.num_phases() - 1);   // ★호라이즌 절단(nph<4) 대비
    Vector6d bp = nsp.pos_at(kk, nsp.gait[kk].duration);
    double px = xb + bp(0) + prm.hip_offsets(L,0) + 0.5 * T_STANCE * c.vadv, py = bp(1) + prm.hip_offsets(L,1);
    if (assign && assign[L] >= 0) clamp_into(sc.f.s[assign[L]], px, py);
    pin(L,0) = px - xb; pin(L,1) = py; pin(L,2) = sc.f.h(px, py);
  }
  o.r = solve_cycle(o.st, sc.hsol, sc.cell, sc.N, z0, 0.0, fm, c, &pin, xtgt);
  o.b = breakdown(o.st, sc.hsol, sc.cell, sc.N);
  o.nvoid = n_void(sc.f, o.st.p, xb);
  o.nviol = n_reach_viol(o.st, o.st.p, c.l_max);
  o.margin = min_support_margin(o.st);
  o.eps_need = max_giac_viol(o.st);
  return o;
}

// ★지형 물림 진단(게이트 ON 일 때만): 발판별
//   dz_map   = p.z − h_map(p)      (|·|>band 이면 **맵 기준 공중부양**)
//   dz_exact = p.z − h_exact(p)    (해석 지형 기준. 맵 2 cm 양자화만큼 차이날 수 있음)
//   gap      = h_close − h_map     (국소 작업면 대비 하강량. >dz_max = 보이드)
//   sdf      = 지지유효 SDF        (≤ −margin 이어야 함)
static void terr_line(const char* tag, const Scene& sc, const TamolsState& st, int n) {
  if (!terrain_hard_on() || !sc.S.h) return;
  std::printf("   %-12s", tag);
  for (int L = 0; L < 4; ++L) {
    Vector3d fp = st.fpos(n, L);
    double hm = bilinear_height(*sc.S.h, sc.cell, sc.N, fp(0), fp(1));
    double gd = bilinear_height(sc.SL.gap_depth, sc.cell, sc.N, fp(0), fp(1));
    double sd = bilinear_height(sc.SL.sdf, sc.cell, sc.N, fp(0), fp(1));
    std::printf(" %s dzmap%+.3f dzex%+.3f gap%.3f sdf%+.3f |", LEGN[L], fp(2)-hm, fp(2)-sc.f.h(fp(0),fp(1)), gd, sd);
  }
  std::printf("\n");
}

static void print_cand(const char* tag, const Scene& sc, const CandRes& o, double xb, double lmax) {
  std::printf("   %-16s ok=%d ineq=%.0e | cost %6.3f = trk %.3f + foot %.3f + nom %.3f + eps %.3f + edge %.3f | void=%d viol=%d 마진=%+.4f\n",
              tag, (int)o.r.ok, o.r.ineq_viol, o.b.total, o.b.track, o.b.foot, o.b.nom, o.b.eps, o.b.edge,
              o.nvoid, o.nviol, o.margin);
  std::printf("   %-16s", "");
  for (int L = 0; L < 4; ++L) { int si; sc.f.h(xb+o.st.p(L,0), o.st.p(L,1), &si);
    std::printf(" %s(%+.3f,%+.3f,z%.3f)%-8s", LEGN[L], xb+o.st.p(L,0), o.st.p(L,1), o.st.p(L,2),
                si>=0 ? sc.f.s[si].tag : "VOID"); }
  std::printf("\n"); terr_line("", sc, o.st, 0); (void)lmax;
}

// 후보 발판을 **고정**하고 명목 base 로 목적함수를 평가(솔버 표류 없이 "목적함수가 어느 돌을 선호하나")
struct PinRes { Br b; double margin, eps_need; Eigen::Matrix<double,4,3> p; };
static PinRes pinned_eval(const Scene& sc, const Params& prm, const Cfg& c, double z0, double xb,
                          const Eigen::Matrix<double,4,3>& fm, const int* assign) {
  PinRes o;
  TamolsState nsp = nominal_spline(prm, z0, c, fm);
  Eigen::Matrix<double,4,3> pl;
  for (int L = 0; L < 4; ++L) {
    int kk = std::min(SWING_PHASE[L], nsp.num_phases() - 1);   // ★호라이즌 절단(nph<4) 대비
    Vector6d bp = nsp.pos_at(kk, nsp.gait[kk].duration);
    double px = xb + bp(0) + prm.hip_offsets(L,0) + 0.5 * T_STANCE * c.vadv, py = bp(1) + prm.hip_offsets(L,1);
    if (assign && assign[L] >= 0) clamp_into(sc.f.s[assign[L]], px, py);
    pl(L,0) = px - xb; pl(L,1) = py; pl(L,2) = sc.f.h(px, py);
  }
  TamolsState st = nominal_state(prm, z0, 0.0, c, fm, pl);
  o.b = breakdown(st, sc.hsol, sc.cell, sc.N);
  o.margin = min_support_margin(st); o.eps_need = max_giac_viol(st); o.p = pl;
  return o;
}

// ═══════════════════════ T3: 다중스텝 예측(함정 시퀀스) ═══════════════════════
//  ★사전등록 (실행 전 확정)
//   3-A TAMOLS 앞다리 발판이 긴 돌의 **먼 쪽 절반**(x>중심)에 놓이는가 (기하는 가까운 쪽)
//   3-B 그 결과 다음 사이클에 다음 열 돌 위에 착지하는가
//   3-C (구조감사) 직전 사이클 발판만 near↔far 로 바꿨을 때 다음 사이클 해가 바뀌는가
//   3-D (지형 타당성) 직전 발판 near/far 가 **다음 사이클의 최대 보폭**을 바꾸는가
//        — 안 바뀌면 "먼쪽 끝에 놓아야 n+1이 닿는다"는 함정 자체가 이 정식화에 성립 불가
static int cmd_t3() {
  std::printf("\n══════ 시험3: 다중스텝 예측(함정 시퀀스) ══════\n");
  std::printf("사전등록  3-A 먼쪽절반 · 3-B 다음열 착지 · 3-C 직전발판 영향 · 3-D 함정 성립성\n");
  const double lmax = 0.42, r_h = std::sqrt(lmax*lmax - BASE_H*BASE_H);
  Cfg c; c.l_max = lmax; c.iters = 80;
  const double ADV = c.vadv * 4 * c.dur;
  Params prm = go2_params(lmax);
  std::printf("  구조: 발판 결정변수 = Matrix<4,3> = **다리당 1개**(horizon = 1 게이트 사이클).\n");
  std::printf("        같은 다리의 연속 두 발판은 같은 문제에 절대 함께 등장하지 않음.\n");
  std::printf("        수평 reach=%.3f m · ADV=%.3f m\n", r_h, ADV);
  Scene sc;
  sc.f.s.push_back(platform(-0.60, 0.28, 0.45));
  sc.f.s.push_back(Stone{0.55,  HIPY, 0.17, 0.10, STONE_TOP, "LONG_L"});
  sc.f.s.push_back(Stone{0.55, -HIPY, 0.17, 0.10, STONE_TOP, "LONG_R"});
  sc.f.s.push_back(Stone{1.05,  HIPY, 0.08, 0.10, STONE_TOP, "NEXT_L"});
  sc.f.s.push_back(Stone{1.05, -HIPY, 0.08, 0.10, STONE_TOP, "NEXT_R"});
  const double LONGC = 0.55;
  double z0 = BASE_H + STONE_TOP, xb = 0.0;
  Eigen::Matrix<double,4,3> fm;
  for (int L = 0; L < 4; ++L){ fm(L,0)=prm.hip_offsets(L,0); fm(L,1)=prm.hip_offsets(L,1); fm(L,2)=sc.f.h(fm(L,0),fm(L,1)); }
  sc.build(xb);
  CandRes t1c = solve_cand(sc, prm, c, z0, xb, fm, nullptr);
  TamolsState nomsp = nominal_spline(prm, z0, c, fm);
  SnapOut sn = geo_snap(sc.f, nomsp, xb, c.vadv, 0.0);
  Eigen::Matrix<double,4,3> snl = to_local(sn.p, xb);
  TamolsState stn = nominal_state(prm, z0, 0.0, c, fm, snl);
  std::printf("\n  [사이클1] LONG 돌 = [%.2f, %.2f] (중심 %.2f)\n", LONGC-0.17, LONGC+0.17, LONGC);
  print_cand("TAMOLS", sc, t1c, xb, lmax);
  report_legs("기하스냅", sc.f, stn, snl, xb, lmax);
  int far_t=0, far_s=0;
  for (int L : {0,1}) { if (xb+t1c.st.p(L,0) > LONGC) ++far_t; if (sn.p(L,0) > LONGC) ++far_s; }
  std::printf("   3-A 먼쪽 절반: TAMOLS %d/2 · 기하 %d/2\n", far_t, far_s);

  double dx = std::max(0.45*ADV, std::min(1.25*ADV, t1c.st.pos_at(3,c.dur)(0)));
  double xb2 = xb + dx;
  Eigen::Matrix<double,4,3> fm2;
  for (int L = 0; L < 4; ++L){ fm2(L,0)=(xb+t1c.st.p(L,0))-xb2; fm2(L,1)=t1c.st.p(L,1); fm2(L,2)=sc.f.h(xb+t1c.st.p(L,0),t1c.st.p(L,1)); }
  Scene sc2; sc2.f = sc.f; sc2.build(xb2);
  CandRes t2c = solve_cand(sc2, prm, c, z0, xb2, fm2, nullptr);
  std::printf("\n  [사이클2] anchor=%.3f\n", xb2);
  print_cand("TAMOLS", sc2, t2c, xb2, lmax);
  int on_next = 0;
  for (int L : {0,1}) { int si; sc2.f.h(xb2+t2c.st.p(L,0), t2c.st.p(L,1), &si);
                        if (si>=0 && std::strncmp(sc2.f.s[si].tag,"NEXT",4)==0) ++on_next; }
  std::printf("   3-B NEXT 착지: TAMOLS %d/2\n", on_next);

  std::printf("\n  [3-C/3-D] 직전 앞발판만 near/far 강제 → 재풀이 · 최대 보폭(vadv=0.6) 비교\n");
  Scene sc3; sc3.f = sc.f; sc3.build(xb2);
  Eigen::Matrix<double,4,3> res[2]; double xend[2];
  for (int v = 0; v < 2; ++v) {
    double fx = LONGC + (v ? +0.145 : -0.145);
    Eigen::Matrix<double,4,3> fm3;
    for (int L = 0; L < 4; ++L) { double px = (L<2)? fx : (xb + prm.hip_offsets(L,0) + dx);
      fm3(L,0)=px-xb2; fm3(L,1)=prm.hip_offsets(L,1); fm3(L,2)=sc3.f.h(px, fm3(L,1)); }
    CandRes a = solve_cand(sc3, prm, c, z0, xb2, fm3, nullptr);
    res[v] = a.st.p;
    Cfg cl = c; cl.vadv = 0.60;                                   // 최대 보폭 요구
    CandRes b2 = solve_cand(sc3, prm, cl, z0, xb2, fm3, nullptr);
    xend[v] = b2.st.pos_at(3, c.dur)(0);
    std::printf("   직전 x=%.3f(%-4s) → 발판 ", fx, v?"far":"near");
    for (int L = 0; L < 4; ++L) std::printf("%s%+.3f ", LEGN[L], xb2+a.st.p(L,0));
    std::printf("| vadv0.6 최대보폭 base_x_end=%.4f (ineq=%.0e)\n", xend[v], b2.r.ineq_viol);
  }
  double dmax = (res[0]-res[1]).cwiseAbs().maxCoeff();
  std::printf("   3-C 발판 최대차 = %.4f m → 직전 발판 영향: %s\n", dmax, dmax<5e-3?"없음":"있음");
  std::printf("   3-D 최대보폭 차 = %.4f m → 직전 발판이 다음 사이클 도달범위를 %s\n",
              std::fabs(xend[0]-xend[1]), std::fabs(xend[0]-xend[1])<5e-3 ? "**바꾸지 않음**(함정 성립 불가)" : "바꿈");
  std::printf("\n  ▶ 판정  3-A %s · 3-B %s\n", far_t>=2?"PASS":"FAIL", on_next>=2?"PASS":"FAIL");
  return 0;
}


// ═══════════════════ T3S: 스탠스 reach(사이클 간 결합) 정량 ═══════════════════
//  ★사전등록 (실행 전 확정)
//   S-1 직전(구) 발판을 near/far 로 강제했을 때 **다음 사이클 달성 가능 보폭**이 달라지는가.
//       달성 가능 보폭 = terminal 제약 x_target 을 이분탐색해 얻은 **feasible 최대값**
//       (3-D 의 base_x_end 는 feasible 이면 항상 x_target 에 딱 붙어 포화 → 도달범위를 못 잼).
//   S-2 방향: 발이 **뒤**에 박혀 있을수록 보폭이 짧아야 한다(뒷발 far-behind < near).
//       앞발은 부호가 반대(앞발이 앞에 있을수록 base 가 더 나갈 수 있음) — 같은 기전의 반대편.
//   S-3 게이트 OFF 에서는 Δ≈0 이어야 한다(=현재 상태 재확인).
//  ※ 강제 발판은 τ=0 에서 **reach 안**이 되도록 배치(δ=0.145m). 시작부터 위반이면
//    "보폭이 준다"가 아니라 "문제가 애초에 infeasible"이 되어 측정이 오염된다.

// 게이트와 무관하게 "스탠스(구 발판) reach" 위반량을 직접 계산 (초과 신장 [m])
static double stance_viol_m(const TamolsState& st) {
  const int S = 6; const double lmax = st.prm.l_max; double worst = 0;
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    for (int L = 0; L < 4; ++L) {
      if (!(st.gait[k].contact[L] && st.fsrc(k, L) < 0)) continue;
      for (int q = 1; q <= S; ++q) {
        double tau = Tk * q / (double)S; Vector6d pose = st.pos_at(k, tau);
        Vector3d d = pose.head<3>() + R_B(pose.tail<3>()) * st.prm.hip_offsets.row(L).transpose() - st.p_meas.row(L).transpose();
        worst = std::max(worst, d.norm() - lmax);
      }
    }
  }
  return worst;
}
// phase k 에서 "구 발판 스탠스" 다리들이 주는 base_x 상한(해석·보수적: 크라우치/pitch 여유 무시)
//   min_L p_meas.x − hip.x + √(l_max²−Δz²),  Δz = z_nom − 발 z
static double base_x_cap(const TamolsState& st, int k, double z_nom) {
  double cap = 1e9;
  for (int L = 0; L < 4; ++L) {
    if (!(st.gait[k].contact[L] && st.fsrc(k, L) < 0)) continue;
    double dz = z_nom - st.p_meas(L, 2);
    double rh = std::sqrt(std::max(1e-9, st.prm.l_max*st.prm.l_max - dz*dz));
    cap = std::min(cap, st.p_meas(L,0) - st.prm.hip_offsets(L,0) + rh);
  }
  return cap;
}

static int cmd_t3s() {
  std::printf("\n══════ 시험3S: 스탠스 reach = 사이클 간 결합 ══════\n");
  std::printf("사전등록  S-1 직전(구) 발판을 near/far 로 옮기면 **그 발이 붙어 있는 동안 몸이 나갈 수 있는 거리**\n");
  std::printf("              (= 해당 phase 끝 base_x, 다음 발이 닿을 수 있는 곳을 결정)가 달라지는가\n");
  std::printf("          S-2 방향: 뒷발이 뒤로 δ 만큼 박히면 base_x 상한도 δ 만큼 줄어야(1:1)\n");
  std::printf("          S-3 게이트 OFF 면 Δ≈0 이어야(현재 상태 재확인)\n");
  std::printf("  ※ 관측점을 '호라이즌 끝 base_x'로 잡으면 안 된다 — 구 발판이 **떨어진 뒤**엔 자유라\n");
  std::printf("     마지막 phase 에서 가속으로 만회한다(속도 상한이 없는 정식화). 결합은 stance 창에서만 문다.\n");
  const double lmax = L_MAX, r_h = std::sqrt(lmax*lmax - BASE_H*BASE_H);
  Cfg c; c.l_max = lmax; c.iters = 80; c.vadv = 0.60;          // ★강한 전진 요구(상한을 물게)
  Params prm = go2_params(lmax);
  const double DELTA = 0.145;
  std::printf("  수평 reach=%.3f m · 강제 δ=%.3f m · vadv=%.2f · 이론 Δ(base_x 상한) = δ = %.3f m\n",
              r_h, DELTA, c.vadv, DELTA);
  Scene sc; sc.f.s.push_back(platform(-0.80, 2.20, 0.60));      // 넓은 평지(발판 선택 교란 제거)
  double z0 = BASE_H + STONE_TOP, xb = 0.0; sc.build(xb);
  Eigen::Matrix<double,4,3> fm0;
  for (int L = 0; L < 4; ++L) { fm0(L,0)=prm.hip_offsets(L,0); fm0(L,1)=prm.hip_offsets(L,1); fm0(L,2)=sc.f.h(fm0(L,0),fm0(L,1)); }
  // 관측 phase: FL(=phase3 스윙)은 phase 0-2 동안 구 발판 → phase2 끝. RL(=phase2 스윙)은 phase 0-1 → phase1 끝.
  const int OBS_FRONT = 2, OBS_REAR = 1;
  struct Case { const char* tag; int legs[2]; double dx; int obs; };
  Case cs[4] = { {"앞발 δ=0",      {0,1}, 0.0,    OBS_FRONT}, {"앞발 +δ(앞으로)", {0,1}, +DELTA, OBS_FRONT},
                 {"뒷발 δ=0",      {2,3}, 0.0,    OBS_REAR },  {"뒷발 −δ(뒤로)",   {2,3}, -DELTA, OBS_REAR } };
  double obsx[2][4], viol[2][4];
  const double XT = 0.50;
  // ── 용량(capacity) 측정: 호라이즌을 phase 0-1 로 **절단** ──
  //   이렇게 하면 관측점(구 발판이 아직 붙어 있는 마지막 순간)이 곧 terminal 이 되어
  //   "마지막 phase 에서 가속으로 만회" 하는 탈출구가 사라진다. x_target 을 올려가며
  //   ineq_viol 이 꺾이는 지점 = 그 발판이 허용하는 base_x 상한(=용량).
  std::printf("\n  ── 용량 스윕(호라이즌 phase 0-1 절단, RL/FL 이 구 발판에 붙어 있는 구간) ──\n");
  std::printf("     x_target 을 올리며 ineq_viol. 꺾이는 곳 = base_x 상한. 크라우치 포함 이론값:\n");
  {
    double zc = z0 - 0.06, dz = zc - STONE_TOP, rh2 = std::sqrt(lmax*lmax - dz*dz);
    std::printf("       δ=0  : %.3f m   ·   δ=−%.3f : %.3f m   (Δ 이론 = %.3f)\n",
                -HIPX + HIPX + rh2, DELTA, -HIPX - DELTA + HIPX + rh2, DELTA);
  }
  Cfg c2 = c; c2.nph = 2; c2.vadv = 1.2;
  std::printf("   %-9s", "x_target");
  for (int gate = 0; gate < 2; ++gate) for (int q = 0; q < 2; ++q)
    std::printf(" | %s %-9s", gate ? "ON " : "OFF", q ? "δ=−0.145" : "δ=0");
  std::printf("\n");
  const double XSW[8] = {0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60};
  for (int d = 0; d < 8; ++d) {
    std::printf("   %-9.2f", XSW[d]);
    for (int gate = 0; gate < 2; ++gate) {
      if (gate) setenv("STANCE_REACH", "1", 1); else unsetenv("STANCE_REACH");
      for (int q = 0; q < 2; ++q) {
        Eigen::Matrix<double,4,3> fm = fm0; double dxq = q ? -DELTA : 0.0;
        fm(2,0) += dxq; fm(3,0) += dxq;
        double xtd = XSW[d];
        CandRes r = solve_cand(sc, prm, c2, z0, xb, fm, nullptr, &xtd);
        std::printf(" | %13.1e", r.r.ineq_viol);
      }
    }
    std::printf("\n");
  }
  unsetenv("STANCE_REACH");
  // ── knee(용량) 이분탐색: ineq_viol<1e-4 를 유지하는 최대 x_target ──
  //   절단 호라이즌에서는 경계가 날카롭다(1e-7 → 1e-2, 5자릿수 점프) → 이분탐색이 유의미.
  std::printf("\n  ── 용량 knee 이분탐색(= 구 발판이 붙어 있는 동안 도달 가능한 최대 base_x) ──\n");
  double cap4[2][2];
  for (int gate = 0; gate < 2; ++gate) {
    if (gate) setenv("STANCE_REACH", "1", 1); else unsetenv("STANCE_REACH");
    for (int q = 0; q < 2; ++q) {
      double f = 0.10, hh = 0.80;
      while (hh - f > 0.005) {
        double mid = 0.5 * (f + hh);
        Eigen::Matrix<double,4,3> fm = fm0; double dxq = q ? -DELTA : 0.0;
        fm(2,0) += dxq; fm(3,0) += dxq;
        CandRes r = solve_cand(sc, prm, c2, z0, xb, fm, nullptr, &mid);
        if (r.r.ineq_viol <= 1e-4 && r.r.eq_viol <= 1e-5) f = mid; else hh = mid;
      }
      cap4[gate][q] = f;
      std::printf("   %s  δ=%-7s 용량 = %.3f m\n", gate?"ON ":"OFF", q?"−0.145":"0", f);
    }
    std::printf("   → Δ(용량) = %+.3f m  %s\n", cap4[gate][1]-cap4[gate][0],
                gate ? "(이론 −0.145, pitch 여유만큼 작음)" : "(결합 없으면 0)");
  }
  unsetenv("STANCE_REACH");
  for (int gate = 0; gate < 2; ++gate) {
    if (gate) setenv("STANCE_REACH", "1", 1); else unsetenv("STANCE_REACH");
    std::printf("\n  ── STANCE_REACH %s (x_target=%.2f) ──\n", gate ? "ON" : "OFF", XT);
    for (int q = 0; q < 4; ++q) {
      Eigen::Matrix<double,4,3> fm = fm0;
      for (int j = 0; j < 2; ++j) fm(cs[q].legs[j], 0) += cs[q].dx;
      double xt = XT;
      CandRes r = solve_cand(sc, prm, c, z0, xb, fm, nullptr, &xt);
      int k = cs[q].obs;
      obsx[gate][q] = r.st.pos_at(k, c.dur)(0);
      viol[gate][q] = stance_viol_m(r.st);
      double cap = base_x_cap(r.st, k, z0);
      std::printf("   %-16s base_x(ph%d끝)=%.4f  [해석상한 %.4f]  스탠스reach 초과=%+.4f m  ok=%d ineq=%.0e\n",
                  cs[q].tag, k, obsx[gate][q], cap, viol[gate][q], (int)r.r.ok, r.r.ineq_viol);
    }
    std::printf("   Δ(앞발 +δ − 기준) = %+.4f m   Δ(뒷발 −δ − 기준) = %+.4f m\n",
                obsx[gate][1]-obsx[gate][0], obsx[gate][3]-obsx[gate][2]);
  }
  unsetenv("STANCE_REACH");
  double dF_off = obsx[0][1]-obsx[0][0], dR_off = obsx[0][3]-obsx[0][2];
  double dF_on  = obsx[1][1]-obsx[1][0], dR_on  = obsx[1][3]-obsx[1][2];
  std::printf("\n  ▶ 판정\n");
  std::printf("   S-3 OFF Δ: 앞 %+.4f · 뒤 %+.4f → %s\n", dF_off, dR_off,
              (std::fabs(dF_off)<5e-3 && std::fabs(dR_off)<5e-3) ? "결합 없음(기존 확인)" : "★예상외");
  std::printf("   S-1 ON  Δ: 앞 %+.4f · 뒤 %+.4f → %s\n", dF_on, dR_on,
              (std::fabs(dF_on)>5e-3 || std::fabs(dR_on)>5e-3) ? "결합 생김" : "★결합 없음 → 단계2 중단");
  std::printf("   S-2 뒷발 −δ 가 상한을 δ 만큼 줄이는가: Δ=%+.4f vs 이론 −%.4f → %s\n", dR_on, DELTA,
              (dR_on < -0.5*DELTA) ? "부합" : "★불일치");
  std::printf("   OFF 해가 스탠스 reach 를 얼마나 어기고 있었나(=기존 계획의 물리적 불가능량):\n");
  for (int q = 0; q < 4; ++q) std::printf("      %-16s OFF %+.4f m → ON %+.4f m\n", cs[q].tag, viol[0][q], viol[1][q]);
  return 0;
}


// ═══════════════ T3M: N=2 다중 사이클 호라이즌 — 함정 시퀀스 ═══════════════
//  ★사전등록 (실행 전 확정, t3 와 동일 지형·동일 판정식)
//   M-A 사이클0 앞발판이 LONG 돌의 **먼 쪽 절반**(x > 중심 0.55)에 놓이는가  (N=1: 0/2, 기하: 0/2)
//   M-B 그 결과 **사이클1 앞발판이 NEXT 돌 위**에 놓이는가                    (N=1: 0/2, 기하: 0/2)
//   M-C N=1 을 같은 요구(같은 총 전진량)로 순차 2회 돌린 대조군과 비교
//  기전: 같은 다리의 연속 두 발판 p[0][L]·p[1][L] 이 한 문제에 함께 있고,
//        p[0][L] 은 (그 다리가 다시 스윙하기 전까지) reach 로 base 를 잡고,
//        p[1][L] 은 그 시점 base 로부터 reach 안이어야 한다 → p[0]→base→p[1] 사슬.
static QpResult solve_multi(TamolsState& st, const Scene& sc, const Params& prm, const Cfg& c,
                            double z0, double vx0, const Eigen::Matrix<double,4,3>& fm,
                            int N, double xtgt) {
  st.prm = prm; use_scene(sc);
  set_walk_multi(st, c.dur, N);
  int P = st.num_phases();
  st.base_pose << 0, 0, z0, 0, 0, 0;
  st.base_vel  << vx0, 0, 0, 0, 0, 0;
  st.p_meas = fm; st.ref_vel = Vector3d(c.vadv, 0, 0);
  cold_init(st, z0, vx0, c);
  const double ADV = c.vadv * 4 * c.dur;                     // 사이클당 전진량
  for (int n = 0; n < N; ++n) for (int i = 0; i < 4; ++i) {
    double px = prm.hip_offsets(i,0) + (n + 0.5) * ADV, py = prm.hip_offsets(i,1);
    double pz = bilinear_height(sc.hsol, sc.cell, sc.N, px, py);
    if (n == 0) { st.p(i,0)=px; st.p(i,1)=py; st.p(i,2)=pz; }
    else { st.p_ext[n-1](i,0)=px; st.p_ext[n-1](i,1)=py; st.p_ext[n-1](i,2)=pz; }
  }
  st.epsilon = VectorXd::Zero(P);
  QpOptions o; o.max_iter = getenv("TAMCAP_ITERS") ? atoi(getenv("TAMCAP_ITERS")) : c.iters;   // env 없으면 기존과 동일
  o.zlo = z0 - 0.06; o.zhi = z0 + 0.06; o.rp_max = 0.20; o.yaw_max = 0.10;
  o.x_target = xtgt; o.y_min = c.y_min; o.y_max = c.y_max; o.gap = false;
  return solve_fast(st, sc.hsol, sc.cell, sc.N, o);
}
// 사이클 n·다리 L 의 터치다운(=그 사이클 스윙 phase 끝) hip→발 거리
static double td_reach_cyc(const TamolsState& st, int n, int L) {
  int k = 4 * n + SWING_PHASE[L];
  Vector6d pose = st.pos_at(k, st.gait[k].duration);
  return (pose.head<3>() + R_B(pose.tail<3>()) * st.prm.hip_offsets.row(L).transpose() - st.fpos(n, L)).norm();
}
static void print_cycle(const char* tag, const Field& f, const TamolsState& st, int n, double xb, double lmax) {
  std::printf("   %-12s", tag);
  for (int L = 0; L < 4; ++L) {
    Vector3d fp = st.fpos(n, L); int si; f.h(xb + fp(0), fp(1), &si);
    double rr = td_reach_cyc(st, n, L);
    std::printf(" %s(%+.3f,%+.3f)%-7s r=%.3f%s", LEGN[L], xb + fp(0), fp(1),
                si >= 0 ? f.s[si].tag : "VOID", rr, rr > lmax + 1e-3 ? "!" : " ");
  }
  std::printf("\n");
}

static int cmd_t3m() {
  std::printf("\n══════ 시험3M: N=2 다중 사이클 호라이즌(함정 시퀀스) ══════\n");
  std::printf("사전등록  M-A 사이클0 앞발판 먼쪽절반(x>0.55) · M-B 사이클1 앞발판 NEXT 착지\n");
  std::printf("          기준선: N=1 순차 0/2 · 기하스냅 0/2 (t3 실측)\n");
  const double lmax = L_MAX, r_h = std::sqrt(lmax*lmax - BASE_H*BASE_H);
  Cfg c; c.l_max = lmax; c.iters = 80; c.vadv = 0.45; c.dur = 0.20;
  Params prm = go2_params(lmax);
  Scene sc;
  sc.f.s.push_back(platform(-0.60, 0.28, 0.45));
  sc.f.s.push_back(Stone{0.55,  HIPY, 0.17, 0.10, STONE_TOP, "LONG_L"});
  sc.f.s.push_back(Stone{0.55, -HIPY, 0.17, 0.10, STONE_TOP, "LONG_R"});
  sc.f.s.push_back(Stone{1.05,  HIPY, 0.08, 0.10, STONE_TOP, "NEXT_L"});
  sc.f.s.push_back(Stone{1.05, -HIPY, 0.08, 0.10, STONE_TOP, "NEXT_R"});
  const double LONGC = 0.55;
  double z0 = BASE_H + STONE_TOP, xb = 0.0;
  sc.build(xb);
  Eigen::Matrix<double,4,3> fm;
  for (int L = 0; L < 4; ++L){ fm(L,0)=prm.hip_offsets(L,0); fm(L,1)=prm.hip_offsets(L,1); fm(L,2)=sc.f.h(fm(L,0),fm(L,1)); }
  const double T2 = 8 * c.dur, XT = 0.9 * c.vadv * T2;
  std::printf("  LONG=[%.2f,%.2f](중심 %.2f) · NEXT=[%.2f,%.2f] · 수평reach=%.3f · 호라이즌 %.1fs · x_target=%.3f\n",
              LONGC-0.17, LONGC+0.17, LONGC, 1.05-0.08, 1.05+0.08, r_h, T2, XT);
  std::printf("  필요조건: 사이클1 앞발판 x≥0.97 ⇒ 그 직전 base_x ≥ 0.97−hip−reach = %.3f\n", 0.97 - HIPX - r_h);
  std::printf("            그 base 를 사이클0 앞발판이 reach 로 붙잡음 ⇒ 사이클0 앞발판 x ≳ %.3f (LONG 중심 %.2f 보다 앞)\n",
              0.97 - HIPX - r_h - (HIPX + r_h) + 2*HIPX, LONGC);

  for (int stnc = 0; stnc < 2; ++stnc) {
    if (stnc) setenv("STANCE_REACH", "1", 1); else unsetenv("STANCE_REACH");
    std::printf("\n  ══ STANCE_REACH %s ══\n", stnc ? "ON" : "OFF");

    // ── N=2 ──
    TamolsState s2; QpResult r2 = solve_multi(s2, sc, prm, c, z0, 0.0, fm, 2, XT);
    double xend2 = s2.pos_at(7, c.dur)(0);
    std::printf("   [N=2] ok=%d iters=%d eq=%.0e ineq=%.0e  base_x끝=%.3f\n", (int)r2.ok, r2.iters, r2.eq_viol, r2.ineq_viol, xend2);
    print_cycle("사이클0", sc.f, s2, 0, xb, lmax); terr_line("  지형", sc, s2, 0);
    print_cycle("사이클1", sc.f, s2, 1, xb, lmax); terr_line("  지형", sc, s2, 1);
    int farA = 0, onB = 0, voidn = 0;
    for (int L : {0,1}) if (s2.fpos(0,L)(0) > LONGC) ++farA;
    for (int L : {0,1}) { Vector3d fp = s2.fpos(1,L); int si; sc.f.h(fp(0), fp(1), &si);
                          if (si>=0 && std::strncmp(sc.f.s[si].tag,"NEXT",4)==0) ++onB; }
    for (int n = 0; n < 2; ++n) for (int L = 0; L < 4; ++L) { Vector3d fp = s2.fpos(n,L); int si; sc.f.h(fp(0),fp(1),&si); if (si<0) ++voidn; }
    std::printf("   M-A 먼쪽절반 %d/2 · M-B NEXT 착지 %d/2 · VOID 발판 %d/8\n", farA, onB, voidn);

    // ── N=1 순차 2회(같은 총 요구) ──
    Cfg c1 = c; TamolsState sA;
    QpResult rA = solve_multi(sA, sc, prm, c1, z0, 0.0, fm, 1, 0.5 * XT);
    double dx = sA.pos_at(3, c1.dur)(0);
    Eigen::Matrix<double,4,3> fm2;
    for (int L = 0; L < 4; ++L) { Vector3d fp = sA.fpos(0,L); fm2(L,0)=fp(0)-dx; fm2(L,1)=fp(1); fm2(L,2)=sc.f.h(fp(0),fp(1)); }
    Scene scB; scB.f = sc.f; scB.build(dx);
    TamolsState sB; QpResult rB = solve_multi(sB, scB, prm, c1, z0, 0.0, fm2, 1, 0.5 * XT);
    std::printf("   [N=1 순차] A ineq=%.0e → 앵커 %.3f → B ineq=%.0e  base_x끝=%.3f\n",
                rA.ineq_viol, dx, rB.ineq_viol, dx + sB.pos_at(3, c1.dur)(0));
    std::printf("   %-12s", "A(사이클0)");
    for (int L = 0; L < 4; ++L) { Vector3d fp = sA.fpos(0,L); int si; sc.f.h(fp(0),fp(1),&si);
      std::printf(" %s(%+.3f,%+.3f)%-7s", LEGN[L], fp(0), fp(1), si>=0?sc.f.s[si].tag:"VOID"); }
    std::printf("\n   %-12s", "B(사이클1)");
    for (int L = 0; L < 4; ++L) { Vector3d fp = sB.fpos(0,L); int si; scB.f.h(dx+fp(0),fp(1),&si);
      std::printf(" %s(%+.3f,%+.3f)%-7s", LEGN[L], dx+fp(0), fp(1), si>=0?scB.f.s[si].tag:"VOID"); }
    std::printf("\n");
    int farA1 = 0, onB1 = 0;
    for (int L : {0,1}) if (sA.fpos(0,L)(0) > LONGC) ++farA1;
    for (int L : {0,1}) { Vector3d fp = sB.fpos(0,L); int si; scB.f.h(dx+fp(0), fp(1), &si);
                          if (si>=0 && std::strncmp(scB.f.s[si].tag,"NEXT",4)==0) ++onB1; }
    std::printf("   M-C(N=1) 먼쪽절반 %d/2 · NEXT 착지 %d/2\n", farA1, onB1);
    std::printf("   ▶ 판정  M-A %s · M-B %s   (N=1 대조 %s/%s)\n",
                farA>=2?"PASS":"FAIL", onB>=2?"PASS":"FAIL", farA1>=2?"PASS":"FAIL", onB1>=2?"PASS":"FAIL");
  }
  // ── 요구 전진량 스윕: "놓을 수 **있는가**"(운동학 사슬) vs "놓기로 **고르는가**"(목적함수) 분리 ──
  //   현 정식화의 목적함수에는 **VOID 페널티가 없다**(foothold_on_ground = (h(p)−p.z)² 인데
  //   void 에서는 h=0 이고 p.z 도 0 으로 내리면 비용 0). 즉 "돌을 고를 이유"가 비용에 없다.
  //   그래서 발판 선택은 전진 요구(x_target)가 밀어낸 만큼만 앞으로 간다 → 요구를 올려 사슬을 시험한다.
  std::printf("\n  ══ 요구(x_target) 스윕: N=2 vs N=1 순차 ══\n");
  std::printf("   %-8s | %-34s | %-34s\n", "x_target", "N=2 사이클1 앞발판 (ineq)", "N=1 순차 B 앞발판 (ineq)");
  setenv("STANCE_REACH", "1", 1);
  const double XSW[5] = {0.648, 0.75, 0.85, 0.95, 1.05};
  for (int d = 0; d < 5; ++d) {
    TamolsState q2; QpResult rq = solve_multi(q2, sc, prm, c, z0, 0.0, fm, 2, XSW[d]);
    int si0, si1; sc.f.h(q2.fpos(1,0)(0), q2.fpos(1,0)(1), &si0); sc.f.h(q2.fpos(1,1)(0), q2.fpos(1,1)(1), &si1);
    TamolsState qA; solve_multi(qA, sc, prm, c, z0, 0.0, fm, 1, 0.5*XSW[d]);
    double dxq = qA.pos_at(3, c.dur)(0);
    Eigen::Matrix<double,4,3> fmq;
    for (int L = 0; L < 4; ++L) { Vector3d fp = qA.fpos(0,L); fmq(L,0)=fp(0)-dxq; fmq(L,1)=fp(1); fmq(L,2)=sc.f.h(fp(0),fp(1)); }
    Scene scq; scq.f = sc.f; scq.build(dxq);
    TamolsState qB; QpResult rB2 = solve_multi(qB, scq, prm, c, z0, 0.0, fmq, 1, 0.5*XSW[d]);
    int sb0, sb1; scq.f.h(dxq+qB.fpos(0,0)(0), qB.fpos(0,0)(1), &sb0); scq.f.h(dxq+qB.fpos(0,1)(0), qB.fpos(0,1)(1), &sb1);
    std::printf("   %-8.3f | FL%.3f%-7s FR%.3f%-7s %.0e | FL%.3f%-7s FR%.3f%-7s %.0e\n", XSW[d],
      q2.fpos(1,0)(0), si0>=0?sc.f.s[si0].tag:"VOID", q2.fpos(1,1)(0), si1>=0?sc.f.s[si1].tag:"VOID", rq.ineq_viol,
      dxq+qB.fpos(0,0)(0), sb0>=0?scq.f.s[sb0].tag:"VOID", dxq+qB.fpos(0,1)(0), sb1>=0?scq.f.s[sb1].tag:"VOID", rB2.ineq_viol);
  }
  // ── 함정이 **정식화 안에서 성립하는지** 직접 시험 ──
  //   위 스윕은 "고르는가"를 못 본다(목적함수에 VOID 페널티가 없어 돌을 고를 이유가 없음).
  //   그래서 먼쪽 배치를 **지형으로 강제**해 분리한다:
  //     LONG_full  = [0.38,0.72] (먼쪽 절반 있음)  vs  LONG_short = [0.38,0.55] (먼쪽 절반 없음)
  //   먼쪽 배치가 n+1 도달에 **필수**라면 short 의 도달 한계가 full 보다 낮아야 한다.
  std::printf("\n  ══ 함정 성립성: LONG 먼쪽 절반 유무가 사이클1 도달을 바꾸는가 ══\n");
  setenv("STANCE_REACH", "1", 1);
  for (int variant = 0; variant < 2; ++variant) {
    Scene sv;
    sv.f.s.push_back(platform(-0.60, 0.28, 0.45));
    double hx = variant ? 0.17 : 0.085, cxs = variant ? 0.55 : 0.465;   // short=[0.38,0.55] · full=[0.38,0.72]
    sv.f.s.push_back(Stone{cxs,  HIPY, hx, 0.10, STONE_TOP, variant?"LONG_L":"SHRT_L"});
    sv.f.s.push_back(Stone{cxs, -HIPY, hx, 0.10, STONE_TOP, variant?"LONG_R":"SHRT_R"});
    sv.f.s.push_back(Stone{1.05,  HIPY, 0.08, 0.10, STONE_TOP, "NEXT_L"});
    sv.f.s.push_back(Stone{1.05, -HIPY, 0.08, 0.10, STONE_TOP, "NEXT_R"});
    sv.build(xb);
    Eigen::Matrix<double,4,3> fmv;
    for (int L = 0; L < 4; ++L){ fmv(L,0)=prm.hip_offsets(L,0); fmv(L,1)=prm.hip_offsets(L,1); fmv(L,2)=sv.f.h(fmv(L,0),fmv(L,1)); }
    std::printf("   ── LONG %s ([%.2f,%.2f]) ──\n", variant?"full ":"short", cxs-hx, cxs+hx);
    for (int nn = 2; nn >= 1; --nn) {
      // 최대 도달: x_target 이분탐색(ineq<1e-4 유지)
      double f = 0.40, hh = 1.60;
      while (hh - f > 0.01) {
        double mid = 0.5*(f+hh);
        TamolsState qq; QpResult rr;
        if (nn == 2) rr = solve_multi(qq, sv, prm, c, z0, 0.0, fmv, 2, mid);
        else { TamolsState qA; solve_multi(qA, sv, prm, c, z0, 0.0, fmv, 1, 0.5*mid);
               double dxq = qA.pos_at(3,c.dur)(0);
               Eigen::Matrix<double,4,3> fq;
               for (int L=0;L<4;++L){ Vector3d fp=qA.fpos(0,L); fq(L,0)=fp(0)-dxq; fq(L,1)=fp(1); fq(L,2)=sv.f.h(fp(0),fp(1)); }
               Scene sq; sq.f = sv.f; sq.build(dxq);
               rr = solve_multi(qq, sq, prm, c, z0, 0.0, fq, 1, 0.5*mid); }
        if (rr.ineq_viol <= 1e-4 && rr.eq_viol <= 1e-5) f = mid; else hh = mid;
      }
      std::printf("      N=%d 최대 도달 x_target = %.3f\n", nn, f);
    }
    // ★진단: 높은 요구에서 발판이 실제 지형 위에 있는가(p.z vs 정확지형 h) — soft cost 라 공중부양 가능
    { TamolsState qd; QpResult rd = solve_multi(qd, sv, prm, c, z0, 0.0, fmv, 2, 1.40);
      std::printf("      [N=2 x_target=1.40 진단 ineq=%.0e] 발판 p.z vs 정확지형h:", rd.ineq_viol);
      for (int n = 0; n < 2; ++n) for (int L = 0; L < 2; ++L) {
        Vector3d fp = qd.fpos(n, L);
        std::printf("  c%d%s x=%.2f z=%.3f h=%.3f%s", n, LEGN[L], fp(0), fp(2), sv.f.h(fp(0), fp(1)),
                    std::fabs(fp(2) - sv.f.h(fp(0), fp(1))) > 0.03 ? "★공중" : "");
      }
      std::printf("\n"); }
  }
  unsetenv("STANCE_REACH");
  return 0;
}


// ═══════════════ COST: N=1 vs N=2 문제 크기·solve 시간·수렴 ═══════════════
//   TAMOLS 의 강점이 "빠른 QP" 였으므로 호라이즌 확장 비용을 반드시 잰다.
//   측정: ①결정변수/제약 행 수 ②RTI(5-iter) cold·warm 시간 ③완전수렴(max_iter=80) 반복수·잔여위반
static int cmd_cost() {
  std::printf("\n══════ 비용 측정: N=1 vs N=2 ══════\n");
  Cfg c; c.l_max = L_MAX; c.vadv = 0.45; c.dur = 0.20;
  Params prm = go2_params(L_MAX);
  Scene sc;
  sc.f.s.push_back(platform(-0.60, 0.28, 0.45));
  sc.f.s.push_back(Stone{0.55,  HIPY, 0.17, 0.10, STONE_TOP, "LONG_L"});
  sc.f.s.push_back(Stone{0.55, -HIPY, 0.17, 0.10, STONE_TOP, "LONG_R"});
  sc.f.s.push_back(Stone{1.05,  HIPY, 0.08, 0.10, STONE_TOP, "NEXT_L"});
  sc.f.s.push_back(Stone{1.05, -HIPY, 0.08, 0.10, STONE_TOP, "NEXT_R"});
  double z0 = BASE_H + STONE_TOP; sc.build(0.0); use_scene(sc);
  Eigen::Matrix<double,4,3> fm;
  for (int L = 0; L < 4; ++L){ fm(L,0)=prm.hip_offsets(L,0); fm(L,1)=prm.hip_offsets(L,1); fm(L,2)=sc.f.h(fm(L,0),fm(L,1)); }

  for (int stnc = 0; stnc < 2; ++stnc) {
    if (stnc) setenv("STANCE_REACH", "1", 1); else unsetenv("STANCE_REACH");
    std::printf("\n  ── STANCE_REACH %s ──\n", stnc ? "ON" : "OFF");
    std::printf("   %-6s %-5s %-5s %-6s %-6s | %-11s %-11s | %-16s\n",
                "N", "phase", "nz", "eq행", "ineq행", "RTI5 cold", "RTI5 warm", "완전수렴(80)");
    for (int N = 1; N <= 2; ++N) {
      double XT = 0.9 * c.vadv * (4 * N) * c.dur;
      // 크기
      TamolsState s0;
      { s0.prm = prm; set_walk_multi(s0, c.dur, N);
        s0.base_pose << 0,0,z0,0,0,0; s0.base_vel << 0,0,0,0,0,0; s0.p_meas = fm;
        s0.ref_vel = Vector3d(c.vadv,0,0); cold_init(s0, z0, 0.0, c);
        double ADV = c.vadv * 4 * c.dur;
        for (int n = 0; n < N; ++n) for (int i = 0; i < 4; ++i) {
          double px = prm.hip_offsets(i,0) + (n+0.5)*ADV, py = prm.hip_offsets(i,1);
          double pz = bilinear_height(sc.hsol, sc.cell, sc.N, px, py);
          if (n==0){s0.p(i,0)=px;s0.p(i,1)=py;s0.p(i,2)=pz;} else {s0.p_ext[n-1](i,0)=px;s0.p_ext[n-1](i,1)=py;s0.p_ext[n-1](i,2)=pz;}
        }
        s0.epsilon = VectorXd::Zero(s0.num_phases()); }
      QpOptions oz; oz.gap=false; oz.y_min=c.y_min; oz.y_max=c.y_max; oz.x_target=XT;
      oz.zlo=z0-0.06; oz.zhi=z0+0.06; oz.rp_max=0.20; oz.yaw_max=0.10;
      Packer pkz(s0.num_phases(), s0.ncyc());
      int neq = (int)eq_constraints(s0).size(), nineq = (int)ineq_constraints(s0, oz).size();

      // RTI 5-iter cold (5회 평균)
      Cfg cr = c; cr.iters = 5;
      double tc = 0; TamolsState sw; QpResult rr;
      for (int rep = 0; rep < 5; ++rep) {
        TamolsState st; auto t0 = std::chrono::high_resolution_clock::now();
        rr = solve_multi(st, sc, prm, cr, z0, 0.0, fm, N, XT);
        auto t1 = std::chrono::high_resolution_clock::now();
        tc += std::chrono::duration<double,std::milli>(t1-t0).count();
        if (rep == 0) sw = st;
      }
      tc /= 5;
      // RTI 5-iter warm (직전 해에서 재출발 = solve_fast 재호출)
      double tw = 0;
      for (int rep = 0; rep < 5; ++rep) {
        TamolsState st = sw; auto t0 = std::chrono::high_resolution_clock::now();
        QpOptions ow = oz; ow.max_iter = 5;
        solve_fast(st, sc.hsol, sc.cell, sc.N, ow);
        auto t1 = std::chrono::high_resolution_clock::now();
        tw += std::chrono::duration<double,std::milli>(t1-t0).count();
      }
      tw /= 5;
      // 완전수렴
      Cfg cf = c; cf.iters = 80;
      TamolsState sf; auto t0 = std::chrono::high_resolution_clock::now();
      QpResult rf = solve_multi(sf, sc, prm, cf, z0, 0.0, fm, N, XT);
      auto t1 = std::chrono::high_resolution_clock::now();
      double tf = std::chrono::duration<double,std::milli>(t1-t0).count();
      std::printf("   N=%-4d %-5d %-5d %-6d %-6d | %6.1f ms   %6.1f ms   | it=%2d %6.0fms ok=%d ineq=%.0e\n",
                  N, s0.num_phases(), pkz.nz, neq, nineq, tc, tw, rf.iters, tf, (int)rf.ok, rf.ineq_viol);
    }
  }
  unsetenv("STANCE_REACH");
  std::printf("\n  ※ 실시간 기준 20ms(50Hz). RTI5 cold 가 기준선.\n");
  return 0;
}

// ═══════════════════════ T1: GIAC(균형 인지 발판) ═══════════════════════
//  ★사전등록
//   1-A TAMOLS 가 at_des 다리 중 하나 이상에서 **최근접이 아닌 돌**을 고르는가
//   1-B 그 선택이 지지 마진(=base xy→지지다각형 경계 부호거리의 phase·τ 최소)을 **키우는가**
//   1-C 필요 eps(=GIAC 위반량)가 기하 대비 **줄어드는가**
//   1-D (기전 귀속) 후보별(R_in / R_out) 초기화로 각각 국소해를 구해 **TAMOLS 목적함수가
//       어느 쪽을 더 낮게 평가하는지** — 목적함수가 넓은 지지(R_out)를 선호하면 기전 존재.
static int cmd_t1() {
  std::printf("\n══════ 시험1: GIAC(균형 인지 발판) ══════\n");
  std::printf("사전등록  1-A 비최근접 돌 · 1-B 지지마진 증가 · 1-C 필요eps 감소 · 1-D 목적함수 선호\n");
  const double lmax = 0.42;
  Cfg c; c.l_max = lmax; c.iters = 80; c.y_min = 0.02; c.y_max = 0.45;
  Params prm = go2_params(lmax);
  Scene sc;
  sc.f.s.push_back(platform(-0.60, 0.28, 0.45));
  sc.f.s.push_back(Stone{0.46,  0.150, 0.10, 0.10, STONE_TOP, "L_nom"});
  sc.f.s.push_back(Stone{0.46, -0.210, 0.10, 0.06, STONE_TOP, "R_near"});   // ★최근접 — 그러나 지지마진 나쁨(스캔 확인)
  sc.f.s.push_back(Stone{0.46, -0.040, 0.10, 0.04, STONE_TOP, "R_good"});   // 더 멀지만 지지마진 좋음
  double z0 = BASE_H + STONE_TOP, xb = 0.0;
  Eigen::Matrix<double,4,3> fm;
  for (int L = 0; L < 4; ++L){ fm(L,0)=prm.hip_offsets(L,0); fm(L,1)=prm.hip_offsets(L,1); fm(L,2)=sc.f.h(fm(L,0),fm(L,1)); }
  sc.build(xb);
  TamolsState nomsp = nominal_spline(prm, z0, c, fm);
  SnapOut sn = geo_snap(sc.f, nomsp, xb, c.vadv, 0.0);
  Eigen::Matrix<double,4,3> snl = to_local(sn.p, xb);
  TamolsState stn = nominal_state(prm, z0, 0.0, c, fm, snl);
  double m_s = min_support_margin(stn), e_s = max_giac_viol(stn);
  // ── 기전 스캔: FR 발판 y 만 바꿔 최소 지지마진이 어떻게 변하는가(명목 base·나머지 명목) ──
  std::printf("\n  [기전 스캔] FR 발판 y 별 최소 지지마진(명목 base·나머지 다리 명목)\n   %8s","FR y");
  { PinRes P0 = pinned_eval(sc, prm, c, z0, xb, fm, nullptr);
    for (double y = -0.04; y >= -0.361; y -= 0.04) std::printf(" %7.3f", y);
    std::printf("\n   %8s","전체");
    for (double y = -0.04; y >= -0.361; y -= 0.04) {
      Eigen::Matrix<double,4,3> q = P0.p; q(1,1) = y; q(1,2) = STONE_TOP;
      TamolsState t = nominal_state(prm, z0, 0.0, c, fm, q);
      std::printf(" %7.4f", min_support_margin(t)); }
    std::printf("\n   %8s","마지막");
    for (double y = -0.04; y >= -0.361; y -= 0.04) {
      Eigen::Matrix<double,4,3> q = P0.p; q(1,1) = y; q(1,2) = STONE_TOP;
      TamolsState t = nominal_state(prm, z0, 0.0, c, fm, q);
      std::printf(" %7.4f", last_phase_margin(t)); }
    std::printf("\n   ※ '전체'는 phase0~1(구 발판 p_meas·신규 발판 무관)이 지배 → 신규 발판 효과는 '마지막' 로 판단\n"); }
  std::printf("\n  [기하스냅] 마진=%+.4f  필요eps=%.2f  (FR 이 고른 돌=%s)\n",
              m_s, e_s, sc.f.s[sn.stone[1]].tag);
  report_legs("기하스냅", sc.f, stn, snl, xb, lmax);
  int iR_in = -1, iR_out = -1, iL = -1;
  for (size_t i = 0; i < sc.f.s.size(); ++i) { if (!std::strcmp(sc.f.s[i].tag,"R_near")) iR_in=i;
    if (!std::strcmp(sc.f.s[i].tag,"R_good")) iR_out=i; if (!std::strcmp(sc.f.s[i].tag,"L_nom")) iL=i; }
  struct V { const char* n; const char* yb; };
  V vs[2] = {{"base y 자유","0"},{"base |y|<=0.05","0.05"}};
  for (int v = 0; v < 2; ++v) {
    if (std::strcmp(vs[v].yb,"0")) setenv("BASE_YBND", vs[v].yb, 1); else unsetenv("BASE_YBND");
    std::printf("\n  ── %s ──\n", vs[v].n);
    CandRes cold = solve_cand(sc, prm, c, z0, xb, fm, nullptr);
    print_cand("cold(명목init)", sc, cold, xb, lmax);
    int a1[4] = {iL, iR_in, -1, -1}, a2[4] = {iL, iR_out, -1, -1};
    CandRes A = solve_cand(sc, prm, c, z0, xb, fm, a1), B = solve_cand(sc, prm, c, z0, xb, fm, a2);
    print_cand("cand FR->R_near", sc, A, xb, lmax);
    print_cand("cand FR->R_good", sc, B, xb, lmax);
    int si; sc.f.h(xb+cold.st.p(1,0), cold.st.p(1,1), &si);
    bool diff = (si != sn.stone[1]);
    std::printf("   ▶ 1-A %s(cold FR 돌=%s vs 기하=%s) · 1-B %s(%+.4f vs %+.4f) · 1-C %s(%.2f vs %.2f)\n",
                diff?"PASS":"FAIL", si>=0?sc.f.s[si].tag:"VOID", sc.f.s[sn.stone[1]].tag,
                last_phase_margin(cold.st) > last_phase_margin(stn) + 1e-3 ? "PASS":"FAIL", last_phase_margin(cold.st), last_phase_margin(stn),
                cold.eps_need < e_s - 1e-3 ? "PASS":"FAIL", cold.eps_need, e_s);
    PinRes PA = pinned_eval(sc, prm, c, z0, xb, fm, a1), PB = pinned_eval(sc, prm, c, z0, xb, fm, a2);
    std::printf("   ▶ 1-D 솔버해 목적함수: R_near %.4f vs R_good %.4f → %s (수렴마진 %+.4f vs %+.4f)\n",
                A.b.total, B.b.total, B.b.total < A.b.total ? "R_good(고마진) 선호" : "R_near(최근접) 선호",
                A.margin, B.margin);
    std::printf("   ▶ 1-D' 발판고정+명목base: R_near cost %.4f(마지막마진%+.4f, 필요eps %.2f) vs R_good %.4f(마지막마진%+.4f, 필요eps %.2f) → %s\n",
                PA.b.total + 50*PA.eps_need*PA.eps_need, last_phase_margin(nominal_state(prm,z0,0.0,c,fm,PA.p)), PA.eps_need,
                PB.b.total + 50*PB.eps_need*PB.eps_need, last_phase_margin(nominal_state(prm,z0,0.0,c,fm,PB.p)), PB.eps_need,
                PB.b.total + 50*PB.eps_need*PB.eps_need < PA.b.total + 50*PA.eps_need*PA.eps_need ? "R_good 선호 PASS" : "R_near 선호 FAIL");
  }
  unsetenv("BASE_YBND");
  return 0;
}

// ═══════════════════════ T2: 몸통 궤적 동시계획 ═══════════════════════
//  ★사전등록
//   2-A TAMOLS base 스플라인 y 가 명목(y≡0)에서 **>=0.03 m** 벗어나는가
//   2-B 그 결과 모든 at_des 다리 터치다운 reach <= l_max 인가
//   2-C 기하 스냅 발판을 **명목 base(y≡0)** 로 평가하면 reach 위반이 >=1 개인가(대조군 실패)
static int cmd_t2() {
  std::printf("\n══════ 시험2: 몸통 궤적 동시계획(횡방향 이격 돌) ══════\n");
  std::printf("사전등록  2-A base y 이탈>=0.03 · 2-B 전 다리 reach<=l_max · 2-C 명목 base 로는 위반\n");
  const double lmax = 0.42, r_h = std::sqrt(lmax*lmax - BASE_H*BASE_H);
  Cfg c; c.l_max = lmax; c.iters = 80; c.y_min = 0.02; c.y_max = 0.52;
  Params prm = go2_params(lmax);
  std::printf("  기하: 수평 reach=%.3f (hip y=%.3f) → base y=0 이면 y>%.3f 인 돌은 도달불가\n",
              r_h, HIPY, HIPY + r_h);
  Scene sc;
  sc.f.s.push_back(platform(-0.60, 0.28, 0.45));
  sc.f.s.push_back(Stone{0.46,  0.47, 0.10, 0.07, STONE_TOP, "L_far"});   // 전체가 base y=0 도달권 밖(y>=0.400)
  sc.f.s.push_back(Stone{0.46, -0.30, 0.10, 0.09, STONE_TOP, "R_in"});    // L_far 가 좌측 명목의 최근접이 되도록 우측을 멀리
  double z0 = BASE_H + STONE_TOP, xb = 0.0;
  Eigen::Matrix<double,4,3> fm;
  for (int L = 0; L < 4; ++L){ fm(L,0)=prm.hip_offsets(L,0); fm(L,1)=prm.hip_offsets(L,1); fm(L,2)=sc.f.h(fm(L,0),fm(L,1)); }
  sc.build(xb);
  TamolsState nomsp = nominal_spline(prm, z0, c, fm);
  SnapOut sn = geo_snap(sc.f, nomsp, xb, c.vadv, 0.0);
  Eigen::Matrix<double,4,3> snl = to_local(sn.p, xb);
  TamolsState stn = nominal_state(prm, z0, 0.0, c, fm, snl);
  int viol_s = n_reach_viol(stn, snl, lmax);
  std::printf("\n  [기하스냅 + 명목 base(y=0)] reach 위반 = %d/4  (FL 이 고른 돌=%s)\n", viol_s, sc.f.s[sn.stone[0]].tag);
  report_legs("기하스냅", sc.f, stn, snl, xb, lmax);
  int iL=-1,iR=-1; for (size_t i=0;i<sc.f.s.size();++i){ if(!std::strcmp(sc.f.s[i].tag,"L_far")) iL=i;
                                                        if(!std::strcmp(sc.f.s[i].tag,"R_in")) iR=i; }
  int as[4] = {iL, iR, -1, -1};
  CandRes cold = solve_cand(sc, prm, c, z0, xb, fm, nullptr);
  CandRes cand = solve_cand(sc, prm, c, z0, xb, fm, as);
  // ── 결정 시험: 왼발판을 L_far 로 하드 구속(foot_y band [0.40,0.52]) 하고
  //    base y 자유(=몸통 동시계획 허용) vs base y 고정(|y|<=0.005) 을 비교 ──
  std::printf("\n  [결정시험] 왼발판을 L_far 에 하드 구속(y∈[0.40,0.52]) — base y 자유 vs 고정\n");
  for (int f = 0; f < 2; ++f) {
    if (f) setenv("BASE_YBND","0.005",1); else unsetenv("BASE_YBND");
    Cfg cf = c; cf.y_min = 0.40; cf.y_max = 0.52;
    CandRes o = solve_cand(sc, prm, cf, z0, xb, fm, as);
    double my = 0; for (int k = 0; k < o.st.num_phases(); ++k) for (int t = 0; t <= 6; ++t)
      my = std::max(my, std::fabs(o.st.pos_at(k, o.st.gait[k].duration*t/6.0)(1)));
    std::printf("   base y %-6s : ineq=%.2e  base|y|max=%.4f  reach위반=%d/4  void=%d/4  FL=(%.3f,%+.3f)\n",
                f?"고정":"자유", o.r.ineq_viol, my, o.nviol, o.nvoid, xb+o.st.p(0,0), o.st.p(0,1));
  }
  unsetenv("BASE_YBND");
  for (int q = 0; q < 2; ++q) {
    const CandRes& o = q ? cand : cold;
    double maxy = 0; for (int k = 0; k < o.st.num_phases(); ++k) for (int t = 0; t <= 6; ++t)
      maxy = std::max(maxy, std::fabs(o.st.pos_at(k, o.st.gait[k].duration*t/6.0)(1)));
    std::printf("\n  [TAMOLS %s] base|y|max=%.4f\n", q?"cand(L_far init)":"cold(명목init)", maxy);
    print_cand(q?"cand":"cold", sc, o, xb, lmax);
    std::printf("   ▶ 2-A %s(%.4f) · 2-B %s(위반 %d/4) · 2-C %s(대조군 위반 %d/4)\n",
                maxy>=0.03?"PASS":"FAIL", maxy, o.nviol==0?"PASS":"FAIL", o.nviol,
                viol_s>=1?"PASS":"FAIL", viol_s);
  }
  return 0;
}

// ═══════════════════════ T4: 돌 선택(미끼 돌) ═══════════════════════
//  ★사전등록
//   4-A TAMOLS 앞다리 발판이 **큰 돌** 위(2/2)인가 (기하 스냅은 최근접=작은 미끼)
//   4-B edge clearance(발판→돌 가장자리 최소거리)가 기하 대비 큰가
//   4-C edge_avoidance ON/OFF 로 **선택의 원인 귀속** — OFF 에서도 PASS 면 원인은 edge 항이 아님
static int cmd_t4() {
  std::printf("\n══════ 시험4: 돌 선택(가깝고 작은 미끼 vs 조금 멀고 큰 돌) ══════\n");
  std::printf("사전등록  4-A 큰 돌 2/2 · 4-B edge clearance 증가 · 4-C edge 항 귀속(ON/OFF)\n");
  const double lmax = 0.42;
  Cfg c; c.l_max = lmax; c.iters = 80; c.y_min = 0.02; c.y_max = 0.45;
  Params prm = go2_params(lmax);
  Scene sc;
  sc.f.s.push_back(platform(-0.60, 0.28, 0.45));
  sc.f.s.push_back(Stone{0.463,  HIPY, 0.035, 0.035, STONE_TOP, "SMALL_L"});   // 한 변 0.07 (미끼: 명목 자리·최근접·clearance<=0.035)
  sc.f.s.push_back(Stone{0.463, -HIPY, 0.035, 0.035, STONE_TOP, "SMALL_R"});
  sc.f.s.push_back(Stone{0.463,  0.322, 0.100, 0.100, STONE_TOP, "BIG_L"});     // 한 변 0.20 · y 0.18 이격 · clearance<=0.100
  sc.f.s.push_back(Stone{0.463, -0.322, 0.100, 0.100, STONE_TOP, "BIG_R"});
  double z0 = BASE_H + STONE_TOP, xb = 0.0;
  Eigen::Matrix<double,4,3> fm;
  for (int L = 0; L < 4; ++L){ fm(L,0)=prm.hip_offsets(L,0); fm(L,1)=prm.hip_offsets(L,1); fm(L,2)=sc.f.h(fm(L,0),fm(L,1)); }
  sc.build(xb);
  auto clear_of = [&](const Field& f, double x, double y) { int si; f.h(x,y,&si); if (si<0) return -1.0;
    return std::min(f.s[si].hx - std::fabs(x-f.s[si].cx), f.s[si].hy - std::fabs(y-f.s[si].cy)); };
  auto nbig = [&](const Field& f, const Eigen::Matrix<double,4,3>& p, double base) { int n=0;
    for (int L : {0,1}) { int si; f.h(base+p(L,0), p(L,1), &si);
      if (si>=0 && !std::strncmp(f.s[si].tag,"BIG",3)) ++n; } return n; };
  TamolsState nomsp = nominal_spline(prm, z0, c, fm);
  SnapOut sn = geo_snap(sc.f, nomsp, xb, c.vadv, 0.0);
  Eigen::Matrix<double,4,3> snl = to_local(sn.p, xb);
  TamolsState stn = nominal_state(prm, z0, 0.0, c, fm, snl);
  std::printf("\n  [기하스냅] 큰돌=%d/2  clear FL=%.3f FR=%.3f\n", nbig(sc.f, snl, xb),
              clear_of(sc.f, sn.p(0,0), sn.p(0,1)), clear_of(sc.f, sn.p(1,0), sn.p(1,1)));
  report_legs("기하스냅", sc.f, stn, snl, xb, lmax);
  int iS=-1,iB=-1,iSr=-1,iBr=-1;
  for (size_t i=0;i<sc.f.s.size();++i){ const char* t=sc.f.s[i].tag;
    if(!std::strcmp(t,"SMALL_L")) iS=i; if(!std::strcmp(t,"SMALL_R")) iSr=i;
    if(!std::strcmp(t,"BIG_L")) iB=i;   if(!std::strcmp(t,"BIG_R")) iBr=i; }
  int aS[4]={iS,iSr,-1,-1}, aB[4]={iB,iBr,-1,-1};
  // ── 기전 프로브: 돌 중심에서 벗어난 거리별 지형 신호(작은돌 vs 큰돌) ──
  std::printf("\n  [기전 프로브] 돌 중심에서 d 만큼 +x 로 벗어난 점의 지형 신호\n");
  std::printf("   %6s | %-28s | %-28s\n","d","SMALL(한변0.07)","BIG(한변0.20)");
  std::printf("   %6s | %7s %7s %7s %7s | %7s %7s %7s %7s\n","", "h_s1","|gh|","|gs1|","edge","h_s1","|gh|","|gs1|","edge");
  {
    edge_layers() = &sc.E; sc.E.w = 1.0;
    const Stone* SS = nullptr; const Stone* BB = nullptr;
    for (auto& q : sc.f.s) { if (!std::strcmp(q.tag,"SMALL_L")) SS=&q; if (!std::strcmp(q.tag,"BIG_L")) BB=&q; }
    for (double d : {0.0, 0.010, 0.020, 0.030, 0.050, 0.080}) {
      std::printf("   %6.3f |", d);
      for (const Stone* Q : {SS, BB}) {
        double x = Q->cx + d, y = Q->cy;
        double hs = bilinear_height(sc.hsol, sc.cell, sc.N, x, y);
        double gx = bilinear_height(sc.L.gh_x, sc.cell, sc.N, x, y), gy = bilinear_height(sc.L.gh_y, sc.cell, sc.N, x, y);
        double sx = bilinear_height(sc.L.gs1_x, sc.cell, sc.N, x, y), sy = bilinear_height(sc.L.gs1_y, sc.cell, sc.N, x, y);
        double e = 3*(gx*gx+gy*gy) + (sx*sx+sy*sy);
        std::printf(" %7.4f %7.3f %7.3f %7.3f |", hs, std::hypot(gx,gy), std::hypot(sx,sy), e);
      }
      std::printf("\n");
    }
    edge_layers() = nullptr;
  }
  // ★지지 유효성 층이 "돌 품질"을 이미 담고 있는가(=목적함수가 안 쓰고 있을 뿐인가)
  //   sdf(발판) = −(유효영역 경계까지 거리) = 고전적 footScore(clearance). 새 가중치 없이 값만 본다.
  if (terrain_hard_on()) {
    PinRes PS = pinned_eval(sc, prm, c, z0, xb, fm, aS), PBg = pinned_eval(sc, prm, c, z0, xb, fm, aB);
    std::printf("\n  [4-E] 지지유효 clearance(−sdf) — 층은 돌 품질을 구분하는가\n");
    std::printf("   %-10s %10s %10s | %-10s %10s %10s\n", "SMALL FL", "clr", "gap", "BIG FL", "clr", "gap");
    auto pr = [&](const PinRes& P) {
      double x = xb + P.p(0,0), y = P.p(0,1);
      std::printf(" %10.3f %10.3f", -bilinear_height(sc.SL.sdf, sc.cell, sc.N, x, y),
                  bilinear_height(sc.SL.gap_depth, sc.cell, sc.N, x, y)); };
    std::printf("   %-10s", "SMALL"); pr(PS); std::printf(" | %-10s", "BIG"); pr(PBg); std::printf("\n");
  }
  const double WS[6] = {0.0, 0.001, 0.01, 0.1, 1.0, 10.0};
  std::printf("\n  [4-C] edge_avoidance 가중 스윕 — 발판고정+명목base 목적함수(SMALL vs BIG)\n");
  std::printf("   %-8s %10s %10s %10s %10s  %s\n","w","SMALL tot","BIG tot","SMALL edge","BIG edge","선호");
  for (double w : WS) {
    sc.E.w = w; edge_layers() = w > 0 ? &sc.E : nullptr;
    PinRes PS = pinned_eval(sc, prm, c, z0, xb, fm, aS), PB = pinned_eval(sc, prm, c, z0, xb, fm, aB);
    std::printf("   %-8.3f %10.4f %10.4f %10.4f %10.4f  %s\n", w, PS.b.total, PB.b.total,
                PS.b.edge, PB.b.edge, PB.b.total < PS.b.total ? "BIG" : "SMALL");
    edge_layers() = nullptr;
  }
  for (int on = 0; on < 2; ++on) {
    sc.E.w = 1.0; edge_layers() = on ? &sc.E : nullptr;
    std::printf("\n  ── edge_avoidance %s (w=1.0, Drake 원가중) ──\n", on?"ON":"OFF");
    CandRes cold = solve_cand(sc, prm, c, z0, xb, fm, nullptr);
    CandRes S = solve_cand(sc, prm, c, z0, xb, fm, aS), B = solve_cand(sc, prm, c, z0, xb, fm, aB);
    print_cand("cold(명목init)", sc, cold, xb, lmax);
    print_cand("cand SMALL", sc, S, xb, lmax);
    print_cand("cand BIG",   sc, B, xb, lmax);
    std::printf("   clear: cold FL=%.3f FR=%.3f | SMALL FL=%.3f | BIG FL=%.3f\n",
                clear_of(sc.f, xb+cold.st.p(0,0), cold.st.p(0,1)), clear_of(sc.f, xb+cold.st.p(1,0), cold.st.p(1,1)),
                clear_of(sc.f, xb+S.st.p(0,0), S.st.p(0,1)), clear_of(sc.f, xb+B.st.p(0,0), B.st.p(0,1)));
    std::printf("   ▶ 4-A cold 큰돌 %d/2 %s · 1-D식 목적함수: SMALL %.4f vs BIG %.4f → %s\n",
                nbig(sc.f, cold.st.p, xb), nbig(sc.f, cold.st.p, xb)>=2?"PASS":"FAIL",
                S.b.total, B.b.total, B.b.total < S.b.total ? "BIG 선호" : "SMALL 선호");
    edge_layers() = nullptr;
  }
  return 0;
}


// ═══════════ TH: 지지 유효성 판별식 자체 검증(부양·일반성) ═══════════
//  ★사전등록
//   H-1 stepping: 돌 위 gap_depth≈0 · 보이드 gap_depth≈gap 깊이(0.15) — 판별이 된다
//   H-2 계단(단조): 모든 tread 에서 gap_depth≈0 — **오검출 없음**(closing=항등)
//   H-3 트렌치(갭): 홈 안 gap_depth≈깊이 — 손코딩 gap_lo/gap_hi 를 대체할 수 있다
//   H-4 sdf 기울기: 보이드 내부(∇h=0)에서도 |∇sdf|≈1 — SQP 탈출 방향이 있다
static int cmd_th() {
  std::printf("\n══════ TH: 지지 유효성 판별식(형태학적 닫힘) 검증 ══════\n");
  std::printf("사전등록  H-1 stepping 판별 · H-2 계단 오검출 0 · H-3 트렌치 판별 · H-4 보이드 내부 기울기\n");
  setenv("TERRAIN_HARD", "1", 1);
  std::printf("  파라미터: closing 반경 R=%.2f m · dz_max=%.3f m · band=%.3f m · margin=%.3f m\n",
              th_rad(), th_dz(), th_band(), th_margin());
  auto dump = [&](const char* tag, Scene& sc, double y, const double* xs, int nx) {
    std::printf("  ── %s ──  유효셀 %d/%d (%.1f%%)\n", tag, sc.SL.n_valid, sc.SL.n_cell,
                100.0 * sc.SL.n_valid / sc.SL.n_cell);
    std::printf("   %-10s", "x"); for (int i=0;i<nx;++i) std::printf(" %7.2f", xs[i]); std::printf("\n");
    const char* nm[4] = {"h(정확)","h_close","gap_depth","sdf"};
    for (int q = 0; q < 4; ++q) { std::printf("   %-10s", nm[q]);
      for (int i = 0; i < nx; ++i) {
        double v = 0;
        if (q==0) v = sc.f.h(xs[i], y);
        else if (q==1) v = bilinear_height(sc.SL.h_close, sc.cell, sc.N, xs[i], y);
        else if (q==2) v = bilinear_height(sc.SL.gap_depth, sc.cell, sc.N, xs[i], y);
        else v = bilinear_height(sc.SL.sdf, sc.cell, sc.N, xs[i], y);
        std::printf(" %7.3f", v); }
      std::printf("\n"); }
    std::printf("   %-10s", "|∇sdf|");
    for (int i = 0; i < nx; ++i) { double gx, gy;
      bilinear_grad(sc.SL.sdf, sc.cell, sc.N, xs[i], y, gx, gy);
      std::printf(" %7.3f", std::hypot(gx, gy)); }
    std::printf("\n");
  };
  // H-1 stepping (t3/t3m 지형)
  { Scene sc;
    sc.f.s.push_back(platform(-0.60, 0.28, 0.45));
    sc.f.s.push_back(Stone{0.55,  HIPY, 0.17, 0.10, STONE_TOP, "LONG_L"});
    sc.f.s.push_back(Stone{0.55, -HIPY, 0.17, 0.10, STONE_TOP, "LONG_R"});
    sc.f.s.push_back(Stone{1.05,  HIPY, 0.08, 0.10, STONE_TOP, "NEXT_L"});
    sc.f.s.push_back(Stone{1.05, -HIPY, 0.08, 0.10, STONE_TOP, "NEXT_R"});
    sc.build(0.0);
    const double xs[9] = {0.10, 0.33, 0.45, 0.55, 0.68, 0.80, 0.85, 1.05, 1.18};
    dump("H-1 stepping (y=+0.142)", sc, HIPY, xs, 9); }
  // H-2 계단(단조) — tread 0.30 · riser 0.15
  { Scene sc;
    sc.f.s.push_back(Stone{0.30, 0.0, 0.90, 0.60, 0.15, "s1"});   // x∈[-0.60,1.20]
    sc.f.s.push_back(Stone{0.60, 0.0, 0.60, 0.60, 0.30, "s2"});   // x∈[ 0.00,1.20]
    sc.f.s.push_back(Stone{0.75, 0.0, 0.45, 0.60, 0.45, "s3"});   // x∈[ 0.30,1.20]
    sc.f.s.push_back(Stone{0.90, 0.0, 0.30, 0.60, 0.60, "s4"});   // x∈[ 0.60,1.20]
    sc.build(0.0);
    const double xs[9] = {-0.40, -0.10, 0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00};
    dump("H-2 계단 단조(y=0)", sc, 0.0, xs, 9); }
  // H-3 트렌치(갭)
  { Scene sc;
    sc.f.s.push_back(Stone{-0.10, 0.0, 0.55, 0.60, 0.15, "near"});  // x∈[-0.65,0.45]
    sc.f.s.push_back(Stone{ 0.95, 0.0, 0.25, 0.60, 0.15, "far"});   // x∈[ 0.70,1.20]
    sc.build(0.0);
    const double xs[9] = {-0.30, 0.20, 0.40, 0.50, 0.575, 0.65, 0.75, 0.95, 1.10};
    dump("H-3 트렌치 0.25m·깊이0.15(y=0)", sc, 0.0, xs, 9); }
  unsetenv("TERRAIN_HARD");
  support_layers() = nullptr;
  return 0;
}


// ═══════════ TH2: 사전등록① "부양 정지" 단독 진단 (t3m 의 x_target 진단 발췌) ═══════════
//   판정: 발판 p.z 가 **실제(해석) 지형 높이와 일치**하는가. 기준선(게이트 OFF) = x=1.40 서 p.z=0.095·h=0.000.
//   TAMCAP_ITERS 로 SQP 반복 상한을 줄여 빠르게 볼 수 있다(기본 80).
static int cmd_th2() {
  std::printf("\n══════ TH2: 부양 정지(사전등록①) ══════\n");
  std::printf("  기준선(OFF): N=2·x_target=1.40 에서 발판 p.z=0.095 인데 정확 지형 h=0.000 (★공중부양)\n");
  Cfg c; c.l_max = L_MAX; c.iters = getenv("TAMCAP_ITERS") ? atoi(getenv("TAMCAP_ITERS")) : 80;
  c.vadv = 0.45; c.dur = 0.20;
  Params prm = go2_params(L_MAX);
  double z0 = BASE_H + STONE_TOP, xb = 0.0;
  setenv("STANCE_REACH", "1", 1);
  for (int variant = 0; variant < 2; ++variant) {
    Scene sv;
    sv.f.s.push_back(platform(-0.60, 0.28, 0.45));
    double hx = variant ? 0.17 : 0.085, cxs = variant ? 0.55 : 0.465;
    sv.f.s.push_back(Stone{cxs,  HIPY, hx, 0.10, STONE_TOP, variant?"LONG_L":"SHRT_L"});
    sv.f.s.push_back(Stone{cxs, -HIPY, hx, 0.10, STONE_TOP, variant?"LONG_R":"SHRT_R"});
    sv.f.s.push_back(Stone{1.05,  HIPY, 0.08, 0.10, STONE_TOP, "NEXT_L"});
    sv.f.s.push_back(Stone{1.05, -HIPY, 0.08, 0.10, STONE_TOP, "NEXT_R"});
    sv.build(xb);
    Eigen::Matrix<double,4,3> fmv;
    for (int L = 0; L < 4; ++L){ fmv(L,0)=prm.hip_offsets(L,0); fmv(L,1)=prm.hip_offsets(L,1); fmv(L,2)=sv.f.h(fmv(L,0),fmv(L,1)); }
    std::printf("\n  ── LONG %s ([%.2f,%.2f]) · 지형하드 %s ──\n", variant?"full ":"short", cxs-hx, cxs+hx,
                terrain_hard_on() ? "ON" : "OFF");
    for (double XT : {0.55, 0.648, 0.75, 1.40}) {
      auto t0 = std::chrono::high_resolution_clock::now();
      TamolsState qd; QpResult rd = solve_multi(qd, sv, prm, c, z0, 0.0, fmv, 2, XT);
      double ms = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
      int nfloat = 0, nvoid = 0;
      std::printf("   x_target=%.3f  it=%2d ineq=%.0e  (%.0f ms)\n", XT, rd.iters, rd.ineq_viol, ms);
      for (int n = 0; n < 2; ++n) { std::printf("     사이클%d ", n);
        for (int L = 0; L < 4; ++L) { Vector3d fp = qd.fpos(n, L);
          double hx2 = sv.f.h(fp(0), fp(1)); int si; sv.f.h(fp(0), fp(1), &si);
          bool fl = std::fabs(fp(2)-hx2) > 0.03; nfloat += fl; nvoid += (si<0);
          std::printf(" %s(%.3f,%+.3f) z=%.3f h=%.3f%s%s", LEGN[L], fp(0), fp(1), fp(2), hx2,
                      fl?"★공중":"", si<0?"[VOID]":""); }
        std::printf("\n"); terr_line("     지형", sv, qd, n); }
      bool feas = (rd.ineq_viol <= 1e-4);
      std::printf("     ▶ 공중부양 %d/8 · VOID %d/8 → %s\n", nfloat, nvoid,
                  !feas ? "판정보류(요구가 infeasible = 해가 수렴 안 함)" : (nfloat==0 ? "부양 정지 PASS" : "★부양 지속 FAIL"));
    }
  }
  unsetenv("STANCE_REACH"); support_layers() = nullptr;
  return 0;
}

// ─────────────────── 자코비안 정합 검사 ───────────────────
static int cmd_jac();
static int cmd_t1(); static int cmd_t2(); static int cmd_t3(); static int cmd_t4(); static int cmd_t3s(); static int cmd_t3m(); static int cmd_cost(); static int cmd_th(); static int cmd_th2();

int main(int argc, char** argv) {
  std::string c = argc > 1 ? argv[1] : "jac";
  if (c == "jac") return cmd_jac();
  if (c == "t1")  return cmd_t1();
  if (c == "t2")  return cmd_t2();
  if (c == "t3")  return cmd_t3();
  if (c == "t3s") return cmd_t3s();
  if (c == "t3m") return cmd_t3m();
  if (c == "cost") return cmd_cost();
  if (c == "t4")  return cmd_t4();
  if (c == "th")  return cmd_th();
  if (c == "th2") return cmd_th2();
  std::printf("usage: test_capability {jac|t1|t2|t3|t3s|t3m|cost|t4|th|th2}\n"); return 1;
}
