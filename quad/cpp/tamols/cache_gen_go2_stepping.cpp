// cache_gen_go2_stepping.cpp — Go2 varied-height stepping-stone field: TAMOLS stone-SELECTION plans.
//   입력: stepping_go2/stones_L{0..9}.csv (gen_stones.py, env _build_stepping_terrain_curriculum
//         GO2_STONE_HVAR=0.05 정확 복제: lane-local x[0.75,4.25]·y±0.9·stone top z=0.15±amp).
//   방식: 코리도를 따라 receding-horizon 스윕 — 매 사이클 base 앵커 주변 heightmap 창을 래스터라이즈
//         (raw = 정확한 돌/보이드, solve용 = gaussian σ(TAMOLS h_s1 역할: 보이드에 gradient basin)),
//         walk 정식화(4-phase 한발씩=GIAC 3발지지)로 solve_fast → 다리당 1개 발판. 발판을 돌 테이블에
//         대조해 "선택한 돌"을 추출, per-leg 돌 시퀀스로 dedup. 출력은 SPATIAL(돌 선택), 시간 아님.
//   돌선택 메커니즘: foothold_on_ground(∇h 따라 xy가 높은 지형=돌로) + kinematic reach(l_max=0.45:
//         보이드 바닥 z=0은 hip에서 0.49m=도달불가 → 발판이 돌 위로 강제) + nominal(hip 아래 h_des).
//   빌드: PIX=/home/jsh/simple-mpc/.pixi/envs/default; g++ -O3 -std=c++17 cache_gen_go2_stepping.cpp \
//         -I/usr/include/eigen3 -I$PIX/include -L$PIX/lib -Wl,-rpath,$PIX/lib -leiquadprog -o cache_gen_go2_stepping
//   실행: ./cache_gen_go2_stepping [outdir=stepping_go2] [level|all]   (env: TAM_SIGMA=셀단위 σ, STEP_VADV)
#include "tamols_online.hpp"
#include "terrain_proc.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
using namespace tamols;

// ── 돌 필드(레벨당): stones_L*.csv 로드. 격자 규칙적(cx per ix, cy per iy) ──
struct StoneField {
  int level = 0, nx = 0, ny = 0;
  double size = 0, pitch = 0;
  std::vector<double> cxs, cys;      // ix→cx, iy→cy (lane-local)
  std::vector<double> top;           // ix*ny+iy → stone top z
  double cx_last() const { return cxs.back(); }
};

static bool load_field(const std::string& path, StoneField& f) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line; std::getline(in, line);            // header
  std::vector<std::array<double,7>> rows;
  int nx = 0, ny = 0;
  while (std::getline(in, line)) {
    std::array<double,7> r;
    if (std::sscanf(line.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                    &r[0],&r[1],&r[2],&r[3],&r[4],&r[5],&r[6]) != 7) continue;
    rows.push_back(r);
    nx = std::max(nx, (int)r[1] + 1); ny = std::max(ny, (int)r[2] + 1);
  }
  f.nx = nx; f.ny = ny; f.size = rows[0][5];
  f.cxs.assign(nx, 0); f.cys.assign(ny, 0); f.top.assign(nx * ny, 0);
  for (auto& r : rows) {
    int ix = (int)r[1], iy = (int)r[2];
    f.cxs[ix] = r[3]; f.cys[iy] = r[4]; f.top[ix * ny + iy] = r[6];
  }
  f.pitch = nx > 1 ? f.cxs[1] - f.cxs[0] : f.size;
  return true;
}

// ── 지형 높이(lane-local, 정확): spawn strip [-0.75,0.75]×|y|≤1 top 0.15 · 돌 top · 그외 평면 0 ──
//   tol>0 = 분류용 완화(1셀). sidx: 밟은 돌 idx(-1 strip, -2 void/plane)
static double field_h(const StoneField& f, double x, double y, double tol, int* sidx) {
  if (sidx) *sidx = -2;
  if (x >= -0.75 - tol && x <= 0.75 + tol && std::fabs(y) <= 1.0 + tol) { if (sidx) *sidx = -1; return 0.15; }
  double hs = 0.5 * f.size;
  int ix = (int)std::lround((x - f.cxs[0]) / f.pitch);
  int iy = (int)std::lround((y - f.cys[0]) / f.pitch);
  if (ix >= 0 && ix < f.nx && iy >= 0 && iy < f.ny &&
      std::fabs(x - f.cxs[ix]) <= hs + tol && std::fabs(y - f.cys[iy]) <= hs + tol) {
    if (sidx) *sidx = ix * f.ny + iy;
    return f.top[ix * f.ny + iy];
  }
  return 0.0;
}

// 최근접 돌(유클리드 중심거리) — naive snap 기준(B'와 동일 규칙)
static int nearest_stone(const StoneField& f, double x, double y) {
  int best = -1; double bd = 1e18;
  for (int ix = 0; ix < f.nx; ++ix) for (int iy = 0; iy < f.ny; ++iy) {
    double dx = x - f.cxs[ix], dy = y - f.cys[iy], d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = ix * f.ny + iy; }
  }
  return best;
}

// ── 로컬 heightmap 창 래스터(창 중심 = (xc, 0), 노드 = a*cell-off) ──
static void raster(const StoneField& f, double xc, int N, double cell, Grid& h) {
  double off = cell * N / 2.0;
  h.resize(N, N);
  for (int a = 0; a < N; ++a) for (int b = 0; b < N; ++b)
    h(a, b) = field_h(f, xc + a * cell - off, b * cell - off, 0.0, nullptr);
}

// ── walk 게이트(accumulated at_des): set_walk_gait와 동일 LS crawl(RR→FR→RL→FL)이되
//    at_des를 trot 패턴처럼 스윙 시점부터 누적(스윙중+이후=1). 효과: ①터미널 nominal_kinematic이
//    4발 전부의 발판을 구동(원판은 마지막 phase 스윙 다리=FL만 → 나머지 3발 xy 무구동=보이드 방치)
//    ②GIAC이 스윙 완료 발을 새 발판 p로 봄(정확한 지지 다각형). 공유 헤더는 미수정(로컬 정의). ──
static void set_walk_gait_acc(TamolsState& st, double phase_dur) {
  int P = 4; st.gait.resize(P);
  int cs[4][4] = {{1,1,1,0},{1,0,1,1},{1,1,0,1},{0,1,1,1}};   // 스윙: RR,FR,RL,FL
  int ad[4][4] = {{0,0,0,1},{0,1,0,1},{0,1,1,1},{1,1,1,1}};   // 스윙부터 누적
  for (int k = 0; k < P; ++k) { st.gait[k].duration = phase_dur;
    for (int i = 0; i < 4; ++i) { st.gait[k].contact[i] = cs[k][i]; st.gait[k].at_des[i] = ad[k][i]; } }
}

// ── stepping replan: online_replan의 walk판 로컬 복제(사유: y밴드·z밴드·l_max를 돌필드에 맞춤 —
//    OnlineCfg엔 y밴드 노출이 없고 L9 돌 행이 y=0/±0.28이라 기본 y[0.10,0.22]엔 돌이 없음) ──
struct StepCfg { double vadv = 0.3, phase_dur = 0.2; int iters = 60;
                 double y_min = 0.04, y_max = 0.34; };
static QpResult stepping_replan(TamolsState& st, const Grid& h, double cell, int ms, double z0,
                                double vx0, const Eigen::Matrix<double,4,3>& foot_meas,
                                const StepCfg& c, const Eigen::Matrix<double,4,3>* p_init) {
  set_walk_gait_acc(st, c.phase_dur);                  // walk 4-phase(RR→FR→RL→FL), at_des 누적
  int P = st.num_phases(); double T = P * c.phase_dur, xf = c.vadv * T;
  st.base_pose << 0, 0, z0, 0, 0, 0;
  st.base_vel  << vx0, 0, 0, 0, 0, 0;
  st.p_meas = foot_meas;
  st.ref_vel = Vector3d(c.vadv, 0, 0);
  // cold init(항상): Hermite 전진 램프(online_replan 정합) + 발판 init
  st.a.assign(P, MatrixXd::Zero(6, 4));
  double c1 = vx0, c3 = (c.vadv - vx0 - 2*(xf - vx0*T)/T)/(T*T), c2 = (xf - vx0*T - c3*T*T*T)/(T*T);
  auto xg = [&](double t){ return c1*t + c2*t*t + c3*t*t*t; };
  auto vg = [&](double t){ return c1 + 2*c2*t + 3*c3*t*t; };
  for (int k = 0; k < P; ++k) {
    st.a[k].col(0) = st.base_pose; st.a[k](2, 0) = z0;
    double t0 = k*c.phase_dur, x0 = xg(t0), x1 = xg(t0+c.phase_dur), v0 = vg(t0), v1 = vg(t0+c.phase_dur);
    st.a[k](0,0) = x0; st.a[k](0,1) = v0;
    st.a[k](0,2) = (3*(x1-x0)/c.phase_dur - 2*v0 - v1)/c.phase_dur;
    st.a[k](0,3) = (2*(x0-x1)/c.phase_dur + v0 + v1)/(c.phase_dur*c.phase_dur);
  }
  if (p_init) st.p = *p_init;
  else for (int i = 0; i < 4; ++i) {                   // 명목: hip + 0.5·xf(터미널 mid-phase 정합)
    st.p(i,0) = st.prm.hip_offsets(i,0) + 0.5*xf; st.p(i,1) = st.prm.hip_offsets(i,1);
    st.p(i,2) = bilinear_height(h, cell, ms, st.p(i,0), st.p(i,1));
  }
  st.epsilon = VectorXd::Zero(P);
  QpOptions o; o.max_iter = c.iters;
  o.zlo = z0 - 0.06; o.zhi = z0 + 0.06; o.rp_max = 0.20; o.yaw_max = 0.10;   // z0_terrain 컨벤션
  o.x_target = xf * 0.9;
  o.y_min = c.y_min; o.y_max = c.y_max;                // 돌 행(y=0·±pitch) 도달 위해 확장
  o.gap = false;
  return solve_fast(st, h, cell, ms, o);
}

// ── 발자국 기록 ──
struct Footfall {
  int cycle, leg;                    // leg: 0FL 1FR 2RL 3RR (planner)
  double fx, fy, fz;                 // 발판(lane-local, z=solver)
  double fz_snap;                    // 정확 지형 z(돌 top/strip 0.15/void 0)
  int support;                       // 돌 idx / -1 strip / -2 void
  int naive;                         // naive 최근접 돌(터치다운 명목점 기준), 필드 밖=-1
  double nom_x, nom_y;               // 터치다운 명목점(per-leg, base(swing끝)+hip)
  bool solver_ok, init_snap;         // init_snap: 스냅 init 재시도로 얻은 해
};
static const char* LEGN[4] = {"FL", "FR", "HL", "HR"};   // 출력명(RL→HL, RR→HR)
static const int SWING_ORDER[4] = {3, 1, 2, 0};          // walk 위상순: RR,FR,RL,FL
static const int SWING_PHASE[4] = {3, 1, 2, 0};          // leg → swing phase idx (FL=3,FR=1,RL=2,RR=0)

int main(int argc, char** argv) {
  // ★2026-09: GIAC_FIX(중력항)·GIAC_ORDER(CW 인접쌍)가 솔버 기본 ON으로 바뀜.
  //   이 오프라인 캐시 생성기의 출력은 이미 배포된 아티팩트라 **레거시 동작으로 고정**한다.
  //   (GIAC_FIX=1 로 실행하려면 이 두 줄을 지울 것)
  if (!getenv("TAMOLS_ALLOW_GIAC")) { setenv("GIAC_FIX", "0", 1); setenv("GIAC_ORDER", "0", 1); }
  // 결정적 동작: 솔버 경로를 바꾸는 env 전부 클리어(해석 jacobian 정합 포함)
  for (const char* e : {"COM_W","COM_WX","COM_LEAD","GIAC_FIX","W_EPS","EPS_MAX","BASE_YBND",
                        "WALK_QLEG","TAM_WNOM","TAM_PRM_MODEL"}) unsetenv(e);
  std::string outdir = argc > 1 ? argv[1] : "stepping_go2";
  int only_level = (argc > 2 && std::string(argv[2]) != "all") ? std::atoi(argv[2]) : -1;
  double sigma = getenv("TAM_SIGMA") ? atof(getenv("TAM_SIGMA")) : 2.0;   // 셀단위(h_s1 역할)
  StepCfg sc; sc.vadv = getenv("STEP_VADV") ? atof(getenv("STEP_VADV")) : 0.3;

  const int N = 101; const double cell = 0.02;         // 창 ±1.01m: 레인폭 전체 + 발 reach 커버
  const double base_h = 0.34, X_END = 4.25;
  const double TOL = 0.02;                             // on-stone 허용(1셀)
  const int MAX_CYC = 40;

  Params prm;                                          // ★Go2 물리(기존 go2 캐시와 달리 h_des도 교정:
  prm.hip_offsets <<                                   //   기본 0.52면 발판 z명목=base-0.52=-0.03 → 보이드 선호로 반전!)
       0.1934,  0.142, 0.0,
       0.1934, -0.142, 0.0,
      -0.1934,  0.142, 0.0,
      -0.1934, -0.142, 0.0;
  prm.mass = 15.0; prm.h_des = base_h; prm.nominal_height = base_h;
  prm.foot_radius = 0.022;
  prm.l_min = 0.10; prm.l_max = 0.45;                  // Go2 다리 최대신장 ~0.43+: 보이드 바닥(0.49)=도달불가

  std::ostringstream meta_lv;
  std::printf("=== Go2 stepping-stone TAMOLS 스톤선택 플랜 (walk·vadv=%.2f·σ=%.1f셀·cell=%.2f) ===\n",
              sc.vadv, sigma, cell);
  std::printf("%3s %5s %5s | %6s %6s %6s | %7s %7s | %5s %5s | %6s %6s | %5s %5s %6s\n",
              "lvl","size","gap","솔브","실패","스냅릿","필드발","돌위%","ΔzMax","cov","다른돌","다른%","selfΔ","끝x","앵커x");

  for (int lvl = 0; lvl < 10; ++lvl) {
    if (only_level >= 0 && lvl != only_level) continue;
    StoneField f; f.level = lvl;
    if (!load_field(outdir + "/stones_L" + std::to_string(lvl) + ".csv", f)) {
      std::printf("L%d: stones csv 로드 실패\n", lvl); return 1; }

    double xb = 0.0, vx0 = 0.0;                        // 앵커 = spawn(스트립 중앙)
    TamolsState st; st.prm = prm;
    Eigen::Matrix<double,4,3> fm;                      // 로컬(앵커 기준) 측정 발위치
    for (int l = 0; l < 4; ++l) { fm(l,0) = prm.hip_offsets(l,0); fm(l,1) = prm.hip_offsets(l,1);
      fm(l,2) = field_h(f, xb + fm(l,0), fm(l,1), 0, nullptr); }

    std::vector<Footfall> ffs;
    int nsolve = 0, nfail = 0, nsnapinit = 0; long it_sum = 0;
    Grid hraw, hsol;
    double stop_x = f.cx_last() - 0.25;                // 앞발 명목이 마지막 돌기둥을 넘지 않게

    for (int cyc = 0; cyc < MAX_CYC && xb < stop_x; ++cyc) {
      raster(f, xb, N, cell, hraw);
      hsol = sigma > 0 ? gaussian_filter(hraw, sigma) : hraw;
      // 발밑 기준 ground = 앵커 주변 "보행 표면"(돌 top 평균·스트립 0.15). 발측정 z를 쓰면
      // 보이드에 빠진 발이 base를 끌어내려 l_max가 보이드 바닥을 도달가능하게 만드는 악순환.
      double gsum = 0; int gn = 0;
      for (int ix = 0; ix < f.nx; ++ix) if (std::fabs(f.cxs[ix] - xb) < 0.45)
        for (int iy = 0; iy < f.ny; ++iy) { gsum += f.top[ix * f.ny + iy]; ++gn; }
      if (xb < 0.75 + 0.45) { gsum += 0.15 * std::max(1, gn); gn += std::max(1, gn); }  // 스트립 겹침=0.15 가중
      double ground = gn ? gsum / gn : 0.15;
      double z0 = base_h + ground;

      // 후보 평가: in-field 발 중 보이드에 놓인 수(적을수록 좋음)
      auto n_off = [&](const TamolsState& s) {
        int off = 0;
        for (int l = 0; l < 4; ++l) {
          double fx = xb + s.p(l,0);
          if (fx <= 0.75 + TOL || fx >= f.cx_last() + 0.5*f.size + TOL) continue;
          int si; field_h(f, fx, s.p(l,1), TOL, &si);
          if (si == -2) ++off;
        }
        return off;
      };
      // 멀티스타트(발판 init × 스텝길이): 시도 A=명목 init·기본 vadv(순수 TO). 발이 보이드에 남으면
      //   ①스냅 init(최근접 돌; TO가 기각/이동 가능=여전히 TO 선택) ②vadv 변주(=사이클 전진량 변주:
      //   0.28m 돌기둥 피치 vs 고정 0.216m 전진의 비정합 해소 — 스텝길이는 TO 결정변수가 아니라
      //   외곽 탐색으로 보완). 채택 순위: ①feasible ②on-stone 多 ③viol 小.
      struct Cand { TamolsState st; QpResult r; bool ok; int off; double vadv; bool snap; };
      auto try_cand = [&](double va, bool use_snap) {
        Cand cd; cd.vadv = va; cd.snap = use_snap;
        cd.st.prm = prm;
        StepCfg scv = sc; scv.vadv = va;
        Eigen::Matrix<double,4,3> pini;
        if (use_snap) {
          double xf = va * 4 * sc.phase_dur;
          for (int l = 0; l < 4; ++l) {
            double nx = xb + prm.hip_offsets(l,0) + 0.5*xf, ny = prm.hip_offsets(l,1);
            int ns = nearest_stone(f, nx, ny);
            if (nx < 0.75) { pini(l,0) = nx - xb; pini(l,1) = ny; }   // 스트립 위=명목 유지
            else { pini(l,0) = f.cxs[ns / f.ny] - xb; pini(l,1) = f.cys[ns % f.ny]; }
            // y밴드 클램프(row0 중심 y=0은 y_min 위반 → 밴드 안 가장 가까운 y로)
            if (l == 0 || l == 2) pini(l,1) = std::max(sc.y_min, std::min(sc.y_max, pini(l,1)));
            else                  pini(l,1) = std::min(-sc.y_min, std::max(-sc.y_max, pini(l,1)));
            pini(l,2) = bilinear_height(hsol, cell, N, pini(l,0), pini(l,1));
          }
        }
        cd.r = stepping_replan(cd.st, hsol, cell, N, z0, vx0, fm, scv, use_snap ? &pini : nullptr);
        ++nsolve; it_sum += cd.r.iters;
        cd.ok = (cd.r.eq_viol < 1e-2 && cd.r.ineq_viol < 1e-2);
        cd.off = n_off(cd.st);
        return cd;
      };
      auto better = [](const Cand& a, const Cand& b) {          // a가 b보다 나은가
        if (a.ok != b.ok) return a.ok;
        if (a.off != b.off) return a.off < b.off;
        return a.r.eq_viol + a.r.ineq_viol < b.r.eq_viol + b.r.ineq_viol;
      };
      Cand best = try_cand(sc.vadv, false);
      if (!best.ok || best.off > 0) {
        const double va_list[5] = {sc.vadv, 0.8 * sc.vadv, 1.2 * sc.vadv, 0.65 * sc.vadv, 1.35 * sc.vadv};
        for (double va : va_list) {
          Cand c2 = try_cand(va, true);
          if (better(c2, best)) best = c2;
          if (best.ok && best.off == 0) break;
        }
      }
      st = best.st; QpResult r = best.r; bool ok = best.ok;
      double vadv_used = best.vadv;
      if (!ok) ++nfail;
      if (best.snap) ++nsnapinit;

      double T = 4 * sc.phase_dur;
      for (int oi = 0; oi < 4; ++oi) {                 // 위상 스윙 순서로 기록
        int l = SWING_ORDER[oi];
        Footfall ff; ff.cycle = cyc; ff.leg = l;
        ff.fx = xb + st.p(l,0); ff.fy = st.p(l,1); ff.fz = st.p(l,2);
        ff.fz_snap = field_h(f, ff.fx, ff.fy, TOL, &ff.support);
        // per-leg 터치다운 명목점: base(자기 swing phase 끝) + hip → naive 최근접 돌
        int kp = SWING_PHASE[l];
        Vector6d bp = st.pos_at(kp, sc.phase_dur);
        ff.nom_x = xb + bp(0) + prm.hip_offsets(l,0);
        ff.nom_y = bp(1) + prm.hip_offsets(l,1);
        ff.naive = (ff.nom_x > 0.75 + TOL) ? nearest_stone(f, ff.nom_x, ff.nom_y) : -1;
        ff.solver_ok = ok; ff.init_snap = best.snap;
        ffs.push_back(ff);
      }
      // 체이닝: 앵커 전진=계획 base 끝 x(클램프, 채택된 vadv 기준), 발측정=스냅 z(물리)
      double xf = vadv_used * T;
      double dx = st.pos_at(3, sc.phase_dur)(0);
      dx = std::max(0.45 * xf, std::min(1.25 * xf, dx));
      double vxe = st.vel_at(3, sc.phase_dur)(0);
      vx0 = std::max(0.0, std::min(sc.vadv, vxe));
      xb += dx;
      for (int oi = 0; oi < 4; ++oi) {
        const Footfall& ff = ffs[ffs.size() - 4 + oi];
        fm(ff.leg, 0) = ff.fx - xb; fm(ff.leg, 1) = ff.fy;
        fm(ff.leg, 2) = (ff.support >= -1) ? ff.fz_snap : 0.0;     // 보이드=바닥 0(정직)
      }
    }

    // ── 통계 (a)-(e) ──
    int nfield = 0, nonstone = 0, nboth = 0, ndiff = 0;
    double max_dz_leg = 0, max_foot_x = 0; int max_ix_front = -1, max_ix_all = -1;
    int last_stone[4] = {-1,-1,-1,-1};
    std::vector<std::array<int,3>> plan_rows;          // (ff순번, leg, stone)
    for (size_t i = 0; i < ffs.size(); ++i) {
      const Footfall& ff = ffs[i];
      bool in_field = ff.fx > 0.75 + TOL && ff.fx < f.cx_last() + 0.5 * f.size + TOL;
      if (in_field) { ++nfield; if (ff.support >= 0) ++nonstone; }
      if (ff.support >= 0 && ff.naive >= 0) { ++nboth; if (ff.support != ff.naive) ++ndiff; }
      max_foot_x = std::max(max_foot_x, ff.fx);
      if (ff.support >= 0) {
        int ix = ff.support / f.ny;
        max_ix_all = std::max(max_ix_all, ix);
        if (ff.leg <= 1) max_ix_front = std::max(max_ix_front, ix);
        if (last_stone[ff.leg] >= 0 && last_stone[ff.leg] != ff.support)
          max_dz_leg = std::max(max_dz_leg, std::fabs(f.top[ff.support] - f.top[last_stone[ff.leg]]));
        if (last_stone[ff.leg] != ff.support) plan_rows.push_back({(int)i, ff.leg, ff.support});
        last_stone[ff.leg] = ff.support;
      }
    }
    double pct = nfield ? 100.0 * nonstone / nfield : 0;
    double pctd = nboth ? 100.0 * ndiff / nboth : 0;
    bool cov = (max_ix_front == f.nx - 1);

    // ── 출력 파일 ──
    { std::ofstream pf(outdir + "/plan_L" + std::to_string(lvl) + ".csv"); pf.precision(10);
      pf << "order,cycle,leg,stone_idx,stone_cx,stone_cy,stone_top_z,foot_x,foot_y,foot_z,naive_idx,differs\n";
      int ord = 0;
      for (auto& pr : plan_rows) {
        const Footfall& ff = ffs[pr[0]]; int s = pr[2];
        pf << ord++ << "," << ff.cycle << "," << LEGN[pr[1]] << "," << s << ","
           << f.cxs[s / f.ny] << "," << f.cys[s % f.ny] << "," << f.top[s] << ","
           << ff.fx << "," << ff.fy << "," << ff.fz_snap << ","
           << ff.naive << "," << (ff.naive >= 0 && s != ff.naive ? 1 : 0) << "\n";
      } }
    { std::ofstream rf(outdir + "/footfalls_L" + std::to_string(lvl) + ".csv"); rf.precision(10);
      rf << "cycle,leg,foot_x,foot_y,foot_z_solver,foot_z_snap,support_idx,naive_idx,nom_x,nom_y,solver_ok,init_snap\n";
      for (auto& ff : ffs)
        rf << ff.cycle << "," << LEGN[ff.leg] << "," << ff.fx << "," << ff.fy << "," << ff.fz << ","
           << ff.fz_snap << "," << ff.support << "," << ff.naive << "," << ff.nom_x << "," << ff.nom_y
           << "," << (ff.solver_ok?1:0) << "," << (ff.init_snap?1:0) << "\n"; }

    meta_lv << (meta_lv.tellp() > 0 ? ",\n" : "")
            << "  {\"level\": " << lvl << ", \"n_solves\": " << nsolve << ", \"n_fail\": " << nfail
            << ", \"n_snap_init\": " << nsnapinit << ", \"mean_iters\": " << (nsolve ? (double)it_sum/nsolve : 0)
            << ", \"n_field_footfalls\": " << nfield << ", \"on_stone_pct\": " << pct
            << ", \"max_adj_dz_chosen\": " << max_dz_leg << ", \"coverage_front_last_col\": " << (cov?"true":"false")
            << ", \"max_foot_x\": " << max_foot_x << ", \"n_both_stone\": " << nboth
            << ", \"n_diff_vs_naive\": " << ndiff << ", \"diff_pct\": " << pctd << "}";
    std::printf("%3d %5.2f %5.2f | %6d %6d %6d | %7d %6.1f%% | %5.3f %5s | %6d %5.1f%% | %5.3f %5.2f\n",
                lvl, f.size, f.pitch - f.size, nsolve, nfail, nsnapinit, nfield, pct,
                max_dz_leg, cov ? "yes" : "NO", ndiff, pctd, max_foot_x, xb);
  }

  std::ofstream mj(outdir + "/plan_meta.json");
  mj << "{\n  \"kind\": \"stepping_stone_plan\", \"model\": \"go2\", \"gait\": \"walk(RR,FR,RL,FL)\",\n";
  mj << "  \"frame\": \"lane-local: x=corridor coord(spawn x=0, strip[-0.75,0.75] top z=0.15, stones[0.75,4.25]);"
        " y rel lane center(world y=y+level*3.0); z absolute(void plane=0)\",\n";
  mj << "  \"vadv\": " << sc.vadv << ", \"phase_dur\": " << sc.phase_dur << ", \"base_h\": " << base_h
     << ", \"cell\": " << cell << ", \"N\": " << N << ", \"sigma_cells\": " << sigma << ",\n";
  mj << "  \"y_band\": [" << sc.y_min << ", " << sc.y_max << "], \"l_max\": " << prm.l_max
     << ", \"h_des\": " << prm.h_des << ", \"on_stone_tol\": " << TOL << ",\n";
  mj << "  \"naive_rule\": \"nearest stone center(Euclid) to per-leg touchdown nominal(base@swing-end + hip)\",\n";
  mj << "  \"levels\": [\n" << meta_lv.str() << "\n  ]\n}\n";
  std::printf("출력: %s/plan_L*.csv footfalls_L*.csv plan_meta.json\n", outdir.c_str());
  return 0;
}
