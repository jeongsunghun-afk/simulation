// cmp_plan_stepping.cpp — ★단계 A: stepping 에서 **TAMOLS 발판 vs 기하 스냅 발판** 실측 비교.
//
//   왜: env(`go2_wtw_env.py::_compute_stepping_footholds`)의 stepping 참조원은 **TAMOLS 가 아니다** —
//       Raibert 반보폭 + 최근접 돌 클램프(기하)다. TAMOLS 캐시는 gap/계단 경로에서만 로드된다.
//       "stepping 에서 TAMOLS vs 기하"를 처음 재려면, 먼저 **같은 상태에서 두 플래너의 발판이
//       얼마나 다른가**를 오프라인으로 재야 한다(차이가 없으면 학습 비교는 base 참조만 재는 실험이 된다).
//
//   방법: 실제 롤아웃 덤프(GO2_FOOT_DIAG, 75열)를 읽어 **매 샘플 스텝의 실제 상태**(base pose·yaw·
//       측정 발위치)를 TAMOLS 초기조건으로 넣고, 같은 지형(stones_L*.csv = env 빌더와 1e-10 일치
//       확인됨)에서 **수정본 솔버**(GIAC_FIX·GIAC_ORDER·STANCE_REACH·TERRAIN_HARD)로 발판을 푼다.
//       덤프 3-5열 = 그 스텝에 기하 스냅이 실제로 내놓은 타겟 → 거리 분포·돌선택 불일치율을 낸다.
//
//   프레임: lane-local(x=코리도, y=레인중심 기준, z=절대). world_y = y + level*3.0.
//           TAMOLS 는 **yaw 정렬 프레임**에서 푼다(env 의 quat_apply_yaw 와 동일 규약):
//           고도맵 창을 base yaw 축으로 래스터 → 해의 local xy 를 Rz(yaw) 로 되돌려 world 비교.
//
//   빌드: PIX=/home/jsh/simple-mpc/.pixi/envs/default; g++ -O3 -std=c++17 cmp_plan_stepping.cpp \
//         -I/usr/include/eigen3 -I$PIX/include -L$PIX/lib -Wl,-rpath,$PIX/lib -leiquadprog -o cmp_plan_stepping
//   실행: ./cmp_plan_stepping <footdiag.txt> <level> [stride=25] [stones_dir=stepping_go2]
//   env : VADV(0.4 명령속도) · PAIRS_CSV(<경로>=쌍별 CSV 덤프) · TAMOLS_LEGACY=1(수정 전 솔버로 대조)
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

// ───────────────────────── 돌 필드(레벨당) ─────────────────────────
struct StoneField {
  int nx = 0, ny = 0; double size = 0, pitch = 0;
  std::vector<double> cxs, cys;   // ix→cx, iy→cy (lane-local)
  std::vector<double> top;        // ix*ny+iy → top z
};
static bool load_field(const std::string& path, StoneField& f) {
  std::ifstream in(path); if (!in) return false;
  std::string line; std::getline(in, line);
  std::vector<std::array<double,7>> rows; int nx = 0, ny = 0;
  while (std::getline(in, line)) {
    std::array<double,7> r;
    if (std::sscanf(line.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                    &r[0],&r[1],&r[2],&r[3],&r[4],&r[5],&r[6]) != 7) continue;
    rows.push_back(r); nx = std::max(nx,(int)r[1]+1); ny = std::max(ny,(int)r[2]+1);
  }
  if (rows.empty()) return false;
  f.nx = nx; f.ny = ny; f.size = rows[0][5];
  f.cxs.assign(nx,0); f.cys.assign(ny,0); f.top.assign(nx*ny,0);
  for (auto& r : rows) { int ix=(int)r[1], iy=(int)r[2];
    f.cxs[ix]=r[3]; f.cys[iy]=r[4]; f.top[ix*ny+iy]=r[6]; }
  f.pitch = nx>1 ? f.cxs[1]-f.cxs[0] : f.size;
  return true;
}
// 정확 지형 높이(lane-local). spawn strip [-0.75,0.75]×|y|≤1 top 0.15 · 돌 top · 그외 0.
// tol>0 = 분류 완화. sidx: 돌 idx / −1 strip / −2 void
static double field_h(const StoneField& f, double x, double y, double tol, int* sidx) {
  if (sidx) *sidx = -2;
  if (x >= -0.75-tol && x <= 0.75+tol && std::fabs(y) <= 1.0+tol) { if (sidx) *sidx = -1; return 0.15; }
  double hs = 0.5*f.size;
  int ix = (int)std::lround((x - f.cxs[0]) / f.pitch);
  int iy = (int)std::lround((y - f.cys[0]) / f.pitch);
  if (ix>=0 && ix<f.nx && iy>=0 && iy<f.ny &&
      std::fabs(x-f.cxs[ix])<=hs+tol && std::fabs(y-f.cys[iy])<=hs+tol) {
    if (sidx) *sidx = ix*f.ny+iy; return f.top[ix*f.ny+iy];
  }
  return 0.0;
}
// 최근접 돌(중심 유클리드) — env `_snap_xy_to_stone` 의 argmin 규칙과 동일
static int nearest_stone(const StoneField& f, double x, double y) {
  int best=-1; double bd=1e18;
  for (int ix=0; ix<f.nx; ++ix) for (int iy=0; iy<f.ny; ++iy) {
    double dx=x-f.cxs[ix], dy=y-f.cys[iy], d=dx*dx+dy*dy;
    if (d<bd) { bd=d; best=ix*f.ny+iy; } }
  return best;
}
// yaw 정렬 창 래스터: 창 중심 = (bx,by) lane-local, 축 = base yaw
static void raster_yaw(const StoneField& f, double bx, double by, double yaw,
                       int N, double cell, Grid& h) {
  double off = cell*N/2.0, cy = std::cos(yaw), sy = std::sin(yaw);
  h.resize(N,N);
  for (int a=0;a<N;++a) for (int b=0;b<N;++b) {
    double lx = a*cell-off, ly = b*cell-off;               // yaw 프레임 좌표
    h(a,b) = field_h(f, bx + cy*lx - sy*ly, by + sy*lx + cy*ly, 0.0, nullptr);
  }
}

// ───────────────────────── walk 게이트(at_des 누적) ─────────────────────────
//   cache_gen_go2_stepping 과 동일: LS crawl(RR→FR→RL→FL), at_des 를 스윙 시점부터 누적
//   (터미널 nominal 이 4발 전부의 발판을 구동 + GIAC 이 새 발판을 지지점으로 봄).
static void set_walk_gait_acc(TamolsState& st, double phase_dur) {
  int P=4; st.gait.resize(P);
  int cs[4][4] = {{1,1,1,0},{1,0,1,1},{1,1,0,1},{0,1,1,1}};
  int ad[4][4] = {{0,0,0,1},{0,1,0,1},{0,1,1,1},{1,1,1,1}};
  for (int k=0;k<P;++k) { st.gait[k].duration=phase_dur;
    for (int i=0;i<4;++i) { st.gait[k].contact[i]=cs[k][i]; st.gait[k].at_des[i]=ad[k][i]; } }
}

struct StepCfg { double vadv=0.4, phase_dur=0.2; int iters=60; double y_min=0.04, y_max=0.34; };

// 실제 상태에서 1회 replan (cache_gen 의 stepping_replan 을 실측 초기조건용으로 일반화)
static QpResult replan_at(TamolsState& st, const Grid& h, double cell, int ms,
                          double bz, double vx0, const Eigen::Matrix<double,4,3>& foot_meas,
                          const StepCfg& c, const Eigen::Matrix<double,4,3>* p_init) {
  set_walk_gait_acc(st, c.phase_dur);
  int P = st.num_phases(); double T = P*c.phase_dur, xf = c.vadv*T;
  st.base_pose << 0,0,bz,0,0,0;        // yaw 정렬 프레임 → yaw=0
  st.base_vel  << vx0,0,0,0,0,0;
  st.p_meas = foot_meas;
  st.ref_vel = Vector3d(c.vadv,0,0);
  st.a.assign(P, MatrixXd::Zero(6,4));
  double c1=vx0, c3=(c.vadv-vx0-2*(xf-vx0*T)/T)/(T*T), c2=(xf-vx0*T-c3*T*T*T)/(T*T);
  auto xg=[&](double t){ return c1*t+c2*t*t+c3*t*t*t; };
  auto vg=[&](double t){ return c1+2*c2*t+3*c3*t*t; };
  for (int k=0;k<P;++k) {
    st.a[k].col(0)=st.base_pose; st.a[k](2,0)=bz;
    double t0=k*c.phase_dur, x0=xg(t0), x1=xg(t0+c.phase_dur), v0=vg(t0), v1=vg(t0+c.phase_dur);
    st.a[k](0,0)=x0; st.a[k](0,1)=v0;
    st.a[k](0,2)=(3*(x1-x0)/c.phase_dur-2*v0-v1)/c.phase_dur;
    st.a[k](0,3)=(2*(x0-x1)/c.phase_dur+v0+v1)/(c.phase_dur*c.phase_dur);
  }
  if (p_init) st.p = *p_init;
  else for (int i=0;i<4;++i) {
    st.p(i,0)=st.prm.hip_offsets(i,0)+0.5*xf; st.p(i,1)=st.prm.hip_offsets(i,1);
    st.p(i,2)=bilinear_height(h, cell, ms, st.p(i,0), st.p(i,1));
  }
  st.epsilon = VectorXd::Zero(P);
  QpOptions o; o.max_iter=c.iters;
  o.zlo=bz-0.06; o.zhi=bz+0.06; o.rp_max=0.20; o.yaw_max=0.10;
  o.x_target = xf*0.9;
  o.y_min=c.y_min; o.y_max=c.y_max;
  o.gap=false;
  return solve_fast(st, h, cell, ms, o);
}

// ───────────────────────── 통계 도우미 ─────────────────────────
static double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  double i = q*(v.size()-1); int lo=(int)std::floor(i), hi=(int)std::ceil(i);
  return v[lo] + (i-lo)*(v[hi]-v[lo]);
}

int main(int argc, char** argv) {
  if (argc < 3) { std::printf("usage: %s <footdiag.txt> <level> [stride=25] [stones_dir]\n", argv[0]); return 2; }
  std::string fdpath = argv[1];
  int lvl = std::atoi(argv[2]);
  int stride = argc>3 ? std::atoi(argv[3]) : 25;
  std::string sdir = argc>4 ? argv[4] : "stepping_go2";

  // ★수정본 솔버 게이트(레거시 캐시 생성기와 달리 전부 ON) — 무엇을 켰는지 출력에 남긴다
  bool legacy = getenv("TAMOLS_LEGACY") && getenv("TAMOLS_LEGACY")[0] != '0';
  if (legacy) { setenv("GIAC_FIX","0",1); setenv("GIAC_ORDER","0",1); }
  else        { setenv("GIAC_FIX","1",1); setenv("GIAC_ORDER","1",1); setenv("STANCE_REACH","1",1); }

  StoneField f;
  if (!load_field(sdir + "/stones_L" + std::to_string(lvl) + ".csv", f)) {
    std::printf("stones_L%d.csv 로드 실패 (%s)\n", lvl, sdir.c_str()); return 1; }
  const double lane_y = lvl * 3.0;
  const double TOL = 0.02;                       // on-stone 허용(1셀)
  const int N = 101; const double cell = 0.02;   // 창 ±1.01 m

  StepCfg sc; sc.vadv = getenv("VADV") ? atof(getenv("VADV")) : 0.4;

  Params prm;                                    // Go2 물리 (cache_gen_go2_stepping 와 동일)
  prm.hip_offsets <<  0.1934, 0.142,0.0,  0.1934,-0.142,0.0, -0.1934, 0.142,0.0, -0.1934,-0.142,0.0;
  prm.mass=15.0; prm.foot_radius=0.022; prm.l_min=0.10; prm.l_max=0.45;

  // ── 덤프 로드(75열) ──
  std::ifstream in(fdpath);
  if (!in) { std::printf("footdiag 열기 실패: %s\n", fdpath.c_str()); return 1; }
  std::vector<std::array<double,75>> rows; std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line); std::array<double,75> r; int i=0;
    while (i<75 && (ss >> r[i])) ++i;
    if (i==75) rows.push_back(r);
  }
  if (rows.size() < 10) { std::printf("덤프 행 부족(%zu)\n", rows.size()); return 1; }

  std::vector<double> d_xy, d_x, d_y;            // TAMOLS↔기하 발판 거리
  long n_pair=0, n_diff_stone=0, n_both_stone=0;
  long n_solve=0, n_cycle=0, n_fail=0, n_tam_void=0, n_geo_void=0, n_inf_t=0, n_inf_g=0;
  std::vector<double> reach_t;                   // TAMOLS 발판의 hip 거리(도달성 점검)
  FILE* pf = nullptr;
  if (getenv("PAIRS_CSV")) { pf = std::fopen(getenv("PAIRS_CSV"), "w");
    if (pf) std::fprintf(pf, "step,leg,tam_x,tam_y,tam_z,geo_x,geo_y,geo_z,d_xy,tam_stone,geo_stone,differs,solver_ok\n"); }

  const char* LEGN[4] = {"FL","FR","RL","RR"};
  for (size_t ri = 0; ri < rows.size(); ri += stride) {
    const auto& r = rows[ri];
    double bx = r[1], by_w = r[2], bz = r[3], yaw = r[4];
    double by = by_w - lane_y;                                  // lane-local
    if (bx <= 0.75 || bx >= f.cxs.back() + 0.5*f.size) continue; // 필드 밖(스트립/종점) 제외
    // base 전진속도 = 유한차분(dt=0.02 s, 50 Hz 제어스텝)
    double vx0 = 0.0;
    if (ri+1 < rows.size()) vx0 = std::max(0.0, std::min(1.0, (rows[ri+1][1]-bx)/0.02));

    Grid hraw; raster_yaw(f, bx, by, yaw, N, cell, hraw);
    Grid hsol = gaussian_filter(hraw, 2.0);                     // 비용/기울기 shaping (cache_gen σ=2셀)
    // ★TERRAIN_HARD: 지지 유효성 하드 제약(원본 고도맵 기준). OFF 면 0행=기존과 동일.
    SupportLayer SL = compute_support(hraw, cell, 0.06, 0.25);
    SupportLayers S; S.h=&hraw; S.sdf=&SL.sdf; S.cell=cell; S.map_size=N;
    S.band=0.01; S.margin=0.025;                                // margin = env 기하스냅의 half−0.025 와 동일
    support_layers() = legacy ? nullptr : &S;

    // 측정 발위치 → yaw 정렬 프레임(anchor=base xy), z 는 절대
    double cy=std::cos(-yaw), sy=std::sin(-yaw);
    Eigen::Matrix<double,4,3> fm;
    for (int L=0; L<4; ++L) {
      double fx = r[5 + L*16 + 0] - bx, fy = r[5 + L*16 + 1] - lane_y - by;
      fm(L,0) = cy*fx - sy*fy; fm(L,1) = sy*fx + cy*fy; fm(L,2) = r[5 + L*16 + 2];
    }

    // ★멀티스타트 — cache_gen_go2_stepping 이 실제로 배포하는 TAMOLS 구성 그대로.
    //   시도 A = 명목 init(순수 TO). 발이 보이드에 남으면 ①스냅 init(최근접 돌; TO 가 기각·이동 가능)
    //   ②vadv 변주(사이클 전진량 = 돌 피치와의 정합). 채택 = feasible > on-stone 多 > viol 小.
    //   (이 보정 없이 재면 TAMOLS 를 실제 배포보다 불리하게 재게 된다.)
    auto n_off = [&](const TamolsState& s) {
      int off=0; double cf2=std::cos(yaw), sf2=std::sin(yaw);
      for (int L=0;L<4;++L) {
        double tx = bx + cf2*s.p(L,0) - sf2*s.p(L,1), ty = by + sf2*s.p(L,0) + cf2*s.p(L,1);
        if (tx <= 0.75+TOL || tx >= f.cxs.back()+0.5*f.size+TOL) continue;
        int si; field_h(f, tx, ty, TOL, &si); if (si==-2) ++off;
      }
      return off;
    };
    struct Cand { TamolsState st; QpResult r; bool ok; int off; };
    auto try_cand = [&](double va, bool use_snap) {
      Cand cd; cd.st.prm = prm; StepCfg scv = sc; scv.vadv = va;
      Eigen::Matrix<double,4,3> pini;
      if (use_snap) {
        double xf = va*4*sc.phase_dur, cf2=std::cos(yaw), sf2=std::sin(yaw);
        for (int L=0;L<4;++L) {
          double lx = prm.hip_offsets(L,0)+0.5*xf, ly = prm.hip_offsets(L,1);
          double nx = bx + cf2*lx - sf2*ly, ny = by + sf2*lx + cf2*ly;
          int ns = nearest_stone(f, nx, ny);
          double sx, sy2;
          if (nx < 0.75) { sx = nx; sy2 = ny; }                 // 스트립 위 = 명목 유지
          else { sx = f.cxs[ns/f.ny]; sy2 = f.cys[ns%f.ny]; }
          double dx2 = sx-bx, dy2 = sy2-by;                     // world → yaw 프레임
          pini(L,0) = cf2*dx2 + sf2*dy2; pini(L,1) = -sf2*dx2 + cf2*dy2;
          if (L==0||L==2) pini(L,1) = std::max(sc.y_min, std::min(sc.y_max, pini(L,1)));
          else            pini(L,1) = std::min(-sc.y_min, std::max(-sc.y_max, pini(L,1)));
          pini(L,2) = bilinear_height(hsol, cell, N, pini(L,0), pini(L,1));
        }
      }
      cd.r = replan_at(cd.st, hsol, cell, N, bz, vx0, fm, scv, use_snap ? &pini : nullptr);
      ++n_solve;
      cd.ok = (cd.r.eq_viol < 1e-2 && cd.r.ineq_viol < 1e-2);
      cd.off = n_off(cd.st);
      return cd;
    };
    auto better = [](const Cand& a, const Cand& b) {
      if (a.ok != b.ok) return a.ok;
      if (a.off != b.off) return a.off < b.off;
      return a.r.eq_viol + a.r.ineq_viol < b.r.eq_viol + b.r.ineq_viol;
    };
    Cand best = try_cand(sc.vadv, false);
    if (!best.ok || best.off > 0) {
      const double va_list[5] = {sc.vadv, 0.8*sc.vadv, 1.2*sc.vadv, 0.65*sc.vadv, 1.35*sc.vadv};
      for (double va : va_list) {
        Cand c2 = try_cand(va, true);
        if (better(c2, best)) best = c2;
        if (best.ok && best.off == 0) break;
      }
    }
    TamolsState& st = best.st;
    bool ok = best.ok;
    ++n_cycle; if (!ok) ++n_fail;
    support_layers() = nullptr;

    // 해 → lane-local world (yaw 되돌림)
    double cf=std::cos(yaw), sf=std::sin(yaw);
    for (int L=0; L<4; ++L) {
      double tx = bx + cf*st.p(L,0) - sf*st.p(L,1);
      double ty = by + sf*st.p(L,0) + cf*st.p(L,1);
      double tz = st.p(L,2);
      double gx = r[5 + L*16 + 3], gy = r[5 + L*16 + 4] - lane_y, gz = r[5 + L*16 + 5];
      double dx = tx-gx, dy = ty-gy, dd = std::hypot(dx,dy);
      d_xy.push_back(dd); d_x.push_back(dx); d_y.push_back(dy); ++n_pair;
      int si_t, si_g; field_h(f, tx, ty, TOL, &si_t); field_h(f, gx, gy, TOL, &si_g);
      // 보이드율은 **필드 안**(스트립 뒤 ~ 마지막 돌기둥 앞) 타겟만 센다 — 코리도 밖은 선택 실패가 아님
      bool inf_t = (tx > 0.75+TOL && tx < f.cxs.back()+0.5*f.size+TOL);
      bool inf_g = (gx > 0.75+TOL && gx < f.cxs.back()+0.5*f.size+TOL);
      if (inf_t) { ++n_inf_t; if (si_t == -2) ++n_tam_void; }
      if (inf_g) { ++n_inf_g; if (si_g == -2) ++n_geo_void; }
      // "다른 돌을 골랐나" = 둘 다 돌 위일 때의 불일치 (void 는 선택이 아니라 실패라 분리 집계)
      int cand_t = (si_t>=0) ? si_t : nearest_stone(f, tx, ty);
      int cand_g = (si_g>=0) ? si_g : nearest_stone(f, gx, gy);
      if (si_t>=0 && si_g>=0) { ++n_both_stone; if (cand_t != cand_g) ++n_diff_stone; }
      reach_t.push_back(std::hypot(std::hypot(st.p(L,0)-prm.hip_offsets(L,0), st.p(L,1)-prm.hip_offsets(L,1)), bz-st.p(L,2)));
      if (pf) std::fprintf(pf, "%d,%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%d,%d,%d,%d\n",
                           (int)r[0], LEGN[L], tx,ty,tz, gx,gy,gz, dd, cand_t, cand_g,
                           (si_t>=0&&si_g>=0&&cand_t!=cand_g)?1:0, ok?1:0);
    }
  }
  if (pf) std::fclose(pf);

  if (n_pair == 0) { std::printf("비교쌍 0 (필드 내 행 없음)\n"); return 1; }
  std::printf("\n=== 단계 A: stepping TAMOLS vs 기하 스냅 발판 (L%d, %s) ===\n", lvl,
              legacy ? "★LEGACY 솔버(GIAC_FIX=0·ORDER=0·하드지형 OFF)" :
                       "★수정본(GIAC_FIX=1·GIAC_ORDER=1·STANCE_REACH=1·TERRAIN_HARD=1)");
  std::printf("덤프: %s  행 %zu (stride %d)  vadv=%.2f\n", fdpath.c_str(), rows.size(), stride, sc.vadv);
  std::printf("계획 사이클 %ld (해 실패 %ld = %.1f%%)   총 솔브 %ld (멀티스타트 포함)   비교쌍 %ld\n", n_cycle, n_fail, 100.0*n_fail/std::max(1L,n_cycle), n_solve, n_pair);
  std::printf("발판 거리 |Δxy| [m]:  중앙 %.4f  IQR [%.4f, %.4f]  p90 %.4f  max %.4f\n",
              pct(d_xy,0.50), pct(d_xy,0.25), pct(d_xy,0.75), pct(d_xy,0.90), pct(d_xy,1.0));
  std::printf("   성분:  Δx 중앙 %+.4f (IQR %+.4f..%+.4f)   Δy 중앙 %+.4f (IQR %+.4f..%+.4f)\n",
              pct(d_x,0.50), pct(d_x,0.25), pct(d_x,0.75), pct(d_y,0.50), pct(d_y,0.25), pct(d_y,0.75));
  std::printf("돌 선택:  둘 다 돌 위 %ld/%ld  →  **다른 돌 %ld = %.1f%%**\n",
              n_both_stone, n_pair, n_diff_stone, 100.0*n_diff_stone/std::max(1L,n_both_stone));
  std::printf("★보이드 발판(필드 내 타겟만):  TAMOLS %ld/%ld (%.1f%%)   기하 %ld/%ld (%.1f%%)\n",
              n_tam_void, n_inf_t, 100.0*n_tam_void/std::max(1L,n_inf_t),
              n_geo_void, n_inf_g, 100.0*n_geo_void/std::max(1L,n_inf_g));
  std::printf("TAMOLS 발판 hip 거리: 중앙 %.3f  p90 %.3f  max %.3f  (l_max=%.2f)\n",
              pct(reach_t,0.50), pct(reach_t,0.90), pct(reach_t,1.0), prm.l_max);
  // ★기계적 판정: 차이가 발 반경 스케일(~2 cm) 이하면 "두 플래너는 사실상 같은 참조"
  double med = pct(d_xy,0.50);
  std::printf("\n판정: 중앙 차이 %.1f cm — %s\n", med*100,
              med < 0.02 ? "★발 반경 이하 = 두 참조가 사실상 동일(정책이 같은 것을 추종)"
                         : (med < 0.05 ? "돌 반크기 수준 = 부분적으로만 다름" : "돌 선택이 실제로 갈린다"));
  return 0;
}
