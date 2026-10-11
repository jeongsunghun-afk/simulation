// cache_gen_go2_stepping_ref.cpp — ★단계 B: Go2 stepping-stone **TAMOLS 오프라인 캐시**
//   (발판 + ★base 궤적). RL env(`go2_wtw_env.py`) 의 stepping 경로가 소비할 참조원.
//
// ─────────────────────────── 왜 새로 만드는가 ───────────────────────────
//   ① env 의 stepping 참조원은 **TAMOLS 가 아니었다** — Raibert + 최근접 돌 클램프(기하)다.
//      기존 TAMOLS 캐시(tamols_cache_go2*)는 gap/계단 경로에서만 로드된다.
//   ② 기존 `cache_gen_go2_stepping.cpp` 는 **코리도 스윕**(receding-horizon)이라 출력이
//      "레벨별 발자국 시퀀스"다 — RL 이 임의 상태에서 조회할 수 있는 **색인된 캐시가 아니다**.
//   ③ 기존 `cache_gen*` 는 재현성 때문에 `GIAC_FIX=0·GIAC_ORDER=0` 으로 **레거시 고정**이다.
//      새 캐시를 그걸로 만들면 또 불구 TAMOLS 를 재게 된다 → 이 파일은 **수정본으로 생성**한다.
//
// ─────────────────── 파라미터화(★결정 근거) ───────────────────
//   gap 캐시는 (vx × gap폭 × gap거리)로 색인된다. stepping 엔 "다음 갭까지 거리" 축이 없다 —
//   대신 **격자가 규칙적**이다: 돌 중심 cx[ix]=0.75+(ix+0.5)·pitch, cy[iy]=y0+iy·pitch,
//   pitch=size+gap 이 레벨로 결정된다(env `_build_stepping_terrain_curriculum`).
//   ⇒ base 주변 국소 지형은 **x·y 양방향으로 주기 `pitch` 의 주기함수**다. 따라서:
//
//     vx     — 명령 전진속도            (gap 캐시의 vx 축과 동일 역할)
//     level  — pitch 의 **절대 스케일** + size/pitch **듀티**(L7 0.536 vs L9 0.357)를 정한다.
//              다리 길이가 고정이라 절대 스케일은 정규화로 없앨 수 없다 ⇒ 독립 축이어야 한다.
//     φx     — 한 pitch 안의 **x 국소 위상**. 규칙격자에서 "다음 돌기둥까지 거리" = 위상이므로
//              **gap 캐시의 `gapd` 축에 정확히 대응**한다.
//     φy     — 한 pitch 안의 **y 국소 위상**. 필요하다: 행 격자가 레인 중심에 정렬돼 있지 않다
//              (ny 짝수인 L7 은 행이 y=0 을 사이에 두고 벌어지고, ny 홀수인 L9 는 y=0 에 행이 있다).
//              게다가 실측 레인편차가 작지 않다(L9 롤아웃 중앙 −0.144 m) ⇒ 무시하면 잘못된 셀을 읽는다.
//
//   한계(정직): 격자 주기성은 **필드 내부**에서만 성립한다. 스폰 스트립 경계와 코리도 끝에서는
//   깨진다 — gap 캐시가 다중 갭 상호작용을 무시하는 것과 같은 종류의 근사다. meta 에 기록한다.
//
// ─────────────────────────── 출력 ───────────────────────────
//   meta.json      축 값 · 형상 · **켠 게이트** · 관례
//   footholds.bin  float32 [n_vx, n_lvl, n_phx, n_phy, 4, 3]  발판, **yaw 정렬 base 앵커 로컬**
//                  (xy = base 기준 상대, z = 지면(돌 top 평균) 기준 상대)
//   baseref.bin    float32 [n_vx, n_lvl, n_phx, n_phy, n_t, 6]  ★base 궤적:
//                  (x, y, z_rel_ground, vx, vy, vz) — 위치만 맞고 속도가 어긋나는 경우를 구분하려고
//                  속도까지 담는다(런타임 덤프에서 분리 진단).
//   solveok.bin    uint8  [n_vx, n_lvl, n_phx, n_phy]  1 = feasible 해(아니면 참조 신뢰 불가)
//
//   빌드: PIX=/home/jsh/simple-mpc/.pixi/envs/default; g++ -O3 -std=c++17 cache_gen_go2_stepping_ref.cpp \
//         -I/usr/include/eigen3 -I$PIX/include -L$PIX/lib -Wl,-rpath,$PIX/lib -leiquadprog -o cache_gen_go2_stepping_ref
//   실행: ./cache_gen_go2_stepping_ref <outdir> [level|all]      (레벨별 병렬 실행 후 merge 로 합침)
//         ./cache_gen_go2_stepping_ref <outdir> merge            (레벨 조각 → 최종 bin/meta)
#include "tamols_online.hpp"
#include "terrain_proc.hpp"
#include "tamols_online.hpp"
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
using namespace tamols;

// ═══════════════ 지형(env 빌더 정합; stones_L*.csv 는 1e-10 일치 확인됨) ═══════════════
struct StoneField {
  int nx=0, ny=0; double size=0, pitch=0;
  std::vector<double> cxs, cys, top;
};
static bool load_field(const std::string& path, StoneField& f) {
  std::ifstream in(path); if (!in) return false;
  std::string line; std::getline(in, line);
  std::vector<std::array<double,7>> rows; int nx=0, ny=0;
  while (std::getline(in, line)) {
    std::array<double,7> r;
    if (std::sscanf(line.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                    &r[0],&r[1],&r[2],&r[3],&r[4],&r[5],&r[6]) != 7) continue;
    rows.push_back(r); nx=std::max(nx,(int)r[1]+1); ny=std::max(ny,(int)r[2]+1);
  }
  if (rows.empty()) return false;
  f.nx=nx; f.ny=ny; f.size=rows[0][5];
  f.cxs.assign(nx,0); f.cys.assign(ny,0); f.top.assign((size_t)nx*ny,0);
  for (auto& r : rows) { int ix=(int)r[1], iy=(int)r[2];
    f.cxs[ix]=r[3]; f.cys[iy]=r[4]; f.top[(size_t)ix*ny+iy]=r[6]; }
  f.pitch = nx>1 ? f.cxs[1]-f.cxs[0] : f.size;
  return true;
}
static double field_h(const StoneField& f, double x, double y, double tol, int* sidx) {
  if (sidx) *sidx = -2;
  if (x >= -0.75-tol && x <= 0.75+tol && std::fabs(y) <= 1.0+tol) { if (sidx) *sidx=-1; return 0.15; }
  double hs = 0.5*f.size;
  int ix=(int)std::lround((x-f.cxs[0])/f.pitch), iy=(int)std::lround((y-f.cys[0])/f.pitch);
  if (ix>=0&&ix<f.nx&&iy>=0&&iy<f.ny &&
      std::fabs(x-f.cxs[ix])<=hs+tol && std::fabs(y-f.cys[iy])<=hs+tol) {
    if (sidx) *sidx = ix*f.ny+iy; return f.top[(size_t)ix*f.ny+iy]; }
  return 0.0;
}
static int nearest_stone(const StoneField& f, double x, double y) {
  int best=-1; double bd=1e18;
  for (int ix=0;ix<f.nx;++ix) for (int iy=0;iy<f.ny;++iy) {
    double dx=x-f.cxs[ix], dy=y-f.cys[iy], d=dx*dx+dy*dy; if (d<bd){bd=d;best=ix*f.ny+iy;} }
  return best;
}
static void raster(const StoneField& f, double bx, double by, int N, double cell, Grid& h) {
  double off = cell*N/2.0; h.resize(N,N);
  for (int a=0;a<N;++a) for (int b=0;b<N;++b)
    h(a,b) = field_h(f, bx + a*cell - off, by + b*cell - off, 0.0, nullptr);
}

// ═══════════════ walk 게이트(at_des 누적) — cache_gen_go2_stepping 와 동일 ═══════════════
static void set_walk_gait_acc(TamolsState& st, double pd) {
  int P=4; st.gait.resize(P);
  int cs[4][4]={{1,1,1,0},{1,0,1,1},{1,1,0,1},{0,1,1,1}};
  int ad[4][4]={{0,0,0,1},{0,1,0,1},{0,1,1,1},{1,1,1,1}};
  for (int k=0;k<P;++k){ st.gait[k].duration=pd;
    for(int i=0;i<4;++i){ st.gait[k].contact[i]=cs[k][i]; st.gait[k].at_des[i]=ad[k][i]; } }
}
struct StepCfg { double vadv=0.4, phase_dur=0.2; int iters=60; double y_min=0.04, y_max=0.34; };

static QpResult replan(TamolsState& st, const Grid& h, double cell, int ms, double z0, double vx0,
                       const Eigen::Matrix<double,4,3>& fm, const StepCfg& c,
                       const Eigen::Matrix<double,4,3>* p_init) {
  set_walk_gait_acc(st, c.phase_dur);
  int P=st.num_phases(); double T=P*c.phase_dur, xf=c.vadv*T;
  st.base_pose << 0,0,z0,0,0,0;
  st.base_vel  << vx0,0,0,0,0,0;
  st.p_meas = fm; st.ref_vel = Vector3d(c.vadv,0,0);
  st.a.assign(P, MatrixXd::Zero(6,4));
  double c1=vx0, c3=(c.vadv-vx0-2*(xf-vx0*T)/T)/(T*T), c2=(xf-vx0*T-c3*T*T*T)/(T*T);
  auto xg=[&](double t){ return c1*t+c2*t*t+c3*t*t*t; };
  auto vg=[&](double t){ return c1+2*c2*t+3*c3*t*t; };
  for (int k=0;k<P;++k){
    st.a[k].col(0)=st.base_pose; st.a[k](2,0)=z0;
    double t0=k*c.phase_dur, x0=xg(t0), x1=xg(t0+c.phase_dur), v0=vg(t0), v1=vg(t0+c.phase_dur);
    st.a[k](0,0)=x0; st.a[k](0,1)=v0;
    st.a[k](0,2)=(3*(x1-x0)/c.phase_dur-2*v0-v1)/c.phase_dur;
    st.a[k](0,3)=(2*(x0-x1)/c.phase_dur+v0+v1)/(c.phase_dur*c.phase_dur);
  }
  if (p_init) st.p=*p_init;
  else for (int i=0;i<4;++i){
    st.p(i,0)=st.prm.hip_offsets(i,0)+0.5*xf; st.p(i,1)=st.prm.hip_offsets(i,1);
    st.p(i,2)=bilinear_height(h,cell,ms,st.p(i,0),st.p(i,1)); }
  st.epsilon=VectorXd::Zero(P);
  QpOptions o; o.max_iter=c.iters;
  o.zlo=z0-0.06; o.zhi=z0+0.06; o.rp_max=0.20; o.yaw_max=0.10;
  o.x_target=xf*0.9; o.y_min=c.y_min; o.y_max=c.y_max; o.gap=false;
  return solve_fast(st,h,cell,ms,o);
}

// ═══════════════ 축 정의 ═══════════════
//   ★축 해상도/반복상한은 **측정해서** 정한다(SREF_NPH·SREF_ITERS 로 프로브 가능).
//   어려운 레벨(L7~L9)은 돌 반폭이 0.05 m 인데 TERRAIN_MARGIN 0.025 + band 0.01 이라 선형화 QP 가
//   자주 infeasible → elastic 폴백(차원 ~1345)이 SQP 반복마다 도는 README 의 알려진 병목에 걸린다.
static const double VX_VALS[] = {0.27, 0.35, 0.43};          // cell B 학습 밴드 [0.25,0.45] 를 3점으로
static const int N_VX  = 3;
static const int N_LVL = 10;                                 // env num_terrain_levels
static int N_PHX = 8, N_PHY = 8;                             // pitch 0.28~0.31 m → 3.5~3.9 cm(≈발반경+셀)
static const int N_T   = 9;                                  // base 궤적 샘플 수(호라이즌 0.8 s → 0.1 s 간격)
static int SREF_ITERS = 20;
static double ON_dsum=0, ON_dmax=0, ON_ms=0; static long ON_n=0, ON_okc=0, ON_okon=0, ON_agree=0;
static int ON_ITERS=5; static int ON_SUPPORT=1; static double CACHE_ms=0;                                  // SQP 상한(측정 기반; README: RTI5=실시간·80=완전수렴)

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: %s <outdir> [level|all|merge]\n", argv[0]); return 2; }
  std::string outdir = argv[1];
  std::string mode = argc>2 ? argv[2] : "all";
  if (getenv("SREF_ITERS")) SREF_ITERS = atoi(getenv("SREF_ITERS"));
  if (getenv("ON_ITERS")) ON_ITERS = atoi(getenv("ON_ITERS"));
  if (getenv("ON_SUPPORT")) ON_SUPPORT = atoi(getenv("ON_SUPPORT"));
  if (getenv("SREF_NPH"))   { N_PHX = N_PHY = atoi(getenv("SREF_NPH")); }

  // ───────── merge: 레벨 조각 → 최종 bin + meta ─────────
  if (mode == "merge") {
    size_t nf = (size_t)N_VX*N_LVL*N_PHX*N_PHY*4*3, nb = (size_t)N_VX*N_LVL*N_PHX*N_PHY*N_T*6;
    size_t nk = (size_t)N_VX*N_LVL*N_PHX*N_PHY;
    std::vector<float> FH(nf, 0.f), BR(nb, 0.f); std::vector<uint8_t> OK(nk, 0);
    int nlv_ok = 0; long tot_ok = 0;
    for (int L=0; L<N_LVL; ++L) {
      char p[512]; std::snprintf(p,sizeof p,"%s/part_L%d.bin", outdir.c_str(), L);
      std::ifstream in(p, std::ios::binary); if (!in) { std::printf("  [merge] 누락: part_L%d.bin\n", L); continue; }
      size_t pf=(size_t)N_VX*N_PHX*N_PHY*4*3, pb=(size_t)N_VX*N_PHX*N_PHY*N_T*6, pk=(size_t)N_VX*N_PHX*N_PHY;
      std::vector<float> f(pf), b(pb); std::vector<uint8_t> k(pk);
      in.read((char*)f.data(), pf*4); in.read((char*)b.data(), pb*4); in.read((char*)k.data(), pk);
      if (!in) { std::printf("  [merge] 손상: part_L%d.bin\n", L); continue; }
      // 조각 레이아웃 [vx][phx][phy][...] → 최종 [vx][lvl][phx][phy][...]
      for (int v=0; v<N_VX; ++v) for (int a=0; a<N_PHX; ++a) for (int b2=0; b2<N_PHY; ++b2) {
        size_t sp = ((size_t)v*N_PHX + a)*N_PHY + b2;
        size_t dp = (((size_t)v*N_LVL + L)*N_PHX + a)*N_PHY + b2;
        std::memcpy(&FH[dp*12], &f[sp*12], 12*sizeof(float));
        std::memcpy(&BR[dp*N_T*6], &b[sp*N_T*6], N_T*6*sizeof(float));
        OK[dp] = k[sp]; tot_ok += k[sp];
      }
      ++nlv_ok;
    }
    auto wr=[&](const char* nm, const void* d, size_t n){ char p[512];
      std::snprintf(p,sizeof p,"%s/%s",outdir.c_str(),nm);
      std::ofstream o(p, std::ios::binary); o.write((const char*)d, n); };
    wr("footholds.bin", FH.data(), nf*4);
    wr("baseref.bin",   BR.data(), nb*4);
    wr("solveok.bin",   OK.data(), nk);
    char mp[512]; std::snprintf(mp,sizeof mp,"%s/meta.json",outdir.c_str());
    std::ofstream m(mp);
    m << "{\n";
    m << "  \"kind\": \"stepping_stone_tamols_ref\", \"model\": \"go2\", \"gait\": \"walk(RR,FR,RL,FL) at_des accumulated\",\n";
    m << "  \"axes\": [\"vx\", \"level\", \"phase_x\", \"phase_y\"],\n";
    m << "  \"axis_rationale\": \"stepping grid is PERIODIC with period pitch(level) in x and y; the local"
         " phase within one pitch is the exact analogue of the gap cache's gapd axis. level stays an"
         " independent axis because it sets the ABSOLUTE pitch and the size/pitch duty, which a"
         " fixed-length leg cannot normalise away.\",\n";
    m << "  \"n_vx\": " << N_VX << ", \"n_level\": " << N_LVL
      << ", \"n_phase_x\": " << N_PHX << ", \"n_phase_y\": " << N_PHY << ", \"n_t\": " << N_T << ",\n";
    m << "  \"vx_vals\": ["; for (int i=0;i<N_VX;++i) m << (i?", ":"") << VX_VALS[i]; m << "],\n";
    m << "  \"phase_def\": \"phase_x = frac((base_x - cx0)/pitch), cx0 = 0.75 + 0.5*pitch;"
         " phase_y = frac((base_y - lane_y - cy0)/pitch), cy0 = -0.5*(ny-1)*pitch."
         " Bin i centre = (i + 0.5)/n.\",\n";
    m << "  \"footholds_shape\": [" << N_VX << ", " << N_LVL << ", " << N_PHX << ", " << N_PHY << ", 4, 3],\n";
    m << "  \"baseref_shape\": ["   << N_VX << ", " << N_LVL << ", " << N_PHX << ", " << N_PHY << ", " << N_T << ", 6],\n";
    m << "  \"foot_order\": [\"FL\",\"FR\",\"RL\",\"RR\"],\n";
    m << "  \"frame\": \"yaw-aligned base anchor: xy relative to base, foothold z relative to local ground"
         " (mean stone top); baseref = (x, y, z_rel_ground, vx, vy, vz), world = base_xy + Rz(yaw)*xy.\",\n";
    m << "  \"horizon_s\": 0.8, \"phase_dur\": 0.2, \"base_h\": 0.34, \"cell\": 0.02, \"window_N\": 101,\n";
    // ★어떤 게이트를 켰는지 = 이 캐시의 정체성. 레거시 캐시와 구별되는 지점.
    m << "  \"solver_gates\": {\"GIAC_FIX\": 1, \"GIAC_ORDER\": 1, \"STANCE_REACH\": 1, \"TERRAIN_HARD\": 1,\n";
    m << "     \"GIAC_NORM\": 0, \"GIAC_NORM_note\": \"eps normalisation gate left OFF: measured eps share of"
         " the objective is 3.8% under the 3-foot WALK stance this cache uses (55.7% under a 2-foot trot"
         " stance, which is the degenerate-support case), eps=0.000 at every converged t1-t4 solution, and"
         " normalisation makes eps CHEAPER (residual/11.9 with W_EPS fixed => 141x weaker), not tighter.\",\n";
    m << "     \"TERRAIN_R\": 0.25, \"TERRAIN_DZ\": 0.06, \"TERRAIN_BAND\": 0.01, \"TERRAIN_MARGIN\": 0.025},\n";
    m << "  \"solver_gates_note\": \"generated with the FIXED solver -- unlike cache_gen*.cpp which pin"
         " GIAC_FIX=0/GIAC_ORDER=0 for artifact reproducibility.\",\n";
    m << "  \"limits\": \"lattice periodicity holds in the FIELD INTERIOR only; the spawn-strip boundary and"
         " the corridor end are not represented (same class of approximation as the gap cache ignoring"
         " multi-gap interaction).\",\n";
    m << "  \"levels_merged\": " << nlv_ok << ", \"cells_feasible\": " << tot_ok
      << ", \"cells_total\": " << nk << "\n}\n";
    std::printf("[merge] 레벨 %d/%d · feasible %ld/%zu (%.1f%%) → %s/{footholds,baseref,solveok}.bin meta.json\n",
                nlv_ok, N_LVL, tot_ok, nk, 100.0*tot_ok/nk, outdir.c_str());
    return 0;
  }

  // ───────── 생성 ─────────
  // ★수정본 솔버로 생성한다(레거시 cache_gen* 와 의도적으로 다름). meta 에 기록.
  setenv("GIAC_FIX","1",1); setenv("GIAC_ORDER","1",1); setenv("STANCE_REACH","1",1);
  // ★GIAC_NORM(eps 정규화)은 **끈 채로** 생성한다. 근거(실측, diag_eps_stepping):
  //   · walk(3발 지지)에서 eps 목적함수 점유율 평균 **3.8%** — 삼키지 않는다(삼키는 건 trot 2발: 55.7%).
  //   · t1~t4 **모든 수렴해에서 eps=0.000** — 정규화해도 4대 기계는 0개 부활(변화 없음).
  //   · 정규화는 GIAC 잔차를 ~11.9로 나누므로 W_EPS 고정 시 같은 물리위반의 페널티가 141× **싸진다**
  //     = 조이는 게 아니라 느슨해진다. 조이는 레버는 EPS_MAX(정규화 후 ≈ d_max/L 로 유도 가능)다.
  //   · t4 ON 에서 수렴 실패(ok=0)가 한 건 생겼다 = 켜면 오히려 약간 나빠진다.
  //   ⇒ 이 캐시는 단계 A(발판 비교)와 **같은 솔버 설정**으로 두어 두 측정이 서로 호환되게 한다.
  unsetenv("GIAC_NORM");
  int only = (mode=="all") ? -1 : std::atoi(mode.c_str());

  const int N = 101; const double cell = 0.02, base_h = 0.34, TOL = 0.02;
  Params prm;
  prm.hip_offsets << 0.1934,0.142,0.0,  0.1934,-0.142,0.0, -0.1934,0.142,0.0, -0.1934,-0.142,0.0;
  prm.mass=15.0; prm.h_des=base_h; prm.nominal_height=base_h;
  prm.foot_radius=0.022; prm.l_min=0.10; prm.l_max=0.45;

  for (int L=0; L<N_LVL; ++L) {
    if (only>=0 && L!=only) continue;
    StoneField f;
    if (!load_field(outdir + "/stones_L" + std::to_string(L) + ".csv", f)) {
      // 조각 파일이라도 남겨 merge 가 누락을 보고하게
      std::printf("L%d: stones csv 없음 — 건너뜀\n", L); continue; }
    const double pitch = f.pitch;
    // 앵커 격자점: 필드 중앙 돌기둥 + 레인중심에 가장 가까운 행(위상은 이 기준으로 정의)
    int ix0 = f.nx/2, iy0 = 0; { double bd=1e18;
      for (int i=0;i<f.ny;++i){ double d=std::fabs(f.cys[i]); if(d<bd){bd=d;iy0=i;} } }

    size_t pf=(size_t)N_VX*N_PHX*N_PHY*4*3, pb=(size_t)N_VX*N_PHX*N_PHY*N_T*6, pk=(size_t)N_VX*N_PHX*N_PHY;
    std::vector<float> FH(pf,0.f), BR(pb,0.f); std::vector<uint8_t> OK(pk,0);
    long n_ok=0, n_cell=0, n_void=0, n_inf=0; double dsum=0;

    for (int v=0; v<N_VX; ++v) {
      StepCfg sc; sc.vadv = VX_VALS[v]; sc.iters = SREF_ITERS;
      for (int a=0; a<N_PHX; ++a) for (int b=0; b<N_PHY; ++b) {
        double phx = (a+0.5)/N_PHX, phy = (b+0.5)/N_PHY;
        double bx = f.cxs[ix0] + phx*pitch;
        double by = f.cys[iy0] + phy*pitch;
        ++n_cell;

        Grid hraw; raster(f, bx, by, N, cell, hraw);
        Grid hsol = gaussian_filter(hraw, 2.0);
        SupportLayer SL = compute_support(hraw, cell, 0.06, 0.25);      // ★지형 하드(부양 금지)
        SupportLayers S; S.h=&hraw; S.sdf=&SL.sdf; S.cell=cell; S.map_size=N;
        S.band=0.01; S.margin=0.025;
        support_layers() = &S;

        // 지면 = 앵커 주변 돌 top 평균 (보이드에 빠진 발이 base 를 끌어내리는 악순환 차단)
        double gsum=0; int gn=0;
        for (int ix=0; ix<f.nx; ++ix) if (std::fabs(f.cxs[ix]-bx) < 0.45)
          for (int iy=0; iy<f.ny; ++iy) { gsum += f.top[(size_t)ix*f.ny+iy]; ++gn; }
        double ground = gn ? gsum/gn : 0.15;
        double z0 = base_h + ground;

        // 측정 발위치 = "지금 돌 위에 서 있다"(hip 명목에서 최근접 돌로 스냅) — 실측 롤아웃과 정합
        Eigen::Matrix<double,4,3> fm;
        for (int l=0;l<4;++l) {
          double hx = bx + prm.hip_offsets(l,0), hy = by + prm.hip_offsets(l,1);
          int ns = nearest_stone(f, hx, hy);
          double sx = f.cxs[ns/f.ny], sy = f.cys[ns%f.ny];
          if (hx < 0.75) { sx = hx; sy = hy; }                          // 스트립 위면 명목 유지
          fm(l,0)=sx-bx; fm(l,1)=sy-by; fm(l,2)=field_h(f,sx,sy,TOL,nullptr);
        }

        // 멀티스타트(cache_gen_go2_stepping 와 동일 규약): 명목 init → 보이드 남으면 스냅 init × vadv 변주
        auto n_off=[&](const TamolsState& s){ int off=0;
          for (int l=0;l<4;++l){ double tx=bx+s.p(l,0), ty=by+s.p(l,1);
            if (tx<=0.75+TOL || tx>=f.cxs.back()+0.5*f.size+TOL) continue;
            int si; field_h(f,tx,ty,TOL,&si); if (si==-2) ++off; } return off; };
        struct Cand { TamolsState st; QpResult r; bool ok; int off; };
        auto try_c=[&](double va, bool snap){
          Cand cd; cd.st.prm=prm; StepCfg scv=sc; scv.vadv=va;
          Eigen::Matrix<double,4,3> pini;
          if (snap) { double xf=va*4*sc.phase_dur;
            for (int l=0;l<4;++l){
              double nx=bx+prm.hip_offsets(l,0)+0.5*xf, ny=by+prm.hip_offsets(l,1);
              int ns=nearest_stone(f,nx,ny);
              double sx = (nx<0.75)? nx : f.cxs[ns/f.ny], sy = (nx<0.75)? ny : f.cys[ns%f.ny];
              pini(l,0)=sx-bx; pini(l,1)=sy-by;
              if (l==0||l==2) pini(l,1)=std::max(sc.y_min,std::min(sc.y_max,pini(l,1)));
              else            pini(l,1)=std::min(-sc.y_min,std::max(-sc.y_max,pini(l,1)));
              pini(l,2)=bilinear_height(hsol,cell,N,pini(l,0),pini(l,1)); } }
          cd.r = replan(cd.st, hsol, cell, N, z0, sc.vadv, fm, scv, snap?&pini:nullptr);
          cd.ok = (cd.r.eq_viol<1e-2 && cd.r.ineq_viol<1e-2);
          cd.off = n_off(cd.st); return cd; };
        auto better=[](const Cand& x, const Cand& y){
          if (x.ok!=y.ok) return x.ok; if (x.off!=y.off) return x.off<y.off;
          return x.r.eq_viol+x.r.ineq_viol < y.r.eq_viol+y.r.ineq_viol; };
        auto _tc0=std::chrono::high_resolution_clock::now();
        Cand best = try_c(sc.vadv,false);
        double _tcache=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-_tc0).count();
        CACHE_ms += _tcache;
        // ★멀티스타트는 **보이드 발판일 때만** 돈다. 원래 목적이 "발이 보이드에 남는 것"의 구제이고,
        //   TERRAIN_HARD 가 켜진 지금은 그 실패가 사실상 사라졌다(단계 A 실측 보이드 0.0%).
        //   반면 `!best.ok`(선형화 QP infeasible)까지 트리거로 두면 어려운 레벨에서 셀마다 6 솔브가
        //   돌아 생성 비용이 6× 튄다 — 얻는 것 없이(재시도해도 여전히 infeasible) 비용만 든다.
        if (best.off>0) {
          const double vl[5] = {sc.vadv, 0.8*sc.vadv, 1.2*sc.vadv, 0.65*sc.vadv, 1.35*sc.vadv};
          for (double va : vl){ Cand c2=try_c(va,true); if (better(c2,best)) best=c2;
            if (best.ok && best.off==0) break; } }
        {   // ── ①-a: **같은 정식화**(캐시 replan)를 싼 반복수로 → 재현 가능한가 + 얼마나 걸리나
          TamolsState so; so.prm = prm; StepCfg sv = sc; sv.iters = ON_ITERS;
          auto t0=std::chrono::high_resolution_clock::now();
          QpResult ro = replan(so, hsol, cell, N, z0, sv.vadv, fm, sv, nullptr);
          ON_ms += std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
          double dmax=0; for (int l=0;l<4;++l){ double d=std::hypot(so.p(l,0)-best.st.p(l,0), so.p(l,1)-best.st.p(l,1)); if(d>dmax)dmax=d; }
          ON_dsum+=dmax; if(dmax>ON_dmax)ON_dmax=dmax; ++ON_n;
          ON_okc += best.ok?1:0; ON_okon += (ro.eq_viol<1e-2 && ro.ineq_viol<1e-2)?1:0;
          if (dmax<0.02) ++ON_agree;
        }
        support_layers() = nullptr;

        size_t sp = ((size_t)v*N_PHX + a)*N_PHY + b;
        for (int l=0;l<4;++l) {
          FH[sp*12 + l*3 + 0] = (float)best.st.p(l,0);
          FH[sp*12 + l*3 + 1] = (float)best.st.p(l,1);
          FH[sp*12 + l*3 + 2] = (float)(best.st.p(l,2) - ground);       // 지면 기준 상대 z
          double tx=bx+best.st.p(l,0), ty=by+best.st.p(l,1);
          if (tx>0.75+TOL && tx<f.cxs.back()+0.5*f.size+TOL) { ++n_inf;
            int si; field_h(f,tx,ty,TOL,&si); if (si==-2) ++n_void; }
        }
        // ★base 궤적 샘플 (0..T 균등, phase 매핑)
        double T = 4*sc.phase_dur;
        for (int j=0;j<N_T;++j) {
          double t = T*j/(N_T-1);
          int k = std::min(3, (int)std::floor(t/sc.phase_dur));
          double tau = t - k*sc.phase_dur;
          Vector6d pp = best.st.pos_at(k, tau), vv = best.st.vel_at(k, tau);
          float* o = &BR[(sp*N_T + j)*6];
          o[0]=(float)pp(0); o[1]=(float)pp(1); o[2]=(float)(pp(2)-ground);
          o[3]=(float)vv(0); o[4]=(float)vv(1); o[5]=(float)vv(2);
        }
        OK[sp] = best.ok ? 1 : 0; n_ok += best.ok?1:0;
        if (b == N_PHY-1) { std::printf("  L%d vx=%.2f phx=%2d/%d  (feasible %ld/%ld)\n",
                                        L, sc.vadv, a+1, N_PHX, n_ok, n_cell); std::fflush(stdout); }
        dsum += std::hypot(best.st.p(0,0)-prm.hip_offsets(0,0), best.st.p(0,1)-prm.hip_offsets(0,1));
      }
      std::printf("L%d vx=%.2f 완료 (누적 %ld/%ld feasible, 보이드 %ld/%ld)\n",
                  L, sc.vadv, n_ok, n_cell, n_void, n_inf); std::fflush(stdout);
    }
    if (ON_n) { std::printf("\n★①-a  L%d  online(iters=%d support=%d) vs 캐시 |  발판차 평균 %.1f mm · 최대 %.1f mm  |  2cm내 일치 %ld/%ld (%.1f%%)  |  feasible 캐시 %ld / 온라인 %ld  |  %.1f ms/solve\n\n",
      L, ON_ITERS, ON_SUPPORT, 1000*ON_dsum/ON_n, 1000*ON_dmax, ON_agree, ON_n, 100.0*ON_agree/ON_n, ON_okc, ON_okon, ON_ms/ON_n);
      std::printf("        \xe2\x94\x94 \xea\xb0\x99\xec\x9d\x80 \xec\x85\x80\xec\x9d\x98 \xec\xba\x90\xec\x8b\x9c(20iter) solve = %.1f ms/cell\n\n", CACHE_ms/ON_n);
      ON_dsum=ON_dmax=ON_ms=CACHE_ms=0; ON_n=ON_okc=ON_okon=ON_agree=0; }
    char p[512]; std::snprintf(p,sizeof p,"%s/part_L%d.bin", outdir.c_str(), L);
    std::ofstream o(p, std::ios::binary);
    o.write((const char*)FH.data(), pf*4);
    o.write((const char*)BR.data(), pb*4);
    o.write((const char*)OK.data(), pk);
    std::printf("L%d 저장: %s  feasible %ld/%ld (%.1f%%)  보이드 발판 %ld/%ld (%.1f%%)  size=%.3f pitch=%.3f\n",
                L, p, n_ok, n_cell, 100.0*n_ok/std::max(1L,n_cell),
                n_void, n_inf, 100.0*n_void/std::max(1L,n_inf), f.size, pitch);
  }
  return 0;
}
