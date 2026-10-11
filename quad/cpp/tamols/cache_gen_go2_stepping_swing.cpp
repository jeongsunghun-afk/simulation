// cache_gen_go2_stepping_swing.cpp — TASK P2: per-footfall SWING TRAJECTORY reference exporter
//   (stepping-stone fields, RL swing-phase 추종용. SPATIAL/per-footfall — 시간 아님, RL이 자기 게이트
//   클록의 swing 진행도 s∈[0,1]로 소비).
//
//   무엇이 TAMOLS-native이고 무엇이 근사인가(정직 명세):
//     · TAMOLS-native: ① 발판/시퀀스 = 이미 검증된 계획(footfalls_L*.csv=TAMOLS 스윕 per-cycle,
//       plan_astarv3_L*.csv=trap A* 체인). ② 지형 레이어 = terrain_proc.hpp process_height_maps의
//       h_s2 "virtual floor"(median→edge mask→dilate→local max→gaussian σ2) — TAMOLS 논문이 스윙
//       무충돌을 explicit constraint 없이 보장하는 그 레이어. ③ xy 스윙 형상 = tamols_track.hpp
//       swing_pos의 linear interp(liftoff→touchdown).
//     · 근사(TAMOLS 솔버는 스윙 스플라인을 출력하지 않음 — 결정변수가 base스플라인+발판+ε뿐):
//       수직 프로파일 = 2-세그먼트 cubic Hermite(끝점 0-기울기, apex 0-기울기), apex 높이 =
//       max(liftoff_z, touchdown_z, path floor max)+0.08, apex 위상 = step-up이면 이른쪽·step-down이면
//       늦은쪽(0.5−0.25·tanh(Δz/0.10), 0.1그리드 스냅=apex가 실제 export 샘플), floor(s)=
//       max(정확지형, h_s2)+0.025로 내부 샘플 클립(=virtual-floor 클리어런스).
//
//   출력: stepping_go2/swing_L{0..9}.csv · swing_trapv3_L{0..9}.csv
//         (columns: level,leg,footfall_order,s,x,y,z — leg∈{FL,FR,HL,HR}, footfall_order=per-leg 0-based,
//          liftoff=그 다리의 직전 발판(order 0은 스폰 스탠스=hip 명목·strip z0.15), K=11 샘플 s=0..1)
//         + swing_meta.json
//   검증(레벨별): 내부샘플 z ≥ 정확지형+0.02 · apex ≥ max(liftoff,touchdown)+0.05 및 경로지형max+0.05 ·
//         끝점=발판 정확 · 수평진행 단조. violations=0 필수.
//
//   빌드: g++ -O3 -std=c++17 cache_gen_go2_stepping_swing.cpp -I/usr/include/eigen3 -o cache_gen_go2_stepping_swing
//   실행: ./cache_gen_go2_stepping_swing [outdir=stepping_go2]     (tamols/ 디렉토리에서)
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

static const int    K          = 11;     // s = 0, 0.1, ..., 1.0
static const double CLIP_CLEAR = 0.025;  // 내부 샘플 floor 클리어런스(검증 0.02보다 여유)
static const double APEX_CLEAR = 0.08;   // apex 상승(검증 0.05보다 여유)
static const double VAL_CLEAR  = 0.02, VAL_APEX = 0.05;

// ── 필드(돌 리스트 + 스폰 스트립) — 정규격자(stones_L*)·불규칙(trapv3) 공용, 겹침=max ──
struct Stone { double cx, cy, half, top; };
struct Field { std::vector<Stone> st; };

static bool load_stones(const std::string& path, Field& f) {
  std::ifstream in(path); if (!in) return false;
  std::string line; std::getline(in, line);                    // header idx,ix,iy,cx,cy,size,top_z
  while (std::getline(in, line)) {
    double a[7];
    if (std::sscanf(line.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf", a,a+1,a+2,a+3,a+4,a+5,a+6) != 7) continue;
    f.st.push_back({a[3], a[4], 0.5*a[5], a[6]});
  }
  return !f.st.empty();
}

// 정확 지형높이(lane-local): 스트립 [-0.75,0.75]×|y|≤1 top 0.15 · 돌 top(겹침 max) · 그외 평면 0
static double field_h(const Field& f, double x, double y) {
  double h = 0.0;
  if (x >= -0.75 && x <= 0.75 && std::fabs(y) <= 1.0) h = 0.15;
  for (const auto& s : f.st)
    if (std::fabs(x - s.cx) <= s.half && std::fabs(y - s.cy) <= s.half) h = std::max(h, s.top);
  return h;
}

// ── h_s2 virtual-floor 레이어: 코리도 전체 래스터 → terrain_proc process_height_maps ──
struct Layer { Grid h; double x0, y0, cell; int R, C; };
static Layer make_hs2(const Field& f) {
  Layer L; L.cell = 0.02; L.x0 = -1.0; L.y0 = -1.2; L.R = 301; L.C = 121;   // x[-1,5]·y[-1.2,1.2]
  Grid raw(L.R, L.C);
  for (int i = 0; i < L.R; ++i) for (int j = 0; j < L.C; ++j)
    raw(i, j) = field_h(f, L.x0 + i * L.cell, L.y0 + j * L.cell);
  TerrainLayers tl = process_height_maps(raw, L.cell);         // σ1=1, σ2=2 (셀단위, TAMOLS 기본)
  L.h = tl.h_s2;
  return L;
}
static double sample_layer(const Layer& L, double x, double y) {
  double u = (x - L.x0) / L.cell, v = (y - L.y0) / L.cell;
  u = std::max(0.0, std::min((double)L.R - 1.001, u));
  v = std::max(0.0, std::min((double)L.C - 1.001, v));
  int i = (int)u, j = (int)v; double fu = u - i, fv = v - j;
  return L.h(i,j)*(1-fu)*(1-fv) + L.h(i+1,j)*fu*(1-fv) + L.h(i,j+1)*(1-fu)*fv + L.h(i+1,j+1)*fu*fv;
}

struct P3 { double x, y, z; };
struct Swing { int leg, order; std::array<double, K> xs, ys, zs; P3 p0, p1; };
static const char* LEGN[4] = {"FL", "FR", "HL", "HR"};
static int leg_idx(const std::string& n) {
  for (int i = 0; i < 4; ++i) if (n == LEGN[i]) return i;
  return -1;
}
// Go2 hip 명목(planner FL,FR,RL→HL,RR→HR) = 스폰 스탠스 xy
static const double HIPX[4] = { 0.1934, 0.1934, -0.1934, -0.1934 };
static const double HIPY[4] = { 0.142, -0.142,  0.142,  -0.142 };

static double smoothstep(double u) { return u*u*(3.0 - 2.0*u); }
static double floor_at(const Field& f, const Layer& L, const P3& p0, const P3& p1, double s) {
  double x = p0.x + s*(p1.x - p0.x), y = p0.y + s*(p1.y - p0.y);
  return std::max(field_h(f, x, y), sample_layer(L, x, y));
}

// ── 스윙 참조 생성: liftoff p0 → touchdown p1, 지형-aware 수직 프로파일 ──
static Swing build_swing(const Field& f, const Layer& L, int leg, int order, const P3& p0, const P3& p1) {
  Swing w; w.leg = leg; w.order = order; w.p0 = p0; w.p1 = p1;
  double H = std::max(p0.z, p1.z);                             // 경로 floor 최대(fine 샘플)
  for (int m = 0; m <= 100; ++m) H = std::max(H, floor_at(f, L, p0, p1, m / 100.0));
  double za = H + APEX_CLEAR;
  double dz = p1.z - p0.z;                                     // apex 위상: step-up 이른쪽·step-down 늦은쪽
  double sa = 0.5 - 0.25 * std::tanh(dz / 0.10);
  sa = std::max(0.3, std::min(0.7, sa));
  sa = std::round(sa * 10.0) / 10.0;                           // 0.1그리드 스냅 → apex=실제 export 샘플
  for (int k = 0; k < K; ++k) {
    double s = k / (double)(K - 1);
    w.xs[k] = p0.x + s*(p1.x - p0.x);
    w.ys[k] = p0.y + s*(p1.y - p0.y);
    double z = (s <= sa) ? p0.z + (za - p0.z) * smoothstep(s / sa)
                         : za  - (za - p1.z) * smoothstep((s - sa) / (1.0 - sa));
    if (k > 0 && k < K - 1) z = std::max(z, floor_at(f, L, p0, p1, s) + CLIP_CLEAR);   // virtual-floor 클립
    w.zs[k] = z;
  }
  w.zs[0] = p0.z; w.zs[K-1] = p1.z;                            // 끝점 = 발판 정확
  return w;
}

// ── 체인 로더: (a) footfalls_L*.csv (per cycle·leg, TAMOLS 스윕) ──
static bool load_chains_footfalls(const std::string& path, std::array<std::vector<P3>,4>& ch) {
  std::ifstream in(path); if (!in) return false;
  std::string line; std::getline(in, line);   // cycle,leg,foot_x,foot_y,foot_z_solver,foot_z_snap,...
  int n = 0;
  while (std::getline(in, line)) {
    std::stringstream ss(line); std::string tok; std::vector<std::string> c;
    while (std::getline(ss, tok, ',')) c.push_back(tok);
    if (c.size() < 6) continue;
    int l = leg_idx(c[1]); if (l < 0) continue;
    ch[l].push_back({std::atof(c[2].c_str()), std::atof(c[3].c_str()), std::atof(c[5].c_str())});   // foot_z_snap
    ++n;
  }
  return n > 0;
}
// ── 체인 로더: (b) plan_astarv3_L*.csv (order,leg,stone_idx,cx,cy,top_z — A* 체인) ──
static bool load_chains_astar(const std::string& path, std::array<std::vector<P3>,4>& ch) {
  std::ifstream in(path); if (!in) return false;
  std::string line; std::getline(in, line);
  int n = 0;
  while (std::getline(in, line)) {
    std::stringstream ss(line); std::string tok; std::vector<std::string> c;
    while (std::getline(ss, tok, ',')) c.push_back(tok);
    if (c.size() < 6) continue;
    int l = leg_idx(c[1]); if (l < 0) continue;
    ch[l].push_back({std::atof(c[3].c_str()), std::atof(c[4].c_str()), std::atof(c[5].c_str())});
    ++n;
  }
  return n > 0;
}

int main(int argc, char** argv) {
  // ★2026-09: GIAC_FIX(중력항)·GIAC_ORDER(CW 인접쌍)가 솔버 기본 ON으로 바뀜.
  //   이 오프라인 캐시 생성기의 출력은 이미 배포된 아티팩트라 **레거시 동작으로 고정**한다.
  //   (GIAC_FIX=1 로 실행하려면 이 두 줄을 지울 것)
  if (!getenv("TAMOLS_ALLOW_GIAC")) { setenv("GIAC_FIX", "0", 1); setenv("GIAC_ORDER", "0", 1); }
  std::string outdir = argc > 1 ? argv[1] : "stepping_go2";
  struct Set { const char* stones_fmt; const char* plan_fmt; const char* out_fmt; bool astar; const char* name; };
  Set sets[2] = {
    { "%s/stones_L%d.csv",        "%s/footfalls_L%d.csv",     "%s/swing_L%d.csv",        false, "safe varied (stones_L*)" },
    { "%s/stones_trapv3_L%d.csv", "%s/plan_astarv3_L%d.csv",  "%s/swing_trapv3_L%d.csv", true,  "trap v3 (stones_trapv3_L*)" },
  };
  char buf[512];
  bool all_ok = true;

  for (int si = 0; si < 2; ++si) {
    std::printf("=== %s → %s ===\n", sets[si].name, sets[si].astar ? "swing_trapv3_L*.csv" : "swing_L*.csv");
    std::printf("%3s | %5s | %8s | %8s | %8s | %8s | %4s\n",
                "lvl", "n_ff", "maxApex", "minClr", "minApexM", "maxEndEr", "viol");
    for (int lvl = 0; lvl < 10; ++lvl) {
      Field f;
      std::snprintf(buf, sizeof buf, sets[si].stones_fmt, outdir.c_str(), lvl);
      if (!load_stones(buf, f)) { std::printf("L%d: stones 로드 실패 (%s)\n", lvl, buf); return 1; }
      std::array<std::vector<P3>,4> ch;
      std::snprintf(buf, sizeof buf, sets[si].plan_fmt, outdir.c_str(), lvl);
      bool ok = sets[si].astar ? load_chains_astar(buf, ch) : load_chains_footfalls(buf, ch);
      if (!ok) { std::printf("L%d: plan 로드 실패 (%s)\n", lvl, buf); return 1; }
      Layer L = make_hs2(f);

      std::vector<Swing> swings;
      for (int l = 0; l < 4; ++l) {
        P3 prev{HIPX[l], HIPY[l], field_h(f, HIPX[l], HIPY[l])};   // 스폰 스탠스(스트립 0.15)
        for (size_t k = 0; k < ch[l].size(); ++k) {
          swings.push_back(build_swing(f, L, l, (int)k, prev, ch[l][k]));
          prev = ch[l][k];
        }
      }

      // ── 검증 ──
      int viol = 0; double max_apex = -1e18, min_clr = 1e18, min_apexm = 1e18, max_ee = 0;
      for (const auto& w : swings) {
        double apex = -1e18;
        for (int k = 0; k < K; ++k) apex = std::max(apex, w.zs[k]);
        max_apex = std::max(max_apex, apex);
        // (1) 내부샘플 z ≥ 정확지형 + 0.02
        for (int k = 1; k < K - 1; ++k) {
          double clr = w.zs[k] - field_h(f, w.xs[k], w.ys[k]);
          min_clr = std::min(min_clr, clr);
          if (clr < VAL_CLEAR - 1e-12) ++viol;
        }
        // (2) apex ≥ max(liftoff,touchdown)+0.05 및 경로 정확지형 max+0.05
        double Hex = std::max(w.p0.z, w.p1.z);
        for (int m = 0; m <= 100; ++m) {
          double s = m / 100.0;
          Hex = std::max(Hex, field_h(f, w.p0.x + s*(w.p1.x-w.p0.x), w.p0.y + s*(w.p1.y-w.p0.y)));
        }
        double am = apex - std::max(std::max(w.p0.z, w.p1.z), Hex);
        min_apexm = std::min(min_apexm, am);
        if (am < VAL_APEX - 1e-12) ++viol;
        // (3) 끝점 = 발판 정확
        double e0 = std::max({std::fabs(w.xs[0]-w.p0.x), std::fabs(w.ys[0]-w.p0.y), std::fabs(w.zs[0]-w.p0.z)});
        double e1 = std::max({std::fabs(w.xs[K-1]-w.p1.x), std::fabs(w.ys[K-1]-w.p1.y), std::fabs(w.zs[K-1]-w.p1.z)});
        max_ee = std::max({max_ee, e0, e1});
        if (e0 > 1e-9 || e1 > 1e-9) ++viol;
        // (4) 수평진행 단조(liftoff→touchdown 방향 내적 ≥ 0)
        double dx = w.p1.x - w.p0.x, dy = w.p1.y - w.p0.y;
        for (int k = 0; k + 1 < K; ++k)
          if ((w.xs[k+1]-w.xs[k])*dx + (w.ys[k+1]-w.ys[k])*dy < -1e-12) ++viol;
      }
      if (viol) all_ok = false;

      // ── 출력 CSV ──
      std::snprintf(buf, sizeof buf, sets[si].out_fmt, outdir.c_str(), lvl);
      std::ofstream of(buf); of.precision(10);
      of << "level,leg,footfall_order,s,x,y,z\n";
      for (const auto& w : swings)
        for (int k = 0; k < K; ++k)
          of << lvl << "," << LEGN[w.leg] << "," << w.order << "," << (k / (double)(K-1))
             << "," << w.xs[k] << "," << w.ys[k] << "," << w.zs[k] << "\n";

      std::printf("%3d | %5d | %8.3f | %8.4f | %8.4f | %8.1e | %4d%s\n",
                  lvl, (int)swings.size(), max_apex, min_clr, min_apexm, max_ee, viol,
                  viol ? "  ★VIOLATION" : "");
    }
  }

  // ── 메타 ──
  { std::ofstream mj(outdir + "/swing_meta.json");
    mj << "{\n  \"kind\": \"per_footfall_swing_references\", \"model\": \"go2\", \"K\": " << K << ",\n";
    mj << "  \"files\": \"swing_L{0..9}.csv(safe varied, from footfalls_L*.csv per-cycle TAMOLS sweep), "
          "swing_trapv3_L{0..9}.csv(trap v3, from plan_astarv3_L*.csv A* chains)\",\n";
    mj << "  \"columns\": \"level,leg(FL|FR|HL|HR),footfall_order(per-leg 0-based),s(swing phase 0..1, 11 pts),x,y,z\",\n";
    mj << "  \"frame\": \"lane-local: x=corridor(spawn strip[-0.75,0.75] top z=0.15, stones x in[0.75,4.25]); "
          "y rel lane center; z absolute(void plane=0) — plan_meta.json과 동일\",\n";
    mj << "  \"liftoff_rule\": \"footfall_order k의 liftoff=그 다리의 (k-1)번째 발판, k=0은 스폰 스탠스 "
          "hip(±0.1934,±0.142) z=0.15(strip)\",\n";
    mj << "  \"native\": \"footholds+sequence(TAMOLS sweep/A* plan), h_s2 virtual floor(terrain_proc "
          "process_height_maps σ1=1 σ2=2 cells, cell 0.02), linear xy(tamols_track swing form)\",\n";
    mj << "  \"approximated\": \"vertical profile: 2-seg cubic Hermite(zero-slope ends+apex), apex_z="
          "max(liftoff,touchdown,path floor max)+" << APEX_CLEAR << ", apex phase 0.5-0.25*tanh(dz/0.10) "
          "snapped to 0.1 grid(step-up early/step-down late), interior samples clipped to "
          "max(exact terrain,h_s2)+" << CLIP_CLEAR << " — TAMOLS solver does NOT output swing splines"
          "(decision vars=base spline+footholds+eps only)\",\n";
    mj << "  \"validation\": \"interior z>=exact terrain+0.02, apex>=max(liftoff,touchdown)+0.05 AND "
          ">=path exact-terrain max+0.05, endpoints exact(<1e-9), monotone horizontal progress\"\n}\n";
  }
  std::printf("출력: %s/swing_L*.csv swing_trapv3_L*.csv swing_meta.json — %s\n",
              outdir.c_str(), all_ok ? "전 레벨 violations=0" : "★위반 있음");
  return all_ok ? 0 : 2;
}
