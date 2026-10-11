// cache_gen.cpp — 오프라인 TAMOLS 플랜 캐시 생성 (DTC offline: 학습 전 사전풀이)
//   격자 (vx × gap_width × gap거리)마다 로컬 TAMOLS 플랜을 풀어 저장. 학습 중 env가 조회+재앵커해 RL에 공급.
//   ★로컬 프레임: base 시작=원점·전방+x. 발판=base중심 기준. env 재앵커: world = base_pos + Rz(yaw)·local.
//   ★도달성: straddle_init 끔 + gap 회피는 nominal 발판이 실제 gap에 빠질 때만(foot-in-gap 게이트) → 먼 gap=nominal.
//   ★width 축: env gap 폭(0.05~0.26)을 커버 → 재앵커+lookup이 폭까지 정합(커리큘럼 보존).
//   출력: footholds.bin(f32 [n_vx,n_w,n_gapd,4,3]) + base/contacts.bin + meta.json.
//   빌드: g++ -O3 -std=c++17 cache_gen.cpp -I/usr/include/eigen3 -I<eiq_inc> -L<eiq_lib> -leiquadprog -o cache_gen
#include "tamols_online.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
using namespace tamols;

int main(int argc, char** argv) {
  // ★2026-09: GIAC_FIX(중력항)·GIAC_ORDER(CW 인접쌍)가 솔버 기본 ON으로 바뀜.
  //   이 오프라인 캐시 생성기의 출력은 이미 배포된 아티팩트라 **레거시 동작으로 고정**한다.
  //   (GIAC_FIX=1 로 실행하려면 이 두 줄을 지울 것)
  if (!getenv("TAMOLS_ALLOW_GIAC")) { setenv("GIAC_FIX", "0", 1); setenv("GIAC_ORDER", "0", 1); }
  std::string outdir = argc > 1 ? argv[1] : "cache";

  // ── 격자 축 ──
  std::vector<double> vx_vals = {0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<double> width_vals = {0.06, 0.10, 0.14, 0.18, 0.22, 0.26};  // env gap 폭 0.05~0.26 커버
  std::vector<double> gapd_vals; for (int j = 0; j < 41; ++j) gapd_vals.push_back(-0.2 + 0.025 * j);  // 로컬 gap_x0
  double dt = 0.02, phase_dur = 0.2, z0 = 0.52;
  // ★foot-in-gap 게이트 공칭 발판 x. 배포된 quad17 gap 캐시(작동, terrain_level~9)와 일치시키려 하드코딩 유지.
  //   ⚠️ 알려진 불일치(2026-08-20): 실제 flat-solve 공칭은 fn≈0.375·rn≈-0.185(sample_plan st.p 실측)로 아래 값과 다름.
  //   게이트가 실제 공칭과 어긋나 있으나 RL이 robustify해 작동 중. 자동화(Go2판처럼 Fprobe)는 캐시를 바꾸므로
  //   quad17 gap RL 재검증과 함께 별도 진행할 것(발판_표준_TAMOLS_방향.md #4). 블라인드 변경 금지.
  double fn = 0.51, rn = 0.06;
  int n_vx = vx_vals.size(), n_w = width_vals.size(), n_gapd = gapd_vals.size();

  Grid h; double cell; int ms; flat_costmap(h, cell, ms);   // 평지 costmap(gap은 cfg로 주입)
  Params prm;
  Eigen::Matrix<double,4,3> foot0;
  for (int i = 0; i < 4; ++i) { foot0(i,0)=prm.hip_offsets(i,0); foot0(i,1)=prm.hip_offsets(i,1); foot0(i,2)=0; }

  int n_samp = 0;   // sample_plan 크기 확정용 1회 프로브
  { TamolsState st; OnlineCfg c; c.vadv=0.4; c.phase_dur=phase_dur; c.warm=false;
    online_replan(st, h, cell, ms, z0, 0, 0, 0, foot0, c);
    std::vector<std::array<double,12>> S; std::vector<std::array<int,4>> C; Eigen::Matrix<double,4,3> F;
    sample_plan(st, dt, S, C, F); n_samp = (int)S.size(); }

  long ncell = (long)n_vx * n_w * n_gapd;
  std::vector<float> fh(ncell*4*3, 0), bs(ncell*n_samp*12, 0), ct(ncell*n_samp*4, 0);
  int nfail = 0;
  auto cidx = [&](int i, int wi, int j){ return ((long)(i*n_w + wi)*n_gapd + j); };

  for (int i = 0; i < n_vx; ++i) {
    for (int wi = 0; wi < n_w; ++wi) {
      TamolsState st; bool warm = false;         // (vx,width)마다 cold 1회 후 warm 체인
      double gw = width_vals[wi];
      for (int j = 0; j < n_gapd; ++j) {
        double gd = gapd_vals[j];
        OnlineCfg c; c.vadv = vx_vals[i]; c.phase_dur = phase_dur; c.rti_iter = warm ? 5 : 60;
        c.warm = warm; c.straddle_init = false;
        // gap 회피는 nominal 발판(앞 fn·뒤 rn)이 실제 gap에 빠질 때만 → 먼 gap=nominal(도달가능).
        bool hit_front = (gd <= fn + 0.03 && gd + gw >= fn - 0.03);
        bool hit_rear  = (gd <= rn + 0.03 && gd + gw >= rn - 0.03);
        if (hit_front || hit_rear) { c.gap_x0 = gd; c.gap_x1 = gd + gw; }
        else { c.gap_x0 = -1; c.gap_x1 = -1; }

        QpResult r = online_replan(st, h, cell, ms, z0, 0, 0, 0, foot0, c);
        warm = true;
        bool ok = (r.eq_viol < 1e-2 && r.ineq_viol < 1e-2);
        if (!ok) { c.warm = false; c.rti_iter = 60; r = online_replan(st, h, cell, ms, z0, 0, 0, 0, foot0, c);
                   ok = (r.eq_viol < 1e-2 && r.ineq_viol < 1e-2); }
        if (!ok) ++nfail;

        std::vector<std::array<double,12>> S; std::vector<std::array<int,4>> C; Eigen::Matrix<double,4,3> F;
        sample_plan(st, dt, S, C, F);
        long base = cidx(i, wi, j);
        for (int l = 0; l < 4; ++l) for (int d = 0; d < 3; ++d) fh[base*12 + l*3 + d] = (float)F(l,d);
        int ns = std::min((int)S.size(), n_samp);
        for (int n = 0; n < ns; ++n) {
          for (int d = 0; d < 12; ++d) bs[(base*n_samp + n)*12 + d] = (float)S[n][d];
          for (int l = 0; l < 4; ++l) ct[(base*n_samp + n)*4 + l] = (float)C[n][l];
        }
      }
    }
  }

  // ── 쓰기 ──
  auto wbin = [&](const std::string& name, const std::vector<float>& v){
    std::ofstream f(outdir + "/" + name, std::ios::binary);
    f.write((const char*)v.data(), v.size()*sizeof(float)); };
  if (system(("mkdir -p " + outdir).c_str())) {}
  wbin("footholds.bin", fh); wbin("base.bin", bs); wbin("contacts.bin", ct);

  std::ofstream mj(outdir + "/meta.json");
  mj << "{\n";
  mj << "  \"n_vx\": " << n_vx << ", \"n_width\": " << n_w << ", \"n_gapd\": " << n_gapd << ", \"n_samp\": " << n_samp << ",\n";
  mj << "  \"dt\": " << dt << ", \"phase_dur\": " << phase_dur << ", \"z0\": " << z0 << ",\n";
  mj << "  \"vx_vals\": ["; for (int i=0;i<n_vx;++i) mj << (i?", ":"") << vx_vals[i]; mj << "],\n";
  mj << "  \"width_vals\": ["; for (int i=0;i<n_w;++i) mj << (i?", ":"") << width_vals[i]; mj << "],\n";
  mj << "  \"gapd_vals\": ["; for (int j=0;j<n_gapd;++j) mj << (j?", ":"") << gapd_vals[j]; mj << "],\n";
  mj << "  \"footholds_shape\": [" << n_vx << ", " << n_w << ", " << n_gapd << ", 4, 3],\n";
  mj << "  \"base_shape\": [" << n_vx << ", " << n_w << ", " << n_gapd << ", " << n_samp << ", 12],\n";
  mj << "  \"contacts_shape\": [" << n_vx << ", " << n_w << ", " << n_gapd << ", " << n_samp << ", 4],\n";
  mj << "  \"foot_order\": \"FL,FR,RL,RR\", \"frame\": \"local: base start=origin, +x fwd; world = base_pos + Rz(yaw)*local\",\n";
  mj << "  \"gapd_meaning\": \"local x-dist base-center to gap near-edge (gap_x0). gap spans [gapd, gapd+width]\",\n";
  mj << "  \"gate\": \"gap avoidance only when nominal foot(front~0.51/rear~0.06) falls in gap; else nominal (reachable)\"\n";
  mj << "}\n";

  std::printf("=== TAMOLS 캐시 생성 완료 ===\n");
  std::printf("격자 %d(vx) × %d(width) × %d(gapd) = %ld 셀, n_samp=%d, 실패 %d\n", n_vx, n_w, n_gapd, ncell, n_samp, nfail);
  std::printf("출력: %s/{footholds,base,contacts}.bin + meta.json\n", outdir.c_str());

  // ── 검증: width=0.18, vx=0.4 컬럼서 발판 x (도달가능·gap회피) ──
  int vi = 2, wi18 = 3;  // vx=0.4, width=0.18
  std::printf("\n[검증] vx=%.1f width=%.2f: gapd별 발판 x (FL,FR,RL,RR) — foot-in-gap서만 straddle, 그외 nominal\n",
              vx_vals[vi], width_vals[wi18]);
  for (int j = 0; j < n_gapd; j += 4) {
    long b = cidx(vi, wi18, j);
    std::printf("  gapd=%+.2f (gap[%.2f,%.2f]): FL=%.3f FR=%.3f RL=%.3f RR=%.3f\n",
      gapd_vals[j], gapd_vals[j], gapd_vals[j]+width_vals[wi18],
      fh[b*12+0], fh[b*12+3], fh[b*12+6], fh[b*12+9]);
  }
  return 0;
}
