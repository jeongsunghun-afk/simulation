// cache_gen_stairs.cpp — 3D 계단 TAMOLS 캐시 (full TAMOLS on stairs: base 상승궤적+tread 발판+접촉)
//   격자 (vx × step_height)마다 계단 등반 플랜을 풀어 저장. env가 재앵커(z 포함)해 RL이 추종.
//   ★평지 gap과 달리 base z가 계단 따라 상승하고 발판이 tread에 안착 = full TAMOLS의 실질 가치.
//   빌드: g++ -O3 -std=c++17 cache_gen_stairs.cpp -I/usr/include/eigen3 -I<eiq_inc> -L<eiq_lib> -leiquadprog -o cache_gen_stairs
#include "tamols_online.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
using namespace tamols;

// 캐시 게이트 선택(env-var TAMOLS_GAIT). 기본=walk(정적안정, 기존 동작 그대로).
//   TAMOLS_GAIT=bound → bound(앞/뒤 쌍), =trot → trot. 미설정/그외 → walk.
static void apply_cache_gait(OnlineCfg& c) {
  const char* g = std::getenv("TAMOLS_GAIT");
  std::string s = g ? g : "walk";
  if (s == "bound")     { c.bound = true;  c.walk = false; }
  else if (s == "trot") { c.bound = false; c.walk = false; }
  else                  { c.bound = false; c.walk = true;  }
}

int main(int argc, char** argv) {
  // ★2026-09: GIAC_FIX(중력항)·GIAC_ORDER(CW 인접쌍)가 솔버 기본 ON으로 바뀜.
  //   이 오프라인 캐시 생성기의 출력은 이미 배포된 아티팩트라 **레거시 동작으로 고정**한다.
  //   (GIAC_FIX=1 로 실행하려면 이 두 줄을 지울 것)
  if (!getenv("TAMOLS_ALLOW_GIAC")) { setenv("GIAC_FIX", "0", 1); setenv("GIAC_ORDER", "0", 1); }
  std::string outdir = argc > 1 ? argv[1] : "stair_cache";

  std::vector<double> vx_vals = {0.2, 0.3, 0.4};                                  // 계단=저속
  std::vector<double> sh_vals = {0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15};  // env L0~L9 step_h
  double step_d = 0.35, dt = 0.02, phase_dur = 0.2, base_h = 0.52;
  int n_vx = vx_vals.size(), n_sh = sh_vals.size();

  int N = 41; double cell = 0.05, off = cell * N / 2.0; int ms = N;
  Params prm;

  // sample 크기 프로브
  int n_samp = 0;
  { Grid h = Grid::Zero(N, N); TamolsState st; OnlineCfg c; c.vadv=0.3; c.phase_dur=phase_dur; c.warm=false; apply_cache_gait(c);
    Eigen::Matrix<double,4,3> f0; for(int i=0;i<4;++i){f0(i,0)=prm.hip_offsets(i,0);f0(i,1)=prm.hip_offsets(i,1);f0(i,2)=0;}
    online_replan(st, h, cell, ms, base_h, 0,0,0, f0, c);
    std::vector<std::array<double,12>> S; std::vector<std::array<int,4>> C; Eigen::Matrix<double,4,3> F;
    sample_plan(st, dt, S, C, F); n_samp = (int)S.size(); }

  long ncell = (long)n_vx * n_sh;
  std::vector<float> fh(ncell*4*3, 0), bs(ncell*n_samp*12, 0), ct(ncell*n_samp*4, 0);
  int nfail = 0;
  auto cidx = [&](int i, int s){ return (long)(i*n_sh + s); };

  for (int i = 0; i < n_vx; ++i) {
    for (int s = 0; s < n_sh; ++s) {
      double sh = sh_vals[s];
      // 계단 heightmap: +x로 sh씩 상승 (base는 steady 등반구간 = 2단 위에서 시작)
      Grid h = Grid::Zero(N, N);
      for (int a = 0; a < N; ++a) for (int b = 0; b < N; ++b) {
        double x = a*cell - off; h(a,b) = (sh > 0 && x > -0.5) ? sh * std::floor((x+0.5)/step_d) : 0.0;
      }
      double z_here = sh * std::floor(0.5/step_d);      // base 현재 발밑 tread 높이
      double z0 = base_h + z_here;
      Eigen::Matrix<double,4,3> foot;
      for (int l = 0; l < 4; ++l) { foot(l,0)=prm.hip_offsets(l,0); foot(l,1)=prm.hip_offsets(l,1);
        foot(l,2)=bilinear_height(h, cell, ms, prm.hip_offsets(l,0), prm.hip_offsets(l,1)); }

      TamolsState st;
      OnlineCfg c; c.vadv = vx_vals[i]; c.phase_dur = phase_dur; c.straddle_init = false;
      apply_cache_gait(c);   // ★기본 walk 정식화(정적안정·한발씩): cmp_gait 검증=trot 대비 발판 tread에 10× 정밀·base z상승 정상(+0.06 vs ~0)·횡드리프트 절반. WALK_QLEG/COM_W는 온라인 컨트롤러용이라 오프라인 캐시선 제외(feas·z상승 악화). §P2.8. TAMOLS_GAIT=bound로 bound 캐시.
      c.z0_terrain = (sh > 0);   // 계단모드(base z밴드=z0±0.06 상대·롤피치 여유)
      QpResult r;
      for (int w = 0; w < 3; ++w) { c.warm = (w > 0); c.rti_iter = (w > 0) ? 5 : 60;
        r = online_replan(st, h, cell, ms, z0, 0, 0, 0, foot, c); }   // warm 체인(cold viol 우회)
      bool ok = (r.eq_viol < 1e-2 && r.ineq_viol < 1e-2); if (!ok) ++nfail;

      std::vector<std::array<double,12>> S; std::vector<std::array<int,4>> C; Eigen::Matrix<double,4,3> F;
      sample_plan(st, dt, S, C, F);
      long base = cidx(i, s);
      // 발판·base를 z_here 기준 상대로 저장(env가 현재 base z에 재앵커) — z_here 빼기
      for (int l = 0; l < 4; ++l) { fh[base*12+l*3+0]=(float)F(l,0); fh[base*12+l*3+1]=(float)F(l,1);
        fh[base*12+l*3+2]=(float)(F(l,2)-z_here); }
      int ns = std::min((int)S.size(), n_samp);
      for (int n = 0; n < ns; ++n) {
        for (int d = 0; d < 12; ++d) { double v = S[n][d]; if (d==2) v -= z0;   // base z를 z0 상대로
          bs[(base*n_samp+n)*12+d] = (float)v; }
        for (int l = 0; l < 4; ++l) ct[(base*n_samp+n)*4+l] = (float)C[n][l];
      }
    }
  }

  auto wbin = [&](const std::string& nm, const std::vector<float>& v){
    std::ofstream f(outdir+"/"+nm, std::ios::binary); f.write((const char*)v.data(), v.size()*sizeof(float)); };
  if (system(("mkdir -p "+outdir).c_str())) {}
  wbin("footholds.bin", fh); wbin("base.bin", bs); wbin("contacts.bin", ct);
  std::ofstream mj(outdir+"/meta.json");
  mj << "{\n  \"kind\": \"stair\", \"n_vx\": "<<n_vx<<", \"n_step_h\": "<<n_sh<<", \"n_samp\": "<<n_samp<<",\n";
  mj << "  \"step_depth\": "<<step_d<<", \"dt\": "<<dt<<", \"phase_dur\": "<<phase_dur<<", \"base_h\": "<<base_h<<",\n";
  mj << "  \"vx_vals\": ["; for(int i=0;i<n_vx;++i) mj<<(i?", ":"")<<vx_vals[i]; mj<<"],\n";
  mj << "  \"step_h_vals\": ["; for(int s=0;s<n_sh;++s) mj<<(s?", ":"")<<sh_vals[s]; mj<<"],\n";
  mj << "  \"footholds_shape\": ["<<n_vx<<", "<<n_sh<<", 4, 3],\n";
  mj << "  \"base_shape\": ["<<n_vx<<", "<<n_sh<<", "<<n_samp<<", 12],\n";
  mj << "  \"contacts_shape\": ["<<n_vx<<", "<<n_sh<<", "<<n_samp<<", 4],\n";
  mj << "  \"frame\": \"local: base start origin, +x fwd, foot/base z RELATIVE to base tread z; world = base_pos + Rz(yaw)*local (z incl)\"\n}\n";

  std::printf("=== 계단 TAMOLS 캐시 생성 완료 ===\n격자 %d(vx)×%d(step_h)=%ld셀, n_samp=%d, 실패 %d → %s\n",
              n_vx, n_sh, ncell, n_samp, nfail, outdir.c_str());
  // 검증: vx=0.3서 step_h별 발판 z(tread에 안착=step_h배수 상대)·base z상승
  int vi = 1;
  std::printf("\n[검증] vx=%.1f: step_h별 발판z(base tread 상대)·base z Δ(상승)\n", vx_vals[vi]);
  for (int s = 0; s < n_sh; ++s) {
    long b = cidx(vi, s);
    double bz_end = bs[(b*n_samp + (n_samp-1))*12 + 2];   // 마지막 sample base z(z0 상대)
    std::printf("  step_h=%.2f: 발판z FL=%+.3f FR=%+.3f RL=%+.3f RR=%+.3f | base zΔ(horizon끝)=%+.3f\n",
      sh_vals[s], fh[b*12+2], fh[b*12+5], fh[b*12+8], fh[b*12+11], bz_end);
  }
  return 0;
}
