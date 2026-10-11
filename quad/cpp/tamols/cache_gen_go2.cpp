// cache_gen_go2.cpp — Go2 기하(z0=0.34, hip=±0.1934/±0.142)용 오프라인 TAMOLS 발판 캐시 생성.
//   cache_gen.cpp(02_Leg z0=0.52/hip±0.225·0.14)의 Go2판. 변경점만:
//     ① prm.hip_offsets = Go2  ② z0 = 0.34  ③ fn/rn(foot-in-gap 게이트 기준 nominal 발판 x)을
//        02_Leg 하드코딩(0.51/0.06) 대신 Go2 flat-solve 결과서 자동 계산(기하 자기정합).
//   width/gapd 격자는 env gap 지형과 정합 위해 02_Leg와 동일 유지(재앵커+lookup 커리큘럼 보존).
//   출력: footholds.bin(f32 [n_vx,n_w,n_gapd,4,3]) + base/contacts.bin + meta.json.
//   빌드: g++ -O3 -std=c++17 cache_gen_go2.cpp -I/usr/include/eigen3 -I<eiq_inc> -L<eiq_lib> -leiquadprog -o cache_gen_go2
#include "tamols_online.hpp"
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
using namespace tamols;

// gap 캐시 gait 선택(env TAMOLS_GAIT). 기본=trot(기존 gap 동작 보존). =walk → walk(정적안정 한발씩).
static void apply_cache_gait(OnlineCfg& c) {
  const char* g = std::getenv("TAMOLS_GAIT");
  std::string s = g ? g : "trot";
  if (s == "walk")      { c.bound = false; c.walk = true;  }
  else if (s == "bound"){ c.bound = true;  c.walk = false; }
  else                  { c.bound = false; c.walk = false; }  // trot
}

int main(int argc, char** argv) {
  // ★2026-09: GIAC_FIX(중력항)·GIAC_ORDER(CW 인접쌍)가 솔버 기본 ON으로 바뀜.
  //   이 오프라인 캐시 생성기의 출력은 이미 배포된 아티팩트라 **레거시 동작으로 고정**한다.
  //   (GIAC_FIX=1 로 실행하려면 이 두 줄을 지울 것)
  if (!getenv("TAMOLS_ALLOW_GIAC")) { setenv("GIAC_FIX", "0", 1); setenv("GIAC_ORDER", "0", 1); }
  std::string outdir = argc > 1 ? argv[1] : "cache_go2";

  // ── 격자 축 (env gap 지형과 동일) ──
  std::vector<double> vx_vals = {0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<double> width_vals = {0.06, 0.10, 0.14, 0.18, 0.22, 0.26};  // env gap 폭 0.05~0.26 커버
  std::vector<double> gapd_vals; for (int j = 0; j < 41; ++j) gapd_vals.push_back(-0.2 + 0.025 * j);  // 로컬 gap_x0
  double dt = 0.02, phase_dur = 0.2, z0 = 0.34;   // ★Go2 base height
  int n_vx = vx_vals.size(), n_w = width_vals.size(), n_gapd = gapd_vals.size();

  Grid h; double cell; int ms; flat_costmap(h, cell, ms);   // 평지 costmap(gap은 cfg로 주입)
  Params prm;
  prm.hip_offsets <<   // ★Go2 hip(=발 공칭) 위치: x=±0.1934, y=±0.142(thigh lateral)
       0.1934,  0.142, 0.0,
       0.1934, -0.142, 0.0,
      -0.1934,  0.142, 0.0,
      -0.1934, -0.142, 0.0;
  Eigen::Matrix<double,4,3> foot0;
  for (int i = 0; i < 4; ++i) { foot0(i,0)=prm.hip_offsets(i,0); foot0(i,1)=prm.hip_offsets(i,1); foot0(i,2)=0; }

  // ── 평지 프로브: n_samp 확정 + nominal 발판 x(fn/rn) 자동 계산 ──
  int n_samp = 0;
  Eigen::Matrix<double,4,3> Fprobe;
  { TamolsState st; OnlineCfg c; c.vadv=0.4; c.phase_dur=phase_dur; c.warm=false; apply_cache_gait(c);
    online_replan(st, h, cell, ms, z0, 0, 0, 0, foot0, c);
    std::vector<std::array<double,12>> S; std::vector<std::array<int,4>> C;
    sample_plan(st, dt, S, C, Fprobe); n_samp = (int)S.size(); }
  double fn = Fprobe(0,0);   // 앞발(FL) 공칭 x  — foot-in-gap 게이트 기준
  double rn = Fprobe(2,0);   // 뒷발(RL) 공칭 x
  std::printf("[Go2 기하] z0=%.2f hip_x=±%.4f hip_y=±%.4f | 자동 게이트 fn(앞)=%.3f rn(뒤)=%.3f n_samp=%d\n",
              z0, 0.1934, 0.142, fn, rn, n_samp);

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
        c.warm = warm; c.straddle_init = false; apply_cache_gait(c);
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
  mj << "  \"model\": \"go2\", \"hip_x\": 0.1934, \"hip_y\": 0.142,\n";
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
  mj << "  \"gate\": \"gap avoidance only when nominal foot(front=" << fn << "/rear=" << rn << ") falls in gap; else nominal (reachable)\"\n";
  mj << "}\n";

  std::printf("=== Go2 TAMOLS 캐시 생성 완료 ===\n");
  std::printf("격자 %d(vx) × %d(width) × %d(gapd) = %ld 셀, n_samp=%d, 실패 %d\n", n_vx, n_w, n_gapd, ncell, n_samp, nfail);
  std::printf("출력: %s/{footholds,base,contacts}.bin + meta.json\n", outdir.c_str());

  // ── 검증: width=0.18, vx=0.4 컬럼서 발판 x (도달가능·gap회피) ──
  int vi = 2, wi18 = 3;
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
