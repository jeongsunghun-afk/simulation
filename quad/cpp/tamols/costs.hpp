// costs.hpp — TAMOLS 비용 (Drake costs.py 활성 3종 C++ 포팅)
//   활성(setup_costs_and_constraints): tracking · foothold_on_ground · nominal_kinematic.
//   (base_pose_alignment·edge_avoidance·previous_solution·smoothness 는 Drake서 주석처리=비활성.)
//   검증: Drake 해에서 각 비용 EvalBinding 값과 C++ 재계산 대조(≈0 오차).
#pragma once
#include "constraints.hpp"   // R_B, TamolsState
#include <cmath>
#include <vector>

namespace tamols {

#ifndef TAMOLS_GRID_ALIAS
#define TAMOLS_GRID_ALIAS
using Grid = Eigen::MatrixXd;   // terrain_proc.hpp와 동일 별칭(중복 include 안전)
#endif

// ── 양선형 높이 보간 (Drake evaluate_height_at_symbolic_xy 정합) ──
//   i=(x+off)/cell, j=(y+off)/cell, off=cell·map_size/2. Σ h[k,l](1−|i−k|)(1−|j−l|), |i−k|<1 & |j−l|<1.
//   grid[k,l]: k=x축 인덱스(행), l=y축 인덱스(열).
inline double bilinear_height(const Grid& h, double cell, int map_size, double x, double y) {
  double off = cell * map_size / 2.0;
  double i = (x + off) / cell, j = (y + off) / cell;
  const int m = (int)h.rows(), n = (int)h.cols();
  double tot = 0;
  int k0 = (int)std::floor(i), l0 = (int)std::floor(j);
  for (int k = k0; k <= k0 + 1; ++k) {
    if (k < 0 || k >= m) continue;
    double wx = 1.0 - std::fabs(i - k); if (wx <= 0) continue;
    for (int l = l0; l <= l0 + 1; ++l) {
      if (l < 0 || l >= n) continue;
      double wy = 1.0 - std::fabs(j - l); if (wy <= 0) continue;
      tot += h(k, l) * wx * wy;
    }
  }
  return tot;
}

// ── tracking 비용 (Drake add_tracking_cost): x속도만 추종 ──
//   Σ_phase Σ_{s=0}^{S-1} (2 T_k/S)(vel_x(τ_s) − ref_vel_x)²,  τ_s = linspace(0,T_k,S+1)[:S]
inline double tracking_cost(const TamolsState& st) {
  const int S = 6;                       // tau_sampling_rate
  double c = 0;
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    for (int s = 0; s < S; ++s) {
      double tau = Tk * (double)s / (double)S;
      double vx = st.vel_at(k, tau)(0);
      double w = 2.0 * Tk / (double)S;
      double d = vx - st.ref_vel(0);
      c += w * d * d;
    }
  }
  return c;
}

// ── foothold_on_ground 비용 (Drake add_foothold_on_ground_cost) ──
//   Σ_i 100 (h(p_i.x,p_i.y) − p_i.z)²
inline double foothold_on_ground_cost(const TamolsState& st, const Grid& h, double cell, int map_size) {
  double c = 0;
  for (int n = 0; n < st.ncyc(); ++n) for (int i = 0; i < 4; ++i) {     // ★N 사이클 전부
    Vector3d pi = st.fpos(n, i);
    double d = bilinear_height(h, cell, map_size, pi(0), pi(1)) - pi(2);
    c += 100.0 * d * d;
  }
  return c;
}

// ── nominal_kinematic 비용 (Drake add_nominal_kinematic_cost) ──
//   마지막 phase, τ=T/2. Σ_i 20 |p_B + R_B·hip_i − l_des − p_i|²,  l_des=[0,0,h_des], p_i=at_des?p:p_meas
inline double nominal_kinematic_cost(const TamolsState& st) {
  Vector3d l_des(0, 0, st.prm.h_des);
  double c = 0;
  for (int n = 0; n < st.ncyc(); ++n) {              // ★사이클마다 그 사이클 마지막 phase 에서
    int k = st.cyc_end_phase(n);
    double Tk = st.gait[k].duration, tau = Tk / 2.0;
    Vector6d pose = st.pos_at(k, tau);
    Vector3d pB = pose.head<3>();
    Eigen::Matrix3d R = R_B(pose.tail<3>());
    for (int i = 0; i < 4; ++i) {
      Vector3d bml = pB + R * st.prm.hip_offsets.row(i).transpose() - l_des;
      Vector3d e = bml - st.foot_at(k, i);
      c += 20.0 * e.dot(e);
    }
  }
  return c;
}

// ── ★edge_avoidance 비용 (2026-09 구현·게이트) ────────────────────────────────
//   정식화 = Drake 프로토타입 `costs.py::add_edge_avoidance_cost` 그대로:
//       cost = Σ_legs [ 3·|∇h(p_i)|²  +  |∇h_s1(p_i)|² ]
//   ∇h·∇h_s1 = 미리 계산된 gradient 격자(terrain_proc.hpp compute_gradients)를
//   발판 xy에서 **양선형 보간**한 값(Drake evaluate_height_at_symbolic_xy 정합).
//   돌 가장자리에서 |∇h|이 크고 돌 중앙 평탄면에서 ≈0 → "가장자리에서 멀수록 좋음".
//   최소제곱 잔차: r = [√(3w)gx, √(3w)gy, √w g1x, √w g1y] (다리당 4행, 총 16행).
//
//   ★게이트: 전역 포인터 g_edge 가 nullptr(기본) 이면 잔차/자코비안에 **아무 행도 추가되지 않음**
//            → OFF 경로는 기존 코드와 바이트 동일. 사용 시 EdgeLayers를 채우고 포인터를 세팅.
struct EdgeLayers {
  const Grid* gh_x  = nullptr;   // ∇h  x
  const Grid* gh_y  = nullptr;   // ∇h  y
  const Grid* gs1_x = nullptr;   // ∇h_s1 x
  const Grid* gs1_y = nullptr;   // ∇h_s1 y
  double cell = 0.02; int map_size = 101;
  double w = 1.0;                // 전체 가중(Drake=1.0)
  double w_raw = 3.0;            // ∇h 항 가중(Drake=3.0)
};
inline const EdgeLayers*& edge_layers() { static const EdgeLayers* p = nullptr; return p; }

// 다리 i의 edge 잔차 4개를 out에 append (게이트 OFF면 아무것도 안 함)
inline void edge_residuals(const TamolsState& st, std::vector<double>& out) {
  const EdgeLayers* E = edge_layers();
  if (!E || E->w <= 0 || !E->gh_x || !E->gh_y || !E->gs1_x || !E->gs1_y) return;
  const double s0 = std::sqrt(E->w * E->w_raw), s1 = std::sqrt(E->w);
  for (int n = 0; n < st.ncyc(); ++n) for (int i = 0; i < 4; ++i) {     // ★N 사이클 전부
    Vector3d fp = st.fpos(n, i); double x = fp(0), y = fp(1);
    out.push_back(s0 * bilinear_height(*E->gh_x,  E->cell, E->map_size, x, y));
    out.push_back(s0 * bilinear_height(*E->gh_y,  E->cell, E->map_size, x, y));
    out.push_back(s1 * bilinear_height(*E->gs1_x, E->cell, E->map_size, x, y));
    out.push_back(s1 * bilinear_height(*E->gs1_y, E->cell, E->map_size, x, y));
  }
}

// 스칼라 비용(진단용): Σ_i w(3|∇h|² + |∇h_s1|²)
inline double edge_avoidance_cost(const TamolsState& st) {
  std::vector<double> r; edge_residuals(st, r);
  double c = 0; for (double v : r) c += v * v; return c;
}

// ── ★지지 유효성 하드 제약 게이트 (2026-09) ──────────────────────────────────
//   "지형을 물리게" 하는 두 줄기. 둘 다 **부등식 하드 제약**(비용 아님) — tamols_qp.hpp 에 행 추가.
//     ① band  : |h(p_xy) − p.z| ≤ band       — 발판이 지형 표면 **위에 붙는다**(공중부양 금지)
//     ② sdf   : sdf_support(p_xy) ≤ −margin  — 그 지형점이 **국소 작업면에서의 지지**다
//   ①만으로는 못 고친다(보이드 바닥이 h=0=p.z 로 정확히 만족). ②만으로도 못 고친다
//   (보이드 위에 p.z 를 띄워 놓으면 xy 는 그대로). **둘이 같이 있어야** 보이드가 배제된다:
//   ① 때문에 p.z=h(=보이드 바닥)로 묶이고, ② 가 그 xy 를 무효로 판정한다.
//   sdf 는 terrain_proc.hpp `compute_support()`(형태학적 닫힘 기반). 보이드 내부에서도 기울기가
//   살아 있어 SQP 가 돌 쪽으로 나갈 방향을 갖는다.
//
//   ★게이트: 전역 포인터 support_layers() 가 nullptr(기본) 이면 제약/자코비안에 **0행 추가**
//            → OFF 경로는 기존과 바이트 동일.
struct SupportLayers {
  const Grid* h   = nullptr;     // 솔버가 최적화 대상으로 쓰는 높이맵(= solve 에 넘기는 것과 동일)
  const Grid* sdf = nullptr;     // 지지유효 SDF [m] (음수=유효영역 안)
  double cell = 0.02; int map_size = 101;
  double band   = 0.01;          // |h(p)−p.z| 허용 [m]
  double margin = 0.0;           // 유효영역 안쪽 여유 [m] (=유효집합 침식량, 발 반경 흡수)
  bool   band_on = true, sdf_on = true;
};
inline const SupportLayers*& support_layers() { static const SupportLayers* p = nullptr; return p; }
// 발판 1개당 추가되는 부등식 행 수(0 = 게이트 OFF)
inline int support_rows_per_foot() {
  const SupportLayers* S = support_layers(); if (!S) return 0;
  return (S->band_on && S->h ? 2 : 0) + (S->sdf_on && S->sdf ? 1 : 0);
}

} // namespace tamols
