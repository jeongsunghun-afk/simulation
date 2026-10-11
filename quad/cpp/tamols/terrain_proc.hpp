// terrain_proc.hpp — TAMOLS 지형 처리 (Drake map_processing.py C++ 포팅)
//   heightmap h → h_s1(gaussian) · h_s2(virtual floor) · ∇h/∇h_s1/∇h_s2(5점 FD)
//   TAMOLS 엣지회피=그래디언트, 발판정합=높이. SDF 불요(D1식 NMPC만 SDF 필요).
//   경계=reflect(scipy 기본). 격자 M(i,j): i=x, j=y. ∇x=axis1(j변화 아님, 원본 axis=1=열)…
//   ※ Drake와 동일 축규약 유지: grad_x=FD along axis=1, grad_y=FD along axis=0.
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace tamols {

using Grid = Eigen::MatrixXd;   // (rows × cols) = h[i][j]

// reflect 인덱스(scipy 'reflect': d c b a | a b c d | d c b a)
inline int reflect_idx(int i, int n) {
  if (n == 1) return 0;
  while (i < 0 || i >= n) { if (i < 0) i = -i - 1; if (i >= n) i = 2 * n - i - 1; }
  return i;
}

// ── 5점 중앙차분 그래디언트 ([1,-8,0,8,-1]/(12·res)) ──
//   grad_x = axis=1(열 방향), grad_y = axis=0(행 방향). Drake compute_gradients 정합.
inline void compute_gradients(const Grid& h, double res, Grid& gx, Grid& gy) {
  const int R = (int)h.rows(), C = (int)h.cols();
  gx.setZero(R, C); gy.setZero(R, C);
  // scipy ndimage.convolve1d = convolution(커널 뒤집음). 반대칭 미분커널이라 correlation 대비 부호반전
  //   → Drake([1,-8,0,8,-1] via convolve1d) 정합 위해 뒤집은 [-1,8,0,-8,1] 사용.
  const double k[5] = {-1, 8, 0, -8, 1};
  const double sc = 1.0 / (12.0 * res);
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j) {
      double sx = 0, sy = 0;
      for (int m = -2; m <= 2; ++m) {
        sx += k[m + 2] * h(i, reflect_idx(j + m, C));   // axis=1
        sy += k[m + 2] * h(reflect_idx(i + m, R), j);   // axis=0
      }
      gx(i, j) = sx * sc; gy(i, j) = sy * sc;
    }
}

// ── 분리형 gaussian filter (reflect) ──
inline Grid gaussian_filter(const Grid& h, double sigma) {
  const int R = (int)h.rows(), C = (int)h.cols();
  int rad = std::max(1, (int)(4.0 * sigma + 0.5));       // scipy gaussian_filter truncate=4.0 정합
  std::vector<double> ker(2 * rad + 1); double sum = 0;
  for (int t = -rad; t <= rad; ++t) { double v = std::exp(-0.5 * (t * t) / (sigma * sigma)); ker[t + rad] = v; sum += v; }
  for (double& v : ker) v /= sum;
  Grid tmp(R, C), out(R, C);
  for (int i = 0; i < R; ++i)                            // 열방향(axis=1)
    for (int j = 0; j < C; ++j) { double s = 0;
      for (int t = -rad; t <= rad; ++t) s += ker[t + rad] * h(i, reflect_idx(j + t, C));
      tmp(i, j) = s; }
  for (int i = 0; i < R; ++i)                            // 행방향(axis=0)
    for (int j = 0; j < C; ++j) { double s = 0;
      for (int t = -rad; t <= rad; ++t) s += ker[t + rad] * tmp(reflect_idx(i + t, R), j);
      out(i, j) = s; }
  return out;
}

// ── 3×3 median filter (reflect) ──
inline Grid median3(const Grid& h) {
  const int R = (int)h.rows(), C = (int)h.cols();
  Grid out(R, C); double w[9];
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j) { int n = 0;
      for (int di = -1; di <= 1; ++di) for (int dj = -1; dj <= 1; ++dj)
        w[n++] = h(reflect_idx(i + di, R), reflect_idx(j + dj, C));
      std::nth_element(w, w + 4, w + 9); out(i, j) = w[4]; }
  return out;
}

// ── TAMOLS 지형 처리: h_s1, h_s2(virtual floor), 그래디언트 ──
struct TerrainLayers {
  Grid h_s1, h_s2;
  Grid gh_x, gh_y, gs1_x, gs1_y, gs2_x, gs2_y;    // ∇h, ∇h_s1, ∇h_s2
};

inline TerrainLayers process_height_maps(const Grid& h_raw, double res,
                                         double sigma1 = 1.0, double sigma2 = 2.0) {
  TerrainLayers L;
  const int R = (int)h_raw.rows(), C = (int)h_raw.cols();
  // h_s1 = gaussian(h, σ1)
  L.h_s1 = gaussian_filter(h_raw, sigma1);
  // virtual floor: median → delta≠0 mask → dilate → 3×3 local max → gaussian(σ2)
  Grid hmed = median3(h_raw);
  Eigen::MatrixXi mask(R, C);                     // delta≠0 = edge/transition
  for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j)
    mask(i, j) = (std::fabs(h_raw(i, j) - hmed(i, j)) > 1e-12) ? 1 : 0;
  Eigen::MatrixXi dil(R, C); dil.setZero();       // 3×3 binary dilation
  for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) if (mask(i, j)) {
    for (int di = -1; di <= 1; ++di) for (int dj = -1; dj <= 1; ++dj) {
      int ii = i + di, jj = j + dj;
      if (ii >= 0 && ii < R && jj >= 0 && jj < C) dil(ii, jj) = 1; } }
  Grid h_dil = h_raw;                             // dilated 셀=3×3 local max(갭 위 virtual floor)
  for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) if (dil(i, j)) {
    double mx = -1e18;
    for (int di = -1; di <= 1; ++di) for (int dj = -1; dj <= 1; ++dj) {
      int ii = i + di, jj = j + dj;
      if (ii >= 0 && ii < R && jj >= 0 && jj < C) mx = std::max(mx, h_raw(ii, jj)); }
    h_dil(i, j) = mx; }
  L.h_s2 = gaussian_filter(h_dil, sigma2);
  // 그래디언트
  compute_gradients(h_raw, res, L.gh_x,  L.gh_y);
  compute_gradients(L.h_s1, res, L.gs1_x, L.gs1_y);
  compute_gradients(L.h_s2, res, L.gs2_x, L.gs2_y);
  return L;
}

// ══════════════════════════════════════════════════════════════════════════════
// ★지지 유효성(support validity) 층 — "작업 높이에서의 지지인가" 판별 (2026-09)
// ══════════════════════════════════════════════════════════════════════════════
//  문제: `foothold_on_ground = 100·(h(p) − p.z)²` 는 **보이드에서도 0** 이다(h=0 이고 p.z→0).
//        즉 "돌 위에 서라"를 보상하는 항이 목적함수에 없다. 세 가지가 전부 안 통한다:
//          ① 등식화(h(p)=p.z)      — 보이드 바닥이 **정확히** 만족한다(h=0, p.z=0).
//          ② edge_avoidance(|∇h|)  — 보이드 바닥은 평평해 |∇h|=0.
//          ③ 발 반경 침식(erosion) — 보이드 바닥은 넓고 평평해 침식을 통과한다.
//        셋 다 "높이 자체"만 보기 때문이다. 필요한 건 **주변 작업면 대비 높이**다.
//
//  판별식 = 회색조 형태학적 닫힘(grayscale morphological closing)
//        h_close = erode_R( dilate_R(h) ),   gap_depth(x) := h_close(x) − h(x) ≥ 0
//    · **단조 지형(계단·경사·단차)에서 closing = 항등** ⇒ gap_depth ≡ 0 (오검출이 원리적으로 없다).
//      1D 증명: 단조증가 h 에 대해 dilate_R(x)=h(x+R), erode_R∘dilate_R(x)=h(x−R+R)=h(x).
//    · 폭 < 2R 인 **구덩이/갭/보이드만** 채워진다 ⇒ gap_depth = 그 구덩이 깊이.
//    ⇒ "주변 작업면보다 dz_max 이상 낮은 곳 = 지지가 아니라 추락"이 정확히 gap_depth > dz_max.
//
//  유효영역 SDF: valid = {gap_depth ≤ dz_max}. sdf<0 = 유효영역 안(절댓값=경계까지 거리),
//        sdf>0 = 밖(=가장 가까운 유효점까지 거리). **보이드 내부(∇h=0)에서도 기울기가 살아 있어**
//        SQP 가 "돌 쪽으로" 나갈 방향을 갖는다(하드 제약을 걸어도 국소 정체하지 않음).
//        sdf(p) ≤ −margin 은 유효집합을 margin 만큼 **침식**한 것과 동일 ⇒ 발 반경은 margin 으로 흡수.

// ── 분리형 box(정사각 SE) 형태학 — 현재 closing 은 원판 SE 를 쓴다(아래). 이건 **빠른 대안**:
//   O(rad) 분리형이라 원판 대비 ~10× 빠르다(121² · R=0.25: 약 2 ms vs 21 ms). 다만 체비셰프라
//   대각으로 √2·R 까지 닿아 침식이 과해진다 → 정확도 대신 속도가 필요할 때만.
inline Grid morph_box(const Grid& h, int rad, bool dilate) {
  const int R = (int)h.rows(), C = (int)h.cols();
  Grid t(R, C), out(R, C);
  for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) {          // axis=1
    double v = h(i, j);
    for (int d = -rad; d <= rad; ++d) { int jj = j + d; if (jj < 0 || jj >= C) continue;
      v = dilate ? std::max(v, h(i, jj)) : std::min(v, h(i, jj)); }
    t(i, j) = v; }
  for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) {          // axis=0
    double v = t(i, j);
    for (int d = -rad; d <= rad; ++d) { int ii = i + d; if (ii < 0 || ii >= R) continue;
      v = dilate ? std::max(v, t(ii, j)) : std::min(v, t(ii, j)); }
    out(i, j) = v; }
  return out;
}
// ── 원판(디스크, 유클리드) 구조요소 형태학 ──
//   정사각 SE(체비셰프)는 **대각 방향으로 √2·R 까지 닿아** 침식이 과해진다 → 폭 2R 미만의 갭인데도
//   대각으로 미충전 영역에 닿아 "안 채워진" 것으로 판정되는 누수가 생긴다(실측: 폭 0.42 m 갭이 valid 로 샘).
//   원판 SE 는 모든 방향으로 정확히 R. 단조지형 항등성(closing=identity)은 대칭 SE 면 그대로 성립.
inline Grid morph_disk(const Grid& h, int rad, bool dilate) {
  const int R = (int)h.rows(), C = (int)h.cols();
  std::vector<std::pair<int,int>> off;                    // 원판 오프셋(1회 계산)
  for (int di = -rad; di <= rad; ++di) for (int dj = -rad; dj <= rad; ++dj)
    if (di * di + dj * dj <= rad * rad) off.emplace_back(di, dj);
  Grid out(R, C);
  for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) {
    double v = h(i, j);
    for (const auto& o : off) { int ii = i + o.first, jj = j + o.second;
      if (ii < 0 || ii >= R || jj < 0 || jj >= C) continue;
      v = dilate ? std::max(v, h(ii, jj)) : std::min(v, h(ii, jj)); }
    out(i, j) = v; }
  return out;
}
inline Grid grayscale_close(const Grid& h, int rad) { return morph_disk(morph_disk(h, rad, true), rad, false); }

// ── 1D 제곱거리변환 (Felzenszwalb & Huttenlocher 2012, 정확·O(n)) ──
//   INF=1e10 (거리제곱 최대 ~3e4 ≪ 1e10 이고 ulp(1e10)≈2e-6 라 (INF+q²)−(INF+v²)=q²−v² 정확)
inline void edt1d(const std::vector<double>& f, std::vector<double>& d) {
  const int n = (int)f.size();
  const double INF = 1e10;
  std::vector<int> v(n, 0); std::vector<double> z(n + 1, 0.0);
  int k = 0; v[0] = 0; z[0] = -INF; z[1] = INF;
  for (int q = 1; q < n; ++q) {
    double s = ((f[q] + (double)q * q) - (f[v[k]] + (double)v[k] * v[k])) / (2.0 * q - 2.0 * v[k]);
    while (s <= z[k]) { --k;
      s = ((f[q] + (double)q * q) - (f[v[k]] + (double)v[k] * v[k])) / (2.0 * q - 2.0 * v[k]); }
    ++k; v[k] = q; z[k] = s; z[k + 1] = INF;
  }
  d.assign(n, 0.0); k = 0;
  for (int q = 0; q < n; ++q) { while (z[k + 1] < q) ++k; d[q] = (double)(q - v[k]) * (q - v[k]) + f[v[k]]; }
}
// src(i,j)!=0 인 셀까지의 **제곱** 거리(셀 단위). 소스가 하나도 없으면 ≈1e10.
inline Grid edt2_sq(const Eigen::MatrixXi& src) {
  const int R = (int)src.rows(), C = (int)src.cols(); const double INF = 1e10;
  Grid d(R, C);
  std::vector<double> f, o;
  for (int i = 0; i < R; ++i) { f.assign(C, 0.0);
    for (int j = 0; j < C; ++j) f[j] = src(i, j) ? 0.0 : INF;
    edt1d(f, o); for (int j = 0; j < C; ++j) d(i, j) = o[j]; }
  for (int j = 0; j < C; ++j) { f.assign(R, 0.0);
    for (int i = 0; i < R; ++i) f[i] = d(i, j);
    edt1d(f, o); for (int i = 0; i < R; ++i) d(i, j) = o[i]; }
  return d;
}

// ── 지지 유효성 층 ──
struct SupportLayer {
  Grid h_close;              // 형태학적 닫힘(=국소 작업면)
  Grid gap_depth;            // h_close − h  (단조지형=0, 구덩이=깊이)
  Eigen::MatrixXi valid;     // gap_depth ≤ dz_max
  Grid sdf;                  // 유효영역 부호거리 [m] (음수=안쪽)
  int  n_valid = 0, n_cell = 0;
};
//  close_radius_m — 채울 구덩이의 최대 **반**폭. 로봇이 한 걸음에 가로지를 수 있는 갭 스케일.
//  dz_max         — 국소 작업면 아래로 허용되는 최대 하강량(= 한 스텝 step-down 여유).
inline SupportLayer compute_support(const Grid& h, double res, double dz_max, double close_radius_m) {
  SupportLayer S;
  const int R = (int)h.rows(), C = (int)h.cols();
  int rad = std::max(1, (int)std::lround(close_radius_m / res));
  S.h_close = grayscale_close(h, rad);
  S.gap_depth = S.h_close - h;
  S.valid.setZero(R, C);
  for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j)
    S.valid(i, j) = (S.gap_depth(i, j) <= dz_max) ? 1 : 0;
  S.n_cell = R * C; S.n_valid = S.valid.sum();
  Eigen::MatrixXi inv(R, C); for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) inv(i, j) = 1 - S.valid(i, j);
  Grid d_to_valid = edt2_sq(S.valid), d_to_inv = edt2_sq(inv);
  S.sdf.resize(R, C);
  for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j)
    { double dv = std::min(1e8, d_to_valid(i, j)), di = std::min(1e8, d_to_inv(i, j));
      double v = S.valid(i, j) ? -res * std::sqrt(di) : res * std::sqrt(dv);
      S.sdf(i, j) = std::max(-1.0, std::min(1.0, v)); }   // ±1 m 클램프(전영역 유효/무효 시 폭주 방지)
  return S;
}

} // namespace tamols
