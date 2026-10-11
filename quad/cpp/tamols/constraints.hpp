// constraints.hpp — TAMOLS 제약 (Drake constraints.py C++ 포팅) + 해 로더
//   검증: Drake feasible 해를 로드해 각 제약 residual≈0(feasibility) 확인.
//   현재: 로더 + 초기(add_initial)·위상연속(junction) [선형]. 다음: kinematic·friction·GIAC.
#pragma once
#include "tamols.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>
#include <functional>
#include <cstdlib>

namespace tamols {

// ── Drake export(/tmp/tamols_sol.txt) 로드 → TamolsState ──
inline bool load_solution(const std::string& path, TamolsState& st) {
  std::ifstream f(path);
  if (!f) return false;
  int P, bd, ord, nl; f >> P >> bd >> ord >> nl;
  st.prm.base_dims = bd; st.prm.spline_order = ord; st.prm.num_legs = nl;
  st.gait.assign(P, GaitPhase{});
  for (int k = 0; k < P; ++k) f >> st.gait[k].duration;
  for (int k = 0; k < P; ++k) for (int i = 0; i < 4; ++i) f >> st.gait[k].contact[i];
  for (int k = 0; k < P; ++k) for (int i = 0; i < 4; ++i) f >> st.gait[k].at_des[i];   // at_des_position
  for (int d = 0; d < 6; ++d) f >> st.base_pose(d);
  for (int d = 0; d < 6; ++d) f >> st.base_vel(d);
  for (int L = 0; L < 4; ++L) for (int c = 0; c < 3; ++c) f >> st.p_meas(L, c);
  st.a.assign(P, MatrixXd(6, ord));
  for (int k = 0; k < P; ++k) for (int d = 0; d < 6; ++d) for (int i = 0; i < ord; ++i) f >> st.a[k](d, i);
  for (int L = 0; L < 4; ++L) for (int c = 0; c < 3; ++c) f >> st.p(L, c);
  for (int L = 0; L < 4; ++L) for (int c = 0; c < 3; ++c) f >> st.prm.hip_offsets(L, c);
  st.epsilon.resize(P); for (int k = 0; k < P; ++k) f >> st.epsilon(k);         // GIAC slack
  for (int c = 0; c < 3; ++c) f >> st.prm.inertia_diag(c);                       // 관성 diag
  return (bool)f;
}

// ── 초기 제약 (Drake add_initial_constraints 앞부분): a0(d,0)=base_pose(d), a0(d,1)=base_vel(d) ──
inline double initial_residual(const TamolsState& st) {
  double e = 0;
  for (int d = 0; d < 6; ++d) {
    e = std::max(e, std::fabs(st.a[0](d, 0) - st.base_pose(d)));   // pos(0)=base_pose
    e = std::max(e, std::fabs(st.a[0](d, 1) - st.base_vel(d)));    // vel(0)=base_vel
  }
  return e;
}

// ── 위상 연속 제약 (junction): pos_k(Tk)=pos_{k+1}(0), vel_k(Tk)=vel_{k+1}(0) ──
inline double junction_residual(const TamolsState& st) {
  double e = 0;
  for (int k = 0; k + 1 < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    Vector6d pos_end = st.pos_at(k, Tk);   // 현 위상 끝
    Vector6d vel_end = st.vel_at(k, Tk);
    for (int d = 0; d < 6; ++d) {
      e = std::max(e, std::fabs(pos_end(d) - st.a[k + 1](d, 0)));   // 다음 위상 시작 pos = a(d,0)
      e = std::max(e, std::fabs(vel_end(d) - st.a[k + 1](d, 1)));   // 다음 위상 시작 vel = a(d,1)
    }
  }
  return e;
}

// ── ZYX 회전 (Drake get_R_B): phi_B=[psi,theta,phi]=[yaw,pitch,roll], R=Rz(psi)Ry(theta)Rx(phi) ──
inline Eigen::Matrix3d R_B(const Vector3d& phi_B) {
  double psi = phi_B(0), theta = phi_B(1), phi = phi_B(2);
  double cz = std::cos(psi), sz = std::sin(psi), cy = std::cos(theta), sy = std::sin(theta), cx = std::cos(phi), sx = std::sin(phi);
  Eigen::Matrix3d Rz, Ry, Rx;
  Rz << cz, -sz, 0, sz, cz, 0, 0, 0, 1;
  Ry << cy, 0, sy, 0, 1, 0, -sy, 0, cy;
  Rx << 1, 0, 0, 0, cx, -sx, 0, sx, cx;
  return Rz * Ry * Rx;
}

// ── friction cone (Drake 간이): 각 phase·τ서 base z-가속 ≥ -9.81 (자유낙하보다 안 빠름). 선형 ──
inline double friction_residual(const TamolsState& st) {
  double e = 0;
  const int S = 6;   // tau_sampling_rate
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    for (int s = 0; s < S; ++s) {
      double tau = Tk * (double)s / (double)S;             // linspace(0,Tk,S+1)[:S]
      double az = st.acc_at(k, tau)(2);                    // base z-가속
      e = std::max(e, std::max(0.0, -9.81 - az));          // az ≥ -9.81 위반량
    }
  }
  return e;
}

// ── ★스탠스 발 reach 게이트 (2026-09, 사이클 간 결합) ──────────────────────────
//   STANCE_REACH : **구 발판(p_meas) 위에 서 있는 다리**에도 reach 제약을 건다. 기본 OFF(=기존과 바이트 동일).
//     기존 reach 는 at_des(=이번 사이클에 새로 놓는 발판) 다리에만 걸려 있어, 지면에 박혀 있는 구 발판이
//     base 전진을 전혀 막지 못했다 → 직전 사이클 발판이 다음 사이클 도달범위에 영향 0(결합 부재).
//     ON: 각 phase τ 샘플에서 l_min² ≤ |base(τ)+R_B(τ)·hip − p_meas|² ≤ l_max².
//   ※ 샘플 s=1..S (τ∈(0,Tk]) — τ=0(phase0)은 초기조건 등식으로 **고정**이라 위반해도 원리적으로 못 고침.
//      phase k>0 의 τ=0 은 phase k−1 의 τ=Tk 와 같은 점이므로 연속성으로 이미 덮인다.
inline bool stance_reach_on() { const char* v = std::getenv("STANCE_REACH"); return v && v[0] != '0'; }
//   "구 발판 위 스탠스" = 접촉 중이면서 아직 새 발판으로 안 옮긴 다리
inline bool is_stance_meas(const GaitPhase& g, int L) { return g.contact[L] && !g.at_des[L]; }
//   다중 사이클(N>1) 정합판: 아직 한 번도 안 디딘(=p_meas) 접촉 다리
inline bool is_stance_meas(const TamolsState& st, int k, int L) { return st.gait[k].contact[L] && st.fsrc(k, L) < 0; }

// ── kinematic reach (Drake): at_des_position 다리서 l_min²≤|base+R_B·hip−foot|²≤l_max² ──
//   (+ STANCE_REACH 시 구 발판 스탠스 다리도)
inline double kinematic_residual(const TamolsState& st) {
  double e = 0;
  const int S = 6;
  const double lo = st.prm.l_min * st.prm.l_min, hi = st.prm.l_max * st.prm.l_max;
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    for (int L = 0; L < 4; ++L) {
      if (st.fsrc(k, L) < 0) continue;                     // at_des(=이미 디딘 발판) 다리만
      for (int s = 0; s < S; ++s) {
        double tau = Tk * (double)s / (double)S;
        Vector6d pose = st.pos_at(k, tau);
        Vector3d base = pose.head<3>();
        Eigen::Matrix3d R = R_B(pose.tail<3>());
        Vector3d diff = base + R * st.prm.hip_offsets.row(L).transpose() - st.foot_at(k, L);
        double total = diff.dot(diff);
        e = std::max(e, std::max(0.0, lo - total));        // ≥ l_min² 위반
        e = std::max(e, std::max(0.0, total - hi));        // ≤ l_max² 위반
      }
    }
  }
  // ★스탠스(구 발판 p_meas) reach — 게이트 OFF 면 이 블록 전체가 no-op
  if (stance_reach_on()) {
    for (int k = 0; k < st.num_phases(); ++k) {
      double Tk = st.gait[k].duration;
      for (int L = 0; L < 4; ++L) {
        if (!is_stance_meas(st, k, L)) continue;
        for (int s = 1; s <= S; ++s) {
          double tau = Tk * (double)s / (double)S;
          Vector6d pose = st.pos_at(k, tau);
          Vector3d diff = pose.head<3>() + R_B(pose.tail<3>()) * st.prm.hip_offsets.row(L).transpose()
                        - st.p_meas.row(L).transpose();
          double total = diff.dot(diff);
          e = std::max(e, std::max(0.0, lo - total));
          e = std::max(e, std::max(0.0, total - hi));
        }
      }
    }
  }
  return e;
}

// ── ★GIAC 게이트 (2026-09, 능력검증) ────────────────────────────────────────────
//   GIAC_FIX  : 중력항(aG = aB − g)을 GIAC 식에 포함. **기본 ON**. `GIAC_FIX=0`으로만 끔.
//               off면 정지자세서 aB≈0 → 17a/17b가 항등적으로 만족 = GIAC 기전 자체가 없음.
//   GIAC_ORDER: 접촉쌍을 **지지폴리곤 CW 인접쌍**으로 (기본 ON, GIAC_FIX 종속).
//               레거시(Drake)는 인덱스순 모든 쌍 (i<j) 인데, 그 반평면 교집합은 지지다각형이 아님.
//               예) stance{FR,RL,RR}(Go2 walk), 중력항 포함 시 레거시 조건의 해집합은
//               {px ≤ −0.1934, py ≥ +0.142} = 뒷발보다 뒤·왼쪽 = 지지삼각형 바깥.
//               det(p_ij, p_B−p_i, (0,0,g)) = g·cross_z(p_j−p_i, p_B−p_i) 이므로
//               "≤ eps" 형태는 **CW 정렬 인접쌍**일 때 정확히 "p_B ∈ 지지다각형"이 된다.
inline bool giac_fix_on()   { const char* v = std::getenv("GIAC_FIX");   return !(v && v[0] == '0'); }
inline bool giac_order_on() { const char* v = std::getenv("GIAC_ORDER"); return giac_fix_on() && !(v && v[0] == '0'); }

// stance 발 인덱스 → GIAC 제약 쌍 목록. ordered=false: 레거시 인덱스순 모든 쌍(Drake 정합).
//   ordered=true: 발 xy를 centroid 기준 각도 내림차순(=CW) 정렬 후 인접쌍(wrap 포함) → n쌍.
//   ※ N=3이면 쌍 개수는 3으로 동일(행수 불변), N=4면 6→4.
inline std::vector<std::pair<int,int>> giac_pairs(const std::vector<int>& stance,
                                                  const std::function<Vector3d(int)>& foot,
                                                  bool ordered) {
  std::vector<std::pair<int,int>> out;
  const int n = (int)stance.size();
  if (n < 3) return out;
  if (!ordered) {
    for (int x = 0; x < n; ++x) for (int y = x + 1; y < n; ++y) out.emplace_back(stance[x], stance[y]);
    return out;
  }
  double cx = 0, cy = 0;
  for (int i : stance) { Vector3d f = foot(i); cx += f(0); cy += f(1); }
  cx /= n; cy /= n;
  std::vector<std::pair<double,int>> ang;
  ang.reserve(n);
  for (int i : stance) { Vector3d f = foot(i); ang.emplace_back(std::atan2(f(1) - cy, f(0) - cx), i); }
  std::sort(ang.begin(), ang.end(), [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second; });   // 내림차순 = CW
  for (int k = 0; k < n; ++k) out.emplace_back(ang[k].second, ang[(k + 1) % n].second);
  return out;
}

// ── 스칼라 삼중곱 det([a b c]) = a·(b×c) (Drake determinant) ──
inline double det3(const Vector3d& a, const Vector3d& b, const Vector3d& c) { return a.dot(b.cross(c)); }

// ── GIAC 안정성 (Drake add_dynamics_constraints, Eq17) ──
//   17a(N>0 마찰콘): (μ·a_z)²−a_x²−a_y² ≥ 0
//   17b(N≥3): 접촉쌍마다 m·det(p_ij,p_B−p_i,a_B) − p_ij·L̇ ≤ eps
//   17c(N==2): |위 식| ≤ eps ,  17d: −det(e_z,p_ij,M_i) ≤ eps  (M_i=(p_B−p_i)×(g−a_B)−L̇/m)
//   eps=epsilon[phase](≥0). residual=최대 위반량(feasible 해서 ≈0).
inline double giac_residual(const TamolsState& st) {
  double e = 0;
  const int S = 6;
  const double mu = st.prm.mu, m = st.prm.mass;
  const Vector3d ez(0, 0, 1), gvec(0, 0, -9.81);
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    std::vector<int> stance;
    for (int i = 0; i < 4; ++i) if (st.gait[k].contact[i]) stance.push_back(i);
    int N = (int)stance.size();
    double eps = (k < st.epsilon.size()) ? st.epsilon(k) : 0.0;
    e = std::max(e, std::max(0.0, -eps));                       // eps ≥ 0
    auto foot = [&](int i) -> Vector3d { return st.foot_at(k, i); };
    const bool _gfix = giac_fix_on();
    auto prs = giac_pairs(stance, foot, giac_order_on());
    for (int s = 0; s < S; ++s) {
      double tau = Tk * (double)s / (double)S;
      Vector3d pB = st.pos_at(k, tau).head<3>();
      Vector3d aB = st.acc_at(k, tau).head<3>();
      Vector3d aG = _gfix ? Vector3d(aB - gvec) : aB;            // ★중력 포함(GIAC 본래정의)
      Vector3d Ld = st.Ldot_at(k, tau);
      if (N > 0) {                                              // 17a: (μ a_z)²−a_x²−a_y² ≥ 0
        double lhs = (mu * aG(2)) * (mu * aG(2)) - aG(0) * aG(0) - aG(1) * aG(1);
        e = std::max(e, std::max(0.0, -lhs));
      }
      if (N >= 3) {                                             // 17b: 접촉쌍
        for (const auto& pr : prs) {
          Vector3d pi = foot(pr.first), pj = foot(pr.second), pij = pj - pi;
          double lhs = m * det3(pij, pB - pi, aG) - pij.dot(Ld);
          e = std::max(e, std::max(0.0, lhs - eps));
        }
      }
      if (N == 2) {                                             // 17c,d: 이중지지
        Vector3d pi = foot(stance[0]), pj = foot(stance[1]), pij = pj - pi;
        double val = m * det3(pij, pB - pi, aG) - pij.dot(Ld);
        e = std::max(e, std::max(0.0, val - eps));              // 17c: |val| ≤ eps
        e = std::max(e, std::max(0.0, -val - eps));
        Vector3d Mi = (pB - pi).cross(gvec - aB) - Ld / m;      // 17d
        double cost = det3(ez, pij, Mi);
        e = std::max(e, std::max(0.0, -cost - eps));
      }
    }
  }
  return e;
}

} // namespace tamols
