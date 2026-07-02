// Phase1: MuJoCo C++ 로드·스텝 벤치 + eiquadprog 링크 확인 (툴체인 검증)
#include <mujoco/mujoco.h>
#include <eiquadprog/eiquadprog-fast.hpp>
#include <Eigen/Dense>
#include <cstdio>
#include <chrono>

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "../quad_real_sphere.mjcf";
  char err[1000] = "";
  mjModel* m = mj_loadXML(path, nullptr, err, 1000);
  if (!m) { std::printf("[C++] 모델 로드 실패: %s\n", err); return 1; }
  mjData* d = mj_makeData(m);
  std::printf("[C++] 로드 성공: nq=%d nv=%d nu=%d 총질량=%.1fkg\n",
              m->nq, m->nv, m->nu, m->body_subtreemass[0]);

  // 1) 순수 mj_step 속도 (물리 시뮬)
  const int N = 20000;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) mj_step(m, d);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::printf("[C++] mj_step %d회: %.1fms → %.0f step/s (%.2fms/step)\n",
              N, ms, N / (ms / 1000.0), ms / N);

  // 2) eiquadprog 링크·동작 확인 (작은 QP: min 0.5 x'Hx + g'x)
  {
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd g(3); g << -1, -2, -3;
    Eigen::MatrixXd Ce(0, 3); Eigen::VectorXd ce(0);
    Eigen::MatrixXd Ci(0, 3); Eigen::VectorXd ci(0);
    Eigen::VectorXd x(3);
    eiquadprog::solvers::EiquadprogFast qp; qp.reset(3, 0, 0);
    auto st = qp.solve_quadprog(H, g, Ce, ce, Ci, ci, x);
    std::printf("[C++] eiquadprog 테스트 해=(%.1f,%.1f,%.1f) status=%d (기대 1,2,3)\n",
                x[0], x[1], x[2], (int)st);
  }

  mj_deleteData(d); mj_deleteModel(m);
  std::printf("[C++] Phase1 툴체인 검증 완료.\n");
  return 0;
}
