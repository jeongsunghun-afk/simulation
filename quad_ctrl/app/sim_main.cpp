// sim 진입점: HAL → Estimator → Controller → HAL 루프(구조 데모).
//   현재는 MockHal(외부 의존 X)로 골격 검증. 2단계에서 MujocoHal로 교체.
//   ★핵심: 컨트롤러는 State만 보고 LowCmd만 낸다 → real 이식 시 이 루프의 HAL/Estimator만 바뀐다.
#include <cstdio>
#include "hal/mock_hal.hpp"                // TODO(2단계): "hal/mujoco_hal.hpp"
#include "estimator/sim_estimator.hpp"
#include "control/controller.hpp"

using namespace qc;

int main() {
  const int NU = 17;
  const double DT = 0.001;

  MockHal hal(NU, DT);                     // TODO: MujocoHal("quad_real_16dof_sphere.mjcf")
  SimEstimator est(NU);
  QuadController ctrl(NU);

  HighCmd hc; hc.mode = Mode::Stand;
  ctrl.set_command(hc);

  LowState ls;
  LowCmd cmd;
  cmd.q_des.setZero(NU); cmd.dq_des.setZero(NU);
  cmd.kp.setConstant(NU, 60); cmd.kd.setConstant(NU, 3); cmd.tau_ff.setZero(NU);

  const int TICKS = 10;
  for (int k = 0; k < TICKS; ++k) {
    hal.read(ls);            // 센서
    est.update(ls);          // 추정 → State
    ctrl.step(est.state(), cmd);  // 제어 → LowCmd
    hal.write(cmd);          // 명령(+1스텝)
  }

  std::printf("[quad_ctrl] 스켈레톤 루프 %d틱 OK (nu=%d dt=%.3f) — HAL/Estimator/Controller 배선 검증\n",
              TICKS, NU, DT);
  return 0;
}
