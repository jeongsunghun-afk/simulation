#pragma once
// 16-DOF 관절맵 — MJCF(quad_real_16dof_sphere) 관절순 ↔ 실모터 Gait SHM 채널.
// ★2026-10-06 실물 허리 관절 제거 → 17-DOF 맵에서 FB_waist 줄(구 index 8)만 삭제. 다른 축의 채널·부호·한계는 그대로.
//   구조: GaitJointCfg{ chan, sign, zero_deg, min_deg, max_deg, vel_max_dps }.
//     chan     = Gait SHM 채널. ★-1 = 미배선(구동 금지). real_hal 이 거부한다.
//     sign     = MJCF 관절 +방향 ↔ 실모터 +방향 (±1).
//     zero_deg = MJCF 0 자세일 때 실모터 절대각(deg).
//     min/max  = 관절 한계(deg, 관절축). 안전 clip.
//     vel_max  = 최대 각속도(deg/s).
//
// ★★채널 배치 = 2026-08-05 Pi 실측 확정(tools/chan_probe). 추측 아님.
//   RobotSharedMem.h 헤더의 29채널 배치(ForeL 0-6·ForeR 7-13·HindL 14-19·HindR 20-25·Waist 26-28)는
//   **최종 전기 로봇 기준**이고, 지금 실제로 서비스되는 채널은 **0~7 뿐**이다:
//     RobotMemGait_GetUpdatedFlag_MotorStatus16() = 0x000000ff (8비트)
//     ch8~28 은 상태 자체가 오지 않는다 → biped/emb 이 쓰던 배치(HL 0-3 · HR 4-7)와 일치.
//   ⇒ 헤더 배치를 믿고 chan 0..16 을 순차 배정하면 **FL/FR 명령이 뒷다리로 나간다**(위험).
//      그래서 미배선 축은 0 이 아니라 **-1** 로 둔다.
//
// ★현재 물리적으로 장착된 모터 = ch0 · ch4 뿐 (2026-08-05 실측: 그 둘만 위치 비영 8.62°/14.60°,
//   나머지는 0.00). PACE 액추에이터 실측도 이 두 축(HL_hip·HR_hip)에서 나왔다.
//   ⇒ 나머지 6축(HL/HR thigh·calf·foot)은 채널은 존재하나 모터 미장착 상태.
//
// ⚠ zero_deg 는 **전 축 미측정**(0.0 placeholder). 다리 장착 후 기준자세 정렬로 확정할 것.
//   min/max 는 biped/emb/config/biped_emb.yaml 실측·외삽값을 계승(그쪽이 같은 하드웨어).
#include "hal/real_hal.hpp"
#include <vector>

#if defined(QC_HAVE_ROBOT_SHM) || __has_include("/usr/include/RobotSharedMem.h")
namespace qc {

// MJCF nu=16 순서: HL(hip,thigh,calf,foot)·HR(4)·FL(4)·FR(4).
//                        chan sign zero  min   max   vel      // 관절
inline std::vector<GaitJointCfg> joint_map_16dof() { return {
  // ── 뒷다리 좌(HL) — 채널 서비스됨 ─────────────────────────────────────────
  {  0, +1,  0.0,  -35,  35, 300 },   // 0  HL_hip    ★sign +1 실측확정(2026-08-05) · 모터 장착됨
  {  1, +1,  0.0, -135,  65, 210 },   // 1  HL_thigh  sign 미검증(모터 미장착)
  {  2, +1,  0.0,  -55,  65, 300 },   // 2  HL_calf   sign 미검증(모터 미장착)
  {  3, +1,  0.0,  -80,  40, 300 },   // 3  HL_foot   sign 미검증(모터 미장착)
  // ── 뒷다리 우(HR) — 채널 서비스됨 ─────────────────────────────────────────
  {  4, +1,  0.0,  -35,  35, 300 },   // 4  HR_hip    ★sign +1 실측확정(거울 아님) · 모터 장착됨
  {  5, +1,  0.0, -135,  65, 210 },   // 5  HR_thigh  sign 미검증(모터 미장착)
  {  6, +1,  0.0,  -55,  65, 300 },   // 6  HR_calf   sign 미검증(모터 미장착)
  {  7, +1,  0.0,  -80,  40, 300 },   // 7  HR_foot   sign 미검증(모터 미장착)
  // ── 아래는 전부 미배선(채널 자체가 서비스되지 않음) ──────────────────────
  //    chan=-1 → real_hal 이 read 에서 제외하고 write 에서 명령하지 않는다.
  { -1, +1,  0.0,  -35,  35, 300 },   // 8  FL_hip    ★미배선
  { -1, +1,  0.0, -135,  65, 210 },   // 9  FL_thigh  ★미배선
  { -1, +1,  0.0,  -55,  65, 300 },   // 10 FL_calf   ★미배선
  { -1, +1,  0.0,  -80,  40, 300 },   // 11 FL_foot   ★미배선
  { -1, +1,  0.0,  -35,  35, 300 },   // 12 FR_hip    ★미배선
  { -1, +1,  0.0, -135,  65, 210 },   // 13 FR_thigh  ★미배선
  { -1, +1,  0.0,  -55,  65, 300 },   // 14 FR_calf   ★미배선
  { -1, +1,  0.0,  -80,  40, 300 },   // 15 FR_foot   ★미배선
}; }

}  // namespace qc
#endif
