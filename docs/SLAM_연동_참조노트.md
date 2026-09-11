# SLAM 연동 참조 노트 — 제어기 ↔ SLAM/Nav 인터페이스 설계 지식

> **범위**: 이 문서는 제어기 쪽에서 확정한 **연동 아키텍처·인터페이스 계약·설계 결정과 그 이유**를 기록한다
> (본인 작업·본인 지식). SLAM 시스템 내부 구현은 다루지 않는다. 목표 블록도=[pipeline_fullstack.html](pipeline_fullstack.html)(F2).

## 1. 전체 아키텍처 (확정안 A: SLAM 라이브 허브)
```
[LiDAR/IMU] → SLAM 라이브(①Estimator ②Localization ③Loop-closure)
                 ├─ map→odom   (드리프트 보정, 점프 허용)
                 ├─ /map_scan  (deskew 완료·odom 프레임 등록된 포인트클라우드)
                 └─ (odom→base는 SLAM이 아니라 ↓제어기 추정기가 원천)
[제어기 스택]  StateEstimator(접촉 KF/InEKF) → odom→base (매끄러움 보장, 고주파)
                 TerrainMap ← /map_scan + odom→base 등록 → elevation map(heightmap)
                 컨트롤러 ← cmd_vel (in) · → odom (out)
```
- **map→base = map→odom ∘ odom→base** 합성으로만 사용.

## 2. 인터페이스 계약 (경계 4개 — 이것만 맞으면 어떤 SLAM이든 교체 가능)
| 경계 | 방향 | 계약 |
|---|---|---|
| `cmd_vel` | nav→제어기 | vx·vy·yaw-rate. 제어기 명령슬루(τ≈0.3s)가 급변 흡수 — nav는 부드러움 책임 없음 |
| `odom`(odom→base) | 제어기→nav/지형 | **원천=제어기 상태추정기**(접촉 KF, controller-independent). 매끄러움·고주파(1kHz급) 보장 |
| `map→odom` | SLAM→nav | 드리프트 보정 전담. **점프 허용**(전역 일관성 담당) |
| `/map_scan` | SLAM→TerrainMap | deskew·odom 등록 완료된 클라우드. TerrainMap은 셀별 높이 융합만 |

## 3. 핵심 설계 결정과 이유 (실측·실패에서 나온 것)
1. **odom→base의 원천은 제어기 추정기다** (SLAM/정책 내부 SE 아님). 이유: 보행 제어는 접촉 정보를 아는 고주파 추정이 필요하고, 정책-내장 SE는 RL 종속이라 스택 간 재사용 불가.
2. **TerrainMap 등록 프레임 = odom→base** (map→base 아님). 이유: map→odom은 루프클로저에서 점프하는데, 지형맵이 그 프레임에 등록되면 **발밑 지형이 순간이동**한다. 지역 지형은 매끄러운 프레임에, 전역 일관성은 map 층에.
3. **depth 픽셀 직접 소비 금지** — 항상 "등록된 클라우드 → 셀 융합 → heightmap" 경로. 이유: 컨트롤러가 소비하는 건 로봇중심 격자 질의(h(x,y)·∇h)이고, 이 추상화가 sim(mj_ray)↔real(클라우드)을 같은 코드로 만든다.
4. **컨트롤러측 플러그 = TerrainProvider 1개 함수** — `fill(sdf, cx, cy)`로 로봇중심 격자(우리 61×41·4cm)를 채우면 끝. 인지 스택은 `height(x,y)` 콜백만 공급(HeightmapTerrainProvider). 검증: 합성 heightmap 주입으로 폐루프 보행 falls=0(배관 검증 완료).
5. **풀스택 통합은 별도 프로젝트** — 제어기 리포는 cmd_vel-in/odom-out 경계만 노출. 결합도를 여기서 끊어야 SLAM 교체·팀 분업이 된다.

## 4. 함정 (겪은 것)
- 지형맵 프레임을 map에 걸면 루프클로저마다 base-z 참조가 튀어 보행 붕괴 위험 (§3-2의 근거).
- SLAM odom을 제어에 직결하면 지연·저주파로 균형 대역폭 부족 — 제어 v̂는 항상 자체 추정.
- heightmap 소비측은 **갱신 주기보다 좌표 일관성**이 중요 — 셀 원점이 로봇과 함께 이동해도 질의 함수가 절대좌표면 문제없음.

## 5. 공개 대안 매핑 (재현 시)
| 역할 | 공개 대안 |
|---|---|
| LiDAR-inertial SLAM 라이브 | HKU-MARS **VoxelSLAM**(원본 오픈소스)·FAST-LIO2·DLIO |
| elevation map | ANYbotics **elevation_mapping**(GridMap)·elevation_mapping_cupy |
| 전역/토폴로지 nav | Nav2 + 토폴로지 플래너류 공개 구현 |
- 위 계약(§2) 4개만 맞추면 어느 조합으로도 우리 제어기와 연동된다 — 그게 이 경계 설계의 목적.
