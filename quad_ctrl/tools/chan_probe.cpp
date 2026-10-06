/* chan_probe.cpp — Gait SHM 채널 배치 실측(읽기 전용, 모터 무동작).
 *
 *   ★목적: quad 16-DOF 관절맵(구 17-DOF)의 `chan` 을 추측이 아니라 실측으로 확정한다.
 *     두 배치가 충돌하고 있다 —
 *       (a) /usr/include/RobotSharedMem.h : 29채널
 *           ForeL 0-6(7) · ForeR 7-13(7) · HindL 14-19(6) · HindR 20-25(6) · Waist 26-28(3)
 *       (b) 실제 검증된 biped/emb + RobotTestGait : 10채널 (HL 0-3 · HR 4-7 · Waist 8-9)
 *           PACE 액추에이터 실측도 ch00(HL_hip) · ch04(HR_hip) 였다.
 *     헤더는 "최종 전기 로봇" 배치이고 지금 배선된 것은 (b) 일 가능성이 높다.
 *     → 어느 채널이 **실제로 살아있는지** 를 보고 결정한다.
 *
 *   ★판정 기준(stt_probe.cpp 의 교훈 계승): updated flag 는 믿을 수 없다.
 *     Emb 는 EtherCAT 이 OP 를 잃어도 마지막 버퍼를 재발행하며 flag 를 세운다.
 *     따라서 "값이 시간에 따라 **변하는지**" 를 본다. 살아있는 축은 limp 상태에서도
 *     엔코더 노이즈로 최소 LSB 단위 흔들림이 있다. 완전 고정 = 죽은 채널(또는 미배선).
 *
 *   ⚠ GetMotorStatus16 은 updated 플래그를 소비한다(RobotSharedMem_Gait.cpp:873).
 *     robot_main 과 **동시에 실행 금지**. 단독으로만 쓸 것.
 *   ⚠ 명령(SetMotorCommand16)은 한 번도 호출하지 않는다 → 모터 절대 무동작.
 *
 *   빌드: g++ -O2 -std=c++17 chan_probe.cpp -o chan_probe -lRobotSharedMem -lrt
 *   실행: ./chan_probe [iterations]      (기본 60회 ≈ 3s @ 50ms)
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "RobotSharedMem.h"

static void sleep_ms(int ms){ struct timespec ts{ ms/1000, (long)(ms%1000)*1000000L }; nanosleep(&ts,nullptr); }

static const int NCH = (int)NUM_OF_MOTOR_GAIT;      // 29 (헤더 최대치까지 전부 훑는다)

int main(int argc, char** argv){
    const int iters = (argc > 1) ? atoi(argv[1]) : 60;

    if (RobotMemGait_InitComm() != 0){ printf("InitComm FAIL — RobotEmbedded 기동 확인\n"); return 1; }
    printf("InitComm OK. 채널 0..%d 스윕 %d회 (읽기 전용, 모터 무동작)\n\n", NCH-1, iters);

    // 채널별 관측 누적
    int    seen[NCH];  int changed[NCH];
    double pmin[NCH],  pmax[NCH];
    float  prev[NCH][3];
    int    status[NCH];
    for (int i=0;i<NCH;i++){ seen[i]=changed[i]=0; pmin[i]=1e9; pmax[i]=-1e9; status[i]=-1;
                             prev[i][0]=prev[i][1]=prev[i][2]=NAN; }
    unsigned long flag_or = 0;

    // IMU 관측(펌웨어 배선 여부 확인용)
    int imu_upd = 0, imu_ok = 0, imu_nonzero = 0, imu_changed = 0;
    float imu_prev[LEN_OF_IMU_DATA]; memset(imu_prev,0,sizeof(imu_prev));
    float imu_last[LEN_OF_IMU_DATA]; memset(imu_last,0,sizeof(imu_last));

    for (int k=0;k<iters;k++){
        flag_or |= RobotMemGait_GetUpdatedFlag_MotorStatus16();
        for (int i=0;i<NCH;i++){
            MotorParam16_t st; memset(&st,0,sizeof(st));
            if (RobotMemGait_GetMotorStatus16(&st, i) != 0) continue;
            seen[i]++;
            const float p=(float)st.fPosition, v=(float)st.fVelocity, t=(float)st.fTorque;
            if (!std::isnan(prev[i][0]) && (p!=prev[i][0] || v!=prev[i][1] || t!=prev[i][2])) changed[i]++;
            prev[i][0]=p; prev[i][1]=v; prev[i][2]=t;
            if (p<pmin[i]) pmin[i]=p;
            if (p>pmax[i]) pmax[i]=p;
            status[i]=(int)st.ucStatus;
        }
        if (RobotMemGait_IsUpdatedIMU()==1){
            imu_upd++;
            float buf[LEN_OF_IMU_DATA]={0};
            if (RobotMemGait_GetIMU(buf, IDX_OF_IMU_ForeC_START, LEN_OF_IMU_DATA)==0){
                imu_ok++;
                for (unsigned i=0;i<LEN_OF_IMU_DATA;i++){
                    if (buf[i]!=0.0f) imu_nonzero++;
                    if (k>0 && buf[i]!=imu_prev[i]) imu_changed++;
                    imu_prev[i]=buf[i]; imu_last[i]=buf[i];
                }
            }
        }
        sleep_ms(50);
    }

    // ── 결과 ────────────────────────────────────────────────────────────────
    printf("MotorStatus16 누적 flag = 0x%08lx\n", flag_or);
    printf("%-5s %-6s %-8s %-11s %-9s %s\n", "chan","읽힘","변화","위치범위[deg]","ucStatus","판정");
    printf("---------------------------------------------------------------------\n");
    int live=0;
    for (int i=0;i<NCH;i++){
        const bool got = seen[i]>0;
        const double span = got ? (pmax[i]-pmin[i]) : 0.0;
        const char* verdict;
        if (!got)                  verdict = "-- 읽기실패";
        else if (changed[i]==0)    verdict = "고정(미배선/정지 의심)";
        else { verdict = "★LIVE(값 변동)"; live++; }
        if (got) printf("%-5d %-6d %-8d %7.2f~%-7.2f 0x%02x      %s\n",
                        i, seen[i], changed[i], pmin[i], pmax[i], status[i], verdict);
        else     printf("%-5d %-6d %-8s %-11s %-9s %s\n", i, seen[i], "-", "-", "-", verdict);
    }
    printf("---------------------------------------------------------------------\n");
    printf("값이 실제로 변하는 채널 = %d개\n", live);
    printf("  → 이 집합이 실배선 축이다. 관절맵 chan 은 여기서만 고른다.\n\n");

    printf("IMU: 갱신플래그 %d/%d · 읽기성공 %d · 비영값 %d · 변화 %d\n",
           imu_upd, iters, imu_ok, imu_nonzero, imu_changed);
    if (imu_changed==0)
        printf("  → ★IMU 미배선/미갱신 확정. 자세·각속도 없음 = KF base 추정 불가 → stand/walk 불가.\n");
    else {
        printf("  → IMU 살아있음. 원값 덤프(convention 확정용):\n");
        printf("     QUAT[0..3] %8.4f %8.4f %8.4f %8.4f  (노름 %.4f)\n",
               imu_last[IDX_OF_IMU_QUAT+0], imu_last[IDX_OF_IMU_QUAT+1],
               imu_last[IDX_OF_IMU_QUAT+2], imu_last[IDX_OF_IMU_QUAT+3],
               std::sqrt((double)imu_last[IDX_OF_IMU_QUAT+0]*imu_last[IDX_OF_IMU_QUAT+0]
                        +(double)imu_last[IDX_OF_IMU_QUAT+1]*imu_last[IDX_OF_IMU_QUAT+1]
                        +(double)imu_last[IDX_OF_IMU_QUAT+2]*imu_last[IDX_OF_IMU_QUAT+2]
                        +(double)imu_last[IDX_OF_IMU_QUAT+3]*imu_last[IDX_OF_IMU_QUAT+3]));
        printf("     ACCL[4..6] %8.4f %8.4f %8.4f   (정지 시 한 축 ≈ ±9.81 이면 중력포함 body 프레임)\n",
               imu_last[IDX_OF_IMU_ACCL+0], imu_last[IDX_OF_IMU_ACCL+1], imu_last[IDX_OF_IMU_ACCL+2]);
        printf("     GYRO[7..9] %8.4f %8.4f %8.4f   (정지 시 ≈0. 단위 deg/s 가정)\n",
               imu_last[IDX_OF_IMU_GYRO+0], imu_last[IDX_OF_IMU_GYRO+1], imu_last[IDX_OF_IMU_GYRO+2]);
        printf("     ARPY[10..12] %8.4f %8.4f %8.4f (deg)\n",
               imu_last[IDX_OF_IMU_ARPY+0], imu_last[IDX_OF_IMU_ARPY+1], imu_last[IDX_OF_IMU_ARPY+2]);
    }
    return 0;
}
