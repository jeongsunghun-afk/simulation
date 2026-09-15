#!/usr/bin/env bash
# emb_ctl.sh — RobotEmbedded(Emb) 기동/종료/로그 헬퍼. 진단 절차를 한 곳에 고정.
#   Emb 는 root 필요(EtherCAT raw socket / SPI). sudoers 미설치면 암호를 물어본다.
#
#   ./emb_ctl.sh start   # 기동(로그: /tmp/emb.log) 후 halGait 초기화(≈5s) + 신선도 확인
#   ./emb_ctl.sh stop    # 종료
#   ./emb_ctl.sh log     # EtherCAT 요약(슬레이브/OP/WKC) + 마지막 로그
#   ./emb_ctl.sh reset   # **싹 종료** — writer·Emb·되살리는 래퍼 셸까지 전부
#   ./emb_ctl.sh fresh   # 신선도만 확인 — Emb 를 터미널에 직접 띄웠을 때
#   ./emb_ctl.sh status  # 프로세스·SHM·프레임 유통 상태
#
# ★Emb 는 EtherCAT 이 OP 를 잃으면 스스로 복구하지 못한다(commEtherCATm.cpp:520 이 조기 return
#   하여 복구 경로인 CheckState 에 영영 도달하지 못함). 유일한 복구는 이 스크립트의 stop→start.
set -u

EMB_DIR=/home/rpetubt/ZSource/RobotEmbedded/build
EMB_BIN=$EMB_DIR/src/RobotEmbedded
# ★로그 기본 /dev/null (2026-09-15): 디버그매크로 5개 ON 상태서 RobotEmbedded 가 /tmp(=SD카드)에
#   2.45GB/h 로그를 쏟아 **SD I/O 포화 → GUI 의 cmd 파일쓰기 스톨 → deploy 명령워치독(500ms) →
#   무경고 limp**(실측: cmd_seq 0.59s 동결 → "명령두절 0.59s>0.50s → limp"). 매크로는 printf 뿐이라
#   데이터엔 무해(defineGeneral.h:44-54). 매크로 OFF 재빌드 전까지 홍수를 여기서 차단한다.
#   실로그가 필요하면 EMB_LOG=/tmp/emb.log 로 override. 매크로 OFF 후엔 실로그 복원 가능.
LOG=${EMB_LOG:-/dev/null}
DIAG=$(cd "$(dirname "$0")" && pwd)

running(){ pgrep -x RobotEmbedded >/dev/null 2>&1; }

# ★중복 기동은 EtherCAT 버스를 깬다. 어떤 명령이든 먼저 알린다.
_n=$(pgrep -x RobotEmbedded 2>/dev/null | wc -l)
if [ "$_n" -gt 1 ] 2>/dev/null; then
    echo "⚠⚠ RobotEmbedded 가 **$_n 개** 떠 있다 — EtherCAT 마스터가 여럿이면 버스가 깨진다."
    echo "   pid: $(pgrep -x RobotEmbedded | tr '\n' ' ')"
    echo "   'diag/emb_ctl.sh stop' 으로 전부 정리한 뒤 다시 시작할 것."
fi

case "${1:-status}" in
start)
    if running; then echo "이미 실행 중 (pid $(pgrep -x RobotEmbedded | tr '\n' ' ')) — 모터 명령 writer 는 하나만."; exit 1; fi
    if pgrep -f "biped_emb.py|RobotTestGait|mot_test" >/dev/null 2>&1; then
        echo "다른 writer(app/RobotTestGait/mot_test)가 실행 중 — 먼저 종료할 것."; exit 1
    fi
    # ★★기동 전 capability 검사 (2026-09-02). ecx_init 은 eth0 **raw 소켓**이라
    #   cap_net_raw 가 없으면 조용히 "[EtherCAT] ecx_init failed" 로 죽는다.
    #   그런데 이 capability 는 **바이너리 파일에 붙는 것**이라, RGA 가 펌웨어를
    #   갱신(=파일 교체)하거나 재빌드하면 소리 없이 사라진다 — 실제로 08/31 RGA
    #   업데이트 후 첫 기동(09-02)이 그렇게 20초를 날리고 죽었다. 여기서 미리 잡는다.
    if ! getcap "$EMB_BIN" 2>/dev/null | grep -q cap_net_raw; then
        echo "✗ $EMB_BIN 에 cap_net_raw 가 없다 — EtherCAT raw 소켓을 못 열어 ecx_init 이 죽는다."
        echo "  (바이너리가 교체/재빌드되면 capability 는 소리 없이 사라진다 — RGA 업데이트 후 필수)"
        echo "  복구:  sudo setcap cap_net_admin,cap_net_raw+eip $EMB_BIN"
        exit 1
    fi
    echo "[emb_ctl] 기동 → $LOG"
    # ★배너 정정 (2026-08-26). 종전 문구는 "전 관절을 4.5초에 걸쳐 0°로 램프한다(Kp=20/Kd=5)"
    #   였는데 **목표자세도 게인도 둘 다 낡았다.** 운전자가 기동할 때마다 보는 줄이라 남겨 둔다.
    #   ① 목표자세 — halGait.cpp:586 이 2026-08-10 부터 목표를 **측정각**으로 덮어쓴다
    #      (m_fGaitCmd_PositionInit[i] = m_fGaitStt_Position[i]). Befo == Curr 이니
    #      half-sine 보간 결과가 상수 = **현재 자세**다 ⇒ 로봇은 움직이지 않는다.
    #      ⇒ 기동 중 **안 움직이는 것이 정상**이다. 움직였다면 그 패치가 빠진 옛 바이너리다.
    #   ② 게인 — "Kp=20/Kd=5" 의 출처는 halGait.cpp:735-736 의 **주석 처리된 옛 줄**
    #      (//<--20, //<--1)이다. 죽은 줄을 문서로 옮겨 적은 것이었다. 실제로 매 틱
    #      램프 중에 축을 잡는 값은 halGait.cpp:743-759 (채널 좌표):
    #      ⚠단 **기동 최초 100틱(≈0.1초)은 무여자**다 — :611-612 가 그 구간엔
    #        m_stSettingCmdInit(:403-404 Kp=0·Kd=0)로 덮어쓴다. 게인은 4500틱 램프부터 걸린다.
    #          hip 100/6.0 · thigh 50/4.0 · knee 80/3.5 · ankle 30/2.0
    #   ⚠"주변 확인" 의 근거가 바뀌었다: 이제 **움직여서**가 아니라 **제자리를 강하게 물어서**다.
    #     램프 게인이 그대로 걸리므로 관절이 굳고, 사람이 잡고 있으면 되민다.
    #   ⚠4.6초(상태수신 100틱 + 램프 4500틱 @1kHz) 동안 SHM 명령이 무시되는 것은 **지금도 참**이다
    #     (m_ucIsGaitInitialized=0). 소프트웨어로 못 막으므로 그냥 기다리는 수밖에 없다.
    echo "⚠ 기동 직후 4.6초는 Emb 초기화 램프 구간 — 그동안 SHM 명령은 **무시된다**(소프트로 못 막음)."
    echo "   목표가 측정각이라 로봇은 **제자리를 유지한다** — 안 움직이는 것이 정상이다(2026-08-26)."
    echo "   단 관절 게인은 그대로 걸린다(hip 100/6.0 · thigh 50/4.0 · knee 80/3.5 · ankle 30/2.0)."
    echo "   ⇒ 제자리를 강하게 물고 되미니, 잡고 있는 손·기대 놓은 물건을 확인할 것!"
    # ★**sudo 없이 먼저 시도한다** (2026-08-12).
    #   바이너리에 capabilities 가 박혀 있어 root 가 필요 없다:
    #       getcap → cap_net_admin,cap_net_raw=eip   (EtherCAT 원시소켓)
    #       /dev/spidev0.0,0.1 은 dialout 그룹 rw, 사용자가 dialout 소속
    #   ⇒ sudo 를 쓰면 얻는 게 없고 잃는 게 크다: root 셸이 고아로 남아 Emb 를 되살리고
    #     EtherCAT 마스터가 중복된다(2026-08-12 4개까지 늘어 전 채널이 얼어붙었다).
    #   ⚠/dev/gpiomem*·/dev/mem 은 여전히 root 전용이다. 그걸 쓴다면 무권한 기동이
    #     실패하므로 **그때만** sudo 로 재시도한다.
    #   ★sudoers 규칙은 "정확히 이 바이너리 경로"만 NOPASSWD 이므로 sh -c 로 감싸지 말 것.
    #     리다이렉션은 호출측(비root) 셸이 수행 → 로그 파일 소유자는 rpetubt.
    # ★stdbuf -oL — **줄단위 버퍼링 강제** (2026-08-12).
    #   터미널로 직접 띄우면 로그가 콸콸 올라오는데 파일로 넘기면 0바이트였다.
    #   stdout 이 파이프·파일이면 libc 가 **블록 버퍼링**(4~8KB)으로 바꾸고,
    #   Emb 는 영원히 도니 버퍼가 flush 되지 않는다. pkill 로 죽이면 **통째로 버려진다.**
    #   ⇒ "RobotEmbedded 는 stdout 에 안 쓴다" 는 오판이었다. 실은 쓰는데 안 보였다.
    #     그 오판 때문에 EtherCAT 동결 원인 규명이 하루 막혔다.
    # ★반복 스팸을 **솎아서** 쌓는다 (2026-08-14). 필터가 아예 없어서 로그가
    #   **495 KB/s = 시간당 1.7GB** 로 자랐다 — 3.5GB 까지 갔고 디스크를 잡아먹었다.
    #   ⚠전부 버리면 안 된다: `[STT]RxCnt` 는 EtherCAT **동결 판별의 증거**다
    #     (RxCnt 가 안 늘면 동결). 그래서 버리는 게 아니라 **1/N 로 솎는다.**
    #     N=500 이면 초당 5줄쯤 남아 동결 판별에는 차고 넘치고, 1 KB/s 로 떨어진다.
    #     에러·경고 등 **그 외 모든 줄은 그대로 통과**한다.
    #   ⚠awk 에 `fflush()` 를 넣는다 — stdout 이 파일이면 libc 가 블록 버퍼링으로
    #     바꾸는 그 문제가 awk 에도 똑같이 생긴다(2026-08-12 에 Emb 에서 겪었다).
    #   EMB_LOG_EVERY=1 로 두면 종전처럼 전량 기록한다(단기 정밀진단용).
    # ★★카운터를 **패턴마다 따로** 둔다 (2026-08-14 개선). 하나를 공용으로 쓰면
    #   **동결 때 불리하다** — `[STT]RxCnt` 가 멎어도 `[engRobot_Proc` 이 카운터를 계속
    #   돌리므로, 드물게 나오는 STT 줄이 하필 건너뛰어질 수 있다.
    #   이 로그의 존재 이유가 동결 진단이므로 그 경우에 유리한 쪽을 택한다.
    #   패턴별이면 각 패턴이 **독립적으로** 1/n 로 남는다.
    _every=${EMB_LOG_EVERY:-500}
    ( cd "$EMB_DIR" && stdbuf -oL -eL "$EMB_BIN" 2>&1 \
        | awk -v n="$_every" '
            /^\[STT\]RxCnt/    { if (++a % n) next }
            /^\[SET\]RxCnt/    { if (++b % n) next }
            /^\[engRobot_Proc/ { if (++d % n) next }
            { print; fflush() }' \
        > "$LOG" & )
    sleep 2
    if ! running; then
        echo "  무권한 기동 실패 → sudo 로 재시도(gpiomem/mem 접근이 필요한 듯):"
        tail -5 "$LOG" 2>/dev/null | sed 's/^/    /'
        # ⚠sudo 경로는 stdbuf 로 감쌀 수 없다 — sudoers 규칙이 "정확히 이 바이너리
        #   경로"만 NOPASSWD 라 stdbuf 를 끼우면 암호를 묻는다. 이 경로로 떨어지면
        #   로그가 블록 버퍼링된다(위 주석). 무권한 기동이 되는 한 이 경로는 안 쓴다.
        ( cd "$EMB_DIR" && sudo "$EMB_BIN" > "$LOG" 2>&1 & )
        sleep 2
    fi
    if ! running; then echo "✗ 기동 실패:"; tail -30 "$LOG" 2>/dev/null; exit 1; fi
    echo "[emb_ctl] 권한: $(ps -o user= -p "$(pgrep -x RobotEmbedded | head -1)" 2>/dev/null)"
    echo "[emb_ctl] pid $(pgrep -x RobotEmbedded | tr '\n' ' ') — halGait 초기화 대기(≈5s)"
    # ★플래그만으로는 부족하다. stt_probe 의 "값이 갱신됨" 판정(신선도)까지 확인한다.
    for i in $(seq 1 20); do
        sleep 1
        if "$DIAG/stt_probe" 6 2>/dev/null | grep -q "값이 갱신됨"; then
            echo "✓ MotorStatus16 신선 — EtherCAT 생존 + Emb 가 SHM 명령을 읽는 상태 (${i}s)"; exit 0
        fi
    done
    # ★실패했으면 **띄운 것을 반드시 정리한다** (2026-08-12).
    #   종전엔 그냥 exit 2 라 죽은 Emb 가 계속 떠 있었다. 사용자가 수동으로 다시 띄우면
    #   **EtherCAT 마스터가 둘**이 되어 버스를 서로 물어뜯는다 — 실제로 4개까지 늘어나
    #   전 채널이 얼어붙었다. 신선하지 않은 Emb 는 남겨둘 가치가 없다(모터 명령은 계속
    #   재전송하면서 상태는 못 읽는 상태다).
    echo "✗ 20s 내 신선한 MotorStatus16 미수신."
    echo "  → 띄운 Emb 를 정리한다(그냥 두면 재시도 시 EtherCAT 마스터가 둘이 된다)."
    sudo pkill -x RobotEmbedded 2>/dev/null; sleep 1
    n=$(pgrep -x RobotEmbedded 2>/dev/null | wc -l)
    echo "  남은 RobotEmbedded: $n"
    "$0" log
    echo
    echo "  다음 순서로 복구할 것:"
    echo "    ① 모터 전원 OFF → 3초 → ON   (EtherCAT 슬레이브 초기화. Emb 재기동만으론 부족)"
    echo "    ② diag/emb_ctl.sh start"
    echo "  ⚠수동으로 RobotEmbedded 를 따로 띄우지 말 것 — 중복 기동이 버스를 깬다."
    exit 2
    ;;
reset)
    # ★"싹 종료" — 모터를 만질 수 있는 모든 프로세스를 없앤다 (2026-08-12).
    #   ⚠stop 만으로는 부족했다. Emb 를 죽여도 **그걸 띄운 래퍼 셸이 살아남아** 다음
    #     명령에서 다시 띄운다. 실기에서 계보가 이랬다:
    #         RobotEmbedded ← sudo ← sudo ← bash(PPID=1, 고아)
    #     emb_ctl.sh 를 sudo 로 실행해서 sudo 가 두 겹이 됐고, 터미널을 닫아도
    #     그 bash 가 살아 EtherCAT 마스터가 4개까지 늘었다.
    #   ⇒ writer → Emb → 래퍼 셸 순으로 지우고, 마지막에 남은 게 없는지 확인한다.
    #   ★자기 자신과 조상은 절대 죽이지 않는다(스크립트가 중간에 죽으면 정리가 안 끝난다).
    echo "[emb_ctl] reset — 모터를 만질 수 있는 프로세스를 전부 정리한다"
    _keep=" $$ $PPID "
    _anc=$PPID; for _ in 1 2 3 4 5; do
        _anc=$(ps -o ppid= -p "$_anc" 2>/dev/null | tr -d ' '); [ -z "$_anc" ] && break
        _keep="$_keep$_anc "; [ "$_anc" = "1" ] && break
    done

    # ★pkill -f 를 쓰지 않는다. **자기 자신을 죽인다.**
    #   2026-08-12: 이 스크립트를 시험하던 셸의 argv 에 "actuator_test.py" 라는 문자열이
    #   들어 있었더니 pkill -f 가 그 셸을 죽여 reset 이 ①에서 끊겼다(exit 144).
    #   패턴이 인자·히스토리·에디터 명령줄에 우연히 들어가는 건 흔한 일이다.
    #   ⇒ PID 를 모아 조상 제외 후 하나씩 죽인다. _kill_matching 로 ①③ 공통 처리.
    _stuck=""
    _kill_matching() {   # $1 = ERE 패턴, $2 = 라벨
        for _p in $(ps -eo pid=,args= | grep -E "$1" | grep -vE "grep -E|ps -eo" \
                    | awk '{print $1}'); do
            case "$_keep" in *" $_p "*) continue;; esac
            _a=$(ps -o args= -p "$_p" 2>/dev/null | cut -c1-70)
            [ -z "$_a" ] && continue
            echo "     kill $_p  $_a"
            # ★sudo 는 **-n(비대화)** 로 쓴다 (2026-08-12). 종전엔 `sudo kill -9 <pid>`
            #   였는데 sudoers 의 NOPASSWD 는 **딱 셋뿐**이다:
            #       RobotEmbedded 바이너리 · pkill -x RobotEmbedded · pkill -9 -x RobotEmbedded
            #   `kill -9 <pid>` 는 거기 없다 → **비밀번호를 묻는다.** 그런데 2>/dev/null 로
            #   프롬프트가 가려져, 화면엔 아무것도 안 뜬 채 sudo 가 입력을 기다린다.
            #   ⇒ reset 이 조용히 멈춘 것처럼 보이고 정리가 안 끝난다.
            #   -n 이면 즉시 실패하고, 무엇을 못 죽였는지 이름을 대고 넘어간다.
            if ! kill -9 "$_p" 2>/dev/null && ! sudo -n kill -9 "$_p" 2>/dev/null; then
                echo "     ✗ 못 죽였다 pid $_p (root 소유) — 아래 수동 명령 참조"
                _stuck="$_stuck $_p"
            fi
        done
    }
    echo "  ① writer 종료 (biped_emb.py · actuator_test.py · RobotTestGait · mot_test)"
    _kill_matching "biped_emb\.py|actuator_test\.py|collect_multichirp\.py|RobotTestGait|mot_test"
    sleep 1

    echo "  ② Emb 종료"
    sudo -n pkill -x RobotEmbedded 2>/dev/null; sleep 1
    if pgrep -x RobotEmbedded >/dev/null 2>&1; then
        echo "     TERM 으로 안 죽는다 → KILL"
        sudo -n pkill -9 -x RobotEmbedded 2>/dev/null; sleep 1
    fi

    echo "  ③ 되살리는 래퍼 셸·sudo 정리"
    _kill_matching "RobotEmbedded|emb_ctl\.sh"
    sleep 1

    # ⚠pgrep -c 는 "0" 을 **찍으면서 exit 1** 이다. `|| echo 0` 을 붙이면
    #   출력이 "0\n0" 이 되어 문자열 비교가 깨진다(2026-08-12 실제로 실패분기로 빠졌다).
    _n=$(pgrep -x RobotEmbedded 2>/dev/null | wc -l)
    _w=0
    for _p in $(ps -eo pid=,args= | grep -E "biped_emb\.py|actuator_test\.py" \
                | grep -vE "grep -E|ps -eo" | awk '{print $1}'); do
        case "$_keep" in *" $_p "*) continue;; esac
        _w=$((_w+1))
    done
    echo
    if [ -n "$_stuck" ]; then
        echo "  ⚠ root 소유라 못 죽인 프로세스: $_stuck"
        echo "    sudoers 에는 RobotEmbedded 관련 3개만 NOPASSWD 다 — 나머지는 수동으로:"
        echo "      sudo kill -9$_stuck"
        echo "    ⚠애초에 **Emb 를 sudo 없이 띄우면** 이 상황이 안 생긴다(RUNBOOK §1)."
    fi
    echo "  남은 것 — Emb $_n 개 · writer $_w 개"
    if [ "$_n" != "0" ] || [ "$_w" != "0" ]; then
        echo "  ✗ 아직 남았다:"; ps -eo pid,ppid,etime,args | grep -E "RobotEmbedded|biped_emb\\.py|actuator_test\\.py" | grep -vE "grep -E|ps -eo"   # ★\.py 를 붙인다 — 뷰어의 biped_emb.**yaml** 경로에 걸렸었다
        exit 1
    fi
    echo "  ✓ 전부 정리됨"
    echo
    echo "  다음: ① 모터 전원 OFF → 3초 → ON   ② diag/emb_ctl.sh start"
    echo "  ⚠ emb_ctl.sh 에 sudo 를 붙이지 말 것 — 스크립트가 안에서 쓴다. 밖에서 한 겹 더"
    echo "    씌우면 sudo 가 이중이 되고 고아 셸이 남는다(이번 사고의 원인)."
    exit 0
    ;;
stop)
    sudo pkill -x RobotEmbedded 2>/dev/null
    sleep 1
    running && { echo "강제 종료"; sudo pkill -9 -x RobotEmbedded; }
    echo "[emb_ctl] 종료됨"
    ;;
log)
    echo "=== EtherCAT 요약 ==="
    grep -aE "EtherCAT|slave|Slave|WKC|OP|SAFEOP|mismatch|Failed|failure|lost|recover" "$LOG" 2>/dev/null | head -40
    echo "=== 마지막 15줄 ==="
    tail -15 "$LOG" 2>/dev/null
    ;;
fresh)
    # ★신선도만 확인한다 — Emb 를 **터미널에 직접 띄웠을 때** 쓴다.
    #   직접 띄우면 로그가 줄단위 버퍼링이라 눈앞에 실시간으로 보인다(동결 원인 규명에
    #   그게 제일 값어치 있다). 대신 start 가 해주던 신선도 확인이 빠지므로 이걸 쓴다.
    #   ⚠플래그·프로세스 존재로는 판별 불가 — Emb 는 EtherCAT OP 를 잃어도 마지막 버퍼를
    #     재발행하고 IsUpdated 까지 1 로 세운다. **값이 실제로 변하는지**를 봐야 한다.
    if ! running; then echo "✗ Emb 가 안 떠 있다."; exit 1; fi
    _n=$(pgrep -x RobotEmbedded 2>/dev/null | wc -l)
    [ "$_n" -gt 1 ] && { echo "✗ Emb 가 $_n 개 — reset 후 하나만 띄울 것."; exit 1; }
    for i in $(seq 1 20); do
        if "$DIAG/stt_probe" 6 2>/dev/null | grep -q "값이 갱신됨"; then
            echo "✓ MotorStatus16 신선 — EtherCAT 생존 (${i}s)"; exit 0
        fi
        sleep 1
    done
    echo "✗ 20s 내 신선한 MotorStatus16 미수신 — EtherCAT OP 이탈."
    echo "  ① reset  ② 모터 전원 OFF → 3초 → ON  ③ 다시 기동"
    exit 2
    ;;
status)
    running && echo "Emb: 실행 중 (pid $(pgrep -x RobotEmbedded | tr '\n' ' '))" || echo "Emb: 정지"
    pgrep -af "biped_emb.py|RobotTestGait|mot_test" || echo "다른 writer: 없음"
    ipcs -m | awk 'NR<4 || /0x000004d2/'
    # EtherCAT 프레임이 실제로 오가는지 — OP 이탈 시 TX 가 0 증가로 멈춘다.
    A=$(cat /sys/class/net/eth0/statistics/tx_packets 2>/dev/null || echo 0)
    sleep 1
    B=$(cat /sys/class/net/eth0/statistics/tx_packets 2>/dev/null || echo 0)
    echo "eth0 tx_packets 1초 증가: $((B-A))  (0 이면 EtherCAT 정지 → Emb 재기동 필요)"
    ;;
*)
    sed -n '2,10p' "$0"; exit 1;;
esac
