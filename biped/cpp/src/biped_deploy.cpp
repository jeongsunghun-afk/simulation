// biped_deploy.cpp — ★C++ 실기 배포 (§9, 핸드오프 미완료 #4).
//
//   biped_sim.cpp 의 `mj_step` 자리에 **실모터 read/write** 를 넣은 것이다.
//   데이터흐름:  HW.read → 관절매핑(deg→rad) → 추정(leg-odom) → 모델 주입 → mj_forward
//              → BipedControl.control → d->ctrl(토크) → 관절매핑(rad→deg) → HW.write_mit
//
//   모드: off / hold / home / jog / stand / walk
//     jog·home 은 위치제어(kp/kd, τ_ff 없음) · stand/walk 는 WBIC 토크(+kd 플로어).
//     emb/app/biped_emb.py 와 **같은 config 키**를 읽는다(jog.max_speed_dps / home.*).
//   ⚠ **모터 명령 writer 는 한 번에 하나만.** 이 바이너리를 돌릴 땐 biped_emb.py 를 끌 것.
//
//   실행:
//     ./biped_deploy                       # 실기(SHM). Emb 기동 후 5초 뒤
//     ./biped_deploy --mock                # 하드웨어 없이 로직 검증
//     ./biped_deploy --mjcf ../biped_from_quad.mjcf --T 30
//
// ★안전장치는 전부 emb/app/biped_emb.py 에서 이식했다. 실측으로 다듬어진 것들이라
//   재발명하지 않는다 — 워치독 · tilt/토크/속도 E-stop(래치) · 종료 시 limp 반복기록.
#include <mujoco/mujoco.h>
#include <set>
#include <dirent.h>
#include <unistd.h>
#include <fstream>
#include <utility>
#include <sstream>
#include "biped_control.hpp"
#include "state_estimator.hpp"
#include "deploy_hw.hpp"
#include "freeze_forensics.hpp"      // ★동결 증거 수집(링버퍼·Emb 생존·NIC·사건로그)

// ── ★SoC 열·클럭 스냅샷 (2026-08-24) ────────────────────────────────────
//   왜 동결 증거에 넣는가 — 실기에서 **88.7°C** 가 찍혔다(Pi5 는 80°C 소프트·85°C 강제
//   클럭다운). 스로틀은 **RT 우선순위로 못 막는다**: 스레드를 가리지 않고 전부 느려진다.
//   `RT_PRIO=50` 이 안 먹혔던 것과 앞뒤가 맞는 유일한 경로라, 동결마다 기록해 둔다.
//   ⚠온도만 봐선 인과를 못 정한다. **동결 때만 높은지 평상시도 높은지**를 갈라야 해서
//     사건로그에 매번 남긴다 — 여러 건을 모아 놓고 봐야 상관이 보인다.
struct SocSnap {
  double temp_c = -1;      // 최고 thermal zone
  double mhz    = -1;      // 최고 현재 클럭
  long   thr    = -1;      // vcgencmd get_throttled (없으면 -1)
  static SocSnap take(){
    SocSnap s;
    for(int z=0; z<8; z++){
      char f[96]; std::snprintf(f,sizeof f,"/sys/class/thermal/thermal_zone%d/temp",z);
      if(FILE* fp=std::fopen(f,"r")){ long v=0; if(std::fscanf(fp,"%ld",&v)==1){
        double c=v/1000.0; if(c>0 && c<200 && c>s.temp_c) s.temp_c=c; } std::fclose(fp); }
    }
    for(int c=0; c<8; c++){
      char f[128]; std::snprintf(f,sizeof f,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq",c);
      if(FILE* fp=std::fopen(f,"r")){ long v=0; if(std::fscanf(fp,"%ld",&v)==1){
        double m=v/1000.0; if(m>s.mhz) s.mhz=m; } std::fclose(fp); }
    }
    if(FILE* fp=popen("vcgencmd get_throttled 2>/dev/null","r")){
      char b[64]={0}; if(std::fgets(b,sizeof b,fp)){ const char* eq=std::strchr(b,'=');
        if(eq) s.thr = std::strtol(eq+1,nullptr,0); } pclose(fp); }
    return s;
  }
};
#include "sim_hw.hpp"
#include <Eigen/Dense>
#include <csignal>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <sched.h>
#include <sys/mman.h>
#include <cmath>
#include <string>
#include <vector>

using namespace bipedhw;

static volatile std::sig_atomic_t g_stop = 0;
static void on_sig(int s){ g_stop = s; }

static double now_s(){
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec*1e-9;
}
static void sleep_s(double s){
  if(s <= 0) return;
  struct timespec ts{ (time_t)s, (long)((s-(double)(time_t)s)*1e9) };
  nanosleep(&ts, nullptr);
}

// RPY(rad) → quat(wxyz). hw_interface.rpy_to_quat 과 동일(ZYX).
static void rpy_to_quat(double r, double p, double y, double* q){
  double cr=std::cos(r/2), sr=std::sin(r/2), cp=std::cos(p/2), sp=std::sin(p/2),
         cy=std::cos(y/2), sy=std::sin(y/2);
  q[0]=cr*cp*cy + sr*sp*sy; q[1]=sr*cp*cy - cr*sp*sy;
  q[2]=cr*sp*cy + sr*cp*sy; q[3]=cr*cp*sy - sr*sp*cy;
}

// ★어떤 경로로 죽어도 **실제로** 무여자로 만든다.
//   bridge_enable(0) 만으로는 정지가 아니다 — g_enabled 플래그만 바꿀 뿐 SHM 명령버퍼는
//   그대로라, Emb 가 마지막 명령을 1kHz 로 영원히 재전송한다(shm_bridge.cpp:115).
//   Kp=Kd=0 을 **반복 기록**해야 진짜 limp 다. (biped_emb.safe_shutdown 과 같은 이유)
static void safe_shutdown(HwIface& hw, int n){
  std::vector<float> z(n,0.f), q(n,0.f);
  int ok=0;
  hw.enable(0);
  for(int i=0;i<25;i++){
    if(hw.write_pos(q.data(), z.data(), z.data(), n)==0) ok++;
    sleep_s(0.002);
  }
  if(ok==0){
    std::fprintf(stderr,
      "\n%s\n!! limp 실패 — SHM 에 무여자 명령을 한 번도 쓰지 못했다."
      "\n!! Emb 는 마지막 명령을 1kHz 로 계속 재전송한다. **모터 전원을 차단할 것**.\n%s\n",
      std::string(68,'!').c_str(), std::string(68,'!').c_str());
  } else {
    // ⚠"무여자" 라 쓰지 않는다 — SHM 에 드라이브 disable 경로가 **없다**(벤더 확인 2026-08-14).
    //   보장되는 건 명령토크 0 까지고, 드라이브는 여전히 여자 상태다.
    std::printf("[deploy] ⚠**체중을 받고 있었다면 지금 주저앉는다** — 무여자는 낙하다.\n"
                "         다시 세우려면: 크레인으로 들어올림 → home → 접지 → hold → stand\n");
    std::printf("[deploy] 종료 — 명령토크 0(Kp=Kd=τ=0) %d/25 회 기록 완료.\n"
                "         ⚠드라이브는 여전히 여자 상태다 — 축이 안 풀리면 물리 리셋/전원 차단.\n", ok);
  }
}

// ── 중복 writer 차단 ────────────────────────────────────────────────────────
// ★★저장소 최상위 불변식: **모터 명령 writer 는 한 번에 하나만.**
//   그런데 강제는 `biped_emb.py`(:190~228)에만 있었고 **C++ 배포에는 없었다** — 경고문만 찍었다.
//       biped_emb.py 먼저 → biped_deploy 나중  →  ❌ 안 막힘  ← 여기(2026-08-24 신설)
//       biped_deploy 먼저 → biped_emb.py 나중  →  ✅ 막힘
//   둘이 뜨면 SHM 에 서로 다른 명령을 번갈아 쓴다. 실기 사고: 2026-08-10 관절 **+18° ↔ −20°** 진동.
//   ⚠Python 가드의 예외 셋을 그대로 옮긴다 — 안 옮기면 **가짜 경보가 진짜 경보를 무디게 한다**:
//     ① 자기 자신·**조상 프로세스**(띄운 셸) 제외
//     ② **빌드 프로세스** 제외 — `cmake --build --target biped_deploy` 등이 이름에 걸린다
//        (2026-08-14 실기에서 실제로 오작동해 writer 가 없는데도 기동을 막았다)
//     ③ pgrep/grep 자신 제외
static std::vector<std::pair<int,std::string>> find_other_writers(){
  static const char* PATS[]  = {"biped_emb.py","biped_deploy","mot_test","actuator_test.py"};
  static const char* TOOLS[] = {"cmake","gmake","make","cc1plus","c++","g++","ld","ninja","sh","bash"};
  const int me = (int)getpid();
  std::set<int> anc;
  for(int p=me, guard=0; p>1 && guard<32; guard++){
    char sp[64]; std::snprintf(sp,sizeof sp,"/proc/%d/stat",p);
    std::ifstream f(sp); if(!f) break;
    std::string line; std::getline(f,line);
    const size_t r = line.rfind(')');
    if(r==std::string::npos) break;
    int ppid=0; if(std::sscanf(line.c_str()+r+1, " %*c %d", &ppid)!=1) break;
    if(ppid<=1) break; anc.insert(ppid); p=ppid;
  }
  std::vector<std::pair<int,std::string>> out;
  DIR* dp = opendir("/proc"); if(!dp) return out;
  while(dirent* e = readdir(dp)){
    const int pid = atoi(e->d_name);
    if(pid<=1 || pid==me || anc.count(pid)) continue;
    char cp[64]; std::snprintf(cp,sizeof cp,"/proc/%d/cmdline",pid);
    std::ifstream f(cp, std::ios::binary); if(!f) continue;
    std::string raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if(raw.empty()) continue;
    std::string cl = raw; for(size_t i=0;i<cl.size();i++) if(cl[i]=='\0') cl[i]=' ';
    if(cl.find("pgrep")!=std::string::npos || cl.find("grep")!=std::string::npos) continue;
    std::string exe0 = cl.substr(0, cl.find(' '));
    { size_t sl=exe0.find_last_of('/'); if(sl!=std::string::npos) exe0 = exe0.substr(sl+1); }
    bool is_tool=false; for(const char* t : TOOLS) if(exe0==t){ is_tool=true; break; }
    if(is_tool) continue;
    for(const char* pat : PATS){
      if(cl.find(pat)!=std::string::npos){
        while(!cl.empty() && cl.back()==' ') cl.pop_back();
        out.push_back(std::make_pair(pid, cl.substr(0, 90))); break; }
    }
  }
  closedir(dp);
  return out;
}

int main(int argc, char** argv){
  std::string mjcf   = "../biped_from_quad.mjcf";     // ★배포는 점발(1pt). §8-g 참조
  std::string cfg_p  = "../emb/config/biped_emb.yaml";
  std::string cmd_p  = "/tmp/biped_cmd.json";
  std::string stt_p  = "/tmp/biped_state.json";
  bool mock = false, simhw = false, force_writer = false; double T = 1e12; std::string start_mode = "off";
  for(int i=1;i<argc;i++){
    std::string a = argv[i];
    if(a=="--force") force_writer = true;          // ★중복 writer 강행(권장 안 함)
    else if(a=="--mock") mock = true;
    // ★--sim : **물리 백엔드**. 모드·워치독·트립이 전부 MuJoCo 플랜트 위에서 돈다.
    //   --mock 은 물리가 없어 "블렌드가 안정한가·float 이 실제로 뜨는가" 를 못 본다.
    else if(a=="--sim") simhw = true;
    else if(a=="--mjcf" && i+1<argc) mjcf = argv[++i];
    else if(a=="--config" && i+1<argc) cfg_p = argv[++i];
    else if(a=="--cmd" && i+1<argc) cmd_p = argv[++i];
    else if(a=="--state" && i+1<argc) stt_p = argv[++i];
    else if(a=="--T" && i+1<argc) T = atof(argv[++i]);
    // ★biped_emb.py 와 **같은 이름의 옵션**을 둔다 — 두 제어기의 기동 규약을 맞춘다.
    //   off  = 무여자로 시작(기본·권장). 명령파일에 뭐가 남아 있든 무시한다.
    //   hold = 생존확인 통과 후 **현재 자세를 그대로** 잡는다(로봇이 정지해 있어야 한다).
    else if(a=="--start-mode" && i+1<argc) start_mode = argv[++i];
    else { std::printf("사용법: %s [--mock] [--mjcf X] [--config X] [--cmd X] [--state X] [--T s]\n"
                       "        [--start-mode off|hold]   기본 off (기동 즉시 무장 방지)\n", argv[0]); return 2; }
  }
  if(start_mode!="off" && start_mode!="hold"){   // start_mode 검증
    std::printf("✗ --start-mode 는 off 또는 hold 다 (받은 값: %s)\n", start_mode.c_str()); return 2; }
  if(const char* e=getenv("QUAD_CMD"))   cmd_p = e;
  if(const char* e=getenv("QUAD_STATE")) stt_p = e;

  // ★★실시간 우선순위 (2026-08-20). 500Hz 루프가 일반 우선순위로 돌면 밀린다 —
  //   실측: 28~51ms 스톨(20~25틱 유실). 이 Pi 는 RobotEmbedded 가 1kHz 로 CPU 90% 를
  //   쓰고 데스크톱(gnome-shell·Xwayland·모니터)까지 얹혀 있어 경합이 실재한다.
  //   ⚠기본 사용자는 rtprio 한도가 0 이라 실패한다 — 그때는 경고만 하고 계속 돈다.
  //     sudo setcap cap_sys_nice+ep <바이너리>
  //     ⚠**재빌드하면 사라진다**(재링크가 파일을 새로 만든다). 실측 확인했다.
  //   ⚠SCHED_FIFO 는 이 스레드가 CPU 를 독점할 수 있다는 뜻이다. 500Hz 루프는
  //     대부분 sleep 이라 안전하지만, 무한루프 버그가 나면 기기가 멎는다.
  { const int rp = getenv("RT_PRIO") ? atoi(getenv("RT_PRIO")) : 80;
    if(rp > 0){
      struct sched_param sp; sp.sched_priority = rp;
      if(sched_setscheduler(0, SCHED_FIFO, &sp) == 0){
        std::printf("[deploy] 실시간 우선순위 **SCHED_FIFO %d** 적용\n", rp);
        if(mlockall(MCL_CURRENT|MCL_FUTURE) != 0)
          std::printf("[deploy] ⚠mlockall 실패 — 페이지폴트로 지터가 남을 수 있다\n");
      } else {
        std::printf("[deploy] ⚠실시간 우선순위 실패(%s) — **루프가 밀릴 수 있다**(실측 28~51ms 스톨).\n"
                    "         sudo setcap cap_sys_nice+ep %s\n"
                    "         ⚠**재빌드(재링크)하면 사라진다** — 소스를 고칠 때마다 다시 걸 것.\n"
                    "         끄려면 RT_PRIO=0\n", std::strerror(errno), argv[0]);
      } } }

  // ── 설정 ──
  EmbCfg cfg; std::string err;
  if(!load_cfg(cfg_p, cfg, err)){ std::printf("✗ %s\n", err.c_str()); return 1; }
  // ★JOG_SPEED_DPS env (2026-08-28, 에어워크 리허설용) — jog 슬루 한계를 세션 한정으로
  //   개방한다. tools/airwalk.py 가 walk 궤적을 감속 재생할 때 20dps 로는 형상이 뭉개진다.
  //   [5,150] 클램프: 150dps 관절 = calf 채널 225dps 로 cfg 트립(200) 위라, 상한은
  //   "트립이 여전히 최후 방어선으로 남는" 범위다. 무효값은 무시.
  if(const char* e = getenv("JOG_SPEED_DPS")){
    const double v = atof(e);
    if(std::isfinite(v) && v >= 5.0 && v <= 150.0){
      std::printf("[deploy] ⚠JOG_SPEED_DPS %.0f → %.0f dps (에어워크 세션용 — 평시엔 지울 것)\n",
                  cfg.jog_speed_dps, v);
      cfg.jog_speed_dps = v;
    } else std::printf("[deploy] ⚠JOG_SPEED_DPS='%s' 무효(5~150) — 설정값 %.0f 유지\n",
                       e, cfg.jog_speed_dps);
  }
  const int NCH = cfg.n_channel;
  const double dt = 1.0/cfg.ctrl_hz;
  JointMap jm(cfg);
  std::printf("[deploy] config=%s · joints=%d · n_channel=%d · ctrl_hz=%.0f\n",
              cfg_p.c_str(), jm.n_leg, NCH, cfg.ctrl_hz);
  { std::string s; for(auto& j : cfg.joints){
      char b[96]; std::snprintf(b,sizeof b," %s(ch%d,s%+.0f,off%+.1f)", j.name.c_str(), j.channel, j.sign, j.offset_deg);
      s += b; }
    std::printf("[deploy] 매핑:%s\n", s.c_str()); }
  // ★★게인을 **두 좌표로 나란히** 찍는다 (2026-08-21).
  //   config 의 kp/kd 는 **채널** 기준인데(드라이버 MIT 의 err 이 채널각이라 그대로 나간다),
  //   값 모니터·상태 JSON 은 **raw** 기준으로 낸다(kp_raw = kp_ch·gear_k²). 관절 아니다.
  //   숫자가 두 배 넘게 달라 보이는 게 정상인데, 그걸 모르면 "설정이 안 먹었다" 로 오해한다
  //   (실제로 2026-08-21 에 그 오해가 있었다 — 채널값 80/30 이 보여 수정이 안 먹은 줄 알았다).
  //   ⇒ 기동 시 둘을 같이 보여 주면 그 자리에서 대조된다.
  { std::string s;
    for(auto& j : cfg.joints){
      const double k2 = j.gear_k*j.gear_k;
      char b[128]; std::snprintf(b,sizeof b, " %s kp %.0f→%.1f · kd %.1f→%.1f;",
                                 j.name.c_str(), j.kp, j.kp*k2, j.kd, j.kd*k2);
      s += b; }
    std::printf("[deploy] 게인(채널→raw, ×gear_k²):%s\n", s.c_str());
    std::printf("         ⚠config 는 **채널**, 모니터·상태는 **관절**이다. 달라 보이는 게 정상.\n"); }
  if(!cfg.installed.empty()){
    std::string s; for(int ch : cfg.installed){ char b[8]; std::snprintf(b,sizeof b," %d",ch); s+=b; }
    std::printf("[deploy] 실장 채널:%s (나머지는 명령은 나가지만 통신 없어 무효)\n", s.c_str());
  }

  // ── 모델 + 컨트롤러 ──
  char merr[1000]={0};
  mjModel* m = mj_loadXML(mjcf.c_str(), nullptr, merr, 1000);
  if(!m){ std::printf("✗ 모델 로드 실패(%s): %s\n", mjcf.c_str(), merr); return 1; }
  mjData* d = mj_makeData(m);
  BipedControl c(m,d); c.reset();
  const int NJ = m->nq - 7, NU = m->nu;
  if(NJ != jm.n_leg){
    std::printf("✗ 모델 관절수(%d) ≠ config joints(%d) — 모델/설정 불일치\n", NJ, jm.n_leg);
    mj_deleteData(d); mj_deleteModel(m); return 1;
  }
  // ★"점발 1pt" 하드코딩이었다(2026-08-13 수정). 평발 MJCF 를 줘도 그렇게 찍혀서,
  //   운전자가 이 줄로 무엇이 올라왔는지 확인하는데 **거짓말을 하고 있었다.**
  //   cmode 는 heel 구 유무로 자동 결정된다(BipedControl: cmode = has_heel ? 1 : 0).
  std::printf("[deploy] 모델=%s (nq=%d nv=%d nu=%d) · cmode=%d **%s**\n",
              mjcf.c_str(), (int)m->nq, (int)m->nv, (int)m->nu, c.cmode,   // ★mjtSize=long → %d 경고
              c.cmode==1 ? "2점 평발(정적 자세유지)" : "1점 점발(stepping 보행)");

  // ★마찰 전방보상 상태를 **반드시 찍는다**. 켜져 있어도 꺼져 있어도 겉보기 동작이
  //   같아서, 지연보상 때 "설정하면 켜진다" 고 9일간 오해했던 것과 같은 함정이다.
  //   2점 평발에서만 켜진다 — 이 값이 20.73s 낙상을 60s 무낙상(tilt 0.1°)으로 바꿨다.
  { const double fc = getenv("FRIC_COMP") ? atof(getenv("FRIC_COMP")) : 1.0;
    const double fv = getenv("FRIC_V0")   ? atof(getenv("FRIC_V0"))   : 0.20;
    const bool all  = getenv("FRIC_ALL_MODES") && atoi(getenv("FRIC_ALL_MODES"));
    if(fc>0 && (c.cmode==1 || all))
      std::printf("[deploy] 마찰보상 **ON** — ×%.2f · v0=%.2f rad/s (JFRIC 0.827/0.604/0.871/0.639)\n", fc, fv);
    else if(fc>0)
      std::printf("[deploy] 마찰보상 OFF — 1점 보행 모드다(2점 평발에서만 켠다)\n");
    else
      std::printf("[deploy] ⚠마찰보상 **OFF**(FRIC_COMP=0) — 2점 stand 는 ~20s 에 넘어진다\n"); }

  BipedEstimator est;
  { std::vector<int> fg={c.sph[0],c.sph[1]};
    std::vector<double> fr={m->geom_size[c.sph[0]*3], m->geom_size[c.sph[1]*3]};
    est.init(m,fg,fr); est.reset(Eigen::Vector3d(0,0,d->qpos[2])); }

  // ── 지연보상(Smith predictor) ────────────────────────────────────────────
  // ★2026-08-14 신설. 종전엔 **시뮬에만** 있었다(deploy_loop.hpp) — 그런데 그 헤더를
  //   include 하는 건 biped_sim/biped_view 뿐이고, 실기 writer 인 이 바이너리는 아니다.
  //   그래서 `LAT_COMP_MS=8.4` 를 줘도 **아무 일도 일어나지 않았다.**
  //   STABILITY_MAP.md 가 그 값을 "배포 조건" 이라 적어둔 탓에 켜져 있다고 오해했다.
  //
  // 원리: 지금 읽는 센서는 T_d 전의 상태이고, 그 사이 내보낸 토크는 아직 안 나타났다.
  //       ⇒ **마지막 명령토크를 유지한 채** 모델을 T_d 만큼 굴려 "지금"을 만들어 제어한다.
  //   왕복지연 실측 **8.39±0.79 ms**(emb/pace/RESULTS.md ⑥). 모델 timestep 2ms → 4 step.
  //   ⚠LCOMP 는 제어주기가 아니라 **m->opt.timestep** 으로 나눈다 — mj_step 이 그만큼
  //     전진하기 때문이다. 지금은 둘 다 2ms 라 같지만 ctrl_hz 를 바꾸면 갈라진다.
  //
  // ★기본 **켜짐 8.4ms**(2026-08-14 근거 확보). 끄려면 `LAT_COMP_MS=0`.
  //   biped_sim 배포경로(EST_CTRL=1 · ACT_LAT_MS=8.4 · T=15s)에서 잰 값이다:
  //     보상없음  vx 0.05/0.15/0.20 에서 낙상 1/1/1  → 켜면 **전부 0**
  //     stand     base·tilt 가 소수점까지 동일 — **완전 중립**(해가 없다)
  //     보상값 오차 6.8/8.4/10.0/16.0 ms 전부 0 낙상(6.8·vx0.20 만 1)
  //       ⇒ 실측 ±0.79ms 는 물론 과보상 쪽으로도 안전하다. 위험한 건 **덜** 주는 쪽.
  //   ⚠속도잡음 증폭도 무시할 만하다: ENCDQ_N 0.0368 rad/s × 8.4ms = **0.018°**.
  const double MDT = m->opt.timestep;
  const double lat_comp_ms = getenv("LAT_COMP_MS") ? atof(getenv("LAT_COMP_MS")) : 8.4;
  // 안전망 — 예측이 튀면 **버리고 실측을 쓴다**. 시뮬판엔 이 가드가 없다(발산해도
  // 넘어질 뿐이지만, 실기에선 그 토크가 그대로 모터로 나간다).
  const double LC_MAX_DEG = getenv("LAT_COMP_MAX_DEG") ? atof(getenv("LAT_COMP_MAX_DEG")) : 5.0;
  const double LC_MAX_Z   = getenv("LAT_COMP_MAX_Z")   ? atof(getenv("LAT_COMP_MAX_Z"))   : 0.05;
  // ★운동학 외삽(`LAT_COMP_KIN=1`) — **이게 이 기기의 기본값이어야 한다.**
  //   동역학 롤아웃은 mj_step 4회인데 이 Pi 에서 mj_step 이 **604 µs** 다(실측 2026-08-14,
  //   두 모델 모두). 4회 = 2.42ms > 제어주기 2.00ms ⇒ **주기를 못 지킨다**
  //   (mock 실측: 504Hz → 285Hz). 지연을 보상하려다 지연을 만드는 셈이다.
  //   외삽은 q += q̇·T 라 사실상 공짜다(실측 501Hz 유지). 8.4ms 에서 빠지는 건 ½q̈T² 항뿐이고,
  //   시뮬 비교에서 **동역학과 낙상수가 동률**이었다(4속도 전부 0). ⇒ 기본값.
  //   `LAT_COMP_KIN=0` 으로 동역학 롤아웃을 강제할 수 있다(오프라인 연구용. 주기를 못 지킨다).
  const bool LC_KIN = getenv("LAT_COMP_KIN") ? atoi(getenv("LAT_COMP_KIN"))!=0 : true;
  int LCOMP = 0; mjData* dpred = nullptr;
  if(lat_comp_ms > 0){
    LCOMP = (int)std::lround(lat_comp_ms/1000.0/MDT);
    if(LCOMP > 0 && LC_KIN){
      std::printf("[deploy] 지연보상 **ON(운동학 외삽)** — %.2fms. 안전망 |Δq|<%.1f° · |Δz|<%.0fmm\n",
                  lat_comp_ms, LC_MAX_DEG, LC_MAX_Z*1e3);
    } else if(LCOMP > 0){
      dpred = mj_makeData(m);
      std::printf("[deploy] 지연보상 **ON(동역학 롤아웃)** — %.2fms = %d step(모델 dt %.1fms)."
                  " 안전망 |Δq|<%.1f° · |Δz|<%.0fmm\n"
                  "[deploy] ⚠⚠ mj_step %d회 ≈ %.2fms > 제어주기 %.2fms — **주기를 못 지킨다.**"
                  " 이 기기에선 LAT_COMP_KIN=1 을 쓸 것.\n",
                  lat_comp_ms, LCOMP, MDT*1e3, LC_MAX_DEG, LC_MAX_Z*1e3,
                  LCOMP, LCOMP*0.604, dt*1e3);
    } else {
      std::printf("[deploy] ⚠LAT_COMP_MS=%.2f 는 모델 1 step(%.1fms) 미만 → 보상 안 함\n",
                  lat_comp_ms, MDT*1e3);
    }
  } else {
    std::printf("[deploy] ⚠지연보상 **OFF**(LAT_COMP_MS=0) — 실측지연 8.39ms 가 보상되지 않는다.\n");
  }
  std::vector<double> u_prev(NU, 0.0);      // ★in-flight 토크. 롤아웃이 이걸 유지한다
  long lc_n = 0, lc_skip = 0; bool lc_warned = false;

  // ── ★중복 writer 차단 (2026-08-24) ──
  if(!mock && !simhw){
    auto others = find_other_writers();
    if(!others.empty()){
      std::printf("\n✗ **모터 명령 writer 가 이미 실행 중이다.** 둘이 뜨면 SHM 에 서로 다른 명령을\n"
                  "  번갈아 써서 관절이 진동한다(2026-08-10 실기 사고: +18° ↔ −20°).\n");
      for(auto& [pid, cl] : others) std::printf("    PID %d: %s\n", pid, cl.c_str());
      std::printf("  → 먼저 종료할 것:  kill");
      for(auto& [pid, cl] : others) std::printf(" %d", pid);
      std::printf("\n  (의도적으로 강행하려면 --force. 권장하지 않는다.)\n\n");
      if(!force_writer){ mj_deleteData(d); mj_deleteModel(m); return 4; }
      std::printf("  ⚠--force — 강행한다.\n");
    }
  }

  // ── 하드웨어 ──
  HwIface* hw = simhw ? (HwIface*)new SimHw(mjcf, &jm, &cfg, NCH, dt)
              : mock  ? (HwIface*)new MockHw(NCH, dt)
                      : (HwIface*)new ShmHw(cfg.lib, NCH);
  if(!hw->init(cfg.recv_wait_ms)){
    auto* sh = dynamic_cast<ShmHw*>(hw);
    std::printf("✗ 하드웨어 초기화 실패\n  %s\n", sh? sh->err.c_str() : "?");
    delete hw; mj_deleteData(d); mj_deleteModel(m); return 1;
  }
  std::printf("[deploy] backend=%s\n", hw->name());

  for(int s : {SIGINT,SIGTERM,SIGHUP,SIGQUIT}) std::signal(s, on_sig);

  // ★★IMU 생존 확인 — tilt E-stop 은 IMU 가 있어야만 동작한다.
  //   이 로봇은 SHM fIMUBuf 가 전부 0 인데 IsUpdatedIMU() 는 1 을 반환한다
  //   (emb/IMU_RECOVERY.md). 그러면 tilt ≡ 0 이라 임계 40° 에 **영원히 도달 못 하고**
  //   tilt E-stop 이 **완전히 무력**해진다. 조용히 넘어가면 보호장치가 있다고
  //   착각하게 되므로 기동 시 명시적으로 경고하고 상태에도 노출한다.
  //   ⚠"값이 0" 보다 "신선한 0" 이 더 위험하다 — freshness 검사로 안 걸러진다.
  bool imu_dead = false;
  { HwState s0; hw->read(s0);
    double a = std::fabs(s0.acc[0])+std::fabs(s0.acc[1])+std::fabs(s0.acc[2]);
    double r = std::fabs(s0.rpy[0])+std::fabs(s0.rpy[1])+std::fabs(s0.rpy[2]);
    // 정상 IMU 는 정지 중에도 가속도계에 중력 ~9.81 이 반드시 잡힌다. 그게 없으면 죽은 것.
    imu_dead = (a < 0.5 && r < 1e-9);
    if(imu_dead){
      std::fprintf(stderr,
        "\n%s\n"
        "!! ⚠⚠ IMU 가 전부 0 → **tilt E-stop 무력**(tilt≡0 이라 임계에 도달 불가).\n"
        "!!    남은 런타임 보호는 워치독 · 토크트립 · 속도트립 **뿐**이다.\n"
        "!!    stand/walk 는 자세 피드백이 필요하다 — IMU 없이 돌리는 것은 위험하다.\n"
        "!!    원인·조치: emb/IMU_RECOVERY.md\n%s\n\n",
        std::string(72,'!').c_str(), std::string(72,'!').c_str());
    } else {
      std::printf("[deploy] IMU 정상 — |acc|합 %.2f (중력 감지). tilt E-stop 유효.\n", a);
    }
  }
  // ★CoM 적분의 xy 축은 IMU 가 살아 있을 때만 의미가 있다 (2026-08-24).
  //   동체 자세를 모르면 cerr[xy] 가 근거 없는 값이라, 적분하면 "오른쪽으로 기움" 을 증폭한다.
  //   z(높이)는 엔코더만으로 관측되므로 IMU 와 무관하게 유효하다. 상세는 biped_control.hpp.
  c.imu_ok = !imu_dead;
  if(c.STAND_KI > 0.0)
    std::printf("[deploy] CoM 적분 STAND_KI=%.2f · 축=(%.0f,%.0f,%.0f)%s\n",
                c.STAND_KI, c.KI_AXIS[0], c.KI_AXIS[1], c.KI_AXIS[2],
                imu_dead ? "  ⚠IMU 사망 → **xy 강제 차단**(z 만 동작)" : "");

  // ── 상태 ──
  std::vector<float> q_ch(NCH), dq_ch(NCH), tau_ch(NCH), kp_ch(NCH), kd_ch(NCH), zero(NCH,0.f);
  // ★모드별 **실제로 내보낸 위치명령**. 발행해서 모니터가 "명령 vs 측정" 을 보여줄 수 있게 한다.
  //   종전엔 안 냈다 — stand 가 순수 토크(kp=0)라 명령각이 무의미했기 때문이다.
  //   그런데 STAND_KP_FLOOR 를 켜면서 **의미가 생겼다**(2026-08-20). 처짐을 보려면 필요하다.
  std::vector<float> qcmd_ch(NCH, 0.f);
  // ★실제로 내보낸 게인도 같이 기록한다. 종전엔 발행값이 **0.0 하드코딩**이었다 —
  //   그래서 모니터의 "≈Nm"(위치오차→토크 환산)도, "kp 걸렸는데 안 따라옴" 판정도
  //   전부 죽어 있었다. 실제로 HL_foot 이 명령 +100° 인데 채널 −0.28° 에 머물러도
  //   모니터는 아무 경고를 못 냈다(2026-08-20).
  std::vector<float> kpcmd_ch(NCH, 0.f), kdcmd_ch(NCH, 0.f);
  // ★★위치모드(home·hold) 게인 배율 (2026-08-20). 위치오차를 줄이려면 kp 를 올린다.
  //   ⚠**두 가지가 같이 움직인다**:
  //   ① 토크트립 — kp_ch 1 당 0.0175 Nm/deg 다(실측). kp 500 이면 8.75 Nm/deg 라
  //      **1.7° 오차에서 트립(15Nm)** 이 걸린다. 게인을 올릴수록 트립이 예민해진다.
  //   ② 감쇠비 ζ ∝ kd/√kp — kp 만 5배 올리면 ζ 가 **2.24배 떨어진다.**
  //      지금도 속도잡음(±3~7dps)이 kd 를 타고 틱틱거리는데, 그게 진동으로 커진다.
  //   ⇒ kd 를 **√배율**만큼 같이 올리는 것을 기본으로 한다(ζ 보존). 따로 주려면 POS_KD_SCALE.
  //   ⚠stand 는 이 배율을 안 쓴다 — 거기선 WBIC 와 싸우면 안 되고 STAND_KP_FLOOR 가 따로 있다.
  //   ★★2026-08-21 **런타임 조절**로 바꿨다(사용자 요청: GUI 에 강성 버튼).
  //   env 는 초기값일 뿐이고, 명령파일의 `pos_kp_scale` 이 오면 그쪽을 따른다.
  //   ⚠**갑자기 바꾸면 안 된다.** 하중을 받아 err 만큼 벌어진 상태에서 배율을 1→5 로
  //     계단으로 올리면 그 축의 토크가 **그 자리에서 5배**가 된다. 접지 중이면 τ_trip 이
  //     바로 걸린다. ⇒ POS_KP_RAMP_S(기본 1.0초) 동안 **선형으로** 옮긴다.
  //   ⚠stand/walk 는 이 배율을 안 쓴다 — 거기선 WBIC 와 싸우면 안 되고 STAND_KP_FLOOR 가 따로다.
  // ★2026-08-21 상한 5 → **10** (사용자: "5배는 해야 잘될 때가 있다" — 여유가 없었다).
  //   ⚠올릴 수 있는 근거: 트립은 **채널토크**로 걸리므로 관절토크 기준 트립은
  //     tau_trip×gear_k 다. 가장 예민한 calf 가 배율당 **7.16°** 이고 ×10 에서 0.72°.
  //     (GUI 가 종전에 4.77°/배율로 찍던 것은 tau_trip 을 관절토크로 본 오차다 — 같이 고쳤다.)
  //   ⚠정적 여유(트립각÷처짐)는 배율과 **무관**하다 — 둘 다 1/s 로 준다.
  //     줄어드는 것은 외란·하중이양·스틱슬립 같은 **과도에 대한 절대 여유**다.
  //     ×10 에서 calf 는 0.72° 과도에 트립한다. 그 위로 올리려면 여기를 다시 볼 것.
  const double KP_SCALE_MAX = getenv("POS_KP_SCALE_MAX") ? atof(getenv("POS_KP_SCALE_MAX")) : 10.0;
  const double KP_RAMP_S    = getenv("POS_KP_RAMP_S")    ? atof(getenv("POS_KP_RAMP_S"))    : 1.0;
  const double KD_SCALE_ENV = getenv("POS_KD_SCALE") ? atof(getenv("POS_KD_SCALE")) : -1.0;
  // ★★무중력(float) 모드 — 매달린 채 중력만 상쇄해 다리를 '무게 없이' 만든다 (2026-08-24).
  //   ⚠두 가지 목적이 겹쳐 있다:
  //     ① 기능 — 손으로 자세를 잡아 줄 수 있다(teach). 정비·캘리브레이션에 쓴다.
  //     ② ★진단 — **중립점이 곧 "제어기가 몇 % 모자라나" 다.** GRAV_SCALE 을 올리며
  //        다리가 뜨기 시작하는 g⁺, 내리며 지기 시작하는 g⁻ 를 잡으면 마찰이 소거된다:
  //            g* = (g⁺+g⁻)/2      ⇒   α·(G_CAD/G_real) = 1/g*
  //        g*=1.15 면 중력보상이 15% 모자란다는 뜻이고, 그게 그대로 stand 처짐의 크기다.
  //        α 인지 질량인지는 못 가리지만(둘 다 CAD 게이지), **제어기가 쓰는 게 G_CAD 라**
  //        "지금 얼마나 모자라나" 에는 정확히 답한다.
  double GRAV_SCALE = getenv("GRAV_SCALE") ? atof(getenv("GRAV_SCALE")) : 1.0;
  // ── ★push(발밀기) 모드 (2026-08-25) — 발밑 저울로 α 를 **외부 기준**으로 잰다 ──
  //   τ_cmd = g*축별·G_model(q) + Jᵀ·(0,0,−F)   (kp=0 · kd=FLOAT_KD)
  //   저울 눈금 = α × F_명령  (중력 몫은 이미 아는 g* 가 중화 → 질량/CoM 오차와 분리된 α)
  //   ★관절마찰(0.6~0.9 Nm ≈ 축당 2~3 N)이 있으므로 **한 점이 아니라 램프 왕복**으로 잰다:
  //     F 를 0→50→0 으로 훑고 올림/내림 저울값의 평균 — g* 브래킷과 같은 소거 원리다.
  //   ⚠calf·foot 토크 경로를 외부 기준으로 검증하는 **유일한** 방법이기도 하다
  //     (g* 는 그 두 축에서 중력≪마찰이라 원리적으로 불가였다).
  //   ⚠GRF 를 τ측정으로 역산하는 길은 이 하드웨어에 없다 — fCurrent=fTorque(명령 에코)라
  //     실측 토크가 존재하지 않는다. 저울이 유일한 진실이다.
  const double PUSH_RATE = getenv("PUSH_RATE") ? atof(getenv("PUSH_RATE")) : 5.0;   // [N/s]
  const double PUSH_MAX  = getenv("PUSH_MAX")  ? atof(getenv("PUSH_MAX"))  : 60.0;  // [N]
  const double PUSH_TAU_MAX = getenv("PUSH_TAU_MAX") ? atof(getenv("PUSH_TAU_MAX")) : 12.0; // [Nm/관절]
  double push_fz_cur = 0.0;
  // ★★hold 중력 전방보상 (2026-08-28) — "hold 로 서기".
  //   ⚠무엇이 문제였나: hold 는 **순수 위치 PD** 다(τ_ff 를 안 보낸다 — write_pos).
  //     중력을 오차로만 이긴다 ⇒ 정상상태 처짐 = τ_중력 /(α·kp_ch·gear_k).
  //     실측(08-28, 크레인 부분지지): hip 3.2° · thigh 0.6° · calf 2.2° · **foot 8.2°**.
  //     그래서 강성배율로 처짐을 줄이려 했고 — ×3·×5 에서 **HL_calf 가 주저앉았다**
  //     (오차 15~23° 로 벌어지는데 토크는 오히려 떨어짐 = 그 채널이 토크를 못 냄).
  //   ⇒ 게인을 올리는 대신 **필요한 토크를 처음부터 준다.** 이건 토크를 더 주는 게 아니다:
  //     중력이 요구하는 총량은 자세가 정하므로 고정이고, kp·오차 항에 있던 몫이 τ_ff 로
  //     **자리를 옮길 뿐**이다. 총 명령토크는 그대로고 **오차만 0 으로 간다.**
  //     ⇒ 트립 여유(15Nm)도 나빠지지 않는다. 이게 강성 상향과 결정적으로 다른 점이다.
  //   식:  τ = g*축별·G_model(q_meas) + Σ_다리 Jᵀ(발)·(0,0,−F) ,  kp·kd 는 그대로 유지
  //   ⚠**IMU 가 필요 없다** — float/push 와 똑같이 베이스를 단위 쿼터니언으로 고정한다.
  //     (2점 stand 는 WBIC 가 base 자세 **피드백**을 쓰므로 IMU 가 죽으면 못 선다.
  //      이건 피드백이 아니라 전방보상이라 그 의존이 없다. 대신 자세를 스스로 못 세운다 —
  //      "무너지지 않고 버티기" 까지가 목표고, 균형 회복은 stand 의 몫이다.)
  //   램프는 **지지율[%/s]** 이다 — 힘[N/s]이 아니다. 중력항까지 같은 비율로 올려야
  //   계단이 안 생기기 때문이다(아래 hold 분기 주석). 기본 10 %/s = 0→100% 에 10초.
  const double HOLD_FF_RATE = getenv("HOLD_FF_RATE") ? atof(getenv("HOLD_FF_RATE")) : 10.0;
  // ★12→14 (2026-09-04 사용자 확정). 실기에서 HL_foot ff 가 11.72 Nm — 12 클램프에
  //   붙어 있어 배분을 60 이상 밀면 소리 없이 잘렸다("올렸는데 안 변함"으로 보임).
  //   토크트립 15 Nm(채널) 은 그대로라 보호는 유지된다.
  const double HOLD_FF_TAU_MAX = getenv("HOLD_FF_TAU_MAX") ? atof(getenv("HOLD_FF_TAU_MAX")) : 14.0;
  //   적용점: 평발 2점 접지의 압력중심은 발바닥 어딘가다. 발끝에만 걸면 발목토크가
  //   과대평가되고, 발목원점(뒤꿈치 구)에만 걸면 발목토크가 0 이 되어 **정작 병목인
  //   foot 축을 하나도 안 도와준다.** 기본은 둘에 반반(=발바닥 중앙).
  const bool   HOLD_FF_MID = !(getenv("HOLD_FF_POINT") && std::string(getenv("HOLD_FF_POINT"))=="toe");
  //   한 다리 100% = mg/2. 모델에서 뽑는다(기체가 바뀌면 자동으로 따라간다).
  const double HOLD_FF_FULL_N = 0.5 * mj_getTotalmass(m) * 9.81;
  // ★foot 하중항 배율 (2026-09-02 실기 1차 반영). foot 은 g*(1.00)로 못 메운다 —
  //   결손이 상수+비율 혼합(r_foot(G)=0.77−0.36/G)이라 저부하(float)에선 상수가 지배해
  //   FOOT_COMP 몫이었지만, **체중 하중(G≈6~7Nm)에선 비율부 0.77 이 지배**한다.
  //   실기 1차: HL_foot 잔차 +2.8° ≈ 결손 20%×6Nm/(α·kp_raw 36.7) + 마찰 1° — 정량 일치.
  //   ⇒ foot 의 전방보상 전체에 1/0.77≈1.30 을 기본 적용(다른 축은 종전대로 g*).
  const double HOLD_FF_FOOT = getenv("HOLD_FF_FOOT") ? atof(getenv("HOLD_FF_FOOT")) : 1.30;
  double hold_fz_cur = 0.0, hold_ff_pct_cur = 0.0;
  double hold_split_cur = 50.0;            // 좌우 배분[% to HL] — 진입 시 50 으로 리셋·2%/s 램프
  std::printf("[deploy] hold 중력지지 — 모델 총질량 %.2f kg · **한 다리 100%% = %.1f N** ·"
              " 램프 %.1f %%/s · 적용점 %s\n"
              "         GUI [중력지지] 버튼(0~100%%)으로 올린다. 0%% 면 종전 hold(순수 위치 PD)와 같다.\n",
              mj_getTotalmass(m), HOLD_FF_FULL_N, HOLD_FF_RATE,
              HOLD_FF_MID ? "발바닥 중앙(발끝+뒤꿈치 반반)" : "발끝");
  //   ★★**축별 배율** (2026-08-24 실기에서 필요해졌다).
  //     실기 관측: HR 은 잘 맞는데 **HL_hip 이 특히 약하고** HL_thigh 가 한쪽으로 흐른다.
  //     무부하 실측이 이미 예측한 것이다 — ROTOR_I HL 7.652e-4 vs HR 7.121e-4(7.5%) ·
  //     강성 6.5~14.0 vs 10.3~15.2 · 마찰 0.711 vs 0.632. **네 지표가 같은 방향**이었다.
  //   ⇒ 공통 배율 하나로는 좌우를 **동시에 중립으로 못 만든다** — HL 을 맞추면 HR 이 뜬다.
  //     축별 g* 를 따로 잡아야 α 의 좌우 분포가 나온다.
  //   GRAV_SCALE_JOINT="1.15,1.10,1.0,1.0,1.0,1.0,1.0,1.0"  (관절 순서 8개)
  //     미지정 축은 GRAV_SCALE 을 쓴다. GUI 배율은 **전 축에 곱해지는 공통 계수**로 남는다.
  std::vector<double> grav_axis(NJ, -1.0);
  if(const char* ga=getenv("GRAV_SCALE_JOINT")){
    std::stringstream ss(ga); std::string t; int i=0;
    while(std::getline(ss,t,',') && i<NJ){ if(!t.empty()) grav_axis[i]=atof(t.c_str()); i++; }
    std::printf("[deploy] 축별 중력배율 GRAV_SCALE_JOINT =");
    for(int j=0;j<NJ;j++) std::printf("  %s %.3f", cfg.joints[j].name.c_str(), grav_axis[j]<0?GRAV_SCALE:grav_axis[j]);
    std::printf("\n");
  }
  //   kd 만 남긴다 — kp=kd=0 순수토크가 30~65Hz 에서 발산한 전례가 그대로 적용된다.
  //   ⚠크면 뻑뻑해서 무중력 느낌이 안 난다. 작으면 발산 위험. 0.30 에서 시작한다.
  // ★foot 상수결손 보상 (2026-08-27 무게추 브래킷 캠페인 확정: r_foot(G)=α−k/G ·
  //   k≈0.36 Nm(관절측) · 좌우 동일 — data/push/PLAN_0826.md 최종표).
  //   드라이브의 정지마찰성 결손이라 **속도가 아니라 명령토크 부호**로 보상한다
  //   (FRIC_COMP 는 속도기반 → 준정적 유지·저속 힘제어에서 0 이라 이 결손을 못 덮는다).
  //   채널좌표에서 더한다(k·tanh(τ_ch/τ0) · k_ch = k_joint/gear_k) — "g* 관절좌표 비가환"
  //   검토 발견과 같은 이유로, 드라이브별 손실 보정은 드라이브(채널) 단위가 정칙이다.
  //   기본 0(꺼짐). 실기 검증 예정값: FOOT_COMP_NM=0.36
  const double FOOT_COMP_NM = getenv("FOOT_COMP_NM") ? atof(getenv("FOOT_COMP_NM")) : 0.0;
  const double FOOT_COMP_T0 = getenv("FOOT_COMP_T0") ? atof(getenv("FOOT_COMP_T0")) : 0.30;
  auto foot_comp = [&](std::vector<float>& tch){
    if(FOOT_COMP_NM <= 0.0) return;
    for(int leg=0; leg<2; leg++){
      const auto& FJ = cfg.joints[leg*4+3];              // foot
      const double kc = FOOT_COMP_NM / FJ.gear_k;        // 관절 Nm → 채널
      const double t0 = std::max(1e-3, FOOT_COMP_T0 / FJ.gear_k);
      tch[FJ.channel] += (float)(kc * std::tanh((double)tch[FJ.channel] / t0));
    }
  };
  if(FOOT_COMP_NM > 0.0)
    std::printf("[deploy] ★foot 상수결손 보상 ON — k=%.2f Nm(관절측)·τ0=%.2f · r_foot(G) 법칙\n",
                FOOT_COMP_NM, FOOT_COMP_T0);
  // ★walk 묶음 (2026-08-27 설계 · sim 정량화 tools/walk_demand_check.py):
  //   실측 플랜트 1점 walk 요구 = calf 채널 |dq| max 610~673 dps · kd 감쇠토크 max 37~41 Nm.
  //   고정 트립(200 dps/15 Nm)은 스윙 자체를 즉살한다 → **walk 모드 한정** 상향, 이탈 시 자동 복원.
  //   (walk 는 kp≈0 순수토크(kpf=0)라 "트립각 = 오차×kp" 논리가 성립하지 않음 — 각 여유 무관)
  //   ⚠원안 "q̇_cmd 전송"은 기각: dq_ch 는 **측정**속도다(ch_to_dq_ctrl 역변환 — 명령이 아님).
  //     측정속도를 dq_des 로 보내면 err≡0 + 센서지연 = 음의 감쇠 — stand_ref 주석의 위치 목표와
  //     같은 함정. 목표 관절속도는 존재하지 않는다(토크 WBIC — 스윙 감쇠는 SW_KD 가 task 공간 담당).
  //     ⇒ 같은 목적(스윙 제동 제거)을 WALK_KD_FLOOR(kd 축소)로 이룬다: 41 Nm × 0.15 ≈ 6 Nm.
  //     지연 안정화용 감쇠가 어느 정도 필요한지는 실기에서만 판가름 — env 로 현장 조절.
  // ⚠env 는 무가드 atof 금지(08-27 검증 확정): 오타→0 이면 walk 진입 즉시 limp,
  //   "nan" 이면 > 비교가 전부 false 라 보호가 통째로 사라진다. 무효값은 기본값 폴백+경고.
  auto env_gd = [&](const char* k, double dv, double lo, double hi)->double{
    const char* e = getenv(k); if(!e) return dv;
    const double v = atof(e);
    if(!std::isfinite(v) || v < lo || v > hi){
      std::printf("[deploy] ⚠%s='%s' 무효(허용 %g~%g) — 기본 %g 사용\n", k, e, lo, hi, dv);
      return dv; }
    return v; };
  const double WALK_VEL_TRIP = env_gd("WALK_VEL_TRIP_DPS", 900.0, 300.0, 3000.0);
  const double WALK_TAU_TRIP = env_gd("WALK_TAU_TRIP_NM",   25.0,   5.0,   80.0);
  const double WALK_KD_FLOOR = env_gd("WALK_KD_FLOOR",       0.15,  0.0,    1.0);
  // ⚠숨은 결합(08-27 검증 확정): 토크트립은 kd 제동이 포함된 명령 에코를 본다 —
  //   calf kd_ch 3.5 × 스윙 673dps ⇒ 전량 41 Nm. WALK_KD_FLOOR > TAU_TRIP/41 이면
  //   **정상 스윙 제동만으로** walk 토크트립이 발화한다. 현장에서 kd 를 올릴 때 같이 볼 것.
  if(WALK_KD_FLOOR * 41.0 > 0.8 * WALK_TAU_TRIP)
    std::printf("[deploy] ⚠WALK_KD_FLOOR %.2f → 스윙 kd 제동 ≈%.0fNm — WALK_TAU_TRIP %.0fNm 근접/초과."
                " kd 를 올리면 트립도 같이 올릴 것 (한계비 %.2f)\n",
                WALK_KD_FLOOR, WALK_KD_FLOOR*41.0, WALK_TAU_TRIP, WALK_TAU_TRIP/41.0);
  std::printf("[deploy] walk 한정: 트립 %.0fdps/%.1fNm · kd×%.2f (타 모드 %.0fdps/%.1fNm·kd 유지)\n",
              WALK_VEL_TRIP, WALK_TAU_TRIP, WALK_KD_FLOOR, cfg.vel_trip_dps, cfg.tau_trip_nm);
  const double FLOAT_KD = getenv("FLOAT_KD") ? atof(getenv("FLOAT_KD")) : 0.30;
  //   ★**기본은 전 축이다.** 무중력은 다리 전체를 무게 없이 만드는 것이 목적이다 —
  //     축이 서로 커플링돼 있어(hip 이 처지면 thigh 의 중력이 바뀐다) 전 축을 같이 놓아야
  //     실제 상황이고, 한 축씩은 나머지가 잡아 주므로 인위적이다.
  //     그리고 τ_ff = G_model(q_meas) 를 **매 틱 다시 계산**하므로 모델을 통한 폐루프가 된다:
  //     α 가 모자라 처져도 접히면서 필요 중력이 줄어 **어딘가에서 평형을 찾는다**(무한낙하 아님).
  //   ★FLOAT_AXES="1,5" — **1회용 디버그**다. 그 채널만 뜨고 나머지는 진입 자세로 위치유지.
  //     쓰는 자리: 전 축을 놓기 전에 **부호를 확인**할 때. 어느 축의 sign 이 반대면 그 축만
  //     폭주하는데, 전 축을 한꺼번에 놓으면 어느 축이 원인인지 안 보인다.
  //     좌우 α 를 따로 재고 싶을 때도 쓴다(FLOAT_AXES=0 / =4 로 hip 좌우).
  std::set<int> float_axes;
  if(const char* fa=getenv("FLOAT_AXES")){
    std::stringstream ss(fa); std::string t;
    while(std::getline(ss,t,',')) if(!t.empty()) float_axes.insert(atoi(t.c_str()));
  }
  std::vector<float> cfg_kp_ch(NCH,0.f), cfg_kd_ch(NCH,0.f);   // 배율 없는 설정 게인(유지축용)
  jm.kp_ch(cfg_kp_ch.data(), 1.0); jm.kd_ch(cfg_kd_ch.data(), 1.0);
  //   ★★**stand/walk 토크 보정** (2026-08-24). 무중력에서 **측정한** 부족분을 넣는다.
  //   무중력 중립점이 g* 면  α·(G_CAD/G_real) = 1/g*  이고, 그건 곧
  //   "제어기가 요구한 토크의 1/g* 배만 실제로 나온다" 는 뜻이다.
  //   ⇒ τ 를 **g\* 배**로 명령하면 실제 출력이 모델이 의도한 값이 된다.
  //   실측(2026-08-24, 매달린 채): HR_hip g* = 1.125·1.167 → **1/g* = 0.873**
  //     등속 스윕 hip 0.867~0.919 · 순수토크 프로브 0.874/0.904 와 겹친다(셋이 0.87).
  //   ⚠**적분(STAND_KI)보다 낫다** — 적분은 모르는 오차를 더듬어 찾지만 이건 잰 값을 넣는다.
  //     그리고 적분은 IMU 가 죽어 xy 를 못 쓰는데, 이건 그 제약이 없다.
  //   ⚠**기본 1.0 = 꺼짐.** 접지 상태에서 토크를 올리는 것이라 크레인을 남긴 채 켤 것.
  //   ⚠이게 α 인지 CAD 질량오차인지는 **아직 안 갈렸다**(다리 링크 저울이 가른다).
  //     어느 쪽이든 "제어기가 모자란 만큼" 을 메우는 것이라 지금 쓰기에는 문제없다.
  double STAND_TAU_SCALE = getenv("STAND_TAU_SCALE") ? atof(getenv("STAND_TAU_SCALE")) : 1.0;
  std::vector<double> tau_axis(NJ, -1.0);            // 축별. <0 = 공통값 사용
  if(const char* ta = getenv("STAND_TAU_SCALE_JOINT")){
    std::stringstream ss(ta); std::string t; int i = 0;
    while(std::getline(ss, t, ',') && i < NJ){ if(!t.empty()) tau_axis[i] = atof(t.c_str()); i++; }
  }
  if(STAND_TAU_SCALE != 1.0 || tau_axis[0] > 0){
    std::printf("[deploy] ★stand 토크보정 ON —");
    for(int j = 0; j < NJ; j++)
      std::printf("  %s %.3f", cfg.joints[j].name.c_str(),
                  tau_axis[j] > 0 ? tau_axis[j] : STAND_TAU_SCALE);
    std::printf("\n"
                "         ⚠접지 상태에서 토크가 그만큼 올라간다 — 크레인을 남긴 채 켤 것.\n");
  }
  // ══ 액추에이터 토크스케일 α 보정 (벤치 실측 2026-09-12) ══════════════════════════════
  //   드라이버가 명령토크의 α(0.834) 만 실현한다(α·τ_cmd). FF 를 1/α 로 올려 **실현 = 의도**.
  //   ★전 모드(hold/float/push/stand/walk) 의 write_mit 에 일괄 적용(write_ff).
  //   ⚠벤치는 bare 7:1(thigh 기저). 벨트 있는 축(calf)은 실효 α 가 더 낮을 수 있어
  //     ACT_ALPHA_JOINT="a0,a1,..,a7"(축순) 로 축별 지정 가능. ⚠접지 토크 ~+20% —
  //     크레인 남긴 채 hold/stand 로 검증 후 walk. 끄려면 ACT_ALPHA=1.
  //   sim 파리티: biped_wbic.py 가 actuator_gear ×α(0.834)로 "약한 로봇" 재현.
  const double ACT_ALPHA = getenv("ACT_ALPHA") ? atof(getenv("ACT_ALPHA")) : 0.834;
  std::vector<double> alpha_ax(NJ, -1.0);
  if(const char* aa = getenv("ACT_ALPHA_JOINT")){
    std::stringstream ss(aa); std::string t; int i=0;
    while(std::getline(ss, t, ',') && i < NJ){ if(!t.empty()) alpha_ax[i]=atof(t.c_str()); i++; }
  }
  std::vector<float> ff_comp(NCH, 1.0f);
  for(int j=0;j<NJ;j++){
    const double al = alpha_ax[j] > 0 ? alpha_ax[j] : ACT_ALPHA;
    ff_comp[cfg.joints[j].channel] = (al > 0.1) ? (float)(1.0/al) : 1.0f;
  }
  std::vector<float> tau_ff_out(NCH, 0.f);
  // FF 토크에 1/α 를 곱해 실기 실현토크를 모델값으로 되돌린다. 전 write_mit 이 이걸 쓴다.
  auto write_ff = [&](const float* qd, const std::vector<float>& tau,
                      const float* kpd, const float* kdd){
    for(int i=0;i<NCH;i++) tau_ff_out[i] = tau[i] * ff_comp[i];
    return hw->write_mit(qd, zero.data(), tau_ff_out.data(), kpd, kdd, NCH);
  };
  if(ACT_ALPHA != 1.0 || alpha_ax[0] > 0){
    std::printf("[deploy] ★액추에이터 α 보정 ON — FF ×1/α (α=%.3f → ×%.3f · ~+%.0f%% 토크).\n"
                "         ⚠접지 토크 그만큼 오른다 — 크레인 남긴 채 hold/stand 검증 후 walk. 끄려면 ACT_ALPHA=1.\n",
                ACT_ALPHA, 1.0/ACT_ALPHA, (1.0/ACT_ALPHA-1.0)*100.0);
  }
  double kp_scale_tgt = getenv("POS_KP_SCALE") ? atof(getenv("POS_KP_SCALE")) : 1.0;
  kp_scale_tgt = std::max(0.0, std::min(KP_SCALE_MAX, kp_scale_tgt));
  double POS_KP = kp_scale_tgt;                    // ★램프 중인 **현재** 배율
  // ★kd 배율 — 우선순위: 명령파일(pos_kd_scale) > env(POS_KD_SCALE) > 자동(√kp, ζ 보존).
  //   자동이 기본인 이유는 ζ ∝ kd/√kp 라 kp 만 올리면 저감쇠 진동이 되기 때문이다.
  //   ⚠그런데 kd 는 **속도잡음을 그대로 토크로 증폭**한다(τ_ripple = kd·dq_noise).
  //     정지 중 잡음이 ±7dps 면 ×10 에서 hip 이 2.3Nm — 트립 15Nm 의 15% 다.
  //     잡음이 지배하는 기체에서는 ζ 를 좀 포기하고 kd 를 낮추는 편이 낫다 ⇒ 조절 가능하게 둔다.
  double kd_scale_cmd = -1.0;                      // 명령파일이 지정한 값(-1 = 미지정)
  double POS_KD = (KD_SCALE_ENV>=0) ? KD_SCALE_ENV : std::sqrt(std::max(1e-9, POS_KP));
  // ★축별 트립각[deg] — **가장 예민한 축**을 찍는다.
  //   ⚠**gear_k 는 1승이다** (2026-08-21 정정. 종전 2승이라 calf 를 4.77°/배율로 찍었다).
  //     트립은 **채널토크**로 걸린다(아래 tau_trip 비교가 hs.tau_nm 을 그대로 본다)
  //     ⇒ 트립 시점의 raw 토크 = tau_trip·gear_k, raw 강성 = kp_ch·gear_k² 이므로
  //         트립각 = (tau_trip·gear_k)/(kp_ch·gear_k²·배율) = tau_trip/(kp_ch·gear_k·배율)
  //     config(biped_emb.yaml "트립각 계산은 1승이다")·GUI(teleop_gui_biped.py)는 이미
  //     1승이었다. **여기만 안 고쳐져 있었다** — 위 :3xx 주석이 "GUI 도 같이 고쳤다" 고
  //     적어 두고 정작 이 코드는 2승인 채였다(주석만 고친 상태).
  //   ⚠kp_ch·gear_k² 는 **raw 좌표** 강성이다(모델각 아님). 모델각 강성은 발목 커플링
  //     때문에 축별 스칼라로 못 쓴다 — 비대각이 있는 행렬이 된다. emb/README 참조.
  auto trip_deg = [&](double sc)->std::pair<std::string,double>{
    std::string who="-"; double best=1e30;
    for(const auto& j : cfg.joints){
      const double nmd = j.kp*j.gear_k*M_PI/180.0*std::max(1e-9,sc);   // ★1승
      if(cfg.tau_trip_nm/nmd < best){ best = cfg.tau_trip_nm/nmd; who = j.name; }
    }
    return {who,best}; };
  {
    auto t1 = trip_deg(POS_KP);
    std::printf("[deploy] 위치게인 배율 kp×%.2f · kd×%.2f (GUI 로 조절 · 최대 ×%.0f · 램프 %.1fs)\n"
                "         ⚠가장 예민한 축 %s — **%.2f° 에서 토크트립**(τ_trip %.0fNm)\n",
                POS_KP, POS_KD, KP_SCALE_MAX, KP_RAMP_S, t1.first.c_str(), t1.second, cfg.tau_trip_nm);
  }
  std::vector<double> q_ctrl(NJ), dq_ctrl(NJ), tau_ctrl(NU);
  // ★토크 통계 누적기 — 루프율로 쌓고 발행마다 비운다(위 ① 주석 참조).
  std::vector<double> ts_sum(jm.n_leg,0.0), ts_sq(jm.n_leg,0.0),
                      ts_min(jm.n_leg, 1e300), ts_max(jm.n_leg,-1e300);
  long ts_n = 0;
  auto ts_reset = [&]{
    std::fill(ts_sum.begin(),ts_sum.end(),0.0); std::fill(ts_sq.begin(),ts_sq.end(),0.0);
    std::fill(ts_min.begin(),ts_min.end(), 1e300);
    std::fill(ts_max.begin(),ts_max.end(),-1e300);
    ts_n=0;
  };
  // ★부팅 초기값을 **0 벡터로 두지 않는다** (2026-08-24). Python 은 기동 시 측정각으로
  //   초기화한다(biped_emb.py:333 `hold_leg = jm.ch_to_q_joint(_raw0.q_deg)`) — C++ 만 0 이었다.
  //   첫 유효 상태를 받는 즉시 측정각으로 덮는다(아래 have_state 지점). 그 전에 쓰일 일은
  //   없지만, 0 벡터가 남아 있으면 어떤 경로로든 전 축 0° 계단이 나갈 수 있다.
  std::vector<float> hold_ch(NCH, 0.f);
  bool hold_ch_primed = false;
  HwState hs;
  // ★home 램프 상태 (2026-08-20 신설). stand 자세로 **속도제한을 걸어** 이동한다.
  //   왜 필요한가: `stand` 는 WBIC posture task 라 램프가 없다. 0° 자세에서 바로 누르면
  //   발목이 채널각 **100.4°** 를 한꺼번에 요구한다(커플링·gear_k 1.2 때문에 모델 −59.8°
  //   가 채널 100.4° 가 된다). 속도트립 200dps 에 그냥 걸린다.
  //   ⇒ home 으로 먼저 그 자세까지 S-curve 로 간 뒤 hold→접지→stand 순서로 간다.
  std::vector<float> home_from(NCH,0.f), home_to(NCH,0.f);
  double home_t0=0, home_T=0; bool home_done=false, home_warned=false;
  // ── 채널 이름(로그용). cfg.joints 에서 channel→name 을 만든다. 없으면 "ch%d".
  std::vector<std::string> chname(NCH);
  for(int i=0;i<NCH;i++) chname[i] = "ch"+std::to_string(i);
  for(const auto& j : cfg.joints) if(j.channel>=0 && j.channel<NCH) chname[j.channel]=j.name;
  // ★★자세유지 토크 비교 (2026-08-21, 사용자 요청).
  //   묻고 있는 것: **같은 자세를 hold(위치제어)로 버틸 때와 stand(WBIC 토크)로 버틸 때
  //   각 축이 내는 토크가 얼마나 다른가.** 둘이 크게 다르면 WBIC 의 모델(질량·중력·접촉
  //   분배)이 실제와 어긋난다는 뜻이고, 그 차이가 곧 stand 가 처지는 이유다.
  //   ⚠성립 조건: **두 모드의 목표자세가 같아야 한다.** cmode=1 이면 home 도 stand 도
  //     Qflat8 이라 성립한다. 1점 점발(cmode=0)에서 home 이 설정 0° 로 가면 stand 목표
  //     (Qhome8)와 달라 비교가 무의미하다 — 그때는 비교표에 경고를 붙인다.
  std::vector<double> tau_acc(NCH,0.0); int tau_n=0; double tau_win_t0=0;
  std::vector<double> tau_avg(NCH,0.0);                 // 최근 창의 평균 토크
  std::vector<double> tau_hold(NCH,0.0), tau_stand(NCH,0.0);
  std::vector<double> q_hold(NCH,0.0),   q_stand(NCH,0.0);
  bool have_tau_hold=false, have_tau_stand=false;
  double mode_t0=0;                                     // 현재 모드 진입시각
  // ★stand 폭주 가드 (2026-09-03 실기 — lite 2차에서 발이 14° 폭주하는 동안 **아무
  //   가드도 안 걸렸다**: 속도가 느려 속도트립 미달, 토크도 15Nm 아래로 스침).
  //   stand 는 기준자세(stand_ref)가 고정이므로 "기준 대비 큰 이탈 지속" 자체가 이상이다.
  //   발동 시 limp 가 아니라 **hold 로 강하**한다 — hold 는 자립 실증된 안전 상태고,
  //   지지율(hold_fz_cur)을 리셋하지 않아 전방보상이 연속으로 이어진다(주저앉음 방지).
  const double STAND_RUNAWAY_DEG = getenv("STAND_RUNAWAY_DEG") ? atof(getenv("STAND_RUNAWAY_DEG")) : 12.0;
  double runaway_t0 = -1.0; bool stand_runaway = false;
  // ★jog 램프 상태 (2026-08-21). **등속 램프**다 — home 의 S-curve 를 쓰지 않는다.
  //   S-curve 는 "A→B 1회 이동" 전제인데 jog 목표는 슬라이더를 끄는 동안 **매 틱 바뀐다.**
  //   그래서 살아 있는 목표를 속도클램프로 따라간다(biped_emb.py control/jog.py 와 같은 방식).
  //   ⚠등속 램프는 출발·도착에서 가속도가 불연속이다(config 주석). jog 는 축 하나씩 천천히
  //     쓰는 검증용이라 실무상 문제가 안 됐지만, 알고 쓰는 것과 모르고 쓰는 것은 다르다.
  // ★안전 종료(soft_off) 상태 — 2026-08-28 사용자 요청.
  //   ⚠`off` 는 **비상 탈출구로 그대로 둔다**(즉시 무여자). 안전 종료는 별도 모드다 —
  //     넘어지는 중에 "천천히" 내려가면 그게 더 위험하기 때문이다.
  //   방식: 현재 자세 → **안전 자세**로 위치제어 서행(기본 20 dps) → 정착 → off.
  //   ⚠게인/배율 램프다운은 쓰지 않는다: 중력은 상수인데 붙잡는 힘만 줄이면 말미에
  //     반드시 "버티다 놓침"이 된다(float_gstar v1·v2 가 그렇게 실패했다).
  //     검증된 방식은 **낙차가 0 인 자세로 옮긴 뒤 끄는 것**이다(v3, 실기 확인).
  std::vector<float> soft_from(NCH,0.f), soft_to(NCH,0.f);
  double soft_t0=0, soft_T=1.0; bool soft_done=false;
  bool soft_latched=false;   // ★완료 래치 — GUI 20ms 재발행이 드라이브를 되살리던 구멍
  std::vector<float> boot_pose(NCH, 0.f); bool boot_pose_set=false;
  std::vector<double> jog_q(NJ, 0.0);        // 램프 중인 명령(**모델각 deg**)
  double jog_prev_t = 0; bool jog_init = false;
  bool ground_refused=false;
  bool push_refused = false;             // ★push 거부 래치 (2026-08-27 검증 반박 반영)
  // ★토크 불응답(사일런트 축사망) 감지 상태 (2026-08-28 실기)
  std::vector<double> unresp_t0(NCH, -1.0);      // 조건 시작 시각(채널별)
  std::vector<char>   unresp(NCH, 0);            // 래치
  double unresp_chk_t = 0.0;      // ★접지 가드 거부 래치(로그 폭주 방지)
  bool still_warned=false;        // ★정지 확인 거부 래치
  // ★★무장 직후 트레이스 (2026-08-20). hold 진입 즉시 속도트립이 반복되는데
  //   원인 후보(게인 점프 / 낡은 측정값 / 부호 / 한쪽 다리)를 말로 가릴 수 없다.
  //   무장 순간부터 0.5초를 **매 틱** CSV 로 남긴다. 트립이 나도 파일은 남는다.
  FILE* trc=nullptr; double trc_t0=0;
  bool have_state=false; int ok_reads=0; double live_t0=0;
  std::string boot_mode="off"; bool mode_locked=false;   // ★기동 시 잔여명령 잠금   // ★센서 준비·생존 확인
  // ★★EtherCAT 동결 감지 (2026-08-20 실기). Emb 는 OP 를 잃어도 프로세스가 계속 돌고
  //   **마지막 버퍼를 재발행하며 갱신 플래그까지 1 로 세운다**(memory: emb-ethercat-freeze).
  //   그래서 health=ok · n_ok=8 · n_fault=0 인데 값만 얼어붙는다 — 침묵 실패다.
  //   실측(2026-08-20): **왼다리 4축만** 동결. dq 가 227dps 로 굳어 있는데 각도는 불변이라
  //   정지가드가 "움직이는 중" 으로 오판했다. 원인을 엉뚱하게 짚게 만든다.
  //   ⇒ 채널별로 (q,dq,tau) 가 **한 번도 안 바뀐 시간**을 세어 직접 이름 붙인다.
  std::vector<float> prv_q(NCH,1e9f), prv_dq(NCH,0.f), prv_t(NCH,0.f);
  std::vector<double> frz_t(NCH,0.0);  bool frz_warned=false; int frozen_now=0;
  // ★★동결 증거 수집기 (2026-08-24). 여덟 번 넘게 났는데 원인을 모르는 이유는
  //   **증거가 남지 않아서**다 — 배너는 스크롤백에만 찍히고 재기동에 사라진다.
  //   ⇒ 직전 N 초를 링버퍼에 계속 굴리고, 동결 순간 CSV+사건로그로 떨어뜨린다.
  //   FREEZE_PRE_S=0 으로 끌 수 있다(메모리 ~2.4MB/3초).
  Ring frz_ring;
  const double FRZ_PRE_S = getenv("FREEZE_PRE_S") ? atof(getenv("FREEZE_PRE_S")) : 3.0;
  if(FRZ_PRE_S > 0) frz_ring.init(FRZ_PRE_S, dt, NCH);
  const std::string FRZ_LOG = getenv("FREEZE_LOG") ? getenv("FREEZE_LOG")
                                                   : "/tmp/biped_freeze_log.tsv";
  int emb_pid = EmbSnap::find_pid();
  NicSnap nic0 = NicSnap::take();          // 기동 시점 기준선 — CRC 증가분을 보려면 필요하다
  EmbSnap emb0 = EmbSnap::take(emb_pid);
  double emb0_t = now_s();
  int frz_event = 0;                        // 이 세션의 동결 횟수(파일명에 쓴다)
  std::printf("[deploy] 동결 증거수집 ON — 직전 %.1fs 링버퍼 · 사건로그 %s\n"
              "         Emb pid=%s · NIC %s carrier=%lld rx_crc=%lld\n",
              FRZ_PRE_S, FRZ_LOG.c_str(),
              emb_pid>0 ? std::to_string(emb_pid).c_str() : "**못 찾음**",
              nic0.ifname.c_str(), nic0.carrier, nic0.rx_crc);
  // ★★stand 진입 블렌드 (2026-08-20 실기). hold→stand 는 위치제어(kp 100/50/80/30)에서
  //   **kp=kd=0 순수토크**로 한 틱에 바뀐다. 그 순간 WBIC 토크가 조금만 모자라도
  //   그대로 주저앉는다(실기 관측). 시뮬은 α=1 이라 안 드러난다 — 실기는 토크 스케일이
  //   ±10% 미검증이고 마찰도 있다.
  //   ⇒ 위치게인을 내리면서 WBIC 토크를 올린다. MIT 모드는 둘을 동시에 받으므로
  //     블렌드 중에는 위치제어가 받쳐 주고, 끝나면 순수토크가 된다.
  double stand_t0=0, stand_T=0; std::vector<float> stand_hold(NCH,0.f), stand_to(NCH,0.f), stand_ref(NCH,0.f);
  std::string mode = "off", prev_mode = "off", last_raw;
  bool estop = false, wd_tripped = false;
  double tau_over_t0 = -1, vel_over_t0 = -1, last_cmd_t = now_s(), last_pub = 0, hz_ema = cfg.ctrl_hz;
  Cmd cmd; double body_h = 0.48;   // ★0.50→0.48 (08-28 높이 스윕 — GUI H_DEF_1PT 와 동기)
  const double watchdog_s = cfg.watchdog_ms/1000.0;

  hw->enable(0);
  std::printf("[deploy] 모드: off/hold/**home**/**jog**/**float**/**push**/stand/walk. GUI 로 조종(%s).\n"
              "[deploy] float = **무중력**(중력보상 ×%.2f · kd×%.2f) — **매달린 채만**. 접지 중이면 거부.\n"
              "[deploy] home = %s 자세로 %.0fdps S-curve 이동(램프 %.1fs 설정).\n"
              "[deploy] jog  = 축별 목표각 추종 · %.0fdps 등속 램프 · 관절한계 클램프.\n",
              cmd_p.c_str(), GRAV_SCALE, FLOAT_KD, c.cmode==1?"2점 평발 stand":"1점 점발",
              cfg.home_speed_dps, cfg.home_min_time_s, cfg.jog_speed_dps);
  // ★위 두 값은 **실제 파싱된 것**을 찍는다 — config 가 안 읽혀도 기본값으로 조용히
  //   도는 걸 막는다. 2026-08-21 까지 실제로 그랬다(키가 safety 분기에 있어 무시됐다).
  std::printf("[deploy] ⚠ 모터 명령 writer 는 한 번에 하나만 — biped_emb.py 와 동시 실행 금지.\n");

  // ★★기동 시 명령파일에 남아 있는 모드를 **그대로 실행하지 않는다** (2026-08-20).
  //   GUI 는 마지막 상태를 파일에 남긴다. 실제로 seq 14871 짜리 `"mode":"stand"` 가
  //   남아 있었고, biped_deploy 는 기동 즉시 그걸 받아 무장했다 —
  //   매달린 채 stand → 접지가드가 hold 로 되돌림 → 흔들리는 로봇에 hold → 속도트립.
  //   기동할 때마다 이 고리를 돌았다.
  //   ⇒ 시작 시점의 내용을 **이미 본 것**으로 기록해 두고, 운전자가 버튼을 눌러
  //     내용이 **바뀌어야** 받는다. biped_emb.py 의 --start-mode off 와 같은 의도다.
  { Cmd c0; if(read_cmd(cmd_p, c0)){ last_raw = c0.raw;
      // ⚠내용 비교만으로는 못 막는다 — GUI 가 seq 를 올리며 **주기적으로 재발행**하므로
      //   같은 stand 도 매번 "새 명령" 이 된다(실측: 그대로 무장했다).
      //   ⇒ **모드**를 잠근다. 운전자가 다른 모드를 한 번 고르기 전까지 무시한다.
      boot_mode = c0.mode; mode_locked = (c0.mode != "off");
      if(c0.mode != "off")
        std::printf("[deploy] ⚠명령파일에 **%s** 가 남아 있다 — 무시하고 off 로 시작한다.\n"
                    "         GUI 에서 버튼을 다시 눌러야 반영된다(기동 즉시 무장 방지).\n",
                    c0.mode.c_str()); } }
  double t0 = now_s(), prev_loop = t0; long long k = 0; bool overrun_warned=false;
  mode_t0 = t0; tau_win_t0 = t0;   // ★0 으로 두면 "모드 2274초째" 같은 값이 나온다(절대시각)

  int rc = 0;
  while(!g_stop && (now_s()-t0) < T){
    double lt = now_s();

    // ① 센서
    hw->read(hs);
    if(!boot_pose_set && hs.mask){ boot_pose = hs.q_deg; boot_pose_set = true; }
    jm.ch_to_q_ctrl(hs.q_deg.data(),  q_ctrl.data());
    jm.ch_to_dq_ctrl(hs.dq_dps.data(), dq_ctrl.data());
    // ★★2026-08-20 **토크 통계는 루프율(500Hz)에서 낸다** — 발행(20Hz)에서 내면 안 된다.
    //   상태는 20Hz 로 나가므로 나이퀴스트가 10Hz 다. 그런데 이 로봇이 발산한 대역은
    //   **30~65Hz** 였다(a117c44: |dq| 196→322 dps). 모니터가 받은 표본으로 표준편차를
    //   계산하면 그 대역이 통째로 접혀 **리플을 과소평가**한다 — 정작 봐야 할 것을 못 본다.
    //   ⇒ 여기서 창(발행주기)마다 누적하고 std·min·max 를 실어 보낸다. 러닝 합이라
    //     비용은 무시할 수준이고, 모니터는 그리기만 하면 된다.
    //   ★min/max 를 같이 내는 이유: 이번 발산은 **첨두** 현상이라 평균·표준편차로는
    //     안 잡힌다. 창 안의 극값이 남아야 한다.
    {
      std::vector<double> tnow(jm.n_leg);
      jm.ch_to_tau_joint(hs.tau_nm.data(), tnow.data());
      for(int i=0;i<jm.n_leg;i++){
        const double v = tnow[i];
        ts_sum[i]+=v; ts_sq[i]+=v*v;
        if(v<ts_min[i]) ts_min[i]=v;
        if(v>ts_max[i]) ts_max[i]=v;
      }
      ts_n++;
    }
    const double D2R = JointMap::D2R;
    const double u = cfg.imu_deg ? D2R : 1.0;
    // ★IMU 프레임 정합 (2026-09-04) — EBIMU 오일러(ARPY)는 NED(전방·우·아래)라
    //   우리 베이스(ENU: 전방·좌·위)로 imu_euler_flip=[1,-1,-1](pitch·yaw 반전)한다.
    //   부호는 imu_peek 로 실측 확정. tilt(hypot)는 부호 무관이라 E-stop 영향 없음.
    //   ⚠자이로는 원시(4-3,Z-up)가 이미 베이스 정렬 → flip 안 함(자세와 같은 프레임).
    //   ⚠offset(imu_offset_m)은 강체라 자세·자이로엔 무영향 — 가속도 선형추정에만
    //     필요한데 추정기는 quat+gyro 만 쓰므로 지금은 config 에 저장만(보정 미적용).
    const auto& fl = cfg.imu_euler_flip;
    const double f0=(fl.size()>=3?fl[0]:1.0), f1=(fl.size()>=3?fl[1]:-1.0), f2=(fl.size()>=3?fl[2]:-1.0);
    double rpy[3] = { hs.rpy[0]*f0*u, hs.rpy[1]*f1*u, hs.rpy[2]*f2*u };
    double gyro[3]= { hs.gyr[0]*u, hs.gyr[1]*u, hs.gyr[2]*u };
    double quat[4]; rpy_to_quat(rpy[0],rpy[1],rpy[2],quat);
    double tilt = std::hypot(rpy[0],rpy[1]) * JointMap::R2D;

    // ★★**센서가 채워지기 전에는 명령을 받지 않는다** (2026-08-20, 트립 5회의 진짜 원인).
    //   기동 직후 첫 틱에 SHM 읽기가 아직 0 이었는데, 명령파일에 이전 hold/home 이
    //   남아 있어 **그 0 을 측정각으로 믿고** 래치했다. 그러면 명령이 0° 가 되어
    //   실제 ±25~85° 에서 0 으로 내리꽂는다 — 트레이스 실측:
    //       ch2 q 25.88 · 명령 0.00 · τ −18.2 Nm · dq −340dps
    //       ch6 q −85.12 · 명령 0.00 · τ +18.5 Nm · dq +460dps
    //   "최대이동 100.4°"(=0°에서 Qflat8 까지) 가 같은 사실을 가리키고 있었다.
    //   ⇒ 유효한 읽기가 몇 틱 쌓이기 전에는 off 를 유지한다.
    // ★동결 판정 — 값이 하나도 안 바뀌면 그 채널의 정지시간을 누적한다.
    for(int i=0;i<NCH;i++){
      if(hs.q_deg[i]==prv_q[i] && hs.dq_dps[i]==prv_dq[i] && hs.tau_nm[i]==prv_t[i]) frz_t[i]+=dt;
      else frz_t[i]=0.0;
      prv_q[i]=hs.q_deg[i]; prv_dq[i]=hs.dq_dps[i]; prv_t[i]=hs.tau_nm[i];
    }
    // ★링버퍼에 이번 틱을 밀어 넣는다. 명령각은 **직전 틱**의 발행값이다(디스패치가 아래라서).
    //   1틱(2ms) 어긋나는데, 동결 원인 상관에는 무관하다 — 대신 항상 채워져 있는 게 중요하다.
    if(!frz_ring.buf.empty())
      frz_ring.push(lt, hs.q_deg.data(), hs.dq_dps.data(), hs.tau_nm.data(),
                    hs.cur_a.data(), qcmd_ch.data());
    // ⚠**mock 은 제외한다** (2026-08-21). MockHw 는 명령이 없으면 값이 안 변하는 게 정상이라
    //   전 채널이 동결로 잡히고 **무장이 막힌다** — `--mock` 으로는 off 말고 아무 모드도
    //   못 들어가서 배포경로를 오프로봇으로 검증할 수 없었다.
    //   기동 생존확인(아래)은 이미 `!mock` 로 제외돼 있었다. 이쪽만 빠져 있어 짝이 안 맞았다.
    //   ★FREEZE_TEST=1 은 mock 에서도 이 검사를 켠다 — **증거수집 경로를 오프로봇에서
    //     검증하기 위한 것**이다(MockHw 는 값이 안 변하므로 즉시 발화한다).
    { std::string fz; int nfz=0;
      const bool fz_en = !mock || (getenv("FREEZE_TEST") && atoi(getenv("FREEZE_TEST")));
      for(int i=0;i<NCH;i++) if(fz_en && cfg.installed_has(i) && frz_t[i]>0.5){
        char b[16]; std::snprintf(b,sizeof b," ch%d",i); fz+=b; nfz++; }
      if(nfz && !frz_warned){ frz_warned=true;
        // ★몇 축이 얼었는지로 **어느 구간이 끊겼는지** 가른다 (2026-08-20).
        //   EtherCAT 슬레이브는 8축 **하나**(LAN9252)다. 그게 끊기면 8축이 **전부** 언다.
        //   4축만 얼면 MCU 는 살아 있고 그 아래 **FDCAN(해당 다리)** 이 끊긴 것이다 —
        //   MCU 가 마지막 값을 EtherCAT 으로 계속 올리므로 갱신 플래그·health 는 정상으로 보인다.
        const bool all_ch = (nfz >= (int)cfg.joints.size());
        // ★★판정 순서 정정 (2026-08-28 실기): 종전엔 **채널 수만 보고** 판정문을 먼저 찍고
        //   물리링크 증거는 그 아래에 붙였다. 그 결과 "4축만 얼었으니 EtherCAT 이 아니다" 라고
        //   단정한 배너 바로 밑에 **carrier 0(링크 끊김)** 이 찍히는 모순이 실제로 나왔다.
        //   ⇒ **carrier 를 먼저 읽고, 0 이면 그것이 최종 판정이다.** 링크가 끊기면 Emb 은
        //   마지막 버퍼를 계속 재발행하므로(commEtherCATm.cpp:520) 어느 채널이 먼저 '정지'로
        //   잡히는지는 타이밍 문제일 뿐이라, 채널 수로 구간을 가르면 틀린다.
        //   ⚠토폴로지: EMB ―EtherCAT→ transfer ―SPI→ MCU ―FDCAN→ MD80 (2026-08-28 정정)
        const NicSnap n_pre = NicSnap::take();
        const bool link_down = (n_pre.carrier == 0);
        std::fprintf(stderr,
          "\n%s\n!! ⛔⛔ **통신 동결** — 다음 채널이 0.5초 넘게 값이 하나도 안 바뀐다:%s\n"
          "!!   %s\n"
          "!!   ⚠health=ok · n_fault=0 으로 보인다 — **믿으면 안 된다.**\n"
          "!!   복구: %s%s\n%s\n\n",
          std::string(72,'!').c_str(), fz.c_str(),
          link_down
            ? "★**EtherCAT 물리 링크 단절**(eth0 carrier=0) — EMB↔transfer 구간이다.\n"
              "!!   ⚠얼어붙은 채널 수는 판단에 쓰지 않는다: 링크가 끊기면 Emb 이 마지막 버퍼를\n"
              "!!   계속 재발행하므로, 어느 채널이 먼저 '정지'로 잡히는지는 타이밍 문제다."
            : (all_ch ? "carrier 는 살아 있는데 8축 **전부** 얼었다 ⇒ **transfer/SPI 또는 MCU** 구간."
                      : "carrier 는 살아 있고 일부만 얼었다 ⇒ **FDCAN(그 다리)**. MCU 는 살아 있다."),
          link_down ? "**케이블·커넥터 점검이 먼저다.** 재기동으로 잠깐 살아나도 재발한다."
                    : "모터 전원 OFF/ON → Emb 재기동.",
          link_down ? "" : "\n!!   ⚠전원 재투입으로 잠깐 풀려도 재발한다 — 배선·커넥터·종단저항을 볼 것.",
          std::string(72,'!').c_str());

        // ═══ 증거 수집 (2026-08-24) ═══════════════════════════════════════
        //   여기까지의 배너는 "얼었다" 만 말한다. 아래가 **왜** 를 좁히는 부분이다.
        frz_event++;
        // ① Emb 가 아직 CPU 를 쓰고 있나 — 0.15초 두 점을 재서 본다.
        //   ⚠이게 **결정적 갈림길**이다. 여덟 번 동안 한 번도 구분한 적이 없다:
        //     CPU 가 늘고 있다 → Emb 는 돌고 EtherCAT 이 죽었다(슬레이브/배선/노이즈)
        //     CPU 가 멈췄다   → Emb 자체가 스톨/데드락이다(전원 OFF/ON 은 헛수고다)
        if(emb_pid<=0) emb_pid = EmbSnap::find_pid();
        EmbSnap e1 = EmbSnap::take(emb_pid);
        sleep_s(0.15);
        EmbSnap e2 = EmbSnap::take(emb_pid);
        const long long dj = (e1.jiffies>=0 && e2.jiffies>=0) ? (e2.jiffies-e1.jiffies) : -1;
        // ② 물리링크 — carrier 0 이면 케이블/커넥터다. CRC 증가는 전기적 노이즈다.
        NicSnap n1 = NicSnap::take();
        const long long dcrc = (n1.rx_crc>=0 && nic0.rx_crc>=0) ? n1.rx_crc-nic0.rx_crc : -1;
        const long long derr = (n1.rx_err>=0 && nic0.rx_err>=0) ? n1.rx_err-nic0.rx_err : -1;
        // ③ 동결 **직전** 1초의 채널별 최대 |τ| · |dq| · |I| — 하중/전류 상관용.
        double tpk[16]={0}, dpk[16]={0}, cpk[16]={0};
        frz_ring.peaks(lt - 0.5, 1.0, tpk, dpk, cpk);   // 얼기 시작한 시점(-0.5s) 이전 1초
        double tmax=0, cmax=0; int tch=-1;
        for(int i=0;i<NCH;i++) if(cfg.installed_has(i)){
          if(tpk[i]>tmax){ tmax=tpk[i]; tch=i; } if(cpk[i]>cmax) cmax=cpk[i]; }
        // ③-b ★**모터 상태워드(ucStatus)** — 세 가설이 여기서 갈린다 (2026-08-24).
        //   RUNBOOK 이 정리한 hardwareStatus 하위 6bit:
        //     0 OverCurrent · **1 OverVoltage** · 2 UnderVoltage
        //     3 MotorTemp   · 4 MosfetTemp     · 5 ADCCurrentOffset
        //   ⇒ bit1 이면 **회생 과전압**(손으로 밀거나 다리가 떨어지며 발전한다).
        //     bit3/4 면 **드라이버 과온**. bit2 면 전원 새그. 전부 0 이면 모터는 멀쩡했고
        //     EtherCAT/마스터 쪽이다.
        //   ⚠ERROR VECTOR 가 hardwareStatus 인지 quickStatus 인지는 미확인이다(RUNBOOK).
        //     그래서 **해석을 강요하지 않고 원값을 같이 남긴다.**
        //   ⚠동결이면 이 값도 **얼어붙은 마지막 값**이다 — 얼기 직전 상태라 오히려 그게 맞다.
        std::string sttx = "";
        long stt_or = 0;
        for(int i=0;i<NCH;i++) if(cfg.installed_has(i) && i<(int)hs.status.size()){
          char b[24]; std::snprintf(b,sizeof b,"%s%02x", sttx.empty()?"":",", hs.status[i]&0xff);
          sttx += b; stt_or |= (hs.status[i]&0xff);
        }
        if(sttx.empty()) sttx = "미보고";
        std::string sttwhy;
        { const char* NM[6]={"과전류","**과전압(회생)**","저전압","모터과온","MOSFET과온","ADC오프셋"};
          for(int b=0;b<6;b++) if(stt_or&(1<<b)){ if(!sttwhy.empty()) sttwhy+="·"; sttwhy+=NM[b]; }
          if(sttwhy.empty()) sttwhy = stt_or ? "비트 정의 밖(원값 확인)" : "**전부 0 — 모터는 정상보고**"; }
        // ③-c SoC 열·클럭·스로틀
        const SocSnap soc = SocSnap::take();

        // ④ 직전 구간 CSV
        char dp[128]; std::snprintf(dp,sizeof dp,"/tmp/freeze_pre_%d.csv", frz_event);
        size_t nrow = frz_ring.dump(dp);

        std::fprintf(stderr,
          "!! ── 증거 ──────────────────────────────────────────────────────\n"
          "!!  Emb 프로세스   pid %d · state %s · 0.15s CPU %+lld jiffies → **%s**\n"
          "!!  물리링크 %s    carrier %lld · rx_crc %+lld · rx_err %+lld → **%s**\n"
          "!!  모터 상태워드  [%s] → **%s**\n"
          "!!  SoC           %.1f°C · %.0fMHz · throttled %s%s\n"
          "!!  직전 1초 최대  |τ| %.2fNm(%s) · |I|(=τ에코, 실측아님) %.2f · 모드 %s(%.1fs째)\n"
          "!!  직전 %.1fs 기록 → %s (%zu행)\n"
          "!!  사건로그(누적) → %s   ← **여기를 여러 번 모아 놓고 봐야 원인이 보인다**\n"
          "%s\n\n",
          emb_pid, e2.state.c_str(), dj,
          dj<0 ? "판정불가(pid 못 찾음)"
               : (dj>0 ? "Emb 는 살아서 돈다 ⇒ EtherCAT 이 죽었다"
                       : "**Emb 가 멈췄다** ⇒ 전원 OFF/ON 이 아니라 Emb 재기동이 답이다"),
          n1.ifname.c_str(), n1.carrier, dcrc, derr,
          n1.carrier==0 ? "**링크 끊김 = 케이블/커넥터**"
                        : (dcrc>0 ? "**CRC 오류 증가 = 전기적 노이즈(모터 전류) 의심**"
                                  : "링크 정상 ⇒ 슬레이브/MCU 쪽"),
          sttx.c_str(), sttwhy.c_str(),
          soc.temp_c, soc.mhz,
          soc.thr<0 ? "(vcgencmd 없음)" : (soc.thr ? "★0x" : "0x"),
          soc.thr<0 ? "" : (soc.thr&0x7 ? " ⚠지금 걸려 있다" : (soc.thr ? " (이력만)" : "")),
          tmax, tch>=0?chname[tch].c_str():"-", cmax, mode.c_str(), lt-mode_t0,
          FRZ_PRE_S, nrow?dp:"(없음)", nrow, FRZ_LOG.c_str(),
          std::string(72,'!').c_str());

        // ⑤ 사건로그 한 줄 append — 재기동해도 남는다.
        char row[1024];
        std::snprintf(row,sizeof row,
          "%s\t%.1f\t%s\t%.1f\t%d\t%s\t%s\t%lld\t%s\t%lld\t%lld\t%lld\t%.2f\t%s\t%.2f\t%.0f\t"
          "%s\t%ld\t%.1f\t%.0f\t%ld\t%s",
          wall_stamp().c_str(), lt-t0, mode.c_str(), lt-mode_t0, nfz,
          link_down ? "EtherCAT" : (all_ch ? "SPI/MCU" : "FDCAN"), fz.c_str(),
          dj, e2.state.c_str(), n1.carrier, dcrc, derr,
          tmax, tch>=0?chname[tch].c_str():"-", cmax, hz_ema,
          sttx.c_str(), stt_or, soc.temp_c, soc.mhz, soc.thr, nrow?dp:"-");
        freeze_log_append(FRZ_LOG,
          "when\tuptime_s\tmode\tin_mode_s\tn_frozen\tclass\tchannels\t"
          "emb_dcpu_jif\temb_state\tnic_carrier\td_rx_crc\td_rx_err\t"
          "pre_tau_max_nm\tpre_tau_ch\tpre_cur_max_a\tloop_hz\t"
          "stt_hex\tstt_or\tsoc_temp_c\tsoc_mhz\tthrottled\tpre_csv", row);
      } else if(!nfz) frz_warned=false;
      if(nfz && mode=="off"){ /* 동결 중에는 무장을 막는다 */ }
      frozen_now = nfz; }
    // ★검토 #3 (2026-08-27): **무장 중** 동결은 즉시 E-stop — 동결값 위에서 토크/속도
    //   트립은 침묵하고(입력이 얼었으니 임계 도달 불가) 추정기·WBIC 는 오제어를 계속한다
    //   (FDCAN 다리 동결 = 실기 반복 재현 유형). 종전엔 off 상태의 무장만 막고
    //   무장 중엔 배너·증거수집뿐이었다.
    if(frozen_now>0 && mode!="off" && !estop){
      std::printf("[deploy] ⛔ E-STOP: 통신 동결 %d채널 (무장 중) — 동결값 위 제어 금지 → limp·래치\n",
                  frozen_now);
      estop = true; mode = "off"; hw->enable(0);
    }

    // ★★기동 시 **생존 확인**: 명령을 받기 전에 전 축이 실제로 갱신되는지 본다.
    //   동결은 세 번 다 같은 4채널이었고, 매번 **무장을 시도한 뒤에야** 드러났다.
    //   그때는 이미 E-stop 이 걸리고 로봇이 흔들린 뒤다. 먼저 확인하는 게 맞다.
    //   1초 동안 각 채널이 한 번이라도 바뀌는지 세고, 안 바뀐 채널이 있으면 무장을 막는다.
    if(!have_state){
      if(hs.mask != 0) ok_reads++; else ok_reads = 0;
      if(ok_reads == 1) live_t0 = lt;
      if(ok_reads >= 5 && lt-live_t0 < 1.0){ /* 아직 관찰 중 */ }
      else if(ok_reads >= 5){
        std::string dead; int nd=0;
        // ⚠mock 은 값이 정지해 있는 게 정상이라 이 검사를 건너뛴다(오탐).
        //   실기 엔코더는 정지 중에도 항상 미세하게 떨린다 — 그게 이 검사의 근거다.
        for(int i=0;i<NCH;i++) if(!mock && cfg.installed_has(i) && frz_t[i] > 0.9){
          char b[16]; std::snprintf(b,sizeof b," ch%d",i); dead+=b; nd++; }
        if(nd){
          std::fprintf(stderr,
            "\n%s\n!! ⛔⛔ **기동 생존확인 실패** — 1초 동안 값이 한 번도 안 바뀐 채널:%s\n"
            "!!   %s\n"
            "!!   **명령을 받지 않는다.** 배선을 고치고 전원 OFF/ON 후 다시 띄울 것.\n"
            "!!   (강제로 진행: LIVE_CHECK=0 — 죽은 축은 제어가 안 된다)\n%s\n\n",
            std::string(72,'!').c_str(), dead.c_str(),
            nd >= (int)cfg.joints.size() ? "8축 전부 → EtherCAT(EMB↔MCU)"
                                         : "일부만 → **FDCAN(그 다리)**. MCU 는 살아 있다.",
            std::string(72,'!').c_str());
          if(!(getenv("LIVE_CHECK") && atoi(getenv("LIVE_CHECK"))==0)){
            safe_shutdown(*hw, NCH); delete hw;
            mj_deleteData(d); mj_deleteModel(m); return 3;
          }
        }
        have_state = true;
        // ★2차 방어 — 첫 유효 상태로 hold 목표를 **선충전**한다(0 벡터 제거).
        hold_ch = hs.q_deg; jm.clamp_ch_via_joint(hold_ch.data()); hold_ch_primed = true;
        if(start_mode=="hold"){ mode="hold";
          hw->enable(1);
          std::printf("[deploy] --start-mode hold — 현재 자세를 잡는다(무장).\n"); }
        std::printf("[deploy] 센서 준비됨 — 명령 수신 시작 (q_ch: %.1f %.1f %.1f %.1f ...)\n",
                    hs.q_deg[0], hs.q_deg[1], hs.q_deg[2], hs.q_deg[3]); }
    }
    // ② 명령 폴링(~50Hz)
    if(have_state && k % (long long)std::max(1.0, 0.02/dt) == 0){
      Cmd nc;
      if(read_cmd(cmd_p, nc)){
        bool fresh = (nc.raw != last_raw);          // ★"파일이 읽히는가"가 아니라 "내용이 바뀌는가"
        last_raw = nc.raw;                          //   (biped_emb.read_cmd_fresh 와 같은 이유)
        if(fresh) last_cmd_t = lt;
        cmd = nc; body_h = nc.body_h;
        // ★강성 배율 수신 — **−1 은 "키 없음"**(옛 GUI)이라 무시한다. 0 은 유효한 값이다.
        if(nc.pos_kd_scale >= 0.0 && std::fabs(nc.pos_kd_scale - kd_scale_cmd) > 1e-6){
          kd_scale_cmd = nc.pos_kd_scale;
          std::printf("[deploy] kd 배율 → ×%.2f (명령). 자동(√kp)을 덮어쓴다.\n", kd_scale_cmd);
        }
        if(nc.pos_kp_scale >= 0.0){
          double want = std::max(0.0, std::min(KP_SCALE_MAX, nc.pos_kp_scale));
          if(std::fabs(want - kp_scale_tgt) > 1e-6){
            auto t2 = trip_deg(want);
            std::printf("[deploy] 강성 kp×%.2f → **×%.2f** (%.1fs 램프) — 트립 예민축 %s **%.2f°**\n",
                        POS_KP, want, KP_RAMP_S, t2.first.c_str(), t2.second);
            if(want > kp_scale_tgt && (mode=="stand" || mode=="walk"))
              std::printf("[deploy]   ⚠지금 %s 다 — 이 배율은 home/hold/jog 에만 걸린다(무영향).\n", mode.c_str());
            kp_scale_tgt = want;
          }
        }
        // ★축별 배율을 명령파일로 받는다 — env 는 기동 1회뿐이라 스윕 중에 못 바꾼다.
        if((int)nc.grav_scale_joint.size() >= NJ){
          bool ch=false;
          // ★[0,3] 클램프 (2026-08-27 검토) — 공통배율(아래)과 동일한 한계를 축별에도.
          //   이 값은 도구(float_gstar/GUI)가 파일로 주입한다 — 오타·스크립트 버그
          //   하나가 무제한 토크 배율이 되어 계단으로 걸리면 안 된다.
          double gcl[NJ];
          for(int j=0;j<NJ;j++){
            gcl[j] = std::max(0.0, std::min(3.0, nc.grav_scale_joint[j]));
            if(std::fabs(gcl[j]-grav_axis[j])>1e-9) ch=true;
          }
          if(ch){
            for(int j=0;j<NJ;j++) grav_axis[j] = gcl[j];
            std::printf("[deploy] 축별 중력배율(명령) =");
            for(int j=0;j<NJ;j++) std::printf(" %.3f", grav_axis[j]);
            std::printf("\n");
          }
        }
        if(nc.grav_scale >= 0.0 && std::fabs(nc.grav_scale-GRAV_SCALE) > 1e-9){
          GRAV_SCALE = std::max(0.0, std::min(3.0, nc.grav_scale));
          std::printf("[deploy] 중력보상 배율 → **×%.3f**%s\n", GRAV_SCALE,
                      (mode=="float") ? "" : "  (float 모드에서만 쓰인다)");
        }
        std::string nm = nc.mode;
        if(nm=="reset") nm = "hold";
        if(nm!="off" && nm!="hold" && nm!="home" && nm!="jog" && nm!="stand" && nm!="walk"
           && nm!="soft_off"
           && nm!="float" && nm!="push") nm = "off";  // ★float=무중력 · push=발밀기(힘제어)
        // ★E-stop 래치는 명시적 off 로만 해제. 그 전까지 모드변경 무시.
        if(estop){
          if(nm=="off"){ estop=false; std::printf("[deploy] E-stop 래치 해제(off 수신) — 재무장 가능\n"); }
          else nm = "off";
        }
        // ★기동 시 남아 있던 모드는 **다른 모드를 한 번 고를 때까지** 무시한다.
        if(mode_locked){
          if(nm == boot_mode) nm = "off";
          else { mode_locked = false;
                 std::printf("[deploy] 명령 잠금 해제 — %s 수신\n", nm.c_str()); }
        }
        if(nm!="stand" && nm!="walk") ground_refused = false;   // ★다른 모드 = 재시도 허용
        if(nm!="push") push_refused = false;                     // ★push 거부도 같은 규약
        // ★안전 종료 완료 래치 (2026-08-28): 완료 뒤에도 GUI 는 soft_off 를 20ms 마다
        //   계속 보낸다. 그대로 두면 재진입 → hw->enable(1) 로 **드라이브가 되살아난다.**
        //   ground_refused/push_refused 와 같은 규약: 다른 모드를 한 번 고를 때까지 off 유지.
        if(nm!="soft_off") soft_latched = false;
        if(soft_latched && nm=="soft_off") nm = "off";
        if(push_refused && nm=="push") nm = "hold";              //   래치 중 재발행은 hold 로
        // ★★거부는 **래치**다 (2026-08-21, mock 실측으로 발견).
        //   종전엔 `ground_refused` 가 **검사만** 건너뛰게 했다. 그런데 GUI 는 stand 를
        //   20ms 마다 재전송하므로, 다음 폴에서 nm!=mode 로 다시 진입하고 검사는 건너뛴 채
        //   **그대로 stand 로 들어갔다.** 즉 거부가 한 틱만 유효했다.
        //   ⚠접지 검사도 같은 경로였다 — 이 커밋 이전엔 "접지 안 됨" 도 한 번만 막혔다.
        //   ⇒ 다른 모드를 한 번 고를 때까지 stand/walk 를 hold 로 눌러 둔다.
        if(ground_refused && (nm=="stand" || nm=="walk")) nm = "hold";
        // ★★**움직이는 중에는 무장하지 않는다** (2026-08-20 실기).
        //   속도트립은 *측정* 속도로 걸린다 — 우리가 명령을 안 줘도, 로봇이 이미
        //   무너지는 중이면 무장하는 순간 그대로 트립한다(ch6 201dps · ch2 204 · ch3 224).
        //   실제 경위: 접지 상태에서 Ctrl+C → safe_shutdown 이 kp=kd=τ=0 을 써서
        //   **로봇이 주저앉았고**, 그 낙하 도중에 hold 를 눌렀다.
        //   ⇒ 트립의 원인은 명령이 아니라 "움직이는 걸 잡으려 한 것" 이다. 미리 막는다.
        if(mode=="off" && nm!="off" && frozen_now>0){
          std::printf("[deploy] ⛔ **EtherCAT 동결 %d채널** — 무장 거부. 전원 OFF/ON 후 Emb 재기동.\n", frozen_now);
          nm = "off";
        } else if(mode=="off" && nm!="off"){
          double vmx=0; int vch=-1;
          for(int i=0;i<NCH;i++) if(cfg.installed_has(i) && std::fabs(hs.dq_dps[i])>vmx){
            vmx=std::fabs(hs.dq_dps[i]); vch=i; }
          const double vlim = cfg.vel_trip_dps * 0.25;      // 트립의 1/4 — 잔진동은 허용
          if(vmx > vlim){
            if(!still_warned){ still_warned = true;
              std::printf("[deploy] ⛔ **아직 움직이고 있다** — ch%d %.0fdps > %.0f. 무장 거부.\n"
                          "         로봇이 정지한 뒤에 누를 것(주저앉는 중이면 크레인으로 받칠 것).\n"
                          "         지금 무장하면 속도트립(%.0fdps)에 즉시 걸린다.\n",
                          vch, vmx, vlim, cfg.vel_trip_dps); }
            nm = "off";
          } else still_warned = false;
        }
        if(nm != mode){
          prev_mode = mode; mode = nm; mode_t0 = lt;
          // ★stand 를 다시 들어오면 비교를 새로 잡는다(hold 스냅샷은 남긴다 — 기준이니까).
          if(mode=="stand") have_tau_stand=false;
          if(((mode!="off" && prev_mode=="off") || mode=="stand" || mode=="walk") && !trc){
            trc = fopen("/tmp/arm_trace.csv","w"); trc_t0 = lt;
            if(trc){ fprintf(trc,"t");
              // cmd=실제 나간 위치명령 · kp/kd=그 틱에 실제 나간 게인(blend 중 매 틱 변한다)
              for(int i=0;i<NCH;i++) fprintf(trc,",q%d,dq%d,tau%d,cmd%d,kp%d,kd%d",i,i,i,i,i,i);
              fprintf(trc,"\n");
              std::printf("[deploy] 트레이스 → /tmp/arm_trace.csv (3초 — 블렌드 전체)\n"); } }
          hw->enable(mode=="off" ? 0 : 1);
          if(mode=="push" && (prev_mode=="stand" || prev_mode=="walk")){
            // ★검토 #7 (2026-08-27): 하중을 받는 stand/walk 에서 오클릭 한 번에
            //   지지토크(WBIC)가 사라지던 경로 봉인 — push 는 매달림+저울 전제라 직행 금지.
            std::printf("[deploy] STOP push 거부 — stand/walk 에서 직행 금지(지지 상실 위험). hold 로 잡는다.\n");
            hold_ch = hs.q_deg; jm.clamp_ch_via_joint(hold_ch.data());
            nm = "hold"; mode = "hold"; push_refused = true;   // ★래치(검증 반박 반영) — GUI 20ms
          }                                                    //   재발행이 50Hz 스래시+처짐추종을 만들던 구멍
          if(mode=="push"){
            // ★매달림 확인 (2026-08-27 · float 와 같은 잣대) — push 전제 = "크레인 팽팽 +
            //   발이 저울에 살짝(수백 g)". 체중이 실린 채 진입하면 지지력이 사라진다.
            //   정상 push 세팅(프리로드 ~0.3kg)은 비 ~1.2 이하로 통과한다(실측).
            for(int j=0;j<NJ;j++){ d->qpos[7+j]=q_ctrl[j]; d->qvel[6+j]=0.0; }
            d->qpos[0]=d->qpos[1]=0; d->qpos[2]=0.5;
            d->qpos[3]=1; d->qpos[4]=d->qpos[5]=d->qpos[6]=0;
            for(int i=0;i<6;i++) d->qvel[i]=0.0;
            mj_forward(m,d);
            std::vector<double> tm(NJ,0.0); jm.ch_to_tau_joint(hs.tau_nm.data(), tm.data());
            double hang=0, meas=0;
            for(int j=0;j<NJ;j++){ hang += std::fabs(d->qfrc_bias[6+j]); meas += std::fabs(tm[j]); }
            const double ratio=(hang>1e-6)?meas/hang:0.0;
            const double gmax = getenv("FLOAT_GROUND_MAX") ? atof(getenv("FLOAT_GROUND_MAX")) : 1.25;
            std::printf("[deploy] 매달림 확인(push) — 실측 |t|합 %.2f vs 예측 %.2f . 비 %.2f (상한 %.2f)\n",
                        meas, hang, ratio, gmax);
            if(ratio > gmax){
              std::printf("[deploy] STOP push 거부 — 하중이 실려 있다. 크레인 이양 후 다시 누를 것.\n");
              hold_ch = hs.q_deg; jm.clamp_ch_via_joint(hold_ch.data());
              nm = "hold"; mode = "hold"; push_refused = true;   // ★래치 — 위와 동일 규약
            }
          }
          if(mode=="push"){
            // ★push 는 float 와 반대로 **접지(저울 접촉)가 전제**다 — 접지 거부를 안 건다.
            //   대신 힘은 반드시 0 에서 시작해 램프로만 오른다(계단 금지 = 저울/발 충격 방지).
            push_fz_cur = 0.0;
            hold_ch = hs.q_deg; jm.clamp_ch_via_joint(hold_ch.data());
            std::printf("[deploy] **발밀기(push)** — 다리 %s · τ = g*·G + Jᵀ(0,0,−F)\n"
                        "         F 는 0 에서 시작, GUI 목표까지 %.1f N/s 램프 (상한 %.0f N)\n"
                        "         ⚠발밑에 저울. 크레인 줄은 팽팽하게(반작용을 줄이 받는다).\n"
                        "         ⚠마찰 소거는 **램프 왕복 평균**으로 — 한 점 값은 ±2~3 N/축 흐리다.\n",
                        cmd.push_leg==1?"HR":"HL", PUSH_RATE, PUSH_MAX);
          }
          if(mode=="float"){
            // ★★**접지 중이면 거부한다** — stand 의 접지 판정을 **역으로** 건다.
            //   발이 땅에 닿은 채 중력보상을 켜면 지면 반력을 모르는 채로 밀어 올린다:
            //   모델은 매달림(반력 0)을 가정하는데 실제로는 반력이 이미 무게를 받치고 있어
            //   **중력보상분이 순수 잉여**가 되고 로봇이 튀어오르거나 한쪽 발로 밀어 넘어진다.
            for(int j=0;j<NJ;j++){ d->qpos[7+j]=q_ctrl[j]; d->qvel[6+j]=0.0; }
            d->qpos[0]=d->qpos[1]=0; d->qpos[2]=0.5;
            d->qpos[3]=1; d->qpos[4]=d->qpos[5]=d->qpos[6]=0;
            for(int i=0;i<6;i++) d->qvel[i]=0.0;
            mj_forward(m,d);
            std::vector<double> tm(NJ,0.0); jm.ch_to_tau_joint(hs.tau_nm.data(), tm.data());
            double hang=0, meas=0;
            for(int j=0;j<NJ;j++){ hang += std::fabs(d->qfrc_bias[6+j]); meas += std::fabs(tm[j]); }
            const double ratio = (hang>1e-6) ? meas/hang : 0.0;
            const double gmax = getenv("FLOAT_GROUND_MAX") ? atof(getenv("FLOAT_GROUND_MAX")) : 1.25;
            std::printf("[deploy] 매달림 확인 — 실측 |t|합 %.2f Nm vs 매달림 예측 %.2f Nm . 비 %.2f (상한 %.2f)\n",
                        meas, hang, ratio, gmax);
            if(ratio > gmax){
              std::printf("[deploy] STOP **접지 중이다** - 무중력 거부, hold 유지.\n"
                          "         크레인으로 들어올려 발을 띄운 뒤 다시 누를 것.\n");
              mode = "hold"; prev_mode = "float";
            } else {
            // ★유지축(FLOAT_AXES 밖)의 목표. 무중력축은 이 값을 안 쓴다.
            hold_ch = hs.q_deg; jm.clamp_ch_via_joint(hold_ch.data());
            std::printf("[deploy] **무중력(중력보상)** — 배율 ×%.3f · kd×%.2f · 축 %s\n"
                        "         ⚠매달린 상태 전용. 손으로 밀어 보며 중립 배율을 찾는다.\n"
                        "         ⚠마찰(관절 0.6~0.9Nm)은 안 지워진다 — 뻑뻑한 게 정상이다.\n",
                        GRAV_SCALE, FLOAT_KD,
                        float_axes.empty() ? "**전축**(기본)" : "지정축만(FLOAT_AXES · 디버그)");
            }
          }
          if(mode=="hold"){
            // ★중력지지는 **매 진입마다 0 에서 다시 램프**한다 — GUI 버튼이 눌린 채로
            //   남아 있어도 진입 순간 계단이 걸리지 않는다(push 와 같은 규약).
            hold_fz_cur = 0.0; hold_ff_pct_cur = 0.0; hold_split_cur = 50.0;
            std::vector<float> raw = hs.q_deg;             // 클램프 전
            // ★★home → hold 는 **home 의 목표를 이어받는다** (2026-08-21, 사용자 요청).
            //   하려는 것: home 자세를 잡고 hold 로 굳힌 뒤 **크레인을 내려 중력을 걸어도
            //   그 자세를 유지**시키는 것. 그런데 종전처럼 진입 시 측정각을 래치하면,
            //   매달린 채 이미 처져 있던 각이 목표가 된다 — 접지하면 거기서 더 처지고
            //   그때 재진입하면 또 그 자리를 목표로 삼는다. **처짐이 누적된다.**
            //   ⇒ 방금 home 이 지시한 자세(home_to)를 그대로 목표로 쓴다. 그러면 hold 는
            //     "지금 자세 굳히기" 가 아니라 "home 자세로 버티기" 가 된다.
            //   ⚠이건 진입 순간 **오차 0 이 아니다**(처진 만큼 계단). 그래서 얼마나 되는지
            //     반드시 찍는다 — 크면 게인이 부족하거나 그 축이 구동되지 않은 것이다.
            //   ⚠HOLD_LATCH=meas 로 종전(측정각 래치) 동작을 되돌릴 수 있다.
            //   ★★**home 이 끝났을 때만 인계한다** (2026-08-24 실기에서 터졌다).
            //     종전엔 prev_mode=="home" 이기만 하면 무조건 home_to 를 목표로 삼았다.
            //     램프 **도중**에 hold 를 누르면 로봇은 아직 출발점 근처인데 목표가
            //     도착점이 되어 **남은 이동량 전부가 계단 명령**이 된다.
            //     실기 실측(2026-08-24): off→home 직후 hold → `진입 오차 최대 HL_foot +60.22°`
            //       → kp_ch 30 × 1.05rad = **31.5 Nm**(채널) → ch3 **365dps** → 속도트립 limp.
            //     즉 hold 가 "정지" 가 아니라 **가속**이었다. 배포경로 감사도 같은 것을
            //     HIGH 로 잡았다("home 램프 도중 HOLD 를 누르면 …").
            //   ⇒ 램프 중이면 **그 순간의 지령각(q_ch)** 을 래치한다. 계단이 0 이고,
            //     측정각이 아니라 **지령**이라 처짐 누적도 안 생긴다(원래 인계의 목적).
            const char* hl = getenv("HOLD_LATCH");
            const bool meas_mode = (hl && std::string(hl)=="meas");
            const bool from_home = (prev_mode=="home") && !meas_mode;
            const bool inherit   = from_home && home_done;
            if(from_home && !home_done){
              hold_ch = q_ch;                       // 램프가 지금 지시하고 있던 자세
              double emx=0; int ech=-1;
              for(int i=0;i<NCH;i++) if(cfg.installed_has(i)){
                double e=(double)q_ch[i]-(double)raw[i];
                if(std::fabs(e)>std::fabs(emx)){ emx=e; ech=i; } }
              const double u = (home_T>0) ? std::min(1.0,(lt-home_t0)/home_T) : 1.0;
              std::printf("[deploy] hold \u2190 **home \ub7a8\ud504 \ub3c4\uc911**(\uc9c4\ud589 %.0f%%) \u2014 \ub3c4\ucc29\uc810\uc774 \uc544\ub2c8\ub77c\n"
                          "         **\uc9c0\uae08 \uc9c0\ub839\uac01**\uc744 \ub798\uce58\ud55c\ub2e4(\uacc4\ub2e8 \ubc29\uc9c0). \uc9c4\uc785 \uc624\ucc28 \ucd5c\ub300 %s %+.2f\u00b0\n",
                          u*100.0, ech>=0?chname[ech].c_str():"-", emx);
            } else if(inherit){
              hold_ch = home_to;
              double emx=0; int ech=-1;
              for(int i=0;i<NCH;i++) if(cfg.installed_has(i)){
                double e=(double)home_to[i]-(double)raw[i];
                if(std::fabs(e)>std::fabs(emx)){ emx=e; ech=i; } }
              std::printf("[deploy] hold ← **home 목표 인계**(측정각 래치 아님).\n"
                          "         진입 오차 최대 %s %+.2f° — 이만큼이 계단 명령이 된다.\n",
                          ech>=0?chname[ech].c_str():"-", emx);
            } else {
              hold_ch = hs.q_deg;
            }
            jm.clamp_ch_via_joint(hold_ch.data());
            // ★hold 진입은 "측정각을 그대로 목표로" 라 **오차 0 이어야** 한다.
            //   그런데 실기에서 진입 즉시 ch2(204dps)·ch3(224dps) 속도트립이 났다.
            //   원인 후보가 둘이라 찍어서 가른다:
            //     (a) clamp_ch_via_joint 의 ch→관절→ch 왕복이 값을 바꾼다(발목은 커플링이라
            //         calf 에 의존한다 — 왕복이 항등이 아닐 수 있다)
            //     (b) 왕복은 항등인데 **게인 인계 점프**다(Emb kd 5.0 → 우리 3.5/2.0).
            //   Δ 가 0 이면 (b), 0 이 아니면 (a) 다. 한 번만 찍는다.
            //
            //   ★★2026-08-26 — **(b) 는 이제 성립하지 않는다.** 지우지 않고 남기는 이유는
            //     아래 판정 규칙("Δ 가 0 이면 (b)")이 그대로 살아 있어서다. 그 규칙을 믿고
            //     읽으면 **있지도 않은 게인 점프**를 원인으로 지목하게 된다.
            //     경위: (b) 는 "Emb 는 kd 5.0 균일" 이라는 전제였다. 그건 halGait.cpp:735-736
            //     의 **주석 처리된 죽은 줄**(//<--20, //<--1)을 읽은 값이었다. 살아 있는 건
            //     :804-820 이고 거기 값이 2026-08-20 에 우리 config 와 같게 맞춰졌다:
            //         hip 100/6.0 · thigh 50/4.0 · calf 80/3.5 · foot 30/2.0
            //     ⇒ **ch2(calf)·ch3(foot) 은 인계 전후 kp·kd 가 완전히 동일하다.**
            //       그 두 채널에서 난 트립(204·224dps)은 게인 점프로 설명할 수 없다.
            //   ⇒ 새 판정: Δ≠0 이면 여전히 (a) 다. **Δ=0 인데 튀면 (a)도 (b)도 아니다** —
            //     남은 후보는 낡은 측정값(센서 미준비)·통신 동결·그 축의 기구 걸림이다.
            //     그때는 /tmp/arm_trace.csv 와 동결 사건로그(/tmp/biped_freeze_log.tsv)를 볼 것.
            //   ⚠home 인계일 때는 이 진단이 성립하지 않는다 — 목표가 측정각이 아니니
            //     Δ 가 0 이 아닌 게 정상이다. 그때는 클램프 영향만 따로 잰다.
            std::vector<float> base = inherit ? home_to : raw;
            std::string dmsg; double dmx=0; int dch=-1;
            for(int i=0;i<NCH;i++){
              double dd = (double)hold_ch[i]-(double)base[i];
              if(std::fabs(dd)>std::fabs(dmx)){ dmx=dd; dch=i; }
              char b[48]; std::snprintf(b,sizeof b," ch%d %+.3f", i, dd); dmsg += b;
            }
            std::printf("[deploy] hold 래치 — 클램프 Δ(목표−측정):%s\n", dmsg.c_str());
            if(std::fabs(dmx)>0.05)
              std::printf("[deploy] ⚠클램프가 ch%d 를 %+.3f° 옮겼다 — 그만큼 **계단 명령**이 된다.\n", dch, dmx);
            else
              // ★2026-08-26: 종전 "움직이면 게인 인계 점프다" 를 버렸다 — Emb 게인이 우리와
              //   같아져서(위 주석 참조) calf·foot 은 인계 점프가 **원천적으로 없다.**
              std::printf("[deploy] ✓클램프 영향 없음(최대 %+.3f°) — 그래도 움직이면 게인 점프가\n"
                          "         아니다(Emb 게인 = 우리 값). 낡은 측정값·동결·기구 걸림을 볼 것.\n", dmx);
            std::printf("[deploy]   인계 게인 kp/kd = hip %.0f/%.1f · thigh %.0f/%.1f · calf %.0f/%.1f · foot %.0f/%.1f\n",
                        cfg.joints[0].kp,cfg.joints[0].kd, cfg.joints[1].kp,cfg.joints[1].kd,
                        cfg.joints[2].kp,cfg.joints[2].kd, cfg.joints[3].kp,cfg.joints[3].kd);
          }
          if(mode=="jog"){
            // ★현재 **측정각에서** 시작한다(클램프 없이) — 명령 점프 방지.
            //   한계로 클램프해서 시작하면, 현재 자세가 jog 범위 밖일 때 진입 즉시
            //   한계까지 순간이동 명령이 나간다(= 막으려던 바로 그 점프).
            //   biped_emb.py Jogger.reset 이 같은 이유로 클램프를 뺐다(2026-08-10).
            jm.ch_to_q_joint(hs.q_deg.data(), jog_q.data());
            jog_prev_t = lt; jog_init = true;
            std::printf("[deploy] jog 진입 — 현재 자세에서 시작 · %.0f dps 제한 · 한계는 관절한계\n",
                        cfg.jog_speed_dps);
          }
          if(mode=="soft_off"){
            // 목표 = SOFT_OFF_DEG(관절 deg 8개 CSV) · 없으면 **기동 시 자세**
            //   (매달림이면 그게 자연 매달림 자세라 낙차가 0 이다 — airwalk 가 검증한 규약)
            soft_from = hs.q_deg; soft_to = boot_pose_set ? boot_pose : hs.q_deg;
            if(const char* e=getenv("SOFT_OFF_DEG")){
              std::vector<double> sv; std::stringstream ss(e); std::string tk;
              while(std::getline(ss,tk,',')) sv.push_back(atof(tk.c_str()));
              if((int)sv.size()>=NJ) jm.q_joint_to_ch(sv.data(), soft_to.data());
            }
            jm.clamp_ch_via_joint(soft_to.data());
            const double rate = getenv("SOFT_OFF_DPS") ? atof(getenv("SOFT_OFF_DPS")) : 20.0;
            double mx=0; for(int i=0;i<NCH;i++) if(cfg.installed_has(i))
              mx = std::max(mx, (double)std::fabs(soft_to[i]-soft_from[i]));
            soft_T = std::max(0.5, mx / std::max(1.0, rate));
            soft_t0 = lt; soft_done = false;
            std::printf("[deploy] ■ 안전 종료 — 최대이동 %.1f° · %.0f dps · %.1fs 서행 후 무여자\n"
                        "         (하중을 받고 있으면 **크레인으로 먼저 받칠 것** — 이건 낙차만 줄인다)\n",
                        mx, rate, soft_T);
          }
          if(mode=="home"){
            home_from = hs.q_deg;
            // ★★목표자세 = **기하 진리 고정** (2026-08-28 사용자 지정): 1점 Qhome8 · 2점 Qflat8.
            //   yaml `home.q_deg` 는 더 읽지 않는다 — 측정 캠페인마다 yaml 을 임시 편집하던
            //   관행이 B자세 실수 커밋(08-27)까지 만든 원흉이다. home = stand 블렌드 종점과
            //   항상 같은 자세가 되고(기하 일관), 전축 0° 는 GUI [Zero 자세](jog)가 담당한다.
            //   측정용 임시 자세는 **HOME_DEG env**(관절 deg 8개 CSV — 1회성, 커밋 불가)로만.
            //   ⚠biped_emb.py 는 여전히 yaml 을 읽는다 — 실기 writer 는 배포기 하나라 무해.
            bool env_home = false;
            if(const char* e = getenv("HOME_DEG")){
              std::vector<double> hv; std::stringstream hss(e); std::string tk;
              while(std::getline(hss, tk, ',')) hv.push_back(atof(tk.c_str()));
              if((int)hv.size() >= NJ){
                jm.q_joint_to_ch(hv.data(), home_to.data());             // env 는 **deg**
                env_home = true;
                std::printf("[deploy] ⚠HOME_DEG 덮어씀(측정용) — home 이 기하 진리가 아니다:");
                for(int j=0;j<NJ;j++) std::printf(" %.1f", hv[j]);
                std::printf("\n");
              } else {
                std::printf("[deploy] ⚠HOME_DEG 원소 %d < %d — 무시하고 기하 진리 사용\n",
                            (int)hv.size(), NJ);
              }
            }
            if(!env_home){
              std::vector<double> qt(NJ);
              for(int j=0;j<NJ;j++) qt[j] = (c.cmode==1 ? c.Qflat8[j] : c.Qhome8[j]);
              jm.q_ctrl_to_ch(qt.data(), home_to.data());                // 기하는 **rad**
            }
            jm.clamp_ch_via_joint(home_to.data());
            double mx=0;
            for(int i=0;i<NCH;i++) if(cfg.installed_has(i))
              mx = std::max(mx, (double)std::fabs(home_to[i]-home_from[i]));
            // ★★궤적도 biped_emb.py 와 같은 **5차식** s=10τ³−15τ⁴+6τ⁵ 로 바꿨다(2026-08-21).
            //   종전 smoothstep 3u²−2u³ 는 시작·끝에서 **가속도가 0 이 아니다**(6Δ/T²).
            //   즉 HOME 을 누르는 순간 토크가 계단으로 튄다. 5차식은 경계 가속도까지 0 이라
            //   그 계단이 없다 — 드라이버 래치오프가 잦은 지금 이건 취향 문제가 아니다.
            //   극값: s'max = 1.875 (τ=0.5) · s''max = 10/√3 ≈ 5.7735
            //   ⚠같은 v/a 한계면 T 가 1.25배 길어진다. 40dps·60dps² · Δ100° → 3.8s→4.7s.
            //   ⚠거리 mx 는 **채널각**으로 잰다(biped_emb.py 는 관절각). 일부러 다르다:
            //     발목 채널 = calf+foot 이라 채널 이동이 관절보다 크고, 속도트립도 채널
            //     기준이다. 관절각으로 재면 발목 채널만 조용히 한계를 넘는다.
            const double S_VMAX = 1.875, S_AMAX = 10.0/std::sqrt(3.0);
            double T1 = S_VMAX*mx/std::max(1e-6, cfg.home_speed_dps);
            double T2 = std::sqrt(S_AMAX*mx/std::max(1e-6, cfg.home_acc_dps2));
            home_T = std::max(std::max(T1,T2), cfg.home_min_time_s);
            home_t0 = lt; home_done = false; home_warned = false;
            std::printf("[deploy] home → **%s** 자세 · 최대이동 %.1f° · %.1fs 램프"
                        "(5차 S-curve · 최대 %.0fdps ≪ 트립 %.0f)\n",
                        env_home ? "HOME_DEG(측정용 임시)"
                                 : (c.cmode==1?"2점 평발 Qflat8":"1점 점발 Qhome8"),
                        mx, home_T, S_VMAX*mx/home_T, cfg.vel_trip_dps);
          }
          // ★★접지 확인 없이 stand 를 못 켜게 막는다 (2026-08-20 실기).
          //   매달린 채 stand 를 누르면 WBIC 가 요구하는 지면반력을 지면이 못 내줘
          //   QP 가 매 틱 실패하고(시뮬 실측 95%) 중력보상 폴백으로 떨어진다.
          //   실기에선 그 직전에 다리가 흔들려 속도트립이 난다(ch7 207dps).
          //   ⚠겉보기엔 "그냥 안 되는" 것처럼 보여 원인을 찾기 어렵다 — 그래서 막는다.
          //
          //   판별: **모델이 예측한 '매달림' 중력토크와 실측 토크를 비교**한다.
          //     매달림이면 다리 자중만 걸리므로 둘이 비슷하다.
          //     접지면 몸무게가 다리를 통해 내려와 실측이 훨씬 커진다.
          //   힘센서가 없어도 되고, 임계를 사람이 정하지 않아도 된다.
          // ★거부는 **래치**한다. 명령파일이 계속 stand 면 50Hz 로 재시도하며 로그가
          //   폭주한다(같은 함정이 hold↔stand 재전이에도 있었다 — 그때 남긴 주석 참조).
          //   래치는 운전자가 stand/walk 가 아닌 모드를 한 번 보내면 풀린다.
          if((mode=="stand" || mode=="walk") && prev_mode!="stand" && prev_mode!="walk"
             && !ground_refused){
            for(int j=0;j<NJ;j++){ d->qpos[7+j]=q_ctrl[j]; d->qvel[6+j]=0.0; }
            d->qpos[0]=d->qpos[1]=0; d->qpos[2]=0.5;
            d->qpos[3]=1; d->qpos[4]=d->qpos[5]=d->qpos[6]=0;
            for(int i=0;i<6;i++) d->qvel[i]=0.0;
            mj_forward(m,d);
            std::vector<double> tau_meas(NJ,0.0);
            jm.ch_to_tau_joint(hs.tau_nm.data(), tau_meas.data());
            double hang=0, meas=0;
            for(int j=0;j<NJ;j++){
              hang += std::fabs(d->qfrc_bias[6+j]);
              meas += std::fabs(tau_meas[j]);
            }
            const double ratio = (hang>1e-6) ? meas/hang : 0.0;
            // ★기준 1.25 — 1.5 는 **너무 높았다**(2026-08-20 정정). 모델로 재보니
            //   매달림 16.21 Nm · 체중전부 25.72 Nm 라 비가 **1.00~1.59** 밖에 안 움직인다.
            //   1.5 를 요구하면 체중의 **95%** 를 넘겨야 통과 = 크레인을 거의 다 내린 상태인데,
            //   hold 는 그 하중을 못 버티고 주저앉는다(kp 100/50/80/30 은 체중용이 아니다).
            //   ⇒ 1.25 ≈ 체중의 **42%**. "확실히 닿았다" 는 알 수 있고 hold 도 버틴다.
            const double need = getenv("GROUND_RATIO") ? atof(getenv("GROUND_RATIO")) : 1.25;
            std::printf("[deploy] 접지 확인 — 실측 |τ|합 %.2f Nm vs 매달림 예측 %.2f Nm · 비 %.2f (기준 %.2f)\n",
                        meas, hang, ratio, need);
            if(ratio < need){
              std::printf("[deploy] ⛔ **접지가 안 됐다** — stand 거부, hold 유지.\n"
                          "         크레인을 내려 발바닥을 붙이고 하중을 로봇에 넘긴 뒤 다시 누를 것.\n"
                          "         (강제로 진행하려면 GROUND_RATIO=0 — 매달림 stand 는 위험하다)\n");
              // ★★**목표를 여기서 래치한다** (2026-08-24, 감사에서 CRITICAL 로 확인).
              //   hold 진입 부작용(:725 부근)은 이 가드보다 **위**에 있어 이미 지나갔다.
              //   래치를 안 하면 hold_ch 가 부팅 초기값(0 벡터)인 채로 :984 가 그걸 내보낸다:
              //     실측(mock, home→stand) q_ch = [-1.0, 4.0, 39.1, **98.8**, -3.3, -6.5, -48.9, **-100.0**]
              //     요구 채널토크 = kp_ch·err[rad] → HR_calf 68.2 Nm · HL_foot 51.7 Nm
              //     = tau_trip 15Nm 의 **3.4~4.5배**. 드라이브 한계 안이라 드라이버가 실제로 낸다.
              //   ⇒ 속도트립이 ~5ms 만에 걸려 limp 로 떨어진다. **부분 접지면 그대로 주저앉는다.**
              //   ⚠화면엔 "거부, hold 유지" 라고 뜬다 — 유지할 hold 가 애초에 없었다.
              //   ⚠Python(biped_emb.py)엔 이 버그가 없다: 가드가 FSM **진입 전에** 모드를 바꿔
              //     `if fsm.entered(HOLD): hold_leg = q_leg.copy()` 가 정상 발화한다. C++ 이관에서만 갈라졌다.
              // ★★램프 도중 거부면 도착점이 아니라 그 순간의 지령각(q_ch)을 래치한다
              //   (2026-08-27 검토) — hold 정상 진입의 2026-08-24 수정과 같은 논리를 이식.
              //   종전엔 home 진행 20% 지점에서 거부되면 남은 이동량 전부가 계단 명령이었다.
              // ★08-27 버그헌트: hold 중 거부(prev_mode=="hold")면 **재래치 금지** —
              //   hs.q_deg 는 PD 처짐이 실린 측정각이라, 거부가 반복 발화할 때마다
              //   목표가 처진 각을 다시 물어 계단식으로 주저앉는다. 기존 목표 유지.
              if(prev_mode!="hold")
                hold_ch = (prev_mode=="home")
                            ? ((home_done && home_to.size()==(size_t)NCH) ? home_to : q_ch)
                            : hs.q_deg;
              jm.clamp_ch_via_joint(hold_ch.data());
              nm = "hold"; mode = "hold"; ground_refused = true;
            }
          }
          // ★★자세 거리 가드 (2026-08-21). 블렌드 목표가 기하 자세로 **이동**하게 바뀌면서
          //   생긴 위험을 막는다. 종전엔 목표가 측정각에 얼어 있어 아무리 멀어도 안 움직였다.
          //   지금은 2.5초 안에 그 거리를 쓸어버린다 — `home` 을 건너뛰면 위험하다:
          //       발목이 100° 떨어져 있으면 피크 ≈ 0.25·kp_raw 43.2·1.75rad ≈ **18.9 Nm**
          //       → 토크트립 15 Nm 초과
          //   ⚠"자동으로 home 을 돌리면 되지 않나" 는 **틀렸다.** home 은 매달린 채 하는
          //     동작이고 stand 는 접지 후다. stand 안에서 home 을 돌리면 **하중이 실린 채로
          //     다리를 크게 움직이게** 된다. 그래서 자동 실행이 아니라 **거부**가 맞다.
          if((mode=="stand" || mode=="walk") && !ground_refused){
            std::vector<double> qt0(NJ); std::vector<float> to0(NCH,0.f);
            for(int j=0;j<NJ;j++) qt0[j] = (c.cmode==1 ? c.Qflat8[j] : c.Qhome8[j]);
            jm.q_ctrl_to_ch(qt0.data(), to0.data());
            jm.clamp_ch_via_joint(to0.data());
            double far=0; int fch=-1;
            for(int j=0;j<NJ;j++){ int ch=cfg.joints[j].channel;
              double dd=std::fabs((double)to0[ch]-(double)hs.q_deg[ch]);
              if(dd>far){ far=dd; fch=ch; } }
            //   임계 15° — 처짐(채널 수 도)은 통과하고 "home 안 돌림"(채널 100°)은 걸린다.
            const double lim = getenv("STAND_POSE_LIM") ? atof(getenv("STAND_POSE_LIM")) : 15.0;
            if(far > lim){
              std::printf("[deploy] ⛔ **자세가 목표에서 멀다** — stand 거부, hold 유지.\n"
                          "         ch%d 가 %.1f° 어긋나 있다(허용 %.1f°). **먼저 home 을 돌릴 것.**\n"
                          "         (강제: STAND_POSE_LIM=999 — 블렌드가 그 거리를 2.5초에 쓸어 트립할 수 있다)\n",
                          fch, far, lim);
              // ★★**목표를 여기서 래치한다** (2026-08-24, 감사에서 CRITICAL 로 확인).
              //   hold 진입 부작용(:725 부근)은 이 가드보다 **위**에 있어 이미 지나갔다.
              //   래치를 안 하면 hold_ch 가 부팅 초기값(0 벡터)인 채로 :984 가 그걸 내보낸다:
              //     실측(mock, home→stand) q_ch = [-1.0, 4.0, 39.1, **98.8**, -3.3, -6.5, -48.9, **-100.0**]
              //     요구 채널토크 = kp_ch·err[rad] → HR_calf 68.2 Nm · HL_foot 51.7 Nm
              //     = tau_trip 15Nm 의 **3.4~4.5배**. 드라이브 한계 안이라 드라이버가 실제로 낸다.
              //   ⇒ 속도트립이 ~5ms 만에 걸려 limp 로 떨어진다. **부분 접지면 그대로 주저앉는다.**
              //   ⚠화면엔 "거부, hold 유지" 라고 뜬다 — 유지할 hold 가 애초에 없었다.
              //   ⚠Python(biped_emb.py)엔 이 버그가 없다: 가드가 FSM **진입 전에** 모드를 바꿔
              //     `if fsm.entered(HOLD): hold_leg = q_leg.copy()` 가 정상 발화한다. C++ 이관에서만 갈라졌다.
              // ★★램프 도중 거부면 도착점이 아니라 그 순간의 지령각(q_ch)을 래치한다
              //   (2026-08-27 검토) — hold 정상 진입의 2026-08-24 수정과 같은 논리를 이식.
              //   종전엔 home 진행 20% 지점에서 거부되면 남은 이동량 전부가 계단 명령이었다.
              // ★08-27 버그헌트: hold 중 거부(prev_mode=="hold")면 **재래치 금지** —
              //   hs.q_deg 는 PD 처짐이 실린 측정각이라, 거부가 반복 발화할 때마다
              //   목표가 처진 각을 다시 물어 계단식으로 주저앉는다. 기존 목표 유지.
              if(prev_mode!="hold")
                hold_ch = (prev_mode=="home")
                            ? ((home_done && home_to.size()==(size_t)NCH) ? home_to : q_ch)
                            : hs.q_deg;
              jm.clamp_ch_via_joint(hold_ch.data());
              nm = "hold"; mode = "hold"; ground_refused = true;
            }
          }
          // ★stand↔walk 상호 전이는 진입 재실행 **금지** (08-27 검증 워크플로 확정 HIGH):
          //   재실행하면 블렌드 재시작(bs 1→0)으로 WBIC 지지토크가 한 틱에 소멸하고
          //   kd/kp 가 계단 점프, c.reset()·u_prev=0 이 지연보상 예측까지 오염한다.
          //   접지가드(위)와 같은 prev_mode 배제. 1점 gait 는 stand 도 제자리 스텝핑이라
          //   컨트롤러 연속이 곧 올바른 의미고, 2점 정적지지는 매 틱 자체 재무장한다.
          // ★stand 폭주 래치 — off 재무장 전까지 stand 재진입 거부 (2026-09-03)
          //   ⚠배너는 1s 로 솎는다 — GUI 가 20ms 재발행이라 안 그러면 초당 50줄 스팸.
          if(mode=="stand" && stand_runaway){
            static double rw_warn_t = -1e9;
            if(lt - rw_warn_t > 1.0){ rw_warn_t = lt;
              std::printf("[deploy] ⛔ STOP **stand 폭주 래치** — off 재무장 전까지 stand 거부, hold 유지.\n"
                          "         (직전 stand 에서 기준자세 이탈 폭주 — 모델 CoM/IMU 문제를 먼저 볼 것)\n"); }
            if(prev_mode!="hold"){ hold_ch = hs.q_deg; jm.clamp_ch_via_joint(hold_ch.data()); }
            nm = "hold"; mode = "hold";
          }
          // ★토크 불응답 래치가 있으면 stand/walk 를 막는다 (2026-08-28)
          if((mode=="stand" || mode=="walk")){
            std::string bad;
            for(int i=0;i<NCH;i++) if(unresp[i]){ bad += " " + chname[i]; }
            if(!bad.empty()){
              std::printf("[deploy] ⛔ STOP **토크 불응답 채널이 있다**:%s — stand/walk 거부, hold 유지.\n"
                          "         그 축은 체중을 못 받친다(주저앉는다). off 재무장/Emb 재기동 후 재시도.\n",
                          bad.c_str());
              if(prev_mode!="hold"){ hold_ch = hs.q_deg; jm.clamp_ch_via_joint(hold_ch.data()); }
              nm = "hold"; mode = "hold"; ground_refused = true;
            }
          }
          if((mode=="stand" || mode=="walk") && prev_mode!="stand" && prev_mode!="walk"){
            stand_hold = hs.q_deg; jm.clamp_ch_via_joint(stand_hold.data());
            // ★★2026-08-21 블렌드 목표를 **측정각에 얼리지 않는다** — 기하 자세로 램프한다.
            //   종전엔 진입 순간 측정각(`stand_hold`)을 블렌드 내내 목표로 썼다. 그런데 그 값은
            //   **하중 상태에 따라 반대로 편향**된다:
            //       매달림 : PD 처짐만큼 아래로   (hip 5.247Nm/kp100 = 3.0° · thigh 2.6° …)
            //       접지   : 지면반력에 밀려 위로
            //   같은 의도 자세인데 언제 눌렀느냐로 hip 수 도가 갈리고, 그 편향이 블렌드 초반
            //   **kp 가 아직 살아 있는 구간**의 서보 목표가 된다.
            //   ⇒ 진입 시엔 측정각(계단 0), 블렌드가 끝날 때는 **Qflat8/Qhome8**(기하 진리).
            //     같은 smoothstep 계수를 쓰므로 추가 튜닝이 없다.
            //   ⚠목표를 처음부터 Qflat8 로 못 박으면 안 된다 — 로봇이 거기 없을 때 계단이 된다
            //     (0° 자세에서 발목 채널 100.4° 요구 → 속도트립. `home` 이 생긴 이유다).
            {
              std::vector<double> qt(NJ);
              for(int j=0;j<NJ;j++) qt[j] = (c.cmode==1 ? c.Qflat8[j] : c.Qhome8[j]);
              jm.q_ctrl_to_ch(qt.data(), stand_to.data());
              jm.clamp_ch_via_joint(stand_to.data());
            }
            // ★2.5초 (2026-08-20). 1.0초로는 ch7 이 202dps 로 임계를 1% 넘겼다.
            //   ⚠발목 채널은 구조상 제일 잘 걸린다 — raw각이 (foot+calf) 라
            //     채널속도 = (q̇_foot + q̇_calf)×1.2 로 **두 관절의 합**이 잡힌다.
            //     관절로는 작은 움직임도 채널로는 2배 넘게 보인다.
            stand_T = getenv("STAND_BLEND_S") ? atof(getenv("STAND_BLEND_S")) : 2.5;
            stand_t0 = lt;
            std::printf("[deploy] stand 진입 — 위치제어→토크 **%.1fs 블렌드**"
                        "(계단 전환은 주저앉는다)\n", stand_T);
            // ★★c.reset() 전에 **측정 자세를 모델에 주입**한다 (2026-08-20 실기).
            //   reset() 은 d->qpos 를 읽어 com_ref_xy · nominal_off · com_ref_z 를 잡는다.
            //   그런데 hold/home 분기는 d 에 아무것도 주입하지 않는다 — 그래서 그때까지
            //   d->qpos 는 **모델 기본자세**다. 그 상태로 reset 하면 WBIC 의 CoM 목표가
            //   실제 로봇과 몇 cm 어긋나고, 무장 순간 그 오차를 지우려 다리가 튄다
            //   (실기: 블렌드 중 ch2 204dps 트립. 실측 자세는 목표와 hip 7° · foot 12° 달랐다).
            for(int j=0;j<NJ;j++){ d->qpos[7+j]=q_ctrl[j]; d->qvel[6+j]=0.0; }
            d->qpos[0]=d->qpos[1]=0.0; d->qpos[2]=0.5;
            for(int a=0;a<4;a++) d->qpos[3+a]=quat[a];
            for(int i=0;i<6;i++) d->qvel[i]=0.0;
            mj_forward(m,d);
            c.reset();
            // ★2점 stand 는 **지금 높이를 유지**한다 — body_h(GUI 슬라이더)로 덮지 않는다.
            //   reset() 이 현재 CoM 높이를 잡아주는데 그걸 0.38/0.5 로 덮으면 무장 즉시
            //   수 cm 를 올리거나 내리려 든다(kp_z=200). 서 있는 걸 유지하는 게 목적이지
            //   높이를 바꾸는 게 아니다. 높이를 바꾸려면 STAND_H 로 명시할 것.
            if(c.cmode!=1) c.com_ref_z = body_h;
            else if(getenv("STAND_H")) c.com_ref_z = atof(getenv("STAND_H"));
            // ★밑창중심과 얼마나 어긋나 있는지 같이 찍는다. reset() 은 **지금 CoM 을 목표로**
            //   잡으므로(=움직이지 않음) 안전하지만, 그 지점이 지지면 밖이면 stand 해도 넘어진다.
            //   여유는 밑창 반길이 7.3cm 다 — 이 숫자가 그보다 크면 stand 를 걸면 안 된다.
            { Eigen::Vector3d fc = 0.5*(c.foot_center(0)+c.foot_center(1));
              const double ex = c.com_ref_xy[0]-fc[0], ey = c.com_ref_xy[1]-fc[1];
              std::printf("[deploy] stand 기준 — CoM 목표 xy(%.3f, %.3f) z %.3f (측정자세 = 지금 자세 유지)\n"
                          "[deploy]   밑창중심 대비 전후 %+.1f cm · 좌우 %+.1f cm  %s\n",
                          c.com_ref_xy[0], c.com_ref_xy[1], c.com_ref_z, ex*100, ey*100,
                          std::fabs(ex)<0.05 ? "✓지지면 안" : "⚠지지면 여유가 적다(반길이 7.3cm)"); }
            est.reset(Eigen::Vector3d(0,0,d->qpos[2]));
            // ★in-flight 토크도 같이 지운다. 안 지우면 직전 세션의 마지막 토크로
            //   첫 틱을 예측한다 — 무장 순간이 가장 위험한 자리다.
            std::fill(u_prev.begin(), u_prev.end(), 0.0);
            lc_n = lc_skip = 0; lc_warned = false;
          }
          std::printf("[deploy] 모드 %s → %s\n", prev_mode.c_str(), mode.c_str());
        }
        c.vx_cmd = (mode=="walk") ? cmd.v  : 0.0;
        c.vy_cmd = (mode=="walk") ? cmd.vy : 0.0;
        c.wz_cmd = (mode=="walk") ? cmd.w  : 0.0;
      }
    }

    // ★stand 폭주 래치 해제 — off 재무장 규약. 전이 분기가 아니라 **매 틱** 본다
    //   (전이 분기 안에 두면 off 처리 경로에 따라 안 타는 수가 있다 — sim 검증에서 발각).
    if(mode=="off" && stand_runaway) stand_runaway = false;

    // ③ 워치독 — 명령 두절이면 limp. 전이를 반드시 출력한다(데드코드 방지).
    // ★★단 **하중 실린 hold 는 면제** (2026-09-03 실기 사고).
    //   크레인 없이 자립 중일 때 GUI 렌더가 1.16s 멈춰 발행이 끊겼고, 워치독이
    //   로봇을 limp 로 떨궜다 — **자립 단계에서 limp = 낙하**다.
    //   워치독의 존재 이유는 "폭주 방지"인데, hold 는 고정 목표 정적 PD 라 명령이
    //   끊겨도 폭주할 것이 없다. 서서 기다리는 것이 떨어지는 것보다 안전하다.
    //   ⇒ hold + 지지율>5% 면 limp 대신 **그 자리를 유지**하고 경고만 찍는다.
    //     명령이 복귀하면 그대로 재개된다(모드가 안 바뀌었으므로 가드 우회도 없다).
    //     GUI 를 영영 못 살리면: 크레인 걸고 배포기 Ctrl+C(=안전종료 시퀀스).
    //   WD_HOLD_LIMP=1 로 종전 동작(무조건 limp)을 되돌릴 수 있다.
    const bool wd_hold_immune = mode=="hold" && hold_ff_pct_cur > 5.0
                             && !(getenv("WD_HOLD_LIMP") && atoi(getenv("WD_HOLD_LIMP")));
    bool wd = (mode!="off") && (lt - last_cmd_t) > watchdog_s && !wd_hold_immune;
    { static bool imm_warned = false;
      const bool silent = (lt - last_cmd_t) > watchdog_s;
      if(wd_hold_immune && silent && !imm_warned){
        imm_warned = true;
        std::printf("[deploy] ⚠명령 두절 %.2fs — **하중 hold 라 limp 하지 않는다**(자립 낙하 방지).\n"
                    "         자세를 유지하며 명령 복귀를 기다린다. GUI 를 살릴 것.\n", lt-last_cmd_t);
        std::fflush(stdout);
      } else if(!silent) imm_warned = false; }
    if(wd != wd_tripped){
      wd_tripped = wd;
      if(wd || lt-last_cmd_t < watchdog_s)   // ★off 강하로 wd 가 꺼진 직후의 허위 '해제' 억제
        std::printf(wd ? "[deploy] 워치독 트립 — 명령 두절 %.2fs > %.2fs → limp\n"
                       : "[deploy] 워치독 해제 — 명령 복귀 (%.2f/%.2f)\n",
                    lt-last_cmd_t, watchdog_s);
      if(wd && mode!="off"){
        // ★검토 #1 (2026-08-27): 종전엔 명령이 복귀하면 같은 틱에 무가드 재무장 —
        //   정지·접지·자세거리 가드와 blend 재시작·c.reset() 이 전부 우회돼, limp 중
        //   처진 자세에 전 게인·완성 토크가 계단으로 걸렸다(낙하 중 재무장 사고 경로).
        //   ⇒ E-stop 과 같은 급으로 off 강하 + 기동잠금 규약 재사용: 재무장은
        //     off→X 정상 전이(모든 가드)를 밟고, 운전자가 다른 모드를 한 번 골라야 풀린다.
        mode = "off";
        hw->enable(0);            // mode 를 먼저 off 로 내리면 아래 enable 게이트를 안 타므로 명시 limp
        boot_mode = (cmd.mode=="reset") ? "hold" : cmd.mode;   // 정제값 저장(reset 매핑 정합)
        mode_locked = true;
        std::printf("[deploy]   ★→ off 강하 + 명령 잠금 — 버튼을 다시 눌러 재무장한다(가드 경유)\n");
      }
      std::fflush(stdout);
    }
    if(mode!="off") hw->enable(wd ? 0 : 1);

    // ④ E-stop (tilt / 토크 / 속도) — 전부 래치. biped_emb.py 이식.
    if(!estop && mode!="off" && tilt > cfg.tilt_estop_deg){
      std::printf("[deploy] ⛔ E-STOP: tilt %.0f° > %.0f° → limp·래치\n", tilt, cfg.tilt_estop_deg);
      estop = true; mode = "off"; hw->enable(0);
    }
    if(!estop && mode!="off"){
      double tau_pk=0, vel_pk=0; int tch=0, vch=0;
      for(int i=0;i<NCH;i++){
        if(std::fabs(hs.tau_nm[i])>tau_pk){ tau_pk=std::fabs(hs.tau_nm[i]); tch=i; }
        if(std::fabs(hs.dq_dps[i])>vel_pk){ vel_pk=std::fabs(hs.dq_dps[i]); vch=i; }
      }
      // ★walk 한정 상향 트립(08-27 묶음) + **이탈 0.5s 유예**(08-27 검증 확정 HIGH):
      //   이탈 틱에 잔류 스윙속도(≤673dps)와 kd 복원 제동이 남아 있는데 cfg 200dps/15Nm 로
      //   즉시 복원하면 "walk 끄기 = limp 낙하" 가 된다. 유예 동안 감쇠가 속도를 죽인다
      //   (τ=kd·q̇ 자기제한 — 수십 ms 면 붕괴). off/estop 은 이 블록 자체를 안 탄다.
      static double walk_last_t = -1e9;
      if(mode=="walk") walk_last_t = lt;
      const bool walk_win = (mode=="walk") || (lt - walk_last_t < 0.5);
      const double tau_trip_eff = walk_win ? WALK_TAU_TRIP : cfg.tau_trip_nm;
      const double vel_trip_eff = walk_win ? WALK_VEL_TRIP : cfg.vel_trip_dps;
      // 토크는 **연속 초과**만 트립(착지 충격 같은 순간 스파이크를 살린다)
      if(tau_pk > tau_trip_eff){
        if(tau_over_t0 < 0) tau_over_t0 = lt;
        else if((lt-tau_over_t0)*1000.0 >= cfg.tau_trip_ms){
          std::printf("[deploy] ⛔ E-STOP: ch%d 토크 %.2fNm > %.2fNm(%s) 가 %.0fms 연속 → limp·래치\n",
                      tch, tau_pk, tau_trip_eff, walk_win?"walk":"cfg", cfg.tau_trip_ms);
          estop = true; mode="off"; hw->enable(0);
        }
      } else tau_over_t0 = -1;
      // ★★속도도 **연속 초과**만 트립한다 (2026-08-24. 종전엔 즉시였다).
      //   ⚠왜 바꿨나: `vel_trip_ms`(20ms)가 config 에 있는데 **C++ 은 파싱조차 안 했다** —
      //     Python(biped_emb.py:555)만 디바운스를 걸고 배포는 단일 샘플로 끊었다.
      //     같은 config 를 읽는 두 writer 가 다르게 동작하는 것 자체가 결함이다.
      //   ⚠실무에서 물린다: **무중력(float) 모드는 사람이 손으로 다리를 민다.**
      //     채널 200dps 는 thigh 기준 관절 200dps(calf 133 · foot 167)라 조금 빠르게 밀면 넘는다.
      //     그 결과가 limp = **다리가 떨어진다.** 잡아 주려던 장치가 떨어뜨리는 셈이다.
      //   ⚠그래도 20ms 는 짧게 둔다 — 진짜 폭주를 지연시키면 안 된다.
      //     20ms 면 200dps 에서 4° 이동이고, 트립각 7.16°(calf) 안이다.
      if(!estop && vel_pk > vel_trip_eff){
        if(vel_over_t0 < 0) vel_over_t0 = lt;
        else if((lt-vel_over_t0)*1000.0 >= cfg.vel_trip_ms){
          std::printf("[deploy] ⛔ E-STOP: ch%d 속도 %.0fdps > %.0fdps(%s) 가 %.0fms 연속 → limp·래치\n",
                      vch, vel_pk, vel_trip_eff, walk_win?"walk":"cfg", cfg.vel_trip_ms);
          estop = true; mode="off"; hw->enable(0);
        }
      } else vel_over_t0 = -1;
    } else { tau_over_t0 = -1; vel_over_t0 = -1; }
    if(estop) hw->enable(0);

    // (트레이스 기록은 **모드 분기 뒤로** 옮겼다 — 아래 "트레이스 기록" 참조)

    // ★강성 배율 램프 — 목표까지 KP_RAMP_S 에 걸쳐 **선형**으로 옮긴다.
    //   계단으로 바꾸면 하중을 받아 err 만큼 벌어진 축의 토크가 그 자리에서 배율만큼
    //   튄다(τ=kp·err). 접지 중이면 그게 곧 τ_trip 이다.
    // kd 배율: 명령 > env > 자동(√kp). kp 램프에 맞춰 매 틱 다시 계산한다.
    POS_KD = (kd_scale_cmd>=0) ? kd_scale_cmd
           : (KD_SCALE_ENV>=0) ? KD_SCALE_ENV
           : std::sqrt(std::max(1e-9, POS_KP));
    if(POS_KP != kp_scale_tgt){
      const double step = (KP_RAMP_S>1e-6) ? (KP_SCALE_MAX/KP_RAMP_S)*dt : 1e9;
      double d0 = kp_scale_tgt - POS_KP;
      POS_KP += std::max(-step, std::min(step, d0));
      if(std::fabs(kp_scale_tgt - POS_KP) < 1e-6) POS_KP = kp_scale_tgt;
      POS_KD = (KD_SCALE_ENV>=0) ? KD_SCALE_ENV : std::sqrt(std::max(1e-9, POS_KP));
    }

    // ★자세유지 토크 — 0.5초 창의 **평균**을 굴린다. 한 틱 값은 마찰·양자화로 튄다.
    {
      for(int i=0;i<NCH;i++) tau_acc[i] += (double)hs.tau_nm[i];
      tau_n++;
      if(lt - tau_win_t0 >= 0.5 && tau_n>0){
        for(int i=0;i<NCH;i++){ tau_avg[i] = tau_acc[i]/tau_n; tau_acc[i]=0.0; }
        tau_n=0; tau_win_t0=lt;
      }
    }

    // ⑤ 모드 디스패치
    if(mode=="off"){
      for(int i=0;i<NCH;i++) q_ch[i]=0.f;
      qcmd_ch = q_ch; kpcmd_ch = zero; kdcmd_ch = zero;
      hw->write_pos(q_ch.data(), zero.data(), zero.data(), NCH);     // enable=0 → 브리지가 0 토크
    } else if(mode=="hold"){
      jm.kp_ch(kp_ch.data(), POS_KP); jm.kd_ch(kd_ch.data(), POS_KD);
      qcmd_ch = hold_ch; kpcmd_ch = kp_ch; kdcmd_ch = kd_ch;
      // 지지율 램프 — 계단 금지. 목표는 GUI(hold_ff_pct), 0 이면 종전 경로 그대로 간다.
      //   ⚠비율은 **전방보상 전체**에 곱한다(중력항 포함). 중력항만 통째로 켜면
      //     램프를 시작하는 순간 관절당 5 Nm 계단이 걸린다 — 목업에서 실제로 나왔다.
      //   ⇒ 0% = 종전 hold 와 **완전히 동일** · 100% = 중력 + 몸무게 절반 전량.
      { const double tgt = std::max(0.0, std::min(100.0, cmd.hold_ff_pct));
        const double stp = HOLD_FF_RATE * dt;
        hold_ff_pct_cur += std::max(-stp, std::min(stp, tgt - hold_ff_pct_cur));
        hold_fz_cur = 0.01 * hold_ff_pct_cur * HOLD_FF_FULL_N; }
      if(hold_ff_pct_cur <= 1e-6){
        hw->write_pos(hold_ch.data(), kp_ch.data(), kd_ch.data(), NCH);
      } else {
        // ★중력 전방보상 — 자세는 그대로 hold_ch 로 잡고 τ_ff 만 얹는다(위 상수 주석 참조).
        //   베이스 고정 + qvel=0 ⇒ qfrc_bias 는 코리올리 없는 **순수 중력항**. IMU 불필요.
        for(int j=0;j<NJ;j++){ d->qpos[7+j]=q_ctrl[j]; d->qvel[6+j]=0.0; }
        d->qpos[0]=d->qpos[1]=0; d->qpos[2]=0.5;
        d->qpos[3]=1; d->qpos[4]=d->qpos[5]=d->qpos[6]=0;
        for(int i=0;i<6;i++) d->qvel[i]=0.0;
        mj_forward(m,d);
        for(int j=0;j<NJ;j++) tau_ctrl[j] = d->qfrc_bias[6+j];      // 중력항(무배율 — 아래서 일괄)
        // 몸무게를 **양 다리에** 건다 — push 와 달리 한쪽이 아니다.
        //   부호는 push 와 같다: τ = G + Jᵀ·(0,0,−F) 가 "발이 F 로 딛고 반작용 F↑ 를 받는다".
        //   ⚠여기선 **전량**을 넣는다 — 지지율은 아래에서 한 번만 곱한다.
        // ★좌우 배분 (2026-09-02 실기 1차): 50:50 고정이었더니 HR_foot 이 −3.3° 과보상
        //   = 실제 하중이 HL 쏠림. GUI 트림(hold_ff_split, % to HL)을 2%/s 로 램프해 따른다.
        { const double tgt_sp = std::max(20.0, std::min(80.0, cmd.hold_ff_split));
          const double stp = 2.0 * dt;
          hold_split_cur += std::max(-stp, std::min(stp, tgt_sp - hold_split_cur)); }
        { static std::vector<mjtNum> jacp;
          for(int leg=0; leg<2; leg++){
            // 총하중 mg = 2·FULL_N. HL 몫 = split%, HR 몫 = 100−split%.
            const double frac = (leg==0) ? 0.01*hold_split_cur : 0.01*(100.0-hold_split_cur);
            const double F_leg = 2.0 * HOLD_FF_FULL_N * frac;
            const int gs[2] = { c.sph[leg], (HOLD_FF_MID && c.has_heel) ? c.sph2[leg] : -1 };
            const double share = (gs[1] >= 0) ? 0.5 : 1.0;
            for(int k=0;k<2;k++){
              if(gs[k] < 0) continue;
              jacp.assign(3*m->nv, 0.0);
              mj_jacGeom(m, d, jacp.data(), nullptr, gs[k]);
              for(int j=0;j<NJ;j++)
                tau_ctrl[j] += jacp[2*m->nv + 6+j] * (-F_leg*share);
            }
          } }
        // ★지지율 × **축별 배율**을 여기서 한 번에 곱한다 — 계단 금지(램프 hff)와
        //   "명령 = 필요토크/전달비" 를 같은 자리에서 처리한다.
        //   foot 만 g* 대신 HOLD_FF_FOOT(1.30) — 하중 대역에선 비율부 0.77 이 지배(상단 주석).
        const double hff = 0.01 * hold_ff_pct_cur;
        for(int j=0;j<NJ;j++){
          const double sc = (j % 4 == 3) ? HOLD_FF_FOOT
                          : (grav_axis[j] >= 0.0 ? grav_axis[j] : GRAV_SCALE);
          tau_ctrl[j] = std::max(-HOLD_FF_TAU_MAX, std::min(HOLD_FF_TAU_MAX, hff*sc*tau_ctrl[j]));
        }
        jm.tau_ctrl_to_ch(tau_ctrl.data(), tau_ch.data());
        foot_comp(tau_ch);
        write_ff(hold_ch.data(), tau_ch, kp_ch.data(), kd_ch.data());   // ★α 보정 FF
        // ★1초 진단 — 브링업 중엔 항상 찍는다(지지율 0 이면 이 가지에 안 들어온다).
        //   읽는 법: 지지율을 올릴수록 **오차가 줄고 τ_ff 가 그만큼 커져야** 정상이다.
        //   오차가 안 줄면 그 축은 명령을 못 따르는 것(구동 결손·기구 걸림)이다.
        { static double last_hff = -1e9;
          if(lt - last_hff > 1.0){ last_hff = lt;
            double emx=0; int ech=-1;
            for(int i=0;i<NCH;i++) if(cfg.installed_has(i)){
              const double e=(double)hold_ch[i]-(double)hs.q_deg[i];
              if(std::fabs(e)>std::fabs(emx)){ emx=e; ech=i; } }
            std::printf("[hold] 지지 %.0f%%(%.1fN/다리) · 배분 HL%.0f:%.0fHR · tau_ff:",
                        hold_ff_pct_cur, hold_fz_cur, hold_split_cur, 100.0-hold_split_cur);
            for(int j=0;j<NJ;j++) std::printf(" %+.2f", tau_ctrl[j]);
            // ★축별 오차도 같이 (2026-09-04 사용자 요청) — 최대값 하나로는 좌우 비대칭·
            //   CoM 역산·유격 서명을 터미널 로그만으로 분석할 수 없었다.
            std::printf("  최대 %s %+.2f°\n       오차(ch):", ech>=0?chname[ech].c_str():"-", emx);
            for(int i=0;i<NCH && i<8;i++)
              std::printf(" %+5.1f", (double)hold_ch[i]-(double)hs.q_deg[i]);
            std::printf("\n");
            std::fflush(stdout); } }
        // ★★하중 hold 세션 로거 (2026-09-04) — 정착 구간 데이터가 어디에도 안 남던
        //   구멍을 막는다: arm_trace 는 전환 3초뿐, 상태 JSON 은 아무도 저장 안 했다.
        //   지지율 >5% 인 동안 10Hz 로 통짜 기록. 기동마다 새로 쓴다(truncate).
        //   열: t · 지지% · 배분 · q(채널°)8 · cmd(채널°)8 · tau측정(채널Nm)8 · ff(관절Nm)8
        //       [+ AUX_MODE 시 aux(출력축°)8] — 있을 때만 헤더/열이 붙는다.
        //   ★aux 가 왜 여기 필요한가 (2026-09-04): 지금 미지수 = foot 전달비 열화(r 0.5~0.65).
        //     aux 는 그걸 직접 가른다 — calf/foot 은 aux=벨트 앞단이라 (aux/gear_k − 링크각)이
        //     벨트 유격, hip/thigh 는 aux=관절각이라 주엔코더와의 차 = 7:1 감속단 비틀림.
        //     벨트 미끄러짐이면 aux 정상·링크만 처짐 / 드라이브 열화면 aux 도 같이 처진다.
        { static FILE* hsf = nullptr; static double hs_last = 0; static bool hs_warn = false;
          if(hold_ff_pct_cur > 5.0 && lt - hs_last > 0.1){
            hs_last = lt;
            float aux_p[16], aux_v[16];
            const bool aux_on = (hw->aux(aux_p, aux_v) == 1);   // 1=AUX_MODE 켜져 값 유효
            if(!hsf){
              hsf = std::fopen("/tmp/hold_session.csv", "w");
              if(hsf){
                std::fprintf(hsf, "t,pct,split");
                for(int i=0;i<8;i++) std::fprintf(hsf, ",q%d", i);
                for(int i=0;i<8;i++) std::fprintf(hsf, ",cmd%d", i);
                for(int i=0;i<8;i++) std::fprintf(hsf, ",tau%d", i);
                for(int i=0;i<8;i++) std::fprintf(hsf, ",ff%d", i);
                if(aux_on) for(int i=0;i<8;i++) std::fprintf(hsf, ",aux%d", i);
                std::fprintf(hsf, "\n");
                std::printf("[hold] 세션 로거 → /tmp/hold_session.csv (10Hz · 지지율>5%%%s)\n",
                            aux_on ? " · aux 포함" : " · aux 없음(AUX_MODE=1 로 켤 것)");
              } else if(!hs_warn){ hs_warn=true; std::printf("[hold] ⚠세션 로거 열기 실패\n"); }
            }
            if(hsf){
              std::fprintf(hsf, "%.3f,%.1f,%.1f", lt, hold_ff_pct_cur, hold_split_cur);
              for(int i=0;i<8;i++) std::fprintf(hsf, ",%.3f", (double)hs.q_deg[i]);
              for(int i=0;i<8;i++) std::fprintf(hsf, ",%.3f", (double)hold_ch[i]);
              for(int i=0;i<8;i++) std::fprintf(hsf, ",%.3f", (double)hs.tau_nm[i]);
              for(int i=0;i<8 && i<NJ;i++) std::fprintf(hsf, ",%.3f", tau_ctrl[i]);
              if(aux_on) for(int i=0;i<8;i++) std::fprintf(hsf, ",%.3f", (double)aux_p[i]);
              std::fprintf(hsf, "\n"); std::fflush(hsf);
            }
          } }
      }
    } else if(mode=="soft_off"){
      double u = (soft_T>0) ? (lt-soft_t0)/soft_T : 1.0;
      u = std::max(0.0, std::min(1.0, u));
      const double sf = u*u*u*(10.0 - 15.0*u + 6.0*u*u);      // home 과 같은 5차 S-curve
      for(int i=0;i<NCH;i++)
        q_ch[i] = soft_from[i] + (float)(sf*(double)(soft_to[i]-soft_from[i]));
      jm.kp_ch(kp_ch.data(), POS_KP); jm.kd_ch(kd_ch.data(), POS_KD);
      qcmd_ch = q_ch; kpcmd_ch = kp_ch; kdcmd_ch = kd_ch;
      hw->write_pos(q_ch.data(), kp_ch.data(), kd_ch.data(), NCH);
      if(u>=1.0 && !soft_done && lt-soft_t0 > soft_T + 1.5){    // 1.5s 정착 후 무여자
        soft_done = true; soft_latched = true;
        mode = "off"; boot_mode = "off"; mode_locked = false;
        hw->enable(0);
        std::printf("[deploy] ✅ 안전 종료 완료 — 무여자. 배포기를 꺼도 된다.\n");
      }
    } else if(mode=="home"){
      // ★5차 S-curve s(τ)=10τ³−15τ⁴+6τ⁵ — biped_emb.py control/home.py 와 **같은 식**.
      //   경계 속도·가속도가 둘 다 0 이라 진입/도착에서 토크가 계단으로 안 튄다.
      double u = (home_T>0) ? (lt-home_t0)/home_T : 1.0;
      u = std::max(0.0, std::min(1.0, u));
      const double sf = u*u*u*(10.0 - 15.0*u + 6.0*u*u);
      for(int i=0;i<NCH;i++)
        q_ch[i] = home_from[i] + (float)(sf*(double)(home_to[i]-home_from[i]));
      jm.kp_ch(kp_ch.data(), POS_KP); jm.kd_ch(kd_ch.data(), POS_KD);
      qcmd_ch = q_ch; kpcmd_ch = kp_ch; kdcmd_ch = kd_ch;
      hw->write_pos(q_ch.data(), kp_ch.data(), kd_ch.data(), NCH);
      if(u>=1.0 && !home_done){
        home_done = true;
        std::printf("[deploy] home 도달 — 그 자세로 유지 중.\n"
                    "         다음: 크레인을 내려 접지 → **hold** 로 하중 이양 → stand\n");
      }
      // ★★**도달했는지 실제로 본다** (2026-08-21 신설, biped_emb.py 의 at_goal 과 같은 계약).
      //   종전 "home 도달" 은 **시간이 다 됐다**는 뜻일 뿐이었다. 그래서 HR_hip 이 두 번
      //   연속 **0° 움직이고** 끝났는데도 화면엔 "도달" 이라고만 떴다(같은 +11.22° 잔차).
      //   ⇒ 시간이 끝난 뒤 0.3s 정착을 주고, 측정각이 목표에서 settle 밖이면 **한 번만** 경고.
      //   ⚠경고일 뿐 정지시키지 않는다 — 매달린 채로 처지는 건 정상이고, 여기서 끊으면
      //     운전자가 다음 단계로 못 간다. 판단은 사람이 한다.
      if(home_done && !home_warned && lt-home_t0 > home_T + 0.3){
        home_warned = true;
        std::string bad; double wmax=0;
        for(int i=0;i<NCH;i++) if(cfg.installed_has(i)){
          double e = (double)hs.q_deg[i] - (double)home_to[i];
          if(std::fabs(e) > cfg.home_settle_deg){
            char b[64]; std::snprintf(b,sizeof b," %s%+.1f°", chname[i].c_str(), e);
            bad += b; wmax = std::max(wmax, std::fabs(e));
          }
        }
        if(!bad.empty())
          std::printf("[deploy] ⚠ home **도달 실패**(허용 %.1f°, 최대 %.1f°):%s\n"
                      "         0° 근처 잔차면 부하 처짐이다. **이동량만큼 그대로 남았으면\n"
                      "         그 축은 구동이 안 된 것이다** — 모니터에서 토크를 볼 것.\n",
                      cfg.home_settle_deg, wmax, bad.c_str());
      }
    } else if(mode=="jog"){
      // ★등속 램프로 **살아 있는 목표**를 따라간다. 목표는 GUI 의 jog_deg(모델각 deg).
      //   jog_deg 가 없으면(옛 GUI) 현재 램프값을 유지 — 갑자기 0 으로 끌려가지 않게.
      if(!jog_init){ jm.ch_to_q_joint(hs.q_deg.data(), jog_q.data()); jog_prev_t = lt; jog_init = true; }
      double el = lt - jog_prev_t; jog_prev_t = lt;
      el = std::max(0.0, std::min(el, 0.05));      // ⚠긴 정지 후 급이동 방지(Jogger.DT_CAP 과 동일)
      const double step = cfg.jog_speed_dps * el;
      if((int)cmd.jog_deg.size() >= NJ){
        for(int j=0;j<NJ;j++){
          double d0 = cmd.jog_deg[j] - jog_q[j];
          jog_q[j] += std::max(-step, std::min(step, d0));
        }
      }
      std::vector<double> qj = jog_q;
      jm.clamp_joint(qj.data());                   // 관절한계(모델각) — 채널이 아니라 여기서
      jm.q_joint_to_ch(qj.data(), q_ch.data());
      jm.kp_ch(kp_ch.data(), POS_KP); jm.kd_ch(kd_ch.data(), POS_KD);
      qcmd_ch = q_ch; kpcmd_ch = kp_ch; kdcmd_ch = kd_ch;
      hw->write_pos(q_ch.data(), kp_ch.data(), kd_ch.data(), NCH);
    } else if(mode=="float"){
      // ★★무중력(중력보상). **매달린 상태 전용** — 접지 중에는 위에서 거부한다.
      //   식:  τ_cmd = GRAV_SCALE · G_model(q_meas)      (kp=0, kd=FLOAT_KD)
      //   드라이버가 α 를 곱하므로 실제로는 α·GRAV_SCALE·G_CAD 가 나간다.
      //   ⇒ 다리가 안 뜨고 안 지는 중립 배율 g* 에서  α·g*·G_CAD = G_real.
      //   ⚠쿨롱마찰(관절 0.60~0.87 Nm)은 **안 지워진다** — 중력만 상쇄한다.
      //     그래서 "놓은 자리에 서 있지만 밀 때는 뻑뻑한" 상태가 된다. 그 폭이 곧
      //     중립점 브래킷의 데드밴드이고, 양방향 평균이 그걸 소거한다.
      for(int j=0;j<NJ;j++){ d->qpos[7+j]=q_ctrl[j]; d->qvel[6+j]=0.0; }
      // ★베이스를 **고정**한다 — 크레인에 매달려 있으므로 부동베이스 추정이 무의미하고,
      //   IMU 도 죽어 있다. qvel=0 이라 qfrc_bias 는 코리올리 없이 **순수 중력항**이 된다.
      d->qpos[0]=d->qpos[1]=0; d->qpos[2]=0.5;
      d->qpos[3]=1; d->qpos[4]=d->qpos[5]=d->qpos[6]=0;
      for(int i=0;i<6;i++) d->qvel[i]=0.0;
      mj_forward(m,d);
      //   축별 배율이 있으면 그걸, 없으면 공통 GRAV_SCALE. GUI 의 배율은 공통 쪽에 실린다.
      for(int j=0;j<NJ;j++)
        tau_ctrl[j] = (grav_axis[j] >= 0.0 ? grav_axis[j] : GRAV_SCALE) * d->qfrc_bias[6+j];
      // ★관절토크 → 채널. tau_ctrl_to_ch 가 **커플링 전치**(τ_raw_calf = τ_calf − c·τ_foot)와
      //   gear_k 나눗셈을 같이 한다. 직접 나누면 발목에서 틀린다.
      jm.tau_ctrl_to_ch(tau_ctrl.data(), tau_ch.data());
      foot_comp(tau_ch);
      jm.kd_ch(kd_ch.data(), FLOAT_KD);
      for(int i=0;i<NCH;i++) kp_ch[i] = 0.f;
      if(getenv("FLOAT_DBG")){
        static int nprint=0;
        if(nprint < 8 && (nprint<4 || lt-mode_t0 > 0.5*nprint)){ nprint++;
          std::printf("[float] tau_joint(qfrc_bias):");
          for(int j=0;j<NJ;j++) std::printf(" %+.2f", tau_ctrl[j]);
          std::printf("\n        tau_ch          :");
          for(int j=0;j<NJ;j++) std::printf(" %+.2f", (double)tau_ch[cfg.joints[j].channel]);
          std::printf("\n        q_joint[deg]    :");
          for(int j=0;j<NJ;j++) std::printf(" %+.1f", q_ctrl[j]*JointMap::R2D);
          std::printf("\n        dq_ch[dps]      :");
          for(int j=0;j<NJ;j++) std::printf(" %+.1f", (double)hs.dq_dps[cfg.joints[j].channel]);
          std::printf("\n"); std::fflush(stdout); }
      }
      // ★축 선택 — FLOAT_AXES="1,5" 면 그 채널만 뜨고 나머지는 **진입 자세로 위치유지**.
      //   한 축씩 확인하는 것이 안전하다(전 축을 한꺼번에 놓으면 자세가 무너진다).
      if(!float_axes.empty()){
        for(int i=0;i<NCH;i++) if(!float_axes.count(i)){
          tau_ch[i] = 0.f; q_ch[i] = hold_ch[i];
          kp_ch[i] = cfg_kp_ch[i]; kd_ch[i] = cfg_kd_ch[i];
        }
      }
      for(int i=0;i<NCH;i++) if(float_axes.empty() || float_axes.count(i)) q_ch[i] = hs.q_deg[i];
      qcmd_ch = q_ch; kpcmd_ch = kp_ch; kdcmd_ch = kd_ch;
      write_ff(q_ch.data(), tau_ch, kp_ch.data(), kd_ch.data());   // ★α 보정 FF
    } else if(mode=="push"){
      // ★발밀기 — float 와 같은 골격(베이스 고정·중력보상) + 선택 다리 발끝에 Jᵀ·F.
      //   J 는 MuJoCo(mj_jacGeom)가 준다 — DH 수식을 복사하지 않는다(참조 구현
      //   motion_controller.py 의 compute_contact_torque 와 같은 τ=Jᵀ·F_ref 패턴).
      for(int j=0;j<NJ;j++){ d->qpos[7+j]=q_ctrl[j]; d->qvel[6+j]=0.0; }
      d->qpos[0]=d->qpos[1]=0; d->qpos[2]=0.5;
      d->qpos[3]=1; d->qpos[4]=d->qpos[5]=d->qpos[6]=0;
      for(int i=0;i<6;i++) d->qvel[i]=0.0;
      mj_forward(m,d);
      for(int j=0;j<NJ;j++)
        tau_ctrl[j] = (grav_axis[j] >= 0.0 ? grav_axis[j] : GRAV_SCALE) * d->qfrc_bias[6+j];
      // 힘 램프 — 계단 금지. 목표는 명령파일(GUI)에서, 상한은 env 에서.
      { const double tgt = std::max(0.0, std::min(PUSH_MAX, cmd.push_fz));
        const double stp = PUSH_RATE * dt;
        push_fz_cur += std::max(-stp, std::min(stp, tgt - push_fz_cur)); }
      // ★Jᵀ·(0,0,−F): 정역학 τ = G − Jᵀ·F_ext(로봇이 받는 반작용 +F↑)
      //   = G + Jᵀ·(0,0,−F). z-행 × (−F) 를 그 다리 관절에 더한다.
      //   (다른 다리 열은 트리 구조상 0 이라 전 축 더해도 무해)
      if(push_fz_cur > 1e-6){
        const int leg = (cmd.push_leg==1) ? 1 : 0;
        // ★PUSH_POINT=heel: 발목 원점 구(sphere2)로 민다 — 힘선이 발목을 지나
        //   발목토크 성분이 0 이 되어 foot 경로가 측정에서 원천 배제된다.
        //   (발끝밀기는 정역학상 뒤꿈치 힘 0 설계라 두 점 접촉이 유지 안 됨 —
        //    08-25 ② 스윕에서 뒤꿈치가 +6→+22mm 떠오른 것이 그 서명)
        int push_geom = c.sph[leg];
        { const char* pp = getenv("PUSH_POINT");
          if(pp && std::string(pp)=="heel"){
            if(c.has_heel) push_geom = c.sph2[leg];
            else { static bool warned=false;
                   if(!warned){ std::printf("[push] ⚠PUSH_POINT=heel 인데 heel 구 없는 모델 — 발끝으로 진행\n"); warned=true; } }
          } }
        static std::vector<mjtNum> jacp;
        jacp.assign(3*m->nv, 0.0);
        mj_jacGeom(m, d, jacp.data(), nullptr, push_geom);
        for(int j=0;j<NJ;j++)
          tau_ctrl[j] += jacp[2*m->nv + 6+j] * (-push_fz_cur);
      }
      // 관절 상한 — push 가 더한 몫이 폭주하지 않게 (중력보상 포함 총량 기준)
      for(int j=0;j<NJ;j++)
        tau_ctrl[j] = std::max(-PUSH_TAU_MAX, std::min(PUSH_TAU_MAX, tau_ctrl[j]));
      jm.tau_ctrl_to_ch(tau_ctrl.data(), tau_ch.data());
      foot_comp(tau_ch);
      jm.kd_ch(kd_ch.data(), FLOAT_KD);
      for(int i=0;i<NCH;i++) kp_ch[i] = 0.f;
      if(getenv("PUSH_DBG")){
        static double lastp=0;
        if(lt-lastp > 1.0){ lastp=lt;
          { const char* pp=getenv("PUSH_POINT");
            std::printf("[push] F=%.1fN(목표 %.1f) 다리 %s·%s · tau_joint:",
                      push_fz_cur, cmd.push_fz, cmd.push_leg==1?"HR":"HL",
                      (pp&&std::string(pp)=="heel")?"뒤꿈치":"발끝"); }
          for(int j=0;j<NJ;j++) std::printf(" %+.3f", tau_ctrl[j]);
          std::printf("  q[deg]:");   // ★같은 틱의 자세 — 독립 검증은 이 q 로 재계산해야 시각차가 없다
          for(int j=0;j<NJ;j++) std::printf(" %+.2f", q_ctrl[j]*JointMap::R2D);
          std::printf("\n"); std::fflush(stdout); }
      }
      for(int i=0;i<NCH;i++) q_ch[i] = hs.q_deg[i];
      qcmd_ch = q_ch; kpcmd_ch = kp_ch; kdcmd_ch = kd_ch;
      write_ff(q_ch.data(), tau_ch, kp_ch.data(), kd_ch.data());   // ★α 보정 FF
    } else {  // stand / walk — 모델기반
      // 접촉: 실기엔 발 힘센서가 없다. 게이트 위상(스탠스 다리)을 접촉으로 쓴다.
      //   ⚠추정에 쓰는 접촉이 제어기 자신의 계획이라 순환처럼 보이지만, 힘센서 없는
      //     운동학 오도메트리의 표준 관행이다. 힘센서가 생기면 여기를 교체할 것.
      std::vector<bool> cts(2,false);
      if(c.cmode==1){ cts[0]=cts[1]=true; }               // 평발 정적 = 양발
      else { cts[c.swing==0?1:0] = true; }                // 점발 = 스탠스 다리만
      est.estimate(m, q_ctrl.data(), dq_ctrl.data(), quat, gyro, cts, dt);

      // 모델에 주입 → mj_forward → 제어
      d->qpos[0]=est.p[0]; d->qpos[1]=est.p[1]; d->qpos[2]=est.p[2];
      for(int a=0;a<4;a++) d->qpos[3+a]=quat[a];
      for(int j=0;j<NJ;j++) d->qpos[7+j]=q_ctrl[j];
      d->qvel[0]=est.v[0]; d->qvel[1]=est.v[1]; d->qvel[2]=est.v[2];
      for(int a=0;a<3;a++) d->qvel[3+a]=gyro[a];
      for(int j=0;j<NJ;j++) d->qvel[6+j]=dq_ctrl[j];

      // ★지연보상 — 마지막 명령토크를 유지한 채 LCOMP step 굴려 "지금"을 만든다.
      //   실패하면 **조용히 실측으로 되돌린다**(예측을 안 쓸 뿐, 제어는 계속된다).
      if(LCOMP>0){
        lc_n++;
        std::vector<double> pq(m->nq), pv(m->nv);
        if(dpred){                                   // (a) 동역학 롤아웃 — in-flight 토크 유지
          mju_copy(dpred->qpos, d->qpos, m->nq);
          mju_copy(dpred->qvel, d->qvel, m->nv);
          mju_zero(dpred->act, m->na); dpred->time = d->time;
          for(int l=0;l<LCOMP;l++){
            for(int i=0;i<NU;i++) dpred->ctrl[i] = u_prev[i];
            mj_step(m, dpred);
          }
          mju_copy(pq.data(), dpred->qpos, m->nq);
          mju_copy(pv.data(), dpred->qvel, m->nv);
        } else {                                     // (b) 운동학 외삽 — q += q̇·T (사실상 공짜)
          const double TT = lat_comp_ms/1000.0;
          mju_copy(pq.data(), d->qpos, m->nq);
          mju_copy(pv.data(), d->qvel, m->nv);        // 속도는 그대로(가속도를 안 쓴다)
          for(int i=0;i<3;i++) pq[i] += d->qvel[i]*TT;             // base 위치
          mju_quatIntegrate(pq.data()+3, d->qvel+3, TT);           // base 자세 ← 자이로
          for(int j=0;j<NJ;j++) pq[7+j] += d->qvel[6+j]*TT;        // 관절
        }
        // 가드 ① 유한성 ② 관절각 튐 ③ 몸통고도 튐.
        //   8.4ms 는 200dps 로 돌아도 1.7° 다 — 5° 를 넘으면 예측이 깨진 것이다.
        bool ok = true;
        for(int i=0;i<m->nq && ok;i++) ok = std::isfinite(pq[i]);
        for(int i=0;i<m->nv && ok;i++) ok = std::isfinite(pv[i]);
        if(ok) for(int j=0;j<NJ;j++)
          if(std::fabs(pq[7+j]-d->qpos[7+j])*JointMap::R2D > LC_MAX_DEG){ ok=false; break; }
        if(ok && std::fabs(pq[2]-d->qpos[2]) > LC_MAX_Z) ok = false;
        if(ok){
          mju_copy(d->qpos, pq.data(), m->nq);
          mju_copy(d->qvel, pv.data(), m->nv);
        } else {
          lc_skip++;
          // 한 번은 반드시 알린다 — 조용히 폴백하면 "보상이 켜져 있다" 고 착각한다.
          if(!lc_warned){ lc_warned = true;
            std::fprintf(stderr, "[deploy] ⚠지연보상 예측이 가드에 걸렸다 — 실측상태로 폴백."
                                 " (|Δq|>%.1f° 또는 |Δz|>%.0fmm 또는 비유한)\n",
                         LC_MAX_DEG, LC_MAX_Z*1e3); }
        }
      }
      mj_forward(m,d);
      c.com_ref_z = body_h;
      c.control(dt);

      // 토크 → 채널. ★tau_max_frac 로 한 번 더 클램프(컨트롤러 내부 클립과 별개의 상위 안전망).
      // ★2026-08-13 두 가지가 바뀌었다.
      //   ① 한계 출처: jnt_actfrcrange(+actuator_trnid 조회) → **actuator ctrlrange**.
      //      종전 주석이 "모델이 바뀌면 조용히 틀린 축의 토크한계를 쓴다" 고 경고했는데,
      //      발목을 tendon 액추에이터로 옮긴 것이 정확히 그 모델 변경이다 —
      //      trnid 가 관절이 아니라 **tendon id** 를 돌려줘 엉뚱한 관절 한계를 읽게 된다.
      //      ctrlrange 는 액추에이터당 하나라 조회 자체가 없어진다.
      //   ② d->ctrl 은 이제 **드라이브 토크**다. 한계도 드라이브 기준이라 여기서 바로 자른다.
      VectorXd u_drv(NU);
      for(int i=0;i<NU;i++){
        double lim = (m->actuator_ctrllimited[i] ? m->actuator_ctrlrange[i*2+1] : 0.0) * cfg.tau_max_frac;
        if(lim<=0) lim = 80.0;
        u_drv[i] = std::max(-lim, std::min(lim, d->ctrl[i]));
      }
      // ★다음 틱의 롤아웃이 유지할 in-flight 토크. **클램프 뒤** 값이어야 한다 —
      //   실제로 나가는 게 이것이고, 예측이 클램프 전 값을 쓰면 모델이 로봇보다 세진다.
      for(int i=0;i<NU;i++) u_prev[i] = u_drv[i];
      // ★관절토크로 되돌려서 넘긴다 — joint_map(tau_ctrl_to_ch)이 **자기가 전단**하므로
      //   드라이브 토크를 그대로 주면 전단이 두 번 걸려 τ_calf−2·τ_foot 이 나간다.
      VectorXd tj = bipedwbic::drive_to_tau(u_drv);
      // ★★토크 보정 — **관절 좌표에서** 곱한다(드라이브에서 곱하면 발목 전단과 섞인다).
      //   ⚠곱한 뒤 **한계를 다시 건다.** 안 그러면 tau_max_frac 안전망을 우회한다.
      //   ⚠`u_prev`(지연보상 롤아웃용)는 **보정 전** 값을 쓴다 — 보정은 α 를 상쇄해
      //     실제 출력을 모델값으로 되돌리는 것이므로, 모델 예측에는 보정 전이 맞다.
      {
        bool any = (STAND_TAU_SCALE != 1.0) || (tau_axis[0] > 0);
        if(any){
          for(int i=0;i<NU && i<NJ;i++)
            tj[i] *= (tau_axis[i] > 0 ? tau_axis[i] : STAND_TAU_SCALE);
          VectorXd u2 = bipedwbic::tau_to_drive(tj);
          for(int i=0;i<NU;i++){
            double lim = (m->actuator_ctrllimited[i] ? m->actuator_ctrlrange[i*2+1] : 0.0) * cfg.tau_max_frac;
            if(lim<=0) lim = 80.0;
            u2[i] = std::max(-lim, std::min(lim, u2[i]));
          }
          tj = bipedwbic::drive_to_tau(u2);
        }
      }
      for(int i=0;i<NU;i++) tau_ctrl[i] = tj[i];
      jm.q_ctrl_to_ch(q_ctrl.data(), q_ch.data());        // 위치/속도는 참고값(kp=kd=0 이라 무영향)
      jm.dq_ctrl_to_ch(dq_ctrl.data(), dq_ch.data());
      jm.tau_ctrl_to_ch(tau_ctrl.data(), tau_ch.data());
      foot_comp(tau_ch);
      // ★순수 토크: kp=kd=0. 드라이버가 tau_ff 만 실행한다.
      // ★블렌드: s=0 → 위치제어(hold 자세) · s=1 → 순수토크(WBIC)
      double sb = (stand_T>0) ? (lt-stand_t0)/stand_T : 1.0;
      sb = std::max(0.0, std::min(1.0, sb));
      const double bs = sb*sb*(3.0-2.0*sb);            // smoothstep
      // ★★**감쇠는 남긴다** (2026-08-20 실기). kp 는 0 으로 보내되 kd 는 유지한다.
      //   블렌드 끝(순수 kp=kd=0 토크)이 이 로봇에선 **불안정**했다 — 세 설정 모두에서
      //   위치게인이 빠지는 후반에 발산했다:
      //       지연ON·마찰ON  30.5Hz τmax 7.43 · 지연OFF 48.8Hz 16.12 · 마찰OFF 31.3Hz 13.21
      //       |dq| 6→12→19→44→196→322 dps (게인이 35% 아래로 내려간 뒤 폭발)
      //   8.4ms 지연은 30~65Hz 에서 위상지연 90~150° 라 순수 토크 피드백으로는 못 잡는다.
      //   ⚠kp 를 남기면 WBIC 와 싸운다(정지한 hold 자세로 되당긴다). kd 는 **속도만**
      //     억제하므로 목표와 싸우지 않고 그 대역만 먹는다. 그래서 kd 만 남긴다.
      //   calf 기준 kd_ch 3.5 = 관절 7.9 Nm·s/rad — 물리 감쇠(0)보다 압도적이다.
      //   STAND_KD_FLOOR 로 조절(0=종전 순수토크 · 1=설정 kd 전량).
      // ★walk 는 kd 를 낮춘다(08-27 묶음): dq_des=0 + kd 전량 = 스윙 제동 max 41 Nm(트립 2.7×).
      //   sim 검증 배터리(8/8)는 드라이버 kd 제동이 없는 플랜트 — 실기를 sim 정합 쪽으로 옮기는 조치.
      //   ★계단 금지(08-27 검증 확정 HIGH): walk↔stand 에서 kdf 를 한 틱에 0.15↔1.0 점프시키면
      //     잔류 스윙속도에 kd 제동 ~41Nm 이 같은 순간 실린다 → 2.0/s 램프(0.15↔1.0 에 ~0.43s).
      //     이탈 방향은 속도 붕괴(수십 ms)가 램프보다 훨씬 빨라 제동 스파이크가 없다.
      const double kdf_tgt = (mode=="walk") ? WALK_KD_FLOOR
                       : (getenv("STAND_KD_FLOOR") ? atof(getenv("STAND_KD_FLOOR")) : 1.0);
      static double kdf_cur = -1.0;
      if(kdf_cur < 0.0) kdf_cur = kdf_tgt;
      { const double stp = 2.0 * dt;
        kdf_cur += std::max(-stp, std::min(stp, kdf_tgt - kdf_cur)); }
      const double kdf = kdf_cur;
      // ★★**kp 도 남긴다** (2026-08-20 실기: home 은 버티는데 stand 에서 처졌다).
      //   kd 는 **속도만** 잡으므로 처짐(정적 오차)에는 아무 도움이 안 된다.
      //   stand 는 kp=0 이라 관절 복원력이 **하나도 없고**, 자세를 잡는 건 모델 토크뿐이다:
      //       τ = h − Jᵀλ     h = **모델의** 중력항
      //   실제가 모델(13.9kg)보다 무겁거나 α(토크 스케일, ±10% 미검증)가 낮으면
      //   **토크가 모자라 처지는데 그걸 되돌릴 게 없다.**
      //   ⇒ feedforward 토크 + 관절 PD. 표준 구성이고, 2점 stand 는 PD 목표(stand_ref)와
      //     WBIC 목표가 **같은 자세**라 서로 싸우지 않는다.
      //   ⚠1점 보행(cmode=0)은 기본 0 이다 — 거기선 목표가 매 스텝 바뀌어 PD 가 방해한다.
      // ★1점 walk 는 kp 서보를 env 로도 못 켠다(08-27 검증 확정): 목표(stand_ref)가 진입 시점
      //   자세로 굳어 있어 매 스텝 움직이는 gait 와 kp 가 정면으로 싸운다. 2점(cmode=1)은
      //   목표=Qflat8=WBIC 자세라 유지(정적지지 전용 — FLAT_WALK 프리뷰 완성 시 재설계).
      const double kpf = (mode=="walk" && c.cmode!=1) ? 0.0
                       : (getenv("STAND_KP_FLOOR") ? atof(getenv("STAND_KP_FLOOR"))
                                                   : (c.cmode==1 ? 0.30 : 0.0));
      const double kd_scale = (1.0-bs) + bs*kdf;      // 블렌드 끝에서 kdf 로 수렴
      const double kp_scale = (1.0-bs) + bs*kpf;      // 블렌드 끝에서 kpf 로 수렴
      jm.kp_ch(kp_ch.data(), kp_scale); jm.kd_ch(kd_ch.data(), kd_scale);
      for(int i=0;i<NCH;i++) tau_ch[i] = (float)(bs*(double)tau_ch[i]);
      // 목표: 측정각 → 기하 자세로 블렌드와 **같은 계수**로 이동. bs=1 이면 순수 Qflat8.
      //   ⚠지금은 bs=1 에서 kp=0 이라 이 목표가 무영향이다. 그래도 측정각을 흘려보내지
      //     않는다 — STAND_KP_FLOOR 를 켜는 순간 **의미 있는 목표가 이미 들어가 있어야** 한다.
      //     측정각을 목표로 두면 err≡0(무효)이거나, 센서지연 때문에 err≈−q̇·τ = **음의 감쇠**가 된다.
      for(int i=0;i<NCH;i++)
        stand_ref[i] = stand_hold[i] + (float)(bs*(double)(stand_to[i]-stand_hold[i]));
      qcmd_ch = stand_ref; kpcmd_ch = kp_ch; kdcmd_ch = kd_ch;
      write_ff(stand_ref.data(), tau_ch, kp_ch.data(), kd_ch.data());   // ★α 보정 FF
      // ★stand 폭주 가드 (2026-09-03 · 상단 선언부 주석). 기준자세 이탈이
      //   STAND_RUNAWAY_DEG 를 0.3s 지속하면 hold 로 강하 + 래치(off 재무장까지 거부).
      if(mode=="stand"){
        double emx=0; int ech=-1;
        for(int i=0;i<NCH;i++) if(cfg.installed_has(i)){
          const double e = std::fabs((double)stand_ref[i]-(double)hs.q_deg[i]);
          if(e>emx){ emx=e; ech=i; } }
        if(emx > STAND_RUNAWAY_DEG){
          if(runaway_t0 < 0) runaway_t0 = lt;
          else if(lt - runaway_t0 > 0.3){
            std::printf("[deploy] ⛔ **stand 폭주** — %s 이탈 %.1f° > %.1f° 가 0.3s 지속.\n"
                        "         hold 로 강하(지지율 유지 — limp 아님) + 래치. off 재무장까지 stand 거부.\n",
                        ech>=0?chname[ech].c_str():"-", emx, STAND_RUNAWAY_DEG);
            std::fflush(stdout);
            hold_ch = hs.q_deg; jm.clamp_ch_via_joint(hold_ch.data());
            prev_mode = mode; mode = "hold"; mode_t0 = lt;
            stand_runaway = true; runaway_t0 = -1.0;
            // ⚠hold_fz_cur 는 **일부러 안 리셋** — 전방보상 연속(강하 순간 주저앉음 방지)
          }
        } else runaway_t0 = -1.0;
      } else runaway_t0 = -1.0;
    }

    // ★트레이스 기록 — 무장 후 3초.
    //   ★★2026-08-28 두 곳을 고쳤다. 종전 트레이스로 한 **추종오차 분석은 전부 무효**다.
    //   ① **위치**: 종전엔 모드 분기 **앞**에서 찍었다 ⇒ 측정은 이번 틱인데 명령은
    //      **직전 틱** 것이었다(외부 분석에서 `cmd[k] ≡ q[k−1]`, max|Δ|=0.000000 으로 검출).
    //   ② **열**: 종전엔 `(mode=="hold")? hold_ch : q_ch` 를 명령각이라 적었는데,
    //      **stand 의 실제 명령각은 `stand_ref`** 이고 `q_ch` 는 stand 분기에서 갱신조차
    //      안 된다(float/push 가 마지막에 넣은 측정각이 그대로 남는다).
    //      ⇒ 모든 모드에서 실제로 나간 값인 `qcmd_ch/kpcmd_ch/kdcmd_ch` 를 찍는다.
    //   ★게인도 같이 찍는다: blend 중에는 kp·kd 가 매 틱 변한다. 이게 없으면
    //     τ_ff = τ_echo − kp·err − kd·(−q̇) 분해를 **추정**할 수밖에 없고, 08-28 리플
    //     조사가 실제로 그 재구성 때문에 불확실성을 안았다. 이제 직접 분해된다.
    if(trc){
      if(lt-trc_t0 <= 3.0){
        fprintf(trc,"%.4f", lt-trc_t0);
        for(int i=0;i<NCH;i++)
          fprintf(trc,",%.3f,%.1f,%.3f,%.3f,%.2f,%.2f", hs.q_deg[i], hs.dq_dps[i], hs.tau_nm[i],
                  (double)qcmd_ch[i], (double)kpcmd_ch[i], (double)kdcmd_ch[i]);
        fprintf(trc,"\n");
      } else { fclose(trc); trc=nullptr; std::printf("[deploy] 트레이스 저장 완료\n"); }
    }

    // ★★자세유지 토크 스냅샷 — hold(위치제어) vs stand(WBIC 토크) 비교용.
    //   같은 자세를 버티는 데 각 축이 실제로 내는 토크를 두 모드에서 각각 잡아 둔다.
    //   ⚠정착을 기다린다: hold 는 진입 1.5s, stand 는 **블렌드가 끝나고** 1.5s.
    //     그 전 값은 계단 응답이라 "유지 토크" 가 아니다.
    {
      const bool hold_settled  = (mode=="hold")  && (lt-mode_t0 > 1.5);
      const bool stand_settled = (mode=="stand") && (lt-mode_t0 > stand_T + 1.5);
      if(hold_settled){
        tau_hold = tau_avg;
        for(int i=0;i<NCH;i++) q_hold[i]=(double)hs.q_deg[i];
        have_tau_hold = true;
      }
      if(stand_settled){
        tau_stand = tau_avg;
        for(int i=0;i<NCH;i++) q_stand[i]=(double)hs.q_deg[i];
        if(!have_tau_stand && have_tau_hold){
          have_tau_stand = true;
          // 자세가 실제로 같은지 먼저 본다 — 다르면 토크 차이는 해석 불가다.
          double qmx=0; int qch=-1;
          for(int i=0;i<NCH;i++) if(cfg.installed_has(i)){
            double e=q_stand[i]-q_hold[i];
            if(std::fabs(e)>std::fabs(qmx)){ qmx=e; qch=i; } }
          std::printf("\n[deploy] ═══ 자세유지 토크: hold(위치제어) vs stand(WBIC) ═══\n"
                      "  축         hold[Nm]  stand[Nm]     Δ[Nm]   |  q_hold   q_stand    Δq[°]\n");
          double tmx=0; int tch=-1;
          for(int i=0;i<NCH;i++){
            if(!cfg.installed_has(i)) continue;
            double dt_ = tau_stand[i]-tau_hold[i], dq_ = q_stand[i]-q_hold[i];
            if(std::fabs(dt_)>std::fabs(tmx)){ tmx=dt_; tch=i; }
            std::printf("  %-10s %8.3f  %8.3f  %+8.3f   | %7.2f  %7.2f  %+7.2f\n",
                        chname[i].c_str(), tau_hold[i], tau_stand[i], dt_,
                        q_hold[i], q_stand[i], dq_);
          }
          std::printf("  최대 Δτ %s %+.3f Nm · 최대 Δq %s %+.2f°\n",
                      tch>=0?chname[tch].c_str():"-", tmx,
                      qch>=0?chname[qch].c_str():"-", qmx);
          // 해석을 사람이 매번 다시 하지 않게 여기 적어 둔다.
          if(std::fabs(qmx) > 2.0)
            std::printf("  ⚠자세가 %.1f° 어긋났다 — **토크 차이를 그대로 읽으면 안 된다.**\n"
                        "    stand 가 그만큼 처졌다는 뜻이고, Δτ 에는 자세 차이의 중력분이 섞여 있다.\n", std::fabs(qmx));
          if(c.cmode!=1)
            std::printf("  ⚠1점 점발이다 — home 목표(설정 0°)와 stand 목표(Qhome8)가 **다른 자세**라\n"
                        "    이 비교는 성립하지 않는다. 2점 평발(cmode=1)에서 볼 것.\n");
          std::printf("  읽는 법: Δτ≈0 이면 WBIC 가 중력을 제대로 보상하고 있다.\n"
                      "           Δτ 가 음이면 stand 가 **덜 내고 있다** — 그만큼 처진다.\n"
                      "  CSV → /tmp/hold_vs_stand.csv\n\n");
          if(FILE* cf = fopen("/tmp/hold_vs_stand.csv","w")){
            fprintf(cf,"ch,name,tau_hold_nm,tau_stand_nm,dtau_nm,q_hold_deg,q_stand_deg,dq_deg\n");
            for(int i=0;i<NCH;i++) if(cfg.installed_has(i))
              fprintf(cf,"%d,%s,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f\n", i, chname[i].c_str(),
                      tau_hold[i], tau_stand[i], tau_stand[i]-tau_hold[i],
                      q_hold[i], q_stand[i], q_stand[i]-q_hold[i]);
            fclose(cf);
          }
        }
      }
    }

    // ── ★토크 불응답(사일런트 축사망) 감지 (2026-08-28 실기에서 당함) ──────────
    //   실기 서명: "신호는 들어오는데 토크제어가 안 먹는" 채널이 **간헐적으로** 생긴다 —
    //   health=ok · fault 0 · estop 없음 · 토크 에코 정상이라 **어떤 플래그로도 안 잡힌다.**
    //   실측(08-28 에어워크 정렬 실패): HR_thigh 가 명령에서 10.1° 벌어진 채 정지,
    //   kp 가 요구하는 토크 8.8 Nm 인데 그 자세의 중력은 0.65 Nm — 붙잡는 것이 중력이 아니다.
    //   ⚠하중 처짐과 반드시 구분해야 한다(2kg 무게추에서 thigh 가 10° 처지는 건 **정상**):
    //     처짐 평형은 kp·err ≈ G/α 라 중력과 균형이 맞고, 불응답은 중력을 크게 웃돈다.
    //   ⇒ 위치모드에서 (큰 오차 + 정지 + 중력 대비 과도 요구토크)가 1s 지속 = 래치.
    //     래치 중 stand/walk 진입 거부(사일런트 무토크 축 위에 서면 그대로 주저앉는다).
    //   ⚠stand/walk 는 kp≈0 이라 이 서명이 성립하지 않는다 — 그래서 위치모드 전용이고,
    //     그 대신 **진입 차단**으로 보호한다.
    if(mode!="off" && mode!="stand" && mode!="walk" && !estop && lt - unresp_chk_t >= 0.1){
      unresp_chk_t = lt;
      // 측정 자세의 중력토크(고정베이스 — float/push 가드와 같은 계산)
      for(int j=0;j<NJ;j++){ d->qpos[7+j]=q_ctrl[j]; d->qvel[6+j]=0.0; }
      d->qpos[0]=d->qpos[1]=0; d->qpos[2]=0.5;
      d->qpos[3]=1; d->qpos[4]=d->qpos[5]=d->qpos[6]=0;
      for(int i=0;i<6;i++) d->qvel[i]=0.0;
      mj_forward(m,d);
      std::vector<double> gj(NJ); std::vector<float> gch(NCH, 0.f);
      for(int j=0;j<NJ;j++) gj[j] = d->qfrc_bias[6+j];
      jm.tau_ctrl_to_ch(gj.data(), gch.data());
      // ★접지 중이면 검사하지 않는다 — 체중이 실리면 관절토크가 "매달림 중력" 을 정당하게
      //   크게 웃돌아 전 축이 오탐된다(float/push 가드와 같은 매달림 판정을 재사용).
      { std::vector<double> tm(NJ,0.0); jm.ch_to_tau_joint(hs.tau_nm.data(), tm.data());
        double hang=0, meas=0;
        for(int j=0;j<NJ;j++){ hang += std::fabs(gj[j]); meas += std::fabs(tm[j]); }
        if(hang>1e-6 && meas/hang > 1.25) goto unresp_skip;   // 접지 — 판정 보류
      }
      // ★★중력지지 중에도 검사하지 않는다 (2026-09-02 실기 오탐 — HL_foot 에서 발화).
      //   τ_ff 가 발을 지면에 밀어붙이므로 "kp 요구토크 vs 매달림 중력" 비교의 전제가
      //   무너진다 — 오차 7° 를 붙잡던 건 드라이버 불응답이 아니라 **지면 반력**이었다.
      //   위의 전역 접지 게이트(meas/hang)로는 못 거른다: 크레인이 체중 대부분을 들면
      //   전신 합계는 "매달림" 으로 보이는데, **발끝만은 이미 눌려** 있었다.
      if(mode=="hold" && hold_ff_pct_cur > 5.0) goto unresp_skip;
      for(int i=0;i<NCH;i++){
        if(!cfg.installed_has(i) || unresp[i]) continue;
        const double err  = (double)qcmd_ch[i] - (double)hs.q_deg[i];      // 채널 deg
        const double tkp  = (double)kpcmd_ch[i] * err * M_PI/180.0;        // kp 가 내는 채널토크
        // ★문턱 근거: 건강한 처짐 평형은 kp·err = (G + 마찰)/α 라 |tkp| ≈ 1.2|G| + ~1 Nm 이다
        //   (α≈0.85 · 채널 마찰 0.4~0.9 Nm). 2|G| + 2.0 은 그 위로 충분히 떨어져 있고,
        //   실측 불응답(tkp 8.8 vs G 0.65)은 크게 넘는다.
        const bool   stuck = std::fabs(err) > 5.0 && std::fabs(hs.dq_dps[i]) < 2.0
                          && std::fabs(tkp) > 2.0*std::fabs((double)gch[i]) + 2.0;
        if(getenv("UNRESP_DBG") && cfg.installed_has(i))
          std::printf("[unresp] ch%d err %+.1f dq %+.1f tkp %+.2f G %+.2f %s\n",
                      i, err, (double)hs.dq_dps[i], tkp, (double)gch[i], stuck?"STUCK":"");
        if(stuck){
          if(unresp_t0[i] < 0) unresp_t0[i] = lt;
          else if(lt - unresp_t0[i] > 1.0){
            unresp[i] = 1;
            std::printf("[deploy] ⛔ **토크 불응답 의심 ch%d(%s)** — 오차 %.1f° 가 %.1fs 유지,"
                        " 축은 정지(%.1f dps)\n"
                        "         kp 요구토크 %.2f Nm vs 그 자세 중력 %.2f Nm — 붙잡는 게 중력이 아니다.\n"
                        "         (드라이버가 명령을 조용히 무시하거나 기구가 물렸다. health=ok 여도 그렇다)\n"
                        "         → **stand/walk 진입을 막는다.** off 로 내렸다가 재무장하거나 Emb 재기동할 것.\n",
                        i, chname[i].c_str(), err, lt-unresp_t0[i], (double)hs.dq_dps[i],
                        tkp, (double)gch[i]);
          }
        } else unresp_t0[i] = -1.0;
      }
      unresp_skip: ;
    }
    if(mode=="off"){ for(int i=0;i<NCH;i++){ unresp[i]=0; unresp_t0[i]=-1.0; } }

    // ⑥ 상태 발행(~20Hz)
    double period = lt - prev_loop; prev_loop = lt;
    if(period > 0) hz_ema = 0.98*hz_ema + 0.02*(1.0/period);
    if(lt - last_pub > 0.05){
      last_pub = lt;
      int n_ok=0, n_fault=0, n_dead=0, n_absent=0;
      // ★ucStatus(=ERROR VECTOR 하위 8비트)를 **원값 그대로** 낸다 (2026-08-20).
      //   종전엔 "fault" 라는 라벨만 나가서, 무엇 때문에 걸렸는지 알 수 없었다.
      //   과전류·과온·엔코더·통신 중 뭔지 알아야 조치가 갈린다.
      // 명령각을 **모델각**으로 바꿔 낸다(측정 q_leg_deg 와 같은 좌표라 바로 비교된다)
      std::string qcmds="[", dqcmds="[";
      { std::vector<double> qc(jm.n_leg); jm.ch_to_q_joint(qcmd_ch.data(), qc.data());
        for(int i2=0;i2<jm.n_leg;i2++){ char b[24];
          std::snprintf(b,sizeof b,"%s%.3f", i2?",":"", qc[i2]); qcmds+=b;
          std::snprintf(b,sizeof b,"%s0.0", i2?",":""); dqcmds+=b; }
        qcmds+="]"; dqcmds+="]"; }
      std::string errs="[";
      std::string health="[", inst="[", qs="[", qchs="[";
      // ★2026-08-13 `q_leg_deg` 에 **채널각**을 넣고 있었다(모델각 계약 위반).
      //   emb/interface/state_pub.py 규약: q_leg_deg = **모델각**(MJCF qpos 와 같은 좌표계).
      //   채널각을 넣으면 뷰어·모니터가 **틀린 자세를 그린다** — gear_k(calf 1.5·foot 1.2)와
      //   커플링만큼 어긋난다. 실기에서 자세를 화면으로 보며 판단하므로 직격이다.
      //   ⇒ 여기서 한 번 변환하고, 채널각은 `q_ch_deg` 로 따로 낸다(캘리브레이션·진단용).
      //   ⚠`q_ch_deg` 가 없으면 diag/couple_check.py 가 "구버전" 이라며 거부한다.
      std::vector<double> q_leg(jm.n_leg);
      jm.ch_to_q_joint(hs.q_deg.data(), q_leg.data());
      for(int i=0;i<jm.n_leg;i++){
        int ch = cfg.joints[i].channel;
        bool ins = cfg.installed_has(ch);
        const char* h = !ins ? "absent"
                      : (ch>=(int)hs.connected.size() || !hs.connected[ch]) ? "dead"
                      : (ch<(int)hs.status.size() && hs.status[ch]!=0) ? "fault" : "ok";
        if(!ins) n_absent++; else if(!std::strcmp(h,"ok")) n_ok++;
        else if(!std::strcmp(h,"fault")) n_fault++; else n_dead++;
        { char b[16]; std::snprintf(b,sizeof b,"%s%d", i?",":"",
            ch<(int)hs.status.size()? hs.status[ch] : -1); errs += b; }
        char b[64];
        std::snprintf(b,sizeof b,"%s\"%s\"", i?",":"", h); health += b;
        std::snprintf(b,sizeof b,"%s%s", i?",":"", ins?"true":"false"); inst += b;
        std::snprintf(b,sizeof b,"%s%.2f", i?",":"", q_leg[i]);      qs   += b;   // 모델각
        std::snprintf(b,sizeof b,"%s%.2f", i?",":"", hs.q_deg[ch]);  qchs += b;   // 채널각
      }
      health+="]"; inst+="]"; qs+="]"; qchs+="]";
      // ★모니터링용 측정·명령 (2026-08-13). Python app(biped_emb.py)과 **같은 키·같은 단위**로 낸다
      //   — 모니터가 두 제어기를 구분하지 않아도 되게. 모델각[deg·deg/s] · 관절토크[Nm].
      //   ⚠이 경로는 **순수 토크모드**(kp=kd=0)라 위치·속도 "명령" 이 없다. q_ch/dq_ch 는
      //     측정에서 만든 참고값이라 명령으로 내보내면 거짓말이 된다 ⇒ 토크만 명령으로 낸다.
      //     kp/kd 를 0 으로 함께 내보내 모니터가 "위치명령 없음" 을 알 수 있게 한다.
      std::vector<double> tau_meas(jm.n_leg);
      jm.ch_to_tau_joint(hs.tau_nm.data(), tau_meas.data());
      std::string dqs="[", taus="[", taucs="[", kps="[", kds="[";
      for(int i=0;i<jm.n_leg;i++){
        char b[64]; const char* sep = i?",":"";
        std::snprintf(b,sizeof b,"%s%.2f", sep, dq_ctrl[i]*JointMap::R2D); dqs   += b;  // 측정 속도
        std::snprintf(b,sizeof b,"%s%.3f", sep, tau_meas[i]);             taus  += b;  // 측정 토크
        std::snprintf(b,sizeof b,"%s%.3f", sep, tau_ctrl[i]);             taucs += b;  // 명령 토크
        // ★채널게인 → **raw 게인**: kp_raw = kp_ch · gear_k²  (emb/README "게인도 좌표가 둘")
        //   ⚠"관절(모델각) 게인" 이 아니다 (2026-08-21 정정). Δq_ch = s·Δq_**raw** 이므로
        //     이 값이 곱해지는 상대는 raw 오차다. 발목만 raw ≠ 모델각이라 거기서 갈린다.
        //     모델각 강성은 K = Aᵀ·diag(kp_raw)·A 로 **비대각이 생겨** 축별 스칼라로 못 낸다
        //     (calf 180 → 223.2). ⇒ 스칼라로 낼 수 있는 정직한 좌표는 raw 뿐이다.
        //   ⚠키 이름이 kp_leg → **kp_raw** 로 바뀌었다. 좌표를 이름에 박아 둔다 —
        //     종전엔 같은 키를 C++ 은 ×k², Python 은 채널 그대로 실어 calf 가 180 vs 80 이었다.
        { const auto& jj = cfg.joints[i]; const double k2 = jj.gear_k*jj.gear_k;
          std::snprintf(b,sizeof b,"%s%.2f", sep, (double)kpcmd_ch[jj.channel]*k2); kps += b; }
        { const auto& jj = cfg.joints[i]; const double k2 = jj.gear_k*jj.gear_k;
          std::snprintf(b,sizeof b,"%s%.2f", sep, (double)kdcmd_ch[jj.channel]*k2); kds += b; }
      }
      dqs+="]"; taus+="]"; taucs+="]"; kps+="]"; kds+="]";
      // ★출력축(aux) 엔코더 + 통신 ACK (2026-09-01, RGA 08/31 펌웨어) — 지원될 때만 키를 낸다.
      //   키 부재 = "기능 없음"(구 .so/mock), 값 0 = "AUX_MODE 꺼짐/에코 정상" — 셋을 구분한다.
      //   aux_deg 는 **채널 원시각**(모델각 아님): 벨트 앞단이라 calf/foot 은 gear_k 나누기 전 값.
      std::string extra_json;
      { float ap[16], av[16]; int al[16], as_[16];
        const int ar = hw->aux(ap, av);
        if(ar >= 0){
          char b[64];
          extra_json += "\"aux_on\":"; extra_json += (ar==1) ? "true," : "false,";
          extra_json += "\"aux_deg\":[";
          for(int i=0;i<NCH;i++){ std::snprintf(b,sizeof b,"%s%.3f", i?",":"", (double)ap[i]); extra_json += b; }
          extra_json += "],\"aux_dps\":[";
          for(int i=0;i<NCH;i++){ std::snprintf(b,sizeof b,"%s%.2f", i?",":"", (double)av[i]); extra_json += b; }
          extra_json += "],";
        }
        if(hw->ack(al, as_) == 0){
          char b[32];
          extra_json += "\"ack_lag\":[";
          for(int i=0;i<NCH;i++){ std::snprintf(b,sizeof b,"%s%d", i?",":"", al[i]); extra_json += b; }
          extra_json += "],\"ack_stale\":[";
          for(int i=0;i<NCH;i++){ std::snprintf(b,sizeof b,"%s%d", i?",":"", as_[i]); extra_json += b; }
          extra_json += "],";
        } }
      // ★창 통계(500Hz 누적) → 발행. 창이 비면(첫 틱) 빈 배열 대신 0 을 낸다.
      std::string tsd="[", tmn="[", tmx="[";
      for(int i=0;i<jm.n_leg;i++){
        char b[64]; const char* sep = i?",":"";
        double mu = ts_n? ts_sum[i]/ts_n : 0.0;
        double sd = ts_n? std::sqrt(std::max(0.0, ts_sq[i]/ts_n - mu*mu)) : 0.0;
        std::snprintf(b,sizeof b,"%s%.3f", sep, sd); tsd += b;
        std::snprintf(b,sizeof b,"%s%.3f", sep, ts_n? ts_min[i]:0.0); tmn += b;
        std::snprintf(b,sizeof b,"%s%.3f", sep, ts_n? ts_max[i]:0.0); tmx += b;
      }
      tsd+="]"; tmn+="]"; tmx+="]";
      // ★자세유지 토크 스냅샷 → **관절 좌표**로 발행(tau_leg_nm 과 같은 좌표라야 같이 읽힌다).
      //   아직 안 잡혔으면 빈 배열 `[]` — 모니터가 "없음" 과 "0 Nm" 을 구분할 수 있어야 한다.
      auto tau_snap = [&](const std::vector<double>& t, bool have){
        if(!have) return std::string("[]");
        std::vector<float> tf(NCH); for(int i=0;i<NCH;i++) tf[i]=(float)t[i];
        std::vector<double> tj(jm.n_leg); jm.ch_to_tau_joint(tf.data(), tj.data());
        std::string s="[";
        for(int i=0;i<jm.n_leg;i++){ char b[32]; std::snprintf(b,sizeof b,"%s%.3f", i?",":"", tj[i]); s+=b; }
        return s+"]"; };
      const std::string thold = tau_snap(tau_hold, have_tau_hold);
      const std::string tstand= tau_snap(tau_stand, have_tau_stand);
      const long ts_n_pub = ts_n;
      ts_reset();                       // 창을 비운다 — 다음 발행까지 다시 쌓는다
      char buf[6144];   // ★5120 → 6144 (2026-09-01): aux/ack 4배열 추가(최대 ~600B)
      // 런타임에 안 변하므로 한 번만 만든다.
      static std::string offs_json;
      if(offs_json.empty()){
        offs_json = "[";
        for(size_t i=0;i<cfg.joints.size();i++){
          char b[32]; std::snprintf(b,sizeof b,"%s%.3f", i?",":"", cfg.joints[i].offset_deg);
          offs_json += b;
        }
        offs_json += "]";
      }
      std::snprintf(buf,sizeof buf,
        "{\"mode\":\"%s\",\"backend\":\"%s\",\"q_leg_deg\":%s,\"q_ch_deg\":%s,"
        "\"dq_leg_dps\":%s,\"tau_leg_nm\":%s,\"tau_cmd_nm\":%s,\"kp_raw\":%s,\"kd_raw\":%s,"
        // ★창 통계 — **500Hz 로 계산**한 값이다(발행 20Hz 표본이 아니라). tau_win_n 은
        //   그 창에 들어간 표본 수 = 통계의 신뢰도. 0 이면 통계를 읽지 말 것.
        "\"tau_std_nm\":%s,\"tau_min_nm\":%s,\"tau_max_nm\":%s,\"tau_win_n\":%ld,"
        "\"rpy_deg\":[%.2f,%.2f,%.2f],\"tilt_deg\":%.2f,\"loop_hz\":%.1f,"
        "\"motors_on\":%s,\"health\":%s,\"installed\":%s,"
        "\"n_ok\":%d,\"n_fault\":%d,\"n_dead\":%d,\"n_absent\":%d,\"n_installed\":%d,"
        "\"est_x\":%.3f,\"est_z\":%.3f,\"estop\":%s,\"tilt_estop_ok\":%s,"
        // ★WBIC QP 건강도 — **접지 판정의 유일한 지표**(biped_control.hpp 주석 참조).
        //   발이 덜 닿으면 QP 가 매 틱 실패하고 중력보상 폴백으로 떨어지는데,
        //   겉보기엔 안정돼 보인다. 이 셋이 그 상태를 드러낸다.
        // ★지연보상 상태. `lat_comp_ms`=0 이면 **꺼져 있다**(env 미설정).
        //   `lc_skip_pct`>0 은 예측이 가드에 걸려 실측으로 폴백 중이라는 뜻 —
        //   켜 놓고도 실제로는 안 걸린 상태라 반드시 보인다.
        "\"qp_fail_pct\":%.1f,\"qp_K\":%d,\"qp_cerr\":[%.4f,%.4f,%.4f],"
        // ★home 진행률(0~1) — GUI 가 이미 표시할 준비가 돼 있다(teleop_gui_biped:526).
        //   10초짜리 램프라 운전자가 "지금 얼마나 갔나" 를 볼 수 있어야 한다.
        "\"lat_comp_ms\":%.2f,\"lc_skip_pct\":%.1f,\"home_progress\":%.3f,\"err\":%s,"
        // ★자세유지 토크 스냅샷(관절 Nm). hold=위치제어로 버틸 때, stand=WBIC 로 버틸 때.
        //   `[]` 는 **아직 안 잡혔다**는 뜻이다(0 Nm 이 아니라).
        // ★강성 배율 — `pos_kp_scale` 은 **지금 실제로 나가고 있는 값**(램프 중이면 중간값),
        //   `pos_kp_target` 은 GUI 가 요구한 값이다. 둘이 다르면 아직 옮겨 가는 중이다.
        "\"q_cmd_deg\":%s,\"dq_cmd_dps\":%s,\"tau_hold_nm\":%s,\"tau_stand_nm\":%s,"
        // ★영점 — **이 프로세스가 기동 시 읽은 값**이다(config 파일의 현재값이 아니다).
        //   config 를 고쳐도 재시작 전에는 안 바뀐다. GUI 가 파일값과 이걸 나란히 놓아
        //   "제어기가 아직 옛 영점을 쓴다" 를 보여 준다.
        //   ⇒ 그래서 GUI 가 채널각↔모델각 역산식을 **복사할 필요가 없다.** 그 복사본이
        //     stale 이 되는 게 이 저장소가 반복해서 당한 버그다(joint_map 규칙 복제 주석 참조).
        "\"pos_kp_scale\":%.3f,\"pos_kp_target\":%.3f,\"pos_kd_scale\":%.3f,\"push_fz\":%.1f,"
        // ★hold 중력지지 — `hold_ff_pct` 는 **지금 실제로 나가는 값**(램프 중이면 중간값),
        //   `hold_ff_n` 은 그때의 한 다리 지지력[N]. 0 이면 순수 위치 PD(종전 hold).
        "\"hold_ff_pct\":%.1f,\"hold_ff_n\":%.1f,\"hold_ff_full_n\":%.1f,\"hold_ff_split\":%.1f,"
        "%s\"offset_deg\":%s}",
        mode.c_str(), hw->name(), qs.c_str(), qchs.c_str(),
        /* dq/tau/tau_cmd/kp/kd 는 다음 줄에서 이어진다 — 아래 5개 뒤에 창통계 4개 */
        dqs.c_str(), taus.c_str(), taucs.c_str(), kps.c_str(), kds.c_str(),
        tsd.c_str(), tmn.c_str(), tmx.c_str(), ts_n_pub,
        rpy[0]*JointMap::R2D, rpy[1]*JointMap::R2D,
        rpy[2]*JointMap::R2D, tilt, hz_ema, (mode!="off"&&!wd)?"true":"false",
        health.c_str(), inst.c_str(), n_ok, n_fault, n_dead, n_absent,
        (int)(jm.n_leg-n_absent), est.p[0], est.p[2], estop?"true":"false",
        imu_dead?"false":"true",
        c.qp_rate*100.0, c.qp_K, c.qp_cerr[0], c.qp_cerr[1], c.qp_cerr[2],
        LCOMP>0 ? lat_comp_ms : 0.0, lc_n ? 100.0*(double)lc_skip/(double)lc_n : 0.0,
        (mode=="home" && home_T>0) ? std::max(0.0,std::min(1.0,(lt-home_t0)/home_T)) : 0.0,
        (errs+"]").c_str(), qcmds.c_str(), dqcmds.c_str(), thold.c_str(), tstand.c_str(),
        POS_KP, kp_scale_tgt, POS_KD, push_fz_cur,
        hold_ff_pct_cur, hold_fz_cur, HOLD_FF_FULL_N, hold_split_cur,
        extra_json.c_str(), offs_json.c_str());
      write_state(stt_p, buf);
    }

    // ⑦ 실시간 페이싱(절대시각 — 누적 드리프트 없음)
    //   ★밀렸을 때 재동기: Pi 부하로 루프가 느려지면 t0+k·dt 가 현재보다 한참 과거가 되고,
    //     부하가 풀리는 순간 sleep 없이 수백 틱을 몰아쳐 돈다(제어주기 붕괴).
    //     10 틱 이상 밀리면 스케줄을 현재로 되맞추고, 지속되면 한 번 경고한다.
    k++;
    double lag = now_s() - (t0 + k*dt);
    if(lag > 10*dt){
      if(!overrun_warned){ overrun_warned = true;
        std::fprintf(stderr, "[deploy] ⚠ 루프가 %.0fms 밀렸다 — 제어주기 %.0fHz 를 못 지키고 있다."
                             " CPU 부하/우선순위 확인.\n", lag*1e3, cfg.ctrl_hz); }
      t0 = now_s() - k*dt;                 // 스케줄 재동기(캐치업 스핀 방지)
    }
    sleep_s(t0 + k*dt - now_s());
  }

  if(g_stop) std::printf("\n[deploy] 신호 %d 수신 → 안전종료\n", (int)g_stop);
  safe_shutdown(*hw, NCH);
  if(LCOMP>0 && lc_n)
    std::printf("[deploy] 지연보상 %.2fms — %ld 틱 중 %ld 폴백(%.1f%%)\n",
                lat_comp_ms, lc_n, lc_skip, 100.0*(double)lc_skip/(double)lc_n);
  if(dpred) mj_deleteData(dpred);
  delete hw; mj_deleteData(d); mj_deleteModel(m);
  return rc;
}
