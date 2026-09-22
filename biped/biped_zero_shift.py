#!/usr/bin/env python3
"""biped_zero_shift.py — exp_log 의 (q_ch − aux) 오프셋으로 **영점 틀어짐** 진단.
   각 관절 delta = q_ch_deg − aux_deg  (주엔코더 − 2차엔코더).
   같은 물리자세에서 delta = [영점차 + 감속단 비틀림(하중의존)].
     → delta 가 **시간에 따라 변하면** 영점/기구 변화(벨트슬립·영점밀림). 비틀림은 하중(자세)의존이라
       old/new 의 aux(=관절측 실제각) 겹침 구간에서 **자세정합**하여 비틀림분을 상쇄해 비교.
   aux==0 은 드롭아웃 → 제거, 통계는 median(노이즈·드롭아웃 강건).

   사용:
     python3 biped_zero_shift.py                    # exp_logs 목록 + 각 로그의 발 delta 미리보기(드리프트 한눈에)
     python3 biped_zero_shift.py 이전.csv 현재.csv   # 두 로그 관절별 자세정합 비교
   ⚠qch/aux 는 좌우 부호규약이 다를 수 있음 → **절대 L-R 갭이 아니라 같은 축의 시간변화(Δ)** 로 판정.
"""
import csv, os, sys, glob, numpy as np
NM = ['HL_hip','HL_thigh','HL_calf','HL_foot','HR_hip','HR_thigh','HR_calf','HR_foot']
HERE = os.path.dirname(os.path.abspath(__file__))
EXPDIR = os.path.join(HERE, 'exp_logs')

def has_aux(row): return ('aux_%s'%NM[0] in row) and ('qch_%s'%NM[0] in row)

def load(path):
    """→ {joint: (aux_valid[deg], delta_valid[deg])}  (pose axis = aux = 관절측 실제각)."""
    rows = list(csv.DictReader(open(path)))
    if not rows: raise SystemExit("빈 CSV: %s" % path)
    if not has_aux(rows[0]):
        print("✗ %s: qch_/aux_ 열 없음 — 구버전 로그(aux 로깅 이전)." % os.path.basename(path)); raise SystemExit(1)
    out = {}
    for n in NM:
        qch = np.array([float(r['qch_%s'%n]) for r in rows])
        aux = np.array([float(r['aux_%s'%n]) for r in rows])
        m = (aux != 0.0) & np.isfinite(aux) & np.isfinite(qch)   # aux 0 = 드롭아웃 제거
        out[n] = (aux[m], (qch - aux)[m])
    return out

def matched(old, new, binw=2.0):
    """aux(자세) 겹침구간을 binw°로 나눠 bin별 median delta 차(new−old)의 중앙값 = 자세정합 Δ."""
    ao, do = old; an, dn = new
    if len(ao) < 3 or len(an) < 3: return None, 0.0
    lo = max(ao.min(), an.min()); hi = min(ao.max(), an.max())
    if hi - lo < binw: return None, 0.0
    edges = np.arange(lo, hi + binw, binw); diffs = []; span = 0.0
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i+1]
        mo = (ao >= a) & (ao < b); mn = (an >= a) & (an < b)
        if mo.sum() >= 3 and mn.sum() >= 3:
            diffs.append(np.median(dn[mn]) - np.median(do[mo])); span += (b - a)
    if not diffs: return None, 0.0
    return float(np.median(diffs)), span

def med(a):
    return float(np.median(a)) if len(a) else float('nan')

# ── 실시간: 매달림(미접지)에서 현재 (q_ch − aux) 스냅샷 (접지실험 로그 없이 '현재' 확보) ──
if len(sys.argv) >= 2 and sys.argv[1] == '--live':
    import json, time
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    SP = os.environ.get('QUAD_STATE', '/dev/shm/biped_state.json')
    acc = {n: [] for n in NM}; poses = {n: [] for n in NM}; nsamp = 0
    t0 = time.time()
    while time.time() - t0 < secs:
        try:
            st = json.load(open(SP)); qch = st.get('q_ch_deg'); aux = st.get('aux_deg')
            if qch and aux and len(qch) >= 8 and len(aux) >= 8:
                for i, n in enumerate(NM):
                    if aux[i] != 0.0 and np.isfinite(aux[i]) and np.isfinite(qch[i]):
                        acc[n].append(qch[i] - aux[i]); poses[n].append(aux[i])
                nsamp += 1
        except Exception:
            pass
        time.sleep(0.05)
    print("■ 실시간 (q_ch − aux) delta[deg] — %s · %.1fs · %d표본" % (os.path.basename(SP), secs, nsamp))
    if nsamp == 0: raise SystemExit("state 못 읽음/aux 없음 — deploy 기동·AUX_MODE=1 확인.")
    print("%-9s %9s %9s %6s" % ("관절", "delta", "aux각°", "표본"))
    for n in NM:
        if acc[n]:
            mark = "  ← 발" if n in ('HL_foot', 'HR_foot') else ""
            print("%-9s %9.2f %9.2f %6d%s" % (n, np.median(acc[n]), np.median(poses[n]), len(acc[n]), mark))
        else:
            print("%-9s %9s   (aux 0/드롭아웃)" % (n, "--"))
    print("→ 이 HL_foot delta 를 무인자 목록의 **이전 로그 HL_foot 값**과 대조. 수 도 차 = 영점 틀어짐.")
    print("  ⚠자세(aux각)가 이전 로그와 크게 다르면 비틀림분이 섞임 — 비슷한 발각에서 비교.")
    sys.exit(0)

# ── 인자 없음: 로그 목록 + 발 delta 미리보기(드리프트 한눈에) ──
if len(sys.argv) < 3:
    files = sorted(glob.glob(os.path.join(EXPDIR, '*.csv')))
    if not files: raise SystemExit("exp_logs 에 CSV 없음. GUI 접지실험(stand/squat/walk) 실행해 로그 생성.")
    print("■ exp_logs — 로그별 (q_ch−aux) median[deg]  (aux 있는 로그만)")
    print("%-40s %6s | %8s %8s %8s %8s" % ("파일", "표본", "HL_foot", "HR_foot", "HL_calf", "HR_calf"))
    print("-" * 92)
    for f in files:
        try:
            rows = list(csv.DictReader(open(f)))
            if not rows or not has_aux(rows[0]):
                print("%-40s %6s | %s" % (os.path.basename(f), len(rows), "  (구버전·aux 없음)")); continue
            d = load(f)
            print("%-40s %6d | %8.2f %8.2f %8.2f %8.2f" % (
                os.path.basename(f), len(rows),
                med(d['HL_foot'][1]), med(d['HR_foot'][1]), med(d['HL_calf'][1]), med(d['HR_calf'][1])))
        except Exception as e:
            print("%-40s  ! %s" % (os.path.basename(f), e))
    print("-" * 92)
    print("→ 같은 자세 계열 로그에서 HL_foot 열이 **이전 대비 몇 도 변했으면 영점 틀어짐**.")
    print("  자세정합 상세비교:  python3 biped_zero_shift.py 이전.csv 현재.csv")
    sys.exit(0)

# ── 두 로그 정밀비교 ──
OLDP, NEWP = sys.argv[1], sys.argv[2]
old = load(OLDP); new = load(NEWP)
print("■ 영점 틀어짐 진단 — (q_ch − aux) delta[deg]")
print("  이전: %s" % os.path.basename(OLDP))
print("  현재: %s" % os.path.basename(NEWP))
print("%-9s %8s %8s %8s | %9s %6s %s" % ("관절","이전δ","현재δ","Δ(전체)","Δ자세정합","겹침°","판정"))
print("-" * 74)
target = {}
for n in NM:
    do = old[n][1]; dn = new[n][1]
    md_o, md_n = med(do), med(dn)
    dfull = md_n - md_o
    dm, span = matched(old[n], new[n])
    key = dm if dm is not None else dfull        # 자세정합 우선, 없으면 전체차
    target[n] = key
    if not np.isfinite(md_o) or not np.isfinite(md_n):
        flag = "자료부족"
    elif abs(key) >= 3.0: flag = "★영점 크게 틀어짐"
    elif abs(key) >= 1.0: flag = "⚠변화 있음"
    else:                 flag = "≈유지"
    dm_s = ("%9.2f" % dm) if dm is not None else "     n/a"
    print("%-9s %8.2f %8.2f %8.2f | %s %6.0f %s" % (n, md_o, md_n, dfull, dm_s, span, flag))
print("-" * 74)
# 좌우 교차확인(같은 축의 시간변화끼리 비교 — 부호규약 무관)
hl, hr = target['HL_foot'], target['HR_foot']
print("→ HL_foot 변화 %+.2f° vs HR_foot 변화 %+.2f°." % (hl, hr))
if abs(hl) >= 1.0 and abs(hr) < 1.0:
    print("  ⇒ **HL_foot 만 틀어짐**(HR_foot 정상) — 왼발 영점/기구 국소 변화 확정적.")
elif abs(hl) >= 1.0 and abs(hr) >= 1.0:
    print("  ⇒ 양발 다 변함 — 공통원인(자세정합 실패·전체 영점) 의심, 겹침°/자세계열 확인.")
else:
    print("  ⇒ HL_foot 변화 미미 — 영점 틀어짐 근거 약함(다른 원인일 수 있음).")
print("  ※ delta 절대값엔 [영점차+비틀림+부호규약] 섞임 — **Δ(시간변화)**만 영점판정 근거.")
