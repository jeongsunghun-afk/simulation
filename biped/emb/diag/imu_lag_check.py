#!/usr/bin/env python3
"""imu_lag_check.py — 500Hz 트레이스로 IMU 자세 지연을 잰다 (2026-09-30).

2점 평발로 서 있으면 몸통 pitch 는 관절각의 함수다. 그래서
  IMU pitch(t) ≈ Σ a_i·q_i(t−L) + c
를 L 마다 최소제곱으로 맞춰 R² 가 가장 큰 L 이 IMU 지연이다(L>0 = IMU 가 늦음).
d(rpy)/dt ↔ gyro 시차도 같이 본다(같은 패킷이면 ≈0 → 자이로도 똑같이 늦다).

배경: 원본 RobotEmbedded 는 IMU UART 를 10ms 마다 1패킷만 읽어 적체 → 지연이 기동 후
1.0→1.8s 로 계속 늘었다. RobotEmbeddedNew(수정판)에서 8ms. Emb 교체·업데이트 후 재확인용.

사용: python3 imu_lag_check.py /dev/shm/arm_trace_stand_*.csv
      (TRACE_SEC=20 으로 stand/hold 중 녹화. 몸통이 몇 도라도 움직인 구간이 있어야 잴 수 있다)
"""
import csv, sys, numpy as np

def load(p):
    r = list(csv.DictReader(open(p)))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0].keys()}

def r2_at(y, X, L):
    n = len(y)
    yy, XX = (y[L:], X[:n-L]) if L >= 0 else (y[:n+L], X[-L:])
    A = np.c_[XX, np.ones(len(yy))]
    a, *_ = np.linalg.lstsq(A, yy, rcond=None)
    return 1 - np.var(yy - A @ a) / np.var(yy)

def main(paths):
    for p in paths:
        d = load(p); t = d["t"]; dt = np.median(np.diff(t)); step = max(1, int(0.01/dt))
        y = d["rpy1"]
        X = np.stack([d[f"q{i}"] for i in (1, 2, 3, 5, 6, 7)], 1)   # 좌우 thigh·calf·foot 채널
        if np.std(y) < 0.2:
            print(f"{p}: pitch 변화가 너무 작다(std {np.std(y):.2f}°) — 몸통이 움직인 구간을 녹화할 것"); continue
        res = [(L*dt, r2_at(y, X, L)) for L in range(-int(0.5/dt), int(2.5/dt)+1, step)]
        Lb, rb = max(res, key=lambda z: z[1])
        dr = np.gradient(y, dt); g = d["gyr1"]; best = (0, 0)
        for L in range(-int(0.5/dt), int(0.5/dt)+1, 2):
            a = dr[max(0, L):len(dr)+min(0, L)]; b = g[max(0, -L):len(g)-max(0, L)]
            c = abs(np.corrcoef(a, b)[0, 1])
            if c > best[0]: best = (c, L*dt)
        verdict = "정상(≈0)" if Lb < 0.05 else "⚠IMU 지연 — Emb IMU 수신 적체 의심(RobotEmbeddedNew 로 기동했는지 확인)"
        print(f"{p}\n  IMU pitch 지연 {Lb*1000:.0f} ms (R² {rb:.3f} · 지연0 가정 R² {r2_at(y, X, 0):.3f})  → {verdict}"
              f"\n  d(pitch)/dt ↔ gyro 시차 {best[1]*1000:+.0f} ms (상관 {best[0]:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(sys.argv[1:])
