#!/bin/bash
# 사용: qrun.sh <이름> [env...] — 사족 실기 유사 조건(추정기·지연 4+6ms·잡음·로터관성 GEARBOX) 공통
B=/home/jsh/quad_plant/quad/cpp/build/trot_sim; M=/home/jsh/quad_plant/quad/mjcf/quad_real_17dof_waist_sphere.mjcf
name=$1; shift
cd /home/jsh/quad_plant/quad/cpp
env EST_CTRL=1 SENSE_LAT_MS=4 ACT_LAT_MS=6 ENCQ_N=7.6e-5 ENCDQ_N=0.037 GYRO_N=0.002 GEARBOX=1 PRINT_EVERY=9999999 \
    "$@" $B $M > /home/jsh/quad_plant/runs/$name.log 2>&1
echo "$name | $(grep '종료' /home/jsh/quad_plant/runs/$name.log | sed 's/| [-0-9]* steps\/s//; s/=== 종료: //; s/ ===//')"
