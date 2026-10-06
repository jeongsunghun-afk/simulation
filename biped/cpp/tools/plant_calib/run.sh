#!/bin/bash
# 사용: run.sh <이름> <vx> [env...]   — 실기 유사 조건(지연 4+6ms·외삽보상 8.4ms·엔코더 잡음·추정기) + 실기 제어 기본값
B=/home/jsh/sim_elastic/biped/cpp/build_el/biped_sim; M=/home/jsh/sim_elastic/biped/biped_pointfoot_payload.mjcf
name=$1; vx=$2; shift 2
cd /home/jsh/sim_elastic/runs
env EST_CTRL=1 SENSE_LAT_MS=4 ACT_LAT_MS=6 LAT_COMP_MS=8.4 LAT_COMP_KIN=1 ENCQ_N=7.6e-5 ENCDQ_N=0.037 \
    WBIC_MIT=2 STANCE_KD=0 FLAT_STEPH=0.03 STAND_CZ=0.50 RET_TAU=1 MPC_ASYNC=2 MPC_LAG_TICKS=2 \
    SIMLOG=/home/jsh/sim_elastic/runs/$name.csv "$@" $B $M $vx ${SIMT:-15} > /home/jsh/sim_elastic/runs/$name.log 2>&1
echo "$name $(grep -o 'falls=[0-9]*' $name.log) $(grep -o '\[plant\] ★탄성[^·]*' $name.log | head -1)"
