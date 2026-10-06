#!/bin/bash
cd /home/jsh/sim_elastic/runs
pool(){ while [ $(jobs -rp | wc -l) -ge 16 ]; do sleep 0.3; done; }
HW="WALK_TRACK=1 TRK_KD=1.5 WALK_FF_SCALE=0.75 WALK_FF_LPF_HZ=10 WALK_FF_NOTCH_HZ=15 WALK_FF_NOTCH_Q=4 PLANT_DRV_CH=42"
for sd in 1 2; do
  pool; SIMT=15 ./run.sh S1_rigid_s$sd 0 SEED=$sd $HW > /dev/null &
  for sc in 1 2 4 8 16; do for z in 0.05 0.3; do
    K=$(python3 -c "print(','.join(str(int(v*$sc)) for v in (400,400,500,330)))")
    pool; SIMT=15 ./run.sh S1_k${sc}_z${z}_s$sd 0 SEED=$sd $HW EL_K=$K EL_ZETA=$z EL_BL_DEG=0.5,0.1,0.1,0.1 > /dev/null &
  done; done
done; wait; echo DONE
