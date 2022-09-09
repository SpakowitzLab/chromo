#!/bin/bash

outdir="/scratch/users/jwakim/chromo_two_mark_phase_separation/output"
num_equilibration=180
sim_ids=`seq 1 32`
for sim_id in ${sim_ids[@]}
do
  echo Sim: $sim_id
  python cmap_approx.py $outdir $sim_id $num_equilibration
done

echo Done!
