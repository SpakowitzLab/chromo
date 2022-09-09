#!/bin/bash

# Generate MSD plots for all simulations in set

output_dir="/scratch/users/jwakim/chromo_two_mark_phase_transition/output"
sim_ids=($(seq 1 20))
seg_length=50
lp=53
save_name="MSD_equilibration.png"

for id in ${sim_ids[@]}
do
    (python MSDs.py $output_dir $id $seg_length $lp $save_name) &
done