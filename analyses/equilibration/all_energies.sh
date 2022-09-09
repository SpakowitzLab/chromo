#!/bin/bash

# Generate energy convergence plots for all simulations in set

output_dir="/scratch/users/jwakim/chromo_two_mark_phase_transition/output"
sim_ids=($(seq 1 20))
save_name="energy_equilibration.png"

for id in ${sim_ids[@]}
do
    (python energies.py $output_dir $id $save_name) &
done