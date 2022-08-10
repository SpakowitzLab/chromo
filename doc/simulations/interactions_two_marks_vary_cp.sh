#!/bin/bash
# Run simulations varying the chemical potential.

chemical_potential=(-0.4 -0.8 -1.2 -1.6)
cross_talk_interact_energy=(-2.0 -1.0 0.0 1.0 2.0)

command=""

for cp in ${chemical_potential[@]}
do
    for cross in ${cross_talk_interact_energy[@]}
    do
        (python mc_two_marks_v3.py $cp -4.0 $cross) &
    done
done