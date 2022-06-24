#!/bin/bash
# Run simulations varying the chemical potential.

interact_energy=(-4.0 -3.0 -2.0 -1.0 0.0)
cross_talk_interact_energy=(-4.0 -2.0 0.0 2.0 4.0)

command=""

for ie in ${interact_energy[@]}
do
    for cross in ${cross_talk_interact_energy[@]}
    do
        (python mc_two_marks_v2.py $ie $ie $cross) &
    done
done
