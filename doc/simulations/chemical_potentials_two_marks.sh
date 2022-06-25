#!/bin/bash
# Run simulations varying the chemical potential.

chemical_potentials=(-2.5 -2.3 -2.0 -1.7 -1.5)
cross_talk_interact_energy=(-4.0 -2.0 0.0 2.0 4.0)

command=""

for cp in ${chemical_potentials[@]}
do
    for cross in ${cross_talk_interact_energy[@]}
    do
        (python mc_two_marks.py $cp $cp $cross) &
    done
done
