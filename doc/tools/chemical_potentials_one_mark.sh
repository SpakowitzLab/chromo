#!/bin/bash
# Run simulations varying the chemical potential.
# Results in over 50% fraction tail occupancy

chemical_potentials=(-2.5 -2.4 -2.3 -2.2 -2.1 -2.0 -1.9 -1.8 -1.7 -1.6 -1.5 -1.4 -1.3 -1.2 -1.1 -1.0 -0.9 -0.8 -0.7 -0.6 -0.5)

command=""

for cp in ${chemical_potentials[@]}
do
    (python mc_one_mark.py $cp) &
done
