#!/bin/bash
# Run simulations varying the chemical potential.
# Attempt to reproduce Langmuir Isotherm

chemical_potentials=(-0.4 -0.3 -0.2 -0.1 -0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)

command=""

for cp in ${chemical_potentials[@]}
do
    (python mc_one_mark.py $cp) &
done
