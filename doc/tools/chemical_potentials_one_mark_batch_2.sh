#!/bin/bash
# Run simulations varying the chemical potential.
# Attempt to reproduce Langmuir Isotherm

chemical_potentials=(-4.6 -4.5 -4.4 -4.3 -4.2 -4.1 -4.0 -3.9 -3.8 -3.7 -3.6 -3.5 -3.4 -3.3 -3.2 -3.1 -3.0 -2.9 -2.8 -2.7 -2.6)

command=""

for cp in ${chemical_potentials[@]}
do
    (python mc_one_mark.py $cp) &
done
