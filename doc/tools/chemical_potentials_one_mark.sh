#!/bin/bash
# Run simulations varying the chemical potential.

chemical_potentials=(-2.5 -2.3 -2.0 -1.7 -1.5)

command=""

for cp in ${chemical_potentials[@]}
do
    (python mc_one_mark.py $cp) &
done
