#!/bin/bash
# Run simulations varying the chemical potential.

chemical_potentials=(-2.5 -2.4 -2.3 -2.2 -2.1 -2 -1.9 -1.8 \
-1.7 -1.6)

command=""

for cp in ${chemical_potentials[@]}
do
    (python mc_varChemPotential.py $cp) &
done
