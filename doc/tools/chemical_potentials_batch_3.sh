#!/bin/bash
# Run simulations varying the chemical potential.
# Third batch of chemical potentials
# We attempt to span sufficient chemical potentials to observe the Langmuir Isotherm

chemical_potentials=(-1.5 -1.4 -1.3 -1.2 -1.1 -1.0 -0.9 -0.8 -0.7 -0.6 -0.5)

command=""

for cp in ${chemical_potentials[@]}
do
    (python mc_varChemPotential.py $cp) &
done
