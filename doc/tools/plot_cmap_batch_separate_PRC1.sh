#!/bin/bash
# Run simulations varying the chemical potential.
# Second batch of chemical potentials

sim_nums=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26)

command=""

for sim in ${sim_nums[@]}
do
    (python plot_cmap_PRC1_contacts.py $sim) &
done