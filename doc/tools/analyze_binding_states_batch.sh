#!/bin/bash
# Analyze binding states for a batch of simulations

sim_ids=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)

command=""

for id in ${sim_ids[@]}
do
    (python analyze_binding_states.py -s $id -p HP1 -l 2) &
done
