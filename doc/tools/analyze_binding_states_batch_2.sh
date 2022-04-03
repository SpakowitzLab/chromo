#!/bin/bash
# Analyze binding states for a batch of simulations

sim_ids=(22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42)

command=""

for id in ${sim_ids[@]}
do
    (python analyze_binding_states.py -s $id -p HP1 -l 2) &
done
