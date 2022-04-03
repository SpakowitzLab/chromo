#!/bin/bash
# Analyze binding states for a batch of simulations

sim_ids=(43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62)

command=""

for id in ${sim_ids[@]}
do
    (python analyze_binding_states.py -s $id -p HP1 -l 2) &
done
