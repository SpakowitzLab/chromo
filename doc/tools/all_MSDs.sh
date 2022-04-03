#!/bin/bash

# Generate MSD plots for all simulations in set

sim_ids=($(seq 1 20))

for id in ${sim_ids[@]}
do
    (python MSD.py $id) &
done