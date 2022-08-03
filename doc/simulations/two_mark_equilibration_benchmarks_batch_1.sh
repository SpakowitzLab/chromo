#!/bin/bash

chain_lengths=(1000 2000 3000 4000)
repeats=(1 2 3)

for length in ${chain_lengths[@]}
do
    for repeat in ${repeats[@]}
    do
        python mc_two_marks_for_equilibration_benchmark.py $length &
    done
    wait
done