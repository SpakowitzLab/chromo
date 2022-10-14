#!/bin/bash

cd ../../output/sim_$1
lines=$(< "binders" wc -l)
cd ../../simulations/examples

if [ $lines -eq 3 ]; then
    python two_mark_factorial_refine_more_snaps.py $1
elif [ $lines -eq 2 ]; then
    python one_mark_factorial_refine_more_snaps.py $1
fi
