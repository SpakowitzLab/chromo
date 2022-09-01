#!/bin/bash

sim_ids=`seq 1 32`
for sim_id in ${sim_ids[@]}
do
  echo Sim: $sim_id
  python plot_cmap_new.py $sim_id
done

echo Done!