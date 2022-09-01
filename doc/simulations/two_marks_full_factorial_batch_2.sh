#!/bin/bash
# Run simulations varying the chemical potential.

cp_HP1=(-0.45 -0.35)
cp_PRC1=(-0.45 -0.35)
self_interact_HP1=(-4.0 -3.0)
self_interact_PRC1=(-4.0 -3.0)

command=""

pids=""
RESULT=0


for cp_HP1_ in ${cp_HP1[@]}
do
  for cp_PRC1_ in ${cp_PRC1[@]}
  do
    for self_interact_HP1_ in ${self_interact_HP1[@]}
    do
      for self_interact_PRC1_ in ${self_interact_PRC1[@]}
      do
        (python two_mark_factorial.py $cp_HP1_ $cp_PRC1_ $self_interact_HP1_ $self_interact_PRC1_ 0.5) &
        pids="$pids $!"
      done
    done
  done
done

for pid in $pids; do
    wait $pid || let "RESULT=1"
done

if [ "$RESULT" == "1" ];
    then
       exit 1
fi
