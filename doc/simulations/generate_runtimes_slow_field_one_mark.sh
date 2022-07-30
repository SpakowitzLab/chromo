#!/bin/bash

python -m cProfile -s "cumulative" -o "rstats_one_mark_slow_field_1" mc_one_mark_for_runtime_slow_field.py &
python -m cProfile -s "cumulative" -o "rstats_one_mark_slow_field_2" mc_one_mark_for_runtime_slow_field.py &
python -m cProfile -s "cumulative" -o "rstats_one_mark_slow_field_3" mc_one_mark_for_runtime_slow_field.py &