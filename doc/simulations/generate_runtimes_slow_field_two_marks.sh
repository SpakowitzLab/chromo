#!/bin/bash

python -m cProfile -s "cumulative" -o "rstats_two_marks_slow_field_1" mc_two_marks_for_runtimes_slow_field.py &
python -m cProfile -s "cumulative" -o "rstats_two_marks_slow_field_2" mc_two_marks_for_runtimes_slow_field.py &
python -m cProfile -s "cumulative" -o "rstats_two_marks_slow_field_3" mc_two_marks_for_runtimes_slow_field.py &