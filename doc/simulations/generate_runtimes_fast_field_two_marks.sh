#!/bin/bash

python -m cProfile -s "cumulative" -o "rstats_two_marks_fast_field_1" mc_two_marks_for_runtimes_fast_field.py &
python -m cProfile -s "cumulative" -o "rstats_two_marks_fast_field_2" mc_two_marks_for_runtimes_fast_field.py &
python -m cProfile -s "cumulative" -o "rstats_two_marks_fast_field_3" mc_two_marks_for_runtimes_fast_field.py &