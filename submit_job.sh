#!/bin/bash
$SBATCH --job-name=chromo
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
source ~/.bashrc
conda activate chromo
cd /home/users/ahirsch1/Chromo_Github/chromo/chromo
python run_simulation.py
echo 'Job complete'#!/bin/bash
o
