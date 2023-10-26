#!/usr/bin/env bash
#SBATCH -A ###
#SBATCH -t 0:10:00
#SBATCH -o out_synth_p.txt
#SBATCH -e err_synth_p.txt
#SBATCH -n 1

python run_synth_p.py $*
