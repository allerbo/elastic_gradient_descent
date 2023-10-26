#!/usr/bin/env bash
#SBATCH -A ###
#SBATCH -t 0:20:00
#SBATCH -o out_synth_rho.txt
#SBATCH -e err_synth_rho.txt
#SBATCH -n 1

python run_synth_rho.py $*
