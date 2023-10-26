for seed in {0..5000}
do
  sbatch start_synth_rho.sh seed=$seed $*
done
