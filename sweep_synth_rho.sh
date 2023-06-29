for p in 50 100 200
do
  for ALG in egdm egd en cd
  do
    for rho2 in 0.1 0.2 0.3 0.4 0.5
    do
      for rho1 in 0.5 0.6 0.7 0.8 0.9
      do
        python3 run_exps.py ALG=\"$ALG\" p=$p rho1=$rho1 rho2=$rho2 SUF=\"_rho\" $*
      done
    done
  done
done
