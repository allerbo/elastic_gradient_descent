for p in 200 100 50
do
  for rho2 in 0.1 0.2 0.3 0.4 0.5
  do
    for rho1 in 0.5 0.6 0.7 0.8 0.9
    do
      for ALG in 0 1 2
      do
        python3 run_exps.py ALG=$ALG p=$p rho1=$rho1 rho2=$rho2 $*
      done
    done
  done
done
