for p in {50..200..10}
do
  echo $p
  for ALG in egdm egd en cd
  do
    python3 run_exps.py ALG=\"$ALG\" p=$p rho1=0.7 rho2=0.3 SUF=\"_p\" $*
  done
done
