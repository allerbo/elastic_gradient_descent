for DATA in wood house black elect energy casp super read twitter 
do
  for ALG in egdm egd en cd
  do
    echo $DATA $ALG
    python3 run_exps.py ALG=\"$ALG\" DATA=\"$DATA\"
  done
done
