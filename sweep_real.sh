for ALG in 0 1 2 4
do
  echo $ALG
  python3 run_exps.py ALG=$ALG DATA=\"diab\"
  python3 run_exps.py ALG=$ALG DATA=\"bs\"
done
