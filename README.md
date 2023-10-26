This is the code used in the article **Elastic Gradient Descent, an Iterative Optimization Method Approximating the Solution Paths of the Elastic Net**, available at http://arxiv.org/abs/2202.02146.


## Figure 1:
```
python path_demo.py              #Uses elastic_desc.py, coordinate_desc.py and gradient_desc.py.
```

## Figure 2:
```
python diff_demo.py              #Uses elastic_desc.py.
```

## Figure 3:
```
python diab_paths_zoom.py        #Uses elastic_desc.py.
```

## Figure 4:
```
python diab_paths_grad.py        #Uses coord_desc.py, elastic_desc.py and elastic_flow.py.
```

## Table 2:
```
bash sweep_real.sh               #Calls run_exps.py. Uses data from in_data/.
python make_tab.py               #Creates table from data generated by run_exps.py.
```

## Figures 5 and 8:
```
bash sweep_synth_p.sh            #Calls run_synth_p.py.
python plot_p_sweep.py           #Creates figure from data generated by run_synth_p.py.
```

## Figure 6
```
python diab_path_flavs.py        #Uses elastic_desc.py.
```

## Figure 7:
```
python path_demo_momentum.py     #Uses elastic_desc.py
```

## Figures 9, 10 and 11:
```
bash sweep_synth_rho.sh          #Calls run_synth_rho.py.
python plot_rho_sweep.py         #Creates figures from data generated by run_synth_rho.py.
```

