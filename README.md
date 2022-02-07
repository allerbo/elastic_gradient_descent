This is the code used in the article **Elastic Gradient Descent and Elastic Gradient Flow: LARS Like Algorithms Approximating the Solution Paths of the Elastic Net**, available at http://arxiv.org/abs/2202.02146.


## Figure 1:
```
python diff_demo.py              #Uses elastic_desc.py.
```

## Figures 2 and 3:
```
julia run_synth.jl               #Uses functions in elastic_gd.jl to generate 
                                 # data for figures.
python plot_rho_sweep.py         #Creates figures from data generated above.
```

## Figure 4
```
python diab_path.py              #Uses elastic_desc.py.
```

## Figure 5
```
python diab_path_flavs.py        #Uses elastic_desc.py and elastic_flow.py.
```

## Additional Figures
```
python run_diabetes_cood.py      #Provides solutions paths for the diabetes data
                                 # using coordinate descent and flow according
                                 # to Equations 15 and 18.
                                 # Uses coordinate_desc.py and coordinate_flow.py.
```

```
python run_diabetes_elastic.py   #Provides solutions paths for the diabetes data
                                 # using elastic gradient descent and flow
                                 # according to Equations 19 and 21.
                                 # Uses elastic_desc1.py and elastic_flow.py.
```
