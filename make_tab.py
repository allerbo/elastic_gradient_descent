import numpy as np
import sys
import math
from sklearn import datasets
import pandas as pd
data_sets=['wood','house','black','elect','energy','casp','super','read','twitter']

def round2(x):
  mults=0
  divs=0
  xt=x
  while xt<10 or xt>=100:
    if xt<10:
      xt*=10
      mults+=1
    if xt>=100:
      xt/=10
      divs+=1
  
  x_r=int(round(xt,0))
  x_str=str(x_r*10**divs*.1**mults)+'0000'
  if x>=10:
    x_out=x_str[:x_str.find('.')]
  elif x>=1:
    x_out=x_str[:3]
  else:
    x_out=x_str[:(x_str.find(str(x_r))+2)]
  
  return x_out

title_dict={}
data_dict={}

data_set='house'
house=datasets.fetch_california_housing()
data=np.hstack((house.data,house.target.reshape((-1,1))))
X=data[:,:-1]
data_dict[data_set]=X.shape
title_dict[data_set]='California\\\\Housing'

data_set='black'
data=pd.read_csv('in_data/bs_2000.csv',sep=',').to_numpy()
X=data[:,1:11]
data_dict[data_set]=X.shape
title_dict[data_set]='U.K.\ Black\\\\Smoke'

data_set='casp'
data=pd.read_csv('in_data/CASP.csv',sep=',').to_numpy()
X=data[:,1:]
data_dict[data_set]=X.shape
title_dict[data_set]='Protein\\\\Structure'

data_set='wood'
data=pd.read_csv('in_data/wood-fibres.csv',sep=',').to_numpy()
X=data[:,1:]
data_dict[data_set]=X.shape
title_dict[data_set]='Aspen\\\\Fibres'

data_set='super'
data=pd.read_csv('in_data/superconduct.csv',sep=',').to_numpy()
X=data[:,:-1]
data_dict[data_set]=X.shape
title_dict[data_set]='Super-\\\\conductors'

data_set='energy'
data=pd.read_csv('in_data/energydata_complete.csv',sep=',').iloc[:,1:].to_numpy()
X=data[:,1:]
data_dict[data_set]=X.shape
title_dict[data_set]='Appliances\\\\Energy Use'

data_set='elect'
data=pd.read_csv('in_data/ElectionData.csv',sep=',').iloc[:,list(range(3,21))+list(range(22,28))].to_numpy()
X=data[:,:-1]
X=np.delete(X,[18,19,20,21,22],1) #too easy if present
data_dict[data_set]=X.shape
title_dict[data_set]='Portugese\\\\Elections'

data_set='twitter'
data=pd.read_csv('in_data/Twitter.csv',sep=',').iloc[:291624,:].to_numpy() #takes too long if using all
X=data[:,:-1]
data_dict[data_set]=X.shape
title_dict[data_set]='Twitter\\\\Popularity'

data_set='read'
data=pd.read_csv('in_data/readability_features.csv',sep=',').to_numpy()
X=data[:,:-1]
data_dict[data_set]=X.shape
title_dict[data_set]='English\\\\Readability'

for data_set in data_sets:
  for alg in ['egd','egdm','en','cd']:
    file_name='logs/'+data_set+'_'+str(alg)+'.txt'
    f = open(file_name)
    lines = f.readlines()
    
    betas=[]
    r2s=[]
    times=[]
    for metric, line_n in zip([betas,r2s,times],[3,4,5]):
	    metric.extend([float(lines[line_n].split(' ')[1]), 
                     float(lines[line_n].split(' ')[2]),
                     float(lines[line_n].split(' ')[3])])
    n,p=data_dict[data_set]
    if alg=='egd':
      out_str =('\\multirow{4}{*}{\\makecell{'+title_dict[data_set]+'\\\\$('+str(n)+' \\times '+str(p)+')$}} & EGD, $\\gamma=0$').ljust(70,' ')+' & '
    elif alg=='egdm':
      out_str = ('& EGD, $\\gamma='+lines[1].split(' ')[4][:-3]+'$').ljust(70,' ')+' & '
    elif alg=='cd':
      out_str = '& CD'.ljust(70,' ')+' & '
    elif alg=='en':
      out_str = '& Elastic Net'.ljust(70,' ')+' & '
    
    out_str += '$'+round2(times[0])+',\\ ('+round2(times[1])+', '+round2(times[2])+')$ & $'+round2(r2s[0])+',\\ ('+round2(r2s[1])+', '+round2(r2s[2])+')$ & $'+str(int(betas[0]))+',\\ ('+str(int(betas[1]))+', '+str(int(betas[2]))+')$ \\\\'
    print(out_str)
  print('\\hline')
  
