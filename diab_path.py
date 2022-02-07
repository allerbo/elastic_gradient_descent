import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import sys
sys.path.insert(1,'..')
from elastic_desc import elastic_desc
from sklearn.linear_model import enet_path

diab=datasets.load_diabetes()
X=diab.data
y=diab.target
labs=diab.feature_names
scaler = StandardScaler()
X=scaler.fit_transform(X)
labs=['age','sex','bmi','bp','tc','ldl','hdl','tch','ltg','glu']

LW=1
FS_LAB=10
FS_TITLE=12
FS_TICK=7

X_MIN=0
X_MAX=150
Y0_MIN=-15
Y0_MAX=30
Y1_MIN=-1.5
Y1_MAX=1.5

ALPHAS=[0.3,0.7]
STEP_SIZE=0.01




fig, axs = plt.subplots(4,len(ALPHAS), figsize=(10,8), gridspec_kw={'height_ratios': [2,2,1,1]})
ls=[]
for a in range(len(ALPHAS)):
  alpha=ALPHAS[a]
  beta_path_eg= elastic_desc(X,y,alpha, STEP_SIZE, 'norm')[0]
  beta_path_en= enet_path(X, y, l1_ratio=alpha, eps=1e-5, fit_intercept=False)[1].T
  print('EG, '+str(alpha)+': '+str(len(np.unique(list(map(lambda i: np.array2string(1*(beta_path_eg!=0)[i,:]), range(beta_path_eg.shape[0])))))-1))
  print('EN, '+str(alpha)+': '+str(len(np.unique(list(map(lambda i: np.array2string(1*(beta_path_en!=0)[i,:]), range(beta_path_en.shape[0])))))-1))
  
  ns_eg=np.sum(np.abs(beta_path_eg),1)
  ns_en=np.sum(np.abs(beta_path_en),1)
  
  for d in range(beta_path_eg.shape[1]):
    if a==0:
      ls.append(axs[0,a].plot(ns_eg,beta_path_eg[:,d], linewidth=LW)[0])
    else:
      axs[0,a].plot(ns_eg,beta_path_eg[:,d], linewidth=LW)
    axs[1,a].plot(ns_en,beta_path_en[:,d], linewidth=LW)
    axs[2,a].plot(ns_eg,beta_path_eg[:,d], linewidth=LW)
    axs[3,a].plot(ns_en,beta_path_en[:,d], linewidth=LW)
  
  
  axs[0,a].set_title('$\\alpha=$'+str(alpha), fontsize=FS_TITLE)
  axs[3,a].set_xlabel('$||\\beta||_1$',fontsize=FS_LAB)
  axs[0,a].set_ylim((Y0_MIN,Y0_MAX))
  axs[1,a].set_ylim((Y0_MIN,Y0_MAX))
  axs[2,a].set_ylim((Y1_MIN,Y1_MAX))
  axs[3,a].set_ylim((Y1_MIN,Y1_MAX))

axs[0,0].set_ylabel('Elastic Gradient Descent\n$\\beta$',fontsize=FS_LAB)
axs[1,0].set_ylabel('Elastic Net\n$\\beta$',fontsize=FS_LAB)
axs[2,0].set_ylabel('Elastic Gradient Descent\n$\\beta$',fontsize=FS_LAB)
axs[3,0].set_ylabel('Elastic Net\n$\\beta$',fontsize=FS_LAB)

for ax in axs.ravel():
  ax.tick_params(axis='x',labelsize=FS_TICK)
  ax.tick_params(axis='y',labelsize=FS_TICK)
  ax.set_xlim((X_MIN,X_MAX))
  ax.hlines(0,X_MIN,X_MAX, colors='k') 


fig.legend(ls, labs, loc='lower center', ncol=10)
fig.tight_layout()
fig.subplots_adjust(bottom=.1)
fig.savefig('figures/diab_path.pdf')

