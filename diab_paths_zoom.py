import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
sys.path.insert(1,'..')
from elastic_desc import elastic_desc
from sklearn.linear_model import enet_path

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import seaborn as sns
colors=sns.color_palette('colorblind')


diab=datasets.load_diabetes()
X=diab.data
y=diab.target
scaler = StandardScaler()
X=scaler.fit_transform(X)
labs=['age','sex','bmi','bp','tc','ldl','hdl','tch','ltg','glu']

beta_path_ed3, grad_path_ed3 = elastic_desc(X,y, 0.3)
beta_path_ed7, grad_path_ed7 = elastic_desc(X,y, 0.7)
beta_path_en3= enet_path(X, y, l1_ratio=0.3, eps=1e-5)[1].T
beta_path_en7= enet_path(X, y, l1_ratio=0.7, eps=1e-5)[1].T

beta_paths_ed=[beta_path_ed3, beta_path_ed7]
beta_paths_en=[beta_path_en3, beta_path_en7]
alphas=[0.3,0.7]


LW=1.7
FS_LAB=10
FS_TITLE=11
FS_TICK=7

X_MIN=0
X_MAX=150
Y0_MIN=-15
Y0_MAX=30
Y1_MIN=-1.5
Y1_MAX=1.5


  
  
fig, axs = plt.subplots(4,2, figsize=(8,8), gridspec_kw={'height_ratios': [2,2,1,1]})
for a, (beta_path_ed, beta_path_en, alpha) in enumerate(zip(beta_paths_ed, beta_paths_en, alphas)):
  ns_ed=np.sum(np.abs(beta_path_ed),1)
  ns_en=np.sum(np.abs(beta_path_en),1)
  for d in range(beta_path_ed.shape[1]):
    axs[0,a].plot(ns_ed,beta_path_ed[:,d], linewidth=LW, color=colors[d])
    axs[1,a].plot(ns_en,beta_path_en[:,d], linewidth=LW, color=colors[d])
    axs[2,a].plot(ns_ed,beta_path_ed[:,d], linewidth=LW, color=colors[d])
    axs[3,a].plot(ns_en,beta_path_en[:,d], linewidth=LW, color=colors[d])
  
  axs[0,a].set_title('$\\alpha=$'+str(alpha), fontsize=FS_TITLE)
  axs[3,a].set_xlabel('$||\\hat{\\beta}||_1$',fontsize=FS_LAB)
  axs[0,a].set_ylim((Y0_MIN,Y0_MAX))
  axs[1,a].set_ylim((Y0_MIN,Y0_MAX))
  axs[2,a].set_ylim((Y1_MIN,Y1_MAX))
  axs[3,a].set_ylim((Y1_MIN,Y1_MAX))

axs[0,0].set_ylabel('Elastic Gradient Descent\n$\\hat{\\beta}$',fontsize=FS_LAB)
axs[1,0].set_ylabel('Elastic Net\n$\\hat{\\beta}$',fontsize=FS_LAB)
axs[2,0].set_ylabel('Elastic Gradient Descent\n$\\hat{\\beta}$',fontsize=FS_LAB)
axs[3,0].set_ylabel('Elastic Net\n$\\hat{\\beta}$',fontsize=FS_LAB)

for ax in axs.ravel():
  ax.tick_params(axis='x',labelsize=FS_TICK)
  ax.tick_params(axis='y',labelsize=FS_TICK)
  ax.set_xlim((X_MIN,X_MAX))
  ax.hlines(0,X_MIN,X_MAX, colors='k') 

lines=[]
for c in range(len(labs)):
  lines.append(Line2D([0],[0],color=colors[c],lw=3))

fig.legend(lines, labs, loc='lower center', ncol=5)
fig.tight_layout()
fig.subplots_adjust(bottom=.13)
fig.savefig('figures/diab_paths_zoom.pdf')

