import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'..')
from coordinate_desc import coord_desc
from elastic_desc import elastic_desc
from elastic_flow import elastic_flow
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


beta_path_cd, grad_path_cd= coord_desc(X,y)

beta_path_ed, grad_path_ed = elastic_desc(X,y, 0.5)
beta_path_ef, grad_path_ef, _, _= elastic_flow(X,y, 0.5)
beta_path_en= enet_path(X, y, l1_ratio=0.5, eps=1e-5, fit_intercept=False)[1].T

beta_paths=[beta_path_cd, beta_path_ed, beta_path_en]
grad_paths=[grad_path_cd, grad_path_ed, grad_path_ef]
ns_beta=[np.sum(np.abs(beta_path_cd),1), np.sum(np.abs(beta_path_ed),1), np.sum(np.abs(beta_path_en),1)]
ns_grad=[np.sum(np.abs(beta_path_cd),1), np.sum(np.abs(beta_path_ed),1), np.sum(np.abs(beta_path_ef),1)]
alphas=[1,0.5,0.5]
ALG_NAMES=['Coordinate Descent', 'Elastic Gradient Descent, $\\alpha=0.5$', 'Elastic Net, $\\alpha=0.5$']


LW=1.7
FS_LAB=10
FS_TITLE=11
FS_TICK=7

X_MIN=0
X_MAX=150
Y0_MIN=-15
Y0_MAX=30


  
  
fig, axs = plt.subplots(len(ALG_NAMES), 2, figsize=(8,8))
for a, (beta_path, grad_path, alg_name, alpha, n_beta, n_grad) in enumerate(zip(beta_paths, grad_paths, ALG_NAMES, alphas, ns_beta, ns_grad)):
  for d in range(beta_path.shape[1]):
    axs[a,0].plot(n_beta,beta_path[:,d], linewidth=LW, color=colors[d])
  axs[a,0].set_ylim((Y0_MIN,Y0_MAX))
  axs[a,0].hlines(0,X_MIN,X_MAX, colors='k')
  axs[a,0].set_ylabel(alg_name+'\n$\\hat{\\beta}$', fontsize=FS_LAB)
  axs[a,0].set_xlabel('$||\\hat{\\beta}||_1$',fontsize=FS_LAB)
  for d in range(grad_path.shape[1]):
    axs[a,1].plot(n_grad,grad_path[:,d]/np.max(np.abs(grad_path),1), linewidth=LW, color=colors[d])
  axs[a,1].hlines(alpha,X_MIN,X_MAX, colors='k')
  axs[a,1].hlines(-alpha,X_MIN,X_MAX, colors='k')
  axs[a,1].set_ylabel('$g/||g||_\infty$',fontsize=FS_LAB)
  axs[a,1].set_xlabel('$||\\hat{\\beta}||_1$',fontsize=FS_LAB)


axs[len(ALG_NAMES)-1,0].set_xlabel('$||\\hat{\\beta}||_1$',fontsize=FS_LAB)
axs[len(ALG_NAMES)-1,1].set_xlabel('$||\\hat{\\beta}||_1$',fontsize=FS_LAB)

axs[0,0].set_title('$\\hat{\\beta}$ Path',fontsize=FS_TITLE)
axs[0,1].set_title('Gradient Path',fontsize=FS_TITLE)

for ax in axs.ravel():
  ax.tick_params(axis='x',labelsize=FS_TICK)
  ax.tick_params(axis='y',labelsize=FS_TICK)
  ax.set_xlim((X_MIN,X_MAX))


fig.tight_layout()
fig.savefig('figures/diab_paths_grad.pdf')
