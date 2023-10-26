import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from elastic_desc import elastic_desc_flav

import seaborn as sns
colors=sns.color_palette('colorblind')

def ma(x, w):
  ma=np.convolve(x, np.ones(w), 'same')/w
  w1=w-1
  for i in range(0,w1-w1//2):
    ma[i]=np.mean(x[:w-(w1-w1//2-i)])
  for i in range(0,w1//2):
   ma[len(ma)-i-1]=np.mean(x[-(w-(w1//2-i)):])
  return ma

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
diab=datasets.load_diabetes()
X=diab.data
y=diab.target
scaler = StandardScaler()
X=scaler.fit_transform(X)

STEP_SIZE=0.01
ALPHA=0.5
ALGS =['std', 'sd_s', 'sd_ns', 'gs_s', 'gs_ns']
ALG_NAMES=['Standard', 'Steepest Descent,\nScaled', 'Steepest Descent,\nNon-scaled', 'Generalized Stagewise,\nScaled', 'Generalized Stagewise,\nNon-scaled']

LW=1.5
LW2=3
FS_LAB=10
FS_TITLE=11
FS_TICK=7

X_MIN=0
X_MAX=150
Y0_MIN=-15
Y0_MAX=30

MA=9


fig, axs = plt.subplots(len(ALG_NAMES), 3, figsize=(8,9))
for a, (alg, alg_name) in enumerate(zip(ALGS, ALG_NAMES)):
  beta_path, grad_path, h_alpha_path, p1s=elastic_desc_flav(X,y,ALPHA, alg, STEP_SIZE)
  ns=np.sum(np.abs(beta_path),1)
  for d in range(beta_path.shape[1]):
    axs[a,0].plot(ns,beta_path[:,d], linewidth=LW, color=colors[d])
    axs[a,1].plot(ns,grad_path[:,d]/np.max(np.abs(grad_path),1), linewidth=LW, color=colors[d])
  axs[a,0].set_ylim((Y0_MIN,Y0_MAX))
  axs[a,0].hlines(0,X_MIN,X_MAX, colors='k')
  axs[a,1].hlines(ALPHA,X_MIN,X_MAX, colors='k')
  axs[a,1].hlines(-ALPHA,X_MIN,X_MAX, colors='k')
  
  #Bounds, desired h_alpha
  if alg=='std':
    axs[a,2].hlines(1,X_MIN,X_MAX, colors='k', linewidth=LW2) #1
  elif alg=='sd_s':
    axs[a,2].hlines(1,X_MIN,X_MAX, colors='k', linewidth=LW2) #1
    axs[a,2].set_ylim((0.8,1.2))
  elif alg=='sd_ns':
    axs[a,2].plot(ns, ma(1+ALPHA*(1-ALPHA)*(np.sqrt(p1s)-1),MA), color=colors[3], linewidth=LW) # upper bound
    axs[a,2].plot(ns, ma(1-ALPHA*(1-ALPHA)*(2-ALPHA)*(1-1/p1s),MA),color=colors[3], linewidth=LW) #lower bound
    axs[a,2].hlines(1,X_MIN,X_MAX, colors='k', linewidth=LW2) #1
  elif alg=='gs_s':
    axs[a,2].hlines(STEP_SIZE,X_MIN,X_MAX, colors='k', linewidth=LW2) #delta t
    axs[a,2].set_ylim((0.8*STEP_SIZE,1.2*STEP_SIZE))
  elif alg=='gs_ns':
    axs[a,2].plot(ns, ma(STEP_SIZE*(1+ALPHA*(1-ALPHA)*(np.sqrt(p1s/STEP_SIZE)-1)),MA), color=colors[3], linewidth=LW) # upperbound
    axs[a,2].plot(ns, ma(STEP_SIZE*(1-ALPHA*(1-ALPHA)*(2-ALPHA)*(1-STEP_SIZE/p1s)),MA),color=colors[3], linewidth=LW) #lower bound
    axs[a,2].plot(ns, STEP_SIZE*np.ones(len(ns)), 'k', linewidth=LW2) #delta t
  
  axs[a,2].plot(ns,ma(h_alpha_path,MA), linewidth=LW, color=colors[0]) #norm
  
  
  #p1
  ax2=axs[a,2].twinx()
  ax2.plot(ns, ma(p1s,MA), color=colors[2], linewidth=1)
  ax2.set_ylim((0,10))
  ax2.tick_params(axis='y',colors=[0.6*.17,0.6*.63,0.6*.17])
  ax2.tick_params(axis='y',labelsize=FS_TICK)
  axs[a,2].set_zorder(ax2.get_zorder()+1) 
  axs[a,2].patch.set_visible(False)
  
  axs[a,0].set_ylabel(ALG_NAMES[a]+'\n$\\hat{\\beta}$', fontsize=FS_LAB)
  axs[a,1].set_ylabel('$g/||g||_\infty$',fontsize=FS_LAB)
  axs[a,2].set_ylabel('$h_\\alpha(\\Delta\\hat{\\beta})$',fontsize=FS_LAB)


for b in range(3):
  axs[len(ALGS)-1,b].set_xlabel('$||\\hat{\\beta}||_1$',fontsize=FS_LAB)
axs[0,0].set_title('$\\hat{\\beta}$ Path',fontsize=FS_TITLE)
axs[0,1].set_title('Gradient Path',fontsize=FS_TITLE)
axs[0,2].set_title('$h_\\alpha$ Path',fontsize=FS_TITLE)


for ax in axs.ravel():
  ax.tick_params(axis='x',labelsize=FS_TICK)
  ax.tick_params(axis='y',labelsize=FS_TICK)
  ax.set_xlim((X_MIN,X_MAX))

labs=['$h_\\alpha$','bounds','$p_1$']
lines=[]
for c in [0,3,2]:
  lines.append(Line2D([0],[0],color=colors[c],lw=2))

fig.legend(lines, labs, loc='lower right', ncol=3)

fig.tight_layout()
fig.subplots_adjust(bottom=0.09)

fig.savefig('figures/diab_paths_flav.pdf')

