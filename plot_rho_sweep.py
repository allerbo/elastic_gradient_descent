import sys, os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pickle

import seaborn as sns
colors=sns.color_palette('colorblind')

plt.rc('text', usetex=True)

FS_SUPTITLE=12
FS_TITLE=10
FS_TICKS=7
FS_RHO=8

ps=[50,100,200]
algs=['egdm','egd','en','cd']
metrics =['sens','spec','pred_err','est_err','time_spent']
rho1s=[0.5,0.6,0.7,0.8,0.9]
rho2s=[0.1,0.2,0.3,0.4,0.5]
seeds=range(5001)


data_dict={}
snr_dict={}
for seed in seeds:
  fi=open('data/synth_rho_'+str(seed)+'.pkl','rb')
  data_dict_seed=pickle.load(fi)
  fi.close()
  for p in ps:
    if not p in data_dict: data_dict[p]={}
    if not p in snr_dict: snr_dict[p]={}
    for alg in algs:
      if not alg in data_dict[p]: data_dict[p][alg]={}
      for rho1 in rho1s:
        if not rho1 in data_dict[p][alg]: data_dict[p][alg][rho1]={}
        for rho2 in rho2s:
          if not rho2 in data_dict[p][alg][rho1]: data_dict[p][alg][rho1][rho2]={}
          if not rho1 in snr_dict[p]: snr_dict[p][rho1]=[]
          snr_dict[p][rho1].append(data_dict_seed[p][alg][rho1][rho2]['snr'])
          for metric in metrics:
            if not metric in data_dict[p][alg][rho1][rho2]: data_dict[p][alg][rho1][rho2][metric]=[]
            data_dict[p][alg][rho1][rho2][metric].append(data_dict_seed[p][alg][rho1][rho2][metric])


def plot_med_quart(ax, p, metric, pane):
  rho1=rho1s[pane]
  x=range(5*pane,5*(pane+1))
  for alg,c in zip(['cd','en','egdm','egd'], [colors[7],colors[1],colors[0], colors[2]]):
    meds=[]
    q1s=[]
    q3s=[]
    for rho2 in rho2s:
      meds.append(np.median(np.array(data_dict[p][alg][rho1][rho2][metric])))
      q1s.append(np.quantile(np.array(data_dict[p][alg][rho1][rho2][metric]),0.25))
      q3s.append(np.quantile(np.array(data_dict[p][alg][rho1][rho2][metric]),0.75))
    ax.plot(x, meds, color=c, linewidth=0.7)
    ax.fill_between(x, q3s, q1s, color=c, alpha=0.3)

for p in ps:
  fig, axs = plt.subplots(5,1, figsize=(0.7*10,0.75*10))
  for ax, metric in zip(axs,metrics):
    for pane in range(5):
      plot_med_quart(ax, p, metric, pane)
  
  x_axis_labs=[]
  for rho1 in rho1s:
    for rho2 in rho2s:
      x_axis_labs.append(str(rho1)+'\n'+str(rho2))
  
  for ax in axs:
    ax.set_xticks(range(len(x_axis_labs)))
    ax.set_xticklabels(x_axis_labs)
    ax.tick_params(axis='x',labelsize=FS_TICKS)
    ax.tick_params(axis='y',labelsize=FS_TICKS)
  
  axs[4].set_yscale('log')
  snrs=[]
  for rho1 in rho1s:
    snrs.append(np.mean(snr_dict[p][rho1]))
  fig.suptitle('n=100, p='+str(p), fontsize=FS_SUPTITLE)
  if p==200:
    axs[0].set_title(f'SNR={snrs[0]:.3g}\qquad\qquad\ \ \  SNR={snrs[1]:.3g}\qquad\qquad\  SNR={snrs[2]:.3g}\qquad\qquad\  SNR={snrs[3]:.3g}\qquad\qquad\ \ \  SNR={snrs[4]:.3g}\n\nSensitivity', fontsize=FS_TITLE)
  else:
    axs[0].set_title(f'SNR={snrs[0]:#.3g}\qquad\qquad\  SNR={snrs[1]:#.3g}\qquad\qquad\  SNR={snrs[2]:#.3g}\qquad\qquad\  SNR={snrs[3]:#.3g}\qquad\qquad\  SNR={snrs[4]:#.3g}\n\nSensitivity', fontsize=FS_TITLE)
  axs[1].set_title('Specificity', fontsize=FS_TITLE)
  axs[2].set_title('Prediction Error, $\\frac{1}{n^*}||X^*(\\hat{\\beta}-\\beta^*)||_2$', fontsize=FS_TITLE)
  axs[3].set_title('Estimation Error, $\\frac{1}{p}||\\hat{\\beta}-\\beta^*||_2$', fontsize=FS_TITLE)
  axs[4].set_title('Execution Time [s]', fontsize=FS_TITLE)
  
  labs=['EGD, $\\gamma=0$', 'EGD, $\\gamma=0.5$', 'Elastic Net', 'Coordinate Descent']
  lines=[]
  for c in [colors[2],colors[0],colors[1], colors[7]]:
    lines.append(Line2D([0],[0],color=c,lw=2))
  
  fig.legend(lines, labs, loc='lower center', ncol=len(labs))
  
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.09)
  
  for ax in axs.ravel():
    ax.text(0,-.22,'$\\rho_1$', fontsize=FS_RHO, transform=ax.transAxes)
    ax.text(0,-.38,'$\\rho_2$', fontsize=FS_RHO, transform=ax.transAxes)
  
  fig.savefig('figures/rho_sweep_'+str(p)+'.pdf')

