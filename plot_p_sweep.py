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

datas=['syn','syn2']
algs=['egdm','egd','en','cd']
metrics =['sens','spec','pred_err','est_err','time_spent']
ps = range(50,201,10)
seeds=range(5001)


snr_dict={}
data_dict={}
for seed in seeds:
  fi=open('data/synth_p_'+str(seed)+'.pkl','rb')
  data_dict_seed=pickle.load(fi)
  fi.close()
  for data in datas:
    if not data in data_dict: data_dict[data]={}
    if not data in snr_dict: snr_dict[data]={}
    for alg in algs:
      if not alg in data_dict[data]: data_dict[data][alg]={}
      for metric in metrics:
        if not metric in data_dict[data][alg]: data_dict[data][alg][metric]={}
        for p in ps:
          if not p in data_dict[data][alg][metric]: data_dict[data][alg][metric][p]=[]
          data_dict[data][alg][metric][p].append(data_dict_seed[data][alg][p][metric])
          if not p in snr_dict[data]: snr_dict[data][p]=[]
          snr_dict[data][p].append(data_dict_seed[data][alg][p]['snr'])


def plot_med_quart(ax, data, metric):
  for alg,c in zip(['cd','en','egdm','egd'], [colors[7],colors[1],colors[0], colors[2]]):
    meds=[]
    q1s=[]
    q3s=[]
    for p in ps:
      meds.append(np.median(np.array(data_dict[data][alg][metric][p])))
      q1s.append(np.quantile(np.array(data_dict[data][alg][metric][p]),0.25))
      q3s.append(np.quantile(np.array(data_dict[data][alg][metric][p]),0.75))
    ax.plot(ps, meds, color=c, linewidth=0.7)
    ax.fill_between(ps, q3s, q1s, color=c, alpha=0.3)

for data in ['syn','syn2']:
  fig, axs = plt.subplots(5,1, figsize=(0.7*10,0.75*10))
  for ax, metric in zip(axs,metrics):
    plot_med_quart(ax, data, metric)
  
  for p in ps:
    print(data,p,np.mean(snr_dict[data][p]))
  
  axs[0].set_title('Sensitivity', fontsize=FS_TITLE)
  axs[1].set_title('Specificity', fontsize=FS_TITLE)
  axs[2].set_title('Prediction Error, $\\frac{1}{n^*}||X^*(\\hat{\\beta}-\\beta^*)||_2$', fontsize=FS_TITLE)
  axs[3].set_title('Estimation Error, $\\frac{1}{p}||\\hat{\\beta}-\\beta^*||_2$', fontsize=FS_TITLE)
  axs[4].set_title('Execution Time [s]', fontsize=FS_TITLE)
  
  axs[4].set_xlabel('p')
  axs[3].set_yscale('log')
  axs[2].set_yscale('log')
  axs[4].set_yscale('log')
  axs[2].set_ylim([.3,10])
  axs[3].set_ylim([.025,.4])
  labs=['EGD, $\\gamma=0$', 'EGD, $\\gamma=0.5$', 'Elastic Net', 'Coordinate Descent']
  lines=[]
  for c in [colors[2],colors[0],colors[1], colors[7]]:
    lines.append(Line2D([0],[0],color=c,lw=2))
  
  fig.legend(lines, labs, loc='lower center', ncol=len(labs))
  
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.1)
  
  if data=='syn2':
    fig.savefig('figures/p_sweep2.pdf')
  else:
    fig.savefig('figures/p_sweep.pdf')

