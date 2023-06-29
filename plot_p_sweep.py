import sys, os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import seaborn as sns
colors=sns.color_palette('colorblind')

plt.rc('text', usetex=True)
rho1s=[0.7]
rho2s=[0.3]
ps=range(50,201,10)

FS_SUPTITLE=12
FS_TITLE=10


def get_stat(lines, l):
  splits=lines[l].split(' ')
  return [float(splits[1]), float(splits[2]), float(splits[3])]

def make_p_dict():
  p_dict={}
  for alg in ['egd','egdm','en','cd']:
    p_dict[alg]={}
    for p in ps:
      file_name='logs/synth_'+str(p)+'_0.7_0.3_'+str(alg)+'_p.txt'
      f = open(file_name)
      lines = f.readlines()
      f.close()
      sens=get_stat(lines,3)
      spec=get_stat(lines,4)
      pred=get_stat(lines,6)
      est=get_stat(lines,7)
      time=get_stat(lines,8)
      snr=get_stat(lines,9)
      p_dict[alg][p]=(sens, spec, pred, est, time, snr)
  return p_dict


def plot_med_quant(ax, p_dict, tp):
  for alg,c in zip(['cd','en','egdm','egd'], [colors[7],colors[1],colors[0], colors[2]]):
    meds=[]
    q1s=[]
    q3s=[]
    for p in ps:
      try:
        meds.append(p_dict[alg][p][tp][0])
        q1s.append(p_dict[alg][p][tp][1])
        q3s.append(p_dict[alg][p][tp][2])
      except:
        meds.append(0)
        q1s.append(0)
        q3s.append(0)
    ax.plot(ps, meds, color=c, linewidth=0.7)
    ax.fill_between(ps, q3s, q1s, color=c, alpha=0.3)

p_dict=make_p_dict()

fig, axs = plt.subplots(5,1, figsize=(0.7*10,0.75*10))

for i, tp in enumerate([0,1,2,3,4]):
  plot_med_quant(axs[i], p_dict, tp)

for p in ps:
  print(p,p_dict['egd'][p][5])

axs[0].set_title('Sensitivity', fontsize=FS_TITLE)
axs[1].set_title('Specificity', fontsize=FS_TITLE)
axs[2].set_title('Prediction Error, $\\frac{1}{n^*}||X^*(\\hat{\\beta}-\\beta^*)||_2$', fontsize=FS_TITLE)
axs[3].set_title('Estimation Error, $\\frac{1}{p}||\\hat{\\beta}-\\beta^*||_2$', fontsize=FS_TITLE)
axs[4].set_title('Execution Time [s]', fontsize=FS_TITLE)

axs[4].set_xlabel('p')
axs[3].set_yscale('log')
axs[2].set_yscale('log')
axs[4].set_yscale('log')
axs[2].set_yticks([1,10])
axs[2].set_ylim([.3,10])
axs[3].set_ylim([.032,.33])
labs=['EGD, $\\gamma=0$', 'EGD, $\\gamma=0.5$', 'Elastic Net', 'Coordinate Descent']
lines=[]
for c in [colors[2],colors[0],colors[1], colors[7]]:
  lines.append(Line2D([0],[0],color=c,lw=2))

fig.legend(lines, labs, loc='lower center', ncol=len(labs))

fig.tight_layout()
fig.subplots_adjust(bottom=0.1)

fig.savefig('figures/p_sweep.pdf')
