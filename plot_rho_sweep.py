import sys, os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import seaborn as sns
colors=sns.color_palette('colorblind')
COLS=[colors[2],colors[0], colors[7]]

plt.rc('text', usetex=True)
rho1s=[0.5, 0.6, 0.7, 0.8, 0.9]
rho2s=[0.1, 0.2, 0.3, 0.4, 0.5]

FS_SUPTITLE=12
FS_TITLE=10
FS_TICKS=7
FS_RHO=8

suf=''

N_EXPS=1000
def get_stat(lines, l):
  splits=lines[l].split(' ')
  return [float(splits[1]), float(splits[2])]

def make_rho_dict(P):
  rho_dict={}
  snr_dict={}
  for alg in range(25):
    rho_dict[alg]={}
    for rho1 in rho1s:
      rho_dict[alg][rho1]={}
      if not rho1 in snr_dict.keys():
        snr_dict[rho1]=[]
      for rho2 in rho2s:
        try:
          file_name='../logs/synth_'+str(P)+'_'+str(rho1)+'_'+str(rho2)+'_'+str(alg)+suf+'.txt'
          f = open(file_name)
          lines = f.readlines()
          f.close()
          start_line=lines.index('Algorithm: '+str(alg)+'\n')
          sens=get_stat(lines,start_line+3)
          spec=get_stat(lines,start_line+4)
          r2=get_stat(lines,start_line+5)
          pred=get_stat(lines,start_line+6)
          est=get_stat(lines,start_line+7)
          time=get_stat(lines,start_line+8)
          snr=get_stat(lines,start_line+9)
          rho_dict[alg][rho1][rho2]=(sens, spec, pred, est, time, snr)
          snr_dict[rho1].append(snr[0])
        except:
          pass
  return rho_dict, snr_dict


def plot_mean_std(ax, rho_dict, pane, tp):
  rho1=rho1s[pane]
  x=range(5*pane,5*(pane+1))
  for alg,c in zip([0,1,2], COLS):
    means=[]
    stds=[]
    for rho2 in rho2s:
      try:
        means.append(rho_dict[alg][rho1][rho2][tp][0])
        stds.append(rho_dict[alg][rho1][rho2][tp][1])
      except:
        means.append(0)
        stds.append(0)
    #l1=ax.plot(x, means, color=c, linewidth=0.7)[0]
    ax.plot(x, means, color=c, linewidth=0.7)
    ax.fill_between(x, list(map(lambda x,y:x+y,means,3*stds/np.sqrt(N_EXPS))), list(map(lambda x,y:x-y,means,3*stds/np.sqrt(N_EXPS))), color=c, alpha=0.5)
    #ax.plot(x, list(map(lambda x,y:x+y,means,stds/np.sqrt(1000))), color=c, linestyle='--', linewidth=0.7)
    #ax.plot(x, list(map(lambda x,y:x-y,means,stds/np.sqrt(1000))), color=c, linestyle='--', linewidth=0.7)

for P in [50,100,200]:
  rho_dict,snr_dict=make_rho_dict(P)
  
  fig1, axs1 = plt.subplots(5,1, figsize=(0.7*10,0.75*10))
  for i, tp in enumerate([0,1,2,3,4]):
    for pane in range(5):
      plot_mean_std(axs1[i], rho_dict, pane, tp)
  
  x_axis_labs=[]
  for rho1 in rho1s:
    for rho2 in rho2s:
      x_axis_labs.append(str(rho1)+'\n'+str(rho2))
  
  for tp in range(5):
    axs1[tp].set_xticks(range(len(x_axis_labs)))
    axs1[tp].set_xticklabels(x_axis_labs)
    axs1[tp].tick_params(axis='x',labelsize=FS_TICKS)
    axs1[tp].tick_params(axis='y',labelsize=FS_TICKS)
  
  #axs1[2].set_yscale('log')
  #print(axs1[3].get_yticks())
  #axs1[3].set_yscale('log')
  #axs1[3].set_yticks([6,10])
  #print(axs1[3].get_yticks())
  #axs1[3].set_yticks(axs1[3].get_yticks())
  axs1[4].set_yscale('log')
#  axs1[3].set_yticklabels([7,10])
  snrs=[]
  for rho1 in rho1s:
    snrs.append(np.mean(snr_dict[rho1]))
  fig1.suptitle('n=100, p='+str(P), fontsize=FS_SUPTITLE)
  if P==200:
    axs1[0].set_title(f'snr={snrs[0]:.3g}\qquad\qquad\quad\ \  snr={snrs[1]:.3g}\qquad\qquad\quad\ \  snr={snrs[2]:.3g}\qquad\qquad\quad\ \  snr={snrs[3]:.3g}\qquad\qquad\quad\ \  snr={snrs[4]:.3g}\n\nSensitivity', fontsize=FS_TITLE)
  else:
    axs1[0].set_title(f'snr={snrs[0]:#.3g}\qquad\qquad\quad\ \  snr={snrs[1]:#.3g}\qquad\qquad\quad\ \  snr={snrs[2]:#.3g}\qquad\qquad\quad\ \  snr={snrs[3]:#.3g}\qquad\qquad\quad\ \  snr={snrs[4]:#.3g}\n\nSensitivity', fontsize=FS_TITLE)
  axs1[1].set_title('Specificity', fontsize=FS_TITLE)
  #axs1[2].set_title('R$^2$', fontsize=FS_TITLE)
  axs1[2].set_title('Prediction Error, $\\frac{1}{n^*}||X^*(\\hat{\\beta}-\\beta^*)||_2$', fontsize=FS_TITLE)
  axs1[3].set_title('Estimation Error, $\\frac{1}{p}||\\hat{\\beta}-\\beta^*||_2$', fontsize=FS_TITLE)
  axs1[4].set_title('Run Time', fontsize=FS_TITLE)
  
  labs=['Elastic Gradient Descent','Elastic Net', 'Coordinate Descent']
  lines=[]
  for c in COLS:
    lines.append(Line2D([0],[0],color=c,lw=2))
  
  fig1.legend(lines, labs, loc='lower center', ncol=3)
  
  fig1.tight_layout()
  fig1.subplots_adjust(bottom=0.09)
  
  for ax in axs1.ravel():
    ax.text(0,-.22,'$\\rho_1$', fontsize=FS_RHO, transform=ax.transAxes)
    ax.text(0,-.38,'$\\rho_2$', fontsize=FS_RHO, transform=ax.transAxes)
  
  fig1.savefig('figures/rho_sweep_'+str(P)+suf+'.pdf')
