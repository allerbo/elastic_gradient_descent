import sys, os
import matplotlib.pyplot as plt
import numpy as np
N_OBS=1000

APDX=''
RHO1s=[0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
RHO2s=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
GREEN=(0,0.4,0)
BLUE= (0,0,0.6)
RED= (0.5,0,0)
GRUE=(0,0.3,0.35)
DARK_RED=(0.35,0,0)

def plot_mean_std(ax, labels_i, means, stds, c):
  l1=ax.plot(labels_i, means, color=c, linewidth=0.7)[0]
  ax.plot(labels_i, list(map(lambda x,y:x+y,means,stds)), color=c, linestyle='--', linewidth=0.7)
  ax.plot(labels_i, list(map(lambda x,y:x-y,means,stds)), color=c, linestyle='--', linewidth=0.7)
  return l1
  
for APDX in ['_norm', '_norm_c', '_eps_c','_eps']:
  fig1, axs1 = plt.subplots(4,1, figsize=(0.75*10,0.85*8))
  fig2, axs2 = plt.subplots(3,1, figsize=(0.75*10,0.85*6))
  axs1b=list(map(lambda ax: ax.twinx(), axs1))
  axs2b=list(map(lambda ax: ax.twinx(), axs2))
  ii=0
  labels=[]
  for RHO1 in RHO1s:
    sens_mse_mean_egs=[]
    sens_mse_std_egs=[]
    spec_mse_mean_egs=[]
    spec_mse_std_egs=[]
    mse_mse_mean_egs=[]
    mse_mse_std_egs=[]
    sens_mse_mean_ens=[]
    sens_mse_std_ens=[]
    spec_mse_mean_ens=[]
    spec_mse_std_ens=[]
    mse_mse_mean_ens=[]
    mse_mse_std_ens=[]
    sens_mse_mean_diffs=[]
    sens_mse_std_diffs=[]
    spec_mse_mean_diffs=[]
    spec_mse_std_diffs=[]
    mse_mse_mean_diffs=[]
    mse_mse_std_diffs=[]
  
    sens_cv_mean_egs=[]
    sens_cv_std_egs=[]
    spec_cv_mean_egs=[]
    spec_cv_std_egs=[]
    mse_cv_mean_egs=[]
    mse_cv_std_egs=[]
    sens_cv_mean_ens=[]
    sens_cv_std_ens=[]
    spec_cv_mean_ens=[]
    spec_cv_std_ens=[]
    mse_cv_mean_ens=[]
    mse_cv_std_ens=[]
    sens_cv_mean_diffs=[]
    sens_cv_std_diffs=[]
    spec_cv_mean_diffs=[]
    spec_cv_std_diffs=[]
    mse_cv_mean_diffs=[]
    mse_cv_std_diffs=[]
    
    path_frac_mean_egs=[]
    path_frac_std_egs=[]
    path_frac_mean_ens=[]
    path_frac_std_ens=[]
    path_frac_mean_diffs=[]
    path_frac_std_diffs=[]
  
    sens_mc_mean_egs=[]
    sens_mc_std_egs=[]
    spec_mc_mean_egs=[]
    spec_mc_std_egs=[]
    sens_mc_mean_ens=[]
    sens_mc_std_ens=[]
    spec_mc_mean_ens=[]
    spec_mc_std_ens=[]
    sens_mc_mean_diffs=[]
    sens_mc_std_diffs=[]
    spec_mc_mean_diffs=[]
    spec_mc_std_diffs=[]
  
    sens_mc1_mean_egs=[]
    sens_mc1_std_egs=[]
    spec_mc1_mean_egs=[]
    spec_mc1_std_egs=[]
    sens_mc1_mean_ens=[]
    sens_mc1_std_ens=[]
    spec_mc1_mean_ens=[]
    spec_mc1_std_ens=[]
    sens_mc1_mean_diffs=[]
    sens_mc1_std_diffs=[]
    spec_mc1_mean_diffs=[]
    spec_mc1_std_diffs=[]
  
  
  
    labels_i=[]
    for RHO2 in RHO2s:
      file_name='logs/run_synth_'+str(RHO1)+'_'+str(RHO2)+APDX+'.txt'
      f = open(file_name)
      lines = f.readlines()
      f.close()
      
      for l in range(len(lines)):
        if lines[l]=='Observations: '+str(N_OBS)+'\n':
          break
      
      stat_list = lines[l+2].split(' ')
      sens_mse_mean_egs.append(float(stat_list[2]))
      sens_mse_std_egs.append(float(stat_list[3][1:-2]))
      spec_mse_mean_egs.append(float(stat_list[5]))
      spec_mse_std_egs.append(float(stat_list[6][1:-2]))
      mse_mse_mean_egs.append(float(stat_list[8]))
      mse_mse_std_egs.append(float(stat_list[9][1:-3]))
  
      stat_list = lines[l+3].split(' ')
      sens_mse_mean_ens.append(float(stat_list[2]))
      sens_mse_std_ens.append(float(stat_list[3][1:-2]))
      spec_mse_mean_ens.append(float(stat_list[5]))
      spec_mse_std_ens.append(float(stat_list[6][1:-2]))
      mse_mse_mean_ens.append(float(stat_list[8]))
      mse_mse_std_ens.append(float(stat_list[9][1:-3]))
      
      stat_list = lines[l+4].split(' ')
      sens_mse_mean_diffs.append(float(stat_list[2]))
      sens_mse_std_diffs.append(float(stat_list[3][1:-2]))
      spec_mse_mean_diffs.append(float(stat_list[5]))
      spec_mse_std_diffs.append(float(stat_list[6][1:-2]))
      mse_mse_mean_diffs.append(float(stat_list[8]))
      mse_mse_std_diffs.append(float(stat_list[9][1:-3]))
      
      stat_list = lines[l+6].split(' ')
      sens_cv_mean_egs.append(float(stat_list[2]))
      sens_cv_std_egs.append(float(stat_list[3][1:-2]))
      spec_cv_mean_egs.append(float(stat_list[5]))
      spec_cv_std_egs.append(float(stat_list[6][1:-2]))
      mse_cv_mean_egs.append(float(stat_list[8]))
      mse_cv_std_egs.append(float(stat_list[9][1:-3]))
  
      stat_list = lines[l+7].split(' ')
      sens_cv_mean_ens.append(float(stat_list[2]))
      sens_cv_std_ens.append(float(stat_list[3][1:-2]))
      spec_cv_mean_ens.append(float(stat_list[5]))
      spec_cv_std_ens.append(float(stat_list[6][1:-2]))
      mse_cv_mean_ens.append(float(stat_list[8]))
      mse_cv_std_ens.append(float(stat_list[9][1:-3]))
      
      stat_list = lines[l+8].split(' ')
      sens_cv_mean_diffs.append(float(stat_list[2]))
      sens_cv_std_diffs.append(float(stat_list[3][1:-2]))
      spec_cv_mean_diffs.append(float(stat_list[5]))
      spec_cv_std_diffs.append(float(stat_list[6][1:-2]))
      mse_cv_mean_diffs.append(float(stat_list[8]))
      mse_cv_std_diffs.append(float(stat_list[9][1:-3]))
  
      stat_list = lines[l+10].split(' ')
      path_frac_mean_egs.append(float(stat_list[1]))
      path_frac_std_egs.append(float(stat_list[2][1:-3]))
  
      stat_list = lines[l+11].split(' ')
      path_frac_mean_ens.append(float(stat_list[1]))
      path_frac_std_ens.append(float(stat_list[2][1:-3]))
  
      stat_list = lines[l+12].split(' ')
      path_frac_mean_diffs.append(float(stat_list[1]))
      path_frac_std_diffs.append(float(stat_list[2][1:-3]))
  
      stat_list = lines[l+14].split(' ')
      sens_mc_mean_egs.append(float(stat_list[2]))
      sens_mc_std_egs.append(float(stat_list[3][1:-2]))
      spec_mc_mean_egs.append(float(stat_list[5]))
      spec_mc_std_egs.append(float(stat_list[6][1:-3]))
  
      stat_list = lines[l+15].split(' ')
      sens_mc_mean_ens.append(float(stat_list[2]))
      sens_mc_std_ens.append(float(stat_list[3][1:-2]))
      spec_mc_mean_ens.append(float(stat_list[5]))
      spec_mc_std_ens.append(float(stat_list[6][1:-3]))
  
      stat_list = lines[l+16].split(' ')
      sens_mc_mean_diffs.append(float(stat_list[2]))
      sens_mc_std_diffs.append(float(stat_list[3][1:-2]))
      spec_mc_mean_diffs.append(float(stat_list[5]))
      spec_mc_std_diffs.append(float(stat_list[6][1:-3]))
  
      stat_list = lines[l+18].split(' ')
      sens_mc1_mean_egs.append(float(stat_list[2]))
      sens_mc1_std_egs.append(float(stat_list[3][1:-2]))
      spec_mc1_mean_egs.append(float(stat_list[5]))
      spec_mc1_std_egs.append(float(stat_list[6][1:-3]))
  
      stat_list = lines[l+19].split(' ')
      sens_mc1_mean_ens.append(float(stat_list[2]))
      sens_mc1_std_ens.append(float(stat_list[3][1:-2]))
      spec_mc1_mean_ens.append(float(stat_list[5]))
      spec_mc1_std_ens.append(float(stat_list[6][1:-3]))
  
      stat_list = lines[l+20].split(' ')
      sens_mc1_mean_diffs.append(float(stat_list[2]))
      sens_mc1_std_diffs.append(float(stat_list[3][1:-2]))
      spec_mc1_mean_diffs.append(float(stat_list[5]))
      spec_mc1_std_diffs.append(float(stat_list[6][1:-3]))
  
      
      
      labels_i.append(ii)
      labels.append(str(RHO1)[-3:]+'\n'+str(RHO2)[-3:])
      ii+=1
        
      l1_1=plot_mean_std(axs1[0], labels_i, sens_mse_mean_egs,  sens_mse_std_egs, GREEN)
      l2_1=plot_mean_std(axs1[0], labels_i, sens_mse_mean_ens,  sens_mse_std_ens, BLUE)
      l3_1=plot_mean_std(axs1b[0],labels_i, sens_mse_mean_diffs,sens_mse_std_diffs, RED)
      plot_mean_std(axs1[1], labels_i, spec_mse_mean_egs,  spec_mse_std_egs, GREEN)
      plot_mean_std(axs1[1], labels_i, spec_mse_mean_ens,  spec_mse_std_ens, BLUE)
      plot_mean_std(axs1b[1],labels_i, spec_mse_mean_diffs,spec_mse_std_diffs, RED)
      plot_mean_std(axs1[2], labels_i, mse_mse_mean_egs,  mse_mse_std_egs, GREEN)
      plot_mean_std(axs1[2], labels_i, mse_mse_mean_ens,  mse_mse_std_ens, BLUE)
      plot_mean_std(axs1b[2],labels_i, mse_mse_mean_diffs,mse_mse_std_diffs, RED)
  
      plot_mean_std(axs1[3], labels_i, path_frac_mean_egs,  path_frac_std_egs, GREEN)
      plot_mean_std(axs1[3], labels_i, path_frac_mean_ens,  path_frac_std_ens, BLUE)
      plot_mean_std(axs1b[3],labels_i, path_frac_mean_diffs,path_frac_std_diffs, RED)
  
      l1_2=plot_mean_std(axs2[0], labels_i, sens_cv_mean_egs,  sens_cv_std_egs, GREEN)
      l2_2=plot_mean_std(axs2[0], labels_i, sens_cv_mean_ens,  sens_cv_std_ens, BLUE)
      l3_2=plot_mean_std(axs2b[0],labels_i, sens_cv_mean_diffs,sens_cv_std_diffs, RED)
      plot_mean_std(axs2[1], labels_i, spec_cv_mean_egs,  spec_cv_std_egs, GREEN)
      plot_mean_std(axs2[1], labels_i, spec_cv_mean_ens,  spec_cv_std_ens, BLUE)
      plot_mean_std(axs2b[1],labels_i, spec_cv_mean_diffs,spec_cv_std_diffs, RED)
      plot_mean_std(axs2[2], labels_i, mse_cv_mean_egs,  mse_cv_std_egs, GREEN)
      plot_mean_std(axs2[2], labels_i, mse_cv_mean_ens,  mse_cv_std_ens, BLUE)
      plot_mean_std(axs2b[2],labels_i, mse_cv_mean_diffs,mse_cv_std_diffs, RED)
  
  FS_TITLE=11
  FS_TICKS=6
  FS_LABELS=9
  FS_RHO=7
  
  axs1[0].set_title('Sensitivity', fontsize=FS_TITLE)
  axs1[1].set_title('Specificity', fontsize=FS_TITLE)
  axs1[2].set_title('Mean Squared Error', fontsize=FS_TITLE)
  axs1[3].set_title('True Path Rate', fontsize=FS_TITLE)
  axs2[0].set_title('Sensitivity', fontsize=FS_TITLE)
  axs2[1].set_title('Specificity', fontsize=FS_TITLE)
  axs2[2].set_title('Mean Squared Error', fontsize=FS_TITLE)
  
  axs1[0].set_ylim([0.55, 1.01])
  axs1b[0].set_ylim([-.06,0.55])
  axs2[0].set_ylim([0.55, 1.01])
  axs2b[0].set_ylim([-.05,0.55])
  axs1[1].set_ylim([-.3, 1.01])
  axs1b[1].set_ylim([-.2,0.67])
  axs2[1].set_ylim([-.3, 1.01])
  axs2b[1].set_ylim([-.2,0.67])
  axs1[2].set_ylim([95, 125])
  axs1b[2].set_ylim([-10,15])
  axs2[2].set_ylim([95, 150])
  axs2b[2].set_ylim([-10,15])
  axs1[3].set_ylim([-.2, 0.2])
  axs1b[3].set_ylim([-.05,0.35])
  
  for ax in np.hstack((axs1.ravel(),axs2.ravel())):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x',labelsize=FS_TICKS)
    ax.set_ylabel('Abolute values',color=GRUE,fontsize=FS_LABELS)
    ax.tick_params(axis='y',colors=GRUE)
    ax.tick_params(axis='y',labelsize=FS_TICKS)
    for x_ in np.arange(5.5,35,6):
      ax.vlines(x_,*ax.get_ylim(), colors='k', linewidth=.5)
      ax.set_xlim((-0.5,35.5))
  
  for ax in axs1b+axs2b:
    ax.set_ylabel('Differences',color=DARK_RED,fontsize=FS_LABELS)
    ax.tick_params(axis='y',colors=DARK_RED)
    ax.tick_params(axis='y',labelsize=FS_TICKS)
    ax.hlines(0,*ax.get_xlim(), colors=DARK_RED)
  
  
  axs1[3].hlines(0,*axs1[3].get_xlim(), colors=GRUE)
  fig1.legend([l1_1, l2_1, l3_1], ['Elastic Gradient Descent', 'Elastic Net', 'Difference'], loc='lower center', ncol=3, fontsize=FS_RHO)
  fig2.legend([l1_2, l2_2, l3_2], ['Elastic Gradient Descent', 'Elastic Net', 'Difference'], loc='lower center', ncol=3, fontsize=FS_RHO)
  
  fig1.suptitle('First Model Selection Criterion (Best MSE)')
  fig2.suptitle('Second Model Selection Criterion (Cross-validation)')
  
  fig1.tight_layout()
  fig2.tight_layout()
  fig1.subplots_adjust(bottom=.08)
  fig2.subplots_adjust(bottom=.11)
  
  for ax in axs1.ravel():
    ax.text(-.018,-.16,'$\\rho_1$', fontsize=FS_RHO, transform=ax.transAxes)
    ax.text(-.018,-.26,'$\\rho_2$', fontsize=FS_RHO, transform=ax.transAxes)
  
  for ax in axs2.ravel():
    ax.text(-.018,-.17,'$\\rho_1$', fontsize=FS_RHO, transform=ax.transAxes)
    ax.text(-.018,-.28,'$\\rho_2$', fontsize=FS_RHO, transform=ax.transAxes)
  
  fig1.savefig('figures/rho_sweep_val'+APDX+'.pdf')
  fig2.savefig('figures/rho_sweep_cv'+APDX+'.pdf')
