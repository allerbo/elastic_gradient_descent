import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from sklearn.linear_model import enet_path
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import sys
sys.path.insert(1,'..')
from elastic_desc import elastic_desc
from elastic_flow import elastic_flow
#X, y = datasets.load_diabetes(return_X_y=True)
def ma(x, w):
  ma=np.convolve(x, np.ones(w), 'same')/w
  w1=w-1
  for i in range(0,w1-w1//2):
    ma[i]=np.mean(x[:w-(w1-w1//2-i)])
  for i in range(0,w1//2):
   ma[len(ma)-i-1]=np.mean(x[-(w-(w1//2-i)):])
  return ma

def might_append(ls,l,b):
  if b:
    ls.append(l)
  return ls

diab=datasets.load_diabetes()
X=diab.data
y=diab.target
scaler = StandardScaler()


X=scaler.fit_transform(X)


LW=1
LW2=3
FS_LAB=10
FS_TITLE=12
FS_TICK=7

X_MIN=0
X_MAX=150
Y0_MIN=-15
Y0_MAX=30


MA=3
ALPHA=0.5
STEP_SIZE=0.01

FLOW=True
for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])


  
ALGS =['norm_s', 'norm_ns', 'unnorm', 'gs_s', 'gs_ns']
ALG_NAMES=['Normalized\nScaled', 'Normalized\nNon-scaled', 'Unnormalized\nGradient Flow', 'Generalized Stagewise\nScaled', 'Generalized Stagewise\nNon-scaled']
  
fig, axs = plt.subplots(len(ALGS), 3, figsize=(10,10))
ls=[]
for a in range(len(ALGS)):
  alg=ALGS[a]
  if alg=='unnorm' and FLOW:
    beta_path, grad_path, h_alpha_path, p1s=elastic_flow(X,y,ALPHA)
    my_ma=1
  else:
    beta_path, grad_path, h_alpha_path, p1s=elastic_desc(X,y,ALPHA, STEP_SIZE, alg)
    my_ma=MA
  
  ns=np.sum(np.abs(beta_path),1)

  delta_beta = np.vstack((np.zeros(beta_path.shape[1]),np.diff(beta_path,axis=0)))/STEP_SIZE
  delta_beta_gs = np.vstack((np.zeros(beta_path.shape[1]),np.diff(beta_path,axis=0)))/STEP_SIZE
  h_alpha_path1 = ALPHA*np.sum(delta_beta*np.sign(delta_beta),1)+(1-ALPHA)*np.sum(delta_beta*delta_beta,1)
  
  for d in range(beta_path.shape[1]):
    axs[a,0].plot(ns,beta_path[:,d], linewidth=LW)
    axs[a,1].plot(ns,grad_path[:,d]/np.max(np.abs(grad_path),1), linewidth=LW)
  axs[a,0].set_ylim((Y0_MIN,Y0_MAX))
  axs[a,0].hlines(0,X_MIN,X_MAX, colors='k')
  axs[a,1].hlines(ALPHA,X_MIN,X_MAX, colors='k')
  axs[a,1].hlines(-ALPHA,X_MIN,X_MAX, colors='k')
  
  #Bounds, desired h_alpha
  if alg=='norm_s':
    axs[a,2].hlines(1,X_MIN,X_MAX, colors='k', linewidth=LW2) #1
    axs[a,2].set_ylim((0.9,1.1))
  elif alg=='gs_s':
    axs[a,2].hlines(STEP_SIZE,X_MIN,X_MAX, colors='k', linewidth=LW2) #eps
    axs[a,2].set_ylim((0.9*STEP_SIZE,1.1*STEP_SIZE))
  elif alg=='gs_ns':
    axs[a,2].plot(ns, ma(STEP_SIZE*(1+ALPHA*(1-ALPHA)*(np.sqrt(p1s/STEP_SIZE)-1)),my_ma), 'C3', linewidth=LW) # upperbound
    axs[a,2].plot(ns, ma(STEP_SIZE*(1-ALPHA*(1-ALPHA)*(2-ALPHA)*(1-STEP_SIZE/p1s)),my_ma),'C3', linewidth=LW) #lower bound
    axs[a,2].plot(ns, STEP_SIZE*np.ones(len(ns)), 'k', linewidth=LW2) #eps
  else:
    ls=might_append(ls,axs[a,2].plot(ns, ma(1+ALPHA*(1-ALPHA)*(np.sqrt(p1s)-1),my_ma), 'C3', linewidth=LW)[0],a==1) # upper bound
    axs[a,2].plot(ns, ma(1-ALPHA*(1-ALPHA)*(2-ALPHA)*(1-1/p1s),my_ma),'C3', linewidth=LW) #lower bound
    axs[a,2].hlines(1,X_MIN,X_MAX, colors='k', linewidth=LW2) #1
  
  ls=might_append(ls,axs[a,2].plot(ns,ma(h_alpha_path,my_ma), linewidth=LW)[0],a==1) #norm
  
  #p1
  ax2=axs[a,2].twinx()
  ls=might_append(ls,ax2.plot(ns, ma(p1s,my_ma), 'C2--', linewidth=LW)[0],a==1)
  ax2.set_ylim((0,10))
  ax2.tick_params(axis='y',colors=[0.6*.17,0.6*.63,0.6*.17])
  ax2.tick_params(axis='y',labelsize=FS_TICK)
  axs[a,2].set_zorder(ax2.get_zorder()+1) 
  axs[a,2].patch.set_visible(False)
  
  axs[a,0].set_ylabel(ALG_NAMES[a]+'\n$\\beta$', fontsize=FS_LAB)
  axs[a,1].set_ylabel('$g/||g||_\infty$',fontsize=FS_LAB)
  axs[a,2].set_ylabel('$h_\\alpha(\\Delta\\beta)$',fontsize=FS_LAB)


for b in range(3):
  axs[len(ALGS)-1,b].set_xlabel('$||\\beta||_1$',fontsize=FS_LAB)
axs[0,0].set_title('$\\beta$ Path',fontsize=FS_TITLE)
axs[0,1].set_title('Gradient Path',fontsize=FS_TITLE)
axs[0,2].set_title('$h_\\alpha$ Path',fontsize=FS_TITLE)

for ax in axs.ravel():
  ax.tick_params(axis='x',labelsize=FS_TICK)
  ax.tick_params(axis='y',labelsize=FS_TICK)
  ax.set_xlim((X_MIN,X_MAX))


ls[0], ls[1] = ls[1], ls[0]
labs=['$h_\\alpha$','bounds','$p_1$']

fig.legend(ls, labs, loc='lower right', ncol=3)

fig.tight_layout()
fig.subplots_adjust(bottom=0.09)

fig.savefig('figures/diab_path_flavs.pdf')
