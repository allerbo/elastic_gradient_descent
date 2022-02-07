import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from sklearn.linear_model import enet_path
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(1,'..')
from elastic_desc import elastic_desc
scaler = StandardScaler()

COLS=[(0.5,0,0), (0,0.4,0), (0,0,0.6)]

rho=.69
ALPHAS=[0,0.5,0.7,1]


LW=1
FS_LAB=8
FS_TITLE=10
FS_TICK=5

Ss=np.array([[1,rho,rho],[rho,1, rho],[rho,rho,1]])
beta_true = [0,0.1,1]
n=1000
np.random.seed(0)
X=np.random.multivariate_normal(np.zeros(len(beta_true)),Ss,n)
X = scaler.fit_transform(X)
y=X.dot(beta_true)+np.random.normal(0,0,n)
beta_hat = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
Ss_hat = 1/n*X.T.dot(X)



STEP_SIZE=0.001

f, axs = plt.subplots(2,len(ALPHAS), figsize=(8,4))
for a in range(len(ALPHAS)):
  alpha=ALPHAS[a]
  beta_path_eg=elastic_desc(X,y,alpha,STEP_SIZE,alg='norm')[0]
  
  alpha=ALPHAS[a]
  if alpha==0:
    betas_en=[]
    I=np.eye(len(beta_true))
    for t in np.logspace(-3,3,200):
      beta_en=(I-np.linalg.pinv(I+t*Ss_hat)).dot(beta_hat)
      betas_en.append(beta_en)
    
    beta_path_en=np.array(betas_en)
  else:
    beta_path_en = enet_path(X, y, l1_ratio=alpha, fit_intercept=False)[1].T
  
  norms_eg=np.sum(np.abs(beta_path_eg),1)
  norms_en=np.sum(np.abs(beta_path_en),1)
  for b in (0,1):
    axs[b,a].axhline(color='k', lw=0.9)
  
  for d in range(beta_path_eg.shape[1]):
    axs[0,a].plot(norms_eg,beta_path_eg[:,d], color=COLS[d])
    axs[1,a].plot(norms_en,beta_path_en[:,d], color=COLS[d])
  
  axs[0,a].set_title('$\\alpha=$'+str(alpha), fontsize=FS_TITLE)
  axs[1,a].set_xlabel('$||\\beta||_1$',fontsize=FS_LAB)
  for b in (0,1):
    axs[b,a].tick_params(axis='x',labelsize=FS_TICK)
    axs[b,a].tick_params(axis='y',labelsize=FS_TICK)
    for i in range(3):
      axs[b,a].axhline(beta_true[i],linestyle=':',color=COLS[i],zorder=1)
    #axs[b,a].set_aspect('equal')

axs[0,0].set_ylabel('Elastic Gradient Descent\n$\\beta$',fontsize=FS_LAB)
axs[1,0].set_ylabel('Elastic Net\n$\\beta$',fontsize=FS_LAB)
plt.tight_layout()
plt.savefig('figures/diff_demo.pdf')

