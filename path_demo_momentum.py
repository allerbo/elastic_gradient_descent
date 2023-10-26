import numpy as np
import matplotlib.pyplot as plt
from elastic_desc import elastic_desc
from matplotlib.lines import Line2D

import seaborn as sns
colors=sns.color_palette('colorblind')


beta_true = [1.2,1.55]
s1=1
s2=2
rho=.55

Ss=np.array([[s1**2,s1*s2*rho],[s1*s2*rho,s2**2]])
n=1000
np.random.seed(0)
X=np.random.multivariate_normal([0,0],Ss,n)
y=X@beta_true+np.random.normal(0,1,n)
Ss_hat=1/n*X.T@X
beta_hat_ols = np.linalg.pinv(X.T@X)@X.T@y

#Contour matrix
beta0 = np.linspace(-.2, 2, 100)
beta1 = np.linspace(-.2, 2, 100)
mse_vals = np.zeros(shape=(beta0.size, beta1.size))
for i, value0 in enumerate(beta0):
  for j, value1 in enumerate(beta1):
    beta_temp = np.array((value0,value1))        
    mse_vals[j, i] = np.mean(np.square(y-X@beta_temp))

levs=np.square(np.linspace(0,4,11))+np.min(mse_vals)

gammas=[0,0.8]

f, ax = plt.subplots(1, 1,figsize=(5,5))
ax.contour(beta0,beta1,mse_vals,colors='C7', linewidths=1, levels=levs)
  
ax.set_aspect('equal')
ax.axhline(0,color='k',zorder=-1)
ax.axvline(0,color='k',zorder=-1)
ax.set_xticks([0,1,2])
ax.set_yticks([0,1,2])
ax.set_xlabel('$\\beta_1$')
ax.set_ylabel('$\\beta_2$')
ax.set_xlim([-.2,2])
ax.set_ylim([-.2,2])


labs=[]
lines=[]
for gamma,col in zip(gammas, [colors[2],colors[0]]):
  betas_ed=elastic_desc(X, y, 0.48, gamma=gamma, STEP_SIZE=1e-2,STOP_ACC=1e-4)[0]
  ax.plot(betas_ed[:,0],betas_ed[:,1],color=col,lw=2)
  lines.append(Line2D([0],[0],color=col,lw=2))
  labs.append('$\\gamma='+str(gamma)+'$')

f.legend(lines, labs, loc='lower center', ncol=len(labs))
plt.tight_layout()
f.subplots_adjust(bottom=0.17)
plt.savefig('figures/path_demo_momentum.pdf')

