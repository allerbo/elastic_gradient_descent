import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
from sklearn.linear_model import enet_path
from gradient_desc import grad_desc
from coordinate_desc import coord_desc
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
#X = scaler.fit_transform(X)
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

betas_ridge=[]
for lbda in np.logspace(10,-5,100):
  betas_ridge.append(np.linalg.pinv(X.T@X+lbda*np.eye(2))@X.T@y)

betas_ridge=np.vstack(betas_ridge)
betas_grad=grad_desc(X, y, STEP_SIZE=1e-4, STOP_ACC=1e-5)[0]

betas_lasso=lasso_path(X,y,eps=1e-5)[1].T
betas_coord=coord_desc(X, y, STEP_SIZE=1e-4,STOP_ACC=1e-5)[0]

betas_en= enet_path(X, y, l1_ratio=0.5)[1].T
betas_ed=elastic_desc(X, y, 0.5, STEP_SIZE=1e-4,STOP_ACC=1e-5)[0]

def l1(x0,x1,alpha=None):
  return np.abs(x0)+np.abs(x1)

def l2(x0,x1,alpha=None):
  return np.square(x0)+np.square(x1)

def en(x0,x1, alpha=0.5):
  return alpha*l1(x0,x1)+(1-alpha)*l2(x0,x1)

lbda0 = np.linspace(-2, 2, 300)
lbda1 = np.linspace(-2, 2, 300)




f, axs = plt.subplots(1, 3,figsize=(9,4))
for ax,betas_l, betas_d, norm, n_levels,title in zip(axs, [betas_ridge, betas_lasso, betas_en], [betas_grad, betas_coord, betas_ed], [l2, l1, en], [[.5, 1.5, 3],[.5,1.4,2.3],[.5,1.5,2.8]],['Ridge and GD ($\\alpha=0$)','Lasso and CD ($\\alpha=1$)','Elastic Net and EGD ($\\alpha=0.5)$']):
  ax.contour(beta0,beta1,mse_vals,colors='C7', linewidths=1, levels=levs)
  lbda_vals = np.zeros(shape=(lbda0.size, lbda1.size))
  for i, value0 in enumerate(lbda0):
    for j, value1 in enumerate(lbda1):
      lbda_vals[j, i] = norm(value0,value1)
  
  ax.plot(betas_l[:,0],betas_l[:,1],color=colors[1],lw=4)
  ax.set_aspect('equal')
  ax.axhline(0,color='k',zorder=-1)
  ax.axvline(0,color='k',zorder=-1)
  ax.set_xticks([0,1,2])
  ax.set_yticks([0,1,2])
  ax.set_xlabel('$\\beta_1$')
  ax.set_ylabel('$\\beta_2$')
  ax.set_xlim([-.2,2])
  ax.set_ylim([-.2,2])
  ax.set_title(title)


lines=[Line2D([0],[0],color=colors[2],lw=2),Line2D([0],[0],color=colors[1],lw=2),Line2D([0],[0],color='C7',lw=1)]
labs=['Iterative Method','Explicit Regularization','Contour Lines of Mean Squared Error']

axs[0].plot(betas_grad[:,0],betas_grad[:,1],color=colors[2],lw=2)
axs[1].plot(betas_coord[:,0],betas_coord[:,1],color=colors[2],lw=2)
axs[2].plot(betas_ed[:,0],betas_ed[:,1],color=colors[2],lw=2)

f.legend(lines, labs, loc='lower center', ncol=len(labs))
plt.tight_layout()
f.subplots_adjust(bottom=0.02)
plt.savefig('figures/path_demo.pdf')

