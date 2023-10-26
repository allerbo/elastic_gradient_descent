import numpy as np
import sys, os, time
from sklearn.linear_model import enet_path
import time
import pickle

def elastic_desc(X, y, alpha, gamma=0, step_size=0.01, max_iter=300000):
  n,p=X.shape
  Xtn=X.T/n
  XtXn=X.T@X/n
  Xtyn=X.T@y/n
  beta=np.zeros((p,1))
  beta_old=np.zeros((p,1))
  betas = [np.copy(beta)]
  n_iters=0
  gn_old=np.inf
  while n_iters<max_iter:
    n_iters+=1
    grad = Xtn@(X@beta-y) if p>2*n else XtXn@beta-Xtyn
    a_grad=np.abs(grad)
    I01=(a_grad/np.max(a_grad)>=alpha)
    el_grad=I01*grad
    beta_diff=beta-beta_old
    beta_old=np.copy(beta)
    beta += gamma*beta_diff-step_size*(alpha*np.sign(el_grad)+(1-alpha)*el_grad)
    betas.append(np.copy(beta))
    gn=np.linalg.norm(grad)
    if gn>gn_old:
      break
    gn_old=gn
  return np.array(betas), n_iters




def make_data_syn(seed, rho1, rho2, p, DATA='syn'):
  np.random.seed(seed)
  if DATA=='syn2':
    p1=40
    p2=p-p1
  else:
    p1=p//2
    p2=p1
  Ss11=(1-rho1)*np.eye(p1)+rho1*np.ones((p1,p1))
  Ss22=(1-rho1)*np.eye(p2)+rho1*np.ones((p2,p2))
  Ss12=rho2*np.ones((p1,p2))
  Ss=np.vstack((np.hstack((Ss11,Ss12)),np.hstack((Ss12.T,Ss22))))
  beta_star=np.vstack((np.random.normal(2,1,(p1,1)),np.zeros((p2,1))))
  
  n_train=100
  n_val=30
  n_test=30
  noise=10
  X_train=np.random.multivariate_normal(np.zeros(beta_star.shape[0]),Ss,n_train)
  y_train=X_train@beta_star+np.random.normal(0,noise,(n_train,1))
  X_val=np.random.multivariate_normal(np.zeros(beta_star.shape[0]),Ss,n_val)
  y_val=X_val@beta_star+np.random.normal(0,noise,(n_val,1))
  X_test=np.random.multivariate_normal(np.zeros(beta_star.shape[0]),Ss,n_test)
  y_test=X_test@beta_star+np.random.normal(0,noise,(n_test,1))
  snr=np.var(X_train@beta_star)/np.var(y_train-X_train@beta_star)
  return X_train, y_train, X_val, y_val, X_test, y_test, beta_star, snr

def get_sens(beta, beta_star):
  tp=np.sum((beta!=0) & (beta_star!=0))
  p = np.sum(beta_star!=0)
  return tp/p

def get_spec(beta, beta_star):
  tn = np.sum((beta==0) & (beta_star==0))
  n = np.sum(beta_star==0)
  return tn/n

def get_mse(y,y_hat):
  return np.mean((y-y_hat)**2)

def get_r2(y,y_hat):
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)

def get_pred_err(beta, beta_star, X_test):
  return 1/X_test.shape[0]*np.linalg.norm(X_test@(beta-beta_star))

def get_est_err(beta, beta_star):
  return 1/beta.shape[0]*np.linalg.norm(beta-beta_star)

def get_beta_size(beta):
  p = np.sum(beta!=0)
  return p

def get_path_frac(beta_path, beta_star):
  norms=np.sum(np.abs(beta_path),1)
  beta_star_bool=beta_star!=0
  correct_betas=[]
  for norm in np.arange(0,norms[-1],0.1):
    idx=np.min(np.where(norm<norms)[0])#last idx where norms<norm
    beta_bool=beta_path[idx,:]!=0
    correct_betas.append(np.sum(beta_bool==beta_star_bool)==beta_star.shape[0])
  return np.mean(correct_betas)

def eval_alpha(beta_path, X_val, y_val):
  best_mse=np.infty
  best_r=-1
  for r in range(beta_path.shape[0]):
    beta=beta_path[r,:]
    mse=get_mse(y_val, X_val@beta)
    if mse<best_mse:
      best_r=r
      best_mse=mse
  return best_r, best_mse

def select_alpha(best_mses, best_rs, beta_paths):
  alpha_idx=np.argmin(best_mses)
  best_beta_path=beta_paths[alpha_idx]
  best_r=best_rs[alpha_idx]
  best_beta=best_beta_path[best_r,:]
  return best_beta, best_beta_path


def sweep_alpha(X_train, y_train, X_val, y_val, ALG, gamma, step_size, alphas=np.linspace(0.1,0.9,9)):
  if ALG=='egdm' or ALG=='egd':
    gamma1 = 0 if ALG=='egd' else gamma
    t1=time.time()
    beta_paths_eg=[]
    best_rs_eg=[]
    best_mses_eg= []
    for alpha in alphas:
      beta_path_eg = elastic_desc(X_train, y_train, alpha, gamma1, step_size)[0]
      best_r_eg, best_mse_eg = eval_alpha(beta_path_eg, X_val, y_val)
      beta_paths_eg.append(beta_path_eg)
      best_rs_eg.append(best_r_eg)
      best_mses_eg.append(best_mse_eg)
    beta, beta_path = select_alpha(best_mses_eg, best_rs_eg, beta_paths_eg)
    time_spent=time.time()-t1
  elif ALG=='en':
    t1=time.time()
    beta_paths_en=[]
    best_rs_en=[]
    best_mses_en= []
    for alpha in alphas:
      beta_path_en= enet_path(X_train, y_train, l1_ratio=alpha)[1].T
      best_r_en, best_mse_en = eval_alpha(beta_path_en, X_val, y_val)
      beta_paths_en.append(beta_path_en)
      best_rs_en.append(best_r_en)
      best_mses_en.append(best_mse_en)
    beta, beta_path = select_alpha(best_mses_en, best_rs_en, beta_paths_en)
    time_spent=time.time()-t1
  elif ALG=='cd':
    t1=time.time()
    beta_path_cd = elastic_desc(X_train, y_train, 1, 0, step_size)[0]
    best_r_cd, best_mse_cd = eval_alpha(beta_path_cd, X_val, y_val)
    beta=beta_path_cd[best_r_cd,:]
    beta_path=beta_path_cd
    time_spent=time.time()-t1
  elif ALG=='gd':
    t1=time.time()
    beta_path_gd = elastic_desc(X_train, y_train, 0, 0, step_size)[0]
    best_r_gd, best_mse_gd = eval_alpha(beta_path_gd, X_val, y_val)
    beta=beta_path_gd[best_r_gd,:]
    beta_path=beta_path_gd
    time_spent=time.time()-t1
  return beta, beta_path, time_spent


seed=0
ALG='egdm'
DATA='syn'
rho1=0.7
rho2=0.3
p=50

gamma=0.5
step_size=0.01

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

data_dict={'syn': {},'syn2': {}}
for DATA in ['syn','syn2']:
  data_dict[DATA]={'egdm': {},'egd': {}, 'en': {},'cd': {}}
  for ALG in ['egdm','egd','en','cd']:
    for p in range(50,201,10):
      X_train, y_train, X_val, y_val, X_test, y_test, beta_star, snr = make_data_syn(seed, rho1, rho2, p, DATA=DATA)
      beta, beta_path, time_spent = sweep_alpha(X_train, y_train, X_val, y_val, ALG, gamma, step_size)
      data_dict[DATA][ALG][p]={'sens': get_sens(beta, beta_star), 'spec': get_spec(beta, beta_star), 'pred_err': get_pred_err(beta, beta_star, X_test), 'est_err': get_est_err(beta, beta_star), 'time_spent': time_spent, 'snr': snr}

fi=open('data/synth_p_'+str(seed)+'.pkl','wb')
pickle.dump(data_dict,fi)
fi.close()

