import numpy as np
import sys, os, time
from sklearn.linear_model import enet_path
import multiprocessing as mp
import time
from sklearn import datasets
import pandas as pd

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
    #beta -= step_size*(alpha*sign.(el_grad)+(1-alpha)*el_grad)
    beta_diff=beta-beta_old
    beta_old=np.copy(beta)
    beta += gamma*beta_diff-step_size*(alpha*np.sign(el_grad)+(1-alpha)*el_grad)
    betas.append(np.copy(beta))
    gn=np.linalg.norm(grad)
    if gn>gn_old:
      break
    gn_old=gn
  return np.array(betas), n_iters

def write_log(log_name, print_str):
  pref = os.environ['SLURM_SUBMIT_DIR']+'/' if 'SLURM_SUBMIT_DIR' in os.environ else ''
  with open(pref+log_name, 'a+') as f:
    f.write(print_str + '\n')

def make_data_real(seed, data_set):
  np.random.seed(seed)
  if data_set=='diab':
    diab=datasets.load_diabetes()
    data=np.hstack((diab.data,diab.target.reshape((-1,1))))
    np.random.shuffle(data)
    X=data[:,:-1]
    y=data[:,-1].reshape((-1,1))
  elif data_set=='house':
    house=datasets.fetch_california_housing()
    data=np.hstack((house.data,house.target.reshape((-1,1))))
    np.random.shuffle(data)
    X=data[:,:-1]
    y=data[:,-1].reshape((-1,1))
  elif data_set=='black':
    #https://www.maths.ed.ac.uk/~swood34
    data=pd.read_csv('in_data/bs_2000.csv',sep=',').to_numpy()
    np.random.shuffle(data)
    X=data[:,1:11]
    y=data[:,0].reshape((-1,1))
  elif data_set=='read':
    #https://www.kaggle.com/code/uocoeeds/building-a-regression-model-with-elastic-net
    data=pd.read_csv('in_data/readability_features.csv',sep=',').to_numpy()
    np.random.shuffle(data)
    X=data[:,:-1]
    y=data[:,-1].reshape((-1,1))
  elif data_set=='casp':
    #https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure
    data=pd.read_csv('in_data/CASP.csv',sep=',').to_numpy()
    np.random.shuffle(data)
    X=data[:,1:]
    y=data[:,0].reshape((-1,1))
  elif data_set=='wood':
    #https://openmv.net/info/wood-fibres
    data=pd.read_csv('in_data/wood-fibres.csv',sep=',').to_numpy()
    np.random.shuffle(data)
    X=data[:,1:]
    y=data[:,0].reshape((-1,1))
  elif data_set=='super':
    #https://archive.ics.uci.edu/dataset/464/superconductivty+data
    data=pd.read_csv('in_data/superconduct.csv',sep=',').to_numpy()
    np.random.shuffle(data)
    X=data[:,:-1]
    y=data[:,-1].reshape((-1,1))
  elif data_set=='twitter':
    #http://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media
    data=pd.read_csv('in_data/Twitter.csv',sep=',').iloc[:291624,:].to_numpy() #takes too long if using all
    np.random.shuffle(data)
    X=data[:,:-1]
    y=data[:,-1].reshape((-1,1))
  elif data_set=='energy':
    #http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
    data=pd.read_csv('in_data/energydata_complete.csv',sep=',').iloc[:,1:].to_numpy()
    np.random.shuffle(data)
    X=data[:,1:]
    y=data[:,0].reshape((-1,1))
  elif data_set=='elect':
    #https://archive.ics.uci.edu/dataset/513/real+time+election+results+portugal+2019
    data=pd.read_csv('in_data/ElectionData.csv',sep=',').iloc[:,list(range(3,21))+list(range(22,28))].to_numpy()
    np.random.shuffle(data)
    X=data[:,:-1]
    X=np.delete(X,[18,19,20,21,22],1) #too easy if present
    y=data[:,-1].reshape((-1,1))
  
  
  X=(X-np.mean(X, 0))/np.std(X,0)
  y=y-np.mean(y)
  n_tot=data.shape[0]
  n_train=round(0.8*n_tot)
  n_val=round(0.1*n_tot)
  n_test=round(0.1*n_tot)
  
  X_train=X[:n_train,:]
  X_val=X[n_train:(n_train+n_val),:]
  X_test=X[(n_train+n_val):,:]
  y_train=y[:n_train]
  y_val=y[n_train:(n_train+n_val)]
  y_test=y[(n_train+n_val):]
  return X_train, y_train, X_val, y_val, X_test, y_test



def make_data_syn(seed, rho1, rho2, p):
  np.random.seed(seed)
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

def print_log(LOG_NAME, data_mat, ALG, DATA, step_size, gamma):
  n_exps=data_mat.shape[0]
  QUANTS=[0.5,0.25,0.75,0.1,0.9,0.05,0.95]
  if DATA=='syn':
    sens = data_mat[:,0]
    spec = data_mat[:,1]
    r2 =  data_mat[:,2]
    pred_err = data_mat[:,3]
    est_err = data_mat[:,4]
    time_spent = data_mat[:,5]
    snr =  data_mat[:,7]
    
    write_log(LOG_NAME, f"Algorithm: {ALG}")
    write_log(LOG_NAME, f"Step size, gamma: {step_size:.3f} {gamma:.3f}")
    write_log(LOG_NAME, f"Experiments: {n_exps}")
    #write_log(LOG_NAME, f"Sensitivity: {np.mean(sens):.3f} {np.std(sens):.3f}")
    #write_log(LOG_NAME, f"Specificity: {np.mean(spec):.3f} {np.std(spec):.3f}")
    #write_log(LOG_NAME, f"R2: {np.mean(r2):.3f} {np.std(r2):.3f}")
    #write_log(LOG_NAME, f"Pred_err: {np.mean(pred_err):.3f} {np.std(pred_err):.3f}")
    #write_log(LOG_NAME, f"Est_err: {np.mean(est_err):.3f} {np.std(est_err):.3f}")
    #write_log(LOG_NAME, f"Time: {np.mean(time_spent):.3f} {np.std(time_spent):.3f}")
    #write_log(LOG_NAME, f"SNR: {np.mean(snr):.3f} {np.std(snr):.3f}")
    write_log(LOG_NAME, 'Sensitivity: '+' '.join([f'{np.quantile(sens,q):.4g}' for q in QUANTS]))
    write_log(LOG_NAME, 'Specificity: '+' '.join([f'{np.quantile(spec,q):.4g}' for q in QUANTS]))
    write_log(LOG_NAME, 'R2: '+' '.join([f'{np.quantile(r2,q):.4g}' for q in QUANTS]))
    write_log(LOG_NAME, 'Pred_err: '+' '.join([f'{np.quantile(pred_err,q):.4g}' for q in QUANTS]))
    write_log(LOG_NAME, 'Est_err: '+' '.join([f'{np.quantile(est_err,q):.4g}' for q in QUANTS]))
    write_log(LOG_NAME, 'Time: '+' '.join([f'{np.quantile(time_spent,q):.4g}' for q in QUANTS]))
    write_log(LOG_NAME, 'SNR: '+' '.join([f'{np.quantile(snr,q):.4g}' for q in QUANTS]))
    write_log(LOG_NAME, "")
  else:
    beta_size =  data_mat[:,0]
    r2 =  data_mat[:,1]
    time_spent = data_mat[:,2]
    write_log(LOG_NAME, f"Algorithm: {ALG}")
    write_log(LOG_NAME, f"Step size, gamma: {step_size:.3f} {gamma:.3f}")
    write_log(LOG_NAME, f"Experiments: {n_exps}")
    write_log(LOG_NAME, 'Beta_size: '+' '.join([f'{np.quantile(beta_size,q):.4g}' for q in QUANTS]))
    write_log(LOG_NAME, 'R2: '+' '.join([f'{np.quantile(r2,q):.4g}' for q in QUANTS]))
    write_log(LOG_NAME, 'Time: '+' '.join([f'{np.quantile(time_spent,q):.4g}' for q in QUANTS]))
    write_log(LOG_NAME, "")

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

def one_experiment_syn(seed, rho1, rho2, p, ALG, gamma, step_size):
  X_train, y_train, X_val, y_val, X_test, y_test, beta_star, snr = make_data_syn(seed, rho1, rho2, p)
  beta, beta_path, time_spent = sweep_alpha(X_train, y_train, X_val, y_val, ALG, gamma, step_size)
  return get_sens(beta, beta_star), get_spec(beta, beta_star), get_r2(y_test, X_test@beta), get_pred_err(beta, beta_star, X_test), get_est_err(beta, beta_star), time_spent, np.linalg.norm(beta_path[-1,:]), snr

def one_experiment_real(seed, ALG, DATA, gamma, step_size):
  #if seed % 1==0:
  #  print(seed)
  X_train, y_train, X_val, y_val, X_test, y_test = make_data_real(seed,DATA)
  beta, beta_path, time_spent = sweep_alpha(X_train, y_train, X_val, y_val, ALG, gamma, step_size)
  return get_beta_size(beta), get_r2(y_test, X_test@beta), time_spent



ALG='egdm'
DATA='syn'
SUF=''
rho1=0.7
rho2=0.3
p=50

gamma=0.5
step_size=0.01
N_EXPS=101

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

n_cpu=mp.cpu_count()
pool=mp.Pool(processes=n_cpu)

#one_experiment_syn(0, rho1, rho2, p, ALG, gamma, step_size)
#one_experiment_bs(0,0, gamma, step_size)

if DATA=='syn':
  LOG_NAME='logs/synth_'+str(p)+'_'+str(rho1)+'_'+str(rho2)+'_'+str(ALG)+SUF+'.txt'
  data_mat = np.array(pool.starmap(one_experiment_syn, [(seed, rho1, rho2, p, ALG, gamma, step_size) for seed in range(N_EXPS)]))
else:
  LOG_NAME='logs/'+DATA+'_'+str(ALG)+'.txt'
  data_mat = np.array(pool.starmap(one_experiment_real, [(seed, ALG, DATA, gamma, step_size) for seed in range(N_EXPS)]))
print_log(LOG_NAME, data_mat, ALG, DATA, step_size, gamma)

