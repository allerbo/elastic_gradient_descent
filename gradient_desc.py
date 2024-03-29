import numpy as np

def grad_desc(X, y, T_MAX=None, STEP_SIZE=0.01, STOP_ACC=None):

  def loop_fun(crit_choose, crit_1, crit_2):
    if crit_choose:
      return crit_1
    else:
      return crit_2
  
  def grad_desc_step(beta_in):
    beta=np.copy(beta_in)
    grad=-1/X.shape[0]*X.T.dot(y-X.dot(beta))
    beta -= STEP_SIZE*grad
    return beta, grad

  X_MAX=None if T_MAX is None else T_MAX/STEP_SIZE 
  if STOP_ACC is None: STOP_ACC=0.1*STEP_SIZE
  beta=np.zeros(X.shape[1])
  betas=[beta]
  grads=[]
  i=0
  while loop_fun(not X_MAX is None, not X_MAX is None and i<X_MAX, np.sum(np.square(grad_desc_step(beta)[1]))>STOP_ACC):
    i+=1
    beta,grad = grad_desc_step(beta)
    betas.append(beta)
    grads.append(grad)
  
  betas = np.array(betas[:-1]) #last beta is one step into the future
  grads = np.array(grads)
  
  return betas, grads

