def elastic_desc(X, y, alpha, STEP_SIZE=0.01, T_MAX=None, STOP_ACC=None):
  import numpy as np

  def loop_fun(crit_choose, crit_1, crit_2):
    if crit_choose:
      return crit_1
    else:
      return crit_2
  
  def elastic_desc_step(beta_in):
    beta=np.copy(beta_in)
    grad=-1/X.shape[0]*X.T.dot(y-X.dot(beta))
    I01=1*(np.abs(grad)/np.max(np.abs(grad))>=alpha)
    el_grad=I01*grad
    beta -= STEP_SIZE*(alpha*np.sign(el_grad)+(1-alpha)*el_grad)
    return beta, grad

  X_MAX=None if T_MAX is None else T_MAX/STEP_SIZE 
  if STOP_ACC is None: STOP_ACC=0.1*STEP_SIZE
  beta=np.zeros(X.shape[1])
  betas=[beta]
  grads=[]
  i=0
  while loop_fun(not X_MAX is None, not X_MAX is None and i<X_MAX, np.sum(np.square(elastic_desc_step(beta)[1]))>STOP_ACC):
    i+=1
    beta,grad = elastic_desc_step(beta)
    betas.append(beta)
    grads.append(grad)
  
  betas = np.array(betas[:-1]) #last beta is one step into the future
  grads = np.array(grads)
  
  return betas, grads

def elastic_desc_flav(X, y, alpha, alg, step_size=0.01):
  import numpy as np
  beta=np.zeros(X.shape[1])
  betas=[beta]
  grads=[]
  h_alphas=[]
  p1s=[]
  old_norm_grad = np.inf
  XtXn=X.T.dot(X)/X.shape[0]
  Xtyn=X.T.dot(y)/X.shape[0]
  for i in range(int(1000/step_size)):
    grad=XtXn.dot(beta)-Xtyn
    el_grad=(np.abs(grad)/np.max(np.abs(grad))>=alpha)*grad
    if alg[:2]=="gs":
      c_alpha=1
      if alg=="gs_s":
        q1=(np.linalg.norm(el_grad,1)/np.linalg.norm(el_grad,2))**2
        c_alpha=((np.sqrt(np.sqrt(q1)*2*alpha*np.sqrt(alpha**2*q1 + 4*step_size*(1-alpha)) + q1*((1-alpha)**3-2*alpha**2)) - np.sqrt(q1)*(1 - alpha)*np.sqrt(1-alpha))/(2*np.sqrt(step_size)*alpha*np.sqrt(1- alpha)))**2
      delta_beta=c_alpha*step_size*alpha*el_grad/np.linalg.norm(el_grad,1)+np.sqrt(c_alpha*step_size)*(1-alpha)*el_grad/np.linalg.norm(el_grad,2)
      beta = beta-delta_beta
    else:
      if alg=='std':
        delta_beta=alpha*np.sign(el_grad)+(1-alpha)*el_grad
      elif alg[:2]=='sd':
        delta_beta=alpha*el_grad/np.linalg.norm(el_grad,1)+(1-alpha)*el_grad/np.linalg.norm(el_grad,2)
        if alg=='sd_s':
          q1=(np.linalg.norm(el_grad,1)/np.linalg.norm(el_grad,2))**2
          c_alpha=(np.sqrt(q1*(alpha**2*q1 + 4*(1-alpha)))-alpha*q1)/(2*(1-alpha)*(np.sqrt(q1)*(1-alpha)+alpha))
          delta_beta*=c_alpha
      beta = beta-step_size*delta_beta

    betas.append(beta)
    grads.append(grad)
    h_alphas.append((delta_beta.dot(alpha*np.sign(delta_beta)+(1-alpha)*delta_beta)))
    p1s.append(np.linalg.norm(np.sign(el_grad),1))
    
    norm_grad=np.sqrt(np.sum(np.square(grad)))
    if norm_grad<0.1*step_size:
      break
    if i % int(1./step_size)==0:
      if np.round(norm_grad/old_norm_grad,4)==1:
        break
      old_norm_grad = norm_grad
  return np.array(betas)[:-1,:], np.array(grads), np.array(h_alphas), np.array(p1s)
