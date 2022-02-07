def elastic_desc(X, y, alpha, step_size, alg='norm'):
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
      if alg=='unnorm':
        delta_beta=alpha*np.sign(el_grad)+(1-alpha)*el_grad
      elif alg[:4]=='norm':
        delta_beta=alpha*el_grad/np.linalg.norm(el_grad,1)+(1-alpha)*el_grad/np.linalg.norm(el_grad,2)
        if alg=='norm_s':
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
