def coord_flow(X, y, STEP_SIZE=0.01, PLOT=False, PRINT=False):
  import numpy as np
  np.set_printoptions(suppress=True)

  #Calculate beta(t)
  def get_beta(t,beta_ti,ii):
    beta = beta_ti+t*ii*np.sign(Ss.dot(beta_hat-beta_ti))#SG
    return beta
  
  #Calculate g(t)
  def get_grad(t,beta_ti,ii):
    grad = Ss.dot(get_beta(t,beta_ti,ii)-beta_hat)
    return grad
  
  def coord_flow_step(beta_ti_in, ii_in, S_M):
    beta_ti=np.copy(beta_ti_in)
    ii=np.copy(ii_in) #Diagonal of \bar{I}
     
    si=np.sign(np.round(Ss.dot(beta_hat-beta_ti),7)) #Signs
    delta_t_opt=np.inf
    d_opt=-1
    #Find next t_i
    for d in list(set(range(p)).difference(set(S_M))):
      for m in S_M:
        for sgn in (-1,1):
          delta_t=(Ss[d,:]+sgn*Ss[m,:]).dot(beta_hat-beta_ti)/(Ss[d,:]+sgn*Ss[m,:]).dot(ii*si)
          if delta_t>1e-8 and delta_t<delta_t_opt:
            delta_t_opt=delta_t
            d_opt=d
    
    #Check if beta(t) has converged.
    if delta_t_opt==np.inf:
      t_maxs=(beta_hat-beta_ti)/(ii*si)
      assert len(set(np.round(t_maxs,7)))==1
      delta_t_max=t_maxs[0]
      beta_ti=get_beta(delta_t_max,beta_ti,ii)
      g_ti=Ss.dot(beta_ti-beta_hat)
      return beta_ti, g_ti, ii, S_M, delta_t_max, -1
      
    beta_ti=get_beta(delta_t_opt,beta_ti,ii)
    g_ti=Ss.dot(beta_ti-beta_hat)

    #Add new element to active set
    S_M.append(d_opt)
    
    #Calculate \bar{I}
    si=np.sign(np.round(Ss.dot(beta_hat-beta_ti),7))
    if len(S_M)>1:
      while 1:
        A=si[S_M].T
        for m_i in range(len(S_M)-1):
          m1=S_M[m_i]
          m2=S_M[m_i+1]
          A=np.vstack((A,si[m1]*Ss[m1,S_M]-si[m2]*Ss[m2,S_M]))
        ii_m=np.linalg.inv(A)[:,0]*si[S_M]
        #Remove "biggest" ii<0 and redo without.
        if np.any(ii_m<0):
          ii_temp=np.zeros(p)
          ii_temp[S_M]=ii_m
          max_viol = np.argmin(ii_m)
          S_M.remove(S_M[max_viol])
        else:
          break
      ii=np.zeros(p)
      ii[S_M]=ii_m
    else:
      ii=np.zeros(p)
      ii[np.argmax(np.abs(g_ti))]=1
     
    #S_M=np.where(ii>0)[0].tolist()
    return beta_ti, g_ti, ii, S_M, delta_t_opt, d_opt
  
  #Initialize
  n=X.shape[0]
  p=X.shape[1]
  Ss=1/n*X.T.dot(X)
  beta_hat = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
  
  beta_ti=np.zeros(p)
  g_ti=-X.T.dot(y)/n
  ii=np.zeros(p)
  ii[np.argmax(np.abs(g_ti))]=1
  S_M=[np.argmax(np.abs(g_ti))]
  t_tot=0
  step=0
   
  betas_plot=[]
  grads_plot=[]
  if PRINT:
    print(step,t_tot,0,-1,'start')
    print('ii      ',ii)
    print('beta_ti   ',beta_ti)
    print('rel grad',get_grad(0,beta_ti,ii)/np.max(np.abs(get_grad(0,beta_ti,ii))))
    print('abs grad',get_grad(0,beta_ti,ii))
    print('')
  
  while np.sum(np.square(get_grad(0,beta_ti,ii)))>1e-10:
    step+=1
    beta_ti_old=np.copy(beta_ti)
    ii_old = np.copy(ii)
    beta_ti, g_ti, ii, S_M, t_step, d_reason= coord_flow_step(beta_ti, ii, S_M)
    t_tot+=t_step
    
    for t_plot in np.arange(0,t_step,STEP_SIZE):
      beta_plot=get_beta(t_plot,beta_ti_old,ii_old)
      betas_plot.append(beta_plot)
      grad_plot=get_grad(t_plot,beta_ti_old,ii_old)
      grads_plot.append(grad_plot)

    if PLOT:
      egd_plot('flow', np.array(betas_plot), np.array(grads_plot), 1, STEP_SIZE)
      
    if PRINT:
      print(step,t_tot,t_step,d_reason)
      print('ii      ',ii)
      print('beta_ti   ',beta_ti)
      print('rel grad',get_grad(0,beta_ti,ii)/np.max(np.abs(get_grad(0,beta_ti,ii))))
      print('abs grad',get_grad(0,beta_ti,ii))
      print('')
    
    if d_reason==-1:
      break
  print(beta_hat) 

  return np.array(betas_plot), np.array(grads_plot)
