def elastic_flow(X, y, ALPHA, GAMMA=0,
                 TAYLOR_COEFFS=4, #Number of coefficients in Taylor expansion.
                 MAX_SEARCH=10,   #Maximum future time considered when searching for new t_i (to speed up calculations).
                 ALPHA_DIFF=1e-3, #Maximum allowed deviation from alpha for coupled parameters. Needed since Taylor expansion is truncated.
                 T_MAX=1e10,      #Maximum time considered.
                 STEP_SIZE=0.01,  #Step size when plotting.
                 PLOT_NAME=None, PRINT=False, USE_OMEGA2=False):
  
  def my_sign(x):
    return np.sign(np.round(x,7))
  
  #Omega1 in Magnus expansion
  def get_Omega1(t,iis):
    assert not np.isnan(t)
    assert not t<0
    assert not t>1.1*MAX_SEARCH
    Omega1=0
    for k in range(iis.shape[0]):
      Omega1+=np.power(t,k+1)/np.math.factorial(k+1)*np.diag(iis[k,:]).dot(Ss)
    Omega1*=-(1-ALPHA)/(1-GAMMA)
    return Omega1
  
  #Omega2 in Magnus expansion
  def get_Omega2(t,iis):
    assert not np.isnan(t)
    assert not t<0
    assert not t>1.1*MAX_SEARCH
  
    Omega2=0
    for l1 in range(2,iis.shape[0]+1):
      for l2 in range(1,l1):
        I_l1_1=np.diag(iis[l1-1,:])
        I_l2_1=np.diag(iis[l2-1,:])
        commutator=I_l1_1.dot(Ss).dot(I_l2_1).dot(Ss)-I_l2_1.dot(Ss).dot(I_l1_1).dot(Ss)
        Omega2+=(l1-l2)/(np.math.factorial(l1)*np.math.factorial(l2)*(l1+l2))*np.power(t,l1+l2)*commutator
    Omega2*=((1-ALPHA)/(1-GAMMA))**2/2
    return Omega2
  
  #Calculate beta(t)
  def get_beta(t,beta_ti,iis):
    Omega=get_Omega1(t,iis)
    alpha_si=ALPHA/(1-ALPHA)*Ss_inv.dot(my_sign(Ss.dot(beta_hat-beta_ti)))
  
    if USE_OMEGA2:
      Omega+=get_Omega2(t,iis)
    beta = beta_hat + alpha_si - expm(Omega).dot(beta_hat+alpha_si-beta_ti)
    return beta
  
  #Calculate g(t)
  def get_grad(t,beta_ti,iis):
    grad = Ss.dot(get_beta(t,beta_ti,iis)-beta_hat)
    return grad

  #Calculate \bar{I}(t)
  def get_It_diag(t,iis):
    It_diag = np.zeros(iis.shape[1])
    for power in range(iis.shape[0]):
      It_diag+=iis[power,:]/np.math.factorial(power) * np.power(t,power)
    return It_diag
  
  #Secand method. Used for calculating next t_i
  def secant(func, t0, args=(), maxiter=50, tol=1e-8, t_max=100):
    f0 = func(t0, *args)
    #if np.isnan(f0):
    #  return -1
    t1=t0+1e-4
    f1 = func(t1, *args)
    if abs(f1) < abs(f0):
      t0, t1, f0, f1 = t1, t0, f1, f0
    t=0
    for itr in range(maxiter):
      if f1 == f0:
        return (t1 + t0) / 2.0
      else:
        if abs(f1) > abs(f0):
          t = (-f0 / f1 * t1 + t0) / (1 - f0 / f1)
        else:
          t = (-f1 / f0 * t0 + t1) / (1 - f1 / f0)
      if np.isclose(t, t1, rtol=0, atol=tol) or t>t_max or t<0:
        return t
      t0, f0 = t1, f1
      t1 = t
      f1 = func(t1, *args)
    return -1
  
  #Find first t_i, among all relevant parameters, d, for all possible criteria, 
  #  a parameter leaves the inactive set
  #  a parameter leaves the free set
  #  a parameter leaves the coupled set
  #  the maximum gradient component changes
  #  the gradient fraction for a coupled parameter deviates too much from ALPHA 
  #    (numerical reason, due to truncated Taylor expansion).
  def first_t(funs, search_inds, t_max):
    t_opt=t_max
    d_opt=-1
    for d in search_inds:
      for fun in funs:
        t_s = 1e-4
        sgn_0=np.sign(fun(1e-4,d))
        t=t_max+0.1
        for t_s in np.arange(0.1+1e-4,t_opt+0.1+1e-4,0.1): #check this
          if np.sign(fun(t_s,d))!=sgn_0:
            t=secant(fun,t0=t_s,args=[d],t_max=t_opt+0.1)
            if t>1e-4 and t<t_opt and np.abs(fun(t,d))<1e-4:
              t_opt=t
              d_opt=d
            break
    return t_opt, d_opt
  
  #Construct 'fun' corresponding to required criterion to use in 'first_t'. Call 'first_t' with this 'fun'.
  def search_t(search_type, beta_ti, m, search_inds, iis, f_value, max_search):
    if PRINT and not search_type=='stop': print('search '+ search_type.ljust(7),end='',flush=True)
    if search_type=='c':
      def ii(t,d):
        ii=0
        for power in range(iis.shape[0]):
          ii+=iis[power,d]/np.math.factorial(power) * np.power(t,power)
        return ii
      funs = [lambda t,d: ii(t,d), lambda t,d: ii(t,d)-1] #A coupled parameter gets a value in {0,1}
      t, d = first_t(funs, search_inds, max_search)
    elif search_type=='stop':
      def fun(t,d):
        return (beta_hat-get_beta(t,beta_ti,iis))[d] #beta(t) = beta_hat_ols
      return first_t([fun], range(p), max_search, True)
    else:
      def fun(t,d):
        return np.log(np.abs(get_grad(t,beta_ti,iis)[d]))-np.log(np.abs(get_grad(t,beta_ti,iis)[m]))-np.log(f_value) #gradient fraction equals 'f_value'
      t, d = first_t([fun], search_inds, max_search)
    if PRINT: print(t,d)
    max_search = min(max_search,t+0.1)
    return  t,d,max_search
  
  #Due to the Taylor expansion being finite, the coupled parameters drift slightly from their correct values.
  #This function corrects them at each t_i.
  def beta_corr(beta_in,S_F,S_C,S_0,m):
    beta_ti=np.copy(beta_in)
    s=(my_sign(Ss[S_C,:].dot(beta_hat-beta_ti))*my_sign(Ss[m,:].dot(beta_hat-beta_ti))).reshape((-1,1))
    A=(Ss[np.ix_(S_C,S_C)]-ALPHA*s.dot(Ss[np.ix_([m],S_C)]))
    b=(Ss[np.ix_(S_C,list(range(p)))]-ALPHA*s.dot(Ss[np.ix_([m],list(range(p)))])).dot(beta_hat)-(Ss[np.ix_(S_C,S_F+S_0)]-ALPHA*s.dot(Ss[np.ix_([m],S_F+S_0)])).dot(beta_ti[S_F+S_0])
    beta_ti[S_C]=np.linalg.solve(A,b)
    return beta_ti
  
  #Derivative of matrix exponential
  def get_expm_der(Xks,max_n=1000, tol=1e-6):
    Xnks=[np.eye(Xks[0].shape[0])]
    for k in range(1,len(Xks)):
      Xnks.append(np.zeros(Xks[k].shape))
    sum_n=0
    sum_n_old=-1
    for n in range(1,max_n):
      Xn_1ks = copy.deepcopy(Xnks) #{(X^(n-1))^(k)}_{k=0,1,...} 
      for k in range(len(Xks)):#Get d^kX^n/dt^k, k=0,1,...
        Xnks[k] = 0
        for i in range(k+1):
          Xnks[k]+=math.comb(k,i)*Xks[i].dot(Xn_1ks[k-i])
      sum_n+=1/np.math.factorial(n)*Xnks[-1]
      if np.all(np.abs(sum_n-sum_n_old)<tol):
        break
      sum_n_old=np.copy(sum_n)
    return sum_n
  
  
  def elastic_flow_step(beta_ti_in, iis_in, S_F, S_C, S_0, m, reason):
    import sys
    beta_ti=np.copy(beta_ti_in)
    iis=np.copy(iis_in) #Taylor coefficients of diagonal of \bar{I}
     
    #Find next t_i, and its reason
    max_search = MAX_SEARCH 
    if reason =='c1':
      t_c1, d_c1, max_search = search_t('c1', beta_ti, m, S_C, iis, ALPHA+ALPHA_DIFF, max_search) #coupled gradient fraction becomes ALPHA+ALPHA_DIFF
    if reason =='c2':
      t_c2, d_c2, max_search = search_t('c2', beta_ti, m, S_C, iis, ALPHA-ALPHA_DIFF, max_search) #coupled gradient fraction becomes ALPHA-ALPHA_DIFF
    t_alpha, d_alpha, max_search = search_t('alpha', beta_ti, m, list(filter(lambda d: d!=m, S_F+S_0)), iis, ALPHA, max_search) #free or zero gradient fraction becomes ALPHA (i.e. coupled)
    t_1, d_1, max_search = search_t('one', beta_ti, m, list(filter(lambda d: d!=m, S_F)), iis, 1., max_search) #new maximum gradient
    t_c, d_c, max_search = search_t('c', beta_ti, m, S_C, iis, None, max_search) #parameter leaves coupled set
    if reason!='c1':
      t_c1, d_c1, max_search = search_t('c1', beta_ti, m, S_C, iis, ALPHA+ALPHA_DIFF, max_search) #coupled gradient fraction becomes ALPHA+ALPHA_DIFF
    if reason!='c2':
      t_c2, d_c2, max_search = search_t('c2', beta_ti, m, S_C, iis, ALPHA-ALPHA_DIFF, max_search) #coupled gradient fraction becomes ALPHA-ALPHA_DIFF
  
    ts_temp=np.array([t_c1,t_c2,t_alpha,t_1,t_c])
    ds_temp=np.array([d_c1,d_c2,d_alpha,d_1,d_c])
     
    if np.all(ds_temp==-1):
      delta_t_opt=MAX_SEARCH
      d_opt=-1
      reason='max'
    else:
      min_idx=np.argmin(ts_temp+1000*(ds_temp==-1)) 
      delta_t_opt=ts_temp[min_idx]
      d_opt=ds_temp[min_idx]
      reason=REASONS[min_idx]
    if reason=='one':
      m=d_1
  
    if reason=='alpha':
      S_C.append(d_alpha)
      if d_alpha in S_F: 
        S_F.remove(d_alpha)
        #alpha_prev=1
      if d_alpha in S_0: 
        S_0.remove(d_alpha)
        #alpha_prev=0
  
    beta_ti=get_beta(delta_t_opt,beta_ti,iis)
    if len(S_C)>0:
      beta_ti=beta_corr(beta_ti,S_F,S_C,S_0,m)
  
    g_ti=(-Ss.dot(beta_hat-beta_ti)).reshape((-1,1))
  
    #Move parameters close to ALPHA to coupled set
    close_alphas=np.where(np.round(np.abs(g_ti/np.max(g_ti)),3)==ALPHA)[0]
    for c in close_alphas:
      if not c in S_C: S_C.append(c)
      if c in S_F: S_F.remove(c)
      if c in S_0: S_0.remove(c)
    
    iis[:,:]=0   #Set all Taylor coefficients to 0. (Late update those for free and coupled parameters.)
    iis[0,S_F]=1 #Set \bar{I}_dd to 1 for free parameters
    
    #Calculate Taylor coefficients for coupled parametrs
    gsi=((1-ALPHA)*g_ti+ALPHA*my_sign(g_ti)).reshape((-1,1))
    for k in range(TAYLOR_COEFFS):
      iis_0=np.copy(iis)
      iis_0[k,:]=0
      
      #Calculate derivatievs of Omega1 and Omega2
      Omega1_ks=[np.zeros((iis_0.shape[1],iis_0.shape[1]))]
      Omega2_ks=[np.zeros((iis_0.shape[1],iis_0.shape[1]))]
      Omega_ks=[np.zeros((iis_0.shape[1],iis_0.shape[1]))]
      for k1 in range(1,k+2):
        Omega1_k1=-(1-ALPHA)/(1-GAMMA)*np.diag(iis_0[k1-1,:]).dot(Ss)
        Omega2_k1=np.zeros((iis_0.shape[1],iis_0.shape[1]))
        if k1 >= 3:
          for l2 in range(1,int((k1-1)/2)+1):
            I_l1_1=np.diag(iis_0[k1-l2-1,:])
            I_l2_1=np.diag(iis_0[l2-1,:])
            commutator=I_l1_1.dot(Ss).dot(I_l2_1).dot(Ss)-I_l2_1.dot(Ss).dot(I_l1_1).dot(Ss)
            Omega2_k1+=((k1-2*l2)*np.math.factorial(k1-1))/(np.math.factorial(l2)*np.math.factorial(k1-l2))*commutator
        Omega2_k1*=((1-ALPHA)/(1-GAMMA))**2/2
        Omega1_ks.append(Omega1_k1)
        Omega2_ks.append(Omega2_k1)
        Omega_ks.append(Omega1_k1+Omega2_k1)
      
      if USE_OMEGA2:
        ck1=-(1-GAMMA)*Ss.dot(get_expm_der(Omega_ks)).dot(beta_hat.reshape((-1,1))-ALPHA/(1-ALPHA)*Ss_inv.dot(my_sign(g_ti))-beta_ti.reshape((-1,1))) #g^(k+1)(t_i)
      else:
        ck1=-(1-GAMMA)*Ss.dot(get_expm_der(Omega1_ks)).dot(beta_hat.reshape((-1,1))-ALPHA/(1-ALPHA)*Ss_inv.dot(my_sign(g_ti))-beta_ti.reshape((-1,1))) #g^(k+1)(t_i)
      
      while 1:
        #Calculate Taylor coefficients
        A=g_ti[S_C].dot(Ss[np.ix_([m],S_C)])-g_ti[m]*Ss[np.ix_(S_C,S_C)]
        if k==0:
          b=(g_ti[m]*Ss[np.ix_(S_C,S_F)]-g_ti[S_C].dot(Ss[np.ix_([m],S_F)])).dot(gsi[S_F])
        else:
          b=ck1[m]*g_ti[S_C]-g_ti[m]*ck1[S_C]
        ii_k_c=(np.linalg.solve(A,b)/gsi[S_C]).flatten()
        
        #If k=0, Remove "biggest" ii<0 or ii>1 and redo without.
        if k==0 and (np.any(ii_k_c<1e-4) or np.any(ii_k_c>1-1e-4)):
          max_viol = np.argmax(np.abs(ii_k_c-0.5))
          if ii_k_c[max_viol]<0.1:
            S_0.append(S_C[max_viol])
          elif ii_k_c[max_viol]>0.9:
            S_F.append(S_C[max_viol])
          else:
            print('ERROR!!')
            sys.exit()
          S_C.remove(S_C[max_viol])
          ii_k_c=np.delete(ii_k_c,max_viol)
        else:
          break

      iis[k,:]=np.zeros(p)
      if k==0:
        iis[0,S_F]=1
      iis[k,S_C]=ii_k_c

      #Is this needed?
      #for c in S_C:
      #  tOmega1_ks=[np.zeros((iis.shape[1],iis.shape[1]))]
      #  tOmega2_ks=[np.zeros((iis.shape[1],iis.shape[1]))]
      #  tOmega_ks=[np.zeros((iis.shape[1],iis.shape[1]))]
      #  for k1 in range(1,k+2):
      #    tOmega1_k1=-(1-ALPHA)*np.diag(iis[k1-1,:]).dot(Ss)
      #    tOmega2_k1=np.zeros((iis.shape[1],iis.shape[1]))
      #    if k1 >= 3:
      #      for l2 in range(1,int((k1-1)/2)+1):
      #        I_l1_1=np.diag(iis[k1-l2-1,:])
      #        I_l2_1=np.diag(iis[l2-1,:])
      #        commutator=I_l1_1.dot(Ss).dot(I_l2_1).dot(Ss)-I_l2_1.dot(Ss).dot(I_l1_1).dot(Ss)
      #        tOmega2_k1+=((k1-2*l2)*np.math.factorial(k1-1))/(np.math.factorial(l2)*np.math.factorial(k1-l2))*commutator
      #    tOmega1_ks.append(tOmega1_k1)
      #    tOmega2_ks.append(tOmega2_k1)
      #    tOmega_ks.append(tOmega1_k1+tOmega2_k1)
      #  g_ti_temp=get_grad(0,beta_ti,iis)
      #  if USE_OMEGA2:
      #    gk_ti_temp=-Ss.dot(get_expm_der(tOmega_ks)).dot(beta_hat.reshape((-1,1))-ALPHA/(1-ALPHA)*Ss_inv.dot(my_sign(g_ti))-beta_ti.reshape((-1,1))) #g^(k+1)(t_i)
      #  else:
      #    gk_ti_temp=-Ss.dot(get_expm_der(tOmega1_ks)).dot(beta_hat.reshape((-1,1))-ALPHA/(1-ALPHA)*Ss_inv.dot(my_sign(g_ti))-beta_ti.reshape((-1,1))) #g^(k+1)(t_i)
    return beta_ti, g_ti, iis, S_F, S_C, S_0, m, delta_t_opt, d_opt, reason
  
  import numpy as np
  import math, copy
  from scipy.linalg import expm
  import matplotlib.pyplot as plt
  np.set_printoptions(suppress=True)
  REASONS=['c1','c2','alpha','one','c']
  #c1:    coupled gradient fraction above ALPHA+ALPHA_DIFF
  #c2:    coupled gradient fraction below ALPHA-ALPHA_DIFF
  #alpha: free or zero gradient fraction becomes ALPHA (i.e. coupled)
  #one:   new maximum gradient
  #c:     parameter leaves coupled set
  
  n=X.shape[0]
  p=X.shape[1]
  Ss=1/n*X.T.dot(X) #Empirical covariannce matrix
  Ss_inv=np.linalg.inv(Ss)
  beta_hat = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y) #beta_ols
  
  beta_ti=np.zeros(p)
  g_ti=-X.T.dot(y)/n
  g_norm_ti_old=np.inf
  old_rel_grad=np.ones(p)
  m=np.argmax(np.abs(g_ti))
  iis=np.zeros((TAYLOR_COEFFS,p))
  iis[0,:]=1.0*(np.abs(g_ti)>=ALPHA*np.max(np.abs(g_ti)))
  S_F=np.where(iis[0,:]==1)[0].tolist()
  S_0=np.where(iis[0,:]==0)[0].tolist()
  S_C=[]
  t_tot=0
  step=0
  reason='start'
   
  betas_plot=[]
  grads_plot=[]
  h_alphas_plot=[]
  p1s_plot=[]
  if PRINT:
    print(step,t_tot,0,-1,'start')
    for r in range(0,iis.shape[0]):
      print('iis       ',iis[r,:])
    print('beta_ti   ',beta_ti)
    print('rel grad',get_grad(0,beta_ti,iis)/np.max(np.abs(get_grad(0,beta_ti,iis))))
    print('abs grad',get_grad(0,beta_ti,iis))
    print('')
  
  
  while t_tot<T_MAX and np.sum(np.square(get_grad(0,beta_ti,iis)))>1e-10:
    step+=1
    beta_ti_old=np.copy(beta_ti)
    iis_old = np.copy(iis)
    beta_ti, g_ti, iis, S_F, S_C, S_0, m, t_step, d_reason, reason = elastic_flow_step(beta_ti, iis, S_F, S_C, S_0, m, reason)
    t_tot+=t_step
  
    if d_reason==-1:
      break
    rel_grad=np.round(get_grad(0,beta_ti,iis)/np.max(np.abs(get_grad(0,beta_ti,iis))),5)
    #print(rel_grad==-old_rel_grad)
    #print('')
    g_norm_ti=np.linalg.norm(g_ti)
    if g_norm_ti>g_norm_ti_old:
      break
    g_norm_ti_old=g_norm_ti
    old_rel_grad=rel_grad
    
    for t_plot in np.arange(0,t_step,STEP_SIZE):
      beta_plot=get_beta(t_plot,beta_ti_old,iis_old)
      betas_plot.append(beta_plot)
      grad_plot=get_grad(t_plot,beta_ti_old,iis_old)
      grads_plot.append(grad_plot)
      
      It_diag = get_It_diag(t_plot,iis_old)
      delta_beta=ALPHA*np.sign(It_diag*grad_plot)+(1-ALPHA)*It_diag*grad_plot
      h_alphas_plot.append((delta_beta.dot(ALPHA*np.sign(delta_beta)+(1-ALPHA)*delta_beta)))
      p1s_plot.append(np.sum(It_diag))
      
  
    if PRINT:
      print(step,t_tot,t_step,d_reason, reason)
      print(S_F,S_C,S_0)
      for r in range(0,iis.shape[0]):
        print('iis       ',iis[r,:])
      print('beta_ti   ',beta_ti)
      print('rel grad',get_grad(0,beta_ti,iis)/np.max(np.abs(get_grad(0,beta_ti,iis))))
      print('abs grad',get_grad(0,beta_ti,iis))
      print('beta_hat',beta_hat) 
      print('')
    
    if not PLOT_NAME is None:
      betas=np.array(betas_plot)
      grads=np.array(grads_plot)
      ts=np.arange(0,betas.shape[0],1)*STEP_SIZE
      f, axs = plt.subplots(2, 1)
      for d in range(betas.shape[1]):
        axs[0].plot(ts, betas[:,d])
        axs[1].plot(ts, grads[:,d]/np.max(np.abs(grads),1))
      
      axs[0].hlines(0,0,STEP_SIZE*betas.shape[0],colors='k')
      axs[1].hlines(ALPHA,0,STEP_SIZE*betas.shape[0],colors='k')
      axs[1].hlines(-ALPHA,0,STEP_SIZE*betas.shape[0],colors='k')
      plt.savefig('figures/'+PLOT_NAME+'.pdf')
      plt.close()
  
  return np.array(betas_plot), np.array(grads_plot), np.array(h_alphas_plot), np.array(p1s_plot)

