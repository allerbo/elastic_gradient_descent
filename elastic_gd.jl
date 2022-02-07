module ElasticGD

export elastic_desc
export println1
export get_mse
export get_sens
export get_spec
export get_sens_spec
export get_path_frac
export train_val_test
export sweep_alpha
export make_folds

using Random
using Statistics
using LinearAlgebra
using Lasso
using StatsBase
using Combinatorics

function elastic_desc(X, y, alpha, step_size, alg="norm_ns", break_early=false)
  beta = zeros(size(X)[2],1)
  grad = nothing
  betas = [beta]
  grads = Array{Float64,2}[]
  old_norm_grad = Inf
  XtXn=X'*X/size(X)[1]
  Xtyn=X'*y/size(X)[1]
  c_alpha=(alpha+sqrt(alpha^2+4)-2)/(2*alpha)
  for i in 1:round(Int,100/step_size)
    grad=XtXn*beta-Xtyn
    I01=(abs.(grad)/maximum(abs.(grad)).>=alpha)
    elastic_grad=I01.*grad

    delta_beta=Nothing
    if alg=="unnorm"
      delta_beta=(alpha*sign.(elastic_grad)+(1-alpha)*elastic_grad)*step_size
    elseif alg[1:4]=="norm"
      delta_beta=(alpha*elastic_grad/norm(elastic_grad,1)+(1-alpha)*elastic_grad/norm(elastic_grad,2))*step_size
      if alg=="norm_s"
        q1=(norm(elastic_grad,1)/norm(elastic_grad,2))^2
        c=(sqrt(q1*(alpha^2*q1 + 4*(1-alpha)))-alpha*q1)/(2*(1-alpha)*(sqrt(q1)*(1-alpha)+alpha))
        delta_beta*=c
      end
    elseif alg[1:2]=="gs"
      c=1
      if alg=="gs_s"
        q1=(norm(elastic_grad,1)/norm(elastic_grad,2))^2
        c=((sqrt(sqrt(q1)*2*alpha*sqrt(alpha^2*q1 + 4*step_size*(1-alpha)) + q1*((1-alpha)^3-2*alpha^2)) - sqrt(q1)*(1 - alpha)*sqrt(1-alpha))/(2*sqrt(step_size)*alpha*sqrt(1- alpha)))^2
      end
      delta_beta=c*step_size*alpha*elastic_grad/norm(elastic_grad,1)+np.sqrt(c*step_size)*(1-alpha)*elastic_grad/norm(elastic_grad,2)
    end
    beta-=delta_beta

    push!(betas,beta)
    push!(grads,grad)
    
    if break_early
      norm_grad=norm(grad)
      if norm_grad<0.1*step_size
        break
      end
      if mod(i,round(Int, 1/step_size))==0
        if round(norm_grad/old_norm_grad; digits=4)==1
          break
        end
        old_norm_grad = norm_grad
      end
    end
  end
  return hcat(betas...)', hcat(grads...)'
end

function println1(LOG_NAME, print_str)
  if "SLURM_SUBMIT_DIR" in keys(ENV)
    f = open(string(ENV["SLURM_SUBMIT_DIR"],"/logs/",LOG_NAME,".txt"), "a")
    println(f, print_str)
    close(f)
  else
    println(print_str)
    #f = open("temp.txt", "a")
    #println(f, print_str)
    #close(f)
  end
end

function get_mse(y,y_hat)
  return mean((y.-y_hat).^2)
end

function get_sens(beta, beta_star)
  tp = sum((beta.!=0) .& (beta_star.!=0))
  p = sum(beta_star.!=0)
  return tp/p
end

function get_spec(beta, beta_star)
  tn = sum((beta.==0) .& (beta_star.==0))
  n = sum(beta_star.==0)
  return tn/n
end

function get_sens_spec(beta_eg, beta_en, beta_star)
  return get_sens(beta_eg, beta_star), get_sens(beta_en, beta_star), get_spec(beta_eg, beta_star), get_spec(beta_en, beta_star)
end

function get_path_frac(beta_path, beta_star)
  norms=sum(abs.(beta_path),dims=2)
  correct_betas=Bool[]
  for norm in 0:0.1:norms[end]
    idx=findlast(norm.>=norms)[1]
    beta_bool=beta_path[idx,:].!=0
    #if !all(beta_bool) #TODO: 0?
    push!(correct_betas, sum(.!((beta_bool.==0) .⊻ (beta_star.==0)))==length(beta_star))
    #end
  end
  return mean(correct_betas)
end


function get_idxs(n, splits)
  @assert sum(splits)==1
  r1 = round(Int, splits[1]*n)
  r2 = r1+round(Int, splits[2]*n)
  train_idxs=1:r1
  val_idxs=r1+1:r2
  if length(splits)==3
    test_idxs=r2+1:n
  else
    test_idxs=1:0
  end
  return train_idxs, val_idxs, test_idxs
end

function make_folds(n,n_folds; test=true, val_frac=0.15)
  all_folds=1:n_folds
  n_val_test=max(1,round(Int,val_frac*n_folds))
  fold_list=Array{Array{Int64,1},1}[]
  if test
    for test_folds in combinations(all_folds,n_val_test)
      train_val_folds=setdiff(all_folds, test_folds)
      for val_folds in combinations(train_val_folds,n_val_test)
        train_folds=setdiff(train_val_folds, val_folds)
        push!(fold_list, [train_folds, val_folds, test_folds])
      end
    end
  else
    for val_folds in combinations(all_folds,n_val_test)
      train_folds=setdiff(all_folds, val_folds)
      push!(fold_list, [train_folds, val_folds])
    end
  end
  n_idx_fold=n÷n_folds
  n_folds_extra=mod(n,n_folds)
  fold_ranges=UnitRange{Int64}[]
  for fold in 1:n_folds
    if fold<=n_folds_extra
      fold_range=(fold-1)*(n_idx_fold+1)+1:fold*(n_idx_fold+1)
    else
      fold_range=n_folds_extra+(fold-1)*n_idx_fold+1:n_folds_extra+fold*n_idx_fold
    end
    push!(fold_ranges,fold_range)
  end
  return fold_list, fold_ranges
end

function normalize_X(X)
  return (X.-mean(X, dims=1))./std(X, dims=1)
end

function normalize_y(y)
  return (y.-mean(y))
end

function train_val_test(X,y; splits=nothing, fold_list=nothing, fold_ranges=nothing, comb_idx=nothing, normalize=true, seed=0)
  Random.seed!(seed)
  p=randperm(size(X)[1])
  Xp=X[p,:]
  yp=y[p]
  if fold_list==nothing
    train_idxs, val_idxs, test_idxs = get_idxs(size(X)[1], splits)
  else
    flc=fold_list[comb_idx]
    if length(flc)==2
      push!(flc,[])
    end
    train_idxs, val_idxs, test_idxs = map(x->collect(Iterators.flatten(fold_ranges[x])), flc)
  end
  X_train=Xp[train_idxs,:]
  X_val=Xp[val_idxs,:]
  X_test=Xp[test_idxs,:]
  y_train=yp[train_idxs]
  y_val=yp[val_idxs]
  y_test=yp[test_idxs]
  if normalize
    X_train = normalize_X(X_train)
    X_val = normalize_X(X_val)
    X_test = normalize_X(X_test)
    y_train = normalize_y(y_train)
    y_val = normalize_y(y_val)
    y_test = normalize_y(y_test)
  end
  return X_train, y_train, X_val, y_val, X_test, y_test
end


function most_common_beta(beta_path, incl_ridge)
  norms=sum(abs.(beta_path),dims=2)
  beta_bools=Array{Bool,1}[]
  for norm in 0:0.1:norms[end]
    idx=findlast(norm.>=norms)[1]
    push!(beta_bools,beta_path[idx,:].!=0)
  end
  beta_counts=countmap(beta_bools)
  best_count=0
  best_beta=ones(size(beta_path)[2])
  for (beta, count) in beta_counts
    if count>best_count && any(beta) && (!all(beta) || incl_ridge)
#      more common        not only 0s      only 1s     include only ones
      best_count=count
      best_beta=beta
    end
  end
  return best_beta
end

function eval_alpha(beta_path, X_val, y_val)
  best_mse=Inf
  best_r=nothing
  for r in 1:size(beta_path)[1]
    beta=beta_path[r,:]
    mse=get_mse(y_val, X_val*beta)
    if mse<best_mse
      best_r=r
      best_mse=mse
    end
  end
  return best_r, best_mse
end

function select_alpha(mses, best_rs, beta_paths; alphas, mse_frac=0)
  alpha_idx=findlast(mses/minimum(mses).<=1+mse_frac)
  best_beta_path=beta_paths[alpha_idx]
  best_r=best_rs[alpha_idx]
  best_beta=best_beta_path[best_r,:]
  return best_beta, best_beta_path, alphas[alpha_idx]
end

function sweep_alpha(X_train, y_train, X_val, y_val, alg; mse_frac=0, n_folds=10, alphas=0.01:0.01:1, step_size=1e-2, nλ=100, λminratio=ifelse(size(X_train, 1) < size(X_train, 2), 0.01, 1e-4))
  best_rs_eg= Int64[]
  best_rs_en= Int64[]
  beta_paths_eg= Array{Float64,2}[]
  beta_paths_en= Array{Float64,2}[]
  mses_eg= Float64[]
  mses_en= Float64[]
  for alpha in alphas
    beta_path_eg = elastic_desc(X_train, y_train, alpha, step_size, alg)[1]
    best_r_eg, best_mse_eg = eval_alpha(beta_path_eg, X_val, y_val)
    push!(beta_paths_eg, beta_path_eg)
    push!(best_rs_eg, best_r_eg)
    push!(mses_eg, best_mse_eg)

    beta_path_en=fit(LassoPath, X_train, y_train; α=alpha, nλ=nλ, λminratio=λminratio).coefs'
    best_r_en, best_mse_en = eval_alpha(beta_path_en, X_val, y_val)
    push!(beta_paths_en, beta_path_en)
    push!(best_rs_en, best_r_en)
    push!(mses_en, best_mse_en)
  end
  beta_mse_eg, beta_path_eg, alpha_eg = select_alpha(mses_eg, best_rs_eg, beta_paths_eg; alphas=alphas, mse_frac=mse_frac)
  beta_mse_en, beta_path_en, alpha_en = select_alpha(mses_en, best_rs_en, beta_paths_en; alphas=alphas, mse_frac=mse_frac)
  beta_cv_eg, beta_cv_en= select_glmnet(X_train, y_train, alpha_eg, alpha_en, alg; n_folds=n_folds, step_size=step_size, nλ=nλ, λminratio=λminratio)
  beta_mc_eg = most_common_beta(beta_path_eg, true)
  beta_mc_en = most_common_beta(beta_path_en, true)
  beta_mc1_eg = most_common_beta(beta_path_eg, false)
  beta_mc1_en = most_common_beta(beta_path_en, false)
  return beta_mse_eg, beta_mse_en, beta_cv_eg, beta_cv_en, beta_path_eg, beta_path_en, alpha_eg, alpha_en, beta_mc_eg, beta_mc_en, beta_mc1_eg, beta_mc1_en
end


function select_glmnet(X_train, y_train, alpha_eg, alpha_en, alg; n_folds=10, step_size=1e-2, nλ=100, λminratio=ifelse(size(X_train, 1) < size(X_train, 2), 0.01, 1e-4))
  fold_list, fold_ranges = make_folds(size(X_train)[1], n_folds; test=false, val_frac=1/n_folds)
  λmax=maximum(abs.(X_train'*y_train/length(y_train)))/alpha_en
  logλmax = log(λmax)
  lbdas=exp.(range(logλmax, stop=logλmax + log(λminratio), length=nλ))
  mses_eg= Array{Float64,1}[]
  mses_en= Array{Float64,1}[]
  for k in 1:n_folds
    X_train_cv, y_train_cv, X_val_cv, y_val_cv, _, _ = train_val_test(X_train, y_train; fold_list=fold_list, fold_ranges=fold_ranges, comb_idx=k, seed=4, normalize=false)
  
    beta_path_eg = elastic_desc(X_train_cv, y_train_cv, alpha_eg, step_size, alg)[1]
    mse_eg=map(beta->get_mse(y_val_cv, X_val_cv*beta), map(r->beta_path_eg[r,:],1:size(beta_path_eg)[1]))
    push!(mses_eg, mse_eg)
  
    beta_path_en=fit(LassoPath, X_train_cv, y_train_cv; α=alpha_en, λ=lbdas).coefs'
    mse_en=map(beta->get_mse(y_val_cv, X_val_cv*beta), map(r->beta_path_en[r,:],1:size(beta_path_en)[1]))
    push!(mses_en, mse_en)
  end
  r_eg=select_pen(mses_eg)

  beta_eg = elastic_desc(X_train, y_train, alpha_eg, step_size, alg)[1][r_eg,:]
  r_en=select_pen(mses_en)
  beta_en=collect(fit(LassoPath, X_train, y_train; α=alpha_en, nλ=nλ, λminratio=λminratio).coefs'[r_en,:])
  return beta_eg, beta_en
end

function select_pen(mses)
  min_len=minimum(map(length,mses))
  mses=hcat(map(mses->mses[1:min_len],mses)...)
  min_mean=minimum(mean(mses, dims=2))
  return findfirst(mean(mses, dims=2).-std(mses, dims=2)/sqrt(size(mses)[2]).<min_mean)[1]
end

function to_norm(beta_path)
  betas=Array{Float64,1}[]
  norms=sum(abs.(beta_path),dims=2)
  for norm_i in 0:1:norms[end]
    idx=findfirst(norm_i.<=norms)[1]
    push!(betas,beta_path[idx,:])
  end
  return hcat(betas...)'
end
end
