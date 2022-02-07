using Distributed
@everywhere include("./elastic_gd.jl")
@everywhere begin
	using LinearAlgebra
	using Random, Distributions
  using .ElasticGD
end

using DelimitedFiles, Formatting


@everywhere function make_data_oa(seed, rho1, rho2, p, sigma, n, beta_var)
	Random.seed!(seed)
	
	Ss11=(1-rho1)*I+rho1*ones(p,p)
	Ss12=rho2*ones(p,p)
	Ss=vcat(hcat(Ss11,Ss12),hcat(Ss12',Ss11))
	
  beta_star=vcat(rand(Normal(2,beta_var),p),zeros(p))
	
	X=rand(MvNormal(Ss),n)'
	y=X*beta_star+rand(Normal(0,sigma),n)
  X_train, y_train, X_val, y_val, X_test, y_test = train_val_test(X,y,splits=[0.6, 0.2, 0.2])

	return X_train, y_train, X_val, y_val, X_test, y_test, beta_star
end


function print_log(LOG_NAME, data_mat)
  n_obs=size(data_mat)[1]
  
	sens_mse_eg = data_mat[:,1]
  sens_mse_en = data_mat[:,2]
  spec_mse_eg = data_mat[:,3]
  spec_mse_en = data_mat[:,4]
  sens_cv_eg =  data_mat[:,5]
  sens_cv_en =  data_mat[:,6]
  spec_cv_eg =  data_mat[:,7]
  spec_cv_en =  data_mat[:,8]
  path_frac_eg = data_mat[:,9]
  path_frac_en = data_mat[:,10]
  mse_mse_eg =  data_mat[:,11]
  mse_mse_en =  data_mat[:,12]
  mse_cv_eg =  data_mat[:,13]
  mse_cv_en =  data_mat[:,14]
  alpha_eg = data_mat[:,15]
  alpha_en = data_mat[:,16]

  sens_mse_diff=sens_mse_eg.-sens_mse_en
  spec_mse_diff=spec_mse_eg.-spec_mse_en
  mse_mse_diff= mse_mse_eg.- mse_mse_en
  sens_cv_diff= sens_cv_eg.- sens_cv_en
  spec_cv_diff= spec_cv_eg.- spec_cv_en
  mse_cv_diff=  mse_cv_eg.-  mse_cv_en
  path_frac_diff=path_frac_eg.-path_frac_en

  sens_mc_eg =  data_mat[:,17]
  sens_mc_en =  data_mat[:,18]
  spec_mc_eg =  data_mat[:,19]
  spec_mc_en =  data_mat[:,20]
  sens_mc1_eg = data_mat[:,21]
  sens_mc1_en = data_mat[:,22]
  spec_mc1_eg = data_mat[:,23]
  spec_mc1_en = data_mat[:,24]

  sens_mc_diff= sens_mc_eg.-sens_mc_en
  spec_mc_diff= spec_mc_eg.-spec_mc_en
  sens_mc1_diff=sens_mc_eg.-sens_mc_en
  spec_mc1_diff=spec_mc_eg.-spec_mc_en

	data_mat_save = hcat(mse_mse_eg, mse_mse_en, mse_cv_eg, mse_cv_en, path_frac_eg, path_frac_en, alpha_eg, alpha_en)
	writedlm(string("data/",LOG_NAME,".txt"), data_mat_save, ' ')
	println1(LOG_NAME, format("Observations: {1:d}", n_obs))
	println1(LOG_NAME, "MSE")
	println1(LOG_NAME, format("eg. sens: {1:.3f} ({2:.3f}). spec: {3:.3f} ({4:.3f}). mse: {5:.3f} ({6:.3f}).", mean(sens_mse_eg), 3/sqrt(n_obs)*std(sens_mse_eg), mean(spec_mse_eg), 3/sqrt(n_obs)*std(spec_mse_eg), mean(mse_mse_eg), 3/sqrt(n_obs)*std(mse_mse_eg)))
	println1(LOG_NAME, format("en. sens: {1:.3f} ({2:.3f}). spec: {3:.3f} ({4:.3f}). mse: {5:.3f} ({6:.3f}).", mean(sens_mse_en), 3/sqrt(n_obs)*std(sens_mse_en), mean(spec_mse_en), 3/sqrt(n_obs)*std(spec_mse_en), mean(mse_mse_en), 3/sqrt(n_obs)*std(mse_mse_en)))
	println1(LOG_NAME, format("diff. sens: {1:.3f} ({2:.3f}). spec: {3:.3f} ({4:.3f}). mse: {5:.3f} ({6:.3f}).", mean(sens_mse_diff), 3/sqrt(n_obs)*std(sens_mse_diff), mean(spec_mse_diff), 3/sqrt(n_obs)*std(spec_mse_diff), mean(mse_mse_diff), 3/sqrt(n_obs)*std(mse_mse_diff)))
	println1(LOG_NAME, "CV")
	println1(LOG_NAME, format("eg. sens: {1:.3f} ({2:.3f}). spec: {3:.3f} ({4:.3f}). mse: {5:.3f} ({6:.3f}).", mean(sens_cv_eg), 3/sqrt(n_obs)*std(sens_cv_eg), mean(spec_cv_eg), 3/sqrt(n_obs)*std(spec_cv_eg), mean(mse_cv_eg), 3/sqrt(n_obs)*std(mse_cv_eg)))
	println1(LOG_NAME, format("en. sens: {1:.3f} ({2:.3f}). spec: {3:.3f} ({4:.3f}). mse: {5:.3f} ({6:.3f}).", mean(sens_cv_en), 3/sqrt(n_obs)*std(sens_cv_en), mean(spec_cv_en), 3/sqrt(n_obs)*std(spec_cv_en), mean(mse_cv_en), 3/sqrt(n_obs)*std(mse_cv_en)))
	println1(LOG_NAME, format("diff. sens: {1:.3f} ({2:.3f}). spec: {3:.3f} ({4:.3f}). mse: {5:.3f} ({6:.3f}).", mean(sens_cv_diff), 3/sqrt(n_obs)*std(sens_cv_diff), mean(spec_cv_diff), 3/sqrt(n_obs)*std(spec_cv_diff), mean(mse_cv_diff), 3/sqrt(n_obs)*std(mse_cv_diff)))

	println1(LOG_NAME, "PATH FRAC")
	println1(LOG_NAME, format("eg: {1:.3f} ({2:.3f}).", mean(path_frac_eg), 3/sqrt(n_obs)*std(path_frac_eg)))
	println1(LOG_NAME, format("en: {1:.3f} ({2:.3f}).", mean(path_frac_en), 3/sqrt(n_obs)*std(path_frac_en)))
	println1(LOG_NAME, format("diff: {1:.3f} ({2:.3f}).", mean(path_frac_diff), 3/sqrt(n_obs)*std(path_frac_diff)))

	println1(LOG_NAME, "MC")
	println1(LOG_NAME, format("eg. sens: {1:.3f} ({2:.3f}). spec: {3:.3f} ({4:.3f}).", mean(sens_mc_eg), 3/sqrt(n_obs)*std(sens_mc_eg), mean(spec_mc_eg), 3/sqrt(n_obs)*std(spec_mc_eg)))
	println1(LOG_NAME, format("en. sens: {1:.3f} ({2:.3f}). spec: {3:.3f} ({4:.3f}).", mean(sens_mc_en), 3/sqrt(n_obs)*std(sens_mc_en), mean(spec_mc_en), 3/sqrt(n_obs)*std(spec_mc_en)))
	println1(LOG_NAME, format("diff. sens: {1:.3f} ({2:.3f}). spec: {3:.3f} ({4:.3f}).", mean(sens_mc_diff), 3/sqrt(n_obs)*std(sens_mc_diff), mean(spec_mc_diff), 3/sqrt(n_obs)*std(spec_mc_diff)))

	println1(LOG_NAME, "MC1")
	println1(LOG_NAME, format("eg. sens: {1:.3f} ({2:.3f}). spec: {3:.3f} ({4:.3f}).", mean(sens_mc1_eg), 3/sqrt(n_obs)*std(sens_mc1_eg), mean(spec_mc1_eg), 3/sqrt(n_obs)*std(spec_mc1_eg)))
	println1(LOG_NAME, format("en. sens: {1:.3f} ({2:.3f}). spec: {3:.3f} ({4:.3f}).", mean(sens_mc1_en), 3/sqrt(n_obs)*std(sens_mc1_en), mean(spec_mc1_en), 3/sqrt(n_obs)*std(spec_mc1_en)))
	println1(LOG_NAME, format("diff. sens: {1:.3f} ({2:.3f}). spec: {3:.3f} ({4:.3f}).", mean(sens_mc1_diff), 3/sqrt(n_obs)*std(sens_mc1_diff), mean(spec_mc1_diff), 3/sqrt(n_obs)*std(spec_mc1_diff)))

	println1(LOG_NAME, "")
end

@everywhere function eg_en_iter(seed, rho1, rho2, p, sigma, n, beta_var, alg, n_folds)
	X_train, y_train, X_val, y_val, X_test, y_test, beta_star = make_data_oa(seed, rho1, rho2, p, sigma, n, beta_var)
  beta_mse_eg, beta_mse_en, beta_cv_eg, beta_cv_en, beta_path_eg, beta_path_en, alpha_eg, alpha_en, beta_mc_eg, beta_mc_en, beta_mc1_eg, beta_mc1_en = sweep_alpha(X_train, y_train, X_val, y_val, alg; n_folds=n_folds)
	sens_mse_eg, sens_mse_en, spec_mse_eg, spec_mse_en = get_sens_spec(beta_mse_eg, beta_mse_en, beta_star)
	sens_cv_eg,  sens_cv_en,  spec_cv_eg,  spec_cv_en  = get_sens_spec(beta_cv_eg,  beta_cv_en,  beta_star)
  path_frac_eg = get_path_frac(beta_path_eg, beta_star)
  path_frac_en = get_path_frac(beta_path_en, beta_star)
  mse_mse_eg=get_mse(y_test, X_test*beta_mse_eg)
  mse_mse_en=get_mse(y_test, X_test*beta_mse_en)
  mse_cv_eg=get_mse(y_test, X_test*beta_cv_eg)
  mse_cv_en=get_mse(y_test, X_test*beta_cv_en)

	sens_mc_eg,  sens_mc_en,  spec_mc_eg,  spec_mc_en  = get_sens_spec(beta_mc_eg,  beta_mc_en,  beta_star)
	sens_mc1_eg, sens_mc1_en, spec_mc1_eg, spec_mc1_en = get_sens_spec(beta_mc1_eg, beta_mc1_en, beta_star)

	return sens_mse_eg, sens_mse_en, spec_mse_eg, spec_mse_en, sens_cv_eg, sens_cv_en, spec_cv_eg, spec_cv_en, path_frac_eg, path_frac_en, mse_mse_eg, mse_mse_en, mse_cv_eg, mse_cv_en, alpha_eg, alpha_en, sens_mc_eg, sens_mc_en, spec_mc_eg, spec_mc_en, sens_mc1_eg, sens_mc1_en, spec_mc1_eg, spec_mc1_en
end

N_OBS=1000
RHO1=0.8
RHO2=0.5
p=10
sigma=10
n=100
BETA_VAR=1
N_FOLDS=10
GPU=""
ALG="norm_ns" #unnorm, norm_ns, norm_s, gs_ns, gs_s
for arg in ARGS
  eval(Meta.parse(arg))
end

LOG_NAME=string("run_synth_",RHO1,"_",RHO2,"_",ALG)
LOG_NAME=string(LOG_NAME,GPU)

nw=nworkers()
n_loops=N_OBS ÷ nw
seed_ranges=Any[]
for i_loop in 1:n_loops
	push!(seed_ranges,(i_loop-1)*nw+1:i_loop*nw)
end
n_last_loop=N_OBS- n_loops*nw
if n_last_loop>0
  push!(seed_ranges,n_loops*nw+1:n_loops*nw+n_last_loop)
end

eg_en_iter(1, RHO1, RHO2, p, sigma, n, BETA_VAR, ALG, N_FOLDS)
data_mat = zeros(0,24)

for seeds in seed_ranges
	data_mat_loc = pmap(seed -> eg_en_iter(seed, RHO1, RHO2, p, sigma, n, BETA_VAR, ALG, N_FOLDS), seeds)
  global data_mat=vcat(data_mat,hcat(map(collect, data_mat_loc)...)')
  
  print_log(LOG_NAME, data_mat)
end
