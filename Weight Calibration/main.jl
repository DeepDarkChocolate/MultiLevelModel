# Normal; Cluster sampling
#126
n = 100
#m = 20
N = 10000
p = 40
r = 0.75

β = vcat(1.0, zeros(10), fill(1.5, 10), zeros(10), fill(3.0, 10))

B = 1000 # 497

eps_theta = 10^(-3)

using Distributions
using Random
using LinearAlgebra
using NamedArrays
using StatsBase
using MixedModels
using DataFrames
using QuadGK
#using Optim
#using Roots
#using RCall
#using Dates
using CSV

using FFTW
using ToeplitzMatrices
using Lasso
using GLMNet
using GLM
using MLBase
using Plots

include("ftns.jl")

if isfile(joinpath(dirname(@__FILE__), "log.txt"))
  rm(joinpath(dirname(@__FILE__), "log.txt"))
end

MSE_res = Array{Float64}(undef, B, 3)

## Simulation start
verbose = false

#@time for simnum in 1:B
@time Threads.@threads for simnum in 1:B
  Random.seed!(simnum)
  open(joinpath(dirname(@__FILE__), "log.txt"), "a+") do io
  write(io, "$simnum\n")
  println(simnum)
  end;

  e = randn(N)

  #a_tilde = exp.(2.5 .+ a) ./ (1 .+ exp.(2.5 .+ a))
  #M = round.(Int, Mi .* a_tilde)

  #x = randn(N, p)
  dist = MvNormal(zeros(p), Symmetric(SymmetricToeplitz(0.2.^(collect(1:p) .- 1))))
  x = transpose(rand(dist, N))
  x = cdf.(Normal(0, 1), x)
  x = [ones(N) x]

  y = x * β + e


  ratio = (1 - r) / r
  z = rand(Normal(0, sqrt(ratio)), N) .+ e
  z_sorted_idx = sortperm(z)
  len = Int(round(N / 4))

  SampleIdx = Vector{Int64}(undef, n)
  w_sampled  = Vector{Float64}(undef, n)
  n_h = [15, 20, 30, 35]
  cumn_h = cumsum(n_h)
  for i in 1:4
    Idx_z = (len * (i - 1) + 1):(len * i)
    Idx = z_sorted_idx[Idx_z]
    from = (i == 1) ? 0 : cumn_h[i-1]
    SampleIdx[(from + 1) : cumn_h[i]] = StatsBase.sample(Idx, n_h[i], replace=false)
    w_sampled[(from + 1) : cumn_h[i]] .= len / n_h[i]
  end
  SampleIdx
  w_sampled

  #y = [vcat((@. rand(Binomial(1, p[i]), 1)...)) for i in 1:length(M)]
  x_sampled = x[SampleIdx, :]
  y_sampled = y[SampleIdx]

  path = fit(LassoPath, diagm(sqrt.(w_sampled)) * x_sampled, diagm(sqrt.(w_sampled)) * y_sampled, standardize=false, intercept = false)
  @show coef(path; select=MinBIC())

  path = fit(LassoPath, x_sampled[:,2:(p+1)], y_sampled, wts = w_sampled, standardize=false)
  @show coef(path; select=MinBIC())

  # True parameter
  Y_t = sum(y)
  X_t = vcat(mapslices(sum, x, dims = 1)...)

  # Horvitz Thompson Estimator
  Y_HT = sum(w_sampled .* y_sampled)
  X_HT = transpose(x_sampled) * w_sampled

  # Regression Estimator
  beta_wls = glm(x_sampled, y_sampled, Normal(), IdentityLink(), wts=w_sampled)
  #beta_wls = glm(diagm(sqrt.(w_sampled)) * x_sampled, diagm(sqrt.(w_sampled)) * y_sampled, Normal(), IdentityLink())

  Y_REG = Y_HT + sum((X_t .- X_HT) .* coef(beta_wls))

  # Ridge Regression

path = fit(LassoPath, diagm(sqrt.(w_sampled)) * x_sampled, diagm(sqrt.(w_sampled)) * y_sampled, standardize=false, intercept = false)
β_Lasso = coef(path; select=MinCVmse(Kfold(100,30)))

length(β_Lasso.nzval)
vcat(mapslices(x -> sum(.!iszero.(x)), Matrix(path.coefs); dims = 1)...)


Y_Lasso = Y_HT + sum((X_t .- X_HT) .* β_Lasso)

x_star = [ones(N) x * β_Lasso]
X_t_star = vcat(mapslices(sum, x_star, dims = 1)...)
x_sampled_star = [ones(n) x_sampled * β_Lasso]
X_HT_star = transpose(x_sampled_star) * w_sampled

beta_wls_star = glm(x_sampled_star, y_sampled, Normal(), IdentityLink(), wts=w_sampled)
Y_Lasso_cal = Y_HT + sum((X_t_star .- X_HT_star) .* coef(beta_wls_star))

lambda_vec = 0:0.1:5
score_vec = zeros(length(lambda_vec))
for i in 1:length(lambda_vec)
  function estfun(train_inds)
    return fit(LassoPath, (diagm(sqrt.(w_sampled)) * x_sampled)[train_inds, :], (diagm(sqrt.(w_sampled)) * y_sampled)[train_inds], standardize=false, intercept = false, α = 0, λ = [lambda_vec[i]])
  end

  function evalfun(path, test_inds)
    return sum((vcat(MLBase.predict(path, x_sampled[test_inds, :])...) .- y_sampled[test_inds]).^2 .* w_sampled[test_inds])
  end

  scores = cross_validate(
    inds -> estfun(inds),        # training function
    (c, inds) -> evalfun(c, inds),  # evaluation function
    n,              # total number of samples
    Kfold(n, 30)
  )
  score_vec[i] = mean(scores)
end
#plot(lambda_vec, score_vec)


path = fit(LassoPath, x_sampled[:,2:(p+1)], y_sampled, wts = w_sampled, standardize=false, α = 0, λ = [lambda_vec[findmin(score_vec)[2]]])
Y_Ridge = Y_HT + sum((X_t .- X_HT) .* coef(path; select=MinBIC()))



  fm = @formula(Y ~ X)
  fm1 = fit(MixedModel, fm, DataFrame(X=vcat(x_sampled...), Y=vcat(y_sampled...)), Normal())

  beta_ini = coef(fm1)

  sigma2e_ini = VarCorr(fm1).s^2
  sigma2a_ini  = VarCorr(fm1).σρ.fac.σ[1]^2

  beta_t = beta_ini
  sigma2e_t = sigma2e_ini
  sigma2a_t = sigma2a_ini

  muis, muis2 = EMalg(x_sampled, y_sampled, beta_t, sigma2e_t, sigma2a_t, M, n, verbose = verbose)

  MSE1 = mean([(muis[i] - mean(y[i]))^2 for i in 1:M])
  MSE2 = mean([(muis2[i] - mean(y[i]))^2 for i in 1:M])
  MSE3 = mean([(mean(y_sampled[i]) - mean(y[i]))^2 for i in 1:M])

  MSE_res[simnum,:] = [MSE1, MSE2, MSE3]
end

CSV.write(joinpath(dirname(@__FILE__), "MSE_res.csv"),  DataFrame(MSE_res), header=false)

#using Plots
#plot([mean(y[i]) for i in 1:M])
#plot!([mean(y_sampled[i]) for i in 1:M])
#plot!(muis)
#plot!(muis2)
