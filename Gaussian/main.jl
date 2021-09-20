# Normal; Cluster sampling
#126
n = 20
#m = 20
N = 1000
M = 50

beta0 = 0.5
beta1 = 2.0
sigma2e = 2.0
sigma2a = 1.0
mua = 0.0

theta = [beta0, beta1, sigma2e, sigma2a]
B = 500 # 497

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

  a = rand(Normal(mua,sqrt(sigma2a)),M)

  e = [rand(Normal(0.0,sqrt(sigma2e)),N) for i in 1:M]

  #a_tilde = exp.(2.5 .+ a) ./ (1 .+ exp.(2.5 .+ a))
  #M = round.(Int, Mi .* a_tilde)

  x = [randn(N) for i in 1:M]

  y = [@. beta0 + beta1 * x[i] + a[i] + e[i] for i in 1:M]

  #y = [vcat((@. rand(Binomial(1, p[i]), 1)...)) for i in 1:length(M)]

  SampleIdx = [StatsBase.sample(1:N, n, replace=false) for i in 1:M] # SRS Sampling

  x_sampled = [x[i][SampleIdx[i]] for i in 1:M]
  y_sampled = [y[i][SampleIdx[i]] for i in 1:M]

  # SRS case

  fm = @formula(Y ~ X + (1|fac))
  fm1 = fit(MixedModel, fm, DataFrame(X=vcat(x_sampled...), Y=vcat(y_sampled...), fac = string.([i for i in 1:M for j in 1:n])), Normal())

  beta_ini = coef(fm1)

  sigma2e_ini = VarCorr(fm1).s^2
  sigma2a_ini  = VarCorr(fm1).σρ.fac.σ[1]^2
  #sigma2a_ini  = sigma2a + rand(1)[1] - 0.5

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
