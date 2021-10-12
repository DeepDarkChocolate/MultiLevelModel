n = 200
m = 9
N = 5000
Mi = 2000

beta0 = 0.0
beta1 = 1.0
sigma2a = 1.0

mua = -0.5
theta = [beta1, mua, sigma2a]
B = 10 # 497
K = 5000 # Number of MC samples generated from approximated Normal distr.

w_h = 1 / 3

eps_theta = 10^(-3)

#interval = 6.0
#rtol = 1e-5
#K2 = 10

interval = 10.0
rtol = 1e-5
K2 = 25

ICS = true # ICS? or Non-ICS

using Distributions
using Random
using LinearAlgebra
using NamedArrays
using StatsBase
using MixedModels
using DataFrames
using QuadGK
using Optim
using Roots
#using RCall
using Dates
using CSV
using Plots
plot()
include("Proposed.jl")

for i in 1:3
Random.seed!(i)
a = rand(Normal(mua,sqrt(sigma2a)),N)

if ICS == true
  a_tilde = exp.(2.5 .+ a) ./ (1 .+ exp.(2.5 .+ a))
else
  a2 = rand(Normal(0,sqrt(1)),N)
  a_tilde = exp.(2.5 .+ a2) ./ (1 .+ exp.(2.5 .+ a2))
end

M = round.(Int, Mi .* a_tilde)

#M = [Mi for i in 1:N] # SRS case

x = [rand(Normal(0.0,1.0),i) for i in M]

p = [@. 1 - 1 / (1 + exp(beta0 + beta1 * x[i] + a[i])) for i in 1:length(M)]
z = [rand(i) for i in M]

y = [ifelse.(z[i] .< p[i], 1, 0) for i in 1:length(M)]
#y = [vcat((@. rand(Binomial(1, p[i]), 1)...)) for i in 1:length(M)]

pi1 = n .* M ./ sum(M)

SampleIdx1 = StatsBase.sample(1:N, Weights(Float64.(M)), n, replace=false)
#SampleIdx1 = StatsBase.sample(1:N, n, replace=false) # SRS Sampling

m1 = Int(m * w_h)
m2 = Int(m - m * w_h)

SampleIdx2 = [vcat(StatsBase.sample( (1:length(z[element]))[z[element] .> 0.5], m1, replace=false),
                StatsBase.sample( (1:length(z[element]))[z[element] .<= 0.5], m2, replace=false))
                for (i, element) in enumerate(SampleIdx1)] # Stratified Sampling

x_sampled = [x[element][SampleIdx2[i]] for (i, element) in enumerate(SampleIdx1)]
y_sampled = [y[element][SampleIdx2[i]] for (i, element) in enumerate(SampleIdx1)]
z_sampled = [z[element][SampleIdx2[i]] for (i, element) in enumerate(SampleIdx1)]

pi1_sampled = pi1[SampleIdx1]
M1 = [sum(z[element] .> 0.5) for element in SampleIdx1]
M2 = [sum(z[element] .<= 0.5) for element in SampleIdx1]

pi2_sampled = [ifelse.(z_sampled[i] .> 0.5, m1 / M1[i], m2 / M2[i])
                for (i, element) in enumerate(SampleIdx1)]

pi2mat_sampled = [
let pi2mat_sampled_tmp = ones(m,m)
pi2mat_sampled_tmp1 = ones(m1,m1) .* m1 * (m1 - 1) / M1[i] / (M1[i] - 1)
pi2mat_sampled_tmp1[diagind(pi2mat_sampled_tmp1)] .= m1 / M1[i]
@. pi2mat_sampled_tmp[z_sampled[1] .> 0.5, z_sampled[1] .> 0.5] = pi2mat_sampled_tmp1

pi2mat_sampled_tmp2 = ones(m2,m2) .* m2 * (m2 - 1) / M2[i] / (M2[i] - 1)
pi2mat_sampled_tmp2[diagind(pi2mat_sampled_tmp2)] .= m2 / M2[i]
@. pi2mat_sampled_tmp[z_sampled[1] .<= 0.5, z_sampled[1] .<= 0.5] = pi2mat_sampled_tmp2

@. pi2mat_sampled_tmp[z_sampled[1] .> 0.5, z_sampled[1] .<= 0.5] = m1 * m2 / M1[i] / M2[i]
@. pi2mat_sampled_tmp[z_sampled[1] .<= 0.5, z_sampled[1] .> 0.5] = m1 * m2 / M1[i] / M2[i]
pi2mat_sampled_tmp
end for i in 1:n]

w1_sampled = @. 1 / pi1_sampled
w2_sampled = [@. 1 / pi2_sampled[i] for i in 1:n]

fm = @formula(Y ~ X + (1|fac))
fm1 = fit(MixedModel, fm, DataFrame(X=vcat(x_sampled...), Y=vcat(y_sampled...), fac = string.([SampleIdx1[i] for i in 1:n for j in 1:length(SampleIdx2[i])])), Binomial())

beta_ini = coef(fm1)
mua_ini = beta_ini[1]
beta_ini[1] = 0.0


sigma2a_ini  = VarCorr(fm1).σρ.fac.σ[1]^2
#sigma2a_ini  = sigma2a + rand(1)[1] - 0.5

ahat = solveahat(x_sampled, y_sampled, w2_sampled, [beta0, beta1], mua, sigma2a)
display(plot!(ahat, M[SampleIdx1], seriestype = :scatter, title = string("ICS = ", ICS, "; Logistic"),
legend=:bottomright))
end
