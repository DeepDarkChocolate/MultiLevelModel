using Distributions
using QuadGK
using FastGaussQuadrature
using LinearAlgebra
using StatsFuns
#using SpecialFunctions
using DistributionsAD, AdvancedVI
using StatsBase
using CSV
using DataFrames
using Random

include("ftns.jl")

if isfile(joinpath(dirname(@__FILE__), "log.txt"))
  rm(joinpath(dirname(@__FILE__), "log.txt")) # If there exists "log.txt", remove it.
end

## Parameters defined
N = 10000
n = 100
B = 100

theta = [0.5, 1.3] # [θ_0, θ_1]
beta = [0.0, 0.4, 0.4] # beta0: to be determined; [β_0, β_1, β_2]

sigma2 = 0.5 # σ^2
phi = 40000 # ϕ

η = vcat(beta, theta, sigma2, phi)

nodes, weights = gausshermite(20) # Number of gausshermite points
#dot( weights, @. 1.0 / (1.0 + exp(-nodes * sqrt(2)))) / sqrt(pi)
#quadgk(v -> pdf(Normal(0.0, 1.0), v) * 1.0 / (1.0 + exp(-v)), -Inf, Inf)


Res = Array{Float64}(undef, B, 18)

## Simulation start
verbose = true
@time for simnum in 1:B # Serial programming
#@time Threads.@threads for simnum in 1:B # Parallel programming
  Random.seed!(simnum)
  open(joinpath(dirname(@__FILE__), "log.txt"), "a+") do io
  write(io, "$simnum\n")
  println(simnum)
  end;

## Sample generation
simnum = 3
Random.seed!(simnum)
x = rand(N) .* 2.0 # x_i ∼ Unif(0, 2)
y = theta[1] .+ theta[2] .* x .+ rand(Normal(0.0, sqrt(sigma2)), N) # y_i ∼ θ_0 + θ_1 * x_i + ε_i

beta[1] = solvebeta(x, y, beta, n) # Find β_0 such that ∑μ_i = n
#sum(@. 1 / (1 + exp(-beta[1] - x * beta[2] - y * beta[3]))) - n # check if it's zero

mu = @. 1 / (1 + exp(-beta[1] - x * beta[2] - y * beta[3])) # logit(μ_i) = β_0 + β_1 x_i + β_2 y_i
Pi = [rand(Beta(mu[i] * phi, (1 - mu[i]) * phi), 1)[1] for i in 1:N] # π_i ∼ Beta(μ_i ϕ, (1 - μ_i) ϕ)

I = ifelse.(rand(N) .< Pi, true, false) # Index set of the sample: I_i ∼ Bernoulli(π_i)

n_obs = sum(I) # Sample size

x_sampled = x[I]
y_sampled = y[I]
w_sampled = 1.0 ./ Pi[I]


## Bayesian inference using Varational approximation
getq(θ) = TuringDiagMvNormal(θ[1:7], exp.(θ[8:14]))
q = vi(eta -> logπ4(eta; x_sampled = x_sampled, y_sampled = y_sampled, w_sampled = w_sampled,
nodes = nodes, weights = weights),
ADVI(10, 10000), getq, vcat(beta, theta, sigma2, phi, ones(7)))

Res[simnum, 1:3] = res1 = q.m[4:6]
Res[simnum, 4:6] = res2 = q.σ[4:6]

z_α = quantile(Normal(), 1 - 0.025)
Res[simnum, 7:9] = Coverage = (@. res1 - z_α * res2 < η[4:6] < res1 + z_α * res2)

if verbose == true
@show q.m
@show q.σ
@show logπ4(vcat(beta, theta, sigma2, phi); x_sampled = x_sampled, y_sampled = y_sampled, w_sampled = w_sampled,
nodes = nodes, weights = weights)
@show logπ4(q.m; x_sampled = x_sampled, y_sampled = y_sampled, w_sampled = w_sampled,
nodes = nodes, weights = weights)

@show res1
@show res2

@show Coverage
@show res1 - z_α * res2
@show η[4:6]
@show res1 + z_α * res2
println("---------------------------------------------")
end

## Design-based Estimation

Xmat_sampled = hcat(fill(1, length(x_sampled)),x_sampled)
thetahat = (transpose(Xmat_sampled) * diagm(w_sampled) * Xmat_sampled) \ (transpose(Xmat_sampled) * diagm(w_sampled) * y_sampled)
sigma2hat = sum(@. w_sampled * (y_sampled - thetahat[1] - thetahat[2] * x_sampled)^2)  / sum(@. w_sampled)

Hmat = transpose(Xmat_sampled) * diagm(w_sampled) * Xmat_sampled
Smat = hcat((@. y_sampled - thetahat[1] - thetahat[2] * x_sampled),
        (@. (y_sampled - thetahat[1] - thetahat[2] * x_sampled) * x_sampled))
VSmat = transpose(Smat) * diagm(@. w_sampled * (w_sampled - 1)) * Smat
iHmat = inv(Hmat)
Vhat_theta = iHmat * VSmat * iHmat + iHmat * sigma2hat
Vhat_sigma2 = sum(@. w_sampled * (w_sampled - 1) *
        ((y_sampled - thetahat[1] - x_sampled * thetahat[2])^2 - sigma2hat)^2) /
        sum(w_sampled)^2 + 2 * sigma2hat^2 / N

Res[simnum, 10:12]  = res3 = vcat(thetahat, sigma2hat)
res4 = vcat(diag(Vhat_theta), Vhat_sigma2)
Res[simnum, 13:15] = res4 = sqrt.(res4)

Res[simnum, 16:18] = Coverage2 = (@. res3 - z_α * res4 < η[4:6] < res3 + z_α * res4)

if verbose == true
@show res3
@show res4

@show Coverage2
@show lower = res3 - z_α * res4
@show η[4:6]
@show upper = res3 + z_α * res4
end

end
## Simulation ends
CSV.write(joinpath(dirname(@__FILE__), string("res", "_n", n, "_B", B, ".csv")),  DataFrame(Res, collect(string.(1:18))), header=false)
