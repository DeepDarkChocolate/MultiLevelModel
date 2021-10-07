using Distributions
using QuadGK
using FastGaussQuadrature
using LinearAlgebra
using StatsFuns
using SpecialFunctions
using DistributionsAD, AdvancedVI
using StatsBase
using CSV
using DataFrames
using Random

include("ftns.jl")

if isfile(joinpath(dirname(@__FILE__), "log.txt"))
  rm(joinpath(dirname(@__FILE__), "log.txt"))
end

## Parameters defined
N = 10000
n = 500
B = 200

theta = [0.5, 1.3]
beta = [0.0, 0.4, 0.4] # beta0: to be determined

sigma2 = 0.5
phi = 4

η = vcat(beta, theta, sigma2, phi)

nodes, weights = gausshermite(20) # Number of gausshermite points
#dot( weights, @. 1.0 / (1.0 + exp(-nodes * sqrt(2)))) / sqrt(pi)
#quadgk(v -> pdf(Normal(0.0, 1.0), v) * 1.0 / (1.0 + exp(-v)), -Inf, Inf)

Res = Array{Float64}(undef, B, 18)

## Simulation start
verbose = false
#@time for simnum in 1:B
@time Threads.@threads for simnum in 1:B
  Random.seed!(simnum)
  open(joinpath(dirname(@__FILE__), "log.txt"), "a+") do io
  write(io, "$simnum\n")
  println(simnum)
  end;

## Sample generation
x = rand(N) .* 2.0
y = theta[1] .+ theta[2] .* x .+ rand(Normal(0.0, sqrt(sigma2)), N)

beta[1] = solvebeta(x, y, beta, n)
#sum(@. 1 / (1 + exp(-beta[1] - x * beta[2] - y * beta[3]))) - n

mu = @. 1 / (1 + exp(-beta[1] - x * beta[2] - y * beta[3]))
Pi = [rand(Beta(mu[i] * phi, (1 - mu[i]) * phi), 1)[1] for i in 1:N]

I = ifelse.(rand(N) .< Pi, true, false)

n_obs = sum(I)

x_sampled = x[I]
y_sampled = y[I]
w_sampled = 1.0 ./ Pi[I]


## Bayesian inference using Varational approximation
getq(θ) = TuringDiagMvNormal(θ[1:7], exp.(θ[8:14]))
q = vi(logπ4, ADVI(10, 10000), getq, vcat(beta, theta, sigma2, phi, rand(7)))

Res[simnum, 1:3] = res1 = q.m[4:6]
Res[simnum, 4:6] = res2 = q.σ[4:6]

z_α = quantile(Normal(), 1 - 0.025)
Res[simnum, 7:9] = Coverage = (@. res1 - z_α * res2 < η[4:6] < res1 + z_α * res2)



if verbose == true
@show q.m
@show q.σ
@show logπ4(vcat(beta, theta, sigma2, phi))
@show logπ4(q.m)

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

CSV.write(joinpath(dirname(@__FILE__), "res.csv"),  DataFrame(Res, collect(string.(1:18))), header=false)
