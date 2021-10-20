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

using Optim, NLSolversBase

using GLM

include("ftns.jl")

data = CSV.read(joinpath(dirname(@__FILE__), "data.csv"), DataFrame, header = false)

nodes, WEIGHTS = gausshermite(20) # Number of gausshermite points
NODES = nodes .* sqrt2
WEIGHTS = WEIGHTS ./ sqrtπ
#dot( WEIGHTS, @. 1.0 / (1.0 + exp(-nodes * sqrt(2)))) / sqrt(pi)
#quadgk(v -> pdf(Normal(0.0, 1.0), v) * 1.0 / (1.0 + exp(-v)), -Inf, Inf)


## Sample generation
x_sampled = log.(data[:, 3])
x_sampled = x_sampled .- mean(x_sampled)
y_sampled = log.(data[:, 4] .* 1000)
w_sampled = data[:, 2]

## OLS Estimation
ols = lm(@formula(y_sampled ~ x_sampled), DataFrame(x_sampled = x_sampled, y_sampled = y_sampled))
theta_ini = coef(ols)
sigma2_ini = deviance(ols) / dof_residual(ols)

## Maximum Likelihood Estimation
logπ4(vcat(theta_ini, sigma2_ini, [1.0/2200.0, 1.0/750.0, 1.0/35.0], collect(1:6), collect(1:3)); x_sampled = x_sampled, y_sampled = y_sampled, w_sampled = w_sampled,
nodes = NODES, weights = WEIGHTS)
#logπ4(vcat(zeros(2), 1.0, fill(0.5, 3), ones(6), ones(3)); x_sampled = x_sampled, y_sampled = y_sampled, w_sampled = w_sampled,
#nodes = NODES, weights = WEIGHTS)

func = TwiceDifferentiable(eta -> -logπ4(eta; x_sampled = x_sampled, y_sampled = y_sampled, w_sampled = w_sampled,
nodes = NODES, weights = WEIGHTS), vcat(theta_ini, sigma2_ini, fill(0.5, 3), ones(6), ones(3)); autodiff=:forward)

optim = optimize(func, vcat(theta_ini, sigma2_ini, [1.0/2200.0, 1.0/750.0, 1.0/35.0], ones(6), ones(3)), NelderMead(), Optim.Options(iterations = 100000))
numerical_hessian = NLSolversBase.hessian!(func,optim.minimizer)
@show optim
@show optim.minimizer
@show optim.minimum

optim = optimize(func, vcat(theta_ini, sigma2_ini, fill(0.5, 3), ones(6), ones(3)), Newton(), Optim.Options(iterations = 100000, g_tol = 1e-12))
numerical_hessian = NLSolversBase.hessian!(func,optim.minimizer)

@show optim
@show optim.minimizer
@show optim.minimum

optim = optimize(eta -> -logπ4(eta; x_sampled = x_sampled, y_sampled = y_sampled, w_sampled = w_sampled,
nodes = NODES, weights = WEIGHTS), vcat(theta_ini.-0.01, sigma2_ini-0.1, fill(0.5, 3), zeros(6), ones(3)))
@show optim.minimizer
@show optim.minimum

@show res2 = sqrt.(diag(inv(numerical_hessian)))[4:6]

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
        sum(w_sampled)^2 + 2 * sigma2hat^2 / sum(w_sampled)

@show res3 = vcat(thetahat, sigma2hat)
res4 = vcat(diag(Vhat_theta), Vhat_sigma2)
@show res4 = sqrt.(res4)


## Simulation ends
CSV.write(joinpath(dirname(@__FILE__), string("res", "_n", n, "_B", B, ".csv")),  DataFrame(Res, :auto), header=false)
