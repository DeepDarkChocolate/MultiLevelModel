using Distributions
using QuadGK
using FastGaussQuadrature
using LinearAlgebra
using StatsFuns
using SpecialFunctions
using DistributionsAD, AdvancedVI
using StatsBase

include("ftns.jl")

quadgk(v -> pdf(Normal(0.0, 1.0), v) * 1.0 / (1.0 + exp(-v)), -Inf, Inf)
nodes, weights = gausshermite(20)
dot( weights, @. 1.0 / (1.0 + exp(-nodes * sqrt(2)))) / sqrt(pi)

N = 10000
n = 500

theta = [0.5, 1.3]
beta = [0.0, 0.4, 0.4] # beta0: to be determined

sigma2 = 0.5
phi = 4

η = vcat(beta, theta, sigma2, phi)

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

getq(θ) = TuringDiagMvNormal(θ[1:7], exp.(θ[8:14]))
#q = vi(logπ3, advi, getq, vcat(randn(5), rand(9)))
q = vi(logπ4, ADVI(10, 10000), getq, vcat(beta, theta, sigma2, phi, rand(7)))

@show q.m
@show q.σ
@show logπ4(vcat(beta, theta, sigma2, phi))
@show logπ4(q.m)

@show res1 = q.m[4:6]
@show res2 = q.σ[4:6]

z_α = quantile(Normal(), 1 - 0.025)
Coverage = (@. res1 - z_α * res2 < η[4:6] < res1 + z_α * res2)
@show Coverage
@show res1 - z_α * res2
@show η[4:6]
@show res1 + z_α * res2

println("---------------------------------------------")

Xmat_sampled = hcat(fill(1, length(x_sampled)),x_sampled)
thetahat = (transpose(Xmat_sampled) * diagm(w_sampled) * Xmat_sampled) \ (transpose(Xmat_sampled) * diagm(w_sampled) * y_sampled)
sigma2hat = sum(@. w_sampled * (y_sampled - thetahat[1] - thetahat[2] * x_sampled)^2)  / sum(@. w_sampled)

Hmat = transpose(Xmat_sampled) * diagm(w_sampled) * Xmat_sampled
#Svec  = zeros(2)
#Svec[1] = mean((@. y_sampled - thetahat[1] - thetahat[2] * x_sampled), StatsBase.weights(w_sampled))
#Svec[2] = mean((@. (y_sampled - thetahat[1] - thetahat[2] * x_sampled) * x_sampled), StatsBase.weights(w_sampled))
Smat = hcat((@. y_sampled - thetahat[1] - thetahat[2] * x_sampled),
        (@. (y_sampled - thetahat[1] - thetahat[2] * x_sampled) * x_sampled))
VSmat = transpose(Smat) * diagm(@. w_sampled * (w_sampled - 1)) * Smat
iHmat = inv(Hmat)
Vhat_theta = iHmat * VSmat * iHmat + iHmat * sigma2hat
Vhat_sigma2 = sum(@. w_sampled * (w_sampled - 1) *
        ((y_sampled - thetahat[1] - x_sampled * thetahat[2])^2 - sigma2hat)^2) /
        sum(w_sampled)^2 + 2 * sigma2hat^2 / N

res3 = vcat(thetahat, sigma2hat)
res4 = vcat(diag(Vhat_theta), Vhat_sigma2)
res4 = sqrt.(res4)
@show res3
@show res4

Coverage = (@. res3 - z_α * res4 < η[4:6] < res3 + z_α * res4)
@show Coverage
@show lower = res3 - z_α * res4
@show η[4:6]
@show upper = res3 + z_α * res4

#=
using Optim
minuslogπ4(eta) = -logπ4(eta)
optim = optimize(minuslogπ4, vcat(beta, theta, sigma2, phi))
@show optim.minimum
@show optim.minimizer
=#
