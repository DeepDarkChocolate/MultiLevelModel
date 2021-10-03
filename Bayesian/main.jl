using Distributions
using QuadGK
using FastGaussQuadrature
using LinearAlgebra
using StatsFuns
using SpecialFunctions

quadgk(v -> pdf(Normal(0.0, 1.0), v) * 1.0 / (1.0 + exp(-v)), -Inf, Inf)
nodes, weights = gausshermite(100000)
dot( weights, @. 1.0 / (1.0 + exp(-nodes * sqrt(2)))) / sqrt(pi)

N = 10000
n = 500

theta = [0.5, 1.3]
beta = [0.0, 0.4, 0.4] # beta0: to be determined

sigma2 = 0.5
phi = 4

x = rand(N) .* 2.0
y = theta[1] .+ theta[2] .* x .+ rand(Normal(0.0, sqrt(sigma2)), N)

beta[1] = solvebeta(x, y, beta, n)
#sum(@. 1 / (1 + exp(-beta[1] - x * beta[2] - y * beta[3]))) - n

mu = @. 1 / (1 + exp(-beta[1] - x * beta[2] - y * beta[3]))
Pi = [rand(Beta(mu[i] * phi, (1 - mu[i]) * phi), 1)[1] for i in 1:N]

z = rand(N)
I = ifelse.(z .< Pi, true, false)

n_obs = sum(I)

x_sampled = x[I]
y_sampled = y[I]
w_sampled = 1.0 ./ Pi[I]

#=
fi = function(beta, theta, sigma, xi, yi)
    mui = beta[1] + beta[2] * xi + beta[3] * (theta[1] + theta[2] * xi)
    sigmai = beta[2]* sigma
    denomi = dot( weights, @. logistic(mui + sqrt2 * nodes * sigmai)) / sqrtπ
    return pdf(Normal(theta[1] + theta[2] * xi, sigma), yi) * logistic(beta[1] + beta[2] * xi + beta[3] * yi) / denomi
end
=#

logfi = function(beta, theta, sigma, phi, xi, yi, wi, weights, nodes)
    muiz = beta[1] + beta[2] * xi + beta[3] * (theta[1] + theta[2] * xi)
    sigmaiz = beta[3]* sigma
    mui = logistic(beta[1] + beta[2] * xi + beta[3] * yi)
    denomi = dot( weights, @. logistic(muiz + sqrt2 * nodes * sigmaiz)) / sqrtπ
    return logpdf(Normal(theta[1] + theta[2] * xi, sigma), yi) +
     log(mui) -
     log(denomi) +
     (1 - mui) * phi * log(wi - 1) -
     phi * log(wi) -
     log(SpecialFunctions.beta((1 - mui) * phi, mui * phi + 1))
end

logfi([-10, beta[2], beta[3]], theta, sqrt(sigma2), phi, x_sampled[1], y_sampled[1], w_sampled[1], weights, nodes)
logfi([beta[1], beta[2], beta[3]], theta, sqrt(sigma2), phi, x_sampled[1], y_sampled[1], w_sampled[1], weights, nodes)
#=
for i in 1:n_obs
    logfi_tmp = logfi([-10, beta[2], beta[3]], theta, sqrt(sigma2), phi, x_sampled[i], y_sampled[i], w_sampled[i], weights, nodes)
    @show logfi_tmp
end

@code_warntype  logfi(beta, theta, sqrt(sigma2), phi, x_sampled[1], y_sampled[1], w_sampled[1], weights, nodes)
=#

logf = function(beta, theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes; logf0 = 0.0)
    return (sum([logfi(beta, theta, sqrt(sigma2), phi, x_sampled[i], y_sampled[i], w_sampled[i], weights, nodes)
    for i in 1:n_obs]) - logf0)::Float64
end

f = function(beta, theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes; logf0 = 0.0)
    return exp(sum([logfi(beta, theta, sqrt(sigma2), phi, x_sampled[i], y_sampled[i], w_sampled[i], weights, nodes)
    for i in 1:n_obs]) - logf0)::Float64
end;
logf0 = logf(beta, theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes)
f(beta, theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0)

#@code_warntype

using Plots
x = collect(range(beta[1] - 4, beta[1] + 4, length = 30))
y = [logf([i, beta[2], beta[3]], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(beta[2] - 4, beta[2]+4, length = 20))
y = [logf([beta[1], i, beta[3]], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(beta[3] - 4, beta[3]+4, length = 20))
y = [logf([beta[1], beta[2], i], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(theta[1] - 4, theta[1]+4, length = 20))
y = [logf(beta, [i, theta[2]], sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(theta[2] - 4, theta[2]+4, length = 20))
y = [logf(beta, [theta[1], i], sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(sigma2 - 0.25, sigma2+0.25, length = 20))
y = [logf(beta, theta, i, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(phi - 1, phi+1, length = 20))
y = [logf(beta, theta, sigma2, i, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)
