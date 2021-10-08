logfi = function(beta, theta, sigma, phi, xi, yi, wi, weights, nodes)
    muiz = beta[1] + beta[2] * xi + beta[3] * (theta[1] + theta[2] * xi)
    sigmaiz = beta[3]* sigma
    mui = logistic(beta[1] + beta[2] * xi + beta[3] * yi)
    denomi = dot( weights, @. logistic(muiz + sqrt2 * nodes * sigmaiz)) / sqrtπ
    return logpdf(Normal(theta[1] + theta[2] * xi, sigma), yi) +
     log(mui) -
     log(denomi) +
     logpdf(BetaPrime((1 - mui) * phi, mui * phi + 1), wi - 1)
end

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

using Plots
x = collect(range(beta[1] - 4, beta[1] + 4, length = 30))
y = [logf([i, beta[2], beta[3]], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(beta[1] - 1, beta[1] + 1, length = 50))
@time y = [f([i, beta[2], beta[3]], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(beta[2] - 4, beta[2]+4, length = 20))
y = [logf([beta[1], i, beta[3]], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(beta[3] - 4, beta[3]+4, length = 20))
y = [logf([beta[1], beta[2], i], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(theta[1] - 4, theta[1]+4, length = 1000))
y = [logf(beta, [i, theta[2]], sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(theta[1] - 0.5, theta[1]+0.5, length = 1000))
y = [logπ4(vcat(beta, [i, theta[2]], sigma2, phi); x_sampled = x_sampled, y_sampled = y_sampled, w_sampled = w_sampled,
nodes = nodes, weights = weights) for i in x]
plot(x, y)

x = collect(range(q.m[5] - 10, q.m[5]+10, length = 100))
y = [logπ4(vcat(q.m[1:4], i, q.m[6:7]); x_sampled = x_sampled, y_sampled = y_sampled, w_sampled = w_sampled,
nodes = nodes, weights = weights) for i in x]
plot!(x, y)

y0 = [logπ4(vcat(q.m[1:4], i, q.m[6:7]); x_sampled = x_sampled, y_sampled = y_sampled, w_sampled = w_sampled,
nodes = nodes, weights = weights) for i in x]
y = [exp.(logπ4(vcat(q.m[1:4], i, q.m[6:7]); x_sampled = x_sampled, y_sampled = y_sampled, w_sampled = w_sampled,
nodes = nodes, weights = weights) .-findmax(y0)[1]) for i in x]

plot!(x, y)


x = collect(range(q.m[5] - 0.1, q.m[5]+0.1, length = 100))
y0 = [logpdf(q, vcat(q.m[1:4], i, q.m[6:7])) for i in x]
y = [exp.(logpdf(q, vcat(q.m[1:4], i, q.m[6:7])) .- findmax(y0)[1]) for i in x]
plot(x, y)
rand(q, 10000)[5,:]

sqrt(var(rand(q, 10000)[5,:]))
tmpsample = rand(q, 10000)
histogram!(tmpsample[5,:])

x[findmax(y)[2]]

x[findmax(y)[2]]

x = collect(range(theta[2] - 4, theta[2]+4, length = 50))
y = [logf(beta, [theta[1], i], sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)
x[findmax(y)[2]]

x = collect(range(sigma2 - 0.25, sigma2+0.25, length = 20))
y = [logf(beta, theta, i, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(phi - 1, phi+1, length = 20))
y = [logf(beta, theta, sigma2, i, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) for i in x]
plot(x, y)

x = collect(range(phi - 1, phi+1, length = 20))
y = [logπ4(vcat(beta, theta, sigma2, i)) for i in x]
plot(x, y)

x = collect(range(phi - 1, phi+1, length = 20))
y = [logπ4(vcat(q.m[1:6], i)) for i in x]
plot(x, y)

###############################################

logfi2 = function(beta, theta, sigma, phi, xi, yi, wi)
    mui = @. logistic(beta[1] + beta[2] * xi + beta[3] * yi)
    mui = @. ifelse(mui > 0, mui, eps(Float64))
    mui = @. ifelse(mui < 1, mui, 1 - eps(Float64))
    return @. logpdf(Normal(theta[1] + theta[2] * xi, sigma), yi) +
      logpdf(BetaPrime((1 - mui) * phi, mui * phi + 1), wi - 1)
     #(1 - mui) * phi * log(wi - 1) -
     #phi * log(wi) -
     #log(SpecialFunctions.beta((1 - mui) * phi, mui * phi + 1))
end


logf2 = function(beta, theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs; logf02 = 0.0)
    return (sum(logpdf.(Normal(0, sqrt(sigma2)), @.y_sampled - theta[1] - x_sampled * theta[2])) - logf02)::Float64
end

logf2 = function(beta, theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs; logf02 = 0.0)
    return (sum([logfi2(beta, theta, sqrt(sigma2), phi, x_sampled[i], y_sampled[i], w_sampled[i])
    for i in 1:n_obs]) - logf02)::Float64
end

f2 = function(beta, theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs; logf02 = 0.0)
    return exp(sum([logfi2(beta, theta, sqrt(sigma2), phi, x_sampled[i], y_sampled[i], w_sampled[i])
    for i in 1:n_obs]) - logf02)::Float64
end
logf02 = logf2(beta, theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs)
f2(beta, theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02)

##########################################


using Plots
x = collect(range(beta[1] - 4, beta[1] + 4, length = 30))
y = [logf([i, beta[2], beta[3]], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) -
      logf2([i, beta[2], beta[3]], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)

x = collect(range(beta[3] - 4, beta[3] + 4, length = 30))
y = [logf([beta[1], beta[2], i], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) -
      logf2([beta[1], beta[2], i], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)

x = collect(range(theta[1] - 10, theta[1]+4, length = 20))
y = [logf(beta, [i, theta[2]], sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) -
      logf2(beta, [i, theta[2]], sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)

x = collect(range(sigma2 - 0.4, sigma2+0.4, length = 20))
y = [logf(beta, theta, i, phi, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) -
      logf2(beta, theta, i, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)

x = collect(range(phi - 2, phi+2, length = 20))
y = [logf(beta, theta, sigma2, i, x_sampled, y_sampled, w_sampled, n_obs, weights, nodes, logf0 = logf0) -
      logf2(beta, theta, sigma2, i, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)


#@code_warntype

x = collect(range(beta[1] - 4, beta[1] + 4, length = 30))
y = [logf2([i, beta[2], beta[3]], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)

x = collect(range(beta[1] - 0.5, beta[1] + 0.5, length = 50))
y = [f2([i, beta[2], beta[3]], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)

x = collect(range(beta[2] - 4, beta[2]+4, length = 50))
y = [logf2([beta[1], i, beta[3]], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)

x = collect(range(beta[2] - 0.3, beta[2]+0.3, length = 50))
y = [f2([beta[1], i, beta[3]], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)

x = collect(range(beta[3] - 4, beta[3]+4, length = 50))
y = [logf2([beta[1], beta[2], i], theta, sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)

x = collect(range(theta[1] - 4, theta[1]+4, length = 20))
y = [logf2(beta, [i, theta[2]], sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)

x = collect(range(theta[1] - 0.2, theta[1]+0.2, length = 50))
y = [f2(beta, [i, theta[2]], sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y / y[findmax(y)[2]])



x = collect(range(theta[2] - 4, theta[2]+4, length = 20))
y = [logf2(beta, [theta[1], i], sigma2, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)

x = collect(range(sigma2 - 0.25, sigma2+0.25, length = 20))
y = [logf2(beta, theta, i, phi, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)

x = collect(range(phi - 1, phi+1, length = 20))
y = [logf2(beta, theta, sigma2, i, x_sampled, y_sampled, w_sampled, n_obs, logf02 = logf02) for i in x]
plot(x, y)
