
logπ4 = function(eta; x_sampled = x_sampled::Vector{Float64},
    y_sampled = y_sampled::Vector{Float64}, w_sampled = w_sampled::Vector{Float64},
    nodes = nodes::Vector{Float64}, weights = weights::Vector{Float64})
    #@show eta
    theta0 = eta[1]
    theta1 = eta[2]
    sigma2 = eta[3]
    mu = [eta[4], eta[5], eta[6]]
    alpha = [zeros(3) eta[7:9] eta[10:12]]
    #alpha = [[0.0 eta[7] eta[10]]; [0.0 eta[8] eta[11]]; [0.0 eta[9] eta[12]]]
    phi = [eta[13], eta[14], eta[15]]

    n = size(x_sampled)[1]
    K = size(nodes)[1]

    #return ((sigma2 > 0.0) & all(phi .> 0.0) & all(0.0 .< mu .< 1.0)) ? (
    return ((sigma2 > 0.0) & all(phi .> 0.0) & all((1 / 10000) < mu[1] < (1 / 1000) <  mu[2] < (1 / 300) < mu[3]  < 1.0)) ? (
    sigma = sqrt(sigma2);
    PI = (exp.(hcat(ones(n), x_sampled, y_sampled) * alpha) * mu)./
    (exp.(hcat(ones(n), x_sampled, y_sampled) * alpha) * ones(3));
    #@show PI;

    #PI_tmp = [sum([mu[h] * exp(alpha[1, h] + alpha[2, h] * x_sampled[i] + alpha[3, h] * y_sampled[i]) for h in 1:3]) / sum([exp(alpha[1, h] + alpha[2, h] * x_sampled[i] + alpha[3, h] * y_sampled[i]) for h in 1:3]) for i in 1:n];
    #@show all(PI .== PI_tmp);

    denom = [sum((exp.(hcat(ones(K), fill(x_sampled[i], K),
    (@. nodes * sigma + theta0 + theta1 * x_sampled[i])) * alpha) * mu) ./
    (exp.(hcat(ones(K), fill(x_sampled[i], K),
    (@. nodes * sigma + theta0 + theta1 * x_sampled[i])) * alpha) * ones(3)) .* weights)
    for i in 1:n];

    fw = [StatsBase.mean((@. pdf(BetaPrime((1 - mu) * phi, mu * phi + 1), w_sampled[i] - 1.0)),
    StatsBase.weights(vcat(exp.(hcat(1.0, x_sampled[i], y_sampled[i]) * alpha)...) .* mu))
    for i in 1:n];
    #fw_tmp = [sum([pdf(BetaPrime((1 - mu[h]) * phi[h], mu[h] * phi[h] + 1), w_sampled[i] - 1.0) * mu[h] * exp(alpha[1, h] + alpha[2, h] * x_sampled[i] + alpha[3, h] * y_sampled[i]) for h in 1:3]) / sum([mu[h] * exp(alpha[1, h] + alpha[2, h] * x_sampled[i] + alpha[3, h] * y_sampled[i]) for h in 1:3]) for i in 1:n];
    #@show norm(fw .- fw_tmp);

    sum(@. logpdf(Normal(0, sigma), @. y_sampled - theta0 - x_sampled * theta1) +
    log(fw) - log(denom) + log(PI))) : -Inf
end

logπ5 = function(eta; x_sampled = x_sampled::Vector{Float64},
    y_sampled = y_sampled::Vector{Float64}, w_sampled = w_sampled::Vector{Float64},
    nodes = nodes::Vector{Float64}, weights = weights::Vector{Float64})
    #@show eta
    theta0 = eta[1]
    theta1 = eta[2]
    sigma2 = eta[3]
    mu = [eta[4], eta[4], eta[4]]
    alpha = [zeros(3) eta[5:7] eta[8:10]]
    #alpha = [[0.0 eta[7] eta[10]]; [0.0 eta[8] eta[11]]; [0.0 eta[9] eta[12]]]
    phi = [eta[11], eta[11], eta[11]]

    n = size(x_sampled)[1]
    K = size(nodes)[1]

    #return ((sigma2 > 0.0) & all(phi .> 0.0) & all(0.0 .< mu .< 1.0)) ? (
    return ((sigma2 > 0.0) & all(phi .> 0.0) & all(0.0 < eta[4]  < 1.0)) ? (
    sigma = sqrt(sigma2);
    PI = (exp.(hcat(ones(n), x_sampled, y_sampled) * alpha) * ones(3))./
    (exp.(hcat(ones(n), x_sampled, y_sampled) * alpha) * ones(3));

    #PI_tmp = [sum([mu[h] * exp(alpha[1, h] + alpha[2, h] * x_sampled[i] + alpha[3, h] * y_sampled[i]) for h in 1:3]) / sum([exp(alpha[1, h] + alpha[2, h] * x_sampled[i] + alpha[3, h] * y_sampled[i]) for h in 1:3]) for i in 1:n];
    #@show all(PI .== PI_tmp);

    denom = [sum((exp.(hcat(ones(K), fill(x_sampled[i], K),
    (@. nodes * sigma + theta0 + theta1 * x_sampled[i])) * alpha) * ones(3)) ./
    (exp.(hcat(ones(K), fill(x_sampled[i], K),
    (@. nodes * sigma + theta0 + theta1 * x_sampled[i])) * alpha) * ones(3)) .* weights)
    for i in 1:n];

    fw = [StatsBase.mean((@. pdf(BetaPrime((1 - eta[4]) * eta[11], eta[4] * eta[11] + 1), w_sampled[i] - 1.0)))
    for i in 1:n];
    #fw_tmp = [sum([pdf(BetaPrime((1 - mu[h]) * phi[h], mu[h] * phi[h] + 1), w_sampled[i] - 1.0) * mu[h] * exp(alpha[1, h] + alpha[2, h] * x_sampled[i] + alpha[3, h] * y_sampled[i]) for h in 1:3]) / sum([mu[h] * exp(alpha[1, h] + alpha[2, h] * x_sampled[i] + alpha[3, h] * y_sampled[i]) for h in 1:3]) for i in 1:n];
    #@show norm(fw .- fw_tmp);

    sum(@. logpdf(Normal(0, sigma), @. y_sampled - theta0 - x_sampled * theta1) +
    log(fw) - log(denom) + log(PI))) : -Inf
end
