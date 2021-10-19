
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

    return ((sigma2 > 0.0) & all(phi .> 0.0) & all(0.0 .< mu .< 1.0)) ? (
    sigma = sqrt(sigma2);
    PI = (exp.(hcat(ones(n), x_sampled, y_sampled) * alpha) * mu)./
    (exp.(hcat(ones(n), x_sampled, y_sampled) * alpha) * ones(3));

    denom = [sum((exp.(hcat(ones(K), fill(x_sampled[i], K), (@. nodes * sigma + theta0 + theta1 * x_sampled[i])) * alpha) * mu) ./
    (exp.(hcat(ones(K), fill(x_sampled[i], K), (@. nodes * sigma + theta0 + theta1 * x_sampled[i])) * alpha) * ones(3)) .* weights)
    for i in 1:n];

    fw = [StatsBase.mean((@. pdf(BetaPrime((1 - mu) * phi, mu * phi + 1), w_sampled[i] - 1.0)), StatsBase.weights(vcat(exp.(hcat(1.0, x_sampled[i], y_sampled[i]) * alpha)...) .* mu))
    for i in 1:n];
    sum(@. logpdf(Normal(0, sigma), @. y_sampled - theta0 - x_sampled * theta1) +
    log(fw) - log(denom) + log(PI))) : -Inf
end
