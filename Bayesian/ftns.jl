function solvebeta(x, y, beta, n)
    beta_t1 = 0.0
    beta_t2 = 0.0
    cnt = 0
    while true
        cnt = cnt + 1
        f = sum(@. 1 / (1 + exp(-beta_t1 - x * beta[2] - y * beta[3]))) - n
        df = sum(@. 1 / (1 + exp(-beta_t1 - x * beta[2] - y * beta[3])) / (1 + exp(beta_t1 + x * beta[2] + y * beta[3])))
        beta_t2 = beta_t1 - f / df
        if abs(beta_t2 - beta_t1) < 10^(-100)
            break
        elseif cnt > 500
            #@warn("beta0 NOT convergent")
            break
        else
            beta_t1 = beta_t2
        end
    end
    return beta_t2
end

logπ4 = function(eta; x_sampled = x_sampled::Vector{Float64},
    y_sampled = y_sampled::Vector{Float64}, w_sampled = w_sampled::Vector{Float64},
    nodes = nodes::Vector{Float64}, weights = weights::Vector{Float64})
    return (eta[6] > 0.0) & (eta[7] > 0.0) ? (mui = @. logistic(eta[1] + eta[2] * x_sampled + eta[3] * y_sampled);
    alpha2 = @. (1 - mui) * eta[7];
    beta2 = @. mui * eta[7] + 1;
    muiz = @. eta[1] + eta[2] * x_sampled + eta[3] * (eta[4] + eta[5] * x_sampled);
    sigmaiz = eta[3]* sqrt(eta[6]);
    denomi = (logistic.(muiz .+ sqrt2 .* transpose(nodes) .* sigmaiz) * weights) ./ sqrtπ;
    sum(logpdf.(Normal(0, sqrt(eta[6])), @. y_sampled - eta[4] - x_sampled * eta[5]) .+
    @. ifelse((alpha2 > 0.0) & (beta2 > 0.0), logpdf(BetaPrime(alpha2, beta2), w_sampled .- 1.0), -Inf) -
    log(denomi) + log(mui)) +
    logpdf(MvNormal([-4, 0.4, 0.4, 0.5, 1.3], diagm(ones(5))), eta[1:5]) +
    logpdf(InverseGamma(3, 1), eta[6]) + logpdf(InverseGamma(3, 8), eta[7])) : -Inf
end
