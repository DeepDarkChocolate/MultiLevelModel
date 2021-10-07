using DistributionsAD, AdvancedVI
using Distributions

advi = ADVI(10, 10_000)

logπ2 = function(beta)
    mui = @. logistic(beta[1] + beta[2] * x_sampled + beta[3] * y_sampled)
    mui = @. ifelse(mui > eps(Float64), mui, eps(Float64))
    mui = @. ifelse(mui < 1 - eps(Float64), mui, 1 - eps(Float64))
    return sum(logpdf.(Normal(0, sqrt(sigma2)), @.y_sampled - theta[1] - x_sampled * theta[2]) .+
    @. logpdf(BetaPrime((1 - mui) * phi, mui * phi + 1), w_sampled .- 1.0))
end

logπ2(randn(3))
logπ2(beta)
logπ2(q.m)

getq(θ) = TuringDiagMvNormal(θ[1:3], exp.(θ[4:6]))
q = vi(logπ2, advi, getq, randn(6))
q.m
q.σ

getq(θ) = TuringDenseMvNormal(θ[1:3], exp.([θ[4] θ[5] θ[6]; θ[5] θ[7] θ[8]; θ[6] θ[8] θ[9]]))
q = vi(logπ2, advi, getq, vcat(randn(3), [0., -100., -100.,  0., -100., 0.]))
q.m
q.C


logπ3 = function(eta)
    mui = @. logistic(eta[1] + eta[2] * x_sampled + eta[3] * y_sampled)
    alpha2 = @. (1 - mui) * eta[7]
    alpha2 = @. ifelse(alpha2 > eps(Float64), alpha2, eps(Float64))
    beta2 = @. mui * eta[7] + 1
    beta2 = @. ifelse(beta2 > eps(Float64), beta2, eps(Float64))
    sigma2 = @. ifelse(eta[6] > eps(Float64), eta[6], eps(Float64))
    return sum(logpdf.(Normal(0, sqrt(sigma2)), @.y_sampled - eta[4] - x_sampled * eta[5]) .+
    @. logpdf(BetaPrime(alpha2, beta2), w_sampled .- 1.0))
end

#=
logπ3 = function(eta)
    mui = @. logistic(eta[1] + eta[2] * x_sampled + eta[3] * y_sampled)
    alpha2 = @. (1.0 - mui) * eta[7]
    beta2 = @. mui * eta[7] + 1.0
    return (eta[6] > eps(Float64)) & (eta[7] > eps(Float64)) ? sum(logpdf.(Normal(0, sqrt(eta[6])), @.y_sampled - eta[4] - x_sampled * eta[5]) .+
    @. logpdf(BetaPrime(alpha2, beta2), w_sampled .- 1.0)) : -Inf
end
=#

getq(θ) = TuringDiagMvNormal(θ[1:7], exp.(θ[8:14]))
#q = vi(logπ3, advi, getq, vcat(randn(5), rand(9)))
q = vi(logπ3, advi, getq, vcat(beta, theta, sigma2, phi, rand(7)))

@show q.m
@show q.σ

logπ3(vcat(beta, theta, sigma2, phi))
logπ3(vcat(q.m[1:6], 3.14))
logπ3(q.m)

logπ4 = function(eta; x_sampled = x_sampled, y_sampled = y_sampled, w_sampled = w_sampled,
    nodes = nodes, weights = weights)
    mui = @. logistic(eta[1] + eta[2] * x_sampled + eta[3] * y_sampled)
    alpha2 = @. (1 - mui) * eta[7]
    alpha2 = @. ifelse(alpha2 > eps(Float64), alpha2, eps(Float64))
    beta2 = @. mui * eta[7] + 1
    beta2 = @. ifelse(beta2 > eps(Float64), beta2, eps(Float64))
    sigma2 = @. ifelse(eta[6] > eps(Float64), eta[6], eps(Float64))
    muiz = @. eta[1] + eta[2] * x_sampled + eta[3] * (eta[4] + eta[5] * x_sampled)
    sigmaiz = eta[3]* sqrt(sigma2)
    denomi = (logistic.(muiz .+ sqrt2 .* transpose(nodes) .* sigmaiz) * weights) ./ sqrtπ
    return sum(logpdf.(Normal(0, sqrt(sigma2)), @. y_sampled - eta[4] - x_sampled * eta[5]) .+
    @. logpdf(BetaPrime(alpha2, beta2), w_sampled .- 1.0) - log(denomi) + log(mui)) +
    logpdf(MvNormal([-4, 0.4, 0.4, 0.5, 1.3], diagm(ones(5))), eta[1:5]) +
    logpdf(InverseGamma(3, 1), sigma2) + logpdf(InverseGamma(3, 8), eta[7] > eps(Float64) ? eta[7] : eps(Float64))
end

getq(θ) = TuringDiagMvNormal(θ[1:7], exp.(θ[8:14]))
#q = vi(logπ3, advi, getq, vcat(randn(5), rand(9)))
q = vi(logπ4, ADVI(10, 10_000), getq, vcat(beta, theta, sigma2, phi, rand(7)))

@show q.m
@show q.σ

@show logπ4(vcat(beta, theta, sigma2, phi))
#@show logπ4(vcat(q.m[1:3], theta, q.m[6:7]))
@show logπ4(q.m)
using Optim
minuslogπ4(eta) = -logπ4(eta)
optim = optimize(minuslogπ4, vcat(beta, theta, sigma2, phi))
@show optim.minimum
@show optim.minimizer

optim = optimize(minuslogπ4, q.m)
@show optim.minimum
@show optim.minimizer

mean(rand(q, 10_000)[7,:])
var(rand(q, 10_000)[7,:])

logπ4 = function(eta; x_sampled = x_sampled::Vector{Float64},
    y_sampled = y_sampled::Vector{Float64}, w_sampled = w_sampled::Vector{Float64},
    nodes = nodes::Vector{Float64}, weights = weights::Vector{Float64})
    mui = @. logistic(eta[1] + eta[2] * x_sampled + eta[3] * y_sampled)
    alpha2 = @. (1 - mui) * eta[7]
    alpha2 = @. ifelse(alpha2 > eps(Float64), alpha2, eps(Float64))
    beta2 = @. mui * eta[7] + 1
    beta2 = @. ifelse(beta2 > eps(Float64), beta2, eps(Float64))
    sigma2 = @. ifelse(eta[6] > eps(Float64), eta[6], eps(Float64))
    muiz = @. eta[1] + eta[2] * x_sampled + eta[3] * (eta[4] + eta[5] * x_sampled)
    sigmaiz = eta[3]* sqrt(sigma2)
    denomi = (logistic.(muiz .+ sqrt2 .* transpose(nodes) .* sigmaiz) * weights) ./ sqrtπ
    return sum(logpdf.(Normal(0, sqrt(sigma2)), @. y_sampled - eta[4] - x_sampled * eta[5]) .+
    @. logpdf(BetaPrime(alpha2, beta2), w_sampled .- 1.0) - log(denomi) + log(mui)) +
    logpdf(MvNormal([-4, 0.4, 0.4, 0.5, 1.3], diagm(ones(5))), eta[1:5]) +
    logpdf(InverseGamma(3, 1), sigma2) + logpdf(InverseGamma(3, 8), eta[7] > eps(Float64) ? eta[7] : eps(Float64))
end

function vectosym(vec, n)
    sym = Array{Float64}(undef, n, n)
    count = 1
    for i in 1:n
        for j in 1:n
            if i <= j
                sym[i, j] = vec[count]
                count = count + 1
            end
            if i < j
                sym[j, i] = sym[i, j]
            end
        end
    end
    return sym
end

tmpvec = fill(-Inf, 35)
tmpvec[1] = tmpvec[8] = tmpvec[14] = tmpvec[19] = tmpvec[23] = tmpvec[26] = tmpvec[28] = 0.0

getq(θ) = TuringDenseMvNormal(θ[1:7], exp.([θ[8]  θ[9]  θ[10] θ[11] θ[12] θ[13] θ[14] ;
                                            θ[9]  θ[15] θ[16] θ[17] θ[18] θ[19] θ[20] ;
                                            θ[10] θ[16] θ[21] θ[22] θ[23] θ[24] θ[25] ;
                                            θ[11] θ[17] θ[22] θ[26] θ[27] θ[28] θ[29] ;
                                            θ[12] θ[18] θ[23] θ[27] θ[30] θ[31] θ[32] ;
                                            θ[13] θ[19] θ[24] θ[28] θ[31] θ[33] θ[34] ;
                                            θ[14] θ[20] θ[25] θ[29] θ[32] θ[34] θ[35]]))
#getq(θ) = TuringDiagMvNormal(θ[1:7], exp.(θ[8:56]))
#q = vi(logπ3, advi, getq, vcat(randn(5), rand(9)))
q = vi(logπ4, ADVI(10, 10_000), getq, vcat(beta, theta, sigma2, phi, tmpvec))

@show q.m
@show q.C

@show logπ4(vcat(beta, theta, sigma2, phi))
@show logπ4(vcat(q.m[1:3], theta, q.m[6:7]))
@show logπ4(q.m)
mean(rand(q, 10_000)[7,:])
var(rand(q, 10_000)[7,:])
