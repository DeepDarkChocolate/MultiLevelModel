#PPS + Stratified Sampling
#126
n = 200
m = 18
N = 5000
Mi = 2000

beta0 = 0.0
beta1 = 2.0
sigma2a = 1.0

mua = -4.0
theta = [beta1, mua, sigma2a]
B = 500 # 497
K = 5000 # Number of MC samples generated from approximated Normal distr.

w_h = 1/3

eps_theta = 10^(-3)

interval = 6.0
rtol = 1e-5
K2 = 15

Random.seed!(1)
a = rand(Normal(mua,sqrt(sigma2a)),N)

a_tilde = exp.(2.5 .+ a) ./ (1 .+ exp.(2.5 .+ a))
M = round.(Int, Mi .* a_tilde)

#M = [Mi for i in 1:N] # SRS case

x = [rand(Normal(2.0,1.0),i) for i in M]

p = [@. 1 - 1 / (1 + exp(beta0 + beta1 * x[i] + a[i])) for i in 1:length(M)]
z = [rand(i) for i in M]

y = [ifelse.(z[i] .< p[i], 1, 0) for i in 1:length(M)]
#y = [vcat((@. rand(Binomial(1, p[i]), 1)...)) for i in 1:length(M)]

pi1 = n .* M ./ sum(M)

SampleIdx1 = StatsBase.sample(1:N, Weights(Float64.(M)), n, replace=false)
#SampleIdx1 = StatsBase.sample(1:N, n, replace=false) # SRS Sampling

m1 = Int(m * w_h)
m2 = Int(m - m * w_h)

SampleIdx2 = [vcat(StatsBase.sample( (1:length(z[element]))[z[element] .> 0.5], m1, replace=false),
                StatsBase.sample( (1:length(z[element]))[z[element] .<= 0.5], m2, replace=false))
                for (i, element) in enumerate(SampleIdx1)] # Stratified Sampling

x_sampled = [x[element][SampleIdx2[i]] for (i, element) in enumerate(SampleIdx1)]
y_sampled = [y[element][SampleIdx2[i]] for (i, element) in enumerate(SampleIdx1)]
z_sampled = [z[element][SampleIdx2[i]] for (i, element) in enumerate(SampleIdx1)]

pi1_sampled = pi1[SampleIdx1]
M1 = [sum(z[element] .> 0.5) for element in SampleIdx1]
M2 = [sum(z[element] .<= 0.5) for element in SampleIdx1]

pi2_sampled = [ifelse.(z_sampled[i] .> 0.5, m1 / M1[i], m2 / M2[i])
                for (i, element) in enumerate(SampleIdx1)]

pi2mat_sampled = [
let pi2mat_sampled_tmp = ones(m,m)
pi2mat_sampled_tmp1 = ones(m1,m1) .* m1 * (m1 - 1) / M1[i] / (M1[i] - 1)
pi2mat_sampled_tmp1[diagind(pi2mat_sampled_tmp1)] .= m1 / M1[i]
@. pi2mat_sampled_tmp[z_sampled[1] .> 0.5, z_sampled[1] .> 0.5] = pi2mat_sampled_tmp1

pi2mat_sampled_tmp2 = ones(m2,m2) .* m2 * (m2 - 1) / M2[i] / (M2[i] - 1)
pi2mat_sampled_tmp2[diagind(pi2mat_sampled_tmp2)] .= m2 / M2[i]
@. pi2mat_sampled_tmp[z_sampled[1] .<= 0.5, z_sampled[1] .<= 0.5] = pi2mat_sampled_tmp2

@. pi2mat_sampled_tmp[z_sampled[1] .> 0.5, z_sampled[1] .<= 0.5] = m1 * m2 / M1[i] / M2[i]
@. pi2mat_sampled_tmp[z_sampled[1] .<= 0.5, z_sampled[1] .> 0.5] = m1 * m2 / M1[i] / M2[i]
pi2mat_sampled_tmp
end for i in 1:n]

w1_sampled = @. 1 / pi1_sampled
w2_sampled = [@. 1 / pi2_sampled[i] for i in 1:n]

fm = @formula(Y ~ X + (1|fac))
fm1 = fit(MixedModel, fm, DataFrame(X=vcat(x_sampled...), Y=vcat(y_sampled...), fac = string.([SampleIdx1[i] for i in 1:n for j in 1:length(SampleIdx2[i])])), Binomial())

beta_ini = coef(fm1)
mua_ini = beta_ini[1]
beta_ini[1] = 0.0

sigma2a_ini  = VarCorr(fm1).σρ.fac.σ[1]^2
#sigma2a_ini  = sigma2a + rand(1)[1] - 0.5

theta_t1 = Vector{Float64}(undef, 3)
theta_t2 = Vector{Float64}(undef, 3)
beta_t1 = copy(beta_ini)
mua_t1 = mua_ini
sigma2a_t1 = sigma2a_ini
beta_t2 = copy(beta_ini)

cnt = 0
rtol_beta = 1e-2

  #global beta_t1, mua_t1, sigma2a_t1, theta_t1, theta_t2
  cnt += 1

  ahat = solveahat(x_sampled, y_sampled, w2_sampled, beta_t1)

  resTMP = updatebetamat_S_fast2(ahat, x_sampled, y_sampled, w1_sampled, w2_sampled, pi2mat_sampled, beta_t1, mua_t1, sigma2a_t1, interval, rtol, K2) # MCMC

  ahat = solveahat(x_sampled, y_sampled, w2_sampled, beta_t1)

  phat = [@. 1 - 1 / (1 + exp(beta_t1[2] * x_sampled[i] + ahat[i])) for i in 1:n]

  #vhat = [sum(@. w2_sampled[i]^2 * (y_sampled[i] - phat[i])^2) / sum(@. phat[i] * (1 - phat[i]) * w2_sampled[i])^2 for i in 1:n]
  vhat = [let ratioi = @. (y_sampled[i] - phat[i]) * w2_sampled[i]
  transpose(ratioi) * (1 .- (pi2_sampled[i] * transpose(pi2_sampled[i]) ./ pi2mat_sampled[i])) * ratioi
  end / sum(@. phat[i] * (1 - phat[i]) * w2_sampled[i])^2 for i in 1:n]
  #vhat = [sum(@. w2_sampled[i]^2 * (y_sampled[i] - phat[i])^2) / sum(@. (y_sampled[i] - phat[i])^2 * w2_sampled[i])^2 for i in 1:n]
  #vhat = [sum(@. w2_sampled[i]^2 * phat[i] * (1 - phat[i])) / sum(@. phat[i] * (1 - phat[i]) * w2_sampled[i])^2 for i in 1:n]

  astar = @. (sigma2a_t1 * ahat + vhat * mua_t1) / (vhat + sigma2a_t1)
  vstar = @. sigma2a_t1 / (vhat + sigma2a_t1) * vhat
  astar[isnan.(astar)] .= mua_t1
  vstar[isnan.(vstar)] .= sigma2a_t1

  beta_t2[2] = beta_t1[2] + updatebetamat_new3(astar, vstar, x_sampled, y_sampled, w1_sampled, w2_sampled, beta_t1, sigma2a_t1, K)
  @show resTMP
  @show beta_t2[2]
  @show mua_t2 = mean(astar, weights(w1_sampled))
  @show sigma2a_t2 = mean(astar.^2 .+ vstar, weights(w1_sampled)) - mua_t2^2


xquad, wquad = gauss(v -> pdf(Normal(mua_t1, sqrt(sigma2a_t1)), v), K2, mua_t1- interval, mua_t1 + interval, rtol=rtol)
#[pdf(Normal(xquad[k], v(xquad[k], x_sampled[1], y_sampled[1], w2_sampled[1], beta_t1, pi2mat_sampled[1], sigma2a_t1, ahat[1])), ahat[1]) for k in 1:K2] .* wquad

i = 175
p1 = plot(xquad, [pdf(Normal(xquad[k], v(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i])), ahat[i]) for k in 1:K2] .* wquad)
p2 = plot(xquad, [pdf(Normal(xquad[k], v2(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i])), ahat[i]) for k in 1:K2] .* wquad)
plot(p1, p2, layout = (1,2))
y_sampled[76]

denomi = sum([pdf(Normal(xquad[k], v(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i])), ahat[i]) for k in 1:K2] .* wquad)
w1_sampled[i] * sum(xquad .^ 2 .* wquad .* [pdf(Normal(xquad[k], v(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i])), ahat[i]) for k in 1:K2]) / denomi

denomi = sum([pdf(Normal(xquad[k], v2(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i])), ahat[i]) for k in 1:K2] .* wquad)
w1_sampled[i] * sum(xquad .^ 2 .* wquad .* [pdf(Normal(xquad[k], v2(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i])), ahat[i]) for k in 1:K2]) / denomi

p1 = plot(xquad, xquad .^ 2 .* wquad .* [pdf(Normal(xquad[k], v(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i])), ahat[i]) for k in 1:K2] ./ sum([pdf(Normal(xquad[k], v(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i])), ahat[i]) for k in 1:K2] .* wquad))
p2 = plot(xquad, xquad .^ 2 .* wquad .* [pdf(Normal(xquad[k], v2(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i])), ahat[i]) for k in 1:K2] ./ sum([pdf(Normal(xquad[k], v2(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i])), ahat[i]) for k in 1:K2] .* wquad))
plot(p1, p2, layout = (1,2))

[sum(y_sampled[i] .== 0.0) for i in 1:n]
findmax([sum(y_sampled[i] .== 0.0) for i in 1:n])

#sum(y_sampled[120] .== 0.0)

print(findmax([sum(y_sampled[i] .== 0.0) for i in 1:n]))

[sum(y_sampled[i] .== 1.0) for i in 1:n]
findmax([sum(y_sampled[i] .== 1.0) for i in 1:n])
print(findmax([sum(y_sampled[i] .== 1.0) for i in 1:n]))

#=
using Plots
i = 26
p1 = plot(xquad, [pdf(Normal(xquad[k], v(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i])), ahat[i]) for k in 1:K2] .* wquad)
p2 = plot(xquad, [pdf(Normal(xquad[k], v2(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i])), ahat[i]) for k in 1:K2] .* wquad)
plot(p1, p2, layout = (1,2))

i = 10
p3 = plot(xquad, [1 / v(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i]) for k in 1:K2] .* wquad)
p4 = plot(xquad, [1 / v2(xquad[k], x_sampled[i], y_sampled[i], w2_sampled[i], beta_t1, pi2mat_sampled[i], sigma2a_t1, ahat[i]) for k in 1:K2] .* wquad)
plot(p3, p4, layout = (1,2))
=#
