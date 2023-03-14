n = 30
N = 2000

beta0 = 0.5
beta1 = 3.0
sigma2e = 3.0
theta = [beta0, beta1, sigma2e]

B = 10000

w_h = 0.7

using Distributions
using Random
using LinearAlgebra
using StatsBase
using NamedArrays
using DataFrames

verbose = false

theta_res = Array{Float64}(undef, B, 3)

for simnum in 1:B
  Random.seed!(simnum)
  x = randn(N)
  e = rand(Normal(0,sqrt(sigma2e)),N)
  y = @. beta0 + x * beta1 + e

  n1 = Int(n * w_h)
  n2 = Int(n - n * w_h)

  SampleIdx = vcat(StatsBase.sample( (1:N)[y .>= 0.0], n1, replace=false),
      StatsBase.sample( (1:N)[y .< 0.0], n2, replace=false))

  x_sampled = x[SampleIdx]
  y_sampled = y[SampleIdx]

  N1 = sum(y .>= 0.0)
  N2 = sum(y .< 0.0)

  pi_sampled = ifelse.(y_sampled .> 0.0, n1 / N1, n2 / N2)
  w_sampled = 1 ./ pi_sampled

  xmat_sampled = hcat(fill(1, n), x_sampled)
  wmat_sampled = diagm(w_sampled)
  betahat = (xmat_sampled' * wmat_sampled * xmat_sampled) \ (xmat_sampled' * wmat_sampled * y_sampled)
  sigma2ehat = sum(@. w_sampled * (y_sampled - beta0 - beta1 * x_sampled)^2) / (sum(w_sampled))
  #sigma2ehat = mean(@. (y_sampled - beta0 - beta1 * x_sampled)^2)
  theta_res[simnum,:] = [betahat[1], betahat[2], sigma2ehat]
end

function summary_data(theta_res)
  Meantheta = vcat(mapslices(mean, theta_res, dims = 1)...)

  Res = zeros(3,3)
  Res[:,1] = round.(Meantheta - theta; digits = 2)
  Res[:,2] =  round.(vcat(mapslices(var, theta_res, dims = 1)...) .* (B-1) ./ B; digits = 4)
  Res[:,3] =   round.(vcat(mapslices(mean, (theta .- transpose(theta_res)).^2, dims = 2)...); digits = 4)

  Res = NamedArray(Res)
  NamedArrays.setnames!(Res, ["beta0", "beta1", "sigma2e"], 1)
  NamedArrays.setnames!(Res, ["Bias", "Var", "MSE"], 2)
  return Res
end
@show n
println(summary_data(theta_res))
