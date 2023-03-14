#PPS + Stratified Sampling
#126
n = 200
m = 27
N = 5000
Mi = 2000

beta0 = 0.0
beta1 = 1.0
sigma2a = 0.5

mua = -0.0
theta = [beta1, mua, sigma2a]
B = 500 # 497
K = 5000 # Number of MC samples generated from approximated Normal distr.

w_h = 1 / 3

eps_theta = 10^(-3)

interval = 6.0
rtol = 1e-5
K2 = 10

using Distributions
using Random
using LinearAlgebra
using NamedArrays
using StatsBase
using MixedModels
using DataFrames
using QuadGK
using Optim
using Roots
#using RCall
using Dates
using CSV

#@rlibrary("pps")

include("ftns_method1.jl")
include("ftns_method2_gq.jl")
include("ftns_method3_gq.jl")
include("ftns_method4.jl")
include("ftns_method5.jl")



theta_res = Array{Float64}(undef, B, 3)
theta_RES = Array{Float64}(undef, B, 3)
#theta_res_new = Array{Float64}(undef, B, 3)
theta_res_new3 = Array{Float64}(undef, B, 3)
theta_RES_new3 = Array{Float64}(undef, B, 3)
theta_res_new4 = Array{Float64}(undef, B, 3)
theta_res_new5 = Array{Float64}(undef, B, 3)
theta_res_new6 = Array{Float64}(undef, B, 3)
theta_res_new7 = Array{Float64}(undef, B, 3)

error_return = trues(B, 5)
time_return = zeros(B)

## Simulation start
dirName = string("m", m, "n", n, "beta1", beta1, "CJS_modified_w2")
#dirName = string("m", m, "n", n, "CJS")
mkdir(dirName)

verbose = true

@time for simnum in 1:B
#@time Threads.@threads for simnum in 1:B
  elapsedtime = @elapsed begin
  Random.seed!(simnum+11)
  open(string(dirName, "/log.txt"), "a+") do io
  write(io, "$simnum\n")
  println(simnum)
  end;

  a = rand(Normal(mua,sqrt(sigma2a)),N)

  a_tilde = exp.(2.5 .+ a) ./ (1 .+ exp.(2.5 .+ a))
  M = round.(Int, Mi .* a_tilde)

  #M = [Mi for i in 1:N] # SRS case

  x = [rand(Normal(0.0,1.0),i) for i in M]

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

  ## normal approximation w/ v

    beta_t1 = copy(beta_ini)
    mua_t1 = mua_ini
    sigma2a_t1 = sigma2a_ini
    beta_t2 = copy(beta_ini)

    cnt = 0
    rtol_beta = 1e-2
    while true
      #global beta_t1, mua_t1, sigma2a_t1, theta_t1, theta_t2
      cnt += 1

      ahat = solveahat(x_sampled, y_sampled, w2_sampled, beta_t1)

      #resTMP = updatebetamat(ahat, x_sampled, y_sampled, w1_sampled, w2_sampled, beta_t1, mua_t1, sigma2a_t1, K * 5) # MCMC
      #resTMP = updatebetamat_fast(ahat, x_sampled, y_sampled, w1_sampled, w2_sampled, pi2mat_sampled, beta_t1, mua_t1, sigma2a_t1, interval, rtol, K2) # gaussian quadrature
      #resTMP = updatebetamat_fast(ahat, x_sampled, y_sampled, w1_sampled, w2_sampled, beta_t1, mua_t1, sigma2a_t1, interval, rtol, K2) # gaussian quadrature
      resTMP = updatebetamat(ahat, x_sampled, y_sampled, w1_sampled, w2_sampled, pi2mat_sampled, beta_t1, mua_t1, sigma2a_t1) # Julia Integral and findroot ftn
      #resTMP = updatebetamat(ahat, x_sampled, y_sampled, w1_sampled, w2_sampled, pi2mat_sampled, beta_t1, mua_t1, sigma2a_t1, 7.0, 1e-5) # gaussian quadrature
      #@show beta_t1[2]
      #@show resTMP[1]

      beta_t2[2] = resTMP[1]
      mua_t2 = resTMP[2]
      sigma2a_t2 = resTMP[3]

      if any([isnan(beta_t2[2]), isnan(mua_t2), isnan(sigma2a_t2)])
        println("simnum", simnum, ", cnt = ", cnt, ": NAN generated(normal approximation w/ v): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = 2 * mua_ini * rand()
        sigma2a_t1 = 2 * sigma2a_ini * rand()
      elseif  sigma2a_t2 > sigma2a * 20.0
        println("sigma2a_t2", simnum, ", cnt = ", cnt, ", sigma2a_t2 = ", sigma2a_t2, ": Too large sigma(normal approximation w/ v): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = 2 * mua_ini * rand()
        sigma2a_t1 = 2 * sigma2a_ini * rand()
      elseif  sigma2a_t2 < sigma2a * 0.05
        println("simnum", simnum, ", cnt = ", cnt, ", sigma2a_t2 = ", sigma2a_t2, ": Too small sigma(normal approximation w/ v): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = 2 * mua_ini * rand()
        sigma2a_t1 = 2 * sigma2a_ini * rand()
      elseif  cnt > 100
        @warn(": Convergence Failed(normal approximation w/ v)")
        println("simnum", simnum, ", cnt = ", cnt)
        theta_t2 = vcat([beta_t2[2], mua_t2, sigma2a_t2])
        error_return[simnum, 1] = false
        break
      else
        theta_t1 = vcat([beta_t1[2], mua_t1, sigma2a_t1])
        theta_t2 = vcat([beta_t2[2], mua_t2, sigma2a_t2])
        if verbose == true
          println("v", theta_t2)
        end
        if norm(theta_t1 - theta_t2) < eps_theta
          break
        else
          beta_t1[2] = beta_t2[2]
          mua_t1 = mua_t2
          sigma2a_t1 = sigma2a_t2
        end
      end
    end

    theta_res[simnum,:] = theta_t2
#=
  ## normal approximation w/ vhat
    beta_t1 = copy(beta_ini)
    mua_t1 = mua_ini
    sigma2a_t1 = sigma2a_ini
    beta_t2 = copy(beta_ini)

    cnt = 0
    while true
      cnt += 1
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
      #beta_t2[2] = updatebetamat_new3_fast(astar, vstar, x_sampled, y_sampled, w1_sampled, w2_sampled, beta_t1, sigma2a_t1, 4.5, 1e-3, 14)

      mua_t2 = mean(astar, weights(w1_sampled))

      sigma2a_t2 = mean(astar.^2 .+ vstar, weights(w1_sampled)) - mua_t2^2

      if any([isnan(beta_t2[2]), isnan(mua_t2), isnan(sigma2a_t2)])
        println("simnum", simnum, ", cnt = ", cnt, ": NAN generated(normal approximation w/ vhat): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = 2 * mua_ini * rand()
        sigma2a_t1 = 2 * sigma2a_ini * rand()
      elseif  sigma2a_t2 > sigma2a * 20.0
        println("sigma2a_t2", simnum, ", cnt = ", cnt, ", sigma2a_t2 = ", sigma2a_t2, ": Too large sigma(normal approximation w/ vhat): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = 2 * mua_ini * rand()
        sigma2a_t1 = 2 * sigma2a_ini * rand()
      elseif  sigma2a_t2 < sigma2a * 0.05
        println("simnum", simnum, ", cnt = ", cnt, ", sigma2a_t2 = ", sigma2a_t2, ": Too small sigma(normal approximation w/ vhat): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = mua_ini
        sigma2a_t1 = 2 * sigma2a_ini * rand()
      elseif  cnt > 100
        @warn(": Convergence Failed(normal approximation w/ vhat)")
        println("simnum", simnum, ", cnt = ", cnt)
        beta_t1[2] = beta_ini[2]
        mua_t1 = mua_ini
        sigma2a_t1 = sigma2a_ini
        theta_t2 = vcat([beta_t2[2], mua_t2, sigma2a_t2])
        error_return[simnum, 2] = false
        break
      else
        theta_t1 = vcat([beta_t1[2], mua_t1, sigma2a_t1])
        theta_t2 = vcat([beta_t2[2], mua_t2, sigma2a_t2])
        if verbose == true
          println("vhat", theta_t2)
        end

        if norm(theta_t1 - theta_t2) < eps_theta
          break
        else
          beta_t1[2] = beta_t2[2]
          mua_t1 = mua_t2
          sigma2a_t1 = sigma2a_t2
        end
      end
    end

    theta_res_new3[simnum,:] = theta_t2

  ## normal approximation using profile likelihood
    beta_t1 = copy(beta_ini)
    mua_t1 = mua_ini
    sigma2a_t1 = sigma2a_ini
    beta_t2 = copy(beta_ini)

    cnt = 0
    while true
      cnt += 1

      #resTMP = updatebetamat_S(x_sampled, y_sampled, w1_sampled, w2_sampled, beta_t1, mua_t1, sigma2a_t1, K * 5) # MCMC
      #resTMP = updatebetamat_S(x_sampled, y_sampled, w1_sampled, w2_sampled, beta_t1, mua_t1, sigma2a_t1, 8.0, 1e-2) # Gaussian quadrature(do not use)
      #resTMP = updatebetamat_S_fast(x_sampled, y_sampled, w1_sampled, w2_sampled, beta_t1, mua_t1, sigma2a_t1, interval, rtol, K2) # Gaussian quadrature
      resTMP = updatebetamat_S_fast(x_sampled, y_sampled, w1_sampled, w2_sampled, pi2mat_sampled, beta_t1, mua_t1, sigma2a_t1, interval, rtol, K2) # Gaussian quadrature
      # Parameters depend on sigma^2
      # len = sqrt(sigma) * 10, rtol = 1e-4, qd points = len * 2.5???;
      # sigma^2 = 3: length * 2.5 = qd points # len = 17.0, rtol = 1e-4, qd points = 50;
      # sigma^2 = 1: length * 2.5 = qd points # len = 10.0, rtol = 1e-4, qd points = 25;
      #resTMP = updatebetamat_S(x_sampled, y_sampled, w1_sampled, w2_sampled, pi2mat_sampled, beta_t1, mua_t1, sigma2a_t1) # Julia Integral and findroot ftn
      #resTMP = updatebetamat_S(x_sampled, y_sampled, w1_sampled, w2_sampled, pi2mat_sampled, beta_t1, mua_t1, sigma2a_t1, 5.0, 1e-4) # Gaussian quadrature

      beta_t2[2] = resTMP[1]
      mua_t2 = resTMP[2]
      sigma2a_t2 = resTMP[3]

      if any([isnan(beta_t2[2]), isnan(mua_t2), isnan(sigma2a_t2)])
        println("simnum", simnum, ", cnt = ", cnt, ": NAN generated(profile likelihood): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = 2 * mua_ini * rand()
        sigma2a_t1 = 2 * sigma2a_ini * rand()
      elseif  sigma2a_t2 > sigma2a * 20.0
        println("simnum", simnum, ", cnt = ", cnt, ", sigma2a_t2 = ", sigma2a_t2, ": Too large sigma(profile likelihood): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = 2 * mua_ini * rand()
        sigma2a_t1 = 2 * sigma2a_ini * rand()
      elseif  sigma2a_t2 < sigma2a * 0.05
        println("simnum", simnum, ", cnt = ", cnt, ", sigma2a_t2 = ", sigma2a_t2, ": Too small sigma(profile likelihood): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = 2 * mua_ini * rand()
        sigma2a_t1 = 2 * sigma2a_ini * rand()
      elseif  cnt > 100
        @warn(": Convergence Failed(profile likelihood)")
        println("simnum", simnum, ", cnt = ", cnt)
        beta_t1[2] = beta_ini[2]
        mua_t1 = mua_ini
        sigma2a_t1 = sigma2a_ini
        theta_t2 = vcat([beta_t2[2], mua_t2, sigma2a_t2])
        error_return[simnum, 3] = false
        break
      else
        theta_t1 = vcat([beta_t1[2], mua_t1, sigma2a_t1])
        theta_t2 = vcat([beta_t2[2], mua_t2, sigma2a_t2])
        if verbose == true
          println("profile", theta_t2)
        end

        if norm(theta_t1 - theta_t2) < eps_theta
          break
        else
          beta_t1[2] = beta_t2[2]
          mua_t1 = mua_t2
          sigma2a_t1 = sigma2a_t2
        end
      end
    end

    theta_res_new4[simnum,:] = theta_t2

  ## Full-pseudo likelihood
    beta_t1 = copy(beta_ini)
    mua_t1 = mua_ini
    sigma2a_t1 = sigma2a_ini
    beta_t2 = copy(beta_ini)

    eta_t1 = 1 / sigma2a_t1
    cnt = 0
    while true
      cnt += 1

      (beta_t2[2], mua_t2, eta_t2) = foo(x_sampled, y_sampled, w1_sampled, w2_sampled, beta_t1, mua_t1, eta_t1, interval, rtol, K2)
      if eta_t2 < 0
        eta_t2 = rand()
      end

      if any([isnan(beta_t2[2]), isnan(mua_t2), isnan(eta_t2)])
        println("simnum", simnum, ", cnt = ", cnt, ": NAN generated(Full): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = 2 * mua_ini * rand()
        eta_t1 = 2 / sigma2a_ini * rand()
      elseif  eta_t2 > 1 / sigma2a * 20.0
        println("simnum", simnum, ", cnt = ", cnt, ", eta = ", eta_t2, ": Too large eta(Full): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = 2 * mua_ini * rand()
        eta_t1 = 2 / sigma2a_ini * rand()
      elseif  mua_t2 < mua - 3.0
        println("simnum", simnum, ", cnt = ", cnt, ", mu = ", mua_t2, ": Too small mua_t2(Full): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = 2 * mua_ini * rand()
        eta_t1 = 2 / sigma2a_ini * rand()
      elseif  eta_t2 < 1 / sigma2a * 0.05
        println("simnum", simnum, ", cnt = ", cnt, ", eta = ", eta_t2, ": Too small eta(Full): perturbation on initial parameters")
        beta_t1[2] = beta_ini[2]
        mua_t1 = 2 * mua_ini * rand()
        eta_t1 = 2 / sigma2a_ini * rand()
      elseif  cnt > 500
        @warn(": Convergence Failed(Composite)")
        println("simnum", simnum, ", cnt = ", cnt)
        beta_t1[2] = beta_ini[2]
        mua_t1 = mua_ini
        eta_t1 = 1 / sigma2a_ini
        theta_t2 = vcat([beta_t2[2], mua_t2, 1 / eta_t1])
        error_return[simnum, 4] = false
        break
      else
        theta_t1 = vcat([beta_t1[2], mua_t1, 1 / eta_t1])
        theta_t2 = vcat([beta_t2[2], mua_t2, 1 / eta_t2])
        if verbose == true
          println("full", theta_t2)
        end

        if norm(theta_t1 - theta_t2) < eps_theta
          break
        else
          beta_t1[2] = beta_t2[2]
          mua_t1 = mua_t2
          eta_t1 = eta_t2
        end
      end
    end

    theta_res_new6[simnum,:] = theta_t2

  ## Composite(Pairwise)-pseudo likelihood
      beta_t1 = copy(beta_ini)
      mua_t1 = mua_ini
      sigma2a_t1 = sigma2a_ini
      beta_t2 = copy(beta_ini)

      eta_t1 = 1 / sigma2a_t1
      #println((beta_ini, mua_ini, 1 / sigma2a_ini))
      cnt = 0
      while true
        cnt += 1

        (beta_t2[2], mua_t2, eta_t2) = foo2(x_sampled, y_sampled, w1_sampled, pi2mat_sampled, beta_t1, mua_t1, eta_t1, interval, rtol, K2)
        if eta_t2 < 0
          eta_t2 = rand()
        end

        if any([isnan(beta_t2[2]), isnan(mua_t2), isnan(eta_t2)])
          println("simnum", simnum, ", cnt = ", cnt, ": NAN generated(Composite): perturbation on initial parameters")
          beta_t1[2] = beta_ini[2]
          mua_t1 = 2 * mua_ini * rand()
          eta_t1 = 2 / sigma2a_ini * rand()
        elseif  eta_t2 > 1 / sigma2a * 20.0
          println("simnum", simnum, ", cnt = ", cnt, ", eta = ", eta_t2, ": Too large eta(Composite): perturbation on initial parameters")
          beta_t1[2] = beta_ini[2]
          mua_t1 = 2 * mua_ini * rand()
          eta_t1 = 2 / sigma2a_ini * rand()
        elseif  beta_t2[2] < beta1 - 3.0
          println("simnum", simnum, ", cnt = ", cnt, ", beta_t2[2] = ", beta_t2[2], ": Too small beta_t2[2](Composite): perturbation on initial parameters")
          beta_t1[2] = beta_ini[2]
          mua_t1 = 2 * mua_ini * rand()
          eta_t1 = 2 / sigma2a_ini * rand()
        elseif  mua_t2 < mua - 3.0
          println("simnum", simnum, ", cnt = ", cnt, ", mu = ", mua_t2, ": Too small mua_t2(Composite): perturbation on initial parameters")
          beta_t1[2] = beta_ini[2]
          mua_t1 = 2 * mua_ini * rand()
          eta_t1 = 2 / sigma2a_ini * rand()
        elseif  eta_t2 < 1 / sigma2a * 0.05
          println("simnum", simnum, ", cnt = ", cnt, ", eta = ", eta_t2, ": Too small eta(Composite): perturbation on initial parameters")
          beta_t1[2] = beta_ini[2]
          mua_t1 = 2 * mua_ini * rand()
          eta_t1 = 2 / sigma2a_ini * rand()
        elseif  cnt > 500
          @warn(": Convergence Failed(Composite)")
          println("simnum", simnum, ", cnt = ", cnt)
          beta_t1[2] = beta_ini[2]
          mua_t1 = mua_ini
          eta_t1 = 1 / sigma2a_ini
          theta_t2 = vcat([beta_t2[2], mua_t2, 1 / eta_t1])
          error_return[simnum, 5] = false
          break
        else
          theta_t1 = vcat([beta_t1[2], mua_t1, 1 / eta_t1])
          theta_t2 = vcat([beta_t2[2], mua_t2, 1 / eta_t2])
          if verbose == true
            println("Composite", theta_t2)
          end

          if norm(theta_t1 - theta_t2) < eps_theta
            break
          else
            beta_t1[2] = beta_t2[2]
            mua_t1 = mua_t2
            eta_t1 = eta_t2
          end
        end
      end

      theta_res_new7[simnum,:] = theta_t2
=#
  end
  time_return[simnum] = elapsedtime
end

function summary_data(theta_res)
  Meantheta = vcat(mapslices(mean, theta_res, dims = 1)...)

  Res = zeros(3,3)
  Res[:,1] = round.(Meantheta - theta; digits = 2)
  Res[:,2] =  round.(vcat(mapslices(var, theta_res, dims = 1)...) .* (B-1) ./ B; digits = 4)
  Res[:,3] =   round.(vcat(mapslices(mean, (theta .- transpose(theta_res)).^2, dims = 2)...); digits = 4)

  Res = NamedArray(Res)
  NamedArrays.setnames!(Res, ["beta1", "mua", "sigma2a"], 1)
  NamedArrays.setnames!(Res, ["Bias", "Var", "MSE"], 2)
  return Res
end

println(summary_data(theta_res_new3))
#println(summary_data(theta_RES_new3))
#println(summary_data(theta_res_new))
println(summary_data(theta_res))
#println(summary_data(theta_RES))
println(summary_data(theta_res_new4))
#println(summary_data(theta_res_new5))
println(summary_data(theta_res_new6))
println(summary_data(theta_res_new7))

CSV.write(string(dirName, "/summary.txt"),  DataFrame(summary_data(theta_res_new3)), header=false)
CSV.write(string(dirName, "/summary.txt"),  DataFrame(summary_data(theta_res)), header=false, append=true)
CSV.write(string(dirName, "/summary.txt"),  DataFrame(summary_data(theta_res_new4)), header=false, append=true)
CSV.write(string(dirName, "/summary.txt"),  DataFrame(summary_data(theta_res_new6)), header=false, append=true)
CSV.write(string(dirName, "/summary.txt"),  DataFrame(summary_data(theta_res_new7)), header=false, append=true)


RES_TOTAL = zeros(18,3)
RES_VEC = summary_data.([theta_res_new3, theta_RES_new3, theta_res, theta_res_new4, theta_res_new6, theta_res_new7])
for j in 1:3
  for i in 1:6
    RES_TOTAL[(j - 1)* 6 + i,:] = RES_VEC[i][j,:]
  end
end
#RES_TOTAL
#print(DataFrame(RES_TOTAL))

println("Number of failure")
println(sum(@. 1 - error_return))

println("Time summary")
println(summarystats(time_return))

println("Total Time")
println(Time(0) + Second(round(sum(time_return))))

#(1:length(error_return[:,1]))[error_return[:,1] .== false]
#(1:length(error_return[:,1]))[error_return[:,2] .== false]



CSV.write(string(dirName, "/theta_res_new3.csv"),  DataFrame(theta_res_new3), header=false)
CSV.write(string(dirName, "/theta_res.csv"),  DataFrame(theta_res), header=false)
CSV.write(string(dirName, "/theta_res_new4.csv"),  DataFrame(theta_res_new4), header=false)
CSV.write(string(dirName, "/theta_res_new6.csv"),  DataFrame(theta_res_new6), header=false)
CSV.write(string(dirName, "/theta_res_new7.csv"),  DataFrame(theta_res_new7), header=false)
CSV.write(string(dirName, "/error_return.csv"),  DataFrame(error_return), header=false)
sum(theta_res_new7[:,3] .> 5.0)
sum(theta_res_new4[:,3] .> 5.0)
[theta_res_new7[:,3] .> 5.0][1]

findmax(theta_res[:,3])
median(theta_res_new4[:,3])
mean(theta_res_new4[:,3])
Plots.histogram(theta_res_new4[:,3])

#=
for k in 1:B
  if theta_res_new7[k,3] .> 5.0
    @show k
  end
end
=#
