# Normal approximation using the profile function
#=
function v_profile(ai, xi, yi, w2i, beta_t1)
  Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + ai))
  return sqrt(sum(@. (w2i - 1) * w2i * (yi - Pi)^2))
end

function S_profile(ai, xi, yi, w2i, beta_t1)
  Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + ai))
  return sum(@. (yi - Pi) * w2i)
end
=#

function solveahat(x, y, w2, beta_t1, mua_t1, sigma2a_t1)
  n = size(x)[1]
  a_t2 = Vector{Float64}(undef, n)
  sigmaa_t1 = sqrt(sigma2a_t1)
  #Threads.@threads for i in 1:n
  for i in 1:n
    #@show i
    ai0 = mua_t1
    xi = x[i]
    yi = y[i]
    w2i = w2[i]
    if all(y[i] .== 0)
      a_t2[i] = -Inf
    elseif all(y[i] .== 1)
      a_t2[i] = Inf
    else
      ai_t1 = ai0
      ai_t2 = ai0
      cnt2 = 0

      while true
        cnt2 += 1
        etai = @. beta_t1[2] * xi + ai_t1
        Pi = @. 1 - 1 / (1 + exp(etai))
        ai_t2 = ai_t1 + sum(@. (yi - Pi) * w2i) ./ sum(@. Pi * (1 - Pi) * w2i)
        #@show ai_t2
        if isnan(ai_t2)
          #@warn(": NAN generated(solveahat)")
          #println("ai_t1 = ", ai_t1)
          #println("ai_t2 = ", ai_t2)
          #println("y[i] = ", y[i])
          #println("i = ", i)
          #println("cnt2 = ", cnt2)
          ai_t1 = rand(Normal(mua_t1,sigmaa_t1),1)[1]
          #break
        #elseif isinf(ai_t2)
        #  @warn(": inf generated(solveahat)")
        #  println("ai_t1 = ", ai_t1)
        #  println("ai_t2 = ", ai_t2)
        #  println("y[i] = ", y[i])
        #  println("i = ", i)
        #  println("cnt2 = ", cnt2)
        #  ai_t2 = ai_t1
        #  break
        elseif abs(ai_t1 - ai_t2) < 10^(-5)
          break
        elseif cnt2 > 500
          @warn("aihat NOT convergent")
          println("ai_t1 = ", ai_t1)
          println("ai_t2 = ", ai_t2)
          println("y[i] = ", y[i])
          println("i = ", i)
          break
        else
          ai_t1 = ai_t2
        end
      end
      a_t2[i] = ai_t2
    end
  end
  return a_t2
end

function v_profile(ai, xi, yi, w2i, beta_t1, pi2mati, m)
  Pi = @. 1 - 1 / (1 + exp(beta_t1[1] + beta_t1[2] * xi + ai))
  ratioi = (yi .- Pi) .* w2i
  #ratioi = (yi .- mean(yi) .- Pi .+ mean(Pi)) .* w2i
  #ratioi = sqrt.(Pi .* (1 .- Pi)) .* w2i
  p2i = @. 1 / w2i
  varhat = transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi
  #varhat = transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi + sum(@. w2i * Pi * (1 - Pi))
  #varhat = sum(@. w2i^2 * Pi * (1 - Pi))
  #if isnan(varhat)
  #  varhat = 0.0001
  #end
  #return sqrt(varhat)
  return sqrt(ifelse(varhat > 0, varhat, 0.0001)) # required?
end

function v_profile(ahati, ai, xi, yi, w2i, beta_t1, pi2mati, m)
  Pi = @. 1 - 1 / (1 + exp(beta_t1[1] + beta_t1[2] * xi + ahati))
  ratioi = (yi .- Pi) .* w2i
  #ratioi = (yi .- mean(yi) .- Pi .+ mean(Pi)) .* w2i
  #ratioi = sqrt.(Pi .* (1 .- Pi)) .* w2i
  p2i = @. 1 / w2i
  varhat = transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi
  #varhat = transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi + sum(@. w2i * Pi * (1 - Pi))
  #varhat = sum(@. w2i^2 * Pi * (1 - Pi))
  if isnan(varhat)
    varhat = 0.0001
  end
  return sqrt(varhat)
  #return sqrt(ifelse(varhat > 0, varhat, 0.0001)) # required?
end


function S_profile(ai, xi, yi, w2i, beta_t1)
  Pi = @. 1 - 1 / (1 + exp(beta_t1[1] + beta_t1[2] * xi + ai))
  return sum(@. (yi - Pi) * w2i)
end

function updatebetamat_S(x, y, w1, w2, beta_t1_ogn, mua_t1, sigma2a_t1, interval, rtol)
  n = size(x)[1]
  mu_num = 0.0
  sigma_num = 0.0
  K = 12

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = copy(beta_t1_ogn)
  beta_t2 = zeros(2)

  cnt = 0

  while true
    beta_denom = 0.0
    beta_num = 0.0
    cnt += 1
    if cnt % 10 == 0
      interval += 1.0
    end

    for i in 1:n
    #println(i)
    #@time begin
      xi = x[i]
      yi = y[i]
      w2i = w2[i]
      #w2_tildei = size(x[i])[1] .* w2i ./sum(w2i)
      m = size(xi)[1]

      res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1, m)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -15, 15)
      #res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), [0.0], LBFGS())
      mode = Optim.minimizer(res)[1]

      xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(0, v_profile(aik, xi, yi, w2i, beta_t1, m)), S_profile(aik, xi, yi, w2i, beta_t1)), K, mode-interval, mode+interval, rtol=rtol)
      denomi = sum(wquad)
      Pi_mat = Array{Float64}(undef, length(xi), K)
      for k in 1:K
        @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
      end
      beta_num += w1[i] * sum([sum(@. w2i * (yi - Pi_mat[:,k]) * xi) for k in 1:K] .* wquad) / denomi
      beta_denom += w1[i] * sum([sum(@. w2i * Pi_mat[:,k] * (1 - Pi_mat[:,k]) * xi ^ 2) for k in 1:K] .* wquad) / denomi
      #beta_num += w1[i] * sum([sum(@. w2_tildei * (yi - Pi_mat[:,k]) * xi) for k in 1:K] .* wquad) / denomi
      #beta_denom += w1[i] * sum([sum(@. w2_tildei * Pi_mat[:,k] * (1 - Pi_mat[:,k]) * xi ^ 2) for k in 1:K] .* wquad) / denomi

    #end
    end
    beta_t2[2] = beta_t1[2] + beta_num / beta_denom
    #@show beta_t2
    if abs(beta_t1[2] - beta_t2[2]) > 1e-2
      beta_t1[2] = beta_t2[2]
    else
      beta_t1[2] = beta_t2[2]
      break
    end
    if cnt > 100
      @warn(": convergence failed: profile likelihood; beta")
      beta_t1[2] = beta_t2[2]
      break
    end
  end
  #@show beta_t1

  #Threads.@threads for i in 1:n
  for i in 1:n
  #println(i)
  #@time begin
    xi = x[i]
    yi = y[i]
    w2i = w2[i]

    res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1, m)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -15, 15)
    #res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), [0.0], LBFGS())
    mode = Optim.minimizer(res)[1]
    #@show mode
    #println("S_profile", map(aik -> S_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("v_profile", map(aik -> v_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("pdf", map(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -5.0:5.0))

    xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(0, v_profile(aik, xi, yi, w2i, beta_t1, m)), S_profile(aik, xi, yi, w2i, beta_t1)), K, mode-interval, mode+interval, rtol=rtol)
    denomi = sum(wquad)
    #@show y[i]
    #@show w2[i]
    #@show w1[i] * sum(@. xquad * wquad) / denomi
    #@show w1[i] * sum(@. xquad^2 * wquad) / denomi

    mu_num += w1[i] * sum(@. xquad * wquad) / denomi
    sigma_num += w1[i] * sum(@. xquad^2 * wquad) / denomi
  #end
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end
#=
function updatebetamat_S_fast(x, y, w1, w2, pi2mat, beta_t1_ogn, mua_t1, sigma2a_t1, interval, rtol, K2)
  n = size(x)[1]
  mu_num = 0.0
  sigma_num = 0.0
  K = K2

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = copy(beta_t1_ogn)
  beta_t2 = zeros(2)

  xquad, wquad = gauss(v -> pdf(Normal(mua_t1, sigmaa_t1), v), K, mua_t1- interval, mua_t1 + interval, rtol=rtol)

  cnt = 0
  while true
    beta_denom = 0.0
    beta_num = 0.0
    cnt += 1
    if cnt % 10 == 0
      interval += 1.0
    end

    for i in 1:n
    #println(i)
    #@time begin
      xi = x[i]
      yi = y[i]
      w2i = w2[i]
      pi2mati = pi2mat[i]
      m = size(xi)[1]
      w2_tildei = m .* w2i ./sum(w2i)

      if all(yi .== 0) | all(yi .== 1)
        #denomi = sum(wquad)
        #Pi_mat = Array{Float64}(undef, length(xi), K)
        #for k in 1:K
        #  @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
        #end
        #beta_num += w1[i] * sum([sum(w2i .* (yi .- Pi_mat[:,k]) .* xi) for k in 1:K] .* wquad) / denomi
        #beta_denom += w1[i] * sum([sum(w2i .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2) for k in 1:K] .* wquad) / denomi
      else
        denomi = sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad)
        Pi_mat = Array{Float64}(undef, length(xi), K)
        for k in 1:K
          @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
        end
        beta_num += w1[i] * sum([sum(w2_tildei .* (yi .- Pi_mat[:,k]) .* xi .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
        beta_denom += w1[i] * sum([sum(w2_tildei .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2.0 .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
      end
    #end
    end
    beta_t2[2] = beta_t1[2] + beta_num / beta_denom
    #@show beta_t2
    if abs(beta_t1[2] - beta_t2[2]) > 1e-5
      beta_t1[2] = beta_t2[2]
    else
      beta_t1[2] = beta_t2[2]
      break
    end
    if cnt > 100
      @warn(": convergence failed: profile likelihood; beta")
      beta_t1[2] = beta_t2[2]
      break
    end
  end
  #@show beta_t1

  #Threads.@threads for i in 1:n
  for i in 1:n
  #println(i)
  #@time begin
    xi = x[i]
    yi = y[i]
    w2i = w2[i]
    pi2mati = pi2mat[i]

    #@show mode
    #println("S_profile", map(aik -> S_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("v_profile", map(aik -> v_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("pdf", map(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -5.0:5.0))
    if all(yi .== 0) | all(yi .== 1)
      #denomi = sum(wquad)

      #mu_num += w1[i] * sum(xquad .* wquad) / denomi
      #sigma_num += w1[i] * sum(xquad .^ 2 .* wquad) / denomi

      #println(w1[i] * sum(xquad .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad))
      #println(w1[i] * sum(xquad .^ 2 .* wquad) / sum(wquad))

    else
      denomi = sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad)
      #@show y[i]
      #@show w2[i]
      #@show w1[i] * sum(@. xquad * wquad) / denomi
      #@show w1[i] * sum(@. xquad^2 * wquad) / denomi

      mu_num += w1[i] * sum(xquad .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / denomi
      sigma_num += w1[i] * sum(xquad .^2.0 .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / denomi
    end
  #end
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end
=#

function updatebetamat_S_fast(x, y, w1, w2, pi2mat, beta_t1_ogn, mua_t1, sigma2a_t1, interval, rtol, K2)
  n = size(x)[1]
  mu_num = 0.0
  sigma_num = 0.0
  K = K2

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = zeros(2)
  beta_t1[1] = mua_t1
  beta_t1[2] = beta_t1_ogn[2]
  beta_t2 = zeros(2)

  xquad, wquad = gauss(v -> pdf(Normal(0.0, sigmaa_t1), v), K, 0.0- interval, 0.0 + interval, rtol=rtol)
  #xquad, wquad = gauss(v -> pdf(Normal(mua_t1, sigmaa_t1), v), K, mua_t1- interval, mua_t1 + interval, rtol=rtol)

  cnt = 0
  while true
    beta_denom11 = 0.0
    beta_denom12 = 0.0
    beta_denom22 = 0.0
    beta_num1 = 0.0
    beta_num2 = 0.0
    cnt += 1
    if cnt % 10 == 0
      interval += 1.0
    end

    for i in 1:n
    #println(i)
    #@time begin
      xi = x[i]
      yi = y[i]
      w2i = w2[i]
      pi2mati = pi2mat[i]
      m = size(xi)[1]
      w2_tildei = m .* w2i ./sum(w2i)

      if all(yi .== 0) | all(yi .== 1)
        #denomi = sum(wquad)
        #Pi_mat = Array{Float64}(undef, length(xi), K)
        #for k in 1:K
        #  @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
        #end
        #beta_num += w1[i] * sum([sum(w2i .* (yi .- Pi_mat[:,k]) .* xi) for k in 1:K] .* wquad) / denomi
        #beta_denom += w1[i] * sum([sum(w2i .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2) for k in 1:K] .* wquad) / denomi
      else
        denomi = sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad)
        Pi_mat = Array{Float64}(undef, length(xi), K)
        for k in 1:K
          @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[1] + beta_t1[2] * xi + xquad[k]))
        end
        beta_num1 += w1[i] * sum([sum(w2_tildei .* (yi .- Pi_mat[:,k]) .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
        beta_num2 += w1[i] * sum([sum(w2_tildei .* (yi .- Pi_mat[:,k]) .* xi .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
        beta_denom11 += w1[i] * sum([sum(w2_tildei .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
        beta_denom12 += w1[i] * sum([sum(w2_tildei .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
        beta_denom22 += w1[i] * sum([sum(w2_tildei .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2.0 .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
      end
    #end
    end
    beta_t2 .= beta_t1 .+ ([[beta_denom11,beta_denom12]  [beta_denom12,beta_denom22]] \ [beta_num1, beta_num2])
    #@show beta_t2
    if norm(beta_t1 .- beta_t2) > 1e-5
      beta_t1 = copy(beta_t2)
    else
      beta_t1 = copy(beta_t2)
      break
    end
    if cnt > 100
      @warn(": convergence failed: profile likelihood; beta")
      beta_t1 = copy(beta_t2)
      break
    end
  end
  #@show beta_t1

  #Threads.@threads for i in 1:n
  for i in 1:n
  #println(i)
  #@time begin
    xi = x[i]
    yi = y[i]
    w2i = w2[i]
    pi2mati = pi2mat[i]

    #@show mode
    #println("S_profile", map(aik -> S_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("v_profile", map(aik -> v_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("pdf", map(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -5.0:5.0))
    if all(yi .== 0) | all(yi .== 1)
      #denomi = sum(wquad)

      #mu_num += w1[i] * sum(xquad .* wquad) / denomi
      #sigma_num += w1[i] * sum(xquad .^ 2 .* wquad) / denomi

      #println(w1[i] * sum(xquad .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad))
      #println(w1[i] * sum(xquad .^ 2 .* wquad) / sum(wquad))

    else
      denomi = sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad)
      #@show y[i]
      #@show w2[i]
      #@show w1[i] * sum(@. xquad * wquad) / denomi
      #@show w1[i] * sum(@. xquad^2 * wquad) / denomi

      mu_num += w1[i] * sum(xquad .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / denomi
      sigma_num += w1[i] * sum(xquad .^2.0 .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / denomi
    end
  #end
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], beta_t1[1], sigma_num / sum(w1) - muhat^2
end

function updatebetamat_S_fast_muhat(x, y, w1, w2, pi2mat, beta_t1_ogn, mua_t1, sigma2a_t1, interval, rtol, K2)
  n = size(x)[1]
  mu_num = 0.0
  sigma_num = 0.0
  K = K2

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = zeros(2)
  beta_t1[1] = mua_t1
  beta_t1[2] = beta_t1_ogn[2]
  beta_t2 = zeros(2)

  xquad, wquad = gauss(v -> pdf(Normal(0.0, sigmaa_t1), v), K, 0.0- interval, 0.0 + interval, rtol=rtol)
  #xquad, wquad = gauss(v -> pdf(Normal(mua_t1, sigmaa_t1), v), K, mua_t1- interval, mua_t1 + interval, rtol=rtol)

  cnt = 0
  while true
    beta_denom11 = 0.0
    beta_denom12 = 0.0
    beta_denom22 = 0.0
    beta_num1 = 0.0
    beta_num2 = 0.0
    cnt += 1
    if cnt % 10 == 0
      interval += 1.0
    end

    for i in 1:n
    #println(i)
    #@time begin
      xi = x[i]
      yi = y[i]
      w2i = w2[i]
      pi2mati = pi2mat[i]
      m = size(xi)[1]
      w2_tildei = m .* w2i ./sum(w2i)

      if all(yi .== 0) | all(yi .== 1)
        #denomi = sum(wquad)
        #Pi_mat = Array{Float64}(undef, length(xi), K)
        #for k in 1:K
        #  @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
        #end
        #beta_num += w1[i] * sum([sum(w2i .* (yi .- Pi_mat[:,k]) .* xi) for k in 1:K] .* wquad) / denomi
        #beta_denom += w1[i] * sum([sum(w2i .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2) for k in 1:K] .* wquad) / denomi
      else
        denomi = sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad)
        Pi_mat = Array{Float64}(undef, length(xi), K)
        for k in 1:K
          @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[1] + beta_t1[2] * xi + xquad[k]))
        end
        beta_num1 += w1[i] * sum([sum(w2_tildei .* (yi .- Pi_mat[:,k]) .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
        beta_num2 += w1[i] * sum([sum(w2_tildei .* (yi .- Pi_mat[:,k]) .* xi .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
        beta_denom11 += w1[i] * sum([sum(w2_tildei .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
        beta_denom12 += w1[i] * sum([sum(w2_tildei .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
        beta_denom22 += w1[i] * sum([sum(w2_tildei .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2.0 .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
      end
    #end
    end
    beta_t2 .= beta_t1 .+ ([[beta_denom11,beta_denom12]  [beta_denom12,beta_denom22]] \ [beta_num1, beta_num2])
    #@show beta_t2
    if norm(beta_t1 .- beta_t2) > 1e-5
      beta_t1 = copy(beta_t2)
    else
      beta_t1 = copy(beta_t2)
      break
    end
    if cnt > 100
      @warn(": convergence failed: profile likelihood; beta")
      beta_t1 = copy(beta_t2)
      break
    end
  end
  #@show beta_t1

  #Threads.@threads for i in 1:n
  for i in 1:n
  #println(i)
  #@time begin
    xi = x[i]
    yi = y[i]
    w2i = w2[i]
    pi2mati = pi2mat[i]

    #@show mode
    #println("S_profile", map(aik -> S_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("v_profile", map(aik -> v_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("pdf", map(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -5.0:5.0))
    if all(yi .== 0) | all(yi .== 1)
      #denomi = sum(wquad)

      #mu_num += w1[i] * sum(xquad .* wquad) / denomi
      #sigma_num += w1[i] * sum(xquad .^ 2 .* wquad) / denomi

      #println(w1[i] * sum(xquad .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad))
      #println(w1[i] * sum(xquad .^ 2 .* wquad) / sum(wquad))

    else
      denomi = sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad)
      #@show y[i]
      #@show w2[i]
      #@show w1[i] * sum(@. xquad * wquad) / denomi
      #@show w1[i] * sum(@. xquad^2 * wquad) / denomi

      mu_num += w1[i] * sum(xquad .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / denomi
      sigma_num += w1[i] * sum(xquad .^2.0 .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / denomi
    end
  #end
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], beta_t1[1] + muhat, sigma_num / sum(w1) - muhat^2
end

function updatebetamat_S_fast2(ahat, x, y, w1, w2, pi2mat, beta_t1_ogn, mua_t1, sigma2a_t1, interval, rtol, K2)
  n = size(x)[1]
  mu_num = 0.0
  sigma_num = 0.0
  K = K2

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = copy(beta_t1_ogn)
  beta_t2 = zeros(2)

  xquad, wquad = gauss(v -> pdf(Normal(mua_t1, sigmaa_t1), v), K, mua_t1- interval, mua_t1 + interval, rtol=rtol)

  cnt = 0
  while true
    beta_denom = 0.0
    beta_num = 0.0
    cnt += 1
    if cnt % 10 == 0
      interval += 1.0
    end

    for i in 1:n
    #println(i)
    #@time begin
      xi = x[i]
      yi = y[i]
      w2i = w2[i]
      pi2mati = pi2mat[i]
      m = size(xi)[1]
      ahati = ahat[i]

      if isinf(ahati)
        denomi = sum(wquad)
        Pi_mat = Array{Float64}(undef, length(xi), K)
        for k in 1:K
          @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
        end
        beta_num += w1[i] * sum([sum(w2i .* (yi .- Pi_mat[:,k]) .* xi) for k in 1:K] .* wquad) / denomi
        beta_denom += w1[i] * sum([sum(w2i .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2) for k in 1:K] .* wquad) / denomi

        #println("inf!, i = ", i)
        #xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) / v(aik, xi, yi, w2i, beta_t1, m), K, mua_t1-interval, mua_t1+interval, rtol=rtol)
      else

        denomi = sum([pdf(Normal(0, v_profile(ahati, xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad)
        Pi_mat = Array{Float64}(undef, length(xi), K)
        for k in 1:K
          @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
        end
      end
      beta_num += w1[i] * sum([sum(w2i .* (yi .- Pi_mat[:,k]) .* xi .* pdf(Normal(0, v_profile(ahati, xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
      beta_denom += w1[i] * sum([sum(w2i .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2.0 .* pdf(Normal(0, v_profile(ahati, xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
    #end
    end
    beta_t2[2] = beta_t1[2] + beta_num / beta_denom
    #@show beta_t2
    if abs(beta_t1[2] - beta_t2[2]) > 1e-5
      beta_t1[2] = beta_t2[2]
    else
      beta_t1[2] = beta_t2[2]
      break
    end
    if cnt > 100
      @warn(": convergence failed: profile likelihood; beta")
      beta_t1[2] = beta_t2[2]
      break
    end
  end
  #@show beta_t1

  #Threads.@threads for i in 1:n
  for i in 1:n

    #@time begin
    xi = x[i]
    yi = y[i]
    w2i = w2[i]
    pi2mati = pi2mat[i]
    ahati = ahat[i]
    #@show i
    #@show ahati
    #@show mode
    #println("S_profile", map(aik -> S_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("v_profile", map(aik -> v_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("pdf", map(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -5.0:5.0))
    if isinf(ahati)
      denomi = sum(wquad)

      mu_num += w1[i] * sum(xquad .* wquad) / denomi
      sigma_num += w1[i] * sum(xquad .^ 2 .* wquad) / denomi

      #println(w1[i] * sum(xquad .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad))
      #println(w1[i] * sum(xquad .^ 2 .* wquad) / sum(wquad))

    else
      denomi = sum([pdf(Normal(0, v_profile(ahati, xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad)
      #@show y[i]
      #@show w2[i]
      #@show w1[i] * sum(@. xquad * wquad) / denomi
      #@show w1[i] * sum(@. xquad^2 * wquad) / denomi

      #println(w1[i] * sum(xquad .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad))
      #println(w1[i] * sum(xquad .* wquad.* [pdf(Normal(0, v_profile(ahati, xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / sum([pdf(Normal(0, v_profile(ahati, xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad))


      mu_num += w1[i] * sum(xquad .* wquad.* [pdf(Normal(0, v_profile(ahati, xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / denomi
      sigma_num += w1[i] * sum(xquad .^2.0 .* wquad.* [pdf(Normal(0, v_profile(ahati, xquad[k], xi, yi, w2i, beta_t1, pi2mati, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / denomi
    end
  #end
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end


#=
function v_profile(ai, xi, yi, w2i, beta_t1, pi2mati)
  Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + ai))
  ratioi = @. (yi - Pi) * w2i
  p2i = @. 1 / w2i
  varhat = transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi
  return sqrt(ifelse(varhat > 0, varhat, 0.01))
end

function updatebetamat_S(x, y, w1, w2, pi2mat, beta_t1_ogn, mua_t1, sigma2a_t1, interval, rtol)
  n = size(x)[1]
  beta_denom = 0.0
  beta_num = 0.0
  mu_num = 0.0
  sigma_num = 0.0
  K = 20

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = copy(beta_t1_ogn)
  beta_t2 = zeros(2)
  while true
    for i in 1:n
    #println(i)
    #@time begin
      xi = x[i]
      yi = y[i]
      w2i = w2[i]
      pi2mati = pi2mat[i]

      res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1, pi2mati)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -15, 15)
      #res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), [0.0], LBFGS())
      mode = Optim.minimizer(res)[1]

      xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(0, v_profile(aik, xi, yi, w2i, beta_t1, pi2mati)), S_profile(aik, xi, yi, w2i, beta_t1)), K, mode-interval, mode+interval, rtol=rtol)
      denomi = sum(wquad)
      Pi_mat = Array{Float64}(undef, length(xi), K)
      for k in 1:K
        @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
      end
      beta_num += w1[i] * sum([sum(@. w2i * (yi - Pi_mat[:,k]) * xi) for k in 1:K] .* wquad) / denomi
      beta_denom += w1[i] * sum([sum(@. w2i * Pi_mat[:,k] * (1 - Pi_mat[:,k]) * xi ^ 2) for k in 1:K] .* wquad) / denomi
    #end
    end
    beta_t2[2] = beta_t1[2] + beta_num / beta_denom
    #@show beta_t2
    if abs(beta_t1[2] - beta_t2[2]) > 1e-2
      beta_t1[2] = beta_t2[2]
    else
      beta_t1[2] = beta_t2[2]
      break
    end
  end


  #Threads.@threads for i in 1:n
  for i in 1:n
  #println(i)
  #@time begin
    xi = x[i]
    yi = y[i]
    w2i = w2[i]
    pi2mati = pi2mat[i]

    res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1, pi2mati)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -15, 15)
    #res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), [0.0], LBFGS())
    mode = Optim.minimizer(res)[1]
    #@show mode
    #println("S_profile", map(aik -> S_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("v_profile", map(aik -> v_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("pdf", map(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -5.0:5.0))

    xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(0, v_profile(aik, xi, yi, w2i, beta_t1, pi2mati)), S_profile(aik, xi, yi, w2i, beta_t1)), K, mode-interval, mode+interval, rtol=rtol)
    denomi = sum(wquad)
    #@show y[i]
    #@show w2[i]
    #@show w1[i] * sum(@. xquad * wquad) / denomi
    #@show w1[i] * sum(@. xquad^2 * wquad) / denomi

    mu_num += w1[i] * sum(@. xquad * wquad) / denomi
    sigma_num += w1[i] * sum(@. xquad^2 * wquad) / denomi
  #end
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end
=#

#=
function v_profile(ai, xi, yi, w2i, beta_t1, m)
  Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + ai))
  return sqrt(sum(@. w2i^2 * Pi * (1 - Pi)))
  #return sqrt(sum(@. w2i^2 * Pi * (1 - Pi)) * m / (m - 1))
end

function S_profile(ai, xi, yi, w2i, beta_t1)
  Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + ai))
  return sum(@. (yi - Pi) * w2i)
end

function updatebetamat_S(x, y, w1, w2, beta_t1_ogn, mua_t1, sigma2a_t1, interval, rtol)
  n = size(x)[1]
  mu_num = 0.0
  sigma_num = 0.0
  K = 12

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = copy(beta_t1_ogn)
  beta_t2 = zeros(2)

  cnt = 0
  while true
    beta_denom = 0.0
    beta_num = 0.0
    cnt += 1
    if cnt % 10 == 0
      interval += 1.0
    end

    for i in 1:n
    #println(i)
    #@time begin
      xi = x[i]
      yi = y[i]
      w2i = w2[i]
      m = size(xi)[1]

      res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1, m)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -15, 15)
      #res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), [0.0], LBFGS())
      mode = Optim.minimizer(res)[1]

      xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(0, v_profile(aik, xi, yi, w2i, beta_t1, m)), S_profile(aik, xi, yi, w2i, beta_t1)), K, mode-interval, mode+interval, rtol=rtol)
      denomi = sum(wquad)
      Pi_mat = Array{Float64}(undef, length(xi), K)
      for k in 1:K
        @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
      end
      beta_num += w1[i] * sum([sum(@. w2i * (yi - Pi_mat[:,k]) * xi) for k in 1:K] .* wquad) / denomi
      beta_denom += w1[i] * sum([sum(@. w2i * Pi_mat[:,k] * (1 - Pi_mat[:,k]) * xi ^ 2) for k in 1:K] .* wquad) / denomi
    #end
    end
    beta_t2[2] = beta_t1[2] + beta_num / beta_denom
    #@show beta_t2
    if abs(beta_t1[2] - beta_t2[2]) > 1e-2
      beta_t1[2] = beta_t2[2]
    else
      beta_t1[2] = beta_t2[2]
      break
    end
    if cnt > 100
      @warn(": convergence failed: profile likelihood; beta")
      beta_t1[2] = beta_t2[2]
      break
    end
  end
  #@show beta_t1

  #Threads.@threads for i in 1:n
  for i in 1:n
  #println(i)
  #@time begin
    xi = x[i]
    yi = y[i]
    w2i = w2[i]

    res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1, m)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -15, 15)
    #res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), [0.0], LBFGS())
    mode = Optim.minimizer(res)[1]
    #@show mode
    #println("S_profile", map(aik -> S_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("v_profile", map(aik -> v_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("pdf", map(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -5.0:5.0))

    xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(0, v_profile(aik, xi, yi, w2i, beta_t1, m)), S_profile(aik, xi, yi, w2i, beta_t1)), K, mode-interval, mode+interval, rtol=rtol)
    denomi = sum(wquad)
    #@show y[i]
    #@show w2[i]
    #@show w1[i] * sum(@. xquad * wquad) / denomi
    #@show w1[i] * sum(@. xquad^2 * wquad) / denomi

    mu_num += w1[i] * sum(@. xquad * wquad) / denomi
    sigma_num += w1[i] * sum(@. xquad^2 * wquad) / denomi
  #end
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end

function updatebetamat_S_fast(x, y, w1, w2, beta_t1_ogn, mua_t1, sigma2a_t1, interval, rtol, K2)
  n = size(x)[1]
  mu_num = 0.0
  sigma_num = 0.0
  K = K2

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = copy(beta_t1_ogn)
  beta_t2 = zeros(2)

  xquad, wquad = gauss(v -> pdf(Normal(mua_t1, sigmaa_t1), v), K, mua_t1- interval, mua_t1 + interval, rtol=rtol)

  cnt = 0
  while true
    beta_denom = 0.0
    beta_num = 0.0
    cnt += 1
    if cnt % 10 == 0
      interval += 1.0
    end

    for i in 1:n
    #println(i)
    #@time begin
      xi = x[i]
      yi = y[i]
      w2i = w2[i]
      m = size(xi)[1]

      denomi = sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad)
      Pi_mat = Array{Float64}(undef, length(xi), K)
      for k in 1:K
        @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
      end
      beta_num += w1[i] * sum([sum(w2i .* (yi .- Pi_mat[:,k]) .* xi .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
      beta_denom += w1[i] * sum([sum(w2i .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2.0 .* pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1))) for k in 1:K] .* wquad) / denomi
    #end
    end
    beta_t2[2] = beta_t1[2] + beta_num / beta_denom
    #@show beta_t2
    if abs(beta_t1[2] - beta_t2[2]) > 1e-5
      beta_t1[2] = beta_t2[2]
    else
      beta_t1[2] = beta_t2[2]
      break
    end
    if cnt > 100
      @warn(": convergence failed: profile likelihood; beta")
      beta_t1[2] = beta_t2[2]
      break
    end
  end
  #@show beta_t1

  #Threads.@threads for i in 1:n
  for i in 1:n
  #println(i)
  #@time begin
    xi = x[i]
    yi = y[i]
    w2i = w2[i]

    #@show mode
    #println("S_profile", map(aik -> S_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("v_profile", map(aik -> v_profile(aik[1], xi, yi, w2i, beta_t1), -5.0:5.0))
    #println("pdf", map(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(0, v_profile(aik[1], xi, yi, w2i, beta_t1)), S_profile(aik[1], xi, yi, w2i, beta_t1)), -5.0:5.0))

    denomi = sum([pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K] .* wquad)
    #@show y[i]
    #@show w2[i]
    #@show w1[i] * sum(@. xquad * wquad) / denomi
    #@show w1[i] * sum(@. xquad^2 * wquad) / denomi

    mu_num += w1[i] * sum(xquad .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / denomi
    sigma_num += w1[i] * sum(xquad .^2.0 .* wquad.* [pdf(Normal(0, v_profile(xquad[k], xi, yi, w2i, beta_t1, m)), S_profile(xquad[k], xi, yi, w2i, beta_t1)) for k in 1:K]) / denomi
  #end
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end
=#

function pij_ftn(ai, xij, beta_t1_2)
  return 1 - 1 / (1 + exp(beta_t1_2 * xij + ai))
end

function vi_ftn_S(Pi::Vector{Float64}, w2i, yi, pi2mati)
  ratioi = @. (yi - Pi) * w2i
  p2i = @. 1 / w2i
  varhat = transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi
  #return sqrt(varhat) # required?
  return  sqrt(ifelse(varhat > 0, varhat, 0.0)) # required?
  #return sqrt(sum(@. w2i^2 * Pi * (1 - Pi)))
end

function Si_ftn_S(Pi::Vector{Float64}, yi, w2i)
  return sum(@. (yi - Pi) * w2i)
end

function hi_ftn(ai, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1)
  Pi = @. 1 - 1 / (1 + exp(beta_t1_2 * xi + ai))
  #@show ai
  #@show Pi
  #@show vi_ftn_S(Pi, w2i)
  #@show Si_ftn_S(Pi, yi, w2i)
  if isnan(vi_ftn_S(Pi, w2i, yi, pi2mati))
    pdfS = 0.0
  elseif vi_ftn_S(Pi, w2i, yi, pi2mati) <= 0.0
    pdfS = 0.0
  else
    pdfS = pdf(Normal(0, vi_ftn_S(Pi, w2i, yi, pi2mati)), Si_ftn_S(Pi, yi, w2i))
  end
  if isinf(pdfS)
    #@warn("Inf generated")
    pdfS = 0.0
  end
  return pdf(Normal(mua_t1, sigmaa_t1), ai) * pdfS
end

function Epi_ftn(xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1, m)
  denom = quadgk(v -> hi_ftn(v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1]
  return [quadgk(v -> pij_ftn(v, xi[j], beta_t1_2) * hi_ftn(v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1] / denom for j in 1:m]
end

function Epi2_ftn(xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1, m)
  denom = quadgk(v -> hi_ftn(v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1]
  return [quadgk(v -> pij_ftn(v, xi[j], beta_t1_2) * (1 - pij_ftn(v, xi[j], beta_t1_2)) * hi_ftn(v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1] / denom for j in 1:m]
end

function Evis_ftn(xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1, m)
  denom = quadgk(v -> hi_ftn(v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1]
  num1 = quadgk(v -> v * hi_ftn(v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1]
  num2 = quadgk(v -> v^2 * hi_ftn(v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1]
  return num1 / denom, num2 / denom
end

function updatebetamat_S(x, y, w1, w2, pi2mat, beta_t1_ogn, mua_t1, sigma2a_t1)
  n = size(x)[1]
  mu_num = 0.0
  sigma_num = 0.0

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = copy(beta_t1_ogn)
  beta_t2 = zeros(2)
  cnt = 0

  while true
    cnt += 1
    beta_denom = 0.0
    beta_num = 0.0
    for i in 1:n
      xi = x[i]
      yi = y[i]
      w1i = w1[i]
      w2i = w2[i]
      pi2mati = pi2mat[i]
      m = size(xi)[1]
      #@show i
      beta_num += w1i * sum(w2i .* (yi .- Epi_ftn(xi, yi, w2i, pi2mati, beta_t1[2], mua_t1, sigmaa_t1, m)) .* xi)
      beta_denom += w1i * sum(w2i .* Epi2_ftn(xi, yi, w2i, pi2mati, beta_t1[2], mua_t1, sigmaa_t1, m) .* xi .^ 2)
    end
    beta_t2[2] = beta_t1[2] + beta_num / beta_denom
    if abs(beta_t1[2] - beta_t2[2]) > 1e-5
      beta_t1[2] = beta_t2[2]
    else
      beta_t1[2] = beta_t2[2]
      break
    end
    if cnt > 100
      @warn(": convergence failed: normal approximation w/ v; beta")
      beta_t1[2] = beta_t2[2]
      break
    end
  end

  #Threads.@threads for i in 1:n
  for i in 1:n
    xi = x[i]
    yi = y[i]
    w1i = w1[i]
    w2i = w2[i]
    pi2mati = pi2mat[i]
    m = size(xi)[1]
    Evi1, Evi2 = Evis_ftn(xi, yi, w2i, pi2mati, beta_t1[2], mua_t1, sigmaa_t1, m)
    mu_num += w1i * Evi1
    sigma_num += w1i * Evi2
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end


#=
function updatebetamat_S(x, y, w1, w2, beta_t1_ogn, mua_t1, sigma2a_t1)
  n = size(x)[1]
  mu_num = 0.0
  sigma_num = 0.0

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = copy(beta_t1_ogn)

  beta_t1_ogn_2 = beta_t1_ogn[2]
  function U_beta(beta_t1_2)
    Uvec = 0.0
    for i in 1:n
      Uveci = 0.0
      xi = x[i]
      yi = y[i]
      w1i = w1[i]
      w2i = w2[i]
      m = size(xi)[1]
      Uvec += w1i * sum(w2i .* (yi .- Epi_ftn(xi, yi, w2i, beta_t1_2, mua_t1, sigmaa_t1, m)) .* xi)
    end
    return Uvec
  end

  #@show U_beta(beta_t1_ogn_2)
  beta_t1[2] = fzero(U_beta, beta_t1_ogn_2 - 5.0, beta_t1_ogn_2 + 5.0)
  #@show beta_t1

  #Threads.@threads for i in 1:n
  for i in 1:n
    xi = x[i]
    yi = y[i]
    w1i = w1[i]
    w2i = w2[i]
    m = size(xi)[1]
    Evi1, Evi2 = Evis_ftn(xi, yi, w2i, beta_t1[2], mua_t1, sigmaa_t1, m)
    mu_num += w1i * Evi1
    sigma_num += w1i * Evi2
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end
=#
