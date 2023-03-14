# Normal approximation using v

function v(ai, xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)
  #Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + aihat))
  #ratioi = (yi .- Pi) .* w2i
  #ratioi = (yi .- mean(yi) .- Pi .+ mean(Pi)) .* w2i
  #ratioi = sqrt.(Pi .* (1 .- Pi)) .* w2i
  #p2i = @. 1 / w2i
  #varhat = (transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi) / sum(@. Pi * (1 - Pi) * w2i)^2
  #@show aihat
  #@show varhat

  Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + ai))
  ratioi = (yi .- Pi) .* w2i
  #ratioi = (yi .- mean(yi) .- Pi .+ mean(Pi)) .* w2i
  #ratioi = sqrt.(Pi .* (1 .- Pi)) .* w2i
  p2i = @. 1 / w2i
  varhat = (transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi) / sum(@. Pi * (1 - Pi) * w2i)^2
  #@show Pi
  #@show ai
  #@show varhat
  #varhat = (transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi + sum(@. w2i * Pi * (1 - Pi))) / sum(@. Pi * (1 - Pi) * w2i)^2
  #varhat = sum(@. w2i^2 * Pi * (1 - Pi)) / sum(@. Pi * (1 - Pi) * w2i)^2
  #if isnan(varhat) #for aihat
  #  varhat = sigma2a_t1
  #end
  return sqrt(varhat)
  #return sqrt(ifelse(varhat > 0, varhat, 0.000001)) # required?
end

function v2(ai, xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)
  #Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + aihat))
  #ratioi = (yi .- Pi) .* w2i
  #ratioi = (yi .- mean(yi) .- Pi .+ mean(Pi)) .* w2i
  #ratioi = sqrt.(Pi .* (1 .- Pi)) .* w2i
  #p2i = @. 1 / w2i
  #varhat = (transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi) / sum(@. Pi * (1 - Pi) * w2i)^2
  #@show aihat
  #@show varhat

  Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + aihat))
  ratioi = (yi .- Pi) .* w2i
  #ratioi = (yi .- mean(yi) .- Pi .+ mean(Pi)) .* w2i
  #ratioi = sqrt.(Pi .* (1 .- Pi)) .* w2i
  p2i = @. 1 / w2i
  varhat = (transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi) / sum(@. Pi * (1 - Pi) * w2i)^2
  #@show Pi
  #@show ai
  #@show varhat
  #varhat = (transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi + sum(@. w2i * Pi * (1 - Pi))) / sum(@. Pi * (1 - Pi) * w2i)^2
  #varhat = sum(@. w2i^2 * Pi * (1 - Pi)) / sum(@. Pi * (1 - Pi) * w2i)^2
  if isnan(varhat) #for aihat
    varhat = sigma2a_t1
  end
  return sqrt(varhat)
  #return sqrt(ifelse(varhat > 0, varhat, 0.000001)) # required?
end

#=
function v(ai, xi, yi, w2i, beta_t1)
  Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + ai))
  return sqrt(sum(@. (w2i - 1) * w2i * (yi - Pi)^2) / sum(@. Pi * (1 - Pi) * w2i)^2)
end
=#
#=
function v(ai, xi, yi, w2i, beta_t1)
  Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + ai))
  return sqrt(1 / sum(@. Pi * (1 - Pi)))
end
=#

#=
function updatebetamat(ahat, x, y, w1, w2, beta_t1, mua_t1, sigma2a_t1, interval, rtol)
  n = size(x)[1]
  beta_denom = 0.0
  beta_num = 0.0
  mu_num = 0.0
  sigma_num = 0.0
  K = 20

  sigmaa_t1 = sqrt(sigma2a_t1)
  #Threads.@threads for i in 1:n
  for i in 1:n
  #println(i)
  #@time begin
    xi = x[i]
    yi = y[i]
    w2i = w2[i]
    aihat = ahat[i]

    if isinf(aihat)
      #println("inf!, i = ", i)
      res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) / v(aik[1], xi, yi, w2i, beta_t1), -15, 15)
      mode = Optim.minimizer(res)[1]
      xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) / v(aik, xi, yi, w2i, beta_t1), K, mua_t1-interval, mua_t1+interval, rtol=rtol)
    else
      res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(aik[1], v(aik[1], xi, yi, w2i, beta_t1)), aihat), -15, 15)
      mode = Optim.minimizer(res)[1]
      xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(aik, v(aik, xi, yi, w2i, beta_t1)), aihat), K, mode-interval, mode+interval, rtol=rtol)
    end
    denomi = sum(wquad)
    Pi_mat = Array{Float64}(undef, length(xi), K)
    for k in 1:K
      @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
    end
    beta_num += w1[i] * sum([sum(@. w2i * (yi - Pi_mat[:,k]) * xi) for k in 1:K] .* wquad) / denomi
    beta_denom += w1[i] * sum([sum(@. w2i * Pi_mat[:,k] * (1 - Pi_mat[:,k]) * xi ^ 2) for k in 1:K] .* wquad) / denomi

    mu_num += w1[i] * sum(@. xquad * wquad) / denomi
    sigma_num += w1[i] * sum(@. xquad^2 * wquad) / denomi
  #end
  end
  muhat = mu_num / sum(w1)

  return beta_num / beta_denom, muhat, sigma_num / sum(w1) - muhat^2
end
=#

function updatebetamat(ahat, x, y, w1, w2, beta_t1_ogn, mua_t1, sigma2a_t1, interval, rtol, rtol_beta)
  n = size(x)[1]
  mu_num = 0.0
  sigma_num = 0.0
  K = 20

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = copy(beta_t1_ogn)
  beta_t2 = zeros(2)
  cnt = 0
  while true
    cnt += 1
    if cnt % 10 == 0
      interval += 1.0
    end
    beta_denom = 0.0
    beta_num = 0.0
    for i in 1:n
    #println(i)
    #@time begin
      xi = x[i]
      yi = y[i]
      w2i = w2[i]
      #w2_tildei = size(x[i])[1] .* w2i ./sum(w2i)
      aihat = ahat[i]
      m = size(xi)[1]

      if isinf(aihat)
        #println("inf!, i = ", i)
        res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) / v(aik[1], xi, yi, w2i, beta_t1, m), mua_t1 - 15, mua_t1 + 15)

        mode = Optim.minimizer(res)[1]
        xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) / v(aik, xi, yi, w2i, beta_t1, m), K, mua_t1-interval, mua_t1+interval, rtol=rtol)
      else
        res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(aik[1], v(aik[1], xi, yi, w2i, beta_t1, m)), aihat), mua_t1 -15, mua_t1 + 15)
        mode = Optim.minimizer(res)[1]
        xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(aik, v(aik, xi, yi, w2i, beta_t1, m)), aihat), K, mode-interval, mode+interval, rtol=rtol)
      end
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
    if abs(beta_t1[2] - beta_t2[2]) > rtol_beta
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
  #println(i)
  #@time begin
    xi = x[i]
    yi = y[i]
    w2i = w2[i]
    aihat = ahat[i]

    if isinf(aihat)
      #println("inf!, i = ", i)
      res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) / v(aik[1], xi, yi, w2i, beta_t1, m), -15, 15)
      mode = Optim.minimizer(res)[1]
      xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) / v(aik, xi, yi, w2i, beta_t1, m), K, mua_t1-interval, mua_t1+interval, rtol=rtol)
    else
      res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(aik[1], v(aik[1], xi, yi, w2i, beta_t1, m)), aihat), -15, 15)
      mode = Optim.minimizer(res)[1]
      xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(aik, v(aik, xi, yi, w2i, beta_t1, m)), aihat), K, mode-interval, mode+interval, rtol=rtol)
    end
    denomi = sum(wquad)

    mu_num += w1[i] * sum(@. xquad * wquad) / denomi
    sigma_num += w1[i] * sum(@. xquad^2 * wquad) / denomi
  #end
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end

function updatebetamat_fast(ahat, x, y, w1, w2, pi2mat, beta_t1_ogn, mua_t1, sigma2a_t1, interval, rtol, K2)
  n = size(x)[1]
  mu_num = 0.0
  sigma_num = 0.0
  K = K2

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = copy(beta_t1_ogn)
  beta_t2 = zeros(2)
  cnt = 0

  xquad, wquad = gauss(v -> pdf(Normal(mua_t1, sigmaa_t1), v), K, mua_t1- interval, mua_t1 + interval, rtol=rtol)
  while true
    cnt += 1
    if cnt % 10 == 0
      interval += 1.0
    end
    beta_denom = 0.0
    beta_num = 0.0
    for i in 1:n
    #println(i)
    #@time begin
      xi = x[i]
      yi = y[i]
      w2i = w2[i]
      aihat = ahat[i]
      pi2mati = pi2mat[i]
      m = size(xi)[1]

      if isinf(aihat)
        denomi = sum([1 / v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat) for k in 1:K] .* wquad)
        Pi_mat = Array{Float64}(undef, length(xi), K)
        for k in 1:K
          @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
        end
        beta_num += w1[i] * sum([sum(w2i .* (yi .- Pi_mat[:,k]) .* xi ./ v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)) for k in 1:K] .* wquad) / denomi
        beta_denom += w1[i] * sum([sum(w2i .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2 ./ v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)) for k in 1:K] .* wquad) / denomi

        #println("inf!, i = ", i)
        #xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) / v(aik, xi, yi, w2i, beta_t1, m), K, mua_t1-interval, mua_t1+interval, rtol=rtol)
      else
        denomi = sum([pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)), aihat) for k in 1:K] .* wquad)
        Pi_mat = Array{Float64}(undef, length(xi), K)
        for k in 1:K
          @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
        end
        beta_num += w1[i] * sum([sum(w2i .* (yi .- Pi_mat[:,k]) .* xi .* pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)), aihat)) for k in 1:K] .* wquad) / denomi
        beta_denom += w1[i] * sum([sum(w2i .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2 .* pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)), aihat)) for k in 1:K] .* wquad) / denomi

        #xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(aik, v(aik, xi, yi, w2i, beta_t1, m)), aihat), K, mode-interval, mode+interval, rtol=rtol)
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
      @warn(": convergence failed: normal approximation w/ v; beta")
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
    aihat = ahat[i]

    if isinf(aihat)
      denomi = sum([1 / v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat) for k in 1:K] .* wquad)

      mu_num += w1[i] * sum(xquad .* wquad .* [1 / v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat) for k in 1:K]) / denomi
      sigma_num += w1[i] * sum(xquad .^ 2 .* wquad .* [1 / v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat) for k in 1:K]) / denomi
    else
      denomi = sum([pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)), aihat) for k in 1:K] .* wquad)
      #@show i
      #println(sum([pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)), aihat) for k in 1:K] .* wquad))
      #println(sum([pdf(Normal(xquad[k], v2(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)), aihat) for k in 1:K] .* wquad))
      #println(w1[i] * sum(xquad .^ 2 .* wquad .* [pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)), aihat) for k in 1:K]) / sum([pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)), aihat) for k in 1:K] .* wquad))
      #println(w1[i] * sum(xquad .^ 2 .* wquad .* [pdf(Normal(xquad[k], v2(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)), aihat) for k in 1:K]) / sum([pdf(Normal(xquad[k], v2(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)), aihat) for k in 1:K] .* wquad))

      mu_num += w1[i] * sum(xquad .* wquad .* [pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)), aihat) for k in 1:K]) / denomi
      sigma_num += w1[i] * sum(xquad .^ 2 .* wquad .* [pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, pi2mati, sigma2a_t1, aihat)), aihat) for k in 1:K]) / denomi
    end
  #end
  end
  muhat = mu_num / sum(w1)
  #@show muhat
  #@show sigma_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end

#=
function v(ai, xi, yi, w2i, beta_t1, pi2mati)
  Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + ai))
  ratioi = @. (yi - Pi) * w2i
  p2i = @. 1 / w2i
  varhat = transpose(ratioi) * (1 .- (p2i * transpose(p2i) ./ pi2mati)) * ratioi / sum(@. Pi * (1 - Pi) * w2i)^2
  return sqrt(ifelse(varhat > 0, varhat, 0.01))
end

function updatebetamat(ahat, x, y, w1, w2, pi2mat, beta_t1_ogn, mua_t1, sigma2a_t1, interval, rtol)
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
      aihat = ahat[i]
      pi2mati = pi2mat[i]

      if isinf(aihat)
        #println("inf!, i = ", i)
        res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) / v(aik[1], xi, yi, w2i, beta_t1, pi2mati), -15, 15)

        mode = Optim.minimizer(res)[1]
        xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) / v(aik, xi, yi, w2i, beta_t1, pi2mati), K, mua_t1-interval, mua_t1+interval, rtol=rtol)
      else
        res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(aik[1], v(aik[1], xi, yi, w2i, beta_t1, pi2mati)), aihat), -15, 15)
        mode = Optim.minimizer(res)[1]
        xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(aik, v(aik, xi, yi, w2i, beta_t1, pi2mati)), aihat), K, mode-interval, mode+interval, rtol=rtol)
      end
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
    aihat = ahat[i]
    pi2mati = pi2mat[i]

    if isinf(aihat)
      #println("inf!, i = ", i)
      res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) / v(aik[1], xi, yi, w2i, beta_t1, pi2mati), -15, 15)
      mode = Optim.minimizer(res)[1]
      xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) / v(aik, xi, yi, w2i, beta_t1, pi2mati), K, mua_t1-interval, mua_t1+interval, rtol=rtol)
    else
      res = optimize(aik -> -pdf(Normal(mua_t1, sigmaa_t1), aik[1]) * pdf(Normal(aik[1], v(aik[1], xi, yi, w2i, beta_t1, pi2mati)), aihat), -15, 15)
      mode = Optim.minimizer(res)[1]
      xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(aik, v(aik, xi, yi, w2i, beta_t1, pi2mati)), aihat), K, mode-interval, mode+interval, rtol=rtol)
    end
    denomi = sum(wquad)

    mu_num += w1[i] * sum(@. xquad * wquad) / denomi
    sigma_num += w1[i] * sum(@. xquad^2 * wquad) / denomi
  #end
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end
=#

#=
function v(ai, xi, yi, w2i, beta_t1, m)
  Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + ai))
  return sqrt(sum(@. w2i^2 * Pi * (1 - Pi)) / sum(@. Pi * (1 - Pi) * w2i)^2)
  #return sqrt(sum(@. w2i^2 * Pi * (1 - Pi)) / sum(@. Pi * (1 - Pi) * w2i)^2 * m / (m - 1))
end

function updatebetamat_fast(ahat, x, y, w1, w2, beta_t1_ogn, mua_t1, sigma2a_t1, interval, rtol, K2)
  n = size(x)[1]
  mu_num = 0.0
  sigma_num = 0.0
  K = K2

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = copy(beta_t1_ogn)
  beta_t2 = zeros(2)
  cnt = 0

  xquad, wquad = gauss(v -> pdf(Normal(mua_t1, sigmaa_t1), v), K, mua_t1- interval, mua_t1 + interval, rtol=rtol)
  while true
    cnt += 1
    if cnt % 10 == 0
      interval += 1.0
    end
    beta_denom = 0.0
    beta_num = 0.0
    for i in 1:n
    #println(i)
    #@time begin
      xi = x[i]
      yi = y[i]
      w2i = w2[i]
      aihat = ahat[i]
      m = size(xi)[1]

      if isinf(aihat)
        denomi = sum([1 / v(xquad[k], xi, yi, w2i, beta_t1, m) for k in 1:K] .* wquad)
        Pi_mat = Array{Float64}(undef, length(xi), K)
        for k in 1:K
          @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
        end
        beta_num += w1[i] * sum([sum(w2i .* (yi .- Pi_mat[:,k]) .* xi ./ v(xquad[k], xi, yi, w2i, beta_t1, m)) for k in 1:K] .* wquad) / denomi
        beta_denom += w1[i] * sum([sum(w2i .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2 ./ v(xquad[k], xi, yi, w2i, beta_t1, m)) for k in 1:K] .* wquad) / denomi

        #println("inf!, i = ", i)
        #xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) / v(aik, xi, yi, w2i, beta_t1, m), K, mua_t1-interval, mua_t1+interval, rtol=rtol)
      else
        denomi = sum([pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, m)), aihat) for k in 1:K] .* wquad)
        Pi_mat = Array{Float64}(undef, length(xi), K)
        for k in 1:K
          @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
        end
        beta_num += w1[i] * sum([sum(w2i .* (yi .- Pi_mat[:,k]) .* xi .* pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, m)), aihat)) for k in 1:K] .* wquad) / denomi
        beta_denom += w1[i] * sum([sum(w2i .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2 .* pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, m)), aihat)) for k in 1:K] .* wquad) / denomi

        #xquad, wquad = gauss(aik -> pdf(Normal(mua_t1, sigmaa_t1), aik) * pdf(Normal(aik, v(aik, xi, yi, w2i, beta_t1, m)), aihat), K, mode-interval, mode+interval, rtol=rtol)
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
      @warn(": convergence failed: normal approximation w/ v; beta")
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
    aihat = ahat[i]

    if isinf(aihat)
      denomi = sum([1 / v(xquad[k], xi, yi, w2i, beta_t1, m) for k in 1:K] .* wquad)

      mu_num += w1[i] * sum(xquad .* wquad .* [1 / v(xquad[k], xi, yi, w2i, beta_t1, m) for k in 1:K]) / denomi
      sigma_num += w1[i] * sum(xquad .^ 2 .* wquad .* [1 / v(xquad[k], xi, yi, w2i, beta_t1, m) for k in 1:K]) / denomi
    else
      denomi = sum([pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, m)), aihat) for k in 1:K] .* wquad)

      mu_num += w1[i] * sum(xquad .* wquad .* [pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, m)), aihat) for k in 1:K]) / denomi
      sigma_num += w1[i] * sum(xquad .^ 2 .* wquad .* [pdf(Normal(xquad[k], v(xquad[k], xi, yi, w2i, beta_t1, m)), aihat) for k in 1:K]) / denomi
    end
  #end
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end
=#


function pij_ftn(ai, xij, beta_t1_2)
  return 1 - 1 / (1 + exp(sum(beta_t1_2 .* xij) + ai))
end

function vi_ftn_v(Pi::Vector{Float64}, w2i, yi, mi)
  return sqrt(w2i * (w2i - 1) * var(yi .- Pi) * mi / sum(@. Pi * (1 - Pi) * w2i)^2)
end

function hi_ftn(aihat, ai, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi)
  Pi = [1 - 1 / (1 + exp(sum(beta_t1_2 .* xi[j,:]) + ai)) for j in 1:mi]
  #@show ai
  #@show Pi
  #@show vi_ftn_v(Pi, w2i)
  if isnan(vi_ftn_v(Pi, w2i, yi, mi))
    pdfS = 0.0
  elseif isinf(aihat)
    pdfS = 1 / vi_ftn_v(Pi, w2i, yi, mi)
  else
    pdfS = pdf(Normal(ai, vi_ftn_v(Pi, w2i, yi, mi)), aihat)
  end
  if isinf(pdfS)
    #@warn("Inf generated")
    pdfS = 0.0
  end
  return pdf(Normal(0.0, sigmaa_t1), ai) * pdfS
end

function Epi_ftn(aihat, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi)
  denom = quadgk(v -> hi_ftn(aihat, v, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi), -Inf, Inf)[1]
  pi1 = [quadgk(v -> pij_ftn(v, xi[j,:], beta_t1_2) * hi_ftn(aihat, v, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi), -Inf, Inf)[1] / denom for j in 1:mi]
  pi2 = [quadgk(v -> pij_ftn(v, xi[j,:], beta_t1_2) * (1 - pij_ftn(v, xi[j,:], beta_t1_2)) * hi_ftn(aihat, v, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi), -Inf, Inf)[1] / denom for j in 1:mi]
  return pi1, pi2
end

#function Epi2_ftn(aihat, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1, m)
#  denom = quadgk(v -> hi_ftn(aihat, v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1]
#  return [quadgk(v -> pij_ftn(v, xi[j], beta_t1_2) * (1 - pij_ftn(v, xi[j], beta_t1_2)) * hi_ftn(aihat, v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1] / denom for j in 1:m]
#end

function Evis_ftn(aihat, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi)
  denom = quadgk(v -> hi_ftn(aihat, v, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi), -Inf, Inf)[1]
  #num1 = quadgk(v -> v * hi_ftn(aihat, v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1]
  num2 = quadgk(v -> v^2 * hi_ftn(aihat, v, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi), -Inf, Inf)[1]
  #return num1 / denom, num2 / denom
  return num2 / denom
end

function updatebetamat(ahat, x, y, w1, w2, beta_t1_ogn, sigma2a_t1)
  n = size(x)[1]
  sigma_num = 0.0

  sigmaa_t1 = sqrt(sigma2a_t1)
  beta_t1 = copy(beta_t1_ogn)
  p = length(beta_t1_ogn)
  beta_t2 = zeros(p)
  cnt = 0

  while true
    cnt += 1
    beta_denom = zeros(p, p)
    beta_num = zeros(p)
    for i in 1:n
      xi = x[i]
      yi = y[i]
      w1i = w1[i]
      w2i = w2[i]
      aihat = ahat[i]
      mi = size(xi)[1]
      #@show i
      Epi1, Epi2 = Epi_ftn(aihat, xi, yi, w2i, beta_t1, sigmaa_t1, mi)
      beta_num += w1i * sum([w2i .* (yi[j] .- Epi1[j]) .* xi[j,:] for j in 1:mi])
      beta_denom += w1i * sum([w2i .* Epi2[j] .* (xi[j,:] * transpose(xi[j,:])) for j in 1:mi])
    end
    beta_t2 = beta_t1 .+ beta_denom \ beta_num
    @show beta_t2
    if norm(beta_t1 .- beta_t2) > 1e-5
      beta_t1 = beta_t2
    else
      beta_t1 = beta_t2
      break
    end
    if cnt > 100
      @warn(": convergence failed: normal approximation w/ v; beta")
      beta_t1 = beta_t2
      break
    end
  end


  #@show U_beta.([1:0.1:5;])
  #@show U_beta(beta_t1_ogn_2)
  #error("done")
  #@show beta_t1

  #Threads.@threads for i in 1:n
  for i in 1:n
    xi = x[i]
    yi = y[i]
    w1i = w1[i]
    w2i = w2[i]
    aihat = ahat[i]
    mi = size(xi)[1]
    Evi2 = Evis_ftn(aihat, xi, yi, w2i, beta_t1, sigmaa_t1, mi)
    #Evi1, Evi2 = Evis_ftn(aihat, xi, yi, w2i, pi2mati, beta_t1[2], mua_t1, sigmaa_t1, m)
    #mu_num += w1i * Evi1
    sigma_num += w1i * Evi2
  end

  return vcat(beta_t1,  sigma_num / sum(w1))
end


#=
function updatebetamat(ahat, x, y, w1, w2, beta_t1_ogn, mua_t1, sigma2a_t1)
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
      aihat = ahat[i]
      m = size(xi)[1]
      #@show i
      #@show w1i * sum(w2i .* (yi .- Epi_ftn(aihat, xi, yi, w2i, beta_t1_2, mua_t1, sigmaa_t1, m)) .* xi)
      Uvec += w1i * sum(w2i .* (yi .- Epi_ftn(aihat, xi, yi, w2i, beta_t1_2, mua_t1, sigmaa_t1, m)) .* xi)
    end
    return Uvec
  end

  #@show U_beta.([1:0.1:5;])
  #@show U_beta(beta_t1_ogn_2)
  #error("done")
  beta_t1[2] = fzero(U_beta, beta_t1_ogn_2 - 5.0, beta_t1_ogn_2 + 5.0)
  #@show beta_t1

  #Threads.@threads for i in 1:n
  for i in 1:n
    xi = x[i]
    yi = y[i]
    w1i = w1[i]
    w2i = w2[i]
    aihat = ahat[i]
    m = size(xi)[1]
    Evi1, Evi2 = Evis_ftn(aihat, xi, yi, w2i, beta_t1[2], mua_t1, sigmaa_t1, m)
    mu_num += w1i * Evi1
    sigma_num += w1i * Evi2
  end
  muhat = mu_num / sum(w1)

  return beta_t1[2], muhat, sigma_num / sum(w1) - muhat^2
end
=#
