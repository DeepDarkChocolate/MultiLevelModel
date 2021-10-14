pivi(beta_t, xi, vi) = 1 .- 1 ./ (1 .+ exp.(xi * beta_t .+ vi))

function pdfvi(vi, xi, yi, w2i, beta_t, sigma2a_t)
  pis = pivi(beta_t, xi, vi)
  #@show pdf(Normal(sum(@. yi - pis), sqrt(sum(@. pis * (1 - pis)))), 0.0) * pdf(Normal(0, sqrt(sigma2a_t)), vi)
  #@show vi
  #@show pis
  #var_tmp = ifelse(sum(@. pis * (1 - pis) * w2i^2) == 0.0, eps(), sum(@. pis * (1 - pis) * w2i^2))
  var_tmp = ifelse(sum(@. (yi - pis)^2 * w2i * (w2i - 1)) == 0.0, eps(), sum(@. (yi - pis)^2 * w2i * (w2i - 1)))
  return pdf(Normal(sum(@. (yi - pis) * w2i), sqrt(var_tmp)), 0.0) * pdf(Normal(0, sqrt(sigma2a_t)), vi)
end

function EMalg(x_sampled, y_sampled, beta_t, sigma2a_t, w1, w2, M; verbose = false, eps_theta = 0.01)
  p = size(beta_t)[1]
  cnt = 0
  #muis = Array{Float64}(undef, M)
  #muis2 = Array{Float64}(undef, M)
  theta_t1 = Array{Float64}(undef, p + 1)
  theta_t2 = Array{Float64}(undef, p +1)
  Svec = Array{Float64}(undef, p)
  Umat = Array{Float64}(undef, p, p)
  while true
    #global muis, muis2, cnt
    cnt += 1
    beta_loop = copy(beta_t)

    while true
      denom = [quadgk(vi -> w1[i] * pdfvi(vi, x_sampled[i], y_sampled[i], w2[i], beta_loop, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
      for j in 1:p
        S = [quadgk(vi -> w1[i] * sum(w2[i] .* (y_sampled[i] .- pivi(beta_loop, x_sampled[i], vi)) .* x_sampled[i][:, j]) * pdfvi(vi, x_sampled[i], y_sampled[i], w2[i], beta_loop, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
        S = @. S / denom
        Svec[j] = sum(S)
      end

      for j in 1:p
        for k in 1:p
          if j <= k
          U = [quadgk(vi -> w1[i] * sum(w2[i] .* pivi(beta_loop, x_sampled[i], vi) .* (1 .- pivi(beta_loop, x_sampled[i], vi)) .* x_sampled[i][:,j] .* x_sampled[i][:,k]) * pdfvi(vi, x_sampled[i], y_sampled[i], w2[i], beta_loop, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
          U = @. U / denom
          Umat[j, k] = sum(U)
          Umat[k, j] = sum(U)
          end
        end
      end

      beta_loop2 = beta_loop .+ (Umat \ Svec)
      if norm(beta_loop2 .- beta_loop) > eps_theta
        beta_loop = copy(beta_loop2)
        if verbose == true
          println("v", beta_loop2)
        end
      else
        break
      end
    end
    beta_t2 = copy(beta_loop)

    denom = [quadgk(vi -> w1[i] * pdfvi(vi, x_sampled[i], y_sampled[i], w2[i], beta_t2, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
    S1 = [quadgk(vi -> w1[i] * vi^2 * pdfvi(vi, x_sampled[i], y_sampled[i], w2[i], beta_t2, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
    S1 = @. S1 / denom
    sigma2a_t2 = mean(S1)

    theta_t1 = vcat(beta_t, sigma2a_t)
    theta_t2 = vcat(beta_t2, sigma2a_t2)

    if  cnt > 100
      @warn(": Convergence Failed(normal approximation w/ v)")
      println("simnum", simnum, ", cnt = ", cnt)
      theta_t2 = vcat(beta_t2, sigma2a_t2)
      break
    end

    if verbose == true
      println("v", theta_t2)
    end
    if norm(theta_t1 - theta_t2) < eps_theta
      #denom = [quadgk(vi -> pdfvi(vi, x_sampled[i], y_sampled[i], w2[i], beta_t2, sigma2a_t2), -Inf, Inf)[1] for i in 1:M]
      #muis = [quadgk(vi -> sum(pivi(beta_t2, x_sampled[i], vi)) * pdfvi(vi, x_sampled[i], y_sampled[i], w2[i], beta_t2, sigma2a_t2), -Inf, Inf)[1] for i in 1:M]
      #muis = @. muis / denom
      break
    else
      beta_t = copy(beta_t2)
      sigma2a_t = sigma2a_t2
    end
  end
  return theta_t2
end

function bootstrap(X_sampled, beta_t, sigma2a_t, w1_sampled, w2_sampled)
  M = size(X_sampled)[1]
  a = rand(Normal(0,sqrt(sigma2a_t)),M)
  p = [1 .- 1 ./ (1 .+ exp.(X_sampled[i] * beta_t .+ a[i])) for i in 1:M]
  Y_sampled = [ifelse.(rand(size(p[i])[1]) .< p[i], 1, 0) for i in 1:M]
  #y = [vcat((@. rand(Binomial(1, p[i]), 1)...)) for i in 1:length(M)]

  beta_t2 = copy(beta_t)
  sigma2a_t2 = sigma2a_t

  theta_t2 = EMalg(X_sampled, Y_sampled, beta_t2, sigma2a_t2, w1_sampled, w2_sampled, M, verbose = false)
  return theta_t2
end
