function EMalg(x_sampled, y_sampled, beta_t, sigma2e_t, sigma2a_t, M, n; verbose = falses, eps_theta = 0.001)
  cnt = 0
  muis = Array{Float64}(undef, M)
  muis2 = Array{Float64}(undef, M)
  beta_t2 = Array{Float64}(undef, 2)
  while true
    #global cnt, sigma2e_t, sigma2a_t, beta_t
    cnt += 1

    vhat = [mean(@. y_sampled[i] - beta_t[1] - beta_t[2] * x_sampled[i]) for i in 1:M]
    Evi = @. vhat * n / sigma2e_t / (n / sigma2e_t + 1 / sigma2a_t)
    Vvi = 1 / (n / sigma2e_t + 1 / sigma2a_t)

    X_s = [fill(1, n * M) vcat(x_sampled...)]
    Y_s = vcat(y_sampled...)
    v_s = repeat(vhat, inner = n)
    beta_t2 = (transpose(X_s) * X_s) \ (transpose(X_s) * (Y_s - v_s))
    sigma2e_t2 = mean([mean(@. (y_sampled[i] - beta_t2[1] - x_sampled[i] * beta_t2[2] - Evi[i])^2) + Vvi for i in 1:M])
    sigma2a_t2 = mean(@. Evi^2 + Vvi)

    theta_t1 = vcat(beta_t, sigma2e_t, sigma2a_t)
    theta_t2 = vcat(beta_t2, sigma2e_t2, sigma2a_t2)

    if  cnt > 100
      @warn(": Convergence Failed(normal approximation w/ v)")
      println("simnum", simnum, ", cnt = ", cnt)
      theta_t2 = vcat(beta_t2, sigma2e_t, sigma2a_t2)
      break
    end

    if verbose == true
      println("v", theta_t2)
    end
    if norm(theta_t1 - theta_t2) < eps_theta
      vhat = [mean(@. y_sampled[i] - beta_t2[1] - beta_t2[2] * x_sampled[i]) for i in 1:M]
      Evi = @. vhat * n / sigma2e_t2 / (n / sigma2e_t2 + 1 / sigma2a_t2)
      muis = [mean(@. beta_t2[1] + x_sampled[i] * beta_t2[2] + Evi[i]) for i in 1:M]

      vhat = [mean(@. y_sampled[i] - beta0 - beta1 * x_sampled[i]) for i in 1:M]
      Evi = @. vhat * n / sigma2e / (n / sigma2e + 1 / sigma2a)
      muis2 = [mean(@. beta0 + x_sampled[i] * beta1 + Evi[i]) for i in 1:M]
      break
    else
      beta_t[1] = beta_t2[1]
      beta_t[2] = beta_t2[2]
      sigma2a_t = sigma2a_t2
      sigma2e_t = sigma2e_t2
    end
  end
  return muis, muis2
end
