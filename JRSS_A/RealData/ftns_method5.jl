function foo2(x_sampled, y_sampled, w1_sampled, beta_t1, sigma2a_t1)
  n = size(x_sampled)[1]

  function lwc(theta, x_sampled, y_sampled, w1_sampled)
    @show theta
    if (theta[8] < 0.0)
      ans = -Inf
    else
      ansvec = zeros(n)
      #xquad, wquad = gauss(v -> pdf(Normal(mua_t1, sqrt(sigma2e_t1)), v), K, mua_t1- interval, mua_t1 + interval, rtol=rtol)
      Threads.@threads for i in 1:n
        wi = w1_sampled[i]
        m = size(y_sampled[i])[1]
        anstmp = 0.0
        for j in 1:m
          for k in 1:(j-1)
            #println((j, k))
            xij = x_sampled[i][j,:]
            xik = x_sampled[i][k,:]
            yij = y_sampled[i][j]
            yik = y_sampled[i][k]
            #anstmp += wjk_i * log(sum([pdf(Normal(theta[1] * xij + vi, sqrt(theta[2])), yij) * pdf(Normal(theta[1] * xik + vi, sqrt(theta[2])), yik) for vi in xquad] .* wquad))
            anstmp += log(quadgk(vi -> ifelse(yij == 1, pij_ftn(vi, xij, beta_t1), (1 - pij_ftn(vi, xij, beta_t1))) * ifelse(yik == 1, pij_ftn(vi, xik, beta_t1), (1 - pij_ftn(vi, xik, beta_t1))) * pdf(Normal(0, sqrt(theta[8])), vi) , -Inf, Inf)[1] )
          end
        end
        ansvec[i] = wi * anstmp
      end
      ans = sum(ansvec)
    end
    return ans
  end

  #inner_optimizer = GradientDescent()
  initial = vcat(beta_t1, sigma2a_t1)

  #lower = [beta_t1[2] - 5.0, 1.0, mua_t1 - 5.0, 0.5]
  #upper = [beta_t1[2] + 5.0, sigma2e_t1 + 5.0, mua_t1 + 5.0, sigma2a_t1 + 5.0]
  res = optimize(theta -> -lwc(theta, x_sampled, y_sampled, w1_sampled), initial, NelderMead(), Optim.Options(iterations = 500))
  #res = optimize(theta -> -lwc(theta, x_sampled, y_sampled, w1_sampled), initial, NelderMead(), Optim.Options(x_tol = 1e-3, f_tol = 1e-2, g_tol = 1e-2))
  #res = optimize(theta -> -lwc(theta, x_sampled, y_sampled, w1_sampled, pi2mat_sampled, rtol), initial, NelderMead(), Optim.Options(x_tol = 1e-3, f_tol = 1e-1, g_tol = 1e-1))
  @show Optim.minimizer(res)
  return Optim.minimizer(res)
end
