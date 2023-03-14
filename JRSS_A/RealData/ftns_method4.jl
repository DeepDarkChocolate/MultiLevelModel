function li_ftn(vi, xi, yi, beta_t1, mi)
  Pi = [1 - 1 / (1 + exp(sum(beta_t1 .* xi[j,:]) + vi)) for j in 1:mi]
  vec = @. ifelse(yi == 1, Pi, (1 - Pi))
  return sum(log.(vec))
  #return sum(@. w2i * (- log(sigma2e_t1) / 2 - 1 / sigma2e_t1 / 2 * (yi - beta_t1 * xi - ai)^2))
end

function foo(x_sampled, y_sampled, w1_sampled, beta_t1, sigma2a_t1)
  n = size(x_sampled)[1]

  function lp(theta, x_sampled, y_sampled, w1_sampled)
    @show theta
    if (theta[8] < 0.0)
      ans = -Inf
    else
      ans = sum(w1_sampled .* [log( quadgk(vi -> exp(li_ftn(vi, x_sampled[i], y_sampled[i], theta[1:7], length(y_sampled[i]))) * pdf(Normal(0, sqrt(theta[8])), vi), -Inf, Inf)[1] ) for i in 1:n])
    end
    return ans
  end

  #inner_optimizer = GradientDescent()
  initial = vcat(beta_t1, sigma2a_t1)
  #lower = [beta_t1[2] - 5.0, 1.0, mua_t1 - 5.0, 0.5]
  #upper = [beta_t1[2] + 5.0, sigma2e_t1 + 5.0, mua_t1 + 5.0, sigma2a_t1 + 5.0]
  res = optimize(theta -> -lp(theta, x_sampled, y_sampled, w1_sampled), initial, NelderMead(), Optim.Options(iterations = 500))
  #res = optimize(theta -> -lp(theta, x_sampled, y_sampled, w1_sampled), initial, NelderMead(), Optim.Options(x_tol = 1e-2, f_tol = 1e-1, g_tol = 1e-1))

  return Optim.minimizer(res)
end
