function li_ftn(ai, xi, yi, w2i, beta_t1_2, sigma2e_t1)
  return sum(@. w2i * logpdf(Normal(beta_t1_2 * xi + ai, sqrt(sigma2e_t1)), yi))
  #return sum(@. w2i * (- log(sigma2e_t1) / 2 - 1 / sigma2e_t1 / 2 * (yi - beta_t1_2 * xi - ai)^2))
end

function wi_tilde(w2i, m)::Vector{Float64}
  return m .* w2i ./ sum(w2i)
end

function foo(x_sampled, y_sampled, w1_sampled, w2_sampled, beta_t1, sigma2e_t1, mua_t1, sigma2a_t1, rtol)
  n = size(x_sampled)[1]

  w2_tilde_sampled = [wi_tilde(w2_sampled[i], length(w2_sampled[i])) for i in 1:n]

  function lp(theta, x_sampled, y_sampled, w1_sampled, w2_tilde_sampled)
    if (theta[4] < 0.0) | (theta[2] < 0.0)
      ans = -Inf
    else
      ans = sum(w1_sampled .* [log( quadgk(vi -> exp(li_ftn(vi, x_sampled[i], y_sampled[i], w2_tilde_sampled[i], theta[1], theta[2])) * pdf(Normal(theta[3], sqrt(theta[4])), vi), -Inf, Inf, rtol=rtol)[1] ) for i in 1:n])
    end
    return ans
  end

  #inner_optimizer = GradientDescent()
  initial = [beta_t1[2], sigma2e_t1, mua_t1, sigma2a_t1]
  #lower = [beta_t1[2] - 5.0, 1.0, mua_t1 - 5.0, 0.5]
  #upper = [beta_t1[2] + 5.0, sigma2e_t1 + 5.0, mua_t1 + 5.0, sigma2a_t1 + 5.0]

  res = optimize(theta -> -lp(theta, x_sampled, y_sampled, w1_sampled, w2_tilde_sampled), initial, NelderMead())

  return Optim.minimizer(res)
end
