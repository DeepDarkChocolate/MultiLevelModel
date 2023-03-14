function solveahat(x, y, w2, beta_t1)
  n = size(x)[1]
  a_t2 = Vector{Float64}(undef, n)
  Threads.@threads for i in 1:n
  #for i in 1:n
    ai0 = 0.0
    xi = x[i]
    yi = y[i]
    w2i = w2[i]

    a_t2[i] = sum(@. w2i * (yi - beta_t1[2] * xi)) / sum(w2i)
  end
  return a_t2
end

# solveahat(x_sampled, y_sampled, pi2_sampled, [0.0, 2.0])


# Normal approximation using vhat
function updatebetamat_new3(astar, vstar, x, y, w1, w2)

  beta_t1_scr = sum(w1 .* [sum(@. w2[i] * (y[i] - astar[i]) * x[i]) for i in 1:n]) /
  sum(w1 .* [sum(@. w2[i] * x[i] ^ 2) for i in 1:n])
  sigma2e_t1 = sum(w1 .* [sum(@. w2[i] * ( (-astar[i] + y[i] - beta_t1_scr * x[i]) ^ 2 + vstar[i] ) ) for i in 1:n]) /
  sum(w1 .* [sum(w2[i]) for i in 1:n])
  muhat = sum(@. w1 * astar) / sum(@. w1)
  sigma2a_t1 = sum(@. w1 * (astar ^ 2 + vstar)) / sum(@. w1) - muhat^2

  return beta_t1_scr, sigma2e_t1, muhat, sigma2a_t1
end
#updatebetamat_new3(astar, vstar, x_sampled, y_sampled, pi1_sampled, pi2_sampled, beta_t1, sigma2a_t1, K)

#=
function updatebetamat_new3_fast(astar, vstar, x, y, w1, w2, beta_t1_ogn, sigma2a_t1, interval, rtol, K2)
  n = size(x)[1]
  K = K2
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
      xi = x[i]
      yi = y[i]
      w2i = w2[i]
      xquad, wquad = gauss(v -> pdf(Normal(astar[i], sqrt(vstar[i])), v), K, astar[i]- interval, astar[i] + interval, rtol=rtol)
      Pi_mat = Array{Float64}(undef, length(xi), K)
      for k in 1:K
        @. Pi_mat[:,k] = 1 - 1 / (1 + exp(beta_t1[2] * xi + xquad[k]))
      end
      beta_num += w1[i] * sum([sum(w2i .* (yi .- Pi_mat[:,k]) .* xi) for k in 1:K] .* wquad)
      beta_denom += w1[i] * sum([sum(w2i .* Pi_mat[:,k] .* (1 .- Pi_mat[:,k]) .* xi .^ 2) for k in 1:K] .* wquad)
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
      @warn(": convergence failed: normal approximation w/ vhat; beta")
      beta_t1[2] = beta_t2[2]
      break
    end
  end

  return beta_t2[2]
end
=#
