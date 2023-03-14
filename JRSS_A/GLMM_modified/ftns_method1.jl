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

# solveahat(x_sampled, y_sampled, pi2_sampled, [0.0, 2.0])


# Normal approximation using vhat
function updatebetamat_new3(astar, vstar, x, y, w1, w2, beta_t1, sigma2a_t1, K)
  n = size(x)[1]
  spres = zeros(2)

  for cnt0 in 1:K
    ak = [rand(Normal(astar[i],sqrt(vstar[i])),1)[1] for i in 1:n]

    pk = [@. 1 - 1 / (1 + exp(beta_t1[2] * x[i] + ak[i])) for i in 1:n]
    tmp_22 = sum(w1 .* [sum(@. w2[i] * pk[i] * (1 - pk[i]) * x[i] ^ 2) for i in 1:n])
    tmp_23 = sum(w1 .* [sum(@. w2[i] * x[i] * (y[i] - pk[i])) for i in 1:n])

    spres = spres + [tmp_22, tmp_23]
  end
  return spres[2] / spres[1]
end
#updatebetamat_new3(astar, vstar, x_sampled, y_sampled, pi1_sampled, pi2_sampled, beta_t1, sigma2a_t1, K)


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
