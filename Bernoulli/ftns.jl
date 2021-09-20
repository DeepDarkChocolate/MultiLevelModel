function solveahat(x, y, beta_t1, sigma2a_t1; mua_t1 = 0.0)
  n = size(x)[1]
  a_t2 = Vector{Float64}(undef, n)
  sigmaa_t1 = sqrt(sigma2a_t1)
  #Threads.@threads for i in 1:n
  for i in 1:n
    #@show i
    ai0 = mua_t1
    xi = x[i]
    yi = y[i]
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
        etai = @. beta_t1[1] + beta_t1[2] * xi + ai_t1
        Pi = @. 1 - 1 / (1 + exp(etai))
        ai_t2 = ai_t1 + sum(@. (yi - Pi)) ./ sum(@. Pi * (1 - Pi))
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
      elseif abs(ai_t1 - ai_t2) < 10^(-7)
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

pivi(beta_t, xi, vi) = @. 1 - 1 / (1 + exp(beta_t[1] + beta_t[2] * xi + vi))

function pdfvi(vi, xi, yi, beta_t, sigma2a_t)
  pis = pivi(beta_t, xi, vi)
  #@show pdf(Normal(sum(@. yi - pis), sqrt(sum(@. pis * (1 - pis)))), 0.0) * pdf(Normal(0, sqrt(sigma2a_t)), vi)
  #@show vi
  #@show pis
  var_tmp = ifelse(sum(@. pis * (1 - pis)) == 0.0, eps(), sum(@. pis * (1 - pis)))
  return pdf(Normal(sum(@. yi - pis), sqrt(var_tmp)), 0.0) * pdf(Normal(0, sqrt(sigma2a_t)), vi)
end

function EMalg(x_sampled, y_sampled, beta_t, sigma2a_t, M, n; verbose = falses, eps_theta = 0.001)
  cnt = 0
  muis = Array{Float64}(undef, M)
  muis2 = Array{Float64}(undef, M)
  while true
    #global muis, muis2, cnt
    cnt += 1
    beta_loop = copy(beta_t)

    while true
      denom = [quadgk(vi -> pdfvi(vi, x_sampled[i], y_sampled[i], beta_loop, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
      S1 = [quadgk(vi -> sum(y_sampled[i] .- pivi(beta_loop, x_sampled[i], vi)) * pdfvi(vi, x_sampled[i], y_sampled[i], beta_loop, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
      S1 = @. S1 / denom
      S2 = [quadgk(vi -> sum((y_sampled[i] .- pivi(beta_loop, x_sampled[i], vi)) .* x_sampled[i]) * pdfvi(vi, x_sampled[i], y_sampled[i], beta_loop, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
      S2 = @. S2 / denom

      U11 = [quadgk(vi -> sum(pivi(beta_loop, x_sampled[i], vi) .* (1 .- pivi(beta_loop, x_sampled[i], vi))) * pdfvi(vi, x_sampled[i], y_sampled[i], beta_loop, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
      U11 = @. U11 / denom
      U12 = [quadgk(vi -> sum(pivi(beta_loop, x_sampled[i], vi) .* (1 .- pivi(beta_loop, x_sampled[i], vi)) .* x_sampled[i]) * pdfvi(vi, x_sampled[i], y_sampled[i], beta_loop, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
      U12 = @. U12 / denom
      U22 = [quadgk(vi -> sum(pivi(beta_loop, x_sampled[i], vi) .* (1 .- pivi(beta_loop, x_sampled[i], vi)) .* x_sampled[i].^2) * pdfvi(vi, x_sampled[i], y_sampled[i], beta_loop, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
      U22 = @. U22 / denom

      beta_loop2 = beta_loop .+ ([[sum(U11), sum(U12)] [sum(U12), sum(U22)]] \ [sum(S1), sum(S2)])
      if norm(beta_loop2 .- beta_loop) > 10^(-5)
        beta_loop[1] = beta_loop2[1]
        beta_loop[2] = beta_loop2[2]
        if verbose == true
          println("v", beta_loop2)
        end
      else
        break
      end
    end
    beta_t2 = copy(beta_loop)

    denom = [quadgk(vi -> pdfvi(vi, x_sampled[i], y_sampled[i], beta_t2, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
    S1 = [quadgk(vi -> vi^2 * pdfvi(vi, x_sampled[i], y_sampled[i], beta_t2, sigma2a_t), -Inf, Inf)[1] for i in 1:M]
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
      denom = [quadgk(vi -> pdfvi(vi, x_sampled[i], y_sampled[i], beta_t2, sigma2a_t2), -Inf, Inf)[1] for i in 1:M]
      muis = [quadgk(vi -> sum(pivi(beta_t2, x_sampled[i], vi)) * pdfvi(vi, x_sampled[i], y_sampled[i], beta_t2, sigma2a_t2), -Inf, Inf)[1] for i in 1:M]
      muis = @. muis / denom

      denom2 = [quadgk(vi -> pdfvi(vi, x_sampled[i], y_sampled[i], [beta0, beta1], sigma2a), -Inf, Inf)[1] for i in 1:M]
      muis2 = [quadgk(vi -> sum(pivi([beta0, beta1], x_sampled[i], vi)) * pdfvi(vi, x_sampled[i], y_sampled[i], [beta0, beta1], sigma2a), -Inf, Inf)[1] for i in 1:M]
      muis2 = @. muis2 / denom2
      break
    else
      beta_t[1] = beta_t2[1]
      beta_t[2] = beta_t2[2]
      sigma2a_t = sigma2a_t2
    end
  end
  return muis, muis2
end
