function solveahat(x, y, w2, beta_t1, sigma2a_t1)
  n = size(x)[1]
  sigmaa_t1 = sqrt(sigma2a_t1)
  a_t2 = Vector{Float64}(undef, n)
  Threads.@threads for i in 1:n
  #for i in 1:n
    ai0 = 0.0
    xi = x[i]
    yi = y[i]
    w2i = w2[i]
    mi = size(x[i])[1]
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
        etai = [sum(beta_t1 .* xi[j,:]) .+ ai_t1 for j in 1:mi]
        Pi = @. 1 - 1 / (1 + exp(etai))
        ai_t2 = ai_t1 + sum(@. (yi - Pi) * w2i) ./ sum(@. Pi * (1 - Pi) * w2i)
        if isnan(ai_t2)
          #@warn(": NAN generated(solveahat)")
          #println("ai_t1 = ", ai_t1)
          #println("ai_t2 = ", ai_t2)
          #println("y[i] = ", y[i])
          #println("i = ", i)
          #println("cnt2 = ", cnt2)
          ai_t1 = rand(Normal(0.0,sigmaa_t1),1)[1]
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

function hi_ftn0(aihat, ai, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi)
  Pi = [1 - 1 / (1 + exp(sum(beta_t1_2 .* xi[j,:]) + aihat)) for j in 1:mi]
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

function Epi_ftn0(aihat, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi)
  denom = quadgk(v -> hi_ftn0(aihat, v, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi), -Inf, Inf)[1]
  pi1 = [quadgk(v -> pij_ftn(v, xi[j,:], beta_t1_2) * hi_ftn0(aihat, v, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi), -Inf, Inf)[1] / denom for j in 1:mi]
  pi2 = [quadgk(v -> pij_ftn(v, xi[j,:], beta_t1_2) * (1 - pij_ftn(v, xi[j,:], beta_t1_2)) * hi_ftn0(aihat, v, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi), -Inf, Inf)[1] / denom for j in 1:mi]
  return pi1, pi2
end

#function Epi2_ftn(aihat, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1, m)
#  denom = quadgk(v -> hi_ftn(aihat, v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1]
#  return [quadgk(v -> pij_ftn(v, xi[j], beta_t1_2) * (1 - pij_ftn(v, xi[j], beta_t1_2)) * hi_ftn(aihat, v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1] / denom for j in 1:m]
#end

function Evis_ftn0(aihat, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi)
  denom = quadgk(v -> hi_ftn0(aihat, v, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi), -Inf, Inf)[1]
  #num1 = quadgk(v -> v * hi_ftn(aihat, v, xi, yi, w2i, pi2mati, beta_t1_2, mua_t1, sigmaa_t1), -Inf, Inf)[1]
  num2 = quadgk(v -> v^2 * hi_ftn0(aihat, v, xi, yi, w2i, beta_t1_2, sigmaa_t1, mi), -Inf, Inf)[1]
  #return num1 / denom, num2 / denom
  return num2 / denom
end

function updatebetamat_new3(ahat, x, y, w1, w2, beta_t1_ogn, sigma2a_t1)
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
      Epi1, Epi2 = Epi_ftn0(aihat, xi, yi, w2i, beta_t1, sigmaa_t1, mi)
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
    Evi2 = Evis_ftn0(aihat, xi, yi, w2i, beta_t1, sigmaa_t1, mi)
    #Evi1, Evi2 = Evis_ftn(aihat, xi, yi, w2i, pi2mati, beta_t1[2], mua_t1, sigmaa_t1, m)
    #mu_num += w1i * Evi1
    sigma_num += w1i * Evi2
  end

  return vcat(beta_t1,  sigma_num / sum(w1))
end
