# Normal approximation using v

function hi_ftn(aihat, ai, xi, yi, w2i, beta_t1_2, sigma2e_t1, mua_t1, sigmaa_t1, pi2mati)
  ratioi = (yi .- mean(yi) .- beta_t1_2 .* (xi .- mean(xi))) .* w2i
  #ratioi = (yi .- beta_t1_2 .* xi .- ai) .* w2i
  pi2i = @. 1 / w2i
  vi = transpose(ratioi) * (1 .- (pi2i * transpose(pi2i) ./ pi2mati)) * ratioi /
  sum(@. w2i)^2
  return pdf(Normal(mua_t1, sigmaa_t1), ai) * pdf(Normal(ai, sqrt(vi)), aihat)
end

function Evis_ftn(aihat, xi, yi, w2i, beta_t1_2, sigma2e_t1, mua_t1, sigmaa_t1, pi2mati)
  denom = quadgk(v -> hi_ftn(aihat, v, xi, yi, w2i, beta_t1_2, sigma2e_t1, mua_t1, sigmaa_t1, pi2mati), -Inf, Inf, rtol=1e-5)[1]
  num1 = quadgk(v -> v * hi_ftn(aihat, v, xi, yi, w2i, beta_t1_2, sigma2e_t1, mua_t1, sigmaa_t1, pi2mati), -Inf, Inf, rtol=1e-5)[1]
  num2 = quadgk(v -> v^2 * hi_ftn(aihat, v, xi, yi, w2i, beta_t1_2, sigma2e_t1, mua_t1, sigmaa_t1, pi2mati), -Inf, Inf, rtol=1e-5)[1]
  return num1 / denom, num2 / denom
end

function updatebetamat(ahat, x, y, w1, w2, pi2mat, beta_t1, sigma2e_t1, mua_t1, sigma2a_t1)
  n = size(x)[1]
  sigmaa_t1 = sqrt(sigma2a_t1)

  #Threads.@threads for i in 1:n
  Ev1 = Vector{Float64}(undef, n)
  Vv1 = Vector{Float64}(undef, n)
  for i in 1:n
    xi = x[i]
    yi = y[i]
    w1i = w1[i]
    w2i = w2[i]
    aihat = ahat[i]
    pi2mati = pi2mat[i]
    Evi1, Evi2 = Evis_ftn(aihat, xi, yi, w2i, beta_t1[2], sigma2e_t1, mua_t1, sigmaa_t1, pi2mati)
    Ev1[i] = Evi1
    Vv1[i] = Evi2 - Evi1^2
  end
  w2_tilde = [size(x[i])[1] .* w2[i] ./sum(w2[i]) for i in 1:n]

  #beta_t1_scr = sum(w1 .* [sum(@. w2[i] * (y[i] - Ev1[i]) * x[i]) for i in 1:n]) /
  #sum(w1 .* [sum(@. w2[i] * x[i] ^ 2) for i in 1:n])
  #sigma2ehat = sum(w1 .* [sum(@. w2[i] * ( (-Ev1[i] + y[i] - beta_t1_scr * x[i]) ^ 2 + Vv1[i]) ) for i in 1:n]) /
  #sum(w1 .* [sum(w2[i]) for i in 1:n])
  beta_t1_scr = sum(w1 .* [sum(@. w2_tilde[i] * (y[i] - Ev1[i]) * x[i]) for i in 1:n]) /
  sum(w1 .* [sum(@. w2_tilde[i] * x[i] ^ 2) for i in 1:n])
  sigma2ehat = sum(w1 .* [sum(@. w2_tilde[i] * ( (-Ev1[i] + y[i] - beta_t1_scr * x[i]) ^ 2 + Vv1[i]) ) for i in 1:n]) /
  sum(w1 .* [sum(w2_tilde[i]) for i in 1:n])
  muhat = sum(@. w1 * Ev1) / sum(@. w1)
  sigma2ahat = sum(@. w1 * (Ev1 ^ 2 + Vv1)) / sum(@. w1) - muhat^2

  return beta_t1_scr, sigma2ehat, muhat, sigma2ahat
end
