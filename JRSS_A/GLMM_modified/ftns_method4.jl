function pi_ftn(Pi::Vector{Float64}, ai::Float64, xi::Vector{Float64}, beta_t1::Vector{Float64})
  @. Pi = 1 - 1 / (1 + exp(beta_t1[2] * xi + ai))
end

function li_ftn(vec::Vector{Float64}, Pi::Vector{Float64}, ai::Float64, xi::Vector{Float64}, yi::Vector{Int64}, w2i::Vector{Float64}, beta_t1::Vector{Float64})
  pi_ftn(Pi, ai, xi, beta_t1)
  @. vec =  ifelse(yi == 1, Pi^w2i, (1 - Pi)^w2i)
  return prod(vec)
end

function wi_tilde(w2i, m)::Vector{Float64}
  return m .* w2i ./ sum(w2i)
end

function pi_tmp(ai::Float64, xi::Vector{Float64}, beta_t1::Vector{Float64})
  Pi = @. 1 - 1 / (1 + exp(beta_t1[2] * xi + ai))
  return Pi
end

#=
function foo(x_sampled, y_sampled, w1_sampled, w2_sampled, K, beta_t1, mua_t1, eta_t1)
  n = size(x_sampled)[1]
  firstp = Array{Float64}(undef, n , 3)
  secondp = Array{Float64}(undef, n , 6)
  vectmp = Vector{Float64}(undef, 3)
  vectmp2 = Vector{Float64}(undef, 6)

  Threads.@threads for i in 1:n
    m = size(x_sampled[i])[1]
    Pi_tmp = similar(x_sampled[i])
    vec_tmp = similar(x_sampled[i])
    w2_tilde_tmp = wi_tilde(w2_sampled[i], m)
    xquad, wquad = gauss(v -> li_ftn(vec_tmp, Pi_tmp, v, x_sampled[i], y_sampled[i], w2_tilde_tmp, beta_t1) * pdf(Normal(mua_t1, sqrt(1 / eta_t1)), v), K, mua_t1-6.0, mua_t1+6.0, rtol=1e-3)
    denomi = sum(wquad)

    Pi_mat = Array{Float64}(undef, length(x_sampled[i]), K)
    for k in 1:K
      Pi_mat[:,k] = pi_tmp(xquad[k], x_sampled[i], beta_t1)
    end
    d1_vec = Vector{Float64}(undef, K)
    for k in 1:K
      d1_vec[k] = sum(@. w2_tilde_tmp * (y_sampled[i] - Pi_mat[:,k]) * x_sampled[i])
    end
    d2_vec = Vector{Float64}(undef, K)
    for k in 1:K
      d2_vec[k] = eta_t1 * (xquad[k] - mua_t1)
    end
    d3_vec = Vector{Float64}(undef, K)
    for k in 1:K
      d3_vec[k] = 1.0 / 2.0 / eta_t1 - 1.0 / 2.0 * (xquad[k] - mua_t1)^2
    end

    d1i = sum(@. d1_vec * wquad)
    d2i = sum(@. d2_vec * wquad)
    d3i = sum(@. d3_vec * wquad)

    A11i = sum(([-sum(@. w2_tilde_tmp * Pi_mat[:,k] * (1 - Pi_mat[:,k]) * x_sampled[i]^2) for k in 1:K] .+ d1_vec.^2) .* wquad)
    A21i = sum(@. d2_vec * d1_vec * wquad)
    A22i = sum(@. (-eta_t1 + d2_vec^2) * wquad)
    A31i = sum(@. d3_vec * d1_vec * wquad)
    A32i = sum(@. (xquad - mua_t1 + d3_vec * d2_vec) * wquad)
    A33i = sum(@. (- 1/2/eta_t1^2 + d3_vec^2) * wquad)

    @. firstp[i,:] = w1_sampled[i] * [d1i, d2i, d3i] / denomi
    @. secondp[i,:] = w1_sampled[i] * ([A11i, A21i, A22i, A31i, A32i, A33i] /
    denomi - [d1i * d1i, d2i * d1i,  d2i * d2i, d3i * d1i, d3i * d2i, d3i * d3i] / denomi^2)
  end

  vectmp .= [sum(firstp[:,k]) for k in 1:3]
  vectmp2 .= [sum(secondp[:,k]) for k in 1:6]

  return ([beta_t1[2], mua_t1, eta_t1] .- [vectmp2[1] vectmp2[2] vectmp2[4]
  vectmp2[2] vectmp2[3] vectmp2[5]
  vectmp2[4] vectmp2[5] vectmp2[6]] \ vectmp)
end
=#

function foo(x_sampled, y_sampled, w1_sampled, w2_sampled, beta_t1, mua_t1, eta_t1, interval, rtol, K2)
  n = size(x_sampled)[1]
  firstp = Array{Float64}(undef, n , 3)
  secondp = Array{Float64}(undef, n , 6)
  vectmp = Vector{Float64}(undef, 3)
  vectmp2 = Vector{Float64}(undef, 6)
  K = K2
  xquad, wquad = gauss(v -> pdf(Normal(mua_t1, sqrt(1 / eta_t1)), v), K, mua_t1-interval, mua_t1+interval, rtol=rtol)

  Threads.@threads for i in 1:n
  #for i in 1:n
    m = size(x_sampled[i])[1]
    Pi_tmp = similar(x_sampled[i])
    vec_tmp = similar(x_sampled[i])
    w2_tilde_tmp = wi_tilde(w2_sampled[i], m)
    Li = [li_ftn(vec_tmp, Pi_tmp, xquad[k], x_sampled[i], y_sampled[i], w2_tilde_tmp, beta_t1) for k in 1:K]
    denomi = sum(Li .* wquad)

    Pi_mat = Array{Float64}(undef, length(x_sampled[i]), K)
    for k in 1:K
      Pi_mat[:,k] = pi_tmp(xquad[k], x_sampled[i], beta_t1)
    end
    d1_vec = Vector{Float64}(undef, K)
    for k in 1:K
      d1_vec[k] = sum(@. w2_tilde_tmp * (y_sampled[i] - Pi_mat[:,k]) * x_sampled[i])
    end
    d2_vec = Vector{Float64}(undef, K)
    for k in 1:K
      d2_vec[k] = eta_t1 * (xquad[k] - mua_t1)
    end
    d3_vec = Vector{Float64}(undef, K)
    for k in 1:K
      d3_vec[k] = 1.0 / 2.0 / eta_t1 - 1.0 / 2.0 * (xquad[k] - mua_t1)^2
    end

    d1i = sum(@. Li * d1_vec * wquad)
    d2i = sum(@. Li * d2_vec * wquad)
    d3i = sum(@. Li * d3_vec * wquad)

    A11i = sum(Li .* ([-sum(@. w2_tilde_tmp * Pi_mat[:,k] * (1 - Pi_mat[:,k]) * x_sampled[i]^2) for k in 1:K] .+ d1_vec.^2) .* wquad)
    A21i = sum(@. Li * d2_vec * d1_vec * wquad)
    A22i = sum(@. Li * (-eta_t1 + d2_vec^2) * wquad)
    A31i = sum(@. Li * d3_vec * d1_vec * wquad)
    A32i = sum(@. Li * (xquad - mua_t1 + d3_vec * d2_vec) * wquad)
    A33i = sum(@. Li * (- 1/2/eta_t1^2 + d3_vec^2) * wquad)

    @. firstp[i,:] = w1_sampled[i] * [d1i, d2i, d3i] / denomi
    @. secondp[i,:] = w1_sampled[i] * ([A11i, A21i, A22i, A31i, A32i, A33i] /
    denomi - [d1i * d1i, d2i * d1i,  d2i * d2i, d3i * d1i, d3i * d2i, d3i * d3i] / denomi^2)
  end

  vectmp .= [sum(firstp[:,k]) for k in 1:3]
  vectmp2 .= [sum(secondp[:,k]) for k in 1:6]

  return ([beta_t1[2], mua_t1, eta_t1] .- [vectmp2[1] vectmp2[2] vectmp2[4]
  vectmp2[2] vectmp2[3] vectmp2[5]
  vectmp2[4] vectmp2[5] vectmp2[6]] \ vectmp)
end
