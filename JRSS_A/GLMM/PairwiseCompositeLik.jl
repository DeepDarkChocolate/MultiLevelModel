
function LL_ftn(ai::Float64, xij::Float64, yij::Int64, xik::Float64, yik::Int64, beta_t1::Vector{Float64})
  pij = 1 - 1 / (1 + exp(beta_t1[2] * xij + ai))
  pik = 1 - 1 / (1 + exp(beta_t1[2] * xik + ai))
  #Lij = exp(yij * log(pij) + (1 - yij) * log(1 - pij))
  #Lik = exp(yik * log(pik) + (1 - yik) * log(1 - pik))
  return ifelse(yij == 1, pij, 1 - pij) * ifelse(yik == 1, pik, 1 - pik)
  #return Lij * Lik
end

#=
function l_wc(x_sampled, y_sampled, w1_sampled, pi2mat_sampled, K, beta_t1, mua_t1, eta_t1)
  n = size(x_sampled)[1]
  res = Array{Float64}(undef, n)
  Threads.@threads for i in 1:n
    sumi = 0.0
    m = size(x_sampled[i])[1]
    for j in 1:m
      for k in 1:(j-1)
        wjk_i = 1 / pi2mat_sampled[i][j, k]
        xij = x_sampled[i][j]
        xik = x_sampled[i][k]
        yij = y_sampled[i][j]
        yik = y_sampled[i][k]
        integral, err = quadgk(v -> LL_ftn(v, xij, yij, xik, yik, beta_t1)* pdf(Normal(mua_t1, sqrt(1 / eta_t1)), v), -Inf, Inf, rtol=1e-4)
        #xquad, wquad = gauss(v -> LL_ftn(v, xij, yij, xik, yik, beta_t1)* pdf(Normal(mua_t1, sqrt(1 / eta_t1)), v), K, mua_t1- 6.0, mua_t1 + 6.0, rtol=1e-4)
        #denom = sum(wquad)
        sumi += (wjk_i * log(integral))
        #if i == 1
        #  @show j, k, wjk_i, log(integral)
        #end
      end
    end
    res[i] = w1_sampled[i] * sumi
  end
  return sum(res)
end


function foo2(x_sampled, y_sampled, w1_sampled, pi2mat_sampled, K, beta_t1, mua_t1, eta_t1)
  n = size(x_sampled)[1]
  firstp = Array{Float64}(undef, n , 3)
  secondp = Array{Float64}(undef, n , 6)
  vectmp = Vector{Float64}(undef, 3)
  vectmp2 = Vector{Float64}(undef, 6)
  Threads.@threads for i in 1:n
  #for i in 1:n
    #println(i)
    m = size(x_sampled[i])[1]
    tmp3 = zeros(3)
    tmp6 = zeros(6)

    for j in 1:m
      for k in 1:(j-1)
        #println((j, k))
        wjk_i = 1 / pi2mat_sampled[i][j, k]
        xij = x_sampled[i][j]
        xik = x_sampled[i][k]
        yij = y_sampled[i][j]
        yik = y_sampled[i][k]

        res = optimize(v -> LL_ftn(v[1], xij, yij, xik, yik, beta_t1)* pdf(Normal(mua_t1, sqrt(1 / eta_t1)), v[1]), -15, 15)
        mode = Optim.minimizer(res)[1]

        xquad, wquad = gauss(v -> LL_ftn(v, xij, yij, xik, yik, beta_t1)* pdf(Normal(mua_t1, sqrt(1 / eta_t1)), v), K, mua_t1- 8.0, mua_t1 + 8.0, rtol=1e-4)
        denom = sum(wquad)
        pij = [1 - 1 / (1 + exp(beta_t1[2] * xij + xquad[k])) for k in 1:K]
        pik = [1 - 1 / (1 + exp(beta_t1[2] * xik + xquad[k])) for k in 1:K]

        d1_vec = Vector{Float64}(undef, K)
        d2_vec = Vector{Float64}(undef, K)
        d3_vec = Vector{Float64}(undef, K)

        d1_vec = @. (yij - pij) * xij + (yik - pik) * xik
        d2_vec = @. eta_t1 * (xquad - mua_t1)
        d3_vec = @. 1.0 / 2.0 / eta_t1 - 1.0 / 2.0 * (xquad - mua_t1)^2

        d1 = sum(@. d1_vec * wquad)
        d2 = sum(@. d2_vec * wquad)
        d3 = sum(@. d3_vec * wquad)

        A11 = sum(@. (-pij * (1 - pij) * xij^2 - pik * (1 - pik) * xik^2 + d1_vec^2) * wquad)
        A21 = sum(@. d2_vec * d1_vec * wquad)
        A22 = sum(@. (-eta_t1 + d2_vec^2) * wquad)
        A31 = sum(@. d3_vec * d1_vec * wquad)
        A32 = sum(@. (xquad - mua_t1 + d3_vec * d2_vec) * wquad)
        A33 = sum(@. (- 1 / 2 / eta_t1^2 + d3_vec^2) * wquad)

        #tmp3[1] = tmp3[1] .+ (wjk_i * log(denom))
        tmp3 .= tmp3 .+ (wjk_i * [d1, d2, d3] / denom)
        tmp6 .= tmp6 .+ (wjk_i * ([A11, A21, A22, A31, A32, A33] / denom - [d1 * d1, d2 * d1, d2* d2, d3 * d1, d3 * d2, d3 * d3] / denom^2))
      end
    end

    #firstp[i,1] = w1_sampled[i] * tmp3[1]
    @. firstp[i,:] = w1_sampled[i] * tmp3
    @. secondp[i,:] = w1_sampled[i] * tmp6
  end

  vectmp .= [sum(firstp[:,k]) for k in 1:3]
  vectmp2 .= [sum(secondp[:,k]) for k in 1:6]

  #return vectmp
  #return vectmp[1]

  return ([beta_t1[2], mua_t1, eta_t1] .- [vectmp2[1] vectmp2[2] vectmp2[4]
  vectmp2[2] vectmp2[3] vectmp2[5]
  vectmp2[4] vectmp2[5] vectmp2[6]] \ vectmp)
end
=#

function foo2(x_sampled, y_sampled, w1_sampled, pi2mat_sampled, beta_t1, mua_t1, eta_t1, interval, rtol, K2)
  n = size(x_sampled)[1]
  firstp = Array{Float64}(undef, n , 3)
  secondp = Array{Float64}(undef, n , 6)
  vectmp = Vector{Float64}(undef, 3)
  vectmp2 = Vector{Float64}(undef, 6)
  K = K2
  xquad, wquad = gauss(v -> pdf(Normal(mua_t1, sqrt(1 / eta_t1)), v), K, mua_t1- interval, mua_t1 + interval, rtol=rtol)
  Threads.@threads for i in 1:n
  #for i in 1:n
    #println(i)
    m = size(x_sampled[i])[1]
    tmp3 = zeros(3)
    tmp6 = zeros(6)
    for j in 1:m
      for k in 1:(j-1)
        #println((j, k))
        wjk_i = 1 / pi2mat_sampled[i][j, k]
        xij = x_sampled[i][j]
        xik = x_sampled[i][k]
        yij = y_sampled[i][j]
        yik = y_sampled[i][k]
        LijLik = [LL_ftn(xquad[k], xij, yij, xik, yik, beta_t1) for k in 1:K]

        denom = sum(@. LijLik * wquad)
        pij = [1 - 1 / (1 + exp(beta_t1[2] * xij + xquad[k])) for k in 1:K]
        pik = [1 - 1 / (1 + exp(beta_t1[2] * xik + xquad[k])) for k in 1:K]

        d1_vec = Vector{Float64}(undef, K)
        d2_vec = Vector{Float64}(undef, K)
        d3_vec = Vector{Float64}(undef, K)

        d1_vec = @. (yij - pij) * xij + (yik - pik) * xik
        d2_vec = @. eta_t1 * (xquad - mua_t1)
        d3_vec = @. 1.0 / 2.0 / eta_t1 - 1.0 / 2.0 * (xquad - mua_t1)^2

        d1 = sum(@. LijLik * d1_vec * wquad)
        d2 = sum(@. LijLik * d2_vec * wquad)
        d3 = sum(@. LijLik * d3_vec * wquad)

        A11 = sum(@. LijLik * (-pij * (1 - pij) * xij^2 - pik * (1 - pik) * xik^2 + d1_vec^2) * wquad)
        A21 = sum(@. LijLik * d2_vec * d1_vec * wquad)
        A22 = sum(@. LijLik * (-eta_t1 + d2_vec^2) * wquad)
        A31 = sum(@. LijLik * d3_vec * d1_vec * wquad)
        A32 = sum(@. LijLik * (xquad - mua_t1 + d3_vec * d2_vec) * wquad)
        A33 = sum(@. LijLik * (- 1 / 2 / eta_t1^2 + d3_vec^2) * wquad)

        #tmp3[1] = tmp3[1] .+ (wjk_i * log(denom))
        tmp3 .= tmp3 .+ (wjk_i * [d1, d2, d3] / denom)
        tmp6 .= tmp6 .+ (wjk_i * ([A11, A21, A22, A31, A32, A33] / denom - [d1 * d1, d2 * d1, d2* d2, d3 * d1, d3 * d2, d3 * d3] / denom^2))
      end
    end

    #firstp[i,1] = w1_sampled[i] * tmp3[1]
    @. firstp[i,:] = w1_sampled[i] * tmp3
    @. secondp[i,:] = w1_sampled[i] * tmp6
  end

  vectmp .= [sum(firstp[:,k]) for k in 1:3]
  vectmp2 .= [sum(secondp[:,k]) for k in 1:6]

  #return vectmp
  #return vectmp[1]

  return ([beta_t1[2], mua_t1, eta_t1] .- [vectmp2[1] vectmp2[2] vectmp2[4]
  vectmp2[2] vectmp2[3] vectmp2[5]
  vectmp2[4] vectmp2[5] vectmp2[6]] \ vectmp)
end
