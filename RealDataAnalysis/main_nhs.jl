p = 9


verbose = true
eps_theta = 10^(-5)

using Distributions
using Random
using LinearAlgebra
using NamedArrays
using StatsBase
using MixedModels
using DataFrames
using QuadGK
using Optim
using Roots
#using RCall
using Dates
using CSV

include("ftns_method1.jl")
include("ftns_method2_gq.jl")
include("ftns_method3_gq.jl")
include("ftns_method4.jl")
include("ftns_method5.jl")

data = CSV.read("nepal_i.csv", DataFrame)
data = data[sortperm(data.V001), :]

Yvec = ifelse.(Int.(data.V313) .== 0, 0, 1)

data_compact = data[!, ["V218R", "V150", "V013R","V113", "V128", "V116", "V129", "V106"]]

Xmat = convert(Matrix, data_compact)
Xmat = hcat(fill(1, length(Xmat[:,1])),Xmat)
Xmat = float(Xmat)

cluster = data.V001
strata = data.V022

SampleIdx = unique(cluster)

X_sampled = [Xmat[cluster .== idx,:] for idx in SampleIdx]
Y_sampled = [Yvec[cluster .== idx] for idx in SampleIdx]

w1_sampled = float(data.V005) / 1000000.0
w2_sampled = [findmax(data.V002[cluster .== cluster[i]])[1] * mean(data.V003[cluster .== cluster[i]]) / sum(data.V001 .== cluster[i]) for i in 1:length(data[:,1])]

w1_sampled = [w1_sampled[cluster .== idx][1] for idx in SampleIdx]
w2_sampled = [w2_sampled[cluster .== idx][1] for idx in SampleIdx]

data_compact = data[!, ["V313", "V218R", "V150", "V013R","V113", "V128", "V116", "V129", "V106", "V001"]]
data_compact.V001 = string.(data_compact.V001)
data_compact.V218R = float(data_compact.V218R)
data_compact.V150 = float(data_compact.V150)
data_compact.V013R = float(data_compact.V013R)
data_compact.V113 = float(data_compact.V113)
data_compact.V128 = float(data_compact.V128)
data_compact.V116 = float(data_compact.V116)
data_compact.V129 = float(data_compact.V129)
data_compact.V106 = float(data_compact.V106)
data_compact.V313 = ifelse.(Int.(data.V313) .== 0, 0, 1)

#fm = @formula(Y ~ time_as + time_pe + gender + income + d_edu1 + m_edu1 + (1|sch_no))
fm = @formula(V313 ~ V218R + V150 + V013R + V113 + V128 + V116 + V129 + V106 + (1|V001))
fm1 = fit(MixedModel, fm, data_compact, Binomial())

beta_ini = coef(fm1)
sigma2a_ini = ifelse(VarCorr(fm1).σρ.V001.σ[1]^2 == 0, 0.0001, VarCorr(fm1).σρ.V001.σ[1]^2) # or 0.4?

theta_t1 = Vector{Float64}(undef, 3)
theta_t2 = Vector{Float64}(undef, 3)
#=
beta_t1 = copy(beta_ini)
sigma2a_t1 = sigma2a_ini
cnt = 0
while true
  global beta_t1, sigma2a_t1, theta_t1, theta_t2, cnt
  cnt += 1

  ahat = solveahat(X_sampled, Y_sampled, w2_sampled, beta_t1, sigma2a_t1)
  resTMP = updatebetamat(ahat, X_sampled, Y_sampled, w1_sampled, w2_sampled, beta_t1, sigma2a_t1)

  beta_t2 = resTMP[1:7]
  sigma2a_t2 = resTMP[8]

  if any(isnan.(resTMP))
    println("simnum", simnum, ", cnt = ", cnt, ": NAN generated(normal approximation w/ v): perturbation on initial parameters")
    beta_t1 = beta_ini
    sigma2a_t1 = 2 * sigma2a_ini * rand()
  elseif  cnt > 100
    @warn(": Convergence Failed(normal approximation w/ v)")
    theta_t2 = vcat([beta_t2, sigma2a_t2]...)
    break
  else
    theta_t1 = vcat([beta_t1, sigma2a_t1]...)
    theta_t2 = vcat([beta_t2, sigma2a_t2]...)
    if verbose == true
      println("v", theta_t2)
    end
    if norm(theta_t1 - theta_t2) < eps_theta
      break
    else
      beta_t1 = copy(beta_t2)
      sigma2a_t1 = sigma2a_t2
    end
  end
end
resTMP1 = copy(theta_t2)

beta_t1 = copy(beta_ini)
sigma2a_t1 = sigma2a_ini
cnt = 0
while true
  global beta_t1, sigma2a_t1, theta_t1, theta_t2, cnt
  cnt += 1

  ahat = solveahat(X_sampled, Y_sampled, w2_sampled, beta_t1, sigma2a_t1)
  resTMP = updatebetamat_new3(ahat, X_sampled, Y_sampled, w1_sampled, w2_sampled, beta_t1, sigma2a_t1)

  beta_t2 = resTMP[1:7]
  sigma2a_t2 = resTMP[8]

  if any(isnan.(resTMP))
    println("simnum", simnum, ", cnt = ", cnt, ": NAN generated(normal approximation w/ vhat): perturbation on initial parameters")
    beta_t1 = beta_ini
    sigma2a_t1 = 2 * sigma2a_ini * rand()
  elseif  cnt > 100
    @warn(": Convergence Failed(normal approximation w/ vhat)")
    theta_t2 = vcat([beta_t2, sigma2a_t2]...)
    break
  else
    theta_t1 = vcat([beta_t1, sigma2a_t1]...)
    theta_t2 = vcat([beta_t2, sigma2a_t2]...)
    if verbose == true
      println("vhat", theta_t2)
    end
    if norm(theta_t1 - theta_t2) < eps_theta
      break
    else
      beta_t1 = copy(beta_t2)
      sigma2a_t1 = sigma2a_t2
    end
  end
end
resTMP2 = copy(theta_t2)
=#
beta_t1 = copy(beta_ini)
sigma2a_t1 = sigma2a_ini
cnt = 0
while true
  global beta_t1, sigma2a_t1, theta_t1, theta_t2, cnt
  cnt += 1

  resTMP = updatebetamat_S(X_sampled, Y_sampled, w1_sampled, w2_sampled, beta_t1, sigma2a_t1)

  beta_t2 = resTMP[1:p]
  sigma2a_t2 = resTMP[p + 1]

  if any(isnan.(resTMP))
    println("simnum", simnum, ", cnt = ", cnt, ": NAN generated(profile): perturbation on initial parameters")
    beta_t1 = beta_ini
    sigma2a_t1 = 2 * sigma2a_ini * rand()
  elseif  cnt > 100
    @warn(": Convergence Failed(profile)")
    theta_t2 = vcat([beta_t2, sigma2a_t2]...)
    break
  else
    theta_t1 = vcat([beta_t1, sigma2a_t1]...)
    theta_t2 = vcat([beta_t2, sigma2a_t2]...)
    if verbose == true
      println("profile", theta_t2)
    end
    if norm(theta_t1 - theta_t2) < eps_theta
      break
    else
      beta_t1 = copy(beta_t2)
      sigma2a_t1 = sigma2a_t2
    end
  end
end
resTMP3 = copy(theta_t2)
println(vcat(round.(resTMP3[1:p]; digits = 5), resTMP3[p + 1]))

#using Printf
#map(x -> @sprintf("%.5f", x), resTMP3)

#resTMP1 = updatebetamat(ahat, X_sampled, Y_sampled, w1_sampled, w2_sampled, beta_t1, sigma2a_t1)
#resTMP2 = updatebetamat_new3(ahat, X_sampled, Y_sampled, w1_sampled, w2_sampled, beta_t1, sigma2a_t1)

beta_t1 = copy(beta_ini)
sigma2a_t1 = sigma2a_ini
resTMP4 = copy(foo(X_sampled, Y_sampled, w1_sampled, beta_t1, sigma2a_t1))

beta_t1 = copy(beta_ini)
sigma2a_t1 = sigma2a_ini
resTMP5 = copy(foo2(X_sampled, Y_sampled, w1_sampled, beta_t1, sigma2a_t1))
#println(round.(resTMP1; digits = 6))

Res = zeros(p+1,3)
Res[:,1] = resTMP3
Res[:,2] = resTMP4
Res[:,3] = resTMP5
Res[p+1,:] = sqrt.(Res[p+1,:])

#Res[:,1] = vcat(round.(resTMP1[1:7]; digits = 5),)
#Res[:,2] =  round.(resTMP2; digits = 5)
#Res[:,3] =   round.(resTMP3; digits = 5)
#Res[:,4] =   round.(resTMP4; digits = 5)
#Res[:,5] =   round.(resTMP5; digits = 5)

Res = NamedArray(Res)
NamedArrays.setnames!(Res, vcat("Constant", names(data_compact)[2:p], "Sigma"), 1)
NamedArrays.setnames!(Res, ["Proposed_Method", "PML", "PCL"], 2)




#@sprintf("%g", Res)
#using Printf
#map(x -> @sprintf("%.5f", x), Res)
CSV.write(string(nepal, ".txt"),  DataFrame(Res), header=false)
