k = 3
dataname = "high"

dirName = string(dataname, "_k", k)

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

data = CSV.read(string(dataname, ".csv"), DataFrame)
data = data[data[:,10] .<= k,:]

Yvec = ifelse.(data[:,10] .< k, 0, 1)

Xmat = convert(Matrix, data[:,2:7])
Xmat = hcat(fill(1, length(Xmat[:,1])),Xmat)
Xmat = float(Xmat)

SampleIdx = unique(data[:,1])

X_sampled = [Xmat[data[:,1] .== idx,:] for idx in SampleIdx]
Y_sampled = [Yvec[data[:,1] .== idx] for idx in SampleIdx]

w1_sampled = [data[:,8][data[:,1] .== idx][1] for idx in SampleIdx]
w2_sampled = [data[:,9][data[:,1] .== idx][1] for idx in SampleIdx]

data_tmp = copy(data)
data_tmp.sch_no = string.(data_tmp.sch_no)
data_tmp.time_as = float(data_tmp.time_as)
data_tmp.time_pe = float(data_tmp.time_pe)
data_tmp.gender = float(data_tmp.gender)
data_tmp.income = float(data_tmp.income)
data_tmp.d_edu1 = float(data_tmp.d_edu1)
data_tmp.m_edu1 = float(data_tmp.m_edu1)
data_tmp.Y = ifelse.(data_tmp.Y .< k, 0.0, 1.0)

fm = @formula(Y ~ time_as + time_pe + gender + income + d_edu1 + m_edu1 + (1|sch_no))
fm1 = fit(MixedModel, fm, data_tmp, Binomial())

beta_ini = coef(fm1)
sigma2a_ini = ifelse(VarCorr(fm1).σρ.sch_no.σ[1]^2 == 0, 0.0001, VarCorr(fm1).σρ.sch_no.σ[1]^2) # or 0.4?

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

  beta_t2 = resTMP[1:7]
  sigma2a_t2 = resTMP[8]

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
println(vcat(round.(resTMP3[1:7]; digits = 5), resTMP3[8]))

#using Printf
#map(x -> @sprintf("%.5f", x), resTMP3)

#resTMP1 = updatebetamat(ahat, X_sampled, Y_sampled, w1_sampled, w2_sampled, beta_t1, sigma2a_t1)
#resTMP2 = updatebetamat_new3(ahat, X_sampled, Y_sampled, w1_sampled, w2_sampled, beta_t1, sigma2a_t1)
#resTMP3 = updatebetamat_S(X_sampled, Y_sampled, w1_sampled, w2_sampled, beta_t1, sigma2a_t1)
#=
beta_t1 = copy(beta_ini)
sigma2a_t1 = sigma2a_ini
resTMP4 = copy(foo(X_sampled, Y_sampled, w1_sampled, beta_t1, sigma2a_t1))

beta_t1 = copy(beta_ini)
sigma2a_t1 = sigma2a_ini
resTMP5 = copy(foo2(X_sampled, Y_sampled, w1_sampled, beta_t1, sigma2a_t1))
#println(round.(resTMP1; digits = 6))
=#
Res = zeros(8,5)
Res[:,1] = resTMP1
Res[:,2] = resTMP2
Res[:,3] = resTMP3
Res[:,4] = resTMP4
Res[:,5] = resTMP5

#Res[:,1] = vcat(round.(resTMP1[1:7]; digits = 5),)
#Res[:,2] =  round.(resTMP2; digits = 5)
#Res[:,3] =   round.(resTMP3; digits = 5)
#Res[:,4] =   round.(resTMP4; digits = 5)
#Res[:,5] =   round.(resTMP5; digits = 5)

Res = NamedArray(Res)
NamedArrays.setnames!(Res, vcat("Constant", names(data)[2:7], "Sigma^2"), 1)
NamedArrays.setnames!(Res, ["Method 1", "Method 2", "Method 3", "PML", "PCL"], 2)




#@sprintf("%g", Res)
#using Printf
#map(x -> @sprintf("%.5f", x), Res)
CSV.write(string(dirName, ".txt"),  DataFrame(Res), header=false)
