
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

#include("ftns_method1.jl")
#include("ftns_method2_gq.jl")
#include("ftns_method3_gq.jl")
#include("ftns_method4.jl")
include("ftns.jl")

data = CSV.read("/home/yonghyun/Documents/GLMM/RealData/nepal_i.csv", DataFrame)
data = data[sortperm(data.V001), :]

Yvec = float(data.V313_yes)
#Yvec = ifelse.(Int.(data.V313) .== 0, 0, 1)

colnames = ["V012", "V101_Hill", "V101_Terai","V102_Rural", "V130_Buddhist",
"V130_Muslim", "V130_Kirat", "V130_Christian", "V133", "V137", "V138", "V151_Female"]
data_compact = data[!, colnames]

Xmat = convert(Matrix, data_compact)
Xmat = hcat(fill(1, length(Xmat[:,1])),Xmat)
Xmat = float(Xmat)

p = size(Xmat)[2]

cluster = data.V001
strata = data.V022

SampleIdx = unique(cluster)

X_sampled = [Xmat[cluster .== idx,:] for idx in SampleIdx]
Y_sampled = [Yvec[cluster .== idx] for idx in SampleIdx]

M = size(X_sampled)[1]

w1_sampled = float(data.V005) / 1000000.0
w2_sampled = [findmax(data.V002[cluster .== cluster[i]])[1] / sum(data.V001 .== cluster[i]) for i in 1:length(data[:,1])]
#w2_sampled = [findmax(data.V002[cluster .== cluster[i]])[1] * mean(data.V003[cluster .== cluster[i]]) / sum(data.V001 .== cluster[i]) for i in 1:length(data[:,1])]
#sum(w2_sampled .<= 1.0)

w1_sampled = [w1_sampled[cluster .== idx][1] for idx in SampleIdx]
w2_sampled = [w2_sampled[cluster .== idx][1] for idx in SampleIdx]

#data_compact = data[!, ["V313", "V218R", "V150", "V013R","V113", "V128", "V116", "V129", "V106", "V001"]]
data_compact.V001 = string.(data.V001)
data_compact.V137 = float.(data.V137)
data_compact.V313_yes = Int.(data.V313_yes)

fm = @formula(V313_yes ~ V012 + V101_Hill + V101_Terai + V102_Rural +
 V130_Buddhist + V130_Muslim + V130_Kirat + V130_Christian + V133 +
 V137 + V138 + V151_Female + (1|V001))
fm1 = fit(MixedModel, fm, data_compact, Binomial())

beta_ini = coef(fm1)
sigma2a_ini = ifelse(VarCorr(fm1).σρ.V001.σ[1]^2 == 0, 0.0001, VarCorr(fm1).σρ.V001.σ[1]^2) # or 0.4?

theta_t1 = Vector{Float64}(undef, p + 1)
theta_t2 = Vector{Float64}(undef, p + 1)

beta_t = copy(beta_ini)
sigma2a_t = sigma2a_ini

theta_t2 = EMalg(X_sampled, Y_sampled, beta_t, sigma2a_t, w1_sampled, w2_sampled, M, verbose = true, eps_theta = 0.001)

BOOTNUM = 200
boot_res = Array{Float64}(undef, BOOTNUM, p + 1)
if isfile(joinpath(dirname(@__FILE__), "log.txt"))
  rm(joinpath(dirname(@__FILE__), "log.txt"))
end
@time Threads.@threads for boot_num in 1:BOOTNUM
  Random.seed!(boot_num)
  open(joinpath(dirname(@__FILE__), "log.txt"), "a+") do io
  write(io, "$boot_num\n")
  println(boot_num)
  end;
  boot_res[boot_num, :] = bootstrap(X_sampled, theta_t2[1:p], theta_t2[p+1], w1_sampled, w2_sampled)
end
CSV.write(joinpath(dirname(@__FILE__), "boot_res.csv"),  DataFrame(boot_res), header=false)

#=
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
=#
