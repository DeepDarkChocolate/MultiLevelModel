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
fm1
beta_ini = coef(fm1)
sigma2a_ini = ifelse(VarCorr(fm1).σρ.V001.σ[1]^2 == 0, 0.0001, VarCorr(fm1).σρ.V001.σ[1]^2) # or 0.4?

theta_t1 = Vector{Float64}(undef, p + 1)
theta_t2 = Vector{Float64}(undef, p + 1)

beta_t = copy(beta_ini)
sigma2a_t = sigma2a_ini

theta_t2 = EMalg(X_sampled, Y_sampled, beta_t, sigma2a_t, w1_sampled, w2_sampled, M, verbose = true, eps_theta = 0.01)
#w2_sampled_tmp = fill(sum([findmax(data.V002[cluster .== cluster[i]])[1] for i in 1:length(data[:,1])]) / sum([sum(data.V001 .== cluster[i]) for i in 1:length(data[:,1])])
#, length(w2_sampled))
#theta_t3 = EMalg(X_sampled, Y_sampled, beta_t, sigma2a_t, w1_sampled, w2_sampled_tmp, M, verbose = true, eps_theta = 0.01)#

ahat = solveahat(X_sampled, Y_sampled, w2_sampled, theta_t2[1:(length(theta_t2) - 1)], theta_t2[length(theta_t2)])
M = [findmax(data.V002[cluster .== cluster[i]])[1] for i in 1:length(data[:,1])]
M = [M[cluster .== idx][1] for idx in SampleIdx]

using Plots
display(plot(ahat, M, seriestype = :scatter, title = string("Nepal Data; Corr = ", round(cor(hcat(ahat, M))[1,2]; digits = 3)),
legend= false))
