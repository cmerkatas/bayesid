using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings, BSON, RCall
using Parameters: @with_kw
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/arbnn.jl")

R"""
    library(astsa)
    D = fmri1
"""
@rget D

data = (D[:, 5] .- mean(D[:, 5])) ./ std(D[:, 5])
plot(data)

plot(qqplot(Normal(), data))
histogram(data)
plot(autocor(data, 1:20), line=:stem)
plot(pacf(data, 1:20), line=:stem)

# split training data first 100 observations and generate the lagged time series via embed
ytemp = data[1:end-14]
lag = 10
D = embed(ytemp, lag+1)


# train data
ytrain = convert(Array{Float64, 2}, hcat(D[:, 1]...))
xtrain = convert(Array{Float64, 2}, D[:, 2:end]')

ytest = data[115:end]
Random.seed!(2);g=NeuralNet(Chain(Dense(10,20,tanh), Dense(20,1)))
@with_kw mutable struct PArgs
    net = g
    maxiter = 40000 # maximum number of iterations
    burnin = 5000 # burnin iterations
    x = xtrain # lagged data
    y = ytrain
    hyper_taus = [1. 1.;1. 1.]
    at = 0.01 # parametric precision  gamma hyperparameter alpha
    bt = 0.01 # parametric gamma hyperparameter beta
    ataus = 0.1ones(2,2) # Gamma hyperprior on network weights precision
    btaus = 0.1ones(2,2) # IG hyperprior on network weights precision
    seed = 1
    stepsize = 0.015
    numsteps = 6
    verb = 1000
    npredict = 14
    save=false
    filename = "/sims/fmri/arbnn"
end

@time pest = arbnn()

# check for thinning
acf = autocor(pest.weights[1:10:end,1], 1:20)  # autocorrelation for lags 1:20
plot(acf, title = "Autocorrelation", legend = false, line=:stem)

ŷ = mean(hcat(pest.predictions...)[5001+1:50:end, :], dims=1)
ŷstd = std(hcat(pest.predictions...)[5001+1:50:end, :], dims=1)
metrics = evaluationmetrics(ŷ , ytest);
println(metrics)
# uncomment and change location accordingly
# writedlm("sims/lynx/bnnparametric/seed1/metrics.txt", hcat(metrics...))
# writedlm("sims/lynx/bnnparametric/seed1/ypred.txt", vcat(ŷ,ŷstd)')

# prediction plot with stds
tsteps=1:128;
newplt = scatter(data, colour = :blue, label = "Data", grid=:false);
plot!(newplt, [114], seriestype =:vline, colour = :green, linestyle =:dash, label = "training data end")

thinned = pest.weights[2001:50:end,:];
fit, sts = predictions(xtrain, thinned);
plot!(newplt, tsteps[size(xtrain,1)+1:114], mean(fit,dims=1)', colour=:black, label=nothing);
plot!(newplt, tsteps[size(xtrain,1)+1:114], mean(fit,dims=1)', ribbon=sts, alpha=0.4, colour =:blue, label="fitted model");
plot!(newplt, tsteps[length(ytemp)+1:end], ŷ', ribbon=ŷstd, colour =:purple, alpha=0.4, label="preditions")
display(newplt)
# uncomment and change location accordingly
# savefig(newplt, "sims/lynx/bnnparametric/seed1/figures/bnnparametricfit-pred-std.pdf")

plot((data .- mean(data)) ./ std(data))