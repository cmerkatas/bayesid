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
data_plot = explore_data(data)

# split training data first 100 observations and generate the lagged time series via embed
ytrain, xtrain, ytest, ntrain = split_data(data, 14, 12);

miact(x) = x + sin(x)^2
Random.seed!(2);g=NeuralNet(Chain(Dense(lag, 20, miact), Dense(20, 1)))
@with_kw mutable struct PArgs
    net = g
    maxiter = 50000 # maximum number of iterations
    burnin = 10000 # burnin iterations
    x = xtrain # lagged data
    y = ytrain
    hyper_taus = 5 .* [1. 1.;1. 1.]
    at = 0.001 # parametric precision  gamma hyperparameter alpha
    bt = 0.001 # parametric gamma hyperparameter beta
    ataus = 5ones(2,2) # Gamma hyperprior on network weights precision
    btaus = 5ones(2,2) # IG hyperprior on network weights precision
    seed = 1
    stepsize = 0.015
    numsteps = 6
    verb = 1000
    npredict = 14
    save=false
    filename = "/sims/fmri/arbnn"
end

@time pest = arbnn();

# check for thinning
acf = autocor(pest.weights[1:10:end,1], 1:20)  # autocorrelation for lags 1:20
plot(acf, title = "Autocorrelation", legend = false, line=:stem)

ŷ = mean(hcat(pest.predictions...)[5001+1:10:end, :], dims=1)
ŷstd = std(hcat(pest.predictions...)[5001+10:10:end, :], dims=1)
metrics = evaluationmetrics(ŷ , ytest);
println(metrics)

thinned = pest.weights[1:20:end,:];
fit, sts = predictions(xtrain, thinned);
plot_results(data, lag, ntrain, fit, std(D[:, 5]).*sts, ŷ, std(D[:, 5]).*ŷstd)