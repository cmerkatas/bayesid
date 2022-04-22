using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings, BSON, RCall
using Parameters: @with_kw
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/arbnn.jl")

R"""
    library(astsa)
    D = Hare
"""
@rget D
data = log.(10, D)
data_plot = explore_data(data)

# split training data first 100 observations and generate the lagged time series via embed
lag = 4
npred = 9
ytrain, xtrain, ytest, ntrain = split_data(data, lag, npred);

Random.seed!(2);g=NeuralNet(Chain(Dense(lag, 20, tanh), Dense(20, 1)))
@with_kw mutable struct PArgs
    net = g
    maxiter = 50000 # maximum number of iterations
    burnin = 10000 # burnin iterations
    x = xtrain # lagged data
    y = ytrain
    hyper_taus = [1. 1.;1. 1.]
    at = 0.05 # parametric precision  gamma hyperparameter alpha
    bt = 0.05 # parametric gamma hyperparameter beta
    ataus = 5ones(2,2) # Gamma hyperprior on network weights precision
    btaus = 5ones(2,2) # Gamma hyperprior on network weights precision
    seed = 1
    stepsize = 0.0015
    numsteps = 10
    verb = 1000
    npredict = 9
    save=false
    filename = "/sims/hare/arbnn"
end

@time pest = arbnn();


ŷ = mean(hcat(pest.predictions...)[1:10:end, :], dims=1)
ŷstd = std(hcat(pest.predictions...)[10:10:end, :], dims=1)
metrics = evaluationmetrics(ŷ , ytest);
println(metrics)

thinned = pest.weights[1:20:end,:];
fit, sts = predictions(xtrain, thinned);
plot_results(data, lag, ntrain, fit, std(data).*sts, ŷ, std(data).*ŷstd; legend=:bottomleft)