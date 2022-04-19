using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings, BSON
using Parameters: @with_kw
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/arbnn.jl")

# load the data and log10 transform
data = readdlm("./data/oldfaithful.txt")[:,1]
data = (data .- mean(data)) ./ std(data)
plot(data, title="old faithful data", legend=nothing)


plot(qqplot(Normal, data))
plot(pacf(data, 1:10), line=:stem)

# split training data first 100 observations and generate the lagged time series via embed
ytemp = data[1:end-16]
D = embed(ytemp, 2)

# train data
ytrain = convert(Array{Float64, 2}, hcat(D[:, 1]...))

xtrain = convert(Array{Float64, 2}, hcat(D[:, 2:end]...))
ytest = data[136:end]
# end
"""
Parametric
"""
Random.seed!(2);g=NeuralNet(Chain(Dense(1,10,tanh), Dense(10,1)))
@with_kw mutable struct PArgs
    net = g
    maxiter = 40000 # maximum number of iterations
    burnin = 5000 # burnin iterations
    x = xtrain # lagged data
    y = ytrain
    hyper_taus = ones(2, 2)#[1. 1.;1. 1.]
    at = 0.001 # parametric precision  gamma hyperparameter alpha
    bt = 0.001 # parametric gamma hyperparameter beta
    ataus = 0.001ones(2,2) # Gamma hyperprior on network weights precision
    btaus = 0.001ones(2,2) # IG hyperprior on network weights precision
    seed = 1
    stepsize = 0.05
    numsteps = 10
    verb = 1000
    npredict = 16
    save=false
    filename = "/sims/faithful/arbnn"
end

@time pest = arbnn()


# check for thinning
acf = autocor(pest.weights[1:20:end,1], 1:20)  # autocorrelation for lags 1:20
plot(acf, title = "Autocorrelation", legend = false, line=:stem)

ŷ = mean(hcat(pest.predictions...)[2001:20:end, :], dims=1)
ŷstd = std(hcat(pest.predictions...)[2001:20:end, :], dims=1)
metrics = evaluationmetrics(ŷ , ytest);
println(metrics)


tsteps=1:151;
newplt=StatsPlots.scatter(data, colour = :blue, label = "data", grid=:false)
plot!(newplt, [135], seriestype =:vline, colour = :green, linestyle =:dash, label = "training data end")

thinned = pest.weights[2001:20:end,:];
fit, sts = predictions(xtrain, thinned);
plot!(newplt, tsteps[size(xtrain,1)+1:135], mean(fit,dims=1)', colour=:black, label=nothing)
plot!(newplt, tsteps[size(xtrain,1)+1:135], mean(fit,dims=1)', ribbon=sts, alpha=0.4, colour =:blue, label="np-bnn fitted model")
plot!(newplt, tsteps[length(ytemp)+1:end], ŷ', ribbon=ŷstd, colour =:purple, alpha=0.4, label="ar-bnn preditions")
display(newplt)
