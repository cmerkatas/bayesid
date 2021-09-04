using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings
using Parameters: @with_kw
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/npbnn.jl")
include("../src/arbnn.jl")

Data = readdlm("./data/usgnp.txt")
plot(Data, title="usgnp", legend=nothing)
dmean, dstd = mean(Data), std(Data)
data = (Data .- dmean) ./ dstd
plot(data)
# split training data first 100 observations and generate the lagged time series via embed
ytemp = data[1:end-12]

D = embed(ytemp, 4)

# train data
ytrain = convert(Array{Float64, 2}, hcat(D[:, 1]...))
xtrain = convert(Array{Float64, 2}, D[:, 2:end]')
ytest = data[165:end]


# initialize neural net
Random.seed!(1);g=NeuralNet(Chain(Dense(3,10,tanh), Dense(10,1)))
# arguments for the main sampler
@with_kw mutable struct Args
    net = g
    maxiter = 20000 # maximum number of iterations
    burnin = 10000 # burnin iterations
    x = xtrain # lagged data
    y = ytrain
    geop = 0.5
    hyper_taus = 0.01*[1. 1. ;1. 1.]
    ap = 1. # beta hyperparameter alpha for the geometric probability
    bp = 1. # beta hyperparameter beta for the geometric probability
    at = 0.05 # atoms  gamma hyperparameter alpha
    bt = 0.05 # atoms gamma hyperparameter beta
    ataus = 0.01ones(2,2) # Gamma hyperprior on network weights precision
    btaus = 0.01ones(2,2) # IG hyperprior on network weights precision
    seed = 123
    stepsize = 0.05
    numsteps = 10
    verb = 1000
    npredict = 12
    filename = "/sims/usgnp/npbnn/"
end
@time est = npbnn();

burnin=10000
# check for thinning
acf = autocor(est.weights[burnin+1:end,1], 1:20)  # autocorrelation for lags 1:20
plot(acf, title = "Autocorrelation", legend = false, line=:stem)


ŷ = mean(hcat(est.predictions...)[burnin+1:end, :], dims=1)
ŷstd = std(hcat(est.predictions...)[burnin+1:end, :], dims=1)
metrics = evaluationmetrics(ŷ , ytest);
println(metrics)


# prediction plot with stds
tsteps=1:164;
ntemp=164
newplt = StatsPlots.scatter(data, colour = :blue, label = "data", ylim = (-5, 5), grid=:false)
plot!(newplt, [164], seriestype =:vline, colour = :green, linestyle =:dash, label = "training data end")

thinned = est.weights[1:end,:];
fit, sts = predictions(xtrain, thinned);
plot!(newplt, tsteps[size(xtrain,1)+1:164], mean(fit,dims=1)', colour=:black, label=nothing)
plot!(newplt, tsteps[size(xtrain,1)+1:164], mean(fit,dims=1)', ribbon=sts, alpha=0.4, colour =:blue, label="np-bnn fitted model")
plot!(newplt, collect(165:176), ŷ', ribbon=dstd.*ŷstd, colour =:purple, alpha=0.4, label="np-bnn preditions")
display(newplt)
