using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings
using Parameters: @with_kw
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/reconstruct.jl")
include("../src/bnnparametric.jl")

# load the data and log10 transform
data = log.(10, readdlm("./data/lynx.txt"))
plot(data, title="log10 canadian lynx data", legend=nothing)

# split training data first 100 observations and generate the lagged time series via embed
ytemp = data[1:end-14]
D = embed(ytemp, 3)

# train data
ytrain = convert(Array{Float64, 2}, hcat(D[:, 1]...))
xtrain = convert(Array{Float64, 2}, D[:, 2:end]')
ytest = data[101:end]
# end

nsims = 10
chainpredictionsmeans = zeros(nsims, 14)
chainpredictionsmedians = similar(chainpredictionsmeans)
for sd in 1:10
    # initialize neural net
    Random.seed!(sd+1);g=NeuralNet(Chain(Dense(2,10,tanh), Dense(10,1)))
    # arguments for the main sampler
    @with_kw mutable struct Args
        net = g
        maxiter = 40000 # maximum number of iterations
        burnin = 2000 # burnin iterations
        x = xtrain # lagged data
        y = ytrain
        geop = 0.5
        hyper_taus = [1. 1. ;1. 1.]
        ap = 1. # beta hyperparameter alpha for the geometric probability
        bp = 1. # beta hyperparameter beta for the geometric probability
        at = 0.05 # atoms  gamma hyperparameter alpha
        bt = 0.05 # atoms gamma hyperparameter beta
        ataus = 5ones(2,2) # Gamma hyperprior on network weights precision
        btaus = 5ones(2,2) # IG hyperprior on network weights precision
        seed = sd
        stepsize = 0.005
        numsteps = 20
        verb = 1000
        npredict = 14
        filename = "/sims/lynx/chains/npbnn/"
    end
    @time est = reconstruct();

    chainpredictionsmeans[sd, :] = mean(hcat(est.predictions...)[2001:50:end, :], dims=1)
    chainpredictionsmedians[sd, :] = median(hcat(est.predictions...)[2001:50:end, :], dims=1)

    println("Done with chain: $sd\n")
end

writedlm("sims/lynx/chains/npbnn/chainpredictionsmeans.txt", chainpredictionsmeans)
writedlm("sims/lynx/chains/npbnn/chainpredictionsmedians.txt", chainpredictionsmedians)

chainpredictionsmedians=copy(chainpredictionsmdedians)

std(chainpredictionsmeans, dims=1)


"""
Parametric
"""
nsims = 10
pchainpredictionsmeans = zeros(nsims, 14)
pchainpredictionsmedians = similar(pchainpredictionsmeans)
for sd in 1:10
    # initialize neural net
    Random.seed!(sd+1);g=NeuralNet(Chain(Dense(2,10,tanh), Dense(10,1)))

    @with_kw mutable struct PArgs
        net = g
        maxiter = 40000 # maximum number of iterations
        burnin = 5000 # burnin iterations
        x = xtrain # lagged data
        y = ytrain
        hyper_taus = [1. 1.;1. 1.]
        at = 0.05 # parametric precision  gamma hyperparameter alpha
        bt = 0.05 # parametric gamma hyperparameter beta
        ataus = 5ones(2,2) # Gamma hyperprior on network weights precision
        btaus = 5ones(2,2) # IG hyperprior on network weights precision
        seed = sd
        stepsize = 0.005
        numsteps = 20
        verb = 1000
        npredict = 14
        filename = "/sims/lynx/chains/bnnparametric"
    end

    @time pest = bnnparametric()

    pchainpredictionsmeans[sd, :] = mean(hcat(pest.predictions...)[2001:50:end, :], dims=1)
    pchainpredictionsmedians[sd, :] = median(hcat(pest.predictions...)[2001:50:end, :], dims=1)

    println("Done with chain: $sd\n")
end

writedlm("sims/lynx/chains/bnnparametric/chainpredictionsmeans.txt", chainpredictionsmeans)
writedlm("sims/lynx/chains/bnnparametric/chainpredictionsmedians.txt", chainpredictionsmedians)


std(chainpredictionsmeans, dims=1)
