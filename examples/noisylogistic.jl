using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings
using Parameters: @with_kw
include("../data/datasets.jl")
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/reconstruct.jl")
include("../src/bnnparametric.jl")

# generate some data from logistic map
nf = 210
ntrain = 200
θ = [1.,0.,-1.71]
x₀ = .5
data = noisypolynomial(x₀, θ, noisemix2; n=nf, seed=11) # already "seeded"
plot(data)

# assume that data are coming from order 1 markovian process then y = F(lagged_x)
ytemp = data[1:ntrain]
D = embed(ytemp, 2)
ytrain = hcat(copy(D[:, 1])...)
xtrain = hcat(copy(D[:, 2])...)
ytest = data[ntrain+1:end]

Random.seed!(1);g=NeuralNet(Chain(Dense(1,10,tanh), Dense(10,1)))
# arguments for the main sampler
@with_kw mutable struct Args
    net = g
    maxiter = 40000 # maximum number of iterations
    burnin = 5000 # burnin iterations
    x = xtrain # lagged data
    y = ytrain
    geop = 0.5
    hyper_taus = 5.0*[1. 1.;1. 1.]
    ap = 1. # beta hyperparameter alpha for the geometric probability
    bp = 1. # beta hyperparameter beta for the geometric probability
    at = 3. # atoms  gamma hyperparameter alpha
    bt = 0.001 # atoms gamma hyperparameter beta
    ataus = 5ones(2,2) # Gamma hyperprior on network weights precision
    btaus = 5ones(2,2) # IG hyperprior on network weights precision
    seed = 10
    stepsize = 0.0015
    numsteps = 3
    verb = 1000
    npredict = 10
end

@time est = reconstruct();


zpl, zpr = -0.5, 0.5;
zp = est.noise;
zp = zp[zp.>zpl];
zp = zp[zp.<zpr];

noiseplot = plot(kde(zp), color=:red, lw=1.5, label=L"\mathrm{est }\hat{f}(z)", grid=:false) #histogram(zp, bins=200, normalize=:pdf, color=:lavender, label="predictive");
x_range = range(zpl, stop=zpr, length=120);
ddnoise = noisemixdensity2.(x_range);
plot!(noiseplot, x_range, ddnoise, color=:black, lw=1.5, label=L"\mathrm{true } f(z)", ylim=(0,25), ylabel="pdf")


allpredictions = hcat(est.predictions...)
thinnedpredictions = allpredictions[1:10:end,:]
mses = zeros(3500)
for t in 1:length(mses)
    mses[t] = Flux.mse(thinnedpredictions[t,:], data[end-9:end])
end
minmse, idx = findmin(mses)

bestplt = scatter(tsteps[ntrain+1:end], ytest, colour = :blue, label = "Data", ylim=(-1.5, 2), grid=:false)
plot!(bestplt, tsteps[ntrain+1:end], mean(thinnedpredictions, dims=1)', color=:purple, ribbon=std(thinnedpredictions, dims=1)', alpha=0.4,label="prediction")

Flux.mse(mean(allpredictions,dims=1), ytest)
Flux.mae(mean(allpredictions,dims=1), ytest)



"""
Parametric
"""
# assume that data are coming from order 1 markovian process then y = F(lagged_x)
ytemp = data[1:ntrain]
D = embed(ytemp, 2)
ytrain = hcat(copy(D[:, 1])...)
xtrain = hcat(copy(D[:, 2])...)
ytest = data[ntrain+1:end]
Random.seed!(1);g=NeuralNet(Chain(Dense(1,10,tanh), Dense(10,1)))
@with_kw mutable struct PArgs
    net = g
    maxiter = 40000 # maximum number of iterations
    burnin = 5000 # burnin iterations
    x = xtrain # lagged data
    y = ytrain
    hyper_taus = 5.0*[1. 1.;1. 1.]
    at = 3. # parametric precision  gamma hyperparameter alpha
    bt = 0.001 # parametric gamma hyperparameter beta
    ataus = 5ones(2,2) # Gamma hyperprior on network weights precision
    btaus = 5ones(2,2) # IG hyperprior on network weights precision
    seed = 9
    stepsize = 0.0015
    numsteps = 3
    verb = 1000
    npredict = 10
    filename = "/sims/parametric"
end


@time pest = bnnparametric()

allpredictions = hcat(pest.predictions...)
Flux.mse(mean(allpredictions,dims=1), ytest)
Flux.mae(mean(allpredictions,dims=1), ytest)
