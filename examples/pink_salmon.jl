using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings, BSON
using Parameters: @with_kw
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/arbnn.jl")

dat = BSON.load("data/pinkSalmon_first30_log.bson")
y = deepcopy(dat[:y])
X = deepcopy(dat[:X])

plot(y)
n, K = length(y), size(X, 2)
y_fit = deepcopy(y[1:n])
X_fit = deepcopy(X[1:n, 1:K])

X_fit = (X_fit .- mean(y_fit)) ./ std(y_fit)  # time series
y_fit = (y_fit .- mean(y_fit)) ./ std(y_fit)

X_fit = Matrix(X_fit')
y_fit = Matrix(y_fit')

"""
Parametric
"""
Random.seed!(2);g=NeuralNet(Chain(Dense(5,10,tanh), Dense(10,1)))
@with_kw mutable struct PArgs
    net = g
    maxiter = 40000 # maximum number of iterations
    burnin = 5000 # burnin iterations
    x = X_fit # lagged data
    y = y_fit
    hyper_taus = ones(2, 2)#[1. 1.;1. 1.]
    at = 0.05 # parametric precision  gamma hyperparameter alpha
    bt = 0.05 # parametric gamma hyperparameter beta
    ataus = 5ones(2,2) # Gamma hyperprior on network weights precision
    btaus = 5ones(2,2) # IG hyperprior on network weights precision
    seed = 1
    stepsize = 0.05
    numsteps = 10
    verb = 100
    npredict = 0
    save=false
    filename = "/sims/salmon/arbnn"
end

@time pest = arbnn()


# check for thinning
acf = autocor(pest.weights[1:50:end,1], 1:20)  # autocorrelation for lags 1:20
plot(acf, title = "Autocorrelation", legend = false, line=:stem)

thinned = pest.weights[2001:50:end,:];
fit, sts = predictions(X_fit, thinned);
newplt=plot(1:size(X_fit,2), mean(fit,dims=1)', colour=:black, label=nothing);
plot!(newplt, 1:size(X_fit,2), mean(fit,dims=1)', ribbon=sts, alpha=0.4, colour =:blue, label="fitted model");
display(newplt)
# uncomment and change location accordingly
# savefig(newplt, "sims/lynx/bnnparametric/seed1/figures/bnnparametricfit-pred-std.pdf")
