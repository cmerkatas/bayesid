using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings
using Parameters: @with_kw
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/npbnn.jl")
include("../src/arbnn.jl")
include("../data/datasets.jl")

# generate data from henon map
θ = [1.38, -1, 0.27]
x₀ = [1, 0.2]
nf = 210
ntrain = 200
data = noisyhenon(x₀, θ, mvmix1; n=nf, seed=6)
plot(data[1,:], data[2,:], seriestype=:scatter)

ytrain = copy(data[:, 2:end])
xtrain = copy(data[:, 1:end-1])
ytest = copy(data[:, ntrain+1:end])

Random.seed!(1);g=NeuralNet(Chain(Dense(2,5,tanh), Dense(5,2)))
# arguments for the main sampler
maxiter = 20000 # maximum number of iterations
burnin = 10000 # burnin iterations
x = xtrain # lagged data
y = ytrain
geop = 0.5
hyper_taus = [1. 1.;1. 1.]
ap = 1. # beta hyperparameter alpha for the geometric probability
bp = 1. # beta hyperparameter beta for the geometric probability
ν₀ = 2 # atoms  gamma hyperparameter alpha
Σ₀ = [1e3 -1;-1 1e3]
ataus = 5ones(2,2) # Gamma hyperprior on network weights precision
btaus = 5ones(2,2) # IG hyperprior on network weights precision
seed = 1
stepsize = 0.0015
numsteps = 3
verb = 100
T = 10
filename = "/sims/henon/npbnn/"

@time est = mvbnn(y, x, T, g, maxiter, burnin, ν₀, Σ₀, geop, ap, bp, hyper_taus, ataus, btaus, stepsize, numsteps;
                    filename = filename, seed = seed, verb);
