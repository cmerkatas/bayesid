# bayesid
Tools for Bayesian system identification.

bayesid delivers methods for System Identification using Bayesian neural networks with nonparametric noise processes [1]. Additionally, a autoregressive Bayesian neural network similar to [2] is implemented.

# Dependencies
It is assumed that the user has or is able to install ```julia```. This project has been developed in ```julia v1.6.0```.

# Code structure
The core files regarding the methods illustrated in [1] are located in ```src``` folder. Subfolders ```mcmc``` and ```models``` contain code for the Hamiltonian Monte Carlo sampler [*] and ```structs``` to implement neural networks using ```Flux```, similar to neural networks implemented at ```Pkg DiffEqFlux.jl```.

Scripts for simulating the logistic data as well as the real data used in the paper are located in ```data``` folder.

The folder ```examples``` contains simple scripts for running the sampler on data from ```data``` folder.

[*] hmc.jl is based on an implementation in SGMCMC.jl--see ```src/mcmc/LICENSE```.

# Environment activation
In order to install all dependencies in the current state, start ```julia``` from the location
```bayesid```  is downloaded.
The project folder contains ```Manifest.toml``` and ```Project.toml``` files so you can type in the ```REPL```

```julia
julia> ]
(@v1.6) pkg> activate .
(bayesid)> instantiate
```
This will make ```julia``` install all the necessary packages and dependencies for the project.

# A simulated data example
Having complete the installation process we can run an example by first loading necessary packages and files
```julia
using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings
using Parameters: @with_kw
include("../data/datasets.jl")
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/npbnn.jl")
```

Next, we generate some data from logistic map

```julia
nf = 210
ntrain = 200
θ = [1.,0.,-1.71]
x₀ = .5
data = noisypolynomial(x₀, θ, noisemix2; n=nf, seed=11)
plot(data)
```

Assuming that the data are coming from order 1 Markovian process then we construct the delayed time series
```julia
ytemp = data[1:ntrain]
D = embed(ytemp, 2)
ytrain = hcat(copy(D[:, 1])...)
xtrain = hcat(copy(D[:, 2])...)
ytest = data[ntrain+1:end]
```
Finally, a neural network is defined and a mutable struct containing the model and HMC parameters specification is defined. We identify the system using the ```npbnn``` method.

```julia
Random.seed!(1);g=NeuralNet(Chain(Dense(1,10,tanh), Dense(10,1)))
# arguments for the main sampler
@with_kw mutable struct Args
    net = g
    maxiter = 40000 # maximum number of iterations
    burnin = 2000 # burnin iterations
    x = xtrain # lagged data
    y = ytrain
    geop = 0.5
    hyper_taus = [1. 1.;1. 1.]
    ap = 1. # beta hyperparameter alpha for the geometric probability
    bp = 1. # beta hyperparameter beta for the geometric probability
    at = 3. # atoms  gamma hyperparameter alpha
    bt = 0.001 # atoms gamma hyperparameter beta
    ataus = 5ones(2,2) # Gamma hyperprior on network weights precision
    btaus = 5ones(2,2) # IG hyperprior on network weights precision
    seed = 1
    stepsize = 0.0015
    numsteps = 3
    verb = 1000
    npredict = 10
    filename = "/sims/logistic/npbnn/"
end

@time est = npbnn();
```
# References
[1] Merkatas, C., & Särkkä, S. (2021). *System identification using Bayesian neural networks with nonparametric noise processes.* (submitted)

[2] Nakada, Y., Matsumoto, T., Kurihara, T., & Yosui, K. (2005). *Bayesian reconstructions and predictions of nonlinear dynamical systems via the hybrid Monte Carlo scheme*. Signal processing, 85(1), 129-145.
