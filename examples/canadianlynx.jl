using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings
using Parameters: @with_kw
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/npbnn.jl")
include("../src/arbnn.jl")
include("../src/plottools.jl")
include("../src/R/RUtils.jl")

# load the data and log10 transform
data = log.(10, readdlm("./data/lynx.txt"));
plot(data, title="log10 canadian lynx data", legend=nothing)

# explore the data
# data_plot = explore_data(data) # requires R packages TimeSeries.OBeu and ts
# savefig(data_plot, "lynxexplore.png")
# split training data first 100 observations and generate the lagged time series via embed
lag = 2;
ytemp = data[1:end-14];
D = embed(ytemp, lag+1);

# train data
ytrain = convert(Array{Float64, 2}, hcat(D[:, 1]...));
xtrain = convert(Array{Float64, 2}, D[:, 2:end]');
ytest = data[101:end];


# for sd in 1:20
# initialize neural net
Random.seed!(2);g=NeuralNet(Chain(Dense(lag,10,tanh), Dense(10,1)));
# arguments for the main sampler
@with_kw mutable struct Args
    net = g
    maxiter = 40000 # maximum number of iterations
    burnin = 5000 # burnin iterations
    x = xtrain # lagged data
    y = ytrain
    geop = 0.5
    hyper_taus = [1. 1. ;1. 1.]
    ap = 1. # beta hyperparameter alpha for the geometric probability
    bp = 1. # beta hyperparameter beta for the geometric probability
    at = 0.05 # atoms  gamma hyperparameter alpha
    bt = 0.05 # atoms gamma hyperparameter beta
    ataus = 5ones(2,2) # Gamma hyperprior on network weights precision
    btaus = 5ones(2,2) # Gamma hyperprior on network weights precision
    seed = 123
    stepsize = 0.005
    numsteps = 20
    verb = 1000
    npredict = 14
    save = false
    filename = "/sims/lynx/npbnn/"
end
@time est = npbnn();


# check for thinning
acf = autocor(est.weights[2001:50:end,1], 1:20) ; # autocorrelation for lags 1:20
plot(acf, title = "Autocorrelation", legend = false, line=:stem)

ŷ = mean(hcat(est.predictions...)[2001:50:end, :], dims=1);
ŷstd = std(hcat(est.predictions...)[2001:50:end, :], dims=1);
metrics = evaluationmetrics(ŷ , ytest);
println(metrics)

thinned = est.weights[2001:10:end,:];
fit, sts = predictions(xtrain, thinned);
plot_results(data, lag, length(ytemp), fit, sts, ŷ, ŷstd; ylim=(1., 5))

# uncomment and change location accordingly
# savefig(newplt, "sims/lynx/npbnn/seed123/figures/fit-pred-std.pdf")

# clusters
clusters = est.clusters;
ergodic_cluster = cumsum(clusters)./collect(1:length(clusters));
clustersplt = plot(ergodic_cluster, ylim=(0,5), lw=1.5, grid=:false, title = "Ergodic means for #clusters",
    seriestype =:line, color = :black, label=:none, xlabel="iterations", ylabel="clusters");
iters=["0", "10000","20000","30000","40000"];
plot!(clustersplt ,xticks=(0:10000:40000,iters))


#=
Fit an ARMA model
=#
auto_arima(vec(ytrain), 15, 15)
# best comes out ARMA(2,2)
armafit, armapred = arima_fit_predict(vec(ytrain), 2, 2, 14);
armametrics = evaluationmetrics(armapred[:pred], ytest);
println(armametrics)


#=
autoregressive BNN with parametric, Gaussian noise
=#
Random.seed!(2);g=NeuralNet(Chain(Dense(2,10,tanh), Dense(10,1)))
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
    btaus = 5ones(2,2) # Gamma hyperprior on network weights precision
    seed = 1
    stepsize = 0.005
    numsteps = 20
    verb = 1000
    npredict = 14
    save=false
    filename = "/sims/lynx/arbnn"
end

@time pest = arbnn()

# check for thinning
acf = autocor(pest.weights[1:50:end,1], 1:20)  # autocorrelation for lags 1:20
plot(acf, title = "Autocorrelation", legend = false, line=:stem)

ŷ = mean(hcat(pest.predictions...)[2001:50:end, :], dims=1)
ŷstd = std(hcat(pest.predictions...)[2001:50:end, :], dims=1)
metrics = evaluationmetrics(ŷ , ytest);
println(metrics)


thinned = pest.weights[2001:50:end,:];
fit, sts = predictions(xtrain, thinned);
plot_results(data, lag, length(ytemp), fit, sts, ŷ, ŷstd; ylim=(1., 5))


