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

# for sd in 1:20
# initialize neural net
Random.seed!(2);g=NeuralNet(Chain(Dense(2,10,tanh), Dense(10,1)))
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
    seed = 123
    stepsize = 0.005
    numsteps = 20
    verb = 1000
    npredict = 14
    filename = "/sims/lynx/npbnn/"
end
@time est = reconstruct();

# check for thinning
acf = autocor(est.weights[2001:50:end,1], 1:20)  # autocorrelation for lags 1:20
plot(acf, title = "Autocorrelation", legend = false, line=:stem)

ŷ = mean(hcat(est.predictions...)[2001:50:end, :], dims=1)
ŷstd = std(hcat(est.predictions...)[2001:50:end, :], dims=1)
metrics = evaluationmetrics(ŷ , ytest);
println(metrics)
# writedlm("sims/lynx/npbnn/seed123/metrics.txt", hcat(metrics...))
# writedlm("sims/lynx/npbnn/seed123/ypred.txt", vcat(ŷ,ŷstd)')

# prediction plot with stds
tsteps=1:114;
newplt = StatsPlots.scatter(data, colour = :blue, label = "data", ylim = (1, 5.), grid=:false);
plot!(newplt, [100], seriestype =:vline, colour = :green, linestyle =:dash, label = "training data end");

thinned = est.weights[2001:50:end,:];
fit, sts = predictions(xtrain, thinned);
plot!(newplt, tsteps[size(xtrain,1)+1:100], mean(fit,dims=1)', colour=:black, label=nothing);
plot!(newplt, tsteps[size(xtrain,1)+1:100], mean(fit,dims=1)', ribbon=sts, alpha=0.4, colour =:blue, label="np-bnn fitted model");
plot!(newplt, tsteps[length(ytemp)+1:end], ŷ', ribbon=ŷstd, colour =:purple, alpha=0.4, label="np-bnn preditions")
display(newplt)
# savefig(newplt, "sims/lynx/npbnn/seed123/figures/fit-pred-std.pdf")

# clusters
clusters = est.clusters;
ergodic_cluster = cumsum(clusters)./collect(1:length(clusters));
clustersplt = plot(ergodic_cluster, ylim=(0,5), lw=1.5, grid=:false, title = "Ergodic means for #clusters",
    seriestype =:line, color = :black, label=:none, xlabel="iterations", ylabel="clusters");
iters=["0", "10000","20000","30000","40000"];
plot!(clustersplt ,xticks=(0:10000:40000,iters))
savefig(clustersplt, "sims/lynx/npbnn/seed123/figures/clusters.pdf")


"""
Parametric
"""
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
    btaus = 5ones(2,2) # IG hyperprior on network weights precision
    seed = 1
    stepsize = 0.005
    numsteps = 20
    verb = 1000
    npredict = 14
    filename = "/sims/lynx/bnnparametric"
end

@time pest = bnnparametric()

# check for thinning
acf = autocor(pest.weights[1:50:end,1], 1:20)  # autocorrelation for lags 1:20
plot(acf, title = "Autocorrelation", legend = false, line=:stem)

ŷ = mean(hcat(pest.predictions...)[2001:50:end, :], dims=1)
ŷstd = std(hcat(pest.predictions...)[2001:50:end, :], dims=1)
metrics = evaluationmetrics(ŷ , ytest);
println(metrics)
# writedlm("sims/lynx/bnnparametric/seed1/metrics.txt", hcat(metrics...))
# writedlm("sims/lynx/bnnparametric/seed1/ypred.txt", vcat(ŷ,ŷstd)')

# prediction plot with stds
tsteps=1:114;
newplt = scatter(data, colour = :blue, label = "Data", ylim = (1, 5.), grid=:false);
plot!(newplt, [100], seriestype =:vline, colour = :green, linestyle =:dash, label = "training data end");

thinned = pest.weights[2001:50:end,:];
fit, sts = predictions(xtrain, thinned);
plot!(newplt, tsteps[size(xtrain,1)+1:100], mean(fit,dims=1)', colour=:black, label=nothing);
plot!(newplt, tsteps[size(xtrain,1)+1:100], mean(fit,dims=1)', ribbon=sts, alpha=0.4, colour =:blue, label="fitted model");
plot!(newplt, tsteps[length(ytemp)+1:end], ŷ', ribbon=ŷstd, colour =:purple, alpha=0.4, label="preditions")
display(newplt)
savefig(newplt, "sims/lynx/bnnparametric/seed1/figures/bnnparametricfit-pred-std.pdf")
