using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings
using Parameters: @with_kw
include("../data/datasets.jl")
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/reconstruct.jl")

# load the data and log10 transform
data = log.(readdlm("data/lynx.txt"))
plot(data, title="log10 canadian lynx data", legend=nothing)

# split training data first 100 observations
# and generate the lagged time series via embed
ytemp = data[1:end-14]
D = embed(ytemp, 3)
# train data
ytrain = convert(Array{Float64, 2}, hcat(D[:, 1]...))
xtrain = convert(Array{Float64, 2}, D[:, 2:end]')

# initialize neural net
Random.seed!(2);g=NeuralNet(Chain(Dense(2,10,tanh), Dense(10,1)))

# arguments for the main sampler
@with_kw mutable struct Args
    net = g
    maxiter = 40000 # maximum number of iterations
    burnin = 5000 # burnin iterations
    x = xtrain # lagged data
    y = ytrain
    geop = 0.5
    hyper_taus = 5.0.*[1. 1. ;1. 1.]
    ap = 1. # beta hyperparameter alpha for the geometric probability
    bp = 1. # beta hyperparameter beta for the geometric probability
    at = 0.05 # atoms  gamma hyperparameter alpha
    bt = 0.05 # atoms gamma hyperparameter beta
    ataus = 5ones(2,2) # Gamma hyperprior on network weights precision
    btaus = 5ones(2,2) # IG hyperprior on network weights precision
    seed = 1234
    stepsize = 0.005
    numsteps = 20
    verb = 1000
    npredict = 14
end

@time est = reconstruct();

# prediction plot
tsteps=1:114;
plt = scatter(data, colour = :blue, label = "Data", ylim = (3, 11.), grid=:false);
plot!(plt, [100], seriestype =:vline, colour = :green, linestyle =:dash,label = "Training Data End")

thinned = est.weights[1:10:end,:];
fit = predictions(xtrain, thinned);
plot!(plt, tsteps[3:100], mean(fit,dims=1)', colour =:black, label = "fitted model")
Flux.mse(mean(fit,dims=1), ytrain)


for i in 1:5:size(thinned, 1)
    plot!(plt, tsteps[3:100], fit[i,:], colour = :blue, lalpha = 0.04, label =:none)
end

testpred = zeros(14)
for t in 1:14
    testpred[t] = mean(est.predictions[t][1:1:end])
end
Flux.mse(testpred, data[end-13:end])

# psigmas = map(std, est.predictions)
# plt
# plot!(plt, tsteps[101:end], testpred, colour =:purple, ribbon=1.96.*psigmas, fillalpha=0.08, label="prediction")
# savefig(plt, "sims/lynx/figures/lynx-predictions-std.pdf")

allpredictions = hcat(est.predictions...)
thinnedpredictions = allpredictions[1:10:end,:]
mses = zeros(35000)
for t in 1:length(mses)
    mses[t] = Flux.mse(allpredictions[t,:], data[end-13:end])
end
minmse, idx = findmin(mses)
for i in 1:5:size(thinnedpredictions, 1)
    plot!(plt, tsteps[101:end], thinnedpredictions[i,:], colour =:purple, lalpha=0.04, label=:none)
end
plt
plot!(plt, tsteps[101:end], testpred, colour =:purple, label="prediction")

# clusters
clusters = est.clusters
ergodic_cluster = cumsum(clusters)./collect(1:length(clusters))
clusters_plt = plot(ergodic_cluster, ylim=(0,5), lw=1.5, grid=:false, title = "Ergodic means for #clusters",
    seriestype =:line, color = :black, label=:none, xlabel="iterations", ylabel="clusters")
iters=["0", "10000","20000","30000","40000"]
plot!(clusters_plt ,xticks=(0:10000:40000,iters))

# prediction plot with stds
tsteps=1:114;
newplt = scatter(data, colour = :blue, label = "Data", ylim = (3, 11.), grid=:false);
plot!(newplt, [100], seriestype =:vline, colour = :green, linestyle =:dash,label = "Training Data End")

thinned = est.weights[1:10:end,:];
fit, sts = predictions(xtrain, thinned);
plot!(newplt, tsteps[3:100], mean(fit,dims=1)', ribbon = sts, alpha=0.4, colour =:blue, label = "fitted model")
Flux.mse(mean(fit,dims=1), ytrain)

predsts = std(allpredictions, dims=1)
plot!(newplt, tsteps[101:end], mean(thinnedpredictions, dims=1)', ribbon=predsts, colour =:purple, alpha=0.4, label="preditions")
