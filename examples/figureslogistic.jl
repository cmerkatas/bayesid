Pkg.activate(".")
using Plots, Random, DelimitedFiles, Flux, LaTeXStrings, KernelDensity, StatsPlots, Distributions
include("../data/datasets.jl")
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/reconstruct.jl")
include("/Users/cmerkatas/github/GSBR/src/ToolBox.jl")

""" load the results for noise and plot
"""
npnoise = readdlm("sims/newlog/samples/sampled_noise.txt")
gsbnoise = readdlm("../GSBR/sims/logistic/gsbr_seed1/noise.txt")
gsbnoise = gsbnoise[5001:end]
partaus = readdlm("sims/parametric/parametric1/samples/samplessampledtaus.txt")[5001:end]
parsigma = sqrt(1.0./mean(partaus))

zpl, zpr = -0.5, 0.5;
npnoise = npnoise[npnoise.>-0.5];npnoise = npnoise[npnoise.<0.5];
gsbnoise = gsbnoise[gsbnoise.>-0.5];gsbnoise = gsbnoise[gsbnoise.<0.5];

xrange = range(zpl, stop=zpr, length=120);
ddnoise = noisemixdensity2.(xrange);
noiseplot = plot(xrange, ddnoise,
        xlabel="z", ylabel="Density",
        color=:black, lw=1.2, label="true f(z)", grid=:false)

plot!(noiseplot, kde(npnoise), color=:blue, lw=1.3, label="npbnn", grid=:false)
plot!(noiseplot, kde(npnoise), line = (:dash, 1.2, :red), label="gsbr", grid=:false)
plot!(noiseplot, xrange, pdf.(Normal(0, parsigma), xrange), lw=1.2, color=:orange, label="arbnn", grid=:false)
savefig(noiseplot, "sims/newlog/figures/noisecomparison.pdf")

"""
load the results for estimation and prediction
"""

# generate the data
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
g=NeuralNet(Chain(Dense(1,10,tanh), Dense(10,1)))

nppredictions = zeros(35000, 10)
gsbrpredictions = similar(nppredictions)
parpredictions = similar(nppredictions)
for t in 1:1:10
        nppredictions[:, t] = readdlm("sims/newlog/samples/sampled_pred$t.txt")
        gsbrpredictions[:, t] = readdlm("../GSBR/sims/logistic/gsbr_seed1/xpred$t.txt")
        parpredictions[:, t] = readdlm("sims/parametric/parametric1/samples/samplessampled_pred$t.txt")
end

# load the weights
npweights = readdlm("sims/newlog/samples/sampled_weights.txt")
parweights = readdlm("sims/newlog/samples/sampled_weights.txt")
thetas =   readdlm("../GSBR/sims/logistic/gsbr_seed1/thetas.txt")
tsteps = 1:nf
gsbfit = zeros(size(thetas, 1), length(xtrain))
gsbfit[:,1] = map(θ -> polyMap(θ, xtrain[1]), thetas[:,1])
nt, nx = size(gsbfit)
for i in 1:1:nt
    for j in 2:1:nx
        gsbfit[i, j] = polyMap(thetas[i, :], gsbfit[i, j-1])
    end
end
gsbsts = std(gsbfit, dims=1)

# std plots
stdplt = scatter(data, colour = :blue, label = "Data", ylim=(-1.5, 2), grid=:false)
plot!(stdplt, [ntrain], seriestype =:vline, colour = :green, linestyle =:dash, label = "Training Data End")

npfit, npsts = predictions(xtrain, npweights);
plot!(stdplt, tsteps[2:ntrain], mean(npfit,dims=1)', ribbon = npsts, alpha=0.4, colour =:blue, label = "fitted model")
plot!(stdplt, tsteps[ntrain+1:end], mean(nppredictions, dims=1)', color=:purple, ribbon=std(nppredictions,dims=1)', alpha=0.4,label="prediction")

# predsplt = scatter(tsteps[ntrain+1:end], ytest, colour = :blue, label = "Data", ylim=(-1.5, 2), grid=:false)
# plot!(predsplt, tsteps[ntrain+1:end], mean(nppredictions, dims=1)', color=:purple, ribbon=std(nppredictions, dims=1)', alpha=0.4, label="npbnn prediction")
# plot!(predsplt, tsteps[ntrain+1:end], mean(gsbrpredictions, dims=1)', color=:red, ribbon=std(gsbrpredictions, dims=1)', alpha=0.2, label="gsbr prediction")
# plot!(predsplt, tsteps[ntrain+1:end], mean(parpredictions, dims=1)', color=:orange, ribbon=std(gsbrpredictions, dims=1)', alpha=0.4, label="gsbr prediction")

nppredhat = mean(nppredictions, dims=1)
gsbpredhat = mean(gsbrpredictions, dims=1)
parpredhat = mean(parpredictions, dims=1)

Flux.mse(nppredhat, ytest)
Flux.mse(gsbpredhat, ytest)
Flux.mse(parpredhat, ytest)

Flux.mae(nppredhat, ytest)
Flux.mae(gsbpredhat, ytest)
Flux.mae(parpredhat, ytest)
