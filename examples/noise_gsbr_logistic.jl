Pkg.activate(".")
using Plots, Random, DelimitedFiles, Flux, LaTeXStrings, KernelDensity, StatsPlots, Distributions
include("../data/datasets.jl")
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/npbnn.jl")
include("/Users/cmerkatas/github/GSBR/src/ToolBox.jl")

"""
load the results for noise and plot
"""
npnoise = readdlm("sims/logistic/npbnn/seed1/samples/sampled_noise.txt")
gsbnoise = readdlm("../GSBR/sims/logistic/gsbr_seed1/noise.txt")
gsbnoise = gsbnoise[5001:end]
partaus = readdlm("sims/logistic/arbnn/seed1/samples/sampledtaus.txt")
parsigma = sqrt(1.0./mean(partaus))

zpl, zpr = -0.5, 0.5;
npnoise = npnoise[npnoise.>-0.5];npnoise = npnoise[npnoise.<0.5];
gsbnoise = gsbnoise[gsbnoise.>-0.5];gsbnoise = gsbnoise[gsbnoise.<0.5];

xrange = range(zpl, stop=zpr, length=120);
ddnoise = noisemixdensity2.(xrange);
noiseplot = plot(xrange, ddnoise,
        xlabel="z", ylabel="Density",
        color=:black, lw=1.2, label="true f(z)", grid=:false)

plot!(noiseplot, kde(npnoise), color=:blue, lw=1.3, label="np-bnn", grid=:false)
plot!(noiseplot, kde(gsbnoise), line = (:dash, 1.2, :red), label="gsbr", grid=:false)
plot!(noiseplot, xrange, pdf.(Normal(0, parsigma), xrange), lw=1.2, color=:orange, label="ar-bnn", grid=:false)
# savefig(noiseplot, "sims/logistic/Figs/noisecomparison.pdf")

"""
load the results for estimation and prediction from GSBR
"""
# generate the data
# generate some data from logistic map
nf = 210
ntrain = 200
θ = [1.,0.,-1.71]
x₀ = .5
data = noisypolynomial(x₀, θ, noisemix2; n=nf, seed=11) # already "seeded"
ytest = data[ntrain+1:end]

gsbrpredictions = zeros(35000, 10)
for t in 1:1:10
        gsbrpredictions[:, t] = readdlm("../GSBR/sims/logistic/gsbr_seed1/xpred$t.txt")
end

gsbpredhat = mean(gsbrpredictions, dims=1)
gsbmetrics = evaluationmetrics(gsbpredhat, ytest')

# metrics from nnet based models are in the associated examples.
