using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings
using Parameters: @with_kw
include("../data/datasets.jl")
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/reconstruct.jl")

# generate some data from chaotic cubic map
nf = 170
ntrain = 150
θ = [1.,0.,-1.71]
x₀ = .5
x = noisypolynomial(x₀, θ, noisemix2; n=nf, seed=11) # already "seeded"
plot(x)
# x = Flux.normalise(x, dims=1)
# assume that data are coming from order 1 markovian process then y = F(lagged_x)
D = embed(x, 2)
ytrain = hcat(copy(D[1:ntrain, 1])...)
xtrain = hcat(copy(D[1:ntrain, 2])...)

Random.seed!(1);g=NeuralNet(Chain(Dense(1,5,tanh), Dense(5,1)))
# arguments for the main sampler
@with_kw mutable struct Args
    net = g
    maxiter = 40000 # maximum number of iterations
    burnin = 5000 # burnin iterations
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
    seed = 3
    stepsize = 0.0015
    numsteps = 3
    verb = 1000
    npredict = 20
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

# prediction plot
tsteps=1:170;
plt = scatter(x, colour = :blue, label = "Data", ylim=(-1.5, 2), grid=:false)
plot!(plt, [150], seriestype =:vline, colour = :green, linestyle =:dash,label = "Training Data End")

thinned = est.weights[1:10:end,:];
fit, sts = predictions(xtrain, thinned);
plot!(plt, tsteps[1:150], mean(fit,dims=1)', colour =:black, label = "fitted model")
Flux.mse(mean(fit,dims=1), ytrain)


for i in 1:10:size(thinned, 1)
    plot!(plt, tsteps[1:150], fit[i,:], colour = :blue, lalpha = 0.04, label =:none)
end
plt
testpred = map(mean, est.predictions)


allpredictions = hcat(est.predictions...)
thinnedpredictions = allpredictions[1:10:end,:]
for i in 1:100:size(thinnedpredictions, 1)
    plot!(plt, tsteps[151:end], thinnedpredictions[i,:], colour =:purple, lalpha=0.04, label=:none)
end
plot!(plt, tsteps[151:end], testpred, color=:purple, label="prediction")
plt



allpredictions = hcat(est.predictions...)
thinnedpredictions = allpredictions[1:10:end,:]
zoomplt=scatter(tsteps[151:end], x[151:end], colour = :blue, label = "Data", ylim=(-1.5, 2), grid=:false)

plot!(zoomplt, tsteps[151:end], testpred, color=:purple, label="prediction")
for i in 1:100:size(thinnedpredictions, 1)
    plot!(zoomplt, tsteps[151:end], thinnedpredictions[i,:], colour =:purple, lalpha=0.04, label=:none)
end
zoomplt
plt

plot(zoomplt)
plot!(plt,
        inset = (zoomplt, bbox(0.05, 0.05, 0.5, 0.25, :bottom, :right)),
        ticks = nothing,
        subplot = zoomplt,
        bg_inside = nothing
)


plot(heatmap(randn(10,20)), boxplot(rand(1:4,1000),randn(1000)), leg=false)
histogram!(randn(1000), inset_subplots = [(1, bbox(0.05,0.95,0.5,0.5, v_anchor=:bottom))], subplot=3, ticks=nothing)


ss = linspace(0,2pi)
pp = plot(ss,sin(ss))
vline!(pp, [pi/2] )
#add more stuff to p
c = contour(x, y, z, inset=(bbox(...), p) ) #inset p into the contour plot
