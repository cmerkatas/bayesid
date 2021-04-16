using Plots, Random, Distributions, Flux, Zygote, LinearAlgebra, StatsBase, StatsPlots, KernelDensity
using DelimitedFiles, LaTeXStrings
using Parameters: @with_kw
include("../data/datasets.jl")
include("../src/models/bnn.jl")
include("../src/mcmc/hmc.jl")
include("../src/utils.jl")
include("../src/npbnn.jl")
include("../src/arbnn.jl")

# generate some data from logistic map
nf = 210
ntrain = 200
θ = [1.,0.,-1.71]
x₀ = .5
data = noisypolynomial(x₀, θ, noisemix2; n=nf, seed=11)
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


zpl, zpr = -0.5, 0.5;
zp = est.noise;
zp = zp[zp.>zpl];
zp = zp[zp.<zpr];

noiseplot = plot(kde(zp), color=:red, lw=1.5, label=L"\mathrm{est }\hat{f}(z)", grid=:false);
x_range = range(zpl, stop=zpr, length=120);
ddnoise = noisemixdensity2.(x_range);
plot!(noiseplot, x_range, ddnoise, color=:black, lw=1.5, label=L"\mathrm{true } f(z)", ylim=(0,25), ylabel="pdf")

# check for thinning
acf = autocor(est.weights[2001:50:end,1], 1:20)  # autocorrelation for lags 1:20
plot(acf, title = "Autocorrelation", legend = false, line=:stem)

ŷ = mean(hcat(est.predictions...)[2001:50:end, :], dims=1)
ŷstd = std(hcat(est.predictions...)[2001:50:end, :], dims=1)
metrics = evaluationmetrics(ŷ , ytest);
println(metrics)
# writedlm("sims/logistic/npbnn/seed1/metrics.txt", hcat(metrics...))
# writedlm("sims/logistic/npbnn/seed1/ypred.txt", vcat(ŷ,ŷstd)')

# prediction plot with stds
tsteps=1:210;
newplt = StatsPlots.scatter(data, colour = :blue, label = "data", ylim = (-1.5, 2.), grid=:false);
plot!(newplt, [200], seriestype =:vline, colour = :green, linestyle =:dash, label = "training data end");

thinned = est.weights[2001:50:end,:];
fit, sts = predictions(xtrain, thinned);
plot!(newplt, tsteps[size(xtrain,1)+1:200], mean(fit,dims=1)', colour=:black, label=nothing);
plot!(newplt, tsteps[size(xtrain,1)+1:200], mean(fit,dims=1)', ribbon=sts, alpha=0.4, colour =:blue, label="np-bnn fitted model");
plot!(newplt, tsteps[length(ytemp)+1:end], ŷ', ribbon=ŷstd', colour =:purple, alpha=0.4, label="np-bnn preditions")
display(newplt)
# savefig(newplt, "sims/logistic/npbnn/seed1/figures/logisticfit-pred-std.pdf")

# zoomed plot
tsteps = 1:210
zoomplt = scatter(tsteps[ntrain+1:end], ytest, colour = :blue, label = "data", ylim=(-1.5, 2), grid=:false)
plot!(zoomplt, tsteps[ntrain+1:end], ŷ', color=:purple, ribbon=ŷstd', alpha=0.4,label="np-bnn prediction")
# savefig(zoomplt, "sims/logistic/npbnn/seed1/figures/logisticzoompred-std.pdf")



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
    hyper_taus = [1. 1.;1. 1.]
    at = 3. # parametric precision  gamma hyperparameter alpha
    bt = 0.001 # parametric gamma hyperparameter beta
    ataus = 5ones(2,2) # Gamma hyperprior on network weights precision
    btaus = 5ones(2,2) # IG hyperprior on network weights precision
    seed = 1
    stepsize = 0.005
    numsteps = 20
    verb = 1000
    npredict = 10
    filename = "/sims/logistic/arbnn"
end


@time pest = arbnn()
xrange=range(-0.5, stop=0.5, length=100)
varhat = 1/sqrt(mean(pest.taus[2001:10:end]))
noiseplot = plot(xrange, pdf.(Normal(0,varhat), xrange), color=:red, lw=1.5, label=L"\mathrm{est }\hat{f}(z)", grid=:false) #histogram(zp, bins=200, normalize=:pdf, color=:lavender, label="predictive");
x_range = range(zpl, stop=zpr, length=120);
ddnoise = noisemixdensity2.(x_range);
plot!(noiseplot, x_range, ddnoise, color=:black, lw=1.5, label=L"\mathrm{true } f(z)", ylim=(0,25), ylabel="pdf")

# check for thinning
acf = autocor(pest.weights[2001:50:end,1], 1:20)  # autocorrelation for lags 1:20
plot(acf, title = "Autocorrelation", legend = false, line=:stem)

ŷ = mean(hcat(pest.predictions...)[2001:50:end, :], dims=1)
ŷstd = std(hcat(pest.predictions...)[2001:50:end, :], dims=1)
metrics = evaluationmetrics(ŷ , ytest);
println(metrics)
# writedlm("sims/logistic/arbnn/seed1/metrics.txt", hcat(metrics...))
# writedlm("sims/logistic/arbnn/seed1/ypred.txt", vcat(ŷ,ŷstd)')

# prediction plot with stds
tsteps=1:210;
newplt = scatter(data, colour = :blue, label = "Data", ylim = (-1.5, 2.), grid=:false);
plot!(newplt, [200], seriestype =:vline, colour = :green, linestyle =:dash, label = "training data end");

thinned = pest.weights[2001:10:end,:];
fit, sts = predictions(xtrain, thinned);
plot!(newplt, tsteps[size(xtrain,1)+1:200], mean(fit,dims=1)', colour=:black, label=nothing);
plot!(newplt, tsteps[size(xtrain,1)+1:200], mean(fit,dims=1)', ribbon=sts, alpha=0.4, colour =:blue, label="fitted model");
plot!(newplt, tsteps[length(ytemp)+1:end], ŷ', ribbon=ŷstd, colour =:purple, alpha=0.4, label="preditions")
display(newplt)
# savefig(newplt, "sims/logistic/arbnn/seed1/figures/logisticfit-pred-std.pdf")
