embed(x,k) = hcat([x[i+k-1:-1:i] for i in 1:length(x)-k+1]...)'

function geometricstickbreak(p::Float64, Nstar::Int64)
    w = zeros(Nstar)
    for j in 1:1:Nstar
        w[j] = p * (1.0 - p)^(j - 1)
    end
    return w
end


function sampleatoms(g::NeuralNet, ws::HMCState, y::Array{Float64,2}, x::Array{Float64,2}, d::Array{Int64,1}, Nstar::Int64, a::Float64, b::Float64)
    @assert size(x,2)==length(y)
    @assert (a > zero(a)) & (b > zero(b))
    n = size(x,2)
    atoms = zeros(Nstar)
    @inbounds for j in 1:1:Nstar
        term, counts = 0.0, 0
        @inbounds for i in 1:1:n
            if d[i].==j
                counts += 1
                term += (y[i] - g(x[:,i], ws.x)[1])^2
            end
        end
        atoms[j] = rand(Gamma(a + 0.5counts, (b + 0.5term)^(-1)))
    end
    return atoms
end


function tgeornd(pp::Float64, k::Int64)
    return rand(Geometric(pp)) + k
end


function sampleclusters(g::NeuralNet, ws::HMCState, y::Array{Float64,2}, x::Array{Float64,2}, atoms::Array{Float64,1}, Nis::Array{Int64,1}, geop::Float64)
    @assert size(x,2)==length(y)
    n = size(x,2)
    clusters, slices = zeros(Int64,n), zeros(Int64,n)
    @inbounds for i in 1:1:n
        nc = 0
        @inbounds for k in 1:Nis[i]
            nc += atoms[k]^(0.5) * exp(-0.5atoms[k] * (y[i] - g(x[:,i], ws.x)[1]).^2)
        end
        probs = 0.0
        uu = rand()
        @inbounds for k in 1:Nis[i]
            probs += atoms[k]^(0.5) * exp(-0.5atoms[k] * (y[i] - g(x[:,i], ws.x)[1]).^2) / nc
            if uu < probs
                clusters[i] = k
                slices[i] = tgeornd(geop, clusters[i])
                break
            end
        end
    end

    return clusters, slices
end


function sampleprob(ap::Float64, bp::Float64, n::Int64, N::Array{Int64,1})
    @assert (ap > zero(ap)) & (bp > zero(bp))
    return rand(Beta(ap + 2n, bp + sum(N) - n))
end


function samplepredictive(w::Array{Float64,1}, atoms::Array{Float64,1}, a::Float64, b::Float64)
    @assert (a > zero(a)) & (b > zero(b))
    W = cumsum(w)
    rd = rand()
    if rd .> W[end]
        tau_star = rand(Gamma(a, 1.0/b))
        return rand(Normal(0.0, 1.0/tau_star^0.5))
    else
        cmpnent = StatsBase.sample(Weights(w))
        tau_star = atoms[cmpnent]
        return rand(Normal(0.0, 1/atoms[cmpnent]^0.5))
    end
end


function predictions(x::Array{Float64,2}, weights::Array{Float64,2})
    np, _ = size(weights)
    preds = zeros(np, size(x,2))
    @inbounds for i in 1:1:np
        preds[i,:] = g(x, weights[i,:])
    end
    stds = std(preds, dims=1)
    return preds, stds
end


function samplehypertaus(g::NeuralNet, hyper_taus::Array{Float64, 2}, alphas, betas)
    updated_taus = similar(hyper_taus)
    for l in 1:length(g.nnet)
        weights, biases = g.nnet[l].W, g.nnet[l].b
        alpha_weights = alphas[1,l] + 0.5length(weights)
        alpha_biases = alphas[2,l] + 0.5length(biases)

        beta_weights = betas[1, l] + 0.5norm(weights)^2
        beta_biases = betas[2, l] + 0.5norm(biases)^2
        updated_taus[1, l] = rand(Gamma(alpha_weights, 1.0/beta_weights))
        updated_taus[2, l] = rand(Gamma(alpha_biases, 1.0/beta_biases))
    end
    return updated_taus
end


function predict(g::NeuralNet, ws::HMCState, x_::Float64, sig::Float64)
    meanstar = g([x_], ws.x)[1]
    return rand(Normal(meanstar, sig))
end

function predictions(x::Array{Float64,2}, weights::Array{Float64,2})
    np, _ = size(weights)
    preds = zeros(np, size(x,2))
    @inbounds for i in 1:1:np
        preds[i,:] = g(x, weights[i,:])
    end
    stds = std(preds, dims=1)
    return preds, stds
end

function evaluationmetrics(ŷ, y)
    # mse
    ŷ = vcat(ŷ...)
    y = vcat(y...)
    mse = mean((ŷ - y).^2)
    rmse = sqrt(mse)

    # mae
    mae = Flux.mae(ŷ, y)

    # mape
    mape = mean(abs.((ŷ - y) ./ y)) * 100

    # theil's U statistic
    u1 = sqrt(sum((ŷ - y).^2)) / sqrt(sum(y.^2))
    rf2 = sqrt(mean(ŷ.^2))
    ry2 = sqrt(mean(y.^2))
    u2 = rmse / (rf2 + ry2)

    return metrics = (mse = mse, rmse = rmse, mae = mae, mape = mape, u1 = u1, u2 = u2)
end
