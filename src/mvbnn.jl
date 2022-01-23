function mvbnn( y::Matrix{Float64},
                x::Matrix{Float64},
                T::Int64,
                g::NeuralNet,
                maxiter::Int64,
                burnin::Int64,
                ν₀::Int64,
                Σ₀::Array{Float64, 2},
                geop::Float64, ap::Float64, bp::Float64,
                hyper_taus::Array{Float64, 2},
                ataus::Array{Float64, 2}, btaus::Array{Float64, 2},
                stepsize::Float64, numsteps::Int64;
                filename="@__DIR__", seed=1, verb=10)

    Random.seed!(seed)
    savelocation = string(pwd(), filename, "/seed$(seed)/samples/")

    mkpath(savelocation)

    ntemp = size(y, 2)
    @assert size(x) == size(y)
    dstate, n = size(y)

    # initialize vectors to store array
    sampled_ws = zeros(maxiter, length(g.ws))
    zp = zeros(dstate, maxiter - burnin)
    geoprob = zeros(maxiter)
    sampled_hypertaus = Array{Array{Float64, 2}}(undef, maxiter)

    # clustering and slice variables
    d, N = ones(Int64, n), ones(Int64, n)
    clusters = zeros(Int64, maxiter)

    # random density weights and locations
    w = zeros(maximum(N))
    Λ = [diagm(ones(dstate)) for i in 1:1:maximum(N)]

    # sampler state
    current_ws = HMCState(g.ws, stepsize, numsteps)
    acc_ratio = 0.0


    # start MCMC loop
    for its in 1:1:maxiter

        Nstar = maximum(N)

        # construct weights
        w = geometricstickbreak(geop, Nstar)

        # sample atoms
        for j in 1:1:Nstar
            Sj, nj = zeros(dstate, dstate), 0
            for i in 1:1:n
                if d[i] .== j
                    nj += 1
                    @inbounds Sj .+= (view(y, :, i) .- g(view(x, :, i))) * reshape(view(y, :, i) .- g(view(x, :, i)), 1, dstate)
                end
            end
            lambdatemp = cholesky(Hermitian(inv(inv(Σ₀) + Sj)))
            @inbounds Λ[j] = rand(Wishart(ν₀ + nj, lambdatemp))
        end

        # sample clusters
        for i in 1:1:n
            nc, prob = 0, 0
            mean_vec = g(x[:, i], current_ws.x)
            for j in 1:1:N[i]
                covmat = Symmetric(inv(Λ[j]))
                @inbounds nc += w[j] * pdf(MultivariateNormal(mean_vec, covmat), y[:, i])#det(Λ[j])^(0.5) * exp.(-0.5 * reshape(view(y, :, i) .- g(view(x, :, i)), 1, 2) * Λ[j] * (view(y, :, i) .- g(view(x, :, i))))
            end
            rd = rand()
            for j in 1:1:N[i]
                covmat = Symmetric(inv(Λ[j]))
                @inbounds prob +=  w[j] * pdf(MultivariateNormal(mean_vec, covmat), y[:, i]) / nc#det(Λ[j])^(0.5) * exp.(-0.5 * transpose(view(y, :, i) .- g(view(x, :, i))) * Λ[j] * (view(y, :, i) .- g(view(x, :, i)))) / nc
                if rd < prob
                    d[i] = j
                    N[i] = tgeornd(geop, d[i])
                    break
                end
            end
        end

        clusters[its] = length(unique(d))

        # update geometric probability
        geop = rand(Beta(ap + 2n, bp + sum(N) - n))
        geoprob[its] = geop

        # update prior sigma for the weights
        hyper_taus = samplehypertaus(g, hyper_taus, ataus, btaus)
        sampled_hypertaus[its] = hyper_taus

        tau_preconditioner = compute_preconditioner(g, hyper_taus)

        # sample network weights
        currwnt_ws, flag = samplestep(g, current_ws, UW, ∇UW, Λ[d], tau_preconditioner, y, x)
        acc_ratio += flag
        sampled_ws[its,:] = current_ws.x

        # sample noise predictive
        if its > burnin
            W = cumsum(w)
            rp = rand()
            if rp > W[end]
                Λstar = rand(Wishart(ν₀, Σ₀))
                zp[:, its-burnin] = rand(MultivariateNormal(zeros(dstate), Λstar \ I))
            else
                for j in 1:1:length(W)
                    if rp < W[j]
                        covmat = Symmetric(inv(Λ[j]))
                        zp[:, its - burnin] =  rand(MultivariateNormal(zeros(dstate), covmat))
                        break
                    end
                end
            end
        end

        # fill in necessary precision matrices for next iter
        if maximum(N) > Nstar
            for add in 1:1:maximum(N) - Nstar
                push!(Λ, diagm(ones(dstate)))
            end
        end


        if mod(its, verb) == 0
            println("MCMC iterations: $its out of $maxiter")
            println("Acceptance ratio: $(acc_ratio/its)")
            println("# of clusters: $(mean(clusters[its]))")
        end
    end

    # predict T future values
    lags = 1#size(x, 1)
    if T > 0
        y = hcat(y, zeros(2, T))
        for t in 1:1:T
            x = hcat(x, reverse(y[:, ntemp+t-lags:ntemp+t-1]))
        end
    end
    preds = [zeros(2, maxiter) for t in 1:T]

    for j in 1:1:size(sampled_ws, 1)
        for t in 1:1:T
            x[:, ntemp+t] = copy(reverse(y[:, ntemp+t-lags:ntemp+t-1]))
            meanstar = g(x[:, ntemp+t], sampled_ws[j, :])
            y[:, ntemp+t] = meanstar
            preds[t][:, j] = y[:, ntemp+t]
        end
    end

    writedlm(string(savelocation, "sampled_weights.txt"), sampled_ws)
    writedlm(string(savelocation, "sampled_noise.txt"), zp)
    writedlm(string(savelocation, "sampled_clusters.txt"), clusters)

    if T > 0
        for t in 1:T
            writedlm(string(savelocation, "sampled_pred$t.txt"), preds[t])
        end
        return est=(weights=sampled_ws, noise=zp, clusters=clusters, precisions=sampled_hypertaus, predictions=preds)
    else
        return est=(weights=sampled_ws, noise=zp, clusters=clusters, precisions=sampled_hypertaus)
    end
end
