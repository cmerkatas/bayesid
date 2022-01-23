function npwishartbnn(; kws...)
    # load algorithms parameters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    savelocation = string(pwd(), args.filename, "/seed$(args.seed)/samples/")
    figlocation = string(pwd(), args.filename, "/seed$(args.seed)/figures/")
    mkpath(savelocation)
    mkpath(figlocation)

    g = args.net # nnet
    x, y = args.x, args.y
    ntemp = size(y, 2)
    maxiter, burnin = args.maxiter, args.burnin
    verbose_every = args.verb

    @assert size(x) == size(y)
    dstate, n = size(y)

    hyper_taus = args.hyper_taus

    geop, ap, bp = args.geop, args.ap, args.bp # alpha and beta hyperparameter for the geometric probability
    nu0, Lambda0 = args.ν₀, args.Λ₀ # dof and precision matrix hyperparameter

    # HMC tuning params
    stepsize = args.stepsize
    numsteps = args.numsteps

    # initialize vectors to store array
    sampled_ws = zeros(maxiter, length(g.ws))
    zp = zeros(dstate, maxiter - burnin)
    geoprob = zeros(maxiter)

    ataus = args.ataus #(alpha_w in layers;alpha_b in layers)
    btaus = args.btaus #(beta_w in layers;beta_b in layers)
    sampled_hypertaus = Array{Array{Float64, 2}}(undef, maxiter)

    # clustering and slice variables
    d, N = ones(Int64,n), ones(Int64,n)
    clusters = zeros(Int64, maxiter)

    # random density weights and locations
    w = zeros(maximum(N))
    tau = [diagm(ones(dstate)) for i in 1:1:maximum(N)]

    # sampler state
    current_ws = HMCState(g.ws, stepsize, numsteps)
    acc_ratio = 0.0

    # start main mcmc loop
    for its in 1:1:maxiter
        Nstar = maximum(N)

        # construct geometric weights
        w = geometricstickbreak(geop, Nstar)

        # sample atoms
        tau = sampleatomswishart(g, current_ws, y, x, d, Nstar, ν₀, Λ₀) # precision matrices tau

        # sample clustering variables and slice variables
        d, N = sampleclusterswishart(g, current_ws, y, x, tau, N, geop)
        clusters[its] = length(unique(d))

        # update geometric probability
        geop = sampleprob(ap, bp, n, N)
        geoprob[its] = geop

        # update prior sigma for the weights
        hyper_taus = samplehypertaus(g, hyper_taus, ataus, btaus)
        sampled_hypertaus[its] = hyper_taus

        tau_preconditioner = compute_preconditioner(g, hyper_taus)

        # sample network weights
        currwnt_ws, flag = samplestep(g, current_ws, UW, ∇UW, tau[d], tau_preconditioner, y, x)
        acc_ratio += flag
        sampled_ws[its,:] = current_ws.x


        # sample noise predictive
        if its > burnin
            zp[:, its-burnin] = samplepredictivewishart(w, tau, ν₀, Λ₀)

        end

        if mod(its, verbose_every) == 0
            println("MCMC iterations: $its out of $maxiter")
            println("Acceptance ratio: $(acc_ratio/its)")
            println("# of clusters: $(mean(clusters[its]))")
        end
    end

    # predict T future values
    T = args.npredict
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
