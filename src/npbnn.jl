function npbnn(; kws...)
    # load algorithms parameters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    if args.save
        savelocation = string(pwd(), args.filename, "/seed$(args.seed)/samples/")
        figlocation =  string(pwd(), args.filename, "/seed$(args.seed)/figures/")
        mkpath(savelocation)
    end
    
    g = args.net # nnet
    x, y = args.x, args.y
    ntemp = length(y)
    maxiter, burnin = args.maxiter, args.burnin
    verbose_every = args.verb

    @assert size(x,2) == length(y)
    n = length(y)

    hyper_taus = args.hyper_taus

    geop, ap, bp = args.geop, args.ap, args.bp # alpha and beta hyperparameter for the geometric probability
    at, bt = args.at, args.bt # precision gamma hyperparameter alpha

    # HMC tuning params
    stepsize = args.stepsize
    numsteps = args.numsteps

    # initialize vectors to store array
    sampled_ws = zeros(maxiter, length(g.ws))
    zp = zeros(maxiter-burnin)
    geoprob = zeros(maxiter)

    ataus = args.ataus #(alpha_w in layers;alpha_b in layers)
    btaus = args.btaus #(beta_w in layers;beta_b in layers)
    sampled_hypertaus = Array{Array{Float64, 2}}(undef, maxiter)

    # clustering and slice variables
    d, N = ones(Int64,n), ones(Int64,n)
    clusters = zeros(Int64, maxiter)

    # random density weights and locations
    w = zeros(maximum(N))
    tau = ones(maximum(N))

    # sampler state
    current_ws = HMCState(g.ws, stepsize, numsteps)
    acc_ratio = 0.0

    # for model selection
    lppd = zeros(maxiter - burnin, n)
    pwaic2 = zeros(maxiter - burnin, n)

    # start main mcmc loop
    for its in 1:1:maxiter
        Nstar = maximum(N)

        # construct geometric weights
        w = geometricstickbreak(geop, Nstar)

        # sample atoms
        tau = sampleatoms(g, current_ws, y, x, d, Nstar, at, bt)

        # sample clustering variables and slice variables
        d, N = sampleclusters(g, current_ws, y, x, tau, N, geop)
        clusters[its] = length(unique(d))

        # update geometric probability
        geop = sampleprob(ap, bp, n, N)
        geoprob[its] = geop

        # update prior sigma for the weights
        hyper_taus = samplehypertaus(g, hyper_taus, ataus, btaus)
        sampled_hypertaus[its] = hyper_taus

        tau_preconditioner = compute_preconditioner(g, hyper_taus)

        # sample network weights
        currwnt_ws, flag = samplestep(g, current_ws, U, ∇U, tau[d], tau_preconditioner, y, x)
        acc_ratio += flag
        sampled_ws[its,:] = current_ws.x


        # sample noise predictive
        if its > burnin
            zp[its-burnin] = samplepredictive(w, tau, at, bt)
            # model selection
            for i in 1:1:n
                lf = logpdf(Normal(g(x[:, i], current_ws.x)[1], sqrt(1 / tau[d[i]])), y[i])
                pwaic2[its - burnin, i] = lf
                lppd[its - burnin, i] = exp(lf)

            end
        end

        if mod(its, verbose_every) == 0
            println("MCMC iterations: $its out of $maxiter")
            println("Acceptance ratio: $(acc_ratio/its)")
            println("# of clusters: $(mean(clusters[its]))")
        end
    end

    # predict T future values
    T = args.npredict
    lags = size(x, 1)
    if T > 0
        y = hcat(y, zeros(1, T))
        for t in 1:1:T
            x = hcat(x, reverse(y[ntemp+t-lags:ntemp+t-1]))
        end
    end
    preds = fill(Float64[], T)
    for t in 1:1:T
      preds[t] = zeros(maxiter)
    end
    for j in 1:1:size(sampled_ws, 1)
        for t in 1:1:T
            x[:, ntemp+t] = copy(reverse(y[ntemp+t-lags:ntemp+t-1]))
            meanstar = g(x[:, ntemp+t], sampled_ws[j, :])[1]
            y[ntemp+t] = meanstar
            preds[t][j] = y[ntemp+t]
        end
    end

    if args.save
        writedlm(string(savelocation, "sampled_weights.txt"), sampled_ws)
        writedlm(string(savelocation, "sampled_noise.txt"), zp)
        writedlm(string(savelocation, "sampled_clusters.txt"), clusters)
    end
    if T > 0
        if args.save
            for t in 1:T
                writedlm(string(savelocation, "sampled_pred$t.txt"), preds[t])
            end
        end
        return est=(weights=sampled_ws, noise=zp, clusters=clusters, precisions=sampled_hypertaus, waic=-2sum(log.(mean(lppd, dims=1)))+2sum(var(pwaic2, dims=1)), predictions=preds)
    else
        return est=(weights=sampled_ws, noise=zp, clusters=clusters, precisions=sampled_hypertaus, waic=-2sum(log.(mean(lppd, dims=1)))+2sum(var(pwaic2, dims=1)))
    end
end
