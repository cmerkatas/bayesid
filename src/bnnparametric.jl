function bnnparametric(; kws...)
    # load algorithms parameters
    args = PArgs(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    savelocation = string(pwd(), args.filename, "/parametric$(args.seed)/samples")
    mkpath(savelocation)

    g = args.net # nnet
    x, y = args.x, args.y
    ntemp = length(y)
    maxiter, burnin = args.maxiter, args.burnin
    verbose_every = args.verb
    T = args.npredict
    lags = size(x,1)
    if T > 0
        y = hcat(y, zeros(1,T))
        for t in 1:T
            x = hcat(x, reverse(y[ntemp+t-lags:ntemp+t-1]))
        end
    end
    preds = fill(Float64[], T)
    for t in 1:1:T
      preds[t] = zeros(maxiter-burnin)
    end

    @assert size(x,2)==length(y)
    n = length(y)

    hyper_taus = args.hyper_taus

    at, bt = args.at, args.bt # precision gamma hyperparameter for parametric noise

    # HMC tuning params
    stepsize = args.stepsize
    numsteps = args.numsteps

    # initialize vectors to store array
    sampled_ws = zeros(maxiter, length(g.ws))
    sampledtau = zeros(maxiter-burnin)

    ataus = args.ataus #(alpha_w in layers;alpha_b in layers)
    btaus = args.btaus #(beta_w in layers;beta_b in layers)
    sampled_hypertaus = Array{Array{Float64, 2}}(undef, maxiter)


    # random density weights and locations
    tau = 0.001 # initial precision

    # sampler state
    current_ws = HMCState(g.ws, stepsize, numsteps)
    acc_ratio = 0.0

    # start main mcmc loop
    for its in 1:maxiter

        # sample atoms
        alphastar = at + 0.5n
        betastar = bt + 0.5sum((y - g(x, current_ws.x)).^2)
        tau = rand(Gamma(alphastar, 1.0/betastar))

        # update prior sigma for the weights
        #prior_sigma = samplesigma(g, current_ws, y, x, as, bs)
        hyper_taus = samplehypertaus(g, hyper_taus, ataus, btaus)
        sampled_hypertaus[its] = hyper_taus

        tau_preconditioner = compute_preconditioner(g, hyper_taus)

        # sample network weights
        currwnt_ws, flag = samplestep(g, current_ws, U, ∇U, tau, tau_preconditioner, y, x)
        acc_ratio += flag
        sampled_ws[its,:] = current_ws.x


        # sample predictive
        if its > burnin
            sampledtau[its-burnin] = tau

            if T > 0
                for t in 1:T
                    meanstar = g(x[:,ntemp+t], current_ws.x)[1]
                    varstar = 1.0 ./ tau
                    y[ntemp+t] = rand(Normal(meanstar, sqrt(varstar)))
                    x[:,ntemp+t] = copy(reverse(y[ntemp+t-lags:ntemp+t-1]))
                    preds[t][its-burnin] = y[ntemp+t]
                end
            end
        end

        if mod(its, verbose_every)==0
            println("MCMC iterations: $its out of $maxiter")
            println("Acceptance ratio: $(acc_ratio/its)")
        end
    end

    writedlm(string(savelocation, "sampledweights.txt"), sampled_ws)
    writedlm(string(savelocation, "sampledtaus.txt"), sampledtau)
    writedlm(string(savelocation, "sampledprecisions.txt"), sampled_hypertaus)

    if T > 0
        for t in 1:T
            writedlm(string(savelocation, "sampled_pred$t.txt"), preds[t])
        end
        return est=(weights=sampled_ws[burnin+1:end,:], taus=sampledtau, precisions=sampled_hypertaus, predictions=preds)
    else
        return est=(weights=sampled_ws[burnin+1:end,:], taus=sampledtau, precisions=sampled_hypertaus)
    end
end
