function arbnn(; kws...)
    # load algorithms parameters
    args = PArgs(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    savelocation = string(pwd(), args.filename, "/seed$(args.seed)/samples/")
    figlocation =  string(pwd(), args.filename, "/seed$(args.seed)/figures/")
    mkpath(figlocation)
    mkpath(savelocation)

    g = args.net # nnet
    x, y = args.x, args.y
    ntemp = length(y)
    maxiter, burnin = args.maxiter, args.burnin
    verbose_every = args.verb

    @assert size(x,2)==length(y)
    n = length(y)

    hyper_taus = args.hyper_taus

    at, bt = args.at, args.bt # precision gamma hyperparameter for parametric noise

    # HMC tuning params
    stepsize = args.stepsize
    numsteps = args.numsteps

    # initialize vectors to store array
    sampled_ws = zeros(maxiter, length(g.ws))
    sampledtau = zeros(maxiter)

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
        hyper_taus = samplehypertaus(g, hyper_taus, ataus, btaus)
        sampled_hypertaus[its] = hyper_taus

        tau_preconditioner = compute_preconditioner(g, hyper_taus)

        # sample network weights
        currwnt_ws, flag = samplestep(g, current_ws, U, ∇U, tau, tau_preconditioner, y, x)
        acc_ratio += flag
        sampled_ws[its,:] = current_ws.x

        sampledtau[its] = tau


        if mod(its, verbose_every)==0
            println("MCMC iterations: $its out of $maxiter")
            println("Acceptance ratio: $(acc_ratio/its)")
        end
    end

    # predict future T values
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
    if T > 0
        for j in 1:1:size(sampled_ws, 1)
            for t in 1:1:T
                x[:,ntemp+t] = copy(reverse(y[ntemp+t-lags:ntemp+t-1]))
                meanstar = g(x[:, ntemp+t], sampled_ws[j, :])[1]
                varstar = 1.0 ./ sampledtau[j]
                y[ntemp+t] = rand(Normal(meanstar, sqrt(varstar)))
                preds[t][j] = y[ntemp+t]
            end
        end
    end


    writedlm(string(savelocation, "sampledweights.txt"), sampled_ws)
    writedlm(string(savelocation, "sampledtaus.txt"), sampledtau)
    writedlm(string(savelocation, "sampledprecisions.txt"), sampled_hypertaus)

    if T > 0
        for t in 1:T
            writedlm(string(savelocation, "sampled_pred$t.txt"), preds[t])
        end
        return est=(weights=sampled_ws[burnin+1:end, :], taus=sampledtau, precisions=sampled_hypertaus, predictions=preds)
    else
        return est=(weights=sampled_ws[burnin+1:end, :], taus=sampledtau, precisions=sampled_hypertaus)
    end
end
