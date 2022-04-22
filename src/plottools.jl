using StatsPlots, StatsBase, DelimitedFiles, LaTeXStrings, KernelDensity, Latexify, RCall
default(; # Plots defaults
    fontfamily="Computer modern",
    label="" # only explicit legend entries
    )
set_default(; # Latexify defaults
    #unitformat=:slash # in case you want `m/s`
    )
scalefontsizes(1.2)


function plot_rpacf(series)
    @rput series
    R"""
        library(TimeSeries.OBeu)
        tspacf = ts.acf(series)
        pacf = tspacf$pacf.parameters$pacf
        lag =  tspacf$pacf.parameters$pacf.lag
        up = tspacf$pacf.parameters$confidence.interval.up
        lo = tspacf$pacf.parameters$confidence.interval.low

    """
    @rget pacf
    @rget lag
    @rget up
    @rget lo

    ls = length(lag)
    print(ls)
    plt = plot(lag, pacf, line=:stem, grid=false, color=:black, lw=1.5,
                xlabel=L"\mathrm{lag}", ylabel=L"\mathrm{PACF}", xticks=0:5:20)
                
    plot!(plt, [0], seriestype=:hline, color=:black)
    plot!(plt, [up], seriestype=:hline, line=:dash, lw=1.5, color=:blue)
    plot!(plt, [lo], seriestype=:hline, line=:dash, lw=1.5, color=:blue)
    return plt
end


function plot_results(data, lag, ntrain, fit, sts, ŷ, ŷstd; kwargs...)
    nfull = size(data, 1)
    tsteps = 1:1:nfull

    plt = scatter(data, label=L"\mathrm{data}", grid=false; kwargs...)
    plot!(plt, 1:lag+1, data[1:lag+1], colour =:blue)
    plot!(plt, [ntrain], seriestype =:vline, colour = :green, linestyle =:dash, label = L"\mathrm{training\, data\, end}")
    lag += 1
    plot!(plt, tsteps[lag:ntrain], mean(fit,dims=1)', ribbon=sts, alpha=0.4, colour=:blue, label=L"\mathrm{fitted\, model}")
    plot!(plt, tsteps[ntrain+1:end], ŷ', ribbon=ŷstd, colour =:purple, alpha=0.4, label=L"\mathrm{preditions}")
    return plt
end


function explore_data(data)
    plt1 = plot(data)
    plt2 = histogram(data, bins = 50)
    plt3 = plot_rpacf(data)#plot(pacf(data, 1:20), line=:stem)
    plt4 = plot(qqplot(Normal(), data))
    
    plt = plot(plt1, plt2, plt3, plt4, layout=(2,2), size=(800, 800))
    return plt
end


