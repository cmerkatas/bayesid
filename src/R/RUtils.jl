using RCall
using StatsPlots, StatsBase, DelimitedFiles, LaTeXStrings, KernelDensity, Latexify, RCall
default(; # Plots defaults
    fontfamily="Computer modern",
    label="" # only explicit legend entries
    )
set_default(; # Latexify defaults
    #unitformat=:slash # in case you want `m/s`
    )
scalefontsizes(1.2)


function plot_acf(series)
    @rput series
    R"""
        library(TimeSeries.OBeu)
        tsacf = ts.acf(series)
        acf = tsacf$acf.parameters$acf
        lag =  tspacf$acf.parameters$acf.lag
        up = tspacf$acf.parameters$confidence.interval.up
        lo = tspacf$acf.parameters$confidence.interval.low

    """
    @rget acf
    @rget lag
    @rget up
    @rget lo

    ls = length(lag)
    print(ls)
    plt = plot(lag, acf, line=:stem, grid=false, color=:black, lw=1.5,
                xlabel=L"\mathrm{lag}", ylabel=L"\mathrm{ACF}", xticks=0:5:20)
                
    plot!(plt, [0], seriestype=:hline, color=:black)
    plot!(plt, [up], seriestype=:hline, line=:dash, lw=1.5, color=:blue)
    plot!(plt, [lo], seriestype=:hline, line=:dash, lw=1.5, color=:blue)
    return plt
end


function plot_pacf(series)
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
    plt = plot(lag, pacf, line=:stem, grid=false, color=:black, lw=1.5, xlabel=L"\mathrm{lag}", ylabel=L"\mathrm{PACF}", xticks=0:5:20)
                
    plot!(plt, [0], seriestype=:hline, color=:black);
    plot!(plt, [up], seriestype=:hline, line=:dash, lw=1.5, color=:blue);
    plot!(plt, [lo], seriestype=:hline, line=:dash, lw=1.5, color=:blue);
    return plt
end


#=
Model fitting
=#
function arima_order(ts, pmax=10, qmax=10)
    @rput ts
    @rput pmax
    @rput qmax
    R"""
        library(astsa)
        library(stats)
        aic = AIC(arima(ts, order = c(1, 0, 1), method="ML"))
        bestorder = c(0, 0, 0)
        bestaic = Inf
        for (i in 1:pmax) for (j in 1:qmax){
                fitaic = AIC(arima(ts, order = c(i, 0, j), method="ML"))
                if (fitaic < bestaic){
                    bestorder = c(i, 0, j)
                    bestarma = arima(ts, order = bestorder, method="ML")
                    bestaic = fitaic
                }
            }
        pq = bestorder
        aic = bestaic
    """
    @rget pq
    @rget aic
    return pq, aic
end

#pq, aic = arima_order(data, 8, 8)

function arima_fit_predict(ts, p, q, npred)
    @rput ts
    @rput p
    @rput q
    @rput npred
    R"""
        library(astsa)
        library(stats)
        arimafit = arima(ts, order=c(p, 0, q), method="ML")
        print(arimafit)
        arimapred = predict(arimafit, npred)
    """
    @rget arimafit
    @rget arimapred
    return arimafit, arimapred
end


function auto_arima(ts, pmax, qmax)
    @rput ts
    @rput pmax
    @rput qmax
    R"""
        library(forecast)
        # recommended setting
        auto.arima(lynx, max.p=pmax, max.q=qmax)#, trace = T, stepwise = F, approximation = F)
    """
end

