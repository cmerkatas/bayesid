using StatsPlots, StatsBase, DelimitedFiles, LaTeXStrings, KernelDensity, Latexify, RCall
default(; # Plots defaults
    fontfamily="Computer modern",
    label="" # only explicit legend entries
    )
set_default(; # Latexify defaults
    #unitformat=:slash # in case you want `m/s`
    )
scalefontsizes(1.2)


function plot_results(data, lag, ntrain, fit, sts, ŷ, ŷstd; kwargs...)
    nfull = size(data, 1)
    tsteps = 1:1:nfull

    plt = scatter(data, label=L"\mathrm{data}", grid=false, colour=:blue; kwargs...)
    plot!(plt, 1:lag+1, data[1:lag+1], colour =:blue)
    plot!(plt, [ntrain], seriestype =:vline, colour = :green, linestyle =:dash, label = L"\mathrm{training\, data\, end}")
    lag += 1
    plot!(plt, tsteps[lag:ntrain], mean(fit,dims=1)', ribbon=sts, alpha=0.4, colour=:blue, label=L"\mathrm{fitted\, model}")
    plot!(plt, tsteps[ntrain+1:end], ŷ', ribbon=ŷstd, colour =:purple, alpha=0.4, label=L"\mathrm{preditions}")
    return plt
end


function plot_results(data, lag, ntrain, fit, sts, ŷ, ŷstd, armap, armas; kwargs...)
    nfull = size(data, 1)
    tsteps = 1:1:nfull

    plt = scatter(data, label=L"\mathrm{data}", colour=:blue, grid=false; kwargs...)
    plot!(plt, 1:lag+1, data[1:lag+1], color =:blue)
    plot!(plt, [ntrain], seriestype =:vline, color = :green, linestyle =:dash, label = L"\mathrm{training\, data\, end}")
    lag += 1
    plot!(plt, tsteps[lag:ntrain], mean(fit,dims=1)', ribbon=sts, alpha=0.8, colour=:blue, label=L"\mathrm{fitted\, model}")
    plot!(plt, tsteps[ntrain+1:end], ŷ', colour =:purple, alpha=0.8, label=L"\mathrm{predictions}")
    plot!(plt, tsteps[ntrain+1:end], armap, colour =:green, alpha=0.8, label=L"\mathrm{ARMA preditcions}")

    return plt
end


function explore_data(data)
    plt1 = plot(data)
    plt2 = histogram(data, bins = 50, color=:lavender)
    plt3 = plot_acf(data)
    plt4 = plot_pacf(data)
    plt = plot(plt1, plt2, plt3, plt4, layout=(2,2), size=(1500, 1500))
    return plt
end


