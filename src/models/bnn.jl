"""
Define structure of neural net.
"""
mutable struct NeuralNet{N, W, RE}
    nnet::N
    ws::W
    re::RE

    function NeuralNet(nnet;ws = nothing)
        _ws, re = Flux.destructure(nnet)
        if ws === nothing
            ws = _ws
        end
        nnet = re(ws)
        new{typeof(nnet), typeof(ws), typeof(re)}(nnet, ws, re)
    end
end
@Flux.functor NeuralNet
(b::NeuralNet)(x, ws=b.ws) = b.re(ws)(x)


function compute_preconditioner(g::NeuralNet, hyper_taus::Array{Float64, 2})
    preconditioner = []
    for l in 1:1:length(g.nnet)
        nw, nb = length(vcat(g.nnet[l].W...)), length(vcat(g.nnet[l].b...))
        push!(preconditioner, hyper_taus[1, l]*ones(nw), hyper_taus[2, l]*ones(nb))
    end
    return vcat(preconditioner...)
end


function U(g::NeuralNet,
           new_w,
           y::AbstractArray,
           x::AbstractArray,
           tau::Union{Float64, Array{Float64,1}},
           tau_preconditioner::Vector)
    g.ws = new_w
    u = -0.5sum((g(x, g.ws) .- y).^2 .* tau') - 0.5norm(sqrt.(tau_preconditioner) .* g.ws)^2
    return u
end


function ∇U(g::NeuralNet,
            new_w,
            y::AbstractArray,
            x::AbstractArray,
            tau::Union{Float64, Array{Float64,1}},
            tau_preconditioner::Vector)
   g.ws = new_w
   ps = Flux.params(g.ws)
   gs=Zygote.gradient(g -> U(g, g.ws, y, x, tau, tau_preconditioner), g)[1].ws
   return gs
end


# UW multivariate likelihood based on wishart
function UW(g::NeuralNet,
           new_w,
           y::AbstractArray,
           x::AbstractArray,
           tau::Vector{Matrix{Float64}},
           tau_preconditioner::Vector)
    g.ws = new_w
    u = -0.5sum([(view(y, :, i) .- g(view(x, :, i), g.ws))' * view(tau[i], :, :) * (view(y, :, i) .- g(view(x, :, i), g.ws)) for i in 1:1:size(y, 2)])
        - 0.5norm(sqrt.(tau_preconditioner) .* g.ws)^2
    return u
end


# ∇UW gradient of multivariate likelihood based on wishart
function ∇UW(g::NeuralNet,
            new_w,
            y::AbstractArray,
            x::AbstractArray,
            tau::Vector{Matrix{Float64}},
            tau_preconditioner::Vector)
   g.ws = new_w
   gs = Zygote.gradient(UW, g, g.ws, y, x, tau, tau_preconditioner)[2]
   return gs
end
