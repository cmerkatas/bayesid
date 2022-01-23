include("noise_processes.jl")

function polyMap(Θ,z)
  g = 0.
  for i in 1:1:length(Θ)
    g += Θ[i]*z^(i-1)
  end
  return g
end

function noisypolynomial(x₀, θ, noiseproc;n=1, seed=4)
    Random.seed!(seed)
    x = zeros(n)
    x[1] = polyMap(θ,x₀) + rand(noiseproc)
    for i in 2:n
        x[i] = polyMap(θ,x[i-1]) + rand(noiseproc)
    end
    return x
end


function henonmap(x::Array{Float64}, θ::Array{Float64})
    z = [θ[1] + θ[2] * x[1]^2 + x[2], θ[3] * x[1]]
    return z
end


function noisyhenon(x₀::Array{Float64}, θ::Array{Float64}, noiseproc::Sampleable;n=1, seed=1)
    Random.seed!(seed)
    x = zeros(length(x₀), n)
    x[:, 1] = henonmap(x₀, θ) + rand(noiseproc)
    for i in 2:n
        x[:, i] = henonmap(x[:, i-1], θ) + rand(noiseproc)
    end
    return x
end
