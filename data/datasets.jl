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
