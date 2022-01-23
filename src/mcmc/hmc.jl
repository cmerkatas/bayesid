abstract type SamplerState end

"""
SamplerState for Hamiltonian Monte Carlo
Fields:
    - x: the current state
    - p: the current momentum
    - stepsize
    - niters: number of iterations
    - mass: mass matrix
"""
mutable struct HMCState <: SamplerState
    x::Array{Float32,1}
    p::Array{Float64,1} # included in sampler state to allow autoregressive updates.
    stepsize::Float64
    niters::Int64
    mass
    function HMCState(x::Array{Float32,1},stepsize::Float64,niters::Int64;p=randn(length(x)),mass=1.0)
        if isa(mass,Number)
          mass = mass * ones(length(x))
        end
        new(x,p,stepsize,niters,mass)
    end
end

"""
sample! performs one HMC update
"""
function samplestep(f::NeuralNet, s::HMCState, Uf::Function, ∇Uf::Function, taus::Union{Float64, Array{Float64,1}, Vector{Matrix{Float64}}}, tau_preconditioner, y::Array{Float64,2}, x::Array{Float64,2})
  # hamiltonian monte carlo (radford neal's version)
  acc_flag = 0
  nparams = length(s.x)
  mass = s.mass
  stepsize = s.stepsize
  niters = s.niters

  s.p = sqrt.(mass).*randn(nparams)
  curx = s.x
  curp = s.p
  s.p += .5*stepsize .* ∇Uf(f, s.x, y, x, taus, tau_preconditioner)
  for iter = 1:niters
    s.x += stepsize * s.p./mass
    s.p += (iter<niters ? stepsize : .5*stepsize) .* ∇Uf(f,s.x,y,x,taus,tau_preconditioner) # two leapfrog steps rolled in one unless at the end.
  end

  logaccratio = Uf(f,s.x,y,x,taus,tau_preconditioner) - Uf(f,curx,y,x,taus,tau_preconditioner) -0.5*sum((s.p.*s.p - curp.*curp)./mass)[1]
  if 0.0 > logaccratio - log(rand())
      #reject
      s.x = curx
      s.p = curp
  else
      #accept
      #negate momentum for symmetric Metropolis proposal
      acc_flag = 1
      s.p = -s.p
  end
  return s, acc_flag
end
