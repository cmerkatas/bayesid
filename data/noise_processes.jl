# noise processes definition
sigma = 0.001
σi = 200*sigma
noisemix1 = MixtureModel([Normal(0, 0.001), Normal(0, σi)], [0.9, 0.1])
noisemixdensity1(x) = pdf.(noisemix1, x)

sigma_param = sqrt(0.002)
paramnoise = Normal(0, sigma_param)
parametricnoisedensity(x) = pdf(Normal(0, sigma_param), x)

sigma23 = sqrt(4*1e-2)
sigma33 = sqrt(1*1e-4)
noisemix2 = MixtureModel(Normal[Normal(0, sigma23), Normal(0, sigma33)], [1.0/3, 2.0/3])
noisemixdensity2(x) = pdf.(noisemix2, x)

#=
2-D noise processes for henon map
=#
zero_mean = zeros(2)
σ² = 0.001
ρ = 0.85
Σ₁ = Symmetric(σ² .* [1 ρ;ρ 1])
Σ₂ = Symmetric(σ² .* [1 -ρ;-ρ 1])
pis = [1/2, 1/2]
mvmix1 = MixtureModel([MultivariateNormal(zero_mean, Σ₁), MultivariateNormal(zero_mean, Σ₂)], pis)
mvmixdensity(x::Array{Float64}) = pdf(mvmix1, x)

# x1range = -0.15:0.001:0.15
# x2range = -0.15:0.001:0.15
# zz = [mvmixdensity([x1,x2]) for x1 in x1range, x2 in x2range]
# heatmap(x1range, x2range, zz, color=:deep)
