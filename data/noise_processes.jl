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
