writedlm("sims/logistic/samples/sampled_weights.txt", est.weights)
writedlm("sims/logistic/samples/sampled_noise.txt", est.noise)
writedlm("sims/logistic/samples/sampled_clusters.txt", est.clusters)
for t in 1:20
    writedlm("sims/logistic/samples/sampled_pred$t.txt", est.predictions[t])
end
writedlm("sims/logistic/samples/sampled_precisions.txt", est.precisions)
writedlm("sims/logistic/samples/thinnedpredictions.txt", thinnedpredictions)

# save the fig
savefig(noiseplot, "sims/logistic/figures/logisticnoise.pdf")

savefig(clusters_plt, "sims/logistic/figures/clusters.pdf")

savefig(zoomplt, "sims/logistic/figures/zoompred.pdf")

savefig(plt, "sims/logistic/figures/logistic-est-pred.pdf")
