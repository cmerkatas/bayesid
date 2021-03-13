writedlm("sims/newlog/samples/sampled_weights.txt", est.weights)
writedlm("sims/newlog/samples/sampled_noise.txt", est.noise)
writedlm("sims/newlog/samples/sampled_clusters.txt", est.clusters)
for t in 1:10
    writedlm("sims/newlog/samples/sampled_pred$t.txt", est.predictions[t])
end
writedlm("sims/newlog/samples/sampled_precisions.txt", est.precisions)
writedlm("sims/newlog/samples/thinnedpredictions.txt", thinnedpredictions)

# save the fig
savefig(noiseplot, "sims/lynx/seed1234_40k/figures/lynx/seed1234_40knoise.pdf")

savefig(clusters_plt, "sims/lynx/seed1234_40k/figures/clusters.pdf")

savefig(zoomplt, "sims/lynx/seed1234_40k/figures/zoompred.pdf")

savefig(newplt, "sims/lynx/seed1234_40k/figures/seed1234_40k-est-pred-std.pdf")

savefig(bestplt, "sims/newlog/figures/bestlog-zoom-pred.pdf")

savefig(stdplt, "sims/newlog/figures/bestlog-est-pred.pdf")
