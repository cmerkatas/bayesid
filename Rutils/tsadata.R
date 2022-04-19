library(MASS)
data(geyser)
#write.table(geyser[60:210, 1], "~/github/bayesid/data/oldfaithful.txt", sep="\n", row.names=FALSE, col.names=FALSE)

data = geyser[60:210, 1]
pacf(data)

library(timeSeries)