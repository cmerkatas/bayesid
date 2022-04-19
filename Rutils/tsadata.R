library(MASS)
data(geyser)
#write.table(geyser[60:210, 1], "~/github/bayesid/data/oldfaithful.txt", sep="\n", row.names=FALSE, col.names=FALSE)

data = geyser[60:210, 1]
pacf(data)

library(timeSeries)

library(TSA)
library(astsa)
load("tsa3.rda")


tsplot(jj, col=4, type="o", ylab="Quarterly Earnings per Share")
tsplot(globtemp, col=4, type="o", ylab="Global Temperature Deviations")
tsplot(gtemp_land, col=4, type="o", ylab="Global Temperature Deviations")
