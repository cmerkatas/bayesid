metrics = function(f, y){
  mse = mean((f-y)^2)
  rmse = sqrt(mse)
  mae = mean(abs(f - y))
  mape = mean(abs((f - y) / y)) * 100
  u1 = sqrt(sum((f-y)^2)) / sqrt(sum(y^2))
  fsq = sqrt(mean(f^2))
  ysq = sqrt(mean(y^2))
  u2 = rmse / (fsq+ysq)
  return (list(mse=mse, rmse=rmse, mae=mae, mape=mape, u1=u1, u2=u2))
}

plot(lynx)
data = log10(lynx)

ntrain = 100
ytrain = data[1:ntrain]
ytest = data[(ntrain+1):length(data)]

# Fit an ar model to train data
lynx.ar = ar(ytrain)
lynx.ar

predictions = predict(lynx.ar, n.ahead=14)
yhat = predictions$pred
yhat

metrics(yhat, ytest)

ts.plot(yhat)
points(101:114,ytest)

# fit a neural network to data
library("forecast", lib.loc="/Library/Frameworks/R.framework/Versions/3.5/Resources/library")
nnfit = nnetar(window(data, end=1920), p=2, size=10, decay=0.05, maxit=150)
nnpred = forecast(nnfit, h=14, model=nnfit)$mean
plot(forecast(nnfit, h=14))
metrics(nnpred, ytest)


library(astsa)
data = fmri1[,5]
data = (data - mean(data)) / sd(data)
plot(data, type="l")
ytrain = data[1:114]
ytest = data[115:length(data)]

fmri.ar = ar(ytrain)
fmri.ar

predictions = predict(fmri.ar, n.ahead=14)
yhat = predictions$pred
yhat

metrics(yhat, ytest)
plot(data)
lines(115:128, ytest, col="red")
lines(115:128, yhat, col="blue")
ytest
yhat
yhatnn = c(-0.057517461258437604,
           -0.9119196086284888,
           -0.6515321984815113,
           -0.7271150902447031,
           -0.44200518307875086,
           -0.5487920043250327,
           -0.689470066148799,
           -0.8090896165965172,
           -0.9332186547037541,
           -0.36404634188419915,
           -0.3092197357887684,
           0.10850919120372934,
           -0.03845206824679782,
           0.3253432422160746)

lines(115:128, yhatnn, col="green")
