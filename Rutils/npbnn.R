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


library(BayesARIMAX)
BayesARIMAX(ytrain, cbind(lag(ytrain, 1), lag(ytrain, 2)), p=2, d=0, q=0)


