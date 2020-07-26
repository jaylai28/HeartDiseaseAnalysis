source ("wrappers.R")
source("prediction.stats.R")
library(rpart)
library(glmnet)
library(boot)
library(kknn)
library(rpart)

#fit a decision tree and use cross validation with 10 folds and 1000 repetitions
heart.train=read.csv("heart.train.ass3.2019.csv")
summary(heart.train)

heart.train1=rpart(HD ~ ., heart.train)

# There are 7 terminal nodes in this tree
# The variables are CA, CP, EXANG, CHOL, AGE
heart.train1

cv = learn.tree.cv(HD~.,data=heart.train,nfolds=10,m=1000)
plot.tree.cv(cv)

#prune the tree
prune.rpart(tree=heart.train1, cp = cv$best.cp)
plot(cv$best.tree)

text(cv$best.tree, pretty=12)

#fit using logistic regression model
file= glm(HD ~ . , data=heart.train, family=binomial)

step.fit.bic = step(file, direction="both", k=log(nrow(heart.train)), trace=0)
summary(step.fit.bic)

#find the important coefficients using the stepwise selection
step.fit.bic$coefficients


#compute the prediction statistics for both the tree and steop-wise logistic regression model
my.pred.stats(predict(file,heart.train,type="response"), heart.train$HD)
my.pred.stats(predict(cv$best.tree,heart.train)[,2], heart.train$HD)



heart.test=read.csv("heart.test.ass3.2019.csv")
#calculate the odds of having hear disease for patient in 45th row using the tree model and
#step-wise logistic regression model
prob = predict(step.fit.bic, heart.test, type = "response")
prob[45]
odd= prob[45]/(1-prob[45])
odd



#use bootstrap procedure and bca option to compute the confidence interval
boot.prob = function(formula, data, indices)
{
  # Create a bootstrapped version of our data
  d = data[indices,]
  
  # Fit a logistic regression to the bootstrapped data
  fit = glm(formula, data=d, family=binomial)

  # Compute the AUC and return it
  dataset=heart.test[45,]
  rv = predict(fit,dataset,type="response")
  return(rv)
}

bs = boot(data=heart.train, statistic=boot.prob, R=5000, formula=HD ~ CP + EXANG  +OLDPEAK + CA + THAL)

boot.ci(bs,conf=0.95,type="bca")
 


#use bootstrap to compute 95% confidence interval for classification accuracy
boot.ca = function(formula, data, indices)
{
  # Create a bootstrapped version of our data
  d = data[indices,]
  
  # Fit a logistic regression to the bootstrapped data
  fit = glm(formula, d, family=binomial)
  
  # Compute the AUC and return it
  target = as.character(fit$terms[[2]])
  rv = my.pred.stats(predict(fit,d,type="response"), d[,target], display=F)
  return(rv$ca)
}

bs = boot(data=heart.train, statistic=boot.ca, R=5000, formula=HD ~ CP + EXANG  +OLDPEAK + CA + THAL)
boot.ci(bs,conf=0.95,type="bca")
plot(bs)


#split the data into training and testing dataset
ms.train = read.csv("ms.train.ass3.2019.csv", header=T)
ms.test = read.csv("ms.test.ass3.2019.csv", header=T)


#use knn to estimate the training data
mZtest.hat = fitted( kknn(intensity~ ., ms.train, ms.test, kernel = 'optimal', k = 1) )
plot(1,mean((MZtest.hat - ms.test$intensity)^2),xlab="k-value",xlim = c(0,25),ylim = c(0,15), main="Mean square errors agaisnt k values from 1 to 25")
error = mean((mZtest.hat - ms.test$intensity)^2)
cat("when k-value = 1,","Mean Squared Error:",error,"\n")

for (i in 1:25) {
  mZtest.hat = fitted( kknn(intensity~ ., ms.train, ms.test, kernel = 'optimal', k = i) )
  error = mean((mZtest.hat - ms.test$intensity)^2)
  cat("when k-value =",i,"Mean Squared Error:",error,"\n")
  points(i,error)
}


#plot the graph for different k values
#when k=2
mZtest1.hat = fitted( kknn(intensity~ ., ms.train, ms.test, kernel = 'optimal', k = 2))
error1 = mean((mZtest1.hat - ms.test$intensity)^2)

plot(ms.train$MZ, ms.train$intensity,type='l',col="blue", main="When k=2")
lines(ms.test$MZ, ms.test$intensity,type='l',col="red")
lines(ms.test$MZ, mZtest1.hat,type='l',col="green")
legend("topright",c("1 = training data points","2 = true spectrum","3 = estimated spectrum"),fill=c("blue","red","green"))


#when k=5
mZtest2.hat = fitted( kknn(intensity~ ., ms.train, ms.test, kernel = 'optimal', k = 5))
error2 = mean((mZtest2.hat - ms.test$intensity)^2)

plot(ms.train$MZ, ms.train$intensity,type='l',col="blue", main="When k=5")
lines(ms.test$MZ, ms.test$intensity,type='l',col="red")
lines(ms.test$MZ, mZtest2.hat,type='l',col="green")
legend("topright",c("1 = training data points","2 = true spectrum","3 = estimated spectrum"),fill=c("blue","red","green"))



#when k=10
mZtest3.hat = fitted( kknn(intensity~ ., ms.train, ms.test, kernel = 'optimal', k = 10))
error3 = mean((mZtest3.hat - ms.test$intensity)^2)

plot(ms.train$MZ, ms.train$intensity,type='l',col="blue", main="When k=10")
lines(ms.test$MZ, ms.test$intensity,type='l',col="red")
lines(ms.test$MZ, mZtest3.hat,type='l',col="green")
legend("topright",c("1 = training data points","2 = true spectrum","3 = estimated spectrum"),fill=c("blue","red","green"))



#when k=25
mZtest4.hat = fitted( kknn(intensity~ ., ms.train, ms.test, kernel = 'optimal', k = 25))
error4 = mean((mZtest4.hat - ms.test$intensity)^2)

plot(ms.train$MZ, ms.train$intensity,type='l',col="blue", main="When k=25")
lines(ms.test$MZ, ms.test$intensity,type='l',col="red")
lines(ms.test$MZ, mZtest4.hat,type='l',col="green")
legend("topright",c("1 = training data points","2 = true spectrum","3 = estimated spectrum"),fill=c("blue","red","green"))



#use cross validation functionality in knn package to select an estimate of the best value of k
knn = train.kknn(intensity ~ ., data = ms.train, kmax=25, kernel="optimal")

mstest.cross = fitted( kknn(intensity ~ ., ms.train, ms.test, kernel = knn$best.parameters$kernel, k = knn$best.parameters$k) )
cat("k-value",knn$best.parameters$k,"is selected")
error= mean((mstest.cross - ms.test$intensity)^2)
error
     

#use decision tree to smooth the data
ms.train = read.csv("ms.train.ass3.2019.csv", header=T)
ms.test = read.csv("ms.test.ass3.2019.csv", header=T)

cv = learn.tree.cv(intensity~.,data=ms.train,nfolds=10,m=1000)
ms.train1=rpart(intensity ~ .,ms.train)
prune.rpart(tree=ms.train1, cp = cv$best.cp)
prediction = predict(cv$best.tree,ms.test)
plot(prediction,type="l")

mean((predict(ms.train1, ms.test) - ms.test$intensity)^2)
