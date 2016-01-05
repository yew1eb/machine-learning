#This is a SVM test with spam dataset
#converge may take more than a hour
#change the max iteration for fast training

#remove all previous data
rm(list=ls())
source("SVM.R")
#load library & data
library(kernlab)
data(spam)

#number of samples
m=dim(spam)[1];  
# number of features 
n=dim(spam)[2];

Y=rep(0,m)
# let spam=1, nonspam=(-1), in SVM the label has to be +1 or -1
Y[ which( spam$type == 'spam')] = 1
Y[ which( spam$type != 'spam')] = (-1)


#separate data X and Y
X = spam[,1:(n-1)]


#get random sample
indices = 1:m

#50% data for training and 50% for testing
train.indices = sample(m, as.integer(m/2))
test.indices = indices[!indices %in% train.indices]
train.X = X[train.indices,]
test.X = X[test.indices,]
train.Y = Y[train.indices]
test.Y = Y[test.indices]

#normalize training data
train.X = scale(train.X);
train.mu = attr(train.X, "scaled:center");
train.var = attr(train.X, "scaled:scale");
#apply normalized data to test data
test.X = scale(test.X, center=train.mu, scale=train.var);

#run SVM
svm.model<-SVM(train.X, train.Y, 0.1, max.iter=3000)

#exam result

#train error
pred.train<-predict(svm.model, train.X)
table(train.Y, pred.train)

train.err.rate<- sum(pred.train != train.Y)/length(pred.train)
train.accurate.rate <- 1 - train.err.rate


#test error
pred.test<-predict(svm.model, test.X)
table(test.Y, pred.test)

test.err.rate<- sum(pred.test != test.Y)/length(pred.test)
test.accurate.rate <- 1 -test.err.rate

