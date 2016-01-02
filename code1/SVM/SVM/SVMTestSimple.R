#remove all previous data
rm(list=ls())
#This is a SVM test with small data set (iris dataset) 
#training time is less than a second.

#load SVM
source("SVM.R")
#load data & function
data(iris)

#select only the first two Species from data
data = iris[iris$Species==c('setosa') | iris$Species==c('versicolor') , ]

#define class y=+1,-1
data$y=0;
data[data$Species==c('setosa'),]$y = -1
data[data$Species==c('versicolor'),]$y=1

#ignore old Species term
data = data[, -5]

#set label to x1,x2,x3,x4
names(data)<-c('x1','x2','x3','x4','y')

#number of samples
m=dim(data)[1];  
# number of features 
n=dim(data)[2];

X = data[,1:(n-1)]
Y = data$y

#get random sample
indices = 1:m
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
svm.model<-SVM(train.X, train.Y, 1,500)

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

