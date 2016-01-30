
#remove all previous data
rm(list=ls())

#load library & data
library(kernlab)
data(spam)
source('~/LogReg_function.R')

#number of samples
n=dim(spam)[1];  
# number of features 
m=dim(spam)[2];

Y=array(rep(0,n), c(n,1));
# let spam==1, nonspam==0
Y[ which( spam$type == 'spam'),] = 1

#separate data X and Y
X = spam[,1:(m-1)];

#set inital theta to 0
theta=array(rep(0,m), c(m,1));
#regularization factor lambda
lambda = 500;
#learning rate alpha
alpha=1;

round=20;
result = as.data.frame(cbind(train = rep(0,round), test = rep(0,round)));

for(k in 1:round){
#get random sample
indices = 1:n
train.indices = sample(n, as.integer(n/2))
test.indices = indices[!indices %in% train.indices]
train.X = X[train.indices,]
test.X = X[test.indices,]
train.Y = as.data.frame(Y[train.indices,])
test.Y = as.data.frame(Y[test.indices,])

#normalize training data
train.X = scale(train.X);
train.mu = attr(train.X, "scaled:center");
train.var = attr(train.X, "scaled:scale");
train.X = cbind( const=array(rep(1,as.integer(n/2)), c(as.integer(n/2),1)) ,train.X);
#apply normalized data to test data
test.X = scale(test.X, center=train.mu, scale=train.var);
test.X = cbind( const=array(rep(1,n-as.integer(n/2)), c(n-as.integer(n/2),1)) ,test.X);

#Training
max_iter=20;
iter=0;
train.cost=rep(0,max_iter+1);
test.cost=rep(0,max_iter+1);
#initial cost
train.cost[1]=cost.fun(theta,train.X,train.Y,lambda);
test.cost[1]=cost.fun(theta,test.X,test.Y,lambda);              

for(i in 1:max_iter){
   
  theta = grad.fun(theta, train.X,train.Y, lambda, alpha)
  train.cost[i+1]=cost.fun(theta,train.X,train.Y,lambda);
  test.cost[i+1]=cost.fun(theta,test.X,test.Y,lambda);
  
  if(abs(train.cost[i+1]-train.cost[i])<0.001){
    iter=i+1;
    break;
  }
    
}

#prediction
train.h_theta = sigmoid(as.matrix(train.X) %*% theta);
test.h_theta = sigmoid(as.matrix(test.X) %*% theta);

train.y_hat= as.data.frame(array(rep(0,dim(train.Y)[1]), c(dim(train.Y[1]),1)));
test.y_hat = as.data.frame(array(rep(0,dim(test.Y)[1]), c(dim(test.Y[1]),1)));

train.y_hat[which(as.data.frame(train.h_theta)$V1 >=0.5 ), ]=1;
test.y_hat[which(as.data.frame(test.h_theta)$V1 >=0.5 ), ]=1;

M= as.data.frame(cbind(y=train.Y, y_hat=train.y_hat));
N= as.data.frame(cbind(y=test.Y, y_hat=test.y_hat));
names(M)<-c('y', 'y_hat')
names(N)<-c('y', 'y_hat')
train.accuracy = (dim ( M[ M$y== M$y_hat, ] )[1]) / dim(M)[1];
test.accuracy = (dim ( N[ N$y== N$y_hat, ] )[1]) / dim(N)[1];

#plot(train.cost[1:iter], type="l")
result$train[k]=train.accuracy;
result$test[k]=test.accuracy;
}        
avg.train = mean(result$train);
avg.test = mean(result$test);
