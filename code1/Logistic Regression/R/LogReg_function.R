sigmoid<-function(X){ 
  Z= 1/(1 + exp(-1*X));
  return(Z);  
}

cost.fun<-function(theta, X, y, lambda){
  #n=number of samples
  n=dim(X)[1];
  #m=number of features
  m=dim(X)[2];
  X=as.matrix(X);
  
  h_theta = sigmoid(X %*% theta);
  J = (1/n)* (sum( ((-1)*y*log(h_theta)) - ((1-y)*log(1-h_theta))) + lambda*0.5*sum(theta[2:m,]^2));
  
  return(J);
}

grad.fun<-function(theta, X, y, lambda, alpha){
  #n=number of samples
  n=dim(X)[1];
  #m=number of features
  m=dim(X)[2];
  X=as.matrix(X);
  
  theta_next = array(rep(0,m), c(m,1));
  
  h_theta = sigmoid(X %*% theta);
  
  #theta0
  theta_next[1,] = theta[1,] - alpha * (1/n) * sum( (h_theta - y)dia*X[,1] );
  
  #theta 1 to m-1
  for(i in 2:m){
    theta_next[i,] = theta[i,] - alpha * (1/n) * (sum( (h_theta - y)*X[,i])+lambda*theta[i,]);    
  }
  return(theta_next);
}