#calculate the error E at index k
getErr<-function(svm.obj, k){

  y.a.x<- svm.obj$y*svm.obj$alpha*(svm.obj$x)
  x.k<- as.matrix(svm.obj$x[k,], ncol=1)
  z<- as.matrix(y.a.x) %*% x.k
  pred<- sum(z)-svm.obj$b
  
  err<- (pred - svm.obj$y[k])
  return(err)
}

getW<-function(svm.obj){  
  z<- svm.obj$y*svm.obj$alpha*(svm.obj$x)
  w<-colSums(z)
  return(w)
}

selectIndex<-function(i, err.i, svm.obj){
  C<-svm.obj$C
  non.zero.index<-which((svm.obj$alpha != 0) && (svm.obj$alpha < C))
  
  max.index<-(-1)
  maxE<-0
  if(length(non.zero.index) >0){
    for(k in non.zero.index){
      if(k == i){
        next
      }
      err = abs(getErr(svm.obj, k)-err.i)
      if(err>maxE){
        maxE<-err
        max.index<-k
      }
    }#end for  
  }
   
  if(max.index < 0){
    max.index<-randSelectIndex(i, svm.obj$m)
  }
  
  return(max.index)
}

updateIndex<-function(svm.obj, index){
  
  err<-getErr(svm.obj, index)
  label<-svm.obj$y[index]
  if( ((err*label) < (-1)*svm.obj$tolerance && (svm.obj$alpha[index] < svm.obj$C) 
      ) || ((err*label) > svm.obj$tolerance && (svm.obj$alpha[index] >0) )){
    
    #select 2nd index 
    #TODO: repace by selectIndex later
    index2<-randSelectIndex(index, svm.obj$m)
    label2<-svm.obj$y[index2]
    err2<-getErr(svm.obj, index2)
    s<-(lable * label2)
    
    #old alpha
    alpha<-svm.obj$alpha[index]
    alpha2<-svm.obj$alpha[index2]
    
    #upper & lower bound for new aplha2
    if(label != label2){
      high = min(svm.obj$C, svm.obj$C + alpha2 - alpha )
      low = max(0 , alpha2 - alpha)
    }else{
      high = min(svm.obj$C, alpha2 + alpha )
      low = max(0 , alpha2 + alpha - svm.obj$C)  
    }
    
    #make sure low != high
    if(low == high){
      return;
    }
    
    #get eta
    x<-svm.obj$x[index,]
    x2<-svm.obj$x[index2,]
    eta = sum(x*x) + sum(x2*x2) - 2*sum(x*x2)  
  
    #make sure eta > 0
    if(eta < 0){
      print("eta > 0")
      return;
    }
    
    #update alpha2
    new.alpha2 <- alpha2 + (label2*(err-err2)/eta)
    new.alpha2 <- getAlpha(new.alpha2, high, low)
    svm.obj$alpha[index2]<-new.alpha2
    
    #update err in svm.obj
    
    #check update value
    if( abs(new.alpha2 - alpha2) < 1e-5){
      #alpha2 change too small
      return(0);
    }
    
    #update alpha
    new.alpha<- s*(alpha2 - new.alpha2)
    svm.obj$alpha[index]<-new.alpha
    
    
  }
  
}

randSelectIndex<-function(i, m){
  j=i
  while(T){
    j= sample(m,1)
    
    if(j!=i) break
  }
  return(j)
}

getAlpha<-function(current, high, low){
  
  if(current > high){
    current = high
  }
  else if(current < low){
    current = low
  }
  
  return(current)
}

update.b<-function(b1, b2, alpha, alpha2, C){
  
  if(alpha>0 && alpha <C){
    b<-b1
  }
  else if(alpha2>0 && alpha2<C){
    b<-b2
  }
  else{
    b<-0.5*(b1 + b2)
  }
  return(b)
}  

predict.SVM<-function(svm.obj, data){
  
  data.matrix<-as.matrix(data) 
  w<-svm.model$w
  pred<-(data.matrix %*% w) - svm.model$b
  pos.index<-which(pred>0)
  
  result<-rep(-1, dim(data)[1])
  result[pos.index]<-1
  #table(test.Y, pred.test)

  return(result)
}