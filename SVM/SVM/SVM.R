#implement SVM with SMO method as a S3 calss
#(only linear kernel for now)
#
#algorithm follows the paper: 
#  Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines
#  by John C. Platt

#Author:Chyi-Kwei Yau
#date: 2012.03.07

source("SVM_func.R")

SVM<-function(x, y, C, max.iter=3000, tolerance=1e-3){
  #x is the input matrix
  #y is the response (+1 or -1)
  m<-dim(x)[1]
  n<-dim(x)[2]
  
  #error check
  #if(m != length(y)){
  #  print("input and response have different length")
  #  return;
  #}
  
  model.attr<-list(); #build class
  class(model.attr) <- "SVM" #set class name
  # set attribute
  model.attr$x<-x
  model.attr$y<-y
  model.attr$C<-C
  model.attr$m<-m
  model.attr$n<-dim(x)[2]
  model.attr$max.iter<-max.iter
  model.attr$tolerance<-tolerance
  model.attrkernel<-kernel
  
  # initialize parameter
  model.attr$alpha<-rep(0, m)
  #model.attr$errorCache<-rep(0,m)
  model.w<-rep(0,n)
  model.attr$b<-0
  
  #training
  iter=0
  numChanged=0
  examAll=T; #go through all data first
  
  start.time<-proc.time()
  
  while( numChanged >0 || examAll){
    
    numChanged=0
    non.zero.alpha<-which( (model.attr$alpha!=0) &  (model.attr$alpha!=model.attr$C) )
    
    cat("iter=", iter, ", support vector number=",length(non.zero.alpha),"\n", sep="")
    
    if(examAll || length(non.zero.alpha)==0 ){    
      cat("  --exam all in this iteration\n")
      update.cand<-seq(1,model.attr$m)
    }
    else{
      cat("  --exam ", length(non.zero.alpha),  " alphas in this iteration\n",sep="")
      update.cand<-non.zero.alpha
    }
    
    #train Full first
    for( index in update.cand){
    #for( index in 1:1){
      err<-getErr(model.attr, index)
      #cat(err,"=err\n")
      
      label<-model.attr$y[index]
      
      # if err is grater than tolerance and not at bound
      if( ((err*label) < (-1)*model.attr$tolerance && (model.attr$alpha[index] < model.attr$C) 
      ) || ((err*label) > model.attr$tolerance && (model.attr$alpha[index] >0) )){
        
        #select 2nd index 
        #TODO: repace by selectIndex later
        index2<-selectIndex(index, err, model.attr)
        #index2<-randSelectIndex(index, model.attr$m)
        #cat(length(index2)," = index2\n")
        
        label2<-model.attr$y[index2]        
        err2<-getErr(model.attr, index2)
        #cat(err2,"= err2\n")
        
        s<-(label * label2)
        C<-model.attr$C
      
        #old alpha
        alpha<-model.attr$alpha[index]
        alpha2<-model.attr$alpha[index2]
        
        #upper & lower bound for new aplha2
        if(label != label2){
          high = min(C, C + alpha2 - alpha )
          low = max(0 , alpha2 - alpha)
        }else{
          #label == label2
          high = min(C, alpha2 + alpha )
          low = max(0 , alpha2 + alpha - C)  
        }
        
        #make sure low != high
        if(low == high){
          #cat("high = low, go to next alpha\n")
          next
        }
        
        #get eta
        x<-model.attr$x[index,]
        x2<-model.attr$x[index2,]
        #use linear kernel for now
        #TODO: replace by kernel function..
        eta = sum(x*x) + sum(x2*x2) - 2*sum(x*x2)
        
        if(eta<=0){
          #cat("eat <=0\n")
          next
        }
        
        #update alpha2
        new.alpha2 <- (alpha2 + (label2*(err-err2)/eta))
        #cat("new.alpha2=",new.alpha2, " high=",high, " low=",low,"\n")
        new.alpha2 <- getAlpha(new.alpha2, high, low)
        
        #check update value
        if( abs(new.alpha2 - alpha2) < 1e-5){
          #alpha2 change too small
          #cat("  --alpha change too small\n")
          next
        }
        
        model.attr$alpha[index2]<-new.alpha2
        
        #update err in index2
        #model.attr$errorCache[index2]<-getErr(model.attr, index2)
      
        
        #update alpha
        new.alpha<- alpha + s*(alpha2 - new.alpha2)
        model.attr$alpha[index]<-new.alpha
        
        #update err in index
        #model.attr$errorCache[index]<-getErr(model.attr, index)
        
        #calculate b1, formula (20)
        b1 =  err + (label*(new.alpha - alpha)*sum(x*x))+(label2*(new.alpha2-alpha2)*sum(x*x2)) + model.attr$b
        
        #calculate b2, formula (21)
        b2 = err2 + (label*(new.alpha - alpha)*sum(x*x2))+(label2*(new.alpha2-alpha2)*sum(x2*x2)) + model.attr$b
        
        #update b
        model.attr$b <- update.b(b1, b2, new.alpha, new.alpha2, model.attr$C)
        #cat("b1=",b1, "b2=", b2, "b=", model.attr$b,"\n")
        
        #alpha change number 
        numChanged<-numChanged+1
      }
    }#end for i
    
    
    #check next iter run full or not
    if(examAll == T){
      examAll=F
    }
    else if(numChanged==0){
      cat("  --no alpha changed in this iteration. Exam all in next iteration\n")
      examAll=T
    }
    cat("  --num alpha changed=",numChanged,"\n",sep="")
    iter=iter+1
    cat("\n")
    
    #force break
    if(iter > max.iter){
      cat("Warning: SVM not converge but reach max iterations!","\n")
      break
    }
    
  }#end while
  
  #calculate w
  model.attr$w<-getW(model.attr)
  
  #end time
  end.time<-proc.time()
  running.time<-end.time - start.time
  cat("\n")
  cat("SVM training finished. Elapsed time=", running.time["elapsed"], " secs")
  
  return(model.attr)
}

