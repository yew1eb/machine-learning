user.base.pred<-function(usr.data,item.data, pred.user, pred.item, top.K.sim){
  
  #get avg rating for pred.user
  mean.pred<-mean(usr.data[[as.character(pred.user)]]$rating)
  
  #get top K user id
  top.K.user<-top.K.sim$uid
  
  #get usr who rate pred item
  rated.user<-item.data[[as.character(pred.item)]]$item
  
  #common users
  common.user<-NULL
  common.user<-intersect(top.K.user, rated.user)
  
  if(!any(common.user)){ 
    return(mean.pred)
  }
  else{
    
    num=0.0
    den=0.0
    
    for(user in common.user){
      
      user.rating<-usr.data[[as.character(user)]]$rating
      mean.user.rating <- mean(user.rating)
      
      user.item <- usr.data[[as.character(user)]]$item  
      index<-which(user.item==as.character(pred.item))
      item.rating<-user.rating[index]
      
      user.sim<-top.K.sim[which(top.K.sim$uid==user),]$sim
      
      num<-num + user.sim * (item.rating - mean.user.rating) 
      den<-den + abs(user.sim)
    }
    pred<-mean.pred + (num/den)
    return(pred)
  }
   #end else 
}
#end pred