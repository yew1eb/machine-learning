iuf.pearson.similarity<-function(usr.list, item.list, user1, user2){
  
  user1.item<-user1$item
  user1.rating<-user1$rating
 
  user2.item<-user2$item
  user2.rating<-user2$rating
 
  common.item<-intersect(user1.item, user2.item)
  
  #return 0 if no common item
  if(!any(common.item)) return(0)
  
  #total user
  n<-length(usr.list)
  
  common.length<-length(common.item)
  common.item.rating1<-rep(0, common.length)
  common.item.rating2<-rep(0, common.length)
  fi<-rep(0, common.length)
  
  for( index in 1:common.length ){
    
    item<-common.item[index]
    rated.times <-length (item.list[[as.character(item)]]$rating)
    
    common.item.rating1[index]<-user1.rating[which(user1.item==item)]
    common.item.rating2[index]<-user2.rating[which(user2.item==item)]
    
    fi[index]<-log(n/rated.times)
  }
  #print(common.item.rating1)
  
  #calculate similarity
  
  #mean over all item or similary item
  sum.fi<-sum(fi)
  sum.f.r1.r2 <-sum( fi * common.item.rating1 * common.item.rating2 )
  sum.f.r1 <-sum( fi * common.item.rating1)
  sum.f.r2 <-sum( fi * common.item.rating2)
  
  sum.f.r1.sqr <-sum( fi * (common.item.rating1^2) )
  sum.f.r2.sqr <-sum( fi * (common.item.rating2^2) )
  
  U<-sum.fi * (sum.f.r1.sqr - (sum.f.r1)^2 )
  V<-sum.fi * (sum.f.r2.sqr - (sum.f.r2)^2 )
  
  num <- (sum.fi*sum.f.r1.r2) - (sum.f.r1*sum.f.r2) 
  
  den<-sqrt(U * V)
  
  if(den!=0){
    iuf.pear.sim<-num/den
  }
  else{
   iuf.pear.sim<-0
  }
  
  return(iuf.pear.sim)
}