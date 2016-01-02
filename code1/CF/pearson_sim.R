pearson.similarity<-function(user1, user2){
  
  user1.item<-user1$item
  user1.rating<-user1$rating
 
  user2.item<-user2$item
  user2.rating<-user2$rating
 
  common.item<-intersect(user1.item, user2.item)
  
  #return 0 if no common item
  if(!any(common.item)) return(0)
  
  common.length<-length(common.item)
  common.item.rating1<-rep(0, common.length)
  common.item.rating2<-rep(0, common.length)
  
  for( index in 1:common.length ){
    
    item<-common.item[index]
    
    common.item.rating1[index]<-user1.rating[which(user1.item==item)]
    common.item.rating2[index]<-user2.rating[which(user2.item==item)]
  }
  #print(common.item.rating1)
 
  #calculate similarity
 
  #mean over all item or similary item
  rating1.mean<-mean(as.vector(common.item.rating1))
  rating2.mean<-mean(as.vector(common.item.rating2))
  
  #
  common.item.rating1<-common.item.rating1-rating1.mean
  common.item.rating2<-common.item.rating1-rating2.mean
  
  
  
  #inner product
  num <- sum(common.item.rating1 * common.item.rating2)
  
  den<-sqrt( sum(common.item.rating1^2) * sum(common.item.rating2^2)  )
  
  if(den!=0){
    pear.sim<-num/den
  }
  else{
    pear.sim<-0
  }
  
  return(pear.sim)
}