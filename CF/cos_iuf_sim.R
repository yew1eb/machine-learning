#similarity for inverse user frequency( modified cosine similairy)
iuf.cosine.similarity<-function(usr.list, item.list, user1, user2){
  
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
  
  #calculate similarity
  
  rating1.sum.sqr<-sum(user1.rating^2)
  rating2.sum.sqr<-sum(user2.rating^2)
  
  #inner product
  rating12.sum <- sum(common.item.rating1 * common.item.rating2 * fi * fi)
  
  num<-rating12.sum
  den<-sqrt(rating1.sum.sqr*rating2.sum.sqr)
  
  iuf.cos.sim<-num/den
  
  return(iuf.cos.sim)
}