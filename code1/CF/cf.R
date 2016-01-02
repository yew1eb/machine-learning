#clear all
rm(list=ls())

#load functions
source("data_load.R")
source("pearson_sim.R")
source("cos_sim.R")
source("pearson_iuf_sim.R")
source("cos_iuf_sim.R")
source("pearson_dv_sim.R")
source("pred.R")

# split data if data are not splitted
if(!file.exists("train_data.csv")){

   #split data into training & testing
   temp.data<-split.data("./ml-1m/ratings.dat", test.size=0.3)
   train.data<-temp.data$train.data
   test.data<-temp.data$test.data

   #rite train & test data to csv file
   write.csv(train.data,"train_data.csv", row.names=F)
   write.csv(test.data,"test_data.csv", row.names=F)
   rm(temp.data)
}else{
  #load data
  train.data<-read.csv("train_data.csv")
  test.data<-read.csv("test_data.csv")  
}
   
#transform data in to user & item list
usr.list<-usr.rating.list(train.data)
item.list<-item.rating.list(train.data)

#testing
#c<-pearson.similarity(usr.list[['1']], usr.list[['4']])
#d<-cosine.similarity(usr.list[['1']], usr.list[['4']])
#e<-user.base.pred(usr.list, item.list, '4132', '1200', pearson.similarity)

#run user based prediction
k<-200
test.data$pred<-0
n<-nrow(test.data)
test.data<-test.data[order(test.data$user),]
all.user<-unique(train.data$user)
pre.user<-0
top.K.sim <-NULL

#item base prediction
for(i in 1:1000){
  
  user<-test.data[i,]$user  
  item<-test.data[i,]$item
  
  if(user != pre.user){
    
    pre.user<-user
    #get all similiarity for user and other
    sim.list<-data.frame('uid'=all.user, 'sim'=rep(0, length(all.user)))
    
    for(user2 in all.user){
      
      rowNum <- which(sim.list$uid == user2)
      if(user2 == user){ 
        sim.list$sim[rowNum]<-(-100)
      }
      else{
        #pearson
        #sim.list$sim[rowNum]<-pearson.similarity(usr.list[[as.character(user)]], usr.list[[as.character(user2)]]) 
        #cosine
        sim.list$sim[rowNum]<-cosine.similarity(usr.list[[as.character(user)]], usr.list[[as.character(user2)]]) 
        # iuf pearson
        #sim.list$sim[rowNum]<-iuf.pearson.similarity(usr.list, item.list, usr.list[[as.character(user)]], usr.list[[as.character(user2)]])
        # iuf cosine
        #sim.list$sim[rowNum]<-iuf.cosine.similarity(usr.list, item.list, usr.list[[as.character(user)]], usr.list[[as.character(user2)]])
        
        #default voting pearson
        #sim.list$sim[rowNum]<-dv.pearson.similarity(usr.list[[as.character(user)]], usr.list[[as.character(user2)]]) 
        
        
      }#end if-else
    }#end for
      
      #sort list
      top.K.sim<-sim.list[order(-sim.list$sim),][1:k,]
    }#end if 
  
  test.data[i,]$pred <-user.base.pred(usr.list, item.list, user, item, top.K.sim)  
}#end pred for 

#first 1000 sample test
new<-test.data[1:1000,]
new$err<-abs(new$pred - new$rating)
mae<-mean(new$err)

#write result to csv file
#write.csv(test.data,"user-based_pred_pearson.csv", row.names=F)
#write.csv(test.data,"user-based_pred_cosine.csv", row.names=F)
#write.csv(test.data,"user-based_pred_pearson_iuf.csv", row.names=F)
#write.csv(test.data,"user-based_pred_cosine_iuf.csv", row.names=F)
#write.csv(test.data,"user-based_pred_pearson_default_vote.csv", row.names=F)


#run item based prediction
k<-200
test.data$pred<-0
n<-nrow(test.data)

#change user -> item
test.data<-test.data[order(test.data$item),]
#change user -> item
all.user<-unique(train.data$item)
#change user -> item
pre.user<-0
top.K.sim <-NULL

#item base prediction
for(i in 1:1000){
  
  #exchange user & item
  user<-test.data[i,]$item  
  item<-test.data[i,]$user
  
  if(user != pre.user){
    
    pre.user<-user
    #get all similiarity for user and other
    sim.list<-data.frame('uid'=all.user, 'sim'=rep(0, length(all.user)))
    
    for(user2 in all.user){
      
      rowNum <- which(sim.list$uid == user2)
      if(user2 == user){ 
        sim.list$sim[rowNum]<-(-100)
      }
      else{
        #cosine 
        #change user -> item
        #sim.list$sim[rowNum]<-cosine.similarity(item.list[[as.character(user)]], item.list[[as.character(user2)]]) 
        
        # iuf cosine
        #change user -> item
        sim.list$sim[rowNum]<-iuf.cosine.similarity(item.list, usr.list, item.list[[as.character(user)]], item.list[[as.character(user2)]])
        
      }#end if-else
    }#end for
    
    #sort list
    top.K.sim<-sim.list[order(-sim.list$sim),][1:k,]
  }#end if 
  
  test.data[i,]$pred <-user.base.pred(item.list, usr.list, user, item, top.K.sim)  
}#end pred for 


#first 1000 sample test
new<-test.data[1:1000,]
new$err<-abs(new$pred - new$rating)
mae<-mean(new$err)
