#import rating data from file

split.data<-function(path, test.size=0.3){
  
  if(file.exists(path)){
    
    #import rating data
    temp.table<-read.table(path,header=F, sep=":")
    
    #only take value in coumn 1.3,5
    temp.table<-temp.table[,c(1,3,5)]
    names(temp.table)<-c("user", "item", "rating")
    
    test.size<-as.double(test.size)
    #get random index
    n<-nrow(temp.table)
    indices<-1:n
    test.indices<-sample(n, round(test.size*n))
    train.indices<-indices[!indices %in% test.indices]
    
    train.data<-temp.table[train.indices,]
    test.data <-temp.table[test.indices,]
    
    #return list
    return.list<-list('train.data'=train.data, 'test.data'=test.data)
    return(return.list)
    
  }
  else
  {
    print("no such file")
    return(list())
  }
  
}

usr.rating.list<-function(data){
  
    #transform table into list
    usr.item.list<-list()
    user.id<-unique(data[,1])
    
    for(index in user.id){
      sub.table<-subset(data, user==index)
    
      content.list<-list("item"=sub.table[,2], "rating"=sub.table[,3])
      
      usr.item.list[[as.character(index)]]<-content.list
    }
        
    return(usr.item.list)
}

item.rating.list<-function(data){
    
    #transform table into list
    item.usr.list<-list()
    item.id<-unique(data[,2])
    
    for(index in item.id){
      sub.table<-subset(data, item==index)
      
      content.list<-list("item"=sub.table[,1], "rating"=sub.table[,3])
      
      item.usr.list[[as.character(index)]]<-content.list
    }
    
    return(item.usr.list)
}