function relustLabel = KNN(inx,data,labels,k)
%% 
%   inx 为 输入测试数据，data为样本数据，labels为样本标签
%%

[datarow , datacol] = size(data);
diffMat = repmat(inx,[datarow,1]) - data ;
distanceMat = sqrt(sum(diffMat.^2,2));
[B , IX] = sort(distanceMat,'ascend');
len = min(k,length(B));
relustLabel = mode(labels(IX(1:len)));

end
