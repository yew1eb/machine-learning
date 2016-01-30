function [lowData,reconMat] = PCA(data,K)

[row , col] = size(data);
meanValue = mean(data);
%varData = var(data,1,1);
normData = data - repmat(meanValue,[row,1]);
covMat = cov(normData(:,1),normData(:,2));%求取协方差矩阵
[eigVect,eigVal] = eig(covMat);%求取特征值和特征向量
[sortMat, sortIX] = sort(eigVal,'descend');
[B,IX] = sort(sortMat(1,:),'descend');
len = min(K,length(IX));
eigVect(:,IX(1:1:len));
lowData =  normData * eigVect(:,IX(1:1:len));
reconMat = (lowData * eigVect(:,IX(1:1:len))') + repmat(meanValue,[row,1]);  % 将降维后的数据转换到新空间

end
