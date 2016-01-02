% K-means算法
function [centSet,clusterAssment] = kMeans(dataSet,K)

[row,col] = size(dataSet);
% 存储质心矩阵
centSet = zeros(K,col);
% 随机初始化质心
for i= 1:col
    minV = min(dataSet(:,i));
    rangV = max(dataSet(:,i)) - minV;
    centSet(:,i) = bsxfun(@plus,minV,rangV*rand(K,1));
    %centSet(:,i) = repmat(minV,[K,1]) + rangV*rand(K,1);
end

% 用于存储每个点被分配的cluster以及到质心的距离
clusterAssment = zeros(row,2);
clusterChange = true;
while clusterChange
    clusterChange = false;
    % 计算每个点应该被分配的cluster
    for i = 1:row
        % 这部分可能可以优化
        minDist = inf;
        minIndex = 0;
        for j = 1:K
            distCal = distEclud(dataSet(i,:) , centSet(j,:));
            if (distCal < minDist)
                minDist = distCal;
                minIndex = j;
            end
        end
        if minIndex ~= clusterAssment(i,1)            
            clusterChange = true;
        end
        clusterAssment(i,1) = minIndex;
        clusterAssment(i,2) = minDist;
    end
    
    % 更新每个cluster 的质心
    for j = 1:K
        simpleCluster = find(clusterAssment(:,1) == j);
        centSet(j,:) = mean(dataSet(simpleCluster',:));
    end
end
end
