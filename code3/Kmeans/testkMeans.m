function testkMeans
clc
clear all
close all
K = 4;
dataSet = load('testSet.txt');
figure 
scatter(dataSet(:,1),dataSet(:,2))
[row,col] = size(dataSet);
% 存储质心矩阵
centSet = zeros(K,col);
% 随机初始化质心
for i= 1:col
    minV = min(dataSet(:,i));
    rangV = max(dataSet(:,i)) - minV;
    centSet(:,i) = repmat(minV,[K,1]) + rangV*rand(K,1);
end

% 用于存储每个点被分配的cluster以及到质心的距离
clusterAssment = zeros(row,2);
clusterChange = true;
while clusterChange
    clusterChange = false;
    % 计算每个点应该被分配的cluster
    for i = 1:row
        % 这部分可能可以优化
        minDist = 10000;
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
     centSet
end
figure
%scatter(dataSet(:,1),dataSet(:,2),5)
for i = 1:K
    pointCluster = find(clusterAssment(:,1) == i);
    scatter(dataSet(pointCluster,1),dataSet(pointCluster,2),5)
    hold on
end
%hold on
scatter(centSet(:,1),centSet(:,2),300,'+')
hold off

end

% 计算欧式距离
function dist = distEclud(vecA,vecB)
    dist  = sum(power((vecA-vecB),2));
end
