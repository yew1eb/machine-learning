function bikMeans
%%
clc
clear
close all
%%
biK = 4;
biDataSet = load('testSet.txt');
[row,col] = size(biDataSet);
% 存储质心矩阵
biCentSet = zeros(biK,col);
% 初始化设定cluster数量为1
numCluster = 1;
%第一列存储每个点被分配的质心，第二列存储点到质心的距离
biClusterAssume = zeros(row,2);
%初始化质心
biCentSet(1,:) = mean(biDataSet);
for i = 1:row
    biClusterAssume(i,1) = numCluster;
    biClusterAssume(i,2) = distEclud(biDataSet(i,:),biCentSet(1,:));
end
while numCluster < biK
    minSSE = 10000;
    %寻找对哪个cluster进行划分最好，也就是寻找SSE最小的那个cluster
    for j = 1:numCluster
        curCluster = biDataSet(find(biClusterAssume(:,1) == j),:);
        [spiltCentSet,spiltClusterAssume] = kMeans(curCluster,2);
        spiltSSE = sum(spiltClusterAssume(:,2));
        noSpiltSSE = sum(biClusterAssume(find(biClusterAssume(:,1)~=j),2));
        curSSE = spiltSSE + noSpiltSSE;
        fprintf('第%d个cluster被划分后的误差为：%f \n' , [j, curSSE])
        if (curSSE < minSSE)
            minSSE = curSSE;
            bestClusterToSpilt = j;
            bestClusterAssume = spiltClusterAssume;
            bestCentSet = spiltCentSet;
        end
    end

     %更新cluster的数目  
    numCluster = numCluster + 1;
    % 必须先更新2的为新的类，再更新1的为原来的类，不然的话，当对第二个cluster进行划分的时候，就会被全部分为同一个类
    bestClusterAssume(find(bestClusterAssume(:,1) == 2),1) = numCluster;
    bestClusterAssume(find(bestClusterAssume(:,1) == 1),1) = bestClusterToSpilt;

    
    % 更新和添加质心坐标  第一行为更新质心，第二行为添加质心
    biCentSet(bestClusterToSpilt,:) = bestCentSet(1,:);
    biCentSet(numCluster,:) = bestCentSet(2,:);

    % 更新被划分的cluster的每个点的质心分配以及误差
    biClusterAssume(find(biClusterAssume(:,1) == bestClusterToSpilt),:) = bestClusterAssume;
end

figure
%scatter(dataSet(:,1),dataSet(:,2),5)
for i = 1:biK
    pointCluster = find(biClusterAssume(:,1) == i);
    scatter(biDataSet(pointCluster,1),biDataSet(pointCluster,2),5)
    hold on
end
%hold on
scatter(biCentSet(:,1),biCentSet(:,2),300,'+')
hold off
biCentSet
end

% 计算欧式距离
function dist = distEclud(vecA,vecB)
    dist  = sum(power((vecA-vecB),2));
end

% K-means算法
function [centSet,clusterAssment] = kMeans(dataSet,K)

[row,col] = size(dataSet);
% 存储质心矩阵
centSet = zeros(K,col);
% 随机初始化质心
for i= 1:col
    minV = min(dataSet(:,i));
    if isempty(minV)
        break;
    end
    rangV = max(dataSet(:,i)) - minV;
    centSet(:,i) = bsxfun(@plus,minV,rangV*rand(K,1));
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
end
end
