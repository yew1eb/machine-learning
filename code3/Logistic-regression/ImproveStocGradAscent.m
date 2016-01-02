function ImproveStocGradAscent
%%
%
%   Description : LogisticRegression using stocGradAsscent
%   Author : Liulongpo
%   Time：2015-4-18 10:57:25
%
%%
clc
clear 
close all
%%
data = load('testSet.txt');
[row , col] = size(data);
dataMat = [ones(row,1) data(:,1:col-1)];
%alpha = 0.01;
numIter = 20;
labelMat = data(:,col);
weightVal = zeros(3,numIter*row);
weight = ones(col,1);
j = 0;

for k = 1:numIter
     randIndex = randperm(row);
    for i = 1:row
        % 改进点 1
        alpha = 4/(1.0+i+k)+0.01; 
        j = j+1;
        % 改进点 2 
        h = sigmoid(dataMat(randIndex(i),:)*weight);
         % 改进点 2
        error = labelMat(randIndex(i)) - h;
        % 改进点 2
        weight = weight + alpha * error * dataMat(randIndex(i),:)';
        weightVal(1,j) = weight(1);
        weightVal(2,j) = weight(2);
        weightVal(3,j) = weight(3);
    end
end

figure
i = 1:numIter*row;
subplot(3,1,1)
plot(i,weightVal(1,:)),title('weight0')%,axis([0 numIter*row 0.8 7])
j = 1:numIter*row;
subplot(3,1,2)
plot(j,weightVal(2,:)),title('weight1')%,axis([0 numIter*row 0.3 1.2])
k = 1:numIter*row;
subplot(3,1,3)
plot(k,weightVal(3,:)),title('weight2')%,axis([0 numIter*row -1.2 -0.1])

figure
scatter(dataMat(find(labelMat(:)==0),2),dataMat(find(labelMat(:)==0),3),5);
hold on
scatter(dataMat(find(labelMat(:) == 1),2),dataMat(find(labelMat(:) == 1),3),5);
hold on
x = -3:0.1:3;
y = -(weight(1)+weight(2)*x)/weight(3);
plot(x,y,'r')
hold off


end

function returnVals = sigmoid(inX)
    % 注意这里的sigmoid函数要用点除
    returnVals = 1.0./(1.0+exp(-inX));
end
