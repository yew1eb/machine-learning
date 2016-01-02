function stocGradAscent
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
alpha = 0.01;
labelMat = data(:,col);
weight = ones(col,1);
for i = 1:row
    h = sigmoid(dataMat(i,:)*weight);
    error = labelMat(i) - h;
    dataMat(i,:)
    weight = weight + alpha * error * dataMat(i,:)'
end

figure
scatter(dataMat(find(labelMat(:)==0),2),dataMat(find(labelMat(:)==0),3),5);
hold on
scatter(dataMat(find(labelMat(:) == 1),2),dataMat(find(labelMat(:) == 1),3),5);
hold on
x = -3:0.1:3;
y = -(weight(1)+weight(2)*x)/weight(3);
plot(x,y)
hold off


end

function returnVals = sigmoid(inX)
    % 注意这里的sigmoid函数要用点除
    returnVals = 1.0./(1.0+exp(-inX));
end
