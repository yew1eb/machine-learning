function weight = gradAscent
%%
%
%   Description : LogisticRegression using gradAsscent
%   Author : Liulongpo
%   Time：2015-4-18 10:57:25
%
%%
clc
close all
clear
%%

data = load('testSet.txt');
[row , col] = size(data);
dataMat = data(:,1:col-1);
dataMat = [ones(row,1) dataMat] ;
labelMat = data(:,col);
alpha = 0.001;
maxCycle = 500;
weight = ones(col,1);
for i = 1:maxCycle
    h = sigmoid((dataMat * weight)');
    error = (labelMat - h');
    weight = weight + alpha * dataMat' * error;
end

figure
scatter(dataMat(find(labelMat(:) == 0),2),dataMat(find(labelMat(:) == 0),3),3);
hold on
scatter(dataMat(find(labelMat(:) == 1),2),dataMat(find(labelMat(:) == 1),3),5);
hold on
x = -3:0.1:3;
y = (-weight(1)-weight(2)*x)/weight(3);
plot(x,y)
hold off

end

function returnVals = sigmoid(inX)
    % 注意这里的sigmoid函数要用点除
    returnVals = 1.0./(1.0+exp(-inX));
end
