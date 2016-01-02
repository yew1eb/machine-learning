function testPCA
%%
clc
clear
close all
%%

filename = 'testSet.txt';
K = 1;
data = load(filename);
[lowData,reconMat] = PCA(data,K);
figure
scatter(data(:,1),data(:,2),5,'r')
hold on
scatter(reconMat(:,1),reconMat(:,2),5)
hold off

end
