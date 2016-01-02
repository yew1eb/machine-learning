clear all
clc

data = load('testSet.txt');
[PX,Model] = GMM(data,4);
[~,index] = max(PX');
cent = Model.Miu;
figure
I = find(index == 1);
scatter(data(I,1),data(I,2))
hold on
scatter(cent(1,1),cent(1,2),150,'filled')
hold on
I = find(index == 2);
scatter(data(I,1),data(I,2))
hold on
scatter(cent(2,1),cent(2,2),150,'filled')
hold on
I = find(index == 3);
scatter(data(I,1),data(I,2))
hold on
scatter(cent(3,1),cent(3,2),150,'filled')
hold on
I = find(index == 4);
scatter(data(I,1),data(I,2))
hold on
scatter(cent(4,1),cent(4,2),150,'filled')

