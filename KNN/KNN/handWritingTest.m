function handWritingTest
%%
clc
clear
close all
%% 获取目录下的所有txt文件名称
d = dir(['digits/trainingDigits/' '*.txt']); % struct 类型
dircell = struct2cell(d); %cell 类型
trainSetLen = size(dircell,2);
K = 4;
dataSize = 1024;
trainLabels = zeros(trainSetLen,1);
trainSet = [];
simpleTrainSet = zeros(1,dataSize);
simpleTestSet = zeros(1,dataSize);

%% 加载数据
fprintf('loading data...')
for i = 1:trainSetLen
    trainName =  dircell(1,i);
    trainFilename = cell2mat(trainName);
    trainLabels(i) = str2num(trainFilename(1));

    fid = fopen(['digits/trainingDigits/' trainFilename],'r');
    traindata = fscanf(fid,'%s');
    for j = 1:dataSize
        simpleTrainSet(j) =  str2num(traindata(j));
    end
    trainSet = [trainSet ; simpleTrainSet];
    fclose(fid);
end

d = dir(['digits/testDigits/' '*.txt']); % struct 类型
dircell = struct2cell(d); %cell 类型
testSetLen = size(dircell,2);
error = 0;
%% 测试数据
for k = 1:testSetLen
    testName =  dircell(1,k);
    testFilename = cell2mat(testName);
    testLabels = str2num(testFilename(1));

    fid = fopen(['digits/testDigits/' testFilename],'r');
    testdata = fscanf(fid,'%s');
    for j = 1:dataSize
        simpleTestSet(j) =  str2num(testdata(j));
    end
    classifyResult = KNN(simpleTestSet,trainSet,trainLabels,K);
    fprintf('识别数字为：%d  真实数字为：%d\n' , [classifyResult , testLabels])
    if(classifyResult~=testLabels)
        error = error+1;
    end
    fclose(fid);
end

fprintf('识别准确率为：%f\n',1-error/testSetLen)

end
