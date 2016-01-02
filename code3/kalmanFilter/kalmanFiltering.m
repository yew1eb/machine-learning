function kalmanFiltering
%%
clc
close all

%%
%
%   Description : kalmanFiltering
%   Author : Liulongpo
%   Time：2015-4-29 16:42:34
%

%%
Z=(1:2:200); %观测值  汽车的位置  也就是我们要修改的量
noise=randn(1,100); %方差为1的高斯噪声
Z=Z+noise;
X=[0 ; 0 ]; %初始状态
P=[1 0;0 1]; %状态协方差矩阵
F=[1 1;0 1]; %状态转移矩阵
Q=[0.0001,0;0 , 0.0001]; %状态转移协方差矩阵
H=[1,0]; %观测矩阵
R=1; %观测噪声协方差矩阵
figure;
hold on;
for i = 1:100
%基于上一状态预测当前状态  
X_ = F*X;
% 更新协方差  Q系统过程的协方差  这两个公式是对系统的预测
P_ = F*P*F'+Q;
% 计算卡尔曼增益
K = P_*H'/(H*P_*H'+R);
% 得到当前状态的最优化估算值  增益乘以残差
X = X_+K*(Z(i)-H*X_);
%更新K状态的协方差
P = (eye(2)-K*H)*P_;
scatter(X(1), X(2),4); %画点，横轴表示位置，纵轴表示速度
end

end
