function varargout = gmm(X, K_or_centroids)
% ============================================================
% Expectation-Maximization iteration implementation of
% Gaussian Mixture Model.
%
% PX = GMM(X, K_OR_CENTROIDS)
% [PX MODEL] = GMM(X, K_OR_CENTROIDS)
%
%  - X: N-by-D data matrix.
%  - K_OR_CENTROIDS: either K indicating the number of
%       components or a K-by-D matrix indicating the
%       choosing of the initial K centroids.
%
%  - PX: N-by-K matrix indicating the probability of each
%       component generating each point.
%  - MODEL: a structure containing the parameters for a GMM:
%       MODEL.Miu: a K-by-D matrix.
%       MODEL.Sigma: a D-by-D-by-K matrix.
%       MODEL.Pi: a 1-by-K vector.
% ============================================================

threshold = 1e-15;
[N, D] = size(X);
% isscalar 判断是否为标量
if isscalar(K_or_centroids)
    K = K_or_centroids;
    % randomly pick centroids
    rndp = randperm(N);
    centroids = X(rndp(1:K), :);
else  % 矩阵，给出每一类的初始化
    K = size(K_or_centroids, 1);
    centroids = K_or_centroids;
end

% initial values
[pMiu pPi pSigma] = init_params();

Lprev = -inf;
while true
    %% Estiamtion Step
    Px = calc_prob();

    % new value for pGamma
    pGamma = Px .* repmat(pPi, N, 1);
    pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K);

    %% Maximization Step
    % new value for parameters of each Component
    Nk = sum(pGamma, 1);
    pMiu = diag(1./Nk) * pGamma' * X;
    pPi = Nk/N;
    for kk = 1:K
        Xshift = X-repmat(pMiu(kk, :), N, 1);
        pSigma(:, :, kk) = (Xshift' * (diag(pGamma(:, kk)) * Xshift)) / Nk(kk);
    end

    %% check for convergence
    L = sum(log(Px*pPi'));
    if L-Lprev < threshold
        break;
    end
    Lprev = L;
end

% 输出参数判定
if nargout == 1
    varargout = {Px};
else
    model = [];
    model.Miu = pMiu;
    model.Sigma = pSigma;
    model.Pi = pPi;
    varargout = {Px, model};
end

function [pMiu pPi pSigma] = init_params()
    pMiu = centroids;  % 均值，也就是K类的中心
    pPi = zeros(1, K); % 概率
    pSigma = zeros(D, D, K); %协方差矩阵，每个都是 D*D

    % hard assign x to each centroids 
    % (X - pMiu)^2 = X^2 + pMiu^2 - 2*X*pMiu
    distmat = repmat(sum(X.*X, 2), 1, K) + repmat(sum(pMiu.*pMiu, 2)', N, 1) - 2*X*pMiu';
    [dummy labels] = min(distmat, [], 2);

    for k=1:K   %初始化参数
        Xk = X(labels == k, :);
        pPi(k) = size(Xk, 1)/N;
        pSigma(:, :, k) = cov(Xk);
    end
end

% 计算概率
function Px = calc_prob()
    Px = zeros(N, K);
    for k = 1:K
        Xshift = X-repmat(pMiu(k, :), N, 1);
        inv_pSigma = inv(pSigma(:, :, k)+diag(repmat(threshold,1,size(pSigma(:, :, k),1)))); % 方差矩阵求逆
        tmp = sum((Xshift*inv_pSigma) .* Xshift, 2);
        coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma)); % det 求方差矩阵的行列式  
        Px(:, k) = coef * exp(-0.5*tmp);
    end
end
end
