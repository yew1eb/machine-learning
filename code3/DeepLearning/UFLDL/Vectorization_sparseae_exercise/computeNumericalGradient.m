function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros   2x1  这是输入X的行数，其实也就是参数W的个数
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 


epsilon = 0.000001;
% 2 x2  得到参数的个数
numW = size(theta,1);
I = eye(numW);
I =  I * epsilon;
for j = 1:numW
    numgrad(j) = (J(theta+I(:,j)) - J(theta-I(:,j))) / (2 * epsilon);
end






%% ---------------------------------------------------------------
end
