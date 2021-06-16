function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = [ones(m,1) X]; %5000 x 401 =A1
Z2 = Theta1* X'; %25 x5000 debiera ser m x h
A2 = sigmoid(Z2)'; %5000 x 25
A2 = [ones(size(A2,1), 1) A2];% 5000 x 26
Z3 = Theta2*A2'; %10 x 5000
h = sigmoid(Z3)'; %5000 x 10

% Transform y from integers in 1:10 into vectors which would be returned by the
% output layer
c = 1:num_labels;
y2 = (c==y);
J1 = (1/m)*sum(sum(-y2.*log(h) - (1-y2).*log(1-h)));

Theta1(:,1)=0;
Theta2(:,1)=0;
reg1 = (lambda/(2*m))*(sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)));
J = J1 + reg1;

D3 = h - y2; % m x r A3=y2, 5000 x 10 r:number of output clasification

D2 = ((Theta2(:,2:end))'*D3') .* sigmoidGradient(Z2); % 25 x 5000 ^

De1 = D2 * (X); % size is (h x m) ⋅ (m x n) --> (h x n) ^'
De2 = D3' * (A2);%size is (r x m) ⋅ (m x [h+1]) --> (r x [h+1])

Theta1_grad = De1./m + (lambda/m)*Theta1;
Theta2_grad = De2./m + (lambda/m)*Theta2;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
