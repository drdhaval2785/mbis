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


% Find activation for the third layer.
%size(Theta1) % 25 * 401
%size(Theta2) % 10 * 26
a1 = [ones(m,1) X]; % 5000 * 401
z2 = a1*Theta1'; % 5000 * 25
a2 = sigmoid(z2); % 5000 * 25
a2 = [ones(m,1) a2]; % 5000 * 26
z3 = a2*Theta2'; % 5000 * 10
a3 = sigmoid(z3); % 5000 * 10
% Prepare a zero matrix of m*k size
yi = zeros(size(a3)); % 5000 * 10
for i = 1:m,
	yi(i,y(i)) = 1;
end
J = (1/m)*sum((sum(-yi.*log(a3)-(1-yi).*log(1-a3)))) + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

DELTA2 = zeros(num_labels,hidden_layer_size+1);
DELTA1 = zeros(hidden_layer_size,input_layer_size+1);

%Vectorized implementation
% This is much more clean.
% 5000 is the size of training set.
% Whatever is happenning in the raws of vector in the conventional implementation, happens in the columns e.g. 10*1 would be 5000*10 in this implementation.
d3_vec = a3 - yi; % 5000 * 10
d2_vec = (d3_vec*Theta2).*(a2.*(1-a2)); % 5000 * 26
d2_vec = d2_vec(:,2:end); % 5000 * 25
DELTA2 = d3_vec'*a2; % 10 * 26
DELTA1 = d2_vec'*a1; % 25 * 401

% Conventional implementation
#{
% Backpropagation
for j = 1:m,
	a1 = [1; X(j,:)(:)]; % 401 * 1
	yout = zeros(num_labels,1);
	yout(y(j)) = 1;
	z2 = Theta1*a1; % 25 * 1
	a2 = sigmoid(z2); % 25 * 1
	a2 = [1; a2(:)]; % 26 * 1
	z3 = Theta2*a2; % 10 * 1
	a3 = sigmoid(z3); % 10 * 1
	d3 = a3 - yout; % 10 * 1
	d2 = (Theta2'*d3).*(a2.*(1-a2)); % 26 * 1
	d2 = d2(2:end); % 25 * 1
	DELTA2 = DELTA2 + d3*(a2'); % 10 * 26
	DELTA1 = DELTA1 + d2*(a1'); % 25 * 401
end
#}

% Ignore the first column of theta2 for regularization
temp2 = Theta2(:,2:end);
temp2 = [zeros(size(Theta2),1) temp2];
% Ignore the first column of theta1 for regularization
temp1 = Theta1(:,2:end);
temp1 = [zeros(size(Theta1),1) temp1];
Theta2_grad = (1/m)*DELTA2 + (lambda/m)*temp2;
Theta1_grad = (1/m)*DELTA1 + (lambda/m)*temp1;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
