%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 10;   % 50 hidden units
num_labels = 2;          % 2 labels, from 1 to 2   
                          % (note that we have mapped "0" to label 2)
threshold = 90; % Can be 95 / 99 for 95%, 99% respectively.

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Normalizing Data ...\n')

%Normalize raining data
fprintf('Normalizing the training data and storing in hottiedata/input/normalized/traindata.mat\n');
X2 = csvread('hottiedata/input/trues.csv');
X2 = rgbnormalize(X2);
X1 = csvread('hottiedata/input/falses.csv');
X1 = rgbnormalize(X1);
X = [X1;X2];
y = [ones(size(X1,1),1).*2;ones(size(X2,1),1)];
z = (1:size(y))(:);
combo = [X y z];
combo = combo(randperm(size(combo,1)),:);
m = size(combo,1);
mtrain = (m * 6) / 10;
mcv = (m * 8) / 10;
Xtrain = combo(1:mtrain,1:input_layer_size);
ytrain = combo(1:mtrain,input_layer_size+1);
ztrain = combo(1:mtrain,input_layer_size+2);
Xcv = combo(mtrain+1:mcv,1:input_layer_size);
ycv = combo(mtrain+1:mcv,input_layer_size+1);
zcv = combo(mtrain+1:mcv,input_layer_size+2);
Xtest = combo(mcv+1:end,1:input_layer_size);
ytest = combo(mcv+1:end,input_layer_size+1);
ztest = combo(mcv+1:end,input_layer_size+2);
lambda = 1;

save('hottiedata/input/normalized/traindata.mat','Xtrain','ytrain','ztrain','m','mtrain','mcv','Xcv','ycv','zcv','Xtest','ytest','ztest');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Loading Parameters ================

% Load the weights into variables Theta1 and Theta2
if exist('hottiedata/input/learntweights.mat') == 2,
	fprintf("Loading the previously learnt NN parameters from hottiedata/input/learntweights.mat");
	load('hottiedata/input/learntweights.mat')
	initial_Theta1 = Theta1;
	initial_Theta2 = Theta2;
else
	initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
	initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels);
end
% Unroll parameters 
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Without learning, classification accuracy
testpredwithoutlearning = predict(initial_Theta1, initial_Theta2, Xtest);
fprintf('\nTest Set Identification from previously learnt parameters: %d / %d\n', sum(double(testpredwithoutlearning==ytest)), size(ytest,1));


%% =================== Part 4: Training NN ===================
fprintf('\nTraining Neural Network... \n')

% Set the maximum iterations
options = optimset('MaxIter', 10);

costFunction = @(p) nnCostFunction(p, ...
								   input_layer_size, ...
								   hidden_layer_size, ...
								   num_labels, Xtrain, ytrain, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
#{
%  You should also try different values of lambda
if exist('hottiedata/input/normalized/learningcurve.mat')==2
	% Create "short hand" for the cost function to be minimized
	load('hottiedata/input/normalized/learningcurve.mat');
	plot(1:mtrain,Costtrain,1:mtrain,Costcv,1:mtrain,Costtest);
	pause();
else
	for i = 1:mtrain,
		% Create "short hand" for the cost function to be minimized
		costFunction = @(p) nnCostFunction(p, ...
										   input_layer_size, ...
										   hidden_layer_size, ...
										   num_labels, Xtrain(1:i,:), ytrain(1:i,:), lambda);

		% Now, costFunction is a function that takes in only one argument (the
		% neural network parameters)
		[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

		% error on test set
		[Costtrain(i), scraptrain] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Xtrain(1:i,:), ytrain(1:i,:), lambda);
		[Costcv(i), scrapcv] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Xcv, ycv, lambda);
		[Costtest(i), scrapcv] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Xtest, ytest, lambda);
		fprintf('m=%d, Costtrain=%d, Costcv=%d',i,Costtrain(i),Costcv(i));
	end
	save('hottiedata/input/normalized/learningcurve.mat','Costtrain','mtrain','Costcv','Costtest');
	plot(1:mtrain,Costtrain,1:mtrain,Costcv,1:mtrain,Costtest);
	pause();
end
#}

#{
% Validation Curve
% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
for i = 1:length(lambda_vec)
	lambda = lambda_vec(i);
	costFunction = @(p) nnCostFunction(p, ...
									   input_layer_size, ...
									   hidden_layer_size, ...
									   num_labels, Xtrain, ytrain, lambda);
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
	[error_train(i) sc1] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Xtrain, ytrain, lambda);
	[error_val(i) sc2] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Xcv, ycv, lambda);
end
save('hottiedata/input/normalized/validationcurve.mat','error_train','lambda_vec','error_val');
plot(lambda_vec,error_train,lambda_vec,error_val);
pause();
#}

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 5: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

%fprintf('\nVisualizing Neural Network... \n')

%displayData(Theta1(:, 2:end));

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ================= Part 6: Implement Predict =================

Predtrain = predict(Theta1, Theta2, Xtrain);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(Predtrain == ytrain)) * 100);

Predcv = predict(Theta1, Theta2, Xcv);
fprintf('\nCross validation Set Identification: %d / %d\n', sum(double(Predcv==ycv)), size(ycv,1));

#{
for i = 1:length(Predcv)
	fprintf('%d %d\n',Predcv(i),ycv(i));
end
#}
Predtest = predict(Theta1, Theta2, Xtest);
fprintf('\nTest Set Identification: %d / %d\n', sum(double(Predtest==ytest)), size(ytest,1));
% Displaying false positive files
full_file = fopen('hottiedata/input/fullset.txt');
number_of_lines = fskipl(full_file, Inf);
frewind(full_file);
falseposcells = cell(number_of_lines, 1);
for i = 1:number_of_lines
    fullcells{i} = fscanf(full_file, '%s', 1);
end

falsenegatives = ztest(Predtest==2 & ytest==1)
falsenegativefile = fullcells(falsenegatives)

falsepositives = ztest(Predtest==1 & ytest==2)
falsepositivefile = fullcells(falsepositives)
#{
for i = 1:length(Predtest)
	fprintf('%d %d\n',Predtest(i),ytest(i));
end
#}
%if mean(double(Predtest==ytest))*100 > threshold
%	save('hottiedata/input/learntweights.mat','Theta1','Theta2')
%end

