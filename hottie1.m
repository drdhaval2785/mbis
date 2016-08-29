%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20X4 Input Images of Digits
hidden_layer_size = 50;   % 50 hidden units
num_labels = 2;          % 2 labels, from 1 to 2   
                          % (note that we have mapped "0" to label 2)
threshold = 80; % Can be 95 / 99 for 95%, 99% respectively. Right now kept at 80%. If the test set accuracy is above threshold, the parameters are stored in hottiedata/input/learntparameters.mat.
lambda = 1; % Regularization parameter.

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Normalizing Training Data ...\n')

% If there is a previously normalized data available, load from there.
if exist('hottiedata/input/normalized/traindata.mat') == 2,
	load('hottiedata/input/normalized/traindata.mat');
% Otherwise normalize the data and store it is the traindata.mat file.
else
	%Normalize raining data
	fprintf('Normalizing the training data and storing in hottiedata/input/normalized/traindata.mat\n');
	% Read the CSV files of pixels into matrices.
	X2 = csvread('hottiedata/input/trues.csv');
	X1 = csvread('hottiedata/input/falses.csv');
	% Add true and false data. It is mandatory to do this before normalization. This error cost me 7-8 hours to correct.
	X = [X1;X2];
	% Normalize the input data for training.
	X =rgbnormalize(X);
	% As the images can be flipped left to right without changing characteristics, Added the flipped data. This will double our training data.
	Xflip = fliplr(X);
	% Added both unflipped and flipped data.
	X = [X;Xflip];
	% Creating a vector with value 2 for negative and 1 for positive. This will be used to compare the output from our training algorithm.
	y = [ones(size(X1,1),1).*2;ones(size(X2,1),1);ones(size(X1,1),1).*2;ones(size(X2,1),1)];
	% Created an index of the image data. We will be randomly shuffling the input data. So a unique index will be easy way to retrieve image file name from data later on.
	z = (1:size(y))(:);
	% Created 400 + 1 +1 = 402 X m dimentional matrix, to reshuffle.
	combo = [X y z];
	% Reshuffled the matrix, to select training and crossvalidation sets randomly.
	combo = combo(randperm(size(combo,1)),:);
	% Total number of observations is denoted by 'm'.
	m = size(combo,1);
	% Print m to screen.
	m
	% 80% of the input data is retained for training set.
	mtrain = floor((m*8)/10);
	Xtrain = combo(1:mtrain,1:input_layer_size); % Input pixel data
	ytrain = combo(1:mtrain,input_layer_size+1); % Input classification - 1 for positive, 2 for negative
	ztrain = combo(1:mtrain,input_layer_size+2); % Input indexing
	% Rest 20% of the input data is used for cross validation set.
	Xcv = combo(mtrain+1:m,1:input_layer_size);
	ycv = combo(mtrain+1:m,input_layer_size+1);
	zcv = combo(mtrain+1:m,input_layer_size+2);
	% Using the test data.
	% Currently we have kept 100 positive and 100 negative images (known), to see the output of our learning.
	% Later on it will be only Xtest. There would not be any ytest associated with it.
	Xtest = csvread('hottiedata/input/tobesorted.csv');
	Xtest = rgbnormalize(Xtest);
	%ytest = [ones(100,1);ones(100,1).*2];
	ztest = (1:size(Xtest,1))(:);
	testcombo = [Xtest ytest ztest];
	testcombo = testcombo(randperm(size(testcombo,1)),:);
	Xtest = testcombo(:,1:input_layer_size);
	%ytest = testcombo(:,input_layer_size+1);
	ztest = testcombo(:,input_layer_size+2);
	% Save into traindata.mat, for future use if any.
	%save('hottiedata/input/normalized/traindata.mat','Xtrain','ytrain','ztrain','m','mtrain','Xcv','ycv','zcv','Xtest','ytest','ztest');
	save('hottiedata/input/normalized/traindata.mat','Xtrain','ytrain','ztrain','m','mtrain','Xcv','ycv','zcv','Xtest','ztest');
end


%% ================ Part 2: Loading Parameters ================

% Load the weights into variables Theta1 and Theta2
if exist('hottiedata/input/learntweights.mat') == 2,
	fprintf("Loading the previously learnt NN parameters from hottiedata/input/learntweights.mat");
	load('hottiedata/input/learntweights.mat')
	% Rename the parameters. This will work as starting weights for neural networks.
	initial_Theta1 = Theta1;
	initial_Theta2 = Theta2;
else
	% Randomly initialize the parameters initial_Theta1 and initial_Theta2
	initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
	initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels);
end

% Unroll parameters 
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Without learning, classification accuracy on Cross Validation test
cvpredwithoutlearning = predict(initial_Theta1, initial_Theta2, Xcv);
fprintf('\nCross Validation Set Identification from previously learnt parameters: %d / %d\n', sum(double(cvpredwithoutlearning==ycv)), size(ycv,1));


%% =================== Part 4: Training NN ===================
fprintf('\nTraining Neural Network... \n')

% Set the maximum iterations
options = optimset('MaxIter', 100);

% Defined a cost function for neural network, to minimize.
costFunction = @(p) nnCostFunction(p, ...
								   input_layer_size, ...
								   hidden_layer_size, ...
								   num_labels, Xtrain, ytrain, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


% Learning Curve. 
% It is used to debug the neural network. Not necessary once you decide on neural network.
#{
if exist('hottiedata/input/normalized/learningcurve.mat')==2
	% Create "short hand" for the cost function to be minimized
	load('hottiedata/input/normalized/learningcurve.mat');
	plot(1:mtrain,Costtrain,1:mtrain,Costcv);
	pause();
else
	for i = 1:floor(mtrain/10),
		i = i*10; % If the training data is large, it is too long to make iterations with every item. So taking iteration only at 10 intervals.
		% Take first i observations for training and get the Cross validation errors for it.
		costFunction = @(p) nnCostFunction(p, ...
										   input_layer_size, ...
										   hidden_layer_size, ...
										   num_labels, Xtrain(1:i,:), ytrain(1:i,:), lambda);

		% Now, costFunction is a function that takes in only one argument (the
		% neural network parameters)
		[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
		
		v(i) = i;
		% error on training set in each iteration.
		[Costtrain(i), scraptrain] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Xtrain(1:i,:), ytrain(1:i,:), lambda);
		% error on cross validation set in each iteration.
		[Costcv(i), scrapcv] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Xcv, ycv, lambda);
		% Print the cost of training and cross validation to the terminal.
		fprintf('m=%d, Costtrain=%d, Costcv=%d',i,Costtrain(i),Costcv(i));
	end
	% Save the data (to plot the data fast in future run).
	save('hottiedata/input/normalized/learningcurve.mat','Costtrain','mtrain','Costcv');
	% Plot the learning curve, for visual feel.
	plot(v,Costtrain,v,Costcv);
end
#}

% Validation Curve
% This is also for debugging Neural Network. Not much used after we decide on neural network architecture.
% It is used to select the value of lambda after seeing it visually. Select lambda with minimum Cost on Cross Validation set.
#{
% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% Initialize empty vectors to store the results.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
% For each value of lambda, train the neural network.
for i = 1:length(lambda_vec)
	lambda = lambda_vec(i);
	costFunction = @(p) nnCostFunction(p, ...
									   input_layer_size, ...
									   hidden_layer_size, ...
									   num_labels, Xtrain, ytrain, lambda);
	% Train neural network
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
	% Cost on training set
	[error_train(i) sc1] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Xtrain, ytrain, lambda);
	% Cost on cross validation set.
	[error_val(i) sc2] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Xcv, ycv, lambda);
end
% Save for future plotting, if needed (Saves on repeat training).
save('hottiedata/input/normalized/validationcurve.mat','error_train','lambda_vec','error_val');
% Plot
plot(lambda_vec,error_train,lambda_vec,error_val);
#}

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



%% ================= Part 5: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

%fprintf('\nVisualizing Neural Network... \n')

%displayData(Theta1(:, 2:end));

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ================= Part 6: Implement Predict =================

% Predict training accuracy	
Predtrain = predict(Theta1, Theta2, Xtrain);
fprintf('\nTraining Set Identification: %d / %d\n', sum(double(Predtrain==ytrain)), size(ytrain,1));
fprintf('\nTraining Set Accuracy: %f\n', mean(double(Predtrain == ytrain)) * 100);

% Predict cross validation set accuracy.
Predcv = predict(Theta1, Theta2, Xcv);
fprintf('\nCross validation Set Identification: %d / %d\n', sum(double(Predcv==ycv)), size(ycv,1));
fprintf('\nCross Validation Set Accuracy: %f\n', mean(double(Predcv == ycv)) * 100);

% The above two predictions are used to identify whether the NN works as per our requirements or not.

% Run the Neural Network on test set (The data to be classified).
Predtest = predict(Theta1, Theta2, Xtest);

% Read the names of files to be sorted into a cell array.
% Borrowed from http://stackoverflow.com/questions/10971332/octave-load-string-from-file-to-matrix
full_file = fopen('hottiedata/input/tosort.txt');
number_of_lines = fskipl(full_file, Inf);
frewind(full_file);
for i = 1:number_of_lines
    fullcells{i} = fscanf(full_file, '%s', 1);
end

% Find indices of the items predicted to be positive by Neural network.
predpositives = ztest(Predtest==1);
% Get the name of file of predicted positive files.
predpositivefile = fullcells(predpositives);
% Write the names to a txt file.
celltocsv(predpositivefile,'hottiedata/output/positive.txt');

% Same three steps for items predicted to be negative by Neural network.
prednegatives = ztest(Predtest==2);
prednegativefile = fullcells(prednegatives);
celltocsv(prednegativefile,'hottiedata/output/negative.txt');


% This section presumes that we have labels for test data also. But it is not the case usually. So bracketed it out. Used for development purpose.
#{
falsenegatives = ztest(Predtest==2 & ytest==1);
falsenegativefile = fullcells(falsenegatives)
celltocsv(falsenagativefile,'hottiedata/output/falsenegative.txt');

falsepositives = ztest(Predtest==1 & ytest==2);
falsepositivefile = fullcells(falsepositives)
celltocsv(falsenagativefile,'hottiedata/output/falsepositive.txt');
#}

% If the prediction on Cross validation set is higher than the threshold, store the parameters of Neural network into a file.
% This may be loaded directly.
if mean(double(Predcv==ycv))*100 > threshold
	save('hottiedata/input/learntweights.mat','Theta1','Theta2')
end

fprintf("Next steps are -");
fprintf("1. run sh postprocess.sh");
fprintf("2. Check the folders Images/Sorted/positives and Images/Sorted/negatives for output.");
