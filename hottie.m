%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 10;   % 50 hidden units
num_labels = 2;          % 2 labels, from 1 to 2   
                          % (note that we have mapped "0" to label 2)
threshold = 80; % Can be 95 / 99 for 95%, 99% respectively.

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Normalizing Data ...\n')

%Normalize raining data
if exist('hottiedata/input/normalized/traindata.mat') == 2,
	fprintf('Loading normalized data from hottiedata/input/normalized/traindata.mat.\nIf you want to test on new data, delete this file and rerun the code.\n')
	load('hottiedata/input/normalized/traindata.mat');
else
	fprintf('Normalizing the training data and storing in hottiedata/input/normalized/traindata.mat');
	X2 = csvread('hottiedata/input/trues.csv');
	X2 = rgbnormalize(X2);
	X1 = csvread('hottiedata/input/falses.csv');
	X1 = rgbnormalize(X1);
	X = [X1;X2];
	y = [ones(size(X1,1),1).*2;ones(size(X2,1),1)];
	combo = [X y];
	combo = combo(randperm(size(combo,1)),:);
	X = combo(:,1:input_layer_size);
	y = combo(:,end);
	m = size(X, 1);
	testX2 = csvread('hottiedata/input/testpositives.csv');
	testX2 = rgbnormalize(testX2);
	testX1 = csvread('hottiedata/input/testnegatives.csv');
	testX1 = rgbnormalize(testX1);
	testX = [testX1;testX2];
	testy = [ones(size(testX1,1),1).*2;ones(size(testX2,1),1)];
	combo1 = [testX testy];
	combo1 = combo1(randperm(size(combo1,1)),:);
	testX = combo1(:,1:input_layer_size);
	testy = combo1(:,end);
	testm = size(testX1,1);
	save('hottiedata/input/normalized/traindata.mat','X','y','m','testX','testy','testm');
end
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
testpredwithoutlearning = predict(initial_Theta1, initial_Theta2, testX);
fprintf('\nTest Set Identification from previously learnt parameters: %d / %d\n', sum(double(testpredwithoutlearning==testy)), size(testy,1));


%% =================== Part 4: Training NN ===================
fprintf('\nTraining Neural Network... \n')

% Set the maximum iterations
options = optimset('MaxIter', 100);

%  You should also try different values of lambda
lambda = 0.5;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

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

pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

testpred = predict(Theta1, Theta2, testX);
fprintf('\nTest Set Identification: %d / %d\n', sum(double(testpred==testy)), size(testy,1));
wrongs = find(double(testpred~=testy));
falsepositives = wrongs(wrongs <= testm);
falsenegatives = wrongs(wrongs > testm) .- testm;
%falsenegatives = wrongs(wrongs > testm)
trainingaccuracy = mean(double(testpred == testy)) * 100;
fprintf('\nTesting Set Accuracy: %f\n', trainingaccuracy);

if trainingaccuracy > threshold
	save('hottiedata/input/learntweights.mat','Theta1','Theta2');
end
treatedpositive = sum(double(testpred==1))
treatednegative = sum(double(testpred==2))
% Displaying false positive files
falsepositive_file = fopen('hottiedata/input/testminus.txt');
number_of_lines = fskipl(falsepositive_file, Inf);
frewind(falsepositive_file);
falseposcells = cell(number_of_lines, 1);
for i = 1:number_of_lines
    falseposcells{i} = fscanf(falsepositive_file, '%s', 1);
end
falsepositivefile = falseposcells(falsepositives)

% Displaying false negative files
falsenegative_file = fopen('hottiedata/input/testplus.txt');
number_of_lines = fskipl(falsenegative_file, Inf);
frewind(falsenegative_file);
falsenegcells = cell(number_of_lines, 1);
for i = 1:number_of_lines
    falsenegcells{i} = fscanf(falsenegative_file, '%s', 1);
end
falsenegativefile = falsenegcells(falsenegatives)

csvwrite('hottiedata/output/falsepositives.csv',falsepositives)
csvwrite('hottiedata/output/falsenegatives.csv',falsenegatives)

