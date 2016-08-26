%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============

%Normalize raining data
fprintf('Normalizing the data to be sorted and storing in hottiedata/input/normalized/sortdata.mat\n');
tobesortedX = csvread('hottiedata/input/tobesorted.csv');
tobesortedX = rgbnormalize(tobesortedX);
save('hottiedata/input/normalized/sortdata.mat','X');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Loading Parameters ================

% Load the weights into variables Theta1 and Theta2
fprintf("Loading the previously learnt NN parameters from hottiedata/input/learntweights.mat\n");
load('hottiedata/input/learntweights.mat')
whos
tobesortedX;
testpredwithoutlearning = predict(Theta1, Theta2, tobesortedX);
sum(testpredwithoutlearning==1)
sum(testpredwithoutlearning==2)

% Displaying false positive files
full_file = fopen('hottiedata/input/tosort.txt');
number_of_lines = fskipl(full_file, Inf);
frewind(full_file);
falseposcells = cell(number_of_lines, 1);
for i = 1:number_of_lines
    fullcells{i} = fscanf(full_file, '%s', 1);
end

positivefiles = fullcells(testpredwithoutlearning==1)
negativefiles = fullcells(testpredwithoutlearning==2)
