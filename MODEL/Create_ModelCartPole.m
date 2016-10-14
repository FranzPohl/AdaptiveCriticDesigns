%% MODEL NETWORK
% Creates and pre-trains the model Network. Needs NeuralNet.m class
% Training and validation data sets are acquired with Dataset.m
% Stochastic Gradient Descent
% ModelNet inputs: - theta
%                  - theta dot
%                  - action (control signal)
%
% ModelNet output: - theta
%                  - theta dot
% author: Franz Pohl
% email: Franz.m.Pohl@gmail.com

clc;
clear;
close all;

%%Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 

%% Load Data Set

load('data_T.mat');  % training data set
load('data_V.mat');  % validation data set

% Test Data
x          = train_x;            % input to Model Network (train)
d          = train_data_xp;      % desired output (train)
train_data = [x; d];
xv         = val_x;              % input to Model Network (validation)
dv         = val_data_xp;        % desired output (validation)
test_data  = [val_x; val_data_xp];
r2d = 180/pi;

%% Construct Model ANN

% network structure
numNeurons = 10;
numInputs  = 5;
numOutputs = 4;
NetworkStruct = [numInputs numNeurons numOutputs];

% Learning Prameters
nbatch = 10;
epochs = 10;
eta    = 0.1;

% Model Neural Network
model = NeuralNet(NetworkStruct);


%% Train Model Network

model.SGD(train_data, epochs, eta, nbatch, test_data)

for i= 1:length(test_data)
    yNN(:,i) = model.FFwrd(val_x(:,i));
end

%% Plot

figure
subplot(4,1,1)
plot(dv(1,:))
hold on
plot(yNN(1,:),'r--')
grid on
ylabel('distance x [m]');
xlabel('time [s]');
legend('plant output','model output', 'Location','NorthEast')

subplot(4,1,2)
plot(dv(2,:))
hold on
plot(yNN(2,:),'r--')
grid on
ylabel('velocity [m/s]');
xlabel('time [s]');
legend('validation data','model output', 'Location','NorthEast')


subplot(4,1,3)
plot(dv(3,:))
hold on
plot(yNN(3,:)*r2d,'r--')
grid on
ylabel('\theta [deg]');
xlabel('time [s]');
legend('validation data','model output', 'Location','NorthEast')

subplot(4,1,4)
plot(dv(4,:))
hold on
plot(yNN(4,:)*r2d,'r--')
grid on
ylabel('\theta dot  [deg/s]');
xlabel('time [s]');
legend('validation data','model output', 'Location','NorthEast')
%print('states','-dpng','-r300');

