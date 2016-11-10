%% DHP Algortihm
clc; clear; close all;

% Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 
addpath(fullfile(mfilepath,'../PLANT')); 
addpath(fullfile(mfilepath,'../MODEL')); 

% Load pre-trained model
load('model.mat')
clear model
%% Neural Nets

% Actor 
numInA     = 2;
numNeuronA = 10;
numOutA    = 1;
actor = NeuralNet([numInA, numNeuronA, numOutA]);
actor.transferFun{end} = sigmoid;

% RL parameters Actor
etaA   = 0.05;
muA    = 0.00;
eps0   = .1;  %exploration rate
epsd   = -.01;
% Critic
numInC     = 2;
numNeuronC = 8;
numOutC    = 2;
critic = NeuralNet([numInC, numNeuronC, numOutC]);

% RL Parameters Critic
gamma   = 0.95;   % discount rate                                                                          
etaC    = 0.1;    % learning rate of critic ANN  
muC     = 0.00;   % momentum factor critic


% Model Neural Net
numInputs = 3;
numNeurons= 10;
numOutputs= 2;
struct =  [numInputs numNeurons numOutputs];
model = NeuralNet(struct);

etaM    = 0.01;                                     % learning rate critic
lambdaM = 0.00;                                     % regularization rate critic
muM     = 0.0;                                      % momentum factor critic  

% Reward Log
Rlog = [];

%% DHP training

% Simulation parameters
tend    = 3; %when it worked used 5
dt      = 0.005;
t       = 0:dt:tend;
n       = length(t);
r2d     = 180/pi;
xdl     = 35;                                       % limitation of angular rate

livestream = false;                                 % set livestream on/off
VideoFile;                                          % create video variables

Ntrials = 400; %400
for trial = 1:Ntrials
    
    clear x;
    clear xn;
    clear u;
    clear r;
    clear errorC;
    clear errorA;
    clear lambda;
    
    % initial state
    x = [randn(1)*0.6; 0]; % xn0
    xn= mapminmax('apply',x,pty); 
    xhat = xn;
    eps = eps0 * exp(epsd*trial );
    
    % START OF TRIAL
    for i = 1:n-1
        
        % critic timestep t
        lambda(:,i) = critic.FFwrd(xn(:,i));
        
        % actor timestep t
        if rand(1) < eps
            u(i) = 2*rand(1) - 1;
            RandomAction = true;
        else 
            RandomAction = false;
            u(i) = actor.FFwrd(xn(:,i));
        end
        
        % denormalize
        denorm = mapminmax('reverse',[xn(:,i);u(i)], ptx);
        
        % plant
        x(:,i+1) = Inverted_Pendulum( x(:,i), denorm(3), dt );
        x(1,i+1) = x(1,i+1) + 2*pi*[abs(x(1,i+1))>pi]*-sign(x(1,i+1));
        xhat(:,i+1)= model.FFwrd([xn(:,i);u(i)]);
        xn(:,i+1)= mapminmax( 'apply', x(:,i+1), pty );
        r(i) = reward( xn(:,i+1), u(i) );
        
        % critic timestep t + 1
        lambda(:,i+1) = critic.FFwrd( xn(:,i+1) );
        
        % get derivatives
        dr = reward_derivative( xn(:,i+1) );
        da_dx = actor.net_derivativeSingle( xn(:,i) );
        dxhat_dx = model.net_derivativeSingle( [xn(:,i);u(i)] );
        
        % train actor and critic
        errorC(:,i) = critic.updateC_DHP( xn(:,i), lambda(:,i:i+1), da_dx, dxhat_dx, dr(1:2), etaC, muC, gamma );
        model.SGD( [xn(:,i);u(i);xn(:,i+1)], 1, 1, etaM, muM, lambdaM );
         
        if RandomAction == false
            errorA(i) = actor.updateA_DHP( xn(:,i), lambda(:,i+1), dxhat_dx(:,3), dr(3), etaA, muA, gamma );
        end
        
        Vlog;
        
        if abs(x(2,i+1)) > xdl
            break;
        end
        
    end
    % END OF TRIAL
    
    Rlog = [Rlog sum(r)];
    mseC(trial) = .5*sum(sum(errorC.^2,1))/(length(xn));
    mseA(trial)  = .5*norm(errorA)/(length(errorA));
    mseM(trial) = model.evaluate([xn(:,1:end-1); u; xn(:,2:end)], lambdaM);
    
    fprintf('Trial %i/%i: TD = %i  Actor = %i Model = %i\n', trial, Ntrials, mseC(trial), mseA(trial), mseM(trial))
    
end

if livestream == true
    close(v)
end

%% Plotting

%save('Exp10DHP','Rlog','mseC','mseA','critic','actor');

PlotACResults




% % Critic
% numInC     = 2;
% numNeuronC = 8;
% numOutC    = 2;
% critic = NeuralNet([numInC, numNeuronC, numOutC]);
% 
% % RL Parameters Critic
% gamma   = 0.95;   % discount rate                                                                          
% etaC    = 0.300;  % learning rate of critic ANN  
% tauC    = 0.01;   % time-step updates critic
% muC     = 0.10;   % momentum factor critic
% 
% % Actor 
% numInA     = 2;
% numNeuronA = 8;
% numOutA    = 1;
% actor = NeuralNet([numInA, numNeuronA, numOutA]);
% actor.transferFun{end} = sigmoid;
% 
% % RL parameters Actor
% etaA   = 0.001;
% tauA   = 0.01;
% muA    = 0.00;