%% DHP Algortihm
clc; clear; close all;

% Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 
addpath(fullfile(mfilepath,'../PLANT')); 
addpath(fullfile(mfilepath,'../MODEL')); 

% Load pre-trained model
load('model.mat')

%% Neural Nets

% Actor 
numInA     = 2;
numNeuronA = 10;
numOutA    = 1;
actor = NeuralNet([numInA, numNeuronA, numOutA]);
actor.transferFun{end} = sigmoid;

% RL parameters Actor
etaA   = 0.05;
tauA   = 0.00;
muA    = 0.00;
eps0    = .1;  %exploration rate

% Critic
numInC     = 2;
numNeuronC = 8;
numOutC    = 2;
critic = NeuralNet([numInC, numNeuronC, numOutC]);

% RL Parameters Critic
gamma   = 0.95;   % discount rate                                                                          
etaC    = 0.1;  % learning rate of critic ANN  
tauC    = 0.00;   % time-step updates critic
muC     = 0.00;   % momentum factor critic

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

%% STEP III HDP CRITIC AND ACTOR CONNECTED

% Limitations
xl = .95*pi;
xdl= 35;
Rlog = [];

% Simulation parameters
tmax    = 5;
dt      = 0.005;
t       = 0:dt:tmax;
n       = length(t);
r2d     =  180/pi;

Ntrials = 400;
for trial = 1:Ntrials
    
    clear x;
    clear xn;
    clear u;
    clear r;
    clear errorC;
    clear errorA;
    clear lambda;
    
    % initial state
    x = [randn(1)*0.4; 0]; % xn0
    xn= mapminmax('apply',x,pty); 
    eps = eps0 * exp( -.1*trial );
    for j = 1:n-1
        
        % critic
        lambda(:,j) = critic.FFwrd(xn(:,j));
        
        % actor
        if rand(1) < eps
            u(j) = 2*rand(1) - 1;
            RandomAction = true;
        else 
            RandomAction = false;
            u(j) = actor.FFwrd(xn(:,j));
        end
        
        denorm = mapminmax('reverse',[xn(:,j);u(j)], ptx);
        
        % Plant
        x(:,j+1) = Inverted_Pendulum( x(:,j), denorm(3), dt );
        x(1,j+1) = x(1,j+1) + 2*pi*[abs(x(1,j+1))>pi]*-sign(x(1,j+1));
        xn(:,j+1)= mapminmax( 'apply', x(:,j+1), pty );
        r(j) = reward( xn(:,j+1) );
        
        % Critic timestep + 1
        lambda(:,j+1) = critic.FFwrd( xn(:,j+1) );
        
        % derivatives
        dr = reward_derivative( xn(:,j+1) );
        da_dx = actor.net_derivativeSingle( xn(:,j) );
        dxhat_dx = model.net_derivativeSingle( [xn(:,j);u(j)] );
        
        % training
        errorC(:,j) = critic.updateC_DHP( xn(:,j), lambda(:,j:j+1), da_dx, dxhat_dx, dr(1:2), etaC, muC, gamma );
       
        if RandomAction == false
            errorA(j) = actor.updateA_DHP( xn(:,j), lambda(:,j+1), dxhat_dx(:,3), dr(3), etaA, muA, gamma );
        end
     
        if abs(x(2,j+1)) > xdl %|| abs(x(1,j+1)) > xl
            break;
        end
        
    end
    
    Rlog = [Rlog sum(r)];
    mseC(trial) = .5*sum(sum(errorC.^2,1))/(length(xn));
    mseA(trial)  = .5*norm(errorA)/(length(errorA));
    fprintf('Trial %i/%i: TD = %i    Actor Error = %i\n', trial, Ntrials, mseC(trial), mseA(trial))
    
end

%% Plotting
PlotACResults