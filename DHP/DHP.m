%% HDP Algortihm
clc; clear; close all;

% Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 
addpath(fullfile(mfilepath,'../PLANT')); 
addpath(fullfile(mfilepath,'../MODEL')); 

% Load pre-trained model
load('model.mat')

% Simulation parameters
tmax    = 3;
dt      = 0.01;
t       = 0:dt:tmax;
n       = length(t);
r2d     =  180/pi;

%% Neural Nets

% Critic
numInC     = 2;
numNeuronC = 8;
numOutC    = 2;
critic = NeuralNet([numInC, numNeuronC, numOutC]);

% RL Parameters Critic
gamma   = 0.95;   % discount rate                                                                          
etaC    = 0.300;  % learning rate of critic ANN  
tauC    = 0.01;   % time-step updates critic
muC     = 0.10;   % momentum factor critic

% Actor 
numInA     = 2;
numNeuronA = 8;
numOutA    = 1;
actor = NeuralNet([numInA, numNeuronA, numOutA]);
actor.transferFun{end} = sigmoid;

% RL parameters Actor
etaA   = 0.001;
tauA   = 0.01;
muA    = 0.00;

%% STEP III HDP CRITIC AND ACTOR CONNECTED

% Limitations
xl = .95*pi;
xdl= 30;
Rlog = [];

Ntrials = 300;
for trial = 1:Ntrials
    
    clear x;
    clear xn;
    clear u;
    clear r;
    clear errorC;
    clear errorA;
    clear lambda;
    
    % initial state
    x = [randn(1)*0.6; randn(1)]; % xn0
    if mod(trial,20)== 0
        x = zeros(2,1);
    end
    xn= mapminmax('apply',x,pty); 
    
    for j = 1:n-1
        
        % critic
        lambda(:,j) = critic.FFwrd(xn(:,j));
        
        % actor
        u(j) = actor.FFwrd(xn(:,j));
        denorm = mapminmax('reverse',[xn(:,j);u(j)], ptx);
        
        % Plant
        x(:,j+1) = Inverted_Pendulum(x(:,j),denorm(3), dt);
        x(1,j+1) = x(1,j+1) + 2*pi*[abs(x(1,j+1))>pi]*-sign(x(1,j+1));
        xn(:,j+1)= mapminmax('apply',x(:,j+1),pty);
        r(j) = reward(xn(:,j+1));
        
        % Critic timestep + 1
        lambda(:,j+1) = critic.FFwrd(xn(:,j+1));
        
        % derivatives
        dr = reward_derivative(xn(:,j+1));
        da_dx = actor.net_derivativeSingle(xn(:,j));
        dxhat_dx = model.net_derivativeSingle([xn(:,j);u(j)]);
        
        % training
        errorC(:,j) = critic.updateC_DHP( xn(:,j), lambda(:,j:j+1), da_dx, dxhat_dx, dr(1:2), etaC, muC, gamma );
        errorA(j) = actor.updateA_DHP( xn(:,j), lambda(:,j+1), dxhat_dx(:,3), dr(3), etaA, muA, gamma );
     
        if abs(x(1,j+1)) > xl || abs(x(2,j+1)) > xdl
            break;
        end
        
    end
    
    Rlog = [Rlog sum(r)];
    mseC(trial) = .5*sum(sum(errorC.^2,1))/(length(xn));
    mseA(trial)  = .5*norm(errorA)/(length(xn));
    fprintf('Trial %i/%i: TD = %i    Actor Error = %i\n', trial, Ntrials, mseC(trial), mseA(trial))
    
end

%% Plotting
% Reward and MSE
figure()
subplot(3,1,1)
plot(1:Ntrials, Rlog)
xlabel('epochs'); ylabel('reward[-]');
subplot(3,1,2)
plot(mseA)
xlabel('epochs'); ylabel('mse Actor')
subplot(3,1,3)
plot(mseC)
xlabel('epochs'); ylabel('mse Critic [-]');
% print('HDP_results','-dpng','-r300');

figure()
subplot(3,1,1)
plot(t(1:length(u)),u)
xlabel('time [s]'); ylabel('actions')
grid on
subplot(3,1,2)
plot(t(1:length(x)),x(1,:)*r2d)
xlabel('time [s]'); ylabel('\theta [deg]')
grid on
subplot(3,1,3)
plot(t(1:length(x)),x(2,:)*r2d)
xlabel('time [s]'); ylabel('\theta_d [deg/s]')
grid on
% print('anglesHDP','-dpng','-r300');

% save('critic2','critic');
% save('actor2','actor');