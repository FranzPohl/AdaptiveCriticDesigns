%% HDP_AC
% Trains Actor and Critic Simultanously. The model of the plant is 
% pretrained.
% grid search 
% update Actor with TD and eligibility traces

clc; clear; close all;

% Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 
addpath(fullfile(mfilepath,'../PLANT')); 
addpath(fullfile(mfilepath,'../MODEL')); 

% Load pre-trained model, critic and actor
load('model.mat')
r2d = 180/pi;

%% Actor and Critic Network

% Actor Neural Net
numInA     = 2;                                     % number of input neurons actor
numNeuronA = 10;                                    % number of hidden neurons actor
numOutA    = 1;                                     % number of output neurons actor
actor = NeuralNet([numInA, numNeuronA, numOutA]);   % construct actor NNN
actor.transferFun{end} = sigmoid;                   % outputlayer transfer functions
        
% Actor RL parameters
etaA    = 0.01;                                     % learning rate actor NN
lambdaA = 0.00;                                     % regularization rate actor NN
muA     = 0.00;                                     % momentum actor NN
Jstar   = 0;                                        % optimum value
eps0    = 0.1;                                      % exploration rate
lambda  = -.05;

% Critic
numInC     = 2;                                     % number of input neurons critic
numNeuronC = 8;                                     % number of hidden neurons critic
numOutC    = 1;                                     % number of output neurons critic             
critic = NeuralNet([numInC, numNeuronC, numOutC]);  % construct critic NNN

% Critic RL parameters
gamma   = 0.97;                                     % discount factor                                                                                
etaC    = 0.05;                                     % learning rate critic
lambdaC = 0.00;                                     % regularization rate critic
muC     = 0.10;                                     % momentum factor critic  

% Choice of Reward function
choice = 3; % 1:binary, 2:quadratic, 3:weight matrix, 4:cosine

% Storage vectors
Rlog = []; Xlog = [];
ActorWeights = [actor.weights{1}(:,1); actor.weights{1}(:,2); actor.weights{2}'];
ActorBias    = [actor.bias{1}; actor.bias{2}];
CriticWeights= [critic.weights{1}(:,1); critic.weights{1}(:,2); critic.weights{2}'];
CriticBias   = [critic.bias{1}; critic.bias{2}];

%% Simulation

% Simulation Settings
tend    = 3;                                        % final time
dt      = 0.005;                                    % time step
t       = 0:dt:tend;                                % time vector
n       = length(t);                                % number of total timesteps
xdl     = 35;                                       % limitation of angular rate
x0      = [randn(1,n).*.6;randn(1,n).*0];           % initial conditions

livestream = false;                                 % set livestream on/off
VideoFile;                                          % create video variables

Ntrials = 300;                                      % total number of trials
for trial = 1:Ntrials
   
    tic
    
    clear r;
    clear x;
    clear u;
    clear J;
    clear xn;
    clear xhat;
    
%     xini = [sign(randn(1))*trial*.01;0];
    x = x0(:,trial);
    x(1) = x(1) + 2*pi * [ abs(x(1))>pi ] * -sign(x(1));
    xn = mapminmax( 'apply', x, pty ); 
    xhat = xn;
    eps = eps0*exp(lambda*trial);
    
    % START OF TRIAL
    for j = 1:n-1
        
        if rand(1) < eps0
            u(j) = 2*rand(1) - 1;   
            RandomAction = true;
        else 
            RandomAction = false;
            u(j) = actor.FFwrd( xn(:,j) ); 
        end
        
        denorm = mapminmax( 'reverse', [ xn(:,j);u(j) ], ptx );
        
        x(:,j+1) = Inverted_Pendulum( x(:,j), denorm(3), dt );
        x(1,j+1) = x(1,j+1) + 2*pi * [abs(x(1,j+1))>pi] * -sign(x(1,j+1));
        xhat(:,j+1) = model.FFwrd( [xn(:,j);u(j)] );
        xn(:,j+1)= mapminmax( 'apply', x(:,j+1), pty );
        
        r(j) = reward( choice, xn(:,j+1), u(j) );
        J(j) = critic.FFwrd( xn(:,j+1) );
        delta_J = J(j) - Jstar;
        
        dJdx = critic.net_derivative( xn(:,j+1) );
        dxdu = model.net_derivative( [xn(:,j); u(j)], dJdx );
        
        critic.updateC_HDP( xn(:,j:j+1), r(j), etaC, muC, gamma, lambdaC );
        
        if RandomAction == false
            actor.updateA_HDP( xn(:,j), delta_J, dxdu(3), etaA, muA );
        end
  
        Vlog;
       
        if abs(x(2,j+1)) > xdl
            break;
        end
        
        % saving weights
%         if mod(j,10) == 0
%             ActorWeights = [ActorWeights, [actor.weights{1}(:,1); actor.weights{1}(:,2); actor.weights{2}']];
%             CriticWeights= [CriticWeights, [critic.weights{1}(:,1); critic.weights{1}(:,2); critic.weights{2}']];
%             ActorBias    = [ActorBias, [actor.bias{1}; actor.bias{2}]];
%             CriticBias   = [CriticBias, [critic.bias{1}; critic.bias{2}]];
%         end
    end

    % Cumulative reward 
    Rlog = [Rlog sum(r)];
    Xlog = [Xlog xn];
    
    % Error
    mseC(trial) = critic.evaluateC(xn, gamma, r);
    mseA(trial)  = .5*norm(J)/(length(xn));
    
    if mseC(trial) < 10^-4 && mseA(trial) < 10^-3
        fprintf('this trial was good with %i theta and %i thetadot \n', x(1,1), x(2,1));
    end
    
    fprintf('Trial %i/%i: TD = %i    Actor Error = %i\n', trial, Ntrials, mseC(trial), mseA(trial))
   
    % Simulation time computation
    comptime( trial ) = toc;
    avgtime = sum( comptime ) / trial;
    rem_time = ( Ntrials - trial ) * avgtime;
    rem_hour = floor( rem_time/3600 ); rem_time = mod( rem_time, 3600 );
    rem_min  = floor( rem_time/60 );   rem_time = mod( rem_time, 60 );
    rem_sec  = floor( rem_time );
    
    fprintf('Trial %i/%i is finished in %.2f seconds, estimated time remaining: %i hours %i min %i sec \n', ...
        trial, Ntrials, comptime( trial ), rem_hour, rem_min, rem_sec );

end

if livestream == true
    close(v)
end

%% Plotting

save('Exp10HDP','Rlog','mseC','mseA','critic','actor');

PlotACResults




