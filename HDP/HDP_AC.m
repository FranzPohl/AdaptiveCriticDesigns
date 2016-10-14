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
load('model_trqLimited.mat')
r2d = 180/pi;

%% Actor and Critic Network

% Actor 
numInA     = 2;
numNeuronA = 8;
numOutA    = 1;
actor = NeuralNet([numInA, numNeuronA, numOutA]);
actor.transferFun{end} = sigmoid;
 
% Actor RL parameters
etaA   = 0.01; 
tauA   = 0.00;
muA    = 0.00;
eps    = 0.0; %exploration rate

% Critic
numInC     = 2;
numNeuronC = 8;
numOutC    = 1;
critic = NeuralNet([numInC, numNeuronC, numOutC]);

% Critic RL parameters
gamma   = 0.97;                                                                           
etaC    = 0.05; 
tauC    = 0.00;   
muC     = 0.00;   
lambdaC = 0.0;

% Choice of Reward function
% 1: binary
% 2: quadratic
% 3: weight matrix
% 4: cosine
choice = 3; 

%% STEP IV HDP CRITIC AND ACTOR CONNECTED AND NOT PRETRAINED
%load('criticUpswing.mat')

% Storage vectors
Rlog = [];
Xlog = [];
ActorWeights = [actor.weights{1}(:,1); actor.weights{1}(:,2); actor.weights{2}'];
ActorBias    = [actor.bias{1}; actor.bias{2}];
CriticWeights= [critic.weights{1}(:,1); critic.weights{1}(:,2); critic.weights{2}'];
CriticBias   = [critic.bias{1}; critic.bias{2}];

% Limitations
LB = [-pi; 0]; %Lower Bound
UB = [+pi; 0]; %Upper Bound
xdl= 35;

% Simulation time
tmax    = 5;
Ntrials = 400;
dt      = 0.005;
t       = 0:dt:tmax;
n       = length(t);
Jstar   = 0; %tmax/dt; 

% Initial Conditions
xini = lhsdesign(Ntrials,2);
xini = bsxfun( @plus, LB, bsxfun(@times, xini', (UB - LB)) );
% xini = bsxfun(@times, randn(2,Ntrials), [2/3*pi; 10/3] );
xini(1,:) = xini(1,:) + 2*pi .* [ abs(xini(1,:))>pi ] .* -sign(xini(1,:));
xini(:,end) = [pi;0];
%xini = [sign(randn(1,Ntrials))*pi; randn(1,Ntrials)];

for trial = 1:Ntrials
   
    tic
    
    clear r;
    clear x;
    clear u;
    clear J;
    clear xn;
    clear xhat;
    
    x = xini(:,trial);
    x(1) = x(1) + 2*pi * [ abs(x(1))>pi ] * -sign(x(1));
    xn = mapminmax( 'apply', x, pty ); 
    xhat = xn;
    
    % START OF TRIAL
    for j = 1:n-1
        
        u(j) = actor.FFwrd( xn(:,j) );    
       
        denorm = mapminmax( 'reverse', [ xn(:,j);u(j) ], ptx );
        
        x(:,j+1) = Inverted_Pendulum( x(:,j), denorm(3), dt );
        x(1,j+1) = x(1,j+1) + 2*pi * [abs(x(1,j+1))>pi] * -sign(x(1,j+1));
        xhat(:,j+1) = model.FFwrd( [xn(:,j);u(j)] );
        xn(:,j+1)= mapminmax( 'apply', x(:,j+1), pty );
        
        r(j) = reward( choice, xn(:,j+1) );
        J(j) = critic.FFwrd( xn(:,j+1) );
        delta_J = J(j) - Jstar;
        
        dJdx = critic.net_derivative( xn(:,j+1) );
        dxdu = model.net_derivative( [xn(:,j); u(j)], dJdx );
        
        critic.updateC_HDP( xn(:,j:j+1), r(j), etaC, muC, gamma, lambdaC );
        actor.updateA_HDP( xn(:,j), delta_J, dxdu(3), etaA, muA );
  
              
        if abs(x(2,j+1)) > xdl
            break;
        end
        
        % saving weights
        if mod(j,10) == 0
            ActorWeights = [ActorWeights, [actor.weights{1}(:,1); actor.weights{1}(:,2); actor.weights{2}']];
            CriticWeights= [CriticWeights, [critic.weights{1}(:,1); critic.weights{1}(:,2); critic.weights{2}']];
            ActorBias    = [ActorBias, [actor.bias{1}; actor.bias{2}]];
            CriticBias   = [CriticBias, [critic.bias{1}; critic.bias{2}]];
        end
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

PlotACResults




