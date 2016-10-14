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
numNeuronA = 12;
numOutA    = 1;
actor = NeuralNet([numInA, numNeuronA, numOutA]);
actor.transferFun{end} = sigmoid;
 
% Actor RL parameters
etaA   = 0.005;
tauA   = 0.00;
muA    = 0.00;


% Critic
numInC     = 2;
numNeuronC = 10;
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
Jstar = 0;

%% STEP IV HDP CRITIC AND ACTOR CONNECTED AND NOT PRETRAINED
%load('criticUpswing.mat')

% Storage vectors
Rlog      = [];
Thetalog  = [];
ThetaDlog = [];
Xlogn     = [];

% Limitations
LB = [-3/2*pi; -10]; %Lower Bound
UB = [3/2*pi; 10];   %Upper Bound
xdl= 40;

% Simulation time
tmax    = 5;
Ntrials = 400;
dt      = 0.005;
t       = 0:dt:tmax;
n       = length(t);

% Initial Conditions
xini = lhsdesign(Ntrials,2);
xini = bsxfun( @plus, LB, bsxfun(@times, xini', (UB - LB)) );
xini(1,:) = xini(1,:) + 2*pi .* [ abs(xini(1,:))>pi ] .* -sign(xini(1,:));
xini(:,end) = [pi;0];

for trial = 1:Ntrials
   
    tic
    
    clear r;
    clear x;
    clear u;
    clear J;
    clear xn;
    clear xhat;
    
    x = NaN(2,n+1);
    x(:,1) = xini(:,trial);
    x(1) = x(1) + 2*pi * [ abs(x(1))>pi ] * -sign(x(1));
    xn = mapminmax( 'apply', x, pty ); 
    xhat = xn;

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
    end
       
    % Log Book
    Rlog  = [Rlog sum(r)];
    Xlogn = [Xlogn, xn];
    Thetalog  = [Thetalog; x(1,:)];
    ThetaDlog  = [ThetaDlog; x(2,:)];
    tvec(trial) = j;

    % Error
    mseC(trial) = critic.evaluateC(xn(:,1:j+1), gamma, r);
    mseA(trial)  = .5*norm(J)/(length(xn(1:j+1)));
    
    if mseC(trial) < 10^-4 && mseA(trial) < 10^-3
        fprintf('this trial was good with %i theta and %i thetadot \n', x(1,1), x(2,1));
    end
    
    if trial == 150 || trial == 165 || trial == 212
        fprintf('trial 158 reached')
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

%% Plotting

print_plots = false;
saveNet = false;

denorm = mapminmax( 'reverse',[xn(:,1:j); u], ptx );
xhat2 = mapminmax( 'reverse',xhat, pty );

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
if print_plots == true
    print('HDP_trqLimited','-dpng','-r300');
end

% Action and angles of last trial
figure()
subplot(3,1,1)
plot(t(1:length(denorm)),denorm(3,:))
xlabel('time [s]'); ylabel('actions')
xlim([0 tmax]);
grid on
subplot(3,1,2)
plot(t(1:j),x(1,1:j)*r2d)
hold on 
plot(t(1:j),xhat2(1,1:j)*r2d)
hold off
xlabel('time [s]'); ylabel('\theta [deg]')
xlim([0 tmax]);
grid on
subplot(3,1,3)
plot(t(1:j),x(2,1:j))
hold on 
plot(t(1:j),xhat2(2,1:j))
hold off
xlabel('time [s]'); ylabel('\theta_d [deg/s]')
xlim([0 tmax]);
grid on
if print_plots == true
    print('anglesHDP_results_trqLimited','-dpng','-r300');
end


xnorm1 = mapminmax('apply',[-pi;-8*pi;0],ptx);
xnorm2 = mapminmax('apply',[pi;8*pi;0],ptx);
x1plot = linspace( xnorm1(1), xnorm2(1), 30 ); 
x2plot = linspace( xnorm1(2), xnorm2(2), 40 );
[X1,X2] = meshgrid( x1plot, x2plot ); % create rectangular meshgrid
Zcrit = zeros( size(X1) );
Zact = zeros( size(X1) );
for i = 1:length(x1plot)
    for k = 1:length(x2plot)
        Zcrit(k,i) = critic.FFwrd([X1(k,i); X2(k,i)]);
        Zact(k,i) = actor.FFwrd([X1(k,i); X2(k,i)]);
    end
end
figure(); clf

% Critic
subplot(2,6,[1:3 7:9]);
surf(X1,X2,Zcrit)
title('Value Function/Critic')
xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('J');
hold on
y = critic.FFwrd([Xlogn(1,:);Xlogn(2,:)]);
plot3(Xlogn(1,:),Xlogn(2,:),y,'k.');
xlim([-1 1]); ylim([-1 1]);
hold off

% Actor
subplot(2,6,[4:6 10:12])
surf(X1,X2,Zact)
title('Control/Actor')
xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('control');
xlim([-1 1])
hold on
y = actor.FFwrd([Xlogn(1,:);Xlogn(2,:)]);
plot3(Xlogn(1,:),Xlogn(2,:),y,'k.');
xlim([-1 1]); ylim([-1 1]);
hold off

if print_plots == true
    print('ACshape_trqLimited','-dpng','-r300');
end

figure()
contourf(X1,X2,Zcrit);
grid on;
xlabel('\theta'); ylabel('\theta_{dot}')
title('Critic');
colorbar

if print_plots == true
    print('Critic','-dpng','-r300');
end

figure()
contourf(X1,X2,Zact);
grid on;
xlabel('\theta'); ylabel('\theta_{dot}')
title('Actor');
colorbar

if print_plots == true
    print('Actor','-dpng','-r300');
end

if saveNet == true
    save('criticUpswing','critic');
    save('actorUpswing','actor');
end

