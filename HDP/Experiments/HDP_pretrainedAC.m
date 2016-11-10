%% HDP Algortihm
clc; clear; close all;

% Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 
addpath(fullfile(mfilepath,'../PLANT')); 
addpath(fullfile(mfilepath,'../MODEL')); 

% Load pre-trained model, critic and actor
load('model.mat')
load('critic.mat')
load('actor.mat')

% Simulation parameters
tmax    = 5;
dt      = 0.01;
t       = 0:dt:tmax;
n       = length(t);
r2d     =  180/pi;

% Record movie_clip parameter
M(n) = struct('cdata',[],'colormap',[]); v = VideoWriter('HDP.avi');
lifestream = false;

%% RL Parameters

gamma   = 0.95;   % discount rate                                                                          
etaC    = 0.100;  % learning rate of critic ANN  
tauC    = 0.01;   % time-step updates critic
muC     = 0.10;   % momentum factor critic

% Actor RL parameters
etaA   = 0.10;
tauA   = 0.01;
muA    = 0.00;

%% STEP III HDP CRITIC AND ACTOR CONNECTED

% Limitations
xl = .95*pi;
xdl= 30;
Xlog = [];
Rlog = [];

Ntrials = 15;
for trial = 1:Ntrials
%      open(v);
    
    clear r;
    clear x;
    clear u;
    clear J;
    clear xn;
    
    x = [randn(1)*0.6; randn(1)]; % xn0
    xn= mapminmax('apply',x,pty); 
    
    for j = 1:n-1
        
        
         u(j) = actor.FFwrd(xn(:,j));
        denorm = mapminmax('reverse',[xn(:,j);u(j)], ptx);
        
        x(:,j+1) = Inverted_Pendulum(x(:,j),denorm(3), dt);
        x(1,j+1) = x(1,j+1) + 2*pi*[abs(x(1,j+1))>pi]*-sign(x(1,j+1));
        xn(:,j+1)= mapminmax('apply',x(:,j+1),pty);
        
        r(j) = reward(xn(:,j+1));
        J(j) = critic.FFwrd(xn(:,j+1));

        dJdx = critic.net_derivative(xn(:,j+1));
        dxdu = model.net_derivative([xn(:,j);u(j)],dJdx);
        
        critic.updateC_HDP( xn(:,j:j+1), r(j), etaC, muC, gamma );
        actor.updateA_HDP( xn(:,j), J(j), dxdu(3), etaA, muA);
        
        if abs(x(1,j+1)) > xl || abs(x(2,j+1)) > xdl
            break;
        end
        
    end
    
    Rlog = [Rlog sum(r)];
    Xlog = [Xlog, xn];
    Jt = critic.FFwrd(xn(:,1:end-1));
    Jtp1 = critic.FFwrd(xn(:,2:end));
    TD = Jt - (gamma*Jtp1 + r);
    mseC(trial) = .5*norm(TD)/(length(xn));
    mseA(trial)  = .5*norm(J)/(length(xn));
    fprintf('Trial %i/%i: TD = %i    Actor Error = %i\n', trial, Ntrials, mseC(trial), mseA(trial))
    
    xnorm1 = mapminmax('apply',[-pi;-6*pi;0],ptx);
    xnorm2 = mapminmax('apply',[pi;6*pi;0],ptx);
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
    figure(1); clf
    subplot(2,6,[1:3 7:9]);
    surf(X1,X2,Zcrit)
    title('Value Function/Critic')
    xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('J');
    subplot(2,6,[4:6 10:12])
    surf(X1,X2,Zact)
    title('Control/Actor')
    xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('control');

    
    % Plotting graphs
    if lifestream == true;
        M(i) = getframe(gcf);
        writeVideo(v,M(i));
    end
    
end
% close(v);

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
print('HDP_results','-dpng','-r300');

figure()
subplot(3,1,1)
plot(t(1:end-1),u)
xlabel('time [s]'); ylabel('actions')
grid on
subplot(3,1,2)
plot(t,x(1,:)*r2d)
xlabel('time [s]'); ylabel('\theta [deg]')
grid on
subplot(3,1,3)
plot(t,x(2,:)*r2d)
xlabel('time [s]'); ylabel('\theta_d [deg/s]')
grid on
% print('anglesHDP','-dpng','-r300');


% Critic
xnorm1 = mapminmax('apply',[-pi;-6*pi;0],ptx);
xnorm2 = mapminmax('apply',[pi;6*pi;0],ptx);
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
subplot(2,6,[1:3 7:9]);
surf(X1,X2,Zcrit)
title('Value Function/Critic')
xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('J');
hold on
y = critic.FFwrd([Xlog(1,:);Xlog(2,:)]);
plot3(Xlog(1,:),Xlog(2,:),y,'k.');
hold off
subplot(2,6,[4:6 10:12])
surf(X1,X2,Zact)
title('Control/Actor')
xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('control');
%print('Results','-dpng','-r300')

figure()
contour(X2,X1,Zcrit);
xlabel('\theta_{dot}'); ylabel('\theta')

% save('critic2','critic');
% save('actor2','actor');