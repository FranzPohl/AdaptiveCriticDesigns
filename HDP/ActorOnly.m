%% HDP Algortihm
clc; clear; close all;

% Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 
addpath(fullfile(mfilepath,'../PLANT')); 
addpath(fullfile(mfilepath,'../MODEL')); 

% Load trained model from model library
load('model.mat')
load('critic.mat')

% Simulation parameters
tmax    = 4;
dt      = 0.01;
t       = 0:dt:tmax;
n       = length(t);
r2d     =  180/pi;

%% Neural Networks

% Actor 
numInA     = 2;
numNeuronA = 8;
numOutA    = 1;
actor = NeuralNet([numInA, numNeuronA, numOutA]);
actor.transferFun{end} = sigmoid;
 
% Actor RL parameters
etaA   = 0.10;
tauA   = 0.010;
muA    = 0.00;
batchA = tauA/dt;

%% STEP II ACTOR TRAINING

% Limitations
ul = [-10 10];
xl = .95*pi;
xdl= 30;
Xlog = [];
Rlog = [];

Ntrials = 50;
for trial = 1:Ntrials
    
    clear r;
    clear x;
    clear u;
    clear xn;
    clear J;
    
    x = [randn(1)*0.3; randn(1)]; % xn0
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
        
        actor.updateA_HDP( xn(:,j), J(j), dxdu(3), etaA, muA);

        if abs(x(1,j+1)) > xl || abs(x(2,j+1)) > xdl
            break;
        end
        
    end
    
    Rlog = [Rlog sum(r)];
    Xlog = [Xlog, xn];

    mse(trial) = .5*norm(J)/(length(x));
    fprintf('Trial %i/%i: Actor Error = %i\n', trial, Ntrials, mse(trial))
       
end

%% Plotting

% Reward and MSE
figure()
subplot(2,1,1)
plot(1:Ntrials, Rlog)
xlabel('epochs'); ylabel('reward[-]');
subplot(2,1,2)
plot(mse)
xlabel('epochs'); ylabel('mse')
% print('actor_results','-dpng','-r300');

subplot(3,1,1)
plot(u)
xlabel('time [s]'); ylabel('actions')
grid on
subplot(3,1,2)
plot(x(1,:)*r2d)
xlabel('time [s]'); ylabel('\theta [deg]')
grid on
subplot(3,1,3)
plot(x(2,:)*r2d)
xlabel('time [s]'); ylabel('\theta_d [deg/s]')
grid on
% print('angles','-dpng','-r300');

% Actor
xnorm1 = mapminmax('apply',[-pi;-8*pi;0],ptx);
xnorm2 = mapminmax('apply',[pi;8*pi;0],ptx);
x1plot = linspace( xnorm1(1), xnorm2(1), 30 ); 
x2plot = linspace( xnorm1(2), xnorm2(2), 40 );

[X1,X2] = meshgrid( x1plot, x2plot ); % create rectangular meshgrid
Zact = zeros( size(X1) );

for i = 1:length(x1plot)
    for k = 1:length(x2plot)
        Zact(k,i) = actor.FFwrd([X1(k,i); X2(k,i)]);
    end
end
figure(); clf
surf(X1,X2,Zact)
title('Controller/Actor')
xlabel('\theta'); ylabel('\theta_{dot}')
hold on
y = actor.FFwrd([Xlog(1,:);Xlog(2,:)]);
plot3(Xlog(1,:),Xlog(2,:),y,'k.');
hold off
% print('actor','-dpng','-r300')

figure()
contour(X2,X1,Zact);
xlabel('\theta_{dot}'); ylabel('\theta')

% save('actor','actor');


%% VIDEO

% % Record movie_clip parameter
% M(n) = struct('cdata',[],'colormap',[]); v = VideoWriter('actor.avi');
% lifestream = false;
% open(v);
%    if lifestream == true;
%         M(i) = getframe(gcf);
%         writeVideo(v,M(i));
%    end
% close(v);

