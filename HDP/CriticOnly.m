%% HDP Algortihm
clc; clear; close all;

% Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 
addpath(fullfile(mfilepath,'../PLANT')); 
addpath(fullfile(mfilepath,'../MODEL')); 

% Load trained model from model library
load('model.mat')

% Simulation parameters
tmax    = 2;
dt      = 0.01;
t       = 0:dt:tmax;
n       = length(t);

%% Neural Networks

% Critic
numInC     = 2;
numNeuronC = 8;
numOutC    = 1;
critic = NeuralNet([numInC, numNeuronC, numOutC]);

% Critic RL parameters
gamma   = 0.95;   % discount rate                                                                          
etaC    = 0.100;  % learning rate of critic ANN  
tauC    = 0.01;   % time-step updates critic
muC     = 0.10;   % momentum factor critic
batchC  = tauC/dt; % batch size critic


%% STEP I CRITIC TRAINING

% Limitations
ul = [-10 10];
xl = .95*pi;
xdl= 30;
Xlog = [];
Rlog = [];

Ntrials = 100;
for trial = 1:Ntrials
    
    clear r;
    clear x;
    clear xn;
    clear u;
    
    x = zeros(2,1);%[randn(1)*0.6; randn(1)*1.5]; % xn0
    xn= mapminmax('apply',x,pty); 
   
    u = rand(1,n-1) * (max(ul)-min(ul)) + min(ul);
    
    for j = 1:n-1
        
        x(:,j+1) = Inverted_Pendulum(x(:,j),u(j), dt);
        x(1,j+1) = x(1,j+1) + 2*pi*[abs(x(1,j+1))>pi]*-sign(x(1,j+1));
        
        xn(:,j+1)= mapminmax('apply',x(:,j+1),pty);
        r(j) = reward(xn(:,j+1)); 

        
        if mod(j,batchC)==0 %What about last data!! Change for that
            critic.updateC_HDP( xn(:,j-batchC+1:j+1), r(j-batchC+1:j), etaC, muC, gamma );
        end
        
        if abs(x(1,j+1)) > xl || abs(x(2,j+1)) > xdl
            break;
        end
        
    end
    
    Rlog = [Rlog r];
    Xlog = [Xlog, xn];
    Jt = critic.FFwrd(xn(:,1:end-1));
    Jtp1 = critic.FFwrd(xn(:,2:end));
    TD = Jt - (gamma*Jtp1 + r);
    mse(trial) = .5*norm(TD)/(length(xn));
    fprintf('Trial %i/%i: Critic Error = %i\n', trial, Ntrials, mse(trial))
  
end


%% Plotting

figure()
plot(mse)
xlabel('epochs');
ylabel('mse [-]');
% print('criticMSE','-dpng','-r300');

% Critic
xnorm1 = mapminmax('apply',[-pi;-6*pi;0],ptx);
xnorm2 = mapminmax('apply',[pi;6*pi;0],ptx);
x1plot = linspace( xnorm1(1), xnorm2(1), 30 ); 
x2plot = linspace( xnorm1(2), xnorm2(2), 40 );

[X1,X2] = meshgrid( x1plot, x2plot ); % create rectangular meshgrid
Zcrit = zeros( size(X1) );

for i = 1:length(x1plot)
    for k = 1:length(x2plot)
        Zcrit(k,i) = critic.FFwrd([X1(k,i); X2(k,i)]);
    end
end
figure(); clf
surf(X1,X2,Zcrit)
title('Value Function/Critic')
xlabel('\theta'); ylabel('\theta_{dot}')
hold on
y = critic.FFwrd([Xlog(1,:);Xlog(2,:)]);
plot3(Xlog(1,:),Xlog(2,:),y,'k.');
hold off
% print('critic','-dpng','-r300');

figure()
contour(X2,X1,Zcrit);
xlabel('\theta_{dot}'); ylabel('\theta')

%save('critic','critic');

%% FOR VIDEO 

% Record movie_clip parameter
% M(n) = struct('cdata',[],'colormap',[]); v = VideoWriter('critic.avi');
% lifestream = false;

% open(v);

%     if lifestream == true;
%         M(i) = getframe(gcf);
%         writeVideo(v,M(i));
%     end

% close(v);
