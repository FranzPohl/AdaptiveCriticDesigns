%% Testing HDP controller
clc; clear; close all;

% Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../ANN')); 
addpath(fullfile(mfilepath,'../../PLANT')); 
addpath(fullfile(mfilepath,'../../MODEL'));

% Load Networks
load('model_trqLimited.mat')
load('actorUpswing.mat')

% Simulation parameters
tmax    = 1.4;
dt      = 0.005;
t       = 0:dt:tmax;
n       = length(t);
r2d     =  180/pi;

% Initial conditions
% x = [randn(1)*0.6; randn(1)]; % xn0
x =[pi;0];
xn= mapminmax('apply',x,pty); 

 for i = 1:n-1
        
        
         u(i) = actor.FFwrd(xn(:,i));
         
         % give disturbing impulse
%          if mod(i,200) == 0
%              u(i) = randn(1)*15;
%          end
         
        denorm = mapminmax('reverse',[xn(:,i);u(i)], ptx);
        
        x(:,i+1) = Inverted_Pendulum(x(:,i),denorm(3), dt);
        x(1,i+1) = x(1,i+1) + 2*pi*[abs(x(1,i+1))>pi]*-sign(x(1,i+1));
        xn(:,i+1)= mapminmax('apply',x(:,i+1),pty);
        
 end
 
%% Plots 

denorm = mapminmax('reverse',[xn(:,1:end-1);u], ptx);

subplot(3,1,1)
plot(t,x(1,:)*r2d)
xlabel('time [s]')
ylabel('\theta [deg]')
hold on
% plot(t,ones(size(t))*180,'r-.')
% plot(t,ones(size(t))*-180,'r-.')
xbar=[2 2 nan 4 4 nan 6 6 nan 8 8 nan 10 10 nan 12 12 nan 14 14].';
ybar=[repmat([-180 180 nan].',6,1); -180; 180];
plot(xbar,ybar,'k:')
xlim([0 tmax])
ylim([-180 180])
legend('\theta control')%,'Disturbance')
hold off

subplot(3,1,2)
plot(t,x(2,:)*r2d)
xlabel('time [s]')
ylabel('\theta_d [deg/s]')
hold on
xbar=[2 2 nan 4 4 nan 6 6 nan 8 8 nan 10 10 nan 12 12 nan 14 14].';
ybar=[repmat([-3000 3000 nan].',6,1); -3000; 3000];
plot(xbar,ybar,'k:')
xlim([0 tmax])
ylim([-3000 3000])
legend('\theta_d control')%,'Disturbance')
hold off

subplot(3,1,3)
plot(t(1:end-1),denorm(3,:))
ylim([-.05 .05]);
xlabel('time [s]');
ylabel('torque [Nm]');

print('performanceHDP','-dpng','-r300')