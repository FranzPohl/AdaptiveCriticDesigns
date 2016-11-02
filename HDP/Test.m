%% HDP

clc; clear; close all;

% Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 
addpath(fullfile(mfilepath,'../PLANT')); 
addpath(fullfile(mfilepath,'../MODEL'));
load('actorSimplePendulum.mat')
load('model.mat')

% Start Simulation
tmax = 11;
dt   = 0.005;
t = 0:dt:tmax;
n = length(t);

x = [2.4; 0];
xn= mapminmax('apply',x, pty);

for i = 1:n-1
    
    if mod(i,600) == 0
        x(:,i) = [randn(1);0];
        x(1,i) = x(1,i) + 2*pi*[abs(x(1,i))>pi]*-sign(x(1,i));
        xn(:,i)= mapminmax('apply',x(:,i), pty);
    end
    
    u(i) = actor.FFwrd( xn(:,i) );
    denorm = mapminmax('reverse',[xn(:,i);u(i)], ptx);
    
    x(:,i+1) = Inverted_Pendulum( x(:,i),u(i),dt );
    x(1,i+1) = x(1,i+1) + 2*pi*[abs(x(1,i+1))>pi]*-sign(x(1,i+1));
    xn(:,i+1)= mapminmax( 'apply', x(:,i+1), pty );
    
end

%% plot

savePlot = true;
r2d = 180/pi;

figure()
subplot(3,1,1)
plot(t(1:length(u)),u)
xlabel('time [s]'); ylabel('actions')
xlim([0 tmax]);
grid on
subplot(3,1,2)
plot(t(1:length(x)),x(1,:)*r2d)
xlabel('time [s]'); ylabel('\theta [deg]')
xlim([0 tmax]);
grid on
subplot(3,1,3)
plot(t(1:length(x)),x(2,:)*r2d)
xlabel('time [s]'); ylabel('\theta_d [deg/s]')
xlim([0 tmax]);
grid on

if savePlot == true;
    print('HDPResults','-deps','-r300');
end