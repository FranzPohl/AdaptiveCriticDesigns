
close all; clear; clc;

mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 
addpath(fullfile(mfilepath,'../PLANT')); 
addpath(fullfile(mfilepath,'../MODEL'));

load('model.mat')


tmax = 6;
dt   = 0.001;
t = 0:dt:tmax;
n = length(t);
xdistort = [1.4; -.7];

%% PID 
% 
% for i = 1:length(xdistort);
%    
%    simOut = sim('PIDInvP',...
%            'StopTime', '3');
% end
% 
% theta = simOut.get('theta');
% thetadot = simOut.get('thetadot');
% 
% theta = [theta.Data(:,1); theta.Data(:,2)];
% thetad= [thetadot.Data(:,1); thetadot.Data(:,2)];

%% DHP

load('Exp1DHP.mat')

xdhp = [xdistort(1); 0];
xn= mapminmax('apply',xdhp, pty);

for i = 1:n-1
    
    if mod(i,3000) == 0
        xdhp(:,i) = [xdistort(2);0];
        xdhp(1,i) = xdhp(1,i) + 2*pi*[abs(xdhp(1,i))>pi]*-sign(xdhp(1,i));
        xn(:,i)= mapminmax('apply',xdhp(:,i), pty);
    end
    
    udhp(i) = actor.FFwrd( xn(:,i) );
    denorm = mapminmax('reverse',[xn(:,i);udhp(i)], ptx);
    
    xdhp(:,i+1) = Inverted_Pendulum( xdhp(:,i),udhp(i),dt );
    xdhp(1,i+1) = xdhp(1,i+1) + 2*pi*[abs(xdhp(1,i+1))>pi]*-sign(xdhp(1,i+1));
    xn(:,i+1)= mapminmax( 'apply', xdhp(:,i+1), pty );
    
end


%% HDP

load('Exp1HDP.mat')

xhdp = [xdistort(1); 0];
xn= mapminmax('apply',xhdp, pty);

for i = 1:n-1
    
    if mod(i,3000) == 0
        xhdp(:,i) = [xdistort(2);0];
        xhdp(1,i) = xhdp(1,i) + 2*pi*[abs(xhdp(1,i))>pi]*-sign(xhdp(1,i));
        xn(:,i)= mapminmax('apply',xhdp(:,i), pty);
    end
    
    uhdp(i) = actor.FFwrd( xn(:,i) );
    denorm = mapminmax('reverse',[xn(:,i);uhdp(i)], ptx);
    
    xhdp(:,i+1) = Inverted_Pendulum( xhdp(:,i),uhdp(i),dt );
    xhdp(1,i+1) = xhdp(1,i+1) + 2*pi*[abs(xhdp(1,i+1))>pi]*-sign(xhdp(1,i+1));
    xn(:,i+1)= mapminmax( 'apply', xhdp(:,i+1), pty );
    
end

%% plot
figure()
r2d = 180/pi;

subplot(2,1,1)

plot(t(1:length(xdhp)),xdhp(1,:)*r2d,'m-..','LineWidth',1)
xlabel('time [s]'); ylabel('\theta [deg]')
xlim([0 tmax]);
hold on
plot(t(1:length(xhdp)),xhdp(1,:)*r2d,'b-.','LineWidth',1)
% plot(t(1:length(xhdp)),theta(1:end-1),'k-.','LineWidth',1)
hold off
% legend('DHP','HDP','PID','Location','SouthEast')
legend('DHP','HDP','Location','SouthEast')

subplot(2,1,2)

plot(t(1:length(xdhp)),xdhp(2,:)*r2d,'m-..','LineWidth',1)
xlabel('time [s]'); ylabel('\theta_d [deg/s]')
xlim([0 tmax]);
hold on
plot(t(1:length(xhdp)),xhdp(2,:)*r2d,'b-.','LineWidth',1)
% plot(t(1:length(xhdp)), thetad(1:end-1),'k-.','LineWidth',1)
hold off
% legend('DHP','HDP','PID','Location','SouthEast')
legend('DHP','HDP','Location','SouthEast')


print('Performance','-dpng','-r300');



