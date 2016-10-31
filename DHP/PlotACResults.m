%% Plot Results of DHP Algorithm

savePlot = true;
saveNets = false;

% Reward and MSE
figure()
subplot(3,1,1)
plot(1:length(Rlog), Rlog)
xlabel('epochs'); ylabel('reward[-]');
subplot(3,1,2)
plot(mseA)
xlabel('epochs'); ylabel('mse Actor')
subplot(3,1,3)
plot(mseC)
xlabel('epochs'); ylabel('mse Critic [-]');
if savePlot ==true
    print('HDP_results','-deps','-r300');
end

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
if savePlot == true;
    print('anglesHDP','-deps','-r300');
end

figure();
plot(lambda(1,:)); 
hold on; 
xlabel('timesteps');
xlim([0,tmax/dt]);
plot(lambda(2,:));
legend('dJ/d\theta','dJ/d\theta_d','Location','SouthEast');
if savePlot == true
    print('dJdx','-deps','-r300');
end


if saveNets == true
   save('criticDHP','critic');
   save('actorDHP','actor');
end