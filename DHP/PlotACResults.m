% Reward and MSE
figure()
subplot(3,1,1)
plot(length(Rlog), Rlog)
xlabel('epochs'); ylabel('reward[-]');
subplot(3,1,2)
plot(mseA)
xlabel('epochs'); ylabel('mse Actor')
subplot(3,1,3)
plot(mseC)
xlabel('epochs'); ylabel('mse Critic [-]');
% print('HDP_results','-dpng','-r300');

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
% print('anglesHDP','-dpng','-r300');

% save('critic2','critic');
% save('actor2','actor');