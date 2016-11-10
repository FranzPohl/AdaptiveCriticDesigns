%% Plot Results of DHP Algorithm

savePlot = false;
saveNets = false;
xhat2 = mapminmax( 'reverse',xhat, pty );

%% Reward and MSE
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
    print('DHPMSE','-deps','-r300');
end

%% Plot Response

figure()
subplot(3,1,1)
plot(t(1:length(u)),u)
xlabel('time [s]'); ylabel('actions')
grid on

subplot(3,1,2)
plot(t(1:length(x)),x(1,:)*r2d)
hold on 
plot(t(1:length(x)),xhat2(1,:)*r2d)
hold off
xlabel('time [s]'); ylabel('\theta [deg]')
grid on
legend('Plant response','Model response','Location','SouthEast');

subplot(3,1,3)
plot(t(1:length(x)),x(2,:)*r2d)
xlabel('time [s]'); ylabel('\theta_d [deg/s]')
grid on
hold on 
plot(t(1:length(x)), xhat2(2,:)*r2d)
hold off
legend('Plant response','Model response','Location','SouthEast');
if savePlot == true;
    print('ResponseDHP','-deps','-r300');
end

figure();
plot(lambda(1,:)); 
hold on; 
xlabel('timesteps');
xlim([0,tend/dt]);
plot(lambda(2,:));
legend('dJ/d\theta','dJ/d\theta_d','Location','SouthEast');
if savePlot == true
    print('dJdx','-deps','-r300');
end


if saveNets == true
   save('criticDHP','critic');
   save('actorDHP','actor');
end