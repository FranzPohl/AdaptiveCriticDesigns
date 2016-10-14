% PLOTS ADAPTIVE CRITIC RESULTS
% 1. ANN Weights
% 2. Reward and MSE
% 3. Policy
% 4. Actor Critic Approximation

print_plots = false;
saveNet = false;

denorm = mapminmax( 'reverse',[xn(:,1:end-1); u(1:length(xn)-1)], ptx );
xhat2 = mapminmax( 'reverse',xhat, pty );

%% 1. ANN Weights

%Actorweights
figure()
subplot(3,1,1)
plot(1:length(ActorWeights(1,:)), ActorWeights(1:numNeuronA,:))
subplot(3,1,2)
plot(1:length(ActorWeights(1,:)), ActorWeights(numNeuronA+1:2*numNeuronA,:));
subplot(3,1,3)
plot(1:length(ActorWeights(1,:)), ActorWeights(2*numNeuronA+1:3*numNeuronA,:))

%Actor Bias
figure()
subplot(2,1,1)
plot(1:length(ActorBias(1,:)), ActorBias(1:numNeuronA,:))
subplot(2,1,2)
plot(1:length(ActorBias(1,:)), ActorBias(numNeuronA+1,:))

%Critic Weights
figure()
subplot(3,1,1)
plot(1:length(CriticWeights(1,:)), CriticWeights(1:numNeuronC,:))
subplot(3,1,2)
plot(1:length(CriticWeights(1,:)), CriticWeights(numNeuronC+1:2*numNeuronC,:));
subplot(3,1,3)
plot(1:length(CriticWeights(1,:)), CriticWeights(2*numNeuronC+1:3*numNeuronC,:));

%Critic Bias
figure()
subplot(2,1,1)
plot(1:length(CriticBias(1,:)), CriticBias(1:numNeuronA,:))
subplot(2,1,2)
plot(1:length(CriticBias(1,:)), CriticBias(numNeuronA+1,:))

%% 2. Reward and MSE

% Reward and MSE
figure()
subplot(3,1,1)
plot(Rlog)
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

%% 3. Policy

% Action and angles of last trial
figure()
subplot(3,1,1)
plot(t(1:length(denorm)),denorm(3,:))
xlabel('time [s]'); ylabel('actions')
xlim([0 tmax]);
grid on
subplot(3,1,2)
plot(t(1:length(x)),x(1,:)*r2d)
hold on 
plot(t(1:length(x)),xhat2(1,:)*r2d)
hold off
xlabel('time [s]'); ylabel('\theta [deg]')
xlim([0 tmax]);
grid on
subplot(3,1,3)
plot(t(1:length(x)),x(2,:))
hold on 
plot(t(1:length(x)),xhat2(2,:))
hold off
xlabel('time [s]'); ylabel('\theta_d [deg/s]')
xlim([0 tmax]);
grid on
if print_plots == true
    print('anglesHDP_results_trqLimited','-dpng','-r300');
end

%% 4. Actor & Critic Approximation

% 3D Surf Plots Data
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

% 3D Surf Critic
subplot(2,6,[1:3 7:9]);
surf(X1,X2,Zcrit)
title('Value Function/Critic')
xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('J');
hold on
y = critic.FFwrd([Xlog(1,:);Xlog(2,:)]);
plot3(Xlog(1,:),Xlog(2,:),y,'k.');
xlim([-1 1]); ylim([-1 1]);
hold off

% 3D Surf Actor
subplot(2,6,[4:6 10:12])
surf(X1,X2,Zact)
title('Control/Actor')
xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('control');
xlim([-1 1])
hold on
y = actor.FFwrd([Xlog(1,:);Xlog(2,:)]);
plot3(Xlog(1,:),Xlog(2,:),y,'k.');
xlim([-1 1]); ylim([-1 1]);
hold off
if print_plots == true
    print('ACshape_trqLimited','-dpng','-r300');
end

% Contour PLot Critic
figure()
contourf(X1,X2,Zcrit);
grid on;
xlabel('\theta'); ylabel('\theta_{dot}')
title('Critic');
colorbar
if print_plots == true
    print('Critic','-dpng','-r300');
end

% Contour Plot Actor
figure()
contourf(X1,X2,Zact);
grid on;
xlabel('\theta'); ylabel('\theta_{dot}')
title('Actor');
colorbar
hold on
if print_plots == true
    print('Actor','-dpng','-r300');
end

% Save Neural Net
if saveNet == true
    save('criticUpswing','critic');
    save('actorUpswing','actor');
end