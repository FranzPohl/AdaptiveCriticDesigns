%% PLOTS DHP CONTROL RESULTS
% 1. Reward and MSE
% 2. Response
% 3. Lambda
% 4. Actor Shape

savePlot = false;
saveNets = false;

denorm = mapminmax( 'reverse',[xn(:,1:end-1); u(1:length(xn)-1)], ptx );
xhat2 = mapminmax( 'reverse',xhat, pty );

%% 1. Reward and MSE

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

%% 2. Response 

figure()
subplot(3,1,1)
plot(t(1:length(denorm)),denorm(3,:));
xlabel('time [s]'); ylabel('actions')
grid on
subplot(3,1,2)
plot(t(1:length(x)),x(1,:)*r2d)
xlabel('time [s]'); ylabel('\theta [deg]')
grid 
hold on
plot(t(1:length(xhat2)),xhat2(1,:)*r2d)
hold off

subplot(3,1,3)
plot(t(1:length(x)),x(2,:)*r2d)
xlabel('time [s]'); ylabel('\theta_d [deg/s]')
grid on
hold on
plot(t(1:length(xhat2)),xhat2(2,:)*r2d)
hold off
if savePlot == true;
    print('anglesHDP','-deps','-r300');
end

%% 3. Value derivative

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

%% 4. Actor shape

% 3D Surf Plots Data
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

% 3D Surf Actor
subplot(2,6,[4:6 10:12])
surf(X1,X2,Zact)
title('Control/Actor')
xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('control');
xlim([-1 1])

%% last policy

state = [pi;0];
xnorm = mapminmax('apply',state,pty);
ipgraph = IPGraphics(state(1));

tmax    = 5;
dt      = 0.005;
t       = 0:dt:tmax;
n       = length(t);

for i=1:n-1
    
    policy(i) = actor.FFwrd( xnorm(:,i) );
    denorms = mapminmax('reverse',[xnorm(:,i); policy(i)], ptx);
    
    state(:,i+1)   = Inverted_Pendulum( state(:,i), denorms(3), dt );
    state(1,i+1) = state(1,i+1) + 2*pi*[abs(state(1,i+1))>pi]*-sign(state(1,i+1));
    xnorm(:,i+1) = mapminmax( 'apply', state(:,i+1), pty); 
    ipgraph.update(state(1,i+1));
    
end
    