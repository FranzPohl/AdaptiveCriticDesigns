
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
        Zact(k,i) =actor.FFwrd([X1(k,i); X2(k,i)]);
    end
end

% SURFPLOT
figure(); clf

% Critic
subplot(2,6,[1:3 7:9]);
surf(X1,X2,Zcrit)
title('Value Function/Critic')
xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('J');
hold on
y = critic.FFwrd([Xlogn(1,:);Xlogn(2,:)]);
plot3(Xlogn(1,:),Xlogn(2,:),y,'k.');
xlim([-1 1]); ylim([-1 1]);
hold off

% Actor
subplot(2,6,[4:6 10:12])
surf(X1,X2,Zact)
title('Control/Actor')
xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('control');
xlim([-1 1])
hold on
y = actor.FFwrd([Xlogn(1,:);Xlogn(2,:)]);
plot3(Xlogn(1,:),Xlogn(2,:),y,'k.');
xlim([-1 1]); ylim([-1 1]);
hold off

% CONTOUR

% Critic
figure()
contourf(X1,X2,Zcrit);
grid on;
xlabel('\theta'); ylabel('\theta_{dot}')
title('Critic');
colorbar

% Actor
figure()
contourf(X1,X2,Zact);
grid on;
xlabel('\theta'); ylabel('\theta_{dot}')
title('Actor');
colorbar