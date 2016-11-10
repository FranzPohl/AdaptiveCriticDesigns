% Video Log (Trial Level)
if livestream == true

    if mod(trial,60) == 0 || trial == 1;

        xval = 0.3*sin(x(1,j));
        yval = 0.3*cos(x(1,j));
        
        % create rectangular meshgrid for plot
        xnorm1 = mapminmax('apply',[-pi;-8*pi;0],ptx);
        xnorm2 = mapminmax('apply',[pi;8*pi;0],ptx);
        x1plot = linspace( -1, 1, 30 ); 
        x2plot = linspace( -1.2, 1.2, 40 );
        [X1,X2] = meshgrid( x1plot, x2plot ); 

        % Compute Actor and Critic Output for mesh
        for z = 1:length(x1plot)
            
            for s = 1:length(x2plot)
                Zact (s,z) = actor.FFwrd([X1(s,z); X2(s,z)]);
                Zcrit(s,z) = critic.FFwrd([X1(s,z);X2(s,z)]);
            end
            
        end

        % plot actor shape
        figure(1)
        subplot(4,5,[4 5 9 10])
        surf(X1,X2,Zact)
        title('Actor')
        xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('torque [Nm]');
        hold on
        y = actor.FFwrd([xn(1,:);xn(2,:)]);
        plot3(xn(1,:), xn(2,:), y, 'k.');
        xlim([-1 1]); ylim([-1 1]);
        hold off
        subplot(4,5,[11 12 16 17])
        contourf(X2,X1,Zact,'ShowText','on'); 
        xlabel('\theta_{dot}'); ylabel('\theta')
        
        % plot Critic Shape
        subplot(4,5,[14 15 19 20])
        surf(X1,X2,Zcrit)
        title('Critic')
        xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('torque [Nm]');
        subplot(4,5,[11 12 16 17])
        contourf(X2,X1,Zcrit,'ShowText','on'); 
        xlabel('\theta_{dot}'); ylabel('\theta')
        

        % plot pendulum
        subplot(4,5,[1 2 6 7])
        plot([0],[0],'ko','MarkerSize',2);          % pivot point
        hold on
        plot([0 xval],[0 yval],'r','LineWidth',2);  % link
        plot(xval,yval,'ko','MarkerSize',5);        % mass
        plot([0 0],[0 0.3],'b--')
        str = ['\theta = ',num2str(round(x(1,j)*180/pi,2)),' [deg]'];
        text(0.2,-0.1,str,'HorizontalAlignment','right');
        axis([-1.2*.3 1.2*.3 -1.2*.3 1.2*.3])
        axis('square')
        hold off

        M(j) = getframe(gcf);
        writeVideo(v,M(j))
    
    end

end

