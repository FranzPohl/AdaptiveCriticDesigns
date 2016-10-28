%% TEST_PLANT
% Testing the plant dynamics and integration (ODE1)
close all; clear; clc;

%% Simulation parameters
tmax = 2;
dt   = 0.01;
t    = 0:dt:tmax;
n    = length(t);

% choice of Dynamic System: 1. Inverted Pendulum 2. Cart-Pole
choice = 1;

%% Simulate

if choice == 1 % inverted Pendulum Test
    
    % loading graphics
    invertedPendulum = IPGraphics(0);

    %initialize x
    x = [pi; 0];
    u = sign(randn(1))*sin(10*t)*.1;
    invertedPendulum.update(x(1,1));    
    
    for i = 1:n
        
        x(:,i+1) = Inverted_Pendulum(x(:,i),u(i),dt);
        x(1,i+1) = x(1,i+1) + 2*pi * [abs(x(1,i+1))>pi] * -sign(x(1,i+1));
        invertedPendulum.update(x(1,i+1));
%         if abs(x(2,i+1)) > 35
%             break;
%         end

    end 
    figure()
    plot(u);
    ylabel('u');
    figure()
    plot(x(1,:));
    ylabel('\theta');
    figure()
    plot(x(2,:))
    ylabel('\theta_{dot}');
  
else   % Cart_pole Test
    for trial = 1:10
        % loading graphics
        cart_pole = CPGraphics(0,0);

        %initialize x
        x = [0; 0; 0; 0];
        s = rand(1);
        u = idinput(n,'prbs',[0 .8],[-10 10]);
        if mod(trial,2)==0
            u(1:6) = 10;
        end
        cart_pole.update(x(1,1),x(3,1));

        for i = 1:n

%             x(:,j+1) = Cart_Pole(x(:,j),u(j));
            x(:,i+1) = Cart_Pole(x(:,i),u(i),dt);
            cart_pole.update(x(1,i+1),x(3,i+1));

%             if abs(x(1,j+1)) > 1.2 || abs(x(3,j+1)) > .5*pi
%                 break;
%             end

        end
    end
    figure(2)
    plot(x(1,:));
    figure(3)
    plot(x(2,:));
    grid on
    figure(4)
    plot(x(3,:));
    figure(5)
    plot(x(4,:));
    figure(6)
    plot(u);
end

    
    

    


    