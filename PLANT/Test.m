%% TEST_PLANT
% Testing the plant dynamics and integration (ODE1)
close all; clear; clc;

%% Simulation parameters
tmax = 5;
int  = false;
dt   = 0.01;
t    = 0:dt:tmax;
n    = length(t);

% choice of Dynamic System: 1. Inverted Pendulum 2. Cart-Pole
choice = 2;

%% Simulate

if choice == 1 % inverted Pendulum Test
    % loading graphics
    invertedPendulum = IPGraphics(0);

    %initialize x
    x = [pi; 0];
    ur = [-.1 .1];
    %u = ones(1,n)*.7;
    u = sin(t*6)*0.05;
    %u = rand(1,n) * (max(ur)-min(ur)) + min(ur);
    %u = random_walk(round(randn(1)),n-1);
    invertedPendulum.update(x(1,1));    
    
    for i = 1:n
        
        x(:,i+1) = Inverted_Pendulum(x(:,i),u(i),dt);
        invertedPendulum.update(x(1,i+1));
        if abs(x(2,i+1)) > 35
            break;
        end

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
%         u = idinput(n,'prbs',[0 .8],[-10 10]);
        u = rand(1,n) * 20 - 10;

        cart_pole.update(x(1,1),x(3,1));

        for i = 1:n

%             x(:,j+1) = Cart_Pole(x(:,j),u(j));
            x(:,i+1) = Cart_Pole(x(:,i),u(i),dt);
            
            if mod(i,1)==0
                cart_pole.update(x(1,i+1),x(3,i+1));
            end

            if abs(x(1,i+1)) > 1.2 || abs(x(3,i+1)) > .5*pi
                break;
            end

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

    
    

    


    