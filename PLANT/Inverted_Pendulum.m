% Dynamcis of a simple inverted pendulum
% xp = PlantDynamics(x,u)
% x  - [rad, rad/s] states, i.e. angle and angular rate
% u  - [Nm] control input 
% xp - [rad, rad/s] new state
function xp = Inverted_Pendulum(x,f,dt)
    
    % Integrate
    [t, xp] = ode23(@fun,[0 dt],x,[],f); 
    xp = xp(end,:)';
end


function dxdt = fun(t,x,f)

% System parameters
    m = 0.1;       % [Kg] Mass of pendulum  0.2   
    l = 0.1;       % [m] length to pendulum center of mass 0.5
    I = m*l^2;     % [kg.m^2] inertia 
    ga = 9.81;     % [m/s^2] gravitational accelleration

    % System dynamics 
    dxdt = [x(2,:);                           % theta-dot                         
            m.*ga.*l/I.*sin(x(1,:)) + 1/I*f]; % theta-double dot         


    
end % FUNCTION ENDS

% function X = Inverted_Pendulum(x,u,dt)  
% %PENDULUMDYNAMICS dynamics of a simple pendulum
% % X = pendulumDynamics(x,u,m,l)           
% 
% % parameters
% m = 0.1;
% l = 0.1;
% J = m*l^2; % intertia
% g = 9.81;  % gravity
% 
% % dynamics
% xd1 = x(2); % angular rate
% xd2 = (1/J) * ( m*g*l*sin(x(1)) - 1e-4*x(2) + u ); % angular acceleration
% 
% % save output
% xdot = [xd1; xd2] ; 
% 
% % Euler integration
% X = x + xdot.*dt;
% 
% end