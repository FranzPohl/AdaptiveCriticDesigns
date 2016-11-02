% Non-Linear dynamcis of a Cart-Pole system
% xp = Cart_Pole(x,u)
% states - state vector containing:
%          x  - horizontal position [m]
%          xp - velocity of cart [m/s]
%      	   th - angle of pendulum (0 = upright) [rad]
%          thd- angular velocity of pendulum [rad/s]
% u  - [N] force to push cart to left or right 
% xp - [rad, rad/s] new state
function xp = Cart_Pole(x,f,dt)
        
      [t, xp] = ode45(@fun,[0 dt],x,[],f); 
      xp = xp(end,:)';
end

function dxdt = fun(t,x,f)

% System Parameters
    mc = 1.0;       % [Kg] Mass of the cart 
    mp = 0.3;       % [Kg] Mass of pendulum 
    M  = mc+mp;     % Total mass
    l  = 1.0;       % [m] length to pendulum center of mass 
    g  = 9.81;      % [m/s^^2] gravitational accelleration
    ffric = 0.1;    % [N] Friction force

               
   dxdt = [x(2,:);
          (f + mp*l.*sin(x(3,:)).*x(4,:).^2 - mp*g.*cos(x(3,:)).*sin(x(3,:)) + ffric*mp.*x(4,:).*cos(x(3,:)))./(M - mp.*cos(x(3,:)).^2);
           x(4,:);
          (f.*cos(x(3,:))+M*g.*sin(x(3,:))-M*ffric*mp.*x(4,:)-mp*l.*cos(x(3,:)).*sin(x(3,:)).*x(4,:).^2)./(M*l - mp*l.*cos(x(3,:)).^2)];

end % FUNCTION ENDS








%% TRASH
%  xdot = x(2,:);
%  xddot=(f - mp*l.*sin(x(3,:)).*x(4,:).^2 + mp*g.*cos(x(3,:)).*sin(x(3,:)))./(M - mp.*cos(x(3,:)).^2); 
%  thetadot = x(4,:);
%  thetaddot= g/l.*sin(x(3,:)) + (xddot./l).*cos(x(3,:));
%  xp = [xdot; xddot; thetadot; thetaddot];
%  X  = x + xp .* dt; 

% MIT UPSWING EXAMPLE PAPER
% function [xp] = Cart_Pole(x, f)
% 
% % System Parameters
%     mc = 1.2;       % [Kg] Mass of the cart 
%     mp = 0.5;       % [Kg] Mass of pendulum 
%     M  = mc+mp;     % Total mass
%     l  = 1.4;       % [m] length to pendulum center of mass 
%     g  = 9.81;       % [m/s^^2] gravitational accelleration
%     dt = 0.01;      % [s] Integration timestep
% 
% %  System Dynamics
% %     sys = @(x, f) [x(2,:);
% %                    1./(mc + mp.*sin(x(3,:)).^2).*(f+mp.*sin(x(3,:)).*(L*x(4,:).^2 + g*cos(x(3,:))));
% %                    x(4,:);
% %                    1./(L.*(mc+mp.*sin(x(3,:)).^2)).*(-f.*cos(x(3,:)) - mp*L.*x(4,:).^2.*cos(x(3,:)).*sin(x(3,:))-M.*g.*sin(x(3,:)))];
%                
%    sys = @(x, f) [x(2,:);
%                  1./(mc+mp.*sin(x(3,:)).^2).*(f+mp.*sin(x(3,:)).*(l*x(4,:).^2 + g.*cos(x(3,:))));
%                   x(4,:);
%                  1./(l.*(mc+mp.*sin(x(3,:)).^2)).*(-f.*cos(x(3,:))-mp*l.*x(4,:).^2.*cos(x(3,:)).*sin(x(3,:))-(mc+mp)*g.*sin(x(3,:)))];
%   
% %  Calculate xdot           
%     xdot = sys(x,f);
%     
% %  Euler integration
%     xp   = x + xdot .* dt; 
% 
% end % FUNCTION ENDS
