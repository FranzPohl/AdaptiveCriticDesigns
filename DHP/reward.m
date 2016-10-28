function r = reward(x,u)
% Reward function calculates the reward based on the state. It is
% independent of the way on how it reached that state (independent of
% action)
% Input: state x
% Ouput: numerical reward r

if nargin == 1
    u = 0;
end
    % case simple inverted pendulum
    %r = -x(:)'*[2.5e-1, 0; 0, 3.0e-3]*x(:) - 3.0e-3*u^2;
%     r = -abs(x(1).^2);
    r = -( x(:)'*[.9, 0; 0, 0.1]*x(:) + .1*u^2 );
    
    % if abs(x(1)) > .5*pi
    %     r = -1;
    % else r = 0;
    % end
end