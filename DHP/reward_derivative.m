function dr = reward_derivative(x,u)
% Derivative of reward function. Note that it is independent of the
% action
% Input: state x
% Ouput: derivative vector of reward r
% reward function r = -(x1^2)

if nargin == 1
    u = 0;
end
    % case simple inverted pendulum
    drdx1 = 2*0.9*x(1);
    drdx2 = 2*0.1*x(2);
    drdu  = 2*0.1*u;
    
    dr = [drdx1; drdx2; drdu];

end