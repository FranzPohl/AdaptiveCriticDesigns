function r = reward(choice, x)
% Reward function calculates the reward based on the state. It is
% independent of the way on how it reached that state (independent of
% action)
% Input: state x
% Ouput: numerical reward r

switch choice
    
    case 1  % binary
        if abs(x(1)) > pi
            r = -1;
        else 
            r = 0;
        end
            
    case 2 % quadratic
        r = -x(1).^2;
                
    case 3 % angle and angular rate
        r = - x(:)'*[.85, 0; 0, .1]*x(:);
        %r = - x(:)'*[.9, 0; 0, .1]*x(:);
                
    case 4 % cosine
        r = .9*cos(x(1)) - 0.1*x(2)/30;
        
    case 5 % more descrete
        if abs( x(1) ) < 0.25 
            r = exp(-.5*15^2*x(1)^2);
        else 
            r = 0;
        end

end