% function U = random_walk(min,max,n)
% Creates an input signal based on random walk principle
% Inputs: min - minimum value of control signal u
%         max - maxiumum value of control signal u
%         n - lenght of control signal 
%
% Output: Control signal u

function u = random_walk(uini, n)

%function inputs
if nargin == 1
    n = 1;
elseif nargin == 0
    uini = 0;
    n = 1;
end

% sigmoid function parameters (determines PDF)
a = 0.4; % the higher the value the steeper the membership function
mu= 0;   % position of steepest slope of membership function

% action initialization
limit = 10; % adjust parameter a when changing this value
u = zeros(1,n);
u(1) = uini;

for i = 1:n
    
    if u(i) == 0
        % if 0 do random action 
        u(i+1) = u(i) + sign(randn(1));
    else
        % Probability of the next action to increase or decrease
        p = max(sigmf(u(i),[a mu]),sigmf(u(i),[-a mu]));
        if rand(1) < p
             u(i+1) = u(i) - .25*sign(u(i));
        else u(i+1) = u(i) + .25*sign(u(i));
        end
    end
    
    % saturation
    if abs(u(i+1)) > limit
        u(i+1) = sign(u(i))*limit;
    end
    
    if n == 1
        u = u(end);
    end
end

