% Sigmoidal Transfer function for Neural Net neurons

classdef sigmoid < TransferFunction
    methods (Static)
       
        % Sigmoidal transferfunction
        function output = fun(input)
            output =  tanh(input); %1./(1+exp(-input)); %!
        end
        
        % Derivative sigmoidal transfer function: sigmoid(x)*(1- sigmoid(x))
        function derivative = dfun(input)
            derivative = 1-tanh(input).^2; %1./(1+exp(-input)).*(1- 1./(1+exp(-input))); %!
        end
    end
end
