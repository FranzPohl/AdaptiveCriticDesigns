% Rectifier Linear Unit (ReLu) Transfer function for Neural Net neurons

classdef ReLu < TransferFunction
    methods (Static)
       
        % sig
        % Sigmoidal transferfunction
        function output = fun(input)
            output = max(0,input);
        end
   
        % dsig
        % Derivative sigmoidal transfer function: (1- tanh^2(x))
        function derivative = dfun(input)
            if input<=0
                derivative = 0;
            else
                derivative = 1;
            end
        end
    end
end
