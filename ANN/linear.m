% Linear Transfer function for Neural Net neurons

classdef linear < TransferFunction
    methods (Static)
        
        %Linear transferfunction feeds through input
        function output = fun(input)
            output = purelin(input);
        end
        
        %Derivative of linear function is: 1
        function derivative = dfun(input)
            derivative = ones(size(input));
        end
    end
end