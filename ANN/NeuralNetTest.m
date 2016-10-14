
%% Test Class Definition
classdef NeuralNetTest < matlab.unittest.TestCase
    
    % Test Method Block
    methods (Test)
        
       
        function testNeuralNetStruct(testCase)  
        % Test Neural Network Constructor
            struct = [3 4 2];
            net = NeuralNet(struct);
            act = size(net.weights{1});
            exp = [4 3];
            testCase.verifyEqual(act,exp)
            
            act = size(net.weights{2});
            exp = [2 4];
            testCase.verifyEqual(act,exp)
            
            act = size(net.bias{1});
            exp = [4 1];
            testCase.verifyEqual(act,exp)
            
            act = size(net.bias{2});
            exp = [2 1];
            testCase.verifyEqual(act,exp)
            
            bias{1} = [-0.0631; 0.7147; -0.2050; -0.1241];
            bias{2} = [1.4897; 1.4090];
            weights{1}= [1.4172    1.6302   -0.3034;
                         0.6715    0.4889    0.2939;
                        -1.2075    1.0347   -0.7873;
                         0.7172    0.7269    0.8884];
            weights{2}= [-1.1471   -0.8095    1.4384   -0.7549;
                         -1.0689   -2.9443    0.3252    1.3703];
            net.iniBias(bias);
            net.iniWeights(weights);
            testCase.verifyEqual(net.weights{1},weights{1})
            testCase.verifyEqual(net.weights{2},weights{2})
            testCase.verifyEqual(net.bias{1},bias{1})
            testCase.verifyEqual(net.bias{2},bias{2})
        end
        
        
        function testFeedForward(testCase)
        % Test Feed Forward Pass
            struct = [3 4 2];
            net = NeuralNet(struct);
            
            bias{1} = [-0.0631; 0.7147; -0.2050; -0.1241];
            bias{2} = [1.4897; 1.4090];
            weights{1}= [1.4172    1.6302   -0.3034;
                         0.6715    0.4889    0.2939;
                        -1.2075    1.0347   -0.7873;
                         0.7172    0.7269    0.8884];
            weights{2}= [-1.1471   -0.8095    1.4384   -0.7549;
                         -1.0689   -2.9443    0.3252   1.3703];
            net.iniBias(bias);
            net.iniWeights(weights);
            
            input = [0.1; -2; 1.6];
            act = net.FFwrd(input);
            exp = [1.0460; 1.2480];      
            testCase.verifyEqual(act,exp);
        end
        
        
        function testSGD(testCase)   
        % Test Stochastic Gradient Descent
            struct = [3 4 2];
            net = NeuralNet(struct);
            
            weights{1}= [1.4172    1.6302   -0.3034;
                         0.6715    0.4889    0.2939;
                        -1.2075    1.0347   -0.7873;
                         0.7172    0.7269    0.8884];
                     
            weights{2}= [-1.1471   -0.8095    1.4384   -0.7549;
                         -1.0689   -2.9443    0.3252   1.3703];
                     
            bias{1} = [-0.0631; 0.7147; -0.2050; -0.1241];
            bias{2} = [1.4897; 1.4090];
                                 
            net.iniBias(bias);
            net.iniWeights(weights);
            
            input = [0.1; -2; 1.6];
            output = [3.2; -0.5];
            data = [input; output];
            epochs = 1;
            eta =0.1;
            batch_size = 1;
            
            net.SGD(data,epochs,eta,batch_size)
            
            act = net.weights{1};
            exp = [1.4172   1.6305  -0.30365;
                   0.7031  -0.14296  0.79939;
                   -1.2075  1.03339 -0.78625;
                   0.67727  1.52533  0.24958];
            testCase.verifyEqual(act,exp);
            
            act = net.weights{2};
            exp = [-1.36222 -0.75186 1.22328 -0.773101
                   -0.89433  2.99107 0.49977  1.385076];
            testCase.verifyEqual(act,exp);
            
            act = net.bias{1};
            exp = [-0.0631; 1.03063; -0.2043; -0.5233];
            testCase.verifyEqual(act,exp);
            
            act = net.bias{2};
            exp = [1.7051; 1.2342];
            testCase.verifyEqual(act,exp);            
            
        end
%         
%                 % Test Function
%         function testASolution(testCase)      
%             % Exercise function under test
%             % act = the value from the function under test
% 
%             % Verify using test qualification
%             % exp = your expected value
%             % testCase.<qualification method>(act,exp);
%         end
    end
end

