% Artificial Neural Network Class

classdef NeuralNet < handle
    properties (GetAccess=private)
        
        numInputs
        numOutputs
        numLayers
        Vw
        Vb
        
    end
    
    properties
        
        transferFun
        weights
        prevWeights
        weightlog
        bias
        prevBias
        biaslog
        prevCost
        
    end
    
    methods
        %% NEURAL NET STRUCT
        
        function obj = NeuralNet(struct) 
        % NeuralNet
        % Creates a artificial neural network object with 1 hidden layer
        % Input: struct is a vector containing the number of neurons for each
        %        layer. e.g. a Neural net with 2 inputs, 1 hidden layer of
        %        20 neurons and 3 output nodes the vector is [2 20 3]
        %
        % obj - neural network object with random ini weights and biases
        %rng(1397524); % to keep consistent results for now
        %rng(14051955);
           obj.numLayers  = length(struct);
           obj.numInputs  = struct(1);    
           obj.numOutputs = struct(end); 
           
           % initialize weights
           for layer = 1:obj.numLayers-1
              obj.weights{layer} =  1/sqrt(struct(layer))*randn(struct(layer+1),struct(layer));  
              obj.Vw{layer} = zeros(size(obj.weights(layer)));
              obj.transferFun{layer} = sigmoid;
           end
           
           % initialize biases
           for layer = 2:obj.numLayers
               obj.bias{layer-1} = randn(struct(layer),1);
               obj.Vb{layer-1} = zeros(size(obj.bias(layer-1)));
           end
           
           % Output layer, Comment if sigmoidal is prefered
           obj.transferFun{obj.numLayers-1} = linear;
           obj.prevCost = +inf;
        end 

        
        
        function iniWeights(obj, weights)
        % Initialize Weights as desired (Watch out to keep size!)
            for layer = 1:obj.numLayers-1
                if size(obj.weights{layer})~= size(weights{layer})
                    fprintf('Initialized weights have to be of same dimension')
                    return;
                end
                    obj.weights{layer} = weights{layer};    
            end
        end

        
        
       function iniBias(obj, bias)
       % Initialize Bias 
            for layer = 1:obj.numLayers-1
                obj.bias{layer} = bias{layer};    
            end
       end
       
       
       
       function outputs = FFwrd(obj, inputs)
       % FFwrd
       % Neural Net feed forward computation
       % inputs - column vector of inputs to the neural net
       %
       % outputs - ANN outputs   
       [~, nSamples] = size(inputs);
          for layer = 1:obj.numLayers-1
              inputs = obj.transferFun{layer}.fun(obj.weights{layer}*inputs + repmat(obj.bias{layer},1,nSamples));
          end

          outputs = inputs;
       end 
        
       
       function WeightLog = getWeights(obj)
       % getWeights
       % Returns the current neural network weights in cell array
       %
       % outputs: WeightLog 
           for layer = 1:obj.numLayers-1
                   WeightLog{layer,1} = obj.weights{layer};
                   WeightLog{layer,2} = obj.bias{layer};
           end
       end
        
        
        %% NEURAL NET TRAINING STOCHASTIC GRADIENT DESCENT
        
        function [mse, mseetr] = SGD(obj, train_data, batch_size, epochs, eta, mu, lambda, test_data)
        % SGD - Stochastic Gradient Descent 
        % trains the NeuralNet object
        % train_data    - input-output tuples [[x];[y]]
        % batch_size    - batch size per training cycle
        % epochs        - training epochs 
        % eta           - learning rate
        % mu            - momentum co-efficient
        % test_data     - validation data with input-output tuples
        %
        % mse - mean squared error
            [~, nSamples] = size(train_data); 
            shuffle = randperm(nSamples);
            for epoch = 1:epochs

                for layer = 1:obj.numLayers-1
                    obj.Vw{layer} = zeros(size(obj.weights{layer}));
                    obj.Vb{layer} = zeros(size(obj.bias{layer}));
                end
                
                % Save weights
                obj.weightlog{1}(:,:,epoch) = obj.weights{1};
                obj.weightlog{2}(:,:,epoch) = obj.weights{2}';
                obj.biaslog{1}(:,epoch) = obj.bias{1};
                obj.biaslog{2}(:,epoch) = obj.bias{2};
                
                % Shuffle training data set
                train_data =  train_data(:,shuffle);

                for i = 1:batch_size:(nSamples-batch_size+1) %UPDATE! 
                    obj.updateNet(nSamples, train_data(:,i:i+batch_size-1), eta, mu, lambda);              
                end
                
                if nargin == 8
                    mse(epoch) = obj.evaluate(test_data,lambda);
                    mseetr(epoch) = obj.evaluate(train_data,lambda);
                    fprintf('Epoch %i: %i \n', epoch, mse(epoch)); 
                    if mse(epoch) < obj.prevCost
                        obj.prevWeights = obj.weights;
                        obj.prevBias    = obj.bias;
                        obj.prevCost = mse(epoch);
                    end
                else
                    mse = 1;
                    mseetr = 1;
                    %fprintf('Epoch %i complete \n', epoch);
                end
                
            end
        end
        
        
        
        function updateNet(obj, nSamples, batch, eta, mu, lambda)
        % Updates the Network Weights and biases using backpropagation
        % inputs: batch data
        %         learning rate eta
        %         moment co-efficient mu
        %
        % Network weights are updated
            for layer = 1:obj.numLayers-1
                dW{layer} = zeros(size(obj.weights{layer}));
                dB{layer} = zeros(size(obj.bias{layer}));
            end 
            
            for i = 1:size(batch,2)
                [deltaW, deltaB] = obj.backpropagation(batch(1:obj.numInputs,i),batch(obj.numInputs+1:(obj.numInputs+obj.numOutputs),i));
                for layer = 1:obj.numLayers-1
                    dW{layer} = dW{layer} + deltaW{layer};
                    dB{layer} = dB{layer} + deltaB{layer};
                end
            end
            
            for layer = 1:obj.numLayers-1
                obj.Vw{layer} = obj.Vw{layer}.*mu - eta/length(batch)*dW{layer}; 
                obj.weights{layer} = (1-eta*lambda/nSamples).*obj.weights{layer} + obj.Vw{layer};
                obj.Vb{layer} = obj.Vb{layer}.*mu - eta/length(batch)*dB{layer};
                obj.bias{layer} = obj.bias{layer} + obj.Vb{layer};
            end
        
        end
        
        
        
        function [deltaW, deltaB] = backpropagation( obj,x,y )
        % backpropagation 
        % inputs: input data x
        %         output data y
        %
        % outputs: delta Weights and biases for each layer
            for layer = 1:obj.numLayers-1
                deltaW{layer} = zeros(size(obj.weights{layer}));
                deltaB{layer} = zeros(size(obj.bias{layer}));
            end
            
            % Forward pass (save all activations and neuron outputs)
            activation{1} = x;
            for layer = 1:obj.numLayers-1
                z{layer} = obj.weights{layer}*activation{layer} + obj.bias{layer};
                activation{layer+1} = obj.transferFun{layer}.fun(z{layer});
            end
            
            % first derivatives
            delta = obj.cost_derivative(activation{end}, y).*obj.transferFun{obj.numLayers-1}.dfun(z{end});
            deltaB{end} = delta;
            deltaW{end} = delta*(activation{end-1})';
            
            % Backpropagate
            for layer = obj.numLayers-2:-1:1
                delta = ((obj.weights{layer+1})'*delta).*obj.transferFun{layer}.dfun(z{layer});
                deltaB{layer} = delta;
                deltaW{layer} = delta*(activation{layer})'; 
            end   
        end
        
        
        
        function cost = evaluate( obj, data, lambda )
        % Evaluation of Neural Network performance
        % inputs: data tuples [[x];[y]]
           [~, nSamples] = size(data);
           x = data(1:obj.numInputs,:);
           y = data(obj.numInputs+1:obj.numInputs + obj.numOutputs,:);
           
           % Regularization
           weightSquaredSum = 0;
           for layer = 1:obj.numLayers-1
               weightSquaredSum = weightSquaredSum + sum(sum(obj.weights{layer}.^2));
           end
           
           yNN = obj.FFwrd(x);
           cost = 0.5*sum((yNN - y).^2,1);
           cost = (sum(cost) + .5*lambda*weightSquaredSum)/nSamples;     
           
           if isnan(cost)
               error('NaN detected, script terminated');
           end
           
        end       
        
        
        
        function best_result( obj )
        % Take the weights and biases from the best training epoch 
        % and use them for the final network structure.
        % Network first needs to be trained
            if obj.prevWeights{1} == 0
                error('No log, train network first')
            else
                obj.weights = obj.prevWeights;
                obj.bias    = obj.prevBias;
            end
        end
        
        
        
        %% DERIVATIVES
        
        function derivative = cost_derivative( obj, net_output, y )
        % computes cost_derivative with respect to error
        % inputs: networ output
        %         desired (output) data y
        %
        % outputs: partial derivative of cost dJ
              derivative =  net_output - y;
        end 
        
        
        
        function derivative = net_derivative( obj, x, prev_derivative )
        % Computes the derivative of the output with respect to the input
        % inputs:  x - inputs to the network
        % outputs: delta - derivative of output w.r.t input
            activation{1} = x;
            for layer = 1:obj.numLayers-1
                z{layer} = obj.weights{layer}*activation{layer} + obj.bias{layer};
                activation{layer+1} = obj.transferFun{layer}.fun(z{layer});
            end
            
            if nargin == 2
            % Backprobagation
                derivative = (obj.weights{end})'*obj.transferFun{obj.numLayers-1}.dfun(z{end});
            else    
                derivative = (obj.weights{end})'*(obj.transferFun{obj.numLayers-1}.dfun(z{end}).*prev_derivative);
            end
                derivative = derivative.*obj.transferFun{obj.numLayers-2}.dfun(z{end-1});
                derivative = (derivative'*obj.weights{end-1})';         
        end
        
        
        
        function derivative = net_derivativeSingle( obj, x, prev_derivative )
        % Computes the derivative of each output w.r.t each input and
        % returns a matrix with numOutputs rows and numInputs columns where
        % the first row would be a row vector with dy1/dx1 dy1/dx2 etc.
        % inputs:  x - inputs to the network
        % outputs: derivative - Matrix containing the derivatives y w.r.t x
            activation{1} = x;
            for layer = 1:obj.numLayers-1
                z{layer} = obj.weights{layer}*activation{layer} + obj.bias{layer};
                activation{layer+1} = obj.transferFun{layer}.fun(z{layer});
            end
            
            if nargin == 2
            % Backprobagation
                for i = 1:obj.numOutputs
                    delta(i,:) = (obj.weights{end}(i,:))*obj.transferFun{obj.numLayers-1}.dfun(z{end}(i));
                    delta(i,:) = delta(i,:).*obj.transferFun{obj.numLayers-2}.dfun(z{end-1})';
                    derivative(i,:) = delta(i,:)*obj.weights{end-1};    
                end
            else 
                 for i = 1:obj.numOutputs
                    delta(i,:) = (obj.weights{end}(i,:))*(obj.transferFun{obj.numLayers-1}.dfun(z{end}(i)).*prev_derivative(i));
                    delta(i,:) = delta(i,:).*obj.transferFun{obj.numLayers-2}.dfun(z{end-1})';
                    derivative(i,:) = delta(i,:)*obj.weights{end-1};  
                 end
            end
                     
        end
        
         
         
        %% HDP TRAINING FUNCTIONS
        
        function updateC_HDP( obj, x, reward, eta, mu, gamma, lambda )
        % updates the critic network weights according to HDP law
        % Inputs: states visited x (size determines batch size)
        %         rewards received
        %         discount factor gamma
        %         learning race eta
        %         momentum factor mu
            batch_size = length(x)-1;
            Jtp1 = obj.FFwrd(x(:,2:end));
            yD = gamma*Jtp1 + reward; % desired output
            batch = [x(:,1:batch_size); yD];
            obj.updateNet(batch_size, batch, eta, mu, lambda); 
        end
                
        
        
        function mseC = evaluateC( obj, x, gamma, reward )
        % Evaluates the critic network by computing the temporal
        % difference error
        % Inputs: x - state vector at two (or more) conescutive timesteps
        %         gamma  - discount factor
        %         reward - value of utility function at state x(t+1)
            Jt = obj.FFwrd( x(:,1:end-1) );
            Jtp1 = obj.FFwrd( x(:,2:end) );
            TD = Jt - ( gamma*Jtp1 + reward );
            mseC = .5*norm(TD)/( length(x) );
        end
        
        
        
        function updateA_HDP( obj, x, delta_J, dJdu, eta, mu )
        % updates the actor network weights according to the evaluation of
        % the critic (HDP law)
        % Inputs: x - State 
        %         u - Control signal
        %         J - Value of next state 
        %         dJdu - Derivative of value w.r.t. control signal
        %         etaA - Actor learning rate
        %         muA  - momentum factor
            lambda = 0;
            for layer = 1:obj.numLayers-1
                dW{layer} = zeros(size(obj.weights{layer}));
                dB{layer} = zeros(size(obj.bias{layer}));
            end
            
            % Forward pass (save all activations and neuron outputs)
            activation{1} = x;
            for layer = 1:obj.numLayers-1
                z{layer} = obj.weights{layer}*activation{layer} + obj.bias{layer};
                activation{layer+1} = obj.transferFun{layer}.fun(z{layer});
            end
            
            % first derivatives
            delta = dJdu.*delta_J.*obj.transferFun{obj.numLayers-1}.dfun(z{end});
            dB{end} = delta;
            dW{end} = delta*(activation{end-1})';
            
            % Backpropagate
            for layer = obj.numLayers-2:-1:1
                delta = ((obj.weights{layer+1})'*delta).*obj.transferFun{layer}.dfun(z{layer});
                dB{layer} = delta;
                dW{layer} = delta*(activation{layer})'; 
            end  
            
            for layer = 1:obj.numLayers-1
                obj.Vw{layer} = obj.Vw{layer}.*mu - eta*dW{layer}; 
                obj.weights{layer} = (1-eta*lambda).*obj.weights{layer} + obj.Vw{layer};
                obj.Vb{layer} = obj.Vb{layer}.*mu - eta*dB{layer};
                obj.bias{layer} = obj.bias{layer} + obj.Vb{layer};
            end
        end
        
        
        %% DHP TRAINING FUNCTIONS
        
        function error = updateC_DHP( obj, x, lambda, da_dx, dxhat_dx, dr_dx, eta, mu, gamma )
        % updates the critic network weights according to DHP law
        % Inputs: states visited x (size determines batch size)
        %         rewards received
        %         discount factor gamma
        %         learning race eta
        %         momentum factor mu
            for i = obj.numInputs
               dJtp1_dx(i) = lambda(:,2)'*dxhat_dx(:,i) + lambda(:,2)'*(dxhat_dx(:,3).*da_dx(i)); 
            end
            y = lambda(:,1);
            yD = gamma * dJtp1_dx' + dr_dx; % desired output
            error = y - yD;
            data = [x; yD];
            obj.updateNet(1, data, eta, mu, 0); 
         end
        
         
         
        function da = updateA_DHP( obj, x, lambda, dxhat_da, dr_da, eta, mu, gamma )
        % updates the actor network weights according to the evaluation of
        % the critic (HDP law)
        % Inputs: x - State 
        %         u - Control signal
        %         J - Value of next state 
        %         dJdu - Derivative of value w.r.t. control signal
        %         etaA - Actor learning rate
        %         muA  - momentum factor
            for layer = 1:obj.numLayers-1
                dW{layer} = zeros(size(obj.weights{layer}));
                dB{layer} = zeros(size(obj.bias{layer}));
            end
            
            % Forward pass (save all activations and neuron outputs)
            activation{1} = x;
            for layer = 1:obj.numLayers-1
                z{layer} = obj.weights{layer}*activation{layer} + obj.bias{layer};
                activation{layer+1} = obj.transferFun{layer}.fun(z{layer});
            end
            
            % first derivatives
            da = gamma*lambda'*dxhat_da + dr_da; 
            delta = da*obj.transferFun{obj.numLayers-1}.dfun(z{end});
            dB{end} = delta;
            dW{end} = delta*(activation{end-1})';
            
            % Backpropagate
            for layer = obj.numLayers-2:-1:1
                delta = ((obj.weights{layer+1})'*delta).*obj.transferFun{layer}.dfun(z{layer});
                dB{layer} = delta;
                dW{layer} = delta*(activation{layer})';
            end  
            
            for layer = 1:obj.numLayers-1
                obj.Vw{layer} = obj.Vw{layer}.*mu - eta*dW{layer}; 
                obj.weights{layer} = obj.weights{layer} + obj.Vw{layer};
                obj.Vb{layer} = obj.Vb{layer}.*mu - eta*dB{layer};
                obj.bias{layer} = obj.bias{layer} + obj.Vb{layer};
            end
        end
        

    end
end
