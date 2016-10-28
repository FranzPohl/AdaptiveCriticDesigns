clear; close all; clc

% Get Class
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../PLANT')); 

%% Simulation Parameters

% time sequence
tmax = 3;
dt = 0.01;
t = 0:dt:tmax;
n = length(t);
Ntrials = 100;

% Data storage
Input  = [];
Output = [];
r2d = 180/pi;

% Limits
ulimit = .1;
xLimit = 1.2;
thetaLimit = pi;
thetadLimit= 30;
ur = [-ulimit ulimit];


%% Simulate

% choice of Dynamic System: 1. Inverted Pendulum 2. Cart-Pole
choice = 1;

if choice == 1
    for trial = 1:Ntrials
        clear x
        clear xp
        clear u
        clear normInp

        x = randn(2,1);
        %u = 10*-sign(x(1,1))*chirp(t,10,tmax,0);
        %u =  25*sign(randn(1))*sin(1*(1:n-1));
        %u = idinput(n-1,'prbs',[0 rand(1)],[-5 5]);
        u = sign(randn(1))*sin(10*t)*.1;
        if mod(trial,2) == 0
             u = rand(n-1) * ( max(ur)-min(ur) ) + min(ur);
        end
        %u = random_walk(randn(1)*2,n-1);    

        for i = 1:n-1

            x(:,i+1) = Inverted_Pendulum(x(:,i),u(i),dt);
            x(1,i+1) = x(1,i+1) + 2*pi*[abs(x(1,i+1))>pi]*-sign(x(1,i+1));

            if abs(x(2,i+1)) > thetadLimit %||abs(x(1,i+1)) > thetaLimit 
                break;
            end

        end
        
        %normalize
        for i=1:length(x)-1
            normInp(:,i) = [x(:,i);u(i)];
        end

        %store data
        Input  = [Input, normInp];
        Output = [Output, x(:,2:end)];
        
    end % end inverted pendulum
    
else % Cart - Pole Problem
    for trial = 1:Ntrials
        clear x
        clear xp
        clear u

        % initialize
        x = [randn(1)*0.3; zeros(3,1)];
        u = 0;
        for i = 1:n-1
            
            %u(i+1) = random_walk(u(i)); 
            u(i) = rand(1) * (max(ur)-min(ur)) + min(ur);
            x(:,i+1) = Cart_Pole(x(:,i),u(i),dt);

            if abs(x(1,i+1)) > xLimit || abs(x(4,i+1)) > thetadLimit
                    break;
            end
        end
        
        Input  = [Input, [x(:,1:end-1); u]];
        Output = [Output, x(:,2:end)];
        
    end % end trials cart-pole
    
end % end data acquisition


%% Normalization Gains

xn1 = linspace(-thetaLimit, thetaLimit,1000);
xn2 = linspace(-thetadLimit, thetadLimit,1000);
un  = linspace(-ulimit, ulimit,1000);
[~, ptx]= mapminmax([xn1;xn2; un]);
[~, pty]= mapminmax([xn1;xn2]);


%% Plot

if choice == 1
    % State
    figure(1)
    subplot(2,1,1)
    sim = length(x);
    plot(t(1:sim),x(1,:)*r2d);
    xlabel('time [s]');
    ylabel('angle [deg]');
    subplot(2,1,2)
    plot(t(1:sim),x(2,:)*r2d);
    xlabel('time [s]');
    ylabel('angular rate [deg/s]');
    % print('states2','-dpng','-r300');

    % Control signal
    figure(2);
    plot(t(1:sim-1),u(1:sim-1));
    xlabel('time [s]');
    ylabel('torque [Nm]');
    % print('ctrl2','-dpng','-r300');
else
    % State
    figure(1)
    subplot(2,1,1)
    sim = length(x);
    plot(t(1:sim),x(1,:));
    xlabel('time [s]');
    ylabel('Distance [m]');
    subplot(2,1,2)
    plot(t(1:sim),x(3,:)*r2d);
    xlabel('time [s]');
    ylabel('angle [deg]');
    % print('states2','-dpng','-r300');

    % Control signal
    figure(2);
    plot(t(1:sim-1),u(1:sim-1));
    xlabel('time [s]');
    ylabel('Force [N]');
    % print('ctrl2','-dpng','-r300');
end

%% Data Post-processing

% Shuffle data set
data_N = length(Input);
shuffle    = randperm(data_N);
Input      = Input(:,shuffle);
Output     = Output(:,shuffle);

% Split Data set 60% training, 40% Validation
ptrain = 0.6;

% Training DataSet
train_input = Input(:,1:round(ptrain*data_N));
train_output = Output(:,1:round(ptrain*data_N));
train_data = [train_input; train_output];

% Validation Data Set
val_input   = Input(:,round(ptrain*data_N)+1:end);
val_output   = Output(:,round(ptrain*data_N)+1:end);
test_data = [val_input; val_output];

save('dataset','train_data','test_data','ptx','pty');

NNtoolinput = [train_input, val_input];
NNtooloutpt = [train_output, val_output];

save('NNtoolInp','NNtoolinput');
save('NNtoolOut','NNtooloutpt');





