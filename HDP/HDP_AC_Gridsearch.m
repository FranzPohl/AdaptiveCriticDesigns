%% HDP_AC
% Trains Actor and Critic Simultanously. The model of the plant is 
% pretrained.
% grid search 
% update Actor with TD and eligibility traces

clc; clear; close all;

% Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 
addpath(fullfile(mfilepath,'../PLANT')); 
addpath(fullfile(mfilepath,'../MODEL')); 

% Load pre-trained model, critic and actor
load('model_trqLimited.mat')

%% Constants

% Actor 
numInA     = 2;
numOutA    = 1;
tauA   = 0.00;
muA    = 0.00;

% Critic
numInC  = 2;
numOutC = 1;
gamma   = 0.97;                                                                           
muC     = 0.00;   
lambdaC = 0.0;

% Choice of Reward function
% 1: binary
% 2: quadratic
% 3: weight matrix
% 4: cosine
choice = 3; 
Jstar = 0;

% Limitations
xdl= 40; 

% Simulation time
tmax    = [5 10 20];
dt      = 0.005;
r2d     = 180/pi;

Ntrials = 400;
Ntrain = 0;
%% GridSearch

numNeuronA = 12;
numNeuronC = 10;
etaA = [.004 .006];
etaC = [.04 .06];

%% STEP IV HDP CRITIC AND ACTOR CONNECTED AND NOT PRETRAINED

for g = tmax
    
    t       = 0:dt:g;
    n       = length(t);
    
    for h = numNeuronA

        for i = numNeuronC

            for k = etaA

                for l = etaC

                    tic 

                    clear mseC
                    clear mseA
                    clear critic
                    clear actor

                    critic = NeuralNet([numInC, i, numOutC]);
                    actor = NeuralNet([numInA, h, numOutA]);
                    actor.transferFun{end} = sigmoid;

                    Rlog = [];


                    for trial = 1:Ntrials

                        clear r;
                        clear x;
                        clear u;
                        clear J;
                        clear xn;
                        clear xhat;


                        if mod(trial, Ntrials) == 0
                            x = [pi; 0]; % xn0
                        else x = sign(randn(2,1)).*rand(2,1).*[pi; 5]; % xn0
                        end

                        %u = sign(randn(1))*sin(t*6);
                        x(1) = x(1) + 2*pi * [ abs(x(1))>pi ] * -sign(x(1));
                        xn= mapminmax( 'apply', x, pty ); 
                        xhat = xn;


                        for j = 1:n-1

                            u(j) = actor.FFwrd( xn(:,j) );    

                            denorm = mapminmax( 'reverse', [ xn(:,j);u(j) ], ptx );

                            x(:,j+1) = Inverted_Pendulum( x(:,j), denorm(3), dt );
                            x(1,j+1) = x(1,j+1) + 2*pi * [abs(x(1,j+1))>pi] * -sign(x(1,j+1));
                            xhat(:,j+1) = model.FFwrd( [xn(:,j);u(j)] );
                            xn(:,j+1)= mapminmax( 'apply', x(:,j+1), pty );

                            r(j) = reward( choice, xn(:,j+1) );
                            J(j) = critic.FFwrd( xn(:,j+1) );
                            delta_J = J(j) - Jstar;

                            dJdx = critic.net_derivative( xn(:,j+1) );
                            dxdu = model.net_derivative( [xn(:,j); u(j)], dJdx );

                            critic.updateC_HDP( xn(:,j:j+1), r(j), l, muC, gamma, lambdaC );

                            actor.updateA_HDP( xn(:,j), delta_J, dxdu(3), k, muA );

                            if abs(x(2,j+1)) > xdl
                                break;
                            end

                        end

                        %END OF TRIAL

                        Rlog = [Rlog sum(r)];

                        mseC(trial) = critic.evaluateC(xn, gamma, r);
                        mseA(trial)  = .5*norm(J)/(length(xn));
                        fprintf('Trial %i/%i: TD = %i    Actor Error = %i\n', trial, Ntrials, mseC(trial), mseA(trial))

                    end

                    Ntrain = Ntrain + 1;
                    %END OF ALL TRIALS - PLOTS
                    fprintf('TRAINING %i ENDED! PLOTTING \n', Ntrain);
                    Xlog = xn;

                    s1 = 'etaA: ';
                    s2 = num2str(k);
                    s3 = ' NumNA: ';
                    s4 = num2str(h);
                    s5 = ' etaC: ';
                    s6 = num2str(l);
                    s7 = ' NumNC: ';
                    s8 = num2str(i);
                    st = strcat(s1,s2,s3,s4,s5,s6,s7,s8);

                    denorm = mapminmax( 'reverse',[xn(:,1:end-1); u(1:length(xn)-1)], ptx );
                    xhat2 = mapminmax( 'reverse',xhat, pty );

                    % Reward and Error
                    figure()
                    subplot(3,1,1)
                    title(st)
                    plot(1:Ntrials, Rlog)
                    xlabel('epochs'); ylabel('reward[-]');
                    subplot(3,1,2)
                    plot(mseA)
                    xlabel('epochs'); ylabel('mse Actor')
                    subplot(3,1,3)
                    plot(mseC)
                    xlabel('epochs'); ylabel('mse Critic [-]');

                    % States
                    figure()
                    subplot(3,1,1)
                    title(st)
                    plot(t(1:length(denorm)),denorm(3,:))
                    xlabel('time [s]'); ylabel('actions')
                    xlim([0 g]);
                    grid on
                    subplot(3,1,2)
                    plot(t(1:length(x)),x(1,:)*r2d)
                    hold on 
                    plot(t(1:length(x)),xhat2(1,:)*r2d)
                    hold off
                    xlabel('time [s]'); ylabel('\theta [deg]')
                    xlim([0 g]);
                    grid on
                    subplot(3,1,3)
                    plot(t(1:length(x)),x(2,:))
                    hold on 
                    plot(t(1:length(x)),xhat2(2,:))
                    hold off
                    xlabel('time [s]'); ylabel('\theta_d [deg/s]')
                    xlim([0 g]);
                    grid on

                    xnorm1 = mapminmax('apply',[-pi;-8*pi;0],ptx);
                    xnorm2 = mapminmax('apply',[pi;8*pi;0],ptx);
                    x1plot = linspace( xnorm1(1), xnorm2(1), 30 ); 
                    x2plot = linspace( xnorm1(2), xnorm2(2), 40 );
                    [X1,X2] = meshgrid( x1plot, x2plot ); % create rectangular meshgrid
                    Zcrit = zeros( size(X1) );
                    Zact = zeros( size(X1) );
                    for s = 1:length(x1plot)
                        for m = 1:length(x2plot)
                            Zcrit(m,s) = critic.FFwrd([X1(m,s); X2(m,s)]);
                            Zact(m,s) = actor.FFwrd([X1(m,s); X2(m,s)]);
                        end
                    end

                    figure(); clf
                    % Critic
                    subplot(2,6,[1:3 7:9]);
                    surf(X1,X2,Zcrit)
                    title(st)
                    xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('J');
                    hold on
                    y = critic.FFwrd([Xlog(1,:);Xlog(2,:)]);
                    plot3(Xlog(1,:),Xlog(2,:),y,'k.');
                    xlim([-1 1]); ylim([-1 1]);
                    hold off

                    % Actor
                    subplot(2,6,[4:6 10:12])
                    surf(X1,X2,Zact)
                    title(st)
                    xlabel('\theta'); ylabel('\theta_{dot}'); zlabel('control');
                    xlim([-1 1])
                    hold on
                    y = actor.FFwrd([Xlog(1,:);Xlog(2,:)]);
                    plot3(Xlog(1,:),Xlog(2,:),y,'k.');
                    xlim([-1 1]); ylim([-1 1]);
                    hold off

                    figure()
                    title(st)
                    contourf(X1,X2,Zcrit);
                    grid on;
                    xlabel('\theta'); ylabel('\theta_{dot}')
                    colorbar

                    figure()
                    contourf(X1,X2,Zact);
                    grid on;
                    xlabel('\theta'); ylabel('\theta_{dot}');
                    s9 = ' Actor';
                    str = strcat(st,s9);
                    title(str)
                    colorbar

                    % Simulation time computation
                    comptime( Ntrain ) = toc;
                    avgtime = sum( comptime ) / Ntrain;
                    rem_time = ( 16 - Ntrain ) * avgtime;
                    rem_hour = floor( rem_time/3600 ); rem_time = mod( rem_time, 3600 );
                    rem_min  = floor( rem_time/60 );   rem_time = mod( rem_time, 60 );
                    rem_sec  = floor( rem_time );

                    fprintf('Trial %i/%i is finished in %.2f seconds, estimated time remaining: %i hours %i min %i sec \n', ...
                        Ntrain, 16, comptime( Ntrain ), rem_hour, rem_min, rem_sec );

                end

            end

        end

    end

end

