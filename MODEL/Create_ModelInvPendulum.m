% function [net, ptx, pty] = Create_ModelInvPendulum()
% MODEL NETWORK
% Creates and pre-trains the model Network. Needs NeuralNet.m class
% Training and validation data sets are acquired with Dataset.m
% Stochastic Gradient Descent
% ModelNet inputs: - theta
%                  - theta dot
%                  - action (control signal)
%
% ModelNet output: - theta
%                  - theta dot
% author: Franz Pohl
% email: Franz.m.Pohl@gmail.com

clear;
close all;
clc;

%%Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 
addpath(fullfile(mfilepath,'../PLANT')); 

%% Data Set

load('dataset_trqLimited.mat');  % training data set

% Normalize Data
[train_data(1:3,:), ptx] =  mapminmax('apply', train_data(1:3,:),ptx);
[train_data(4:5,:), pty] =  mapminmax('apply', train_data(4:5,:),pty);
[test_data(1:3,:)] = mapminmax('apply',test_data(1:3,:), ptx);
[test_data(4:5,:)] = mapminmax('apply',test_data(4:5,:), pty);

%% Neural Network 

numInputs = 3;
numNeurons= 10;
numOutputs= 2;
struct =  [numInputs numNeurons numOutputs];
model = NeuralNet(struct);

% Hyper parameters
batch_size = 400; % number of elements per batch 318
epochs = 1;     % number of training epochs
eta = 0.5;       % learning rate 0.5
mu  = 0.6;        % momentum co-efficient 0.7
lamda = 0.01;     % realization parameter

% Train Neural Net
[msee, mseetr] = model.SGD(train_data, batch_size, epochs, eta, mu, lamda, test_data);
model.best_result   % Use best performed net during training

%% Error Plots
% figure()
% plot(msee);
% hold on;
% plot(mseetr);
% legend('cost validation data','cost train data');
% xlabel('epochs');
% ylabel('cost');
% grid on
% print('train','-dpng','-r300');

%% Simulation
% Layer 1
%bias{1} = [-506.75052295180706;507.86201274906114;-556.05090666769968;-0.7011431304244099;-0.0022605116536111021;0.0025364502704699247;0.6195009699537356;-474.56553021888618;-17.021904137344052;-17.064799036746159];
%weights{1} = [-506.90855020666265 -56.231418310579549 -1.580252615746365;-507.36968631952715 -56.904515292958919 -1.271470535136888;555.50544442238549 62.323357905228534 1.3720569159903666;0.062224873917703802 0.15637643962146469 0.0040303545629624595;-0.5157434612983155 -0.036603847236640098 -4.8609307266634535e-05;0.65558738436532982 0.04542296253441503 0.00011273515537897863;0.056538975865271172 0.14202907121800329 0.0036274324739371615;-474.71739801176847 -52.639057176624455 -1.486115096147145;-18.983043182060019 1.2250341079639306 -9.9580733363523048;-19.043964438908777 1.1966590116130593 -9.9277346117788206];
bias{1} = [624.86335446371311;-664.22084656606933;0.74756932025969558;-0.58667458036322662;-0.70596734594927335;-0.00033708087635474731;0.76355977566746891;-0.56549676161204765;1.1386082208197239;-1182.1555857698727];
weights{1} = [-624.43211069916413 -23.124511579755389 -0.50932692530595924;663.81615028592432 24.534688472838333 0.55907115123810303;0.01764119529790344 -0.15262549104074274 -0.0015754218273751608;0.015018333845149464 -0.13011490934581399 -0.0013613483435603779;-0.336845484890827 -0.07229802238478808 0.0019654410329405627;-1.4021961341748677 -0.0012487626975661245 -1.4382471854454554e-05;0.14384882758645126 0.0301171274269386 -0.0010900758278010818;0.19119389821869009 0.72053610550463609 0.55393848826035008;0.41312127294790318 0.99757189353955267 0.6942128155741053;-1183.6738259414308 -44.898398244768615 -0.053394585017258037];

% Layer 2
%bias{2} = [-0.045559121587252449;-0.2896061276052494];
%weights{2} = [15.995715333231933 -9.6138507744048116 -10.606942942666747 0.15217617660496818 -5.4710757881499399 -2.8042503389596556 0.23819302644967494 -15.003098432729155 -0.43213684092726151 0.4313532961518996;-0.00035068875023050959 0.0010465002610465864 0.00099904864105787279 4.6602487566400299 2.9897813727486402 1.7590423571999283 5.6466272746290578 0.00037585158395185016 0.00087336594628367216 -0.00090042250802271277];
bias{2} = [-11.280929104554833;-0.43178049044584293];
weights{2} =  [-12.855692546102274 -13.850609219740988 0.76214935553064977 1.1333408633421864 3.9328794398195233 -0.0002882365974228327 21.453801854202606 0.018235803727317985 0.013002521487677282 1.0084953007784339;-0.0075248203814413384 -0.0072720217924831781 -4.5383780162301948 -6.2225422919581606 0.0096897959878230501 -0.12726667509803077 0.048585851826141911 0.00011240225592427912 4.3411290731915032e-05 -0.00023842810812357712];

model.iniWeights(weights);
model.iniBias(bias);


tmax = 2;
dt = 0.01;
t = 0:dt:tmax;
n = length(t);
ur = [-.7 .7]; 
    
% Initialize x
x(:,1) = [randn(1)*.4;randn(1)*.6];
xhat(:,1) = x(:,1);
u = rand(1,n-1) * (max(ur)-min(ur)) + min(ur);

for j = 1:n-1

    x(:,j+1) = Inverted_Pendulum(x(:,j),u(j), dt);
    x(1,j+1) = x(1,j+1) + 2*pi*[abs(x(1,j+1))>pi]*-sign(x(1,j+1));
    xn = mapminmax('apply',[x(:,j);u(:,j)],ptx);
    xhat(:,j+1) = model.FFwrd(xn);

end

% De-normalize data to compare results
xhat = mapminmax('reverse',xhat,pty);

% Plot results
figure()
subplot(2,1,1)
plot(t,x(1,:).*180/pi,t,xhat(1,:).*180/pi,'g-.');
legend('plant output','model output')
grid on
ylabel('\theta [deg]');
xlabel('time [s]');
subplot(2,1,2)
plot(t,x(2,:).*180/pi,t,xhat(2,:).*180/pi,'g-.');
legend('plant output','model output')
ylabel('$\dot{\theta}$ [deg/s]','interpreter','latex');
xlabel('time [s]');
%print('results','-dpng','-r300')

save('model_trqLimited2.mat','model','pty','ptx');

% end