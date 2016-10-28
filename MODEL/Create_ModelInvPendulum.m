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

load('dataset.mat');  % training data set

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
bias{1} = [-367.33351786551094;-0.51739355500137585;-342.39193410067429;25.577106903167248;316.17684760489243;0.00049743204736269636;294.11915731725878;-0.00053130098792851565;0.79450845459630548;12.861912848388849];
weights{1} = [367.25349473578945 35.503500451405429 0.62535926915834672;0.086857296045550828 0.1130542034729148 0.0036413235498353813;342.28340870169819 33.170266901100774 0.59920302649660373;6.3611384780116627 20.52948439585424 1.6577565978553894;316.08988575414833 30.301953642028167 0.5145445854413675;-0.62995664584807931 -0.034519284631969883 -0.00030050724345323277;294.03654093543906 28.184514692659324 0.49119656816879698;0.7959649004195577 0.042682403562103607 0.00038262782712702781;0.12361908220904816 0.16095877953559504 0.0051866176747095115;11.364627378110862 0.63961614197580197 -2.3817056138880979];

% Layer 2
%bias{2} = [-0.045559121587252449;-0.2896061276052494];
%weights{2} = [15.995715333231933 -9.6138507744048116 -10.606942942666747 0.15217617660496818 -5.4710757881499399 -2.8042503389596556 0.23819302644967494 -15.003098432729155 -0.43213684092726151 0.4313532961518996;-0.00035068875023050959 0.0010465002610465864 0.00099904864105787279 4.6602487566400299 2.9897813727486402 1.7590423571999283 5.6466272746290578 0.00037585158395185016 0.00087336594628367216 -0.00090042250802271277];
bias{2} = [0.026648922816836854;1.1810581557897166];
weights{2} =  [-14.619159875665114 0.32652622164938538 13.627056783262827 0.043339805239560186 -11.708804182808121 -4.5480185343000992 10.713475118925569 -2.3804984227965678 0.12204528452486861 0.0088018341211691734;-0.00043429378803837206 7.7072093198668101 0.00046000900097539798 0.00013185306999927382 0.00012160228816902956 6.2665335703080443 -0.00011363799673001438 4.1057208474105531 3.7624900862983504 -1.6423356055191471e-06];

model.iniWeights(weights);
model.iniBias(bias);


tmax = 2;
dt = 0.01;
t = 0:dt:tmax;
n = length(t);
ur = [-.1 .1]; 
    
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

save('model.mat','model','pty','ptx');

% end