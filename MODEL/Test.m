%% PART 0 INVERTED PENDULUM SIMULATION
clear; close all; clc;

% Add libraries
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../PLANT')); 

tmax= 8;
dt = 0.01;
t = 0:dt:tmax;
n = length(t);

% x = randn(2,1);
x = [pi;0];
u = +.5;

ipgraph = IPGraphics(x(1));

for i = 1:n-1
    x(:,i+1) = Inverted_Pendulum(x(:,i),u,dt);
%     if mod(i,5)==0
        ipgraph.update(x(1,i+1));
%     end
end

%% PART I FEED FORWARD COMPARISON
clear; close all; clc;

% Get Class
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 

[ts,ys] = simplefit_dataset;
data = [ts; ys];
n = length(data);
shuffle = randperm(n);
data = data(:,shuffle);
dataIn = data(1,:);
dataOu = data(2,:);


% Neural Network Toolbox
net1 = feedforwardnet(10);
net1 = configure(net1,dataIn,dataOu);
net1 = train(net1,dataIn,dataOu);
YNN1 = net1(ts);

% Neural Network 
struct =  [1 10 1];
net = NeuralNet(struct);

weights{1} = [11.892152117723265;-6.9491554918877316;-5.8696658861203774;-11.800564905621393;-12.696374882544227;8.810725795174708;-5.7449272152027486;-6.6511164262680182;-7.2315323336443313;-11.230677084091271];
weights{2} = [0.13685517727374838 -0.6266500108564097 1.0386510810092393 0.063350539572816778 -0.087224742517914344 0.095486161539514766 0.15877936142687107 0.2197583413058404 -0.51004508245133129 -0.35947826047823833];

bias{1} = [-10.809945852111845;5.5577481031989251;2.9596995671480455;4.6079429955113138;2.2941699034697254;-0.32814959641544389;-2.4594303738464527;-3.9448722294214846;-6.4753843894138727;-11.376702958239747];
bias{2} = -0.24827674423839891;

net.iniWeights(weights);
net.iniBias(bias);

% Normalization of the Input Data Samples
% x1_step1_xoffset = 0;
% x1_step1_gain = 0.200475452649894;
% x1_step1_ymin = -1;
% 
% xnn = bsxfun(@minus,ts,x1_step1_xoffset);
% xnn = bsxfun(@times,xnn,x1_step1_gain);
% tsNormalized = bsxfun(@plus,xnn,x1_step1_ymin);
[tsNormalized,ps] = mapminmax(ts,-1,1);

% De-normalization of output samples
yNNtmp = net.FFwrd(tsNormalized);
% y1_step1_ymin = -1;
% y1_step1_gain = 0.2;
% y1_step1_xoffset = 0;
% 
% x = bsxfun(@minus,yNNtmp,y1_step1_ymin);
% x = bsxfun(@rdivide,x,y1_step1_gain);
% YNN = bsxfun(@plus,x,y1_step1_xoffset);
YNN = mapminmax('reverse',yNNtmp,ps);

% Plots
figure()
plot(ts,YNN1,'r-');
hold on;
plot(ts,ys,'b-.')%,ts,YNN,'w-');
legend('nntool','plant', 'NeuralNet')
grid on;
hold off;



%% PART II TRAINING WITH NORMALIZATION
clear; close all; clc;

% Get Class
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 

% Data min-max mapping
[ts,ys] = simplefit_dataset;
[tsn,ps1] = mapminmax(ts);
[ysn, ps] = mapminmax(ys); 
data = [tsn; ysn];
n = length(data);

% shuffle data
shuffle = randperm(n);
data = data(:,shuffle);

% split data
ptrain = 0.6;
train_data = data(:,1:round(ptrain*size(data,2)));
test_data  = data(:,round(ptrain*size(data,2))+1:end);

% Neural Network 
struct =  [1 10 1];
net = NeuralNet(struct);

% training
eta = 0.5;%0.5
mu = 0.6; %0.6
lamda = 0.003;
epochs = 600;
batch_size = 10;
[msee, mseetr] = net.SGD(train_data, batch_size, epochs, eta, mu, lamda, test_data);
y = net.FFwrd(tsn);
%Runtime 43sec

% Reverse min-max mapping
YNN = mapminmax('reverse',y,ps);

net.best_result
y = net.FFwrd(tsn);
YNN2 = mapminmax('reverse',y,ps);

% Plots
figure()
plot(ts,ys,'g-',ts,YNN,'b-.',ts,YNN2,'r-.');
legend('plant','NeuralNet','BestResult')
grid on;

% figure()
% plot(mse);
% hold on;
% plot(msetr); 
% legend('cost validation data','cost train data');

figure()
plot(msee);
hold on;
plot(mseetr);
legend('cost validation data','cost train data');
xlabel('epochs');
ylabel('cost');
grid on

% figure()
% subplot(4,1,1)
% plot(1:epochs, net.weightlog{1});
% ylabel('IW');
% subplot(4,1,2)
% plot(1:epochs,net.weightlog{2});
% ylabel('LW');
% subplot(4,1,3)
% plot(1:epochs,net.biaslog{1});
% ylabel('IB');
% subplot(4,1,4)
% plot(1:epochs,net.biaslog{2});
% ylabel('LB');

%% PART II TRAINING WITH NORMALIZATION
clear; close all; clc;

% Get Class
mfilepath = fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ANN')); 
addpath(fullfile(mfilepath,'../PLANT')); 

load('test_data.mat');
load('train_data.mat');

[train_data(1:3,:), ptr1] = mapminmax(train_data(1:3,:));
[train_data(4:5,:), ptr2] = mapminmax(train_data(4:5,:));
[test_data(1:3,:), pt1]   = mapminmax('apply',test_data(1:3,:), ptr1);
[test_data(4:5,:), pt2]   = mapminmax('apply',test_data(4:5,:), ptr2);

% Neural Network 
struct =  [3 10 2];
net = NeuralNet(struct);

% training
eta = 0.3;
mu = 0.5; 
lamda = 0.01;
epochs = 50;
batch_size = 188;
[msee, mseetr] = net.SGD(train_data, batch_size, epochs, eta, mu, lamda, test_data);
net.best_result

%% Plots
figure()
plot(msee);
hold on;
plot(mseetr);
legend('cost validation data','cost train data');
xlabel('epochs');
ylabel('cost');
grid on
print('train','-dpng','-r300');

%% Simulation
tmax = 2;
dt   = 0.001;
t    = 0:dt:tmax;
n    = length(t);
ur   = [-10 10];
    
%initialize x
x(:,1) = [randn(1)*.4;randn(1)*.6];
xhat(:,1) = x(:,1);
u = rand(1,n-1) * (max(ur)-min(ur)) + min(ur);

for j = 1:n-1

    x(:,j+1) = Inverted_Pendulum(x(:,j),u(j),dt);
    x(1,j+1) = x(1,j+1) + 2*pi*[abs(x(1,j+1))>pi]*-sign(x(1,j+1));
    xn = mapminmax('apply',[x(:,j);u(:,j)],ptr1);
    xhat(:,j+1) = net.FFwrd(xn);

end

xhat = mapminmax('reverse',xhat,ptr2);
%%
figure()
subplot(2,1,1)
plot(t,x(1,:).*180/pi,t,xhat(1,:).*180/pi,'g-.');
legend('plant output','model output')
subplot(2,1,2)
plot(t,x(2,:).*180/pi,t,xhat(2,:).*180/pi,'g-.');
legend('plant output','model output')
print('results','-dpng','-r300')


    
