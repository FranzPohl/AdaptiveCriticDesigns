clear; close all; clc;

load('Exp1DHP')
Rmat(1,:)    = Rlog;
mseCmat(1,:) = mseC;
mseAmat(1,:) = mseA;

load('Exp2DHP')
Rmat(2,:)    = Rlog;
mseCmat(2,:) = mseC;
mseAmat(2,:) = mseA;

load('Exp3DHP')
Rmat(3,:)    = Rlog;
mseCmat(3,:) = mseC;
mseAmat(3,:) = mseA;

load('Exp4DHP')
Rmat(4,:)    = Rlog;
mseCmat(4,:) = mseC;
mseAmat(4,:) = mseA;

load('Exp5DHP')
Rmat(5,:)    = Rlog;
mseCmat(5,:) = mseC;
mseAmat(5,:) = mseA;

load('Exp6DHP')
Rmat(6,:)    = Rlog;
mseCmat(6,:) = mseC;
mseAmat(6,:) = mseA;

load('Exp7DHP')
Rmat(7,:)    = Rlog;
mseCmat(7,:) = mseC;
mseAmat(7,:) = mseA;

load('Exp8DHP')
Rmat(8,:)    = Rlog;
mseCmat(8,:) = mseC;
mseAmat(8,:) = mseA;

load('Exp9DHP')
Rmat(9,:)    = Rlog;
mseCmat(9,:) = mseC;
mseAmat(9,:) = mseA;

load('Exp10DHP')
Rmat(10,:)    = Rlog;
mseCmat(10,:) = mseC;
mseAmat(10,:) = mseA;


x = 1:200;
Rmean = mean(Rmat,1);
% fR = fit(x',Rmean','poly3');
% pR = predint(fR,x,0.5,'observation','off');
mseCm = mean(mseCmat,1);
% fmC = fit(x',mseCm','poly9');
% pC = predint(fmC,x,0.5,'observation','off');
mseAm = mean(mseAmat,1);
% fmA = fit(x',mseAm','poly9');
% pA = predint(fmA,x,0.5,'observation','off');

Rvar  = std(Rmat,0,1);
mseCv = std(mseCmat,0,1);
mseAv = std(mseAmat,0,1);

%% Plots 

saveplots = true;

figure()
plot(Rmean,'r.')
grid on
hold on 
plot(Rmean+Rvar./3,'k-.')
plot(Rmean-Rvar./3,'k-.')
% plot(x, pR,'m--')
% plot(fR,x,Rmean);
xlabel('trials','FontSize',13);
ylabel('Cost','FontSize',13)
hold off
legend('avg Cost','sigma','Location','NorthEast')
if saveplots == true
    print('RDHP','-dpng','-r300');
end

figure()
plot(mseCm,'r.')
grid on
hold on 
plot(mseCm + mseCv,'k-.')
plot(mseCm - mseCv,'k-.')
% plot(x, pC,'m--')
grid on
% plot(fmC,x,mseCm);
xlabel('trials','FontSize',13);
ylabel('mse Critic','FontSize',13)
hold off
legend('avg mse','sigma','Location','NorthEast')
if saveplots == true
    print('mCDHP','-dpng','-r300');
end

figure()
plot(mseAm,'r.')
grid on
hold on 
plot(mseAm - mseAv,'k-.')
plot(mseAm + mseAv,'k-.')
% plot(x, pA,'m--')
grid on
% plot(fmA,x,mseAm);
xlabel('trials','FontSize',13);
ylabel('mse Actor','FontSize',13)
hold off
legend('avg mse','sigma','Location','NorthEast')
if saveplots == true
    print('mADHP','-dpng','-r300');
end
