clc
clear
close all

%%%%%%%%%%%%%%%%%%%
%% LOAD PACKAGES %%
%%%%%%%%%%%%%%%%%%%

pkg load stk
addpath ../_utilities
warning('off','all')

%%%%%%%%%%%%%%
%% SETTINGS %%
%%%%%%%%%%%%%%

%% optimizer
Nbrute = 64;

%% plotting
clim_Yi = [0 10];
clim_Ui = [0 3];

%%%%%%%%%%%%%%%
%% COLORMAPS %%
%%%%%%%%%%%%%%%
y0 = viridis;
x0 = linspace(0,1,size(y0,1))';
x1 = linspace(0,1,20)';
colmap_y = interp1(x0,y0,x1);

y0 = [.5 1 .5 ; .5 0 0];
x0 = [0 ; 1];
colmap_u = interp1(x0,y0,x1);

%%%%%%%%%%%%%%%
%% LOAD DATA %%
%%%%%%%%%%%%%%%
disp('Loading data ...')

%% load covariates
disp('  |- Loading covariates ...')
x1_cov = csvread('../_data4example/cov_lon.csv');
x2_cov = csvread('../_data4example/cov_lat.csv');
mask = csvread('../_data4example/mask.csv');
tree = csvread('../_data4example/cov_tree.csv');
pop = csvread('../_data4example/cov_population.csv');
d2c = csvread('../_data4example/cov_dist2coast.csv');

%% load first-party data
disp('  |- Loading first-party data ...')
x1_1pd = csvread('../_data4example/1pd_lon.csv');
x2_1pd = csvread('../_data4example/1pd_lat.csv');
y_1pd = csvread('../_data4example/1pd_windspeed.csv');

%% load third-party data
disp('  |- Loading third-party data ...')
x1_3pd = csvread('../_data4example/3pd_lon.csv');
x2_3pd = csvread('../_data4example/3pd_lat.csv');
y_3pd = csvread('../_data4example/3pd_windspeed.csv');
    
%% plot raw data
disp('  |- Plotting station data ...')
figure()
colormap(colmap_y)
scatter(x1_3pd,x2_3pd,30,y_3pd,'o')
hold on
scatter(x1_1pd,x2_1pd,30,y_1pd,'s','filled')
hold off
set(gca,'clim',clim_Yi)
h = colorbar();
xlabel('longitude [deg]')
ylabel('latitude [deg]')
ylabel(h,'Hourly mean wind speed [m/s]')
legend('KNMI','WOW')
title('Data')
grid on
print('-dpng','figures/station_data.png','-S500,400')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SET UP REGRESSION INPUT %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('')
disp('Setting up regression input ...')

disp('  |- Concatenating station data ...')
x_1pd = [x1_1pd x2_1pd];
x_3pd = [x1_3pd x2_3pd];

x = [x_1pd ; x_3pd]; % code assumes 1pd comes before 3pd
y = [y_1pd ; y_3pd]; % code assumes 1pd comes before 3pd

N1 = length(y_1pd);
N2 = length(y_3pd);
N = N1+N2;

disp('  |- Concatenating output coordinates ...')
xi = [reshape(x1_cov,numel(x1_cov),1) reshape(x2_cov,numel(x2_cov),1)];
ni = size(xi,1);

disp('  |- Preparing covariates and proxies ...')
treestat = interp2(x1_cov,x2_cov,tree,x(:,1),x(:,2));
popstat = interp2(x1_cov,x2_cov,pop,x(:,1),x(:,2));
d2cstat = interp2(x1_cov,x2_cov,d2c,x(:,1),x(:,2));
treei = interp2(x1_cov,x2_cov,tree,xi(:,1),xi(:,2));
popi = interp2(x1_cov,x2_cov,pop,xi(:,1),xi(:,2));
d2ci = interp2(x1_cov,x2_cov,d2c,xi(:,1),xi(:,2));
noiserounding = [0.3 0.003];
noiseproxy = [[0*y_1pd ; 0*y_3pd+1] [0*y_1pd ; y_3pd]];
driftX = [ones(N,1) [zeros(N1,1);ones(N2,1)] d2cstat popstat treestat popstat.^2 treestat.^2];
driftXi = [ones(ni,1) zeros(ni,1) d2ci popi treei popi.^2 treei.^2];   

disp('  |- Preparing optimizer settings ...')
hyperinit = [nan .1 .1];
log10hypersearchrange = [.5 .5];
reldist = cosd(mean(mean(x2_cov)));
      
%%%%%%%%%%%%%%%%%%%%
%% RUN REGRESSION %%
%%%%%%%%%%%%%%%%%%%%
disp('')
disp('Running regression ...')
[yi,ui,theta,noisehyper,xvalRMSE,xvalQUANT,xvalRHrstd,xvalRUrms] = ...
   regressionkriging(x,y,noiserounding,noiseproxy,driftX,driftXi,xi,...
   hyperinit,log10hypersearchrange,Nbrute,N1,reldist);

disp('  |- Regression monitors:')   
disp(['      |- theta      = ' num2str(theta)])
disp(['      |- noisehyper = ' num2str(noisehyper)])
disp(['      |- xvalRMSE   = ' num2str(xvalRMSE)])
disp(['      |- xvalQUANT  = ' num2str(xvalQUANT)])
disp(['      |- xvalRHrstd = ' num2str(xvalRHrstd)])
disp(['      |- xvalRUrms  = ' num2str(xvalRUrms)])

%%%%%%%%%%%%%%%
%% PLOT MAPS %%
%%%%%%%%%%%%%%%
disp('')
disp('Plotting maps ...')

%% reshape map (vector to grid)
disp('  |- Reshaping gridded data ...') 
Yi = reshape(yi,size(x1_cov));
Ui = reshape(ui,size(x1_cov));
id = find(mask==0);
Yi(id) = nan;
Ui(id) = nan;

%% plot maps
disp('  |- Plotting mean ...') 
mx = mean(mean(x2_cov));
figure()
imagesc(x1_cov,x2_cov,flipud(Yi))
hold on
plot(x1_1pd,mx-(x2_1pd-mx),'ko','color',[.5 .5 .5])
hold off
grid on
colormap(colmap_y)
set(gca,'clim',clim_Yi)
h = colorbar();
ylabel(h,'Hourly mean wind speed [m/s]')
xticks([]), yticks([])
print('-dpng','figures/posterior_mean.png','-S500,400')

disp('  |- Plotting uncertainty ...') 
mx = mean(mean(x2_cov));
figure()
imagesc(x1_cov,x2_cov,flipud(Yi))
hold on
plot(x1_1pd,mx-(x2_1pd-mx),'ko','color',[.5 .5 .5])
hold off
grid on
colormap(colmap_u)
set(gca,'clim',clim_Ui)
h = colorbar();
ylabel(h,'Uncertainty [m/s]')
xticks([]), yticks([])
print('-dpng','figures/posterior_std.png','-S500,400')
      

