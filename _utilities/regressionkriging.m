function [yi,ui,theta,noisehyper,xvalRMSE,xvalQUANT,xvalRHrstd,xvalRUrms] = regressionkriging(x,y,noiserounding,noiseproxy,driftX,driftXi,xi,hyperinit,log10hypersearchrange,Nbrute,N1,reldist)
  
  %% REGRESSION KRIGING WITH BIAS AND NOISE BUDGET
  
  %% INPUTS
  % x: lon lat station locations
  % y: observed station values
  % noiserounding: rounding error in station values
  % noiseproxy: noise budget at station locations
  % driftX: bias budget and drift at station locations
  % driftXi: zero-bias-budget and drift at query locations (e.g. grid points)
  % xi: lon lat query locations (e.g. grid points)
  % hyperinit: initial hyperparameters
  % log10hypersearchrange
  % Nbrute: number of brute force iterations for hyperparameter optimisation
  % N1: number of high-fidelity points for x-validation (assumed to be at top of x & y vectors)
  % reldist: km 1 deg lat / km 1 deg lon
  
  %% OUTPUTS
  % yi: posterior mean
  % ui: posterior uncertainty
  % theta: optimised length scale
  % noisehyper: optimised noise budget weights
  % xvalRMSE: rms x-validation error (should be low for accurate yi)
  % xvalQUANT: quantile x-validation error (should be low for accurate yi)
  % xvalRHrstd: relative rank histogram standard deviation (should be low for accurate ui)
  
  %%%%%%%%%%%%%%%%%%%%
  %% LOCAL SETTINGS %%
  %%%%%%%%%%%%%%%%%%%%
  NmemorySave = 50;
  xvalquant = 0.8;
  
  %%%%%%%%%%%%%%%%%%%%
  %% NORMALIZE DATA %%
  %%%%%%%%%%%%%%%%%%%%
  y0 = y;
  sigma0 = std(y(1:N1));
  if sigma0 > 0
    y = y/sigma0;
    noiserounding = noiserounding/sigma0;
    noiseproxy = noiseproxy/sigma0;
    driftX = driftX/sigma0;
    driftXi = driftXi/sigma0;
  else
    sigma0 = 1;
  endif
  
  N = size(x,1);
  prefactor = sqrt(N/(N-size(driftX,2)-length(hyperinit)));

  %%%%%%%%%%%%%%%%%%
  %% COMPUTE LAGS %%
  %%%%%%%%%%%%%%%%%%
  % computing lags is computationally expensive, therefore they
  % are pre-computed and stored for later iterations
  N = size(x,1);
  ni = size(xi,1);
  ndim = size(x,2);
  H2 = zeros(N,N);
  Hi2 = zeros(ni,N);
  reldist = [reldist 1];
  for k = 1:ndim
    [X,Y] = meshgrid(x(:,k),x(:,k));
    H2 = H2 + (reldist(k)*(Y-X)).^2;
    [X,Y] = meshgrid(x(:,k),xi(:,k));
    Hi2 = Hi2 + (reldist(k)*(Y-X)).^2;
  endfor
  H = sqrt(H2);
  Hi = sqrt(Hi2);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% HYPERPARAMETER ESTIMATION %%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % we use brute force minization of the leave-one-out
  % cross-validation RMS error
  %
  % note: for research purposes it is interesting to run the optimizer with
  % a wide search range and a large number of iterations. however, this
  % is expensive and not very suitable for operations. an alternative would
  % be to have tailored values of hyperinit for different variables, and then
  % reduce the search range and number of brute force optimisations. this does
  % not require changes in the current function, it can be specified in the
  % settings when this function is called.
  
  % define goal function
  noisehyper = nan;
  goalfunction = @(loghyper) hyperxvalGoalFunction(loghyper,y,noiserounding,H,driftX,noiseproxy,N1);
  loghyperinit = log10(hyperinit);
  
  % compute median lag between 1PD stations, which is used as initial value if
  % nan is provided for the initial length scale
  if isnan(hyperinit(1))
    Hm = H(1:N1,1:N1);
    Hm(find(Hm==0)) = Inf;
    Hm = min(Hm,[],2);
    theta = median(Hm);
    loghyperinit(1) = log10(theta);
  endif
  
  if Nbrute > 0;
    
    % define search space for brute force optimisation
    searchrangetheta = log10hypersearchrange(1);
    searchrangenoise = log10hypersearchrange(2);
    nhyp = size(loghyperinit,2);
    v = rand('state');
    rand('state',0);
    %HYP = -1 + 2*rand(Nbrute,nhyp);
    doe = stk_sampling_maximinlhs(Nbrute,nhyp);
    HYP = -1 + 2*struct(doe).data;
    rand('state',v);
    for k = 1:nhyp
      if k == 1
        HYP(:,k) = loghyperinit(k) + searchrangetheta*HYP(:,k);
      else
        HYP(:,k) = loghyperinit(k) + searchrangenoise*HYP(:,k);
      endif
    endfor
    
    % iterate over search space to compute goal function
    GOAL = nan(Nbrute,1);
    for k = 1:Nbrute
      GOAL(k) = goalfunction(HYP(k,:));
    endfor
    
    % minimize goal function
    [~,idmin] = min(GOAL);
    loghyper0 = HYP(idmin,:);
  else
    % use Nelder-Mead Simplex minimizer if Nbrute is specified as zero
    % note: this does not really work well ...
    loghyper0 = fminsearch(goalfunction,loghyperinit);
  endif
  
  % unpack hyperparameters
  theta = 10.^(loghyper0(1));
  if length(loghyper0)>1
    noisehyper = 10.^(loghyper0(2:end));
    noise = sqrt( (noiseproxy.^2)*(noisehyper.^2)' );
  else
    noise = noiseproxy;
  endif
  noiserounding = [noiserounding(1)*ones(N1,1);noiserounding(2)*ones(N-N1,1)];
  noise = sqrt(noise.^2 + noiserounding.^2) + 0.00001;
  
  %%%%%%%%%%%%%%%%%%%%%%%%
  %% KRIGING PREDICTION %%
  %%%%%%%%%%%%%%%%%%%%%%%%
  
  % compute covariance matrices
  P = exp(-0.5*(H.^2)/(theta^2));
  R = diag(noise.^2);
  A = P+R;
  b = exp(-0.5*(Hi.^2)/(theta^2));
  
  % compute drift
  c = (driftX'*(A\driftX))\(driftX'*(A\y));
  drift = driftX*c;
  drifti = driftXi*c;
  
  % compute posterior mean
  yi = drifti + b*(A\(y-drift));
  
  % compute posterior variance (for uncertainty map)
  sigmaCorr = sqrt((y-drift)'*(A\(y-drift))/N);
  yi = sigma0*yi;
  M1 = driftX'*(A\driftX);
  M2 = driftXi'-driftX'*(A\b');
  ui = nan*yi;
  id0 = round(linspace(0,length(ui),NmemorySave+1));
  for k = 1:NmemorySave
    id1 = id0(k)+1;
    id2 = id0(k+1);
    ui(id1:id2) = prefactor*sigmaCorr*sqrt(1-diag(b(id1:id2,:)*(A\b(id1:id2,:)'))+diag(M2(:,id1:id2)'*(M1\M2(:,id1:id2))));
  endfor
  ui = sigma0*ui;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% X-VALIATION OF PREDICTION %%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % this can be used to check the prediction
  E = nan(N1,1);
  Er = nan(N1,1);
  b = exp(-0.5*(H.^2)/(theta^2));
  id0 = 1:N1;
  for k = 1:N1
    idout = k;
    idin = find(id0~=k);
    idin = [idin (N1+1):length(y)];
    c = (driftX(idin,:)'*(A(idin,idin)\driftX(idin,:)))\(driftX(idin,:)'*(A(idin,idin)\y(idin,:)));
    drift = driftX*c;
    yixval = drift(idout) + b(idout,idin)*(A(idin,idin)\(y(idin)-drift(idin)));
    M1 = driftX(idin,:)'*(A(idin,idin)\driftX(idin,:));
    M2 = driftX'-driftX'*(A\b');
    uixval = prefactor*sigmaCorr*sqrt(1-diag(b(idout,idin)*(A(idin,idin)\b(idout,idin)'))+diag(M2(:,idout)'*(M1\M2(:,idout))));
    E(k) = yixval - y(idout);
    Er(k) = (yixval-y(idout))/sqrt(uixval.^2+noiserounding(idout).^2);
  endfor
  Y0 = y0(1:N1);
  E00 = sigma0*E;
  nH(1) = sum(Er<=-.67);
  nH(2) = sum((Er>-.67).*(Er<=.0));
  nH(3) = sum((Er>.0).*(Er<=.67));
  nH(4) = sum(Er>.67);
  xvalRHrstd = std(nH)/std([N1 0 0 0]); % rank histogram relative std
  xvalRUrms = sqrt(mean(Er.^2)); % rank histogram relative std
  xvalQUANT = sigma0*quantile(abs(E),xvalquant); % xval quantile
  E = sort(E);
  E0 = E(2:end-1);
  xvalRMSE = sigma0*sqrt(mean(E.^2)); % xval RMS error