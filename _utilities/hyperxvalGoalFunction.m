function out = hyperxvalGoalFunction(loghyper,y,noiserounding,H,driftX,noiseproxy,N1)
  
    N = size(y,1); 
    theta = 10.^(loghyper(1));
    if length(loghyper)>1
      noisehyper = 10.^(loghyper(2:end));
      noise = sqrt( (noiseproxy.^2)*(noisehyper.^2)' );
    else
      noise = noiseproxy;
    endif
    noiserounding = [noiserounding(1)*ones(N1,1);noiserounding(2)*ones(N-N1,1)];
    noise = sqrt(noise.^2 + noiserounding.^2) + 0.00001;
    P = exp(-0.5*(H.^2)/(theta^2));
    R = diag(noise.^2);
    A = P+R;
    c = (driftX'*(A\driftX))\(driftX'*(A\y));
    drift = driftX*c;
    %out = log(det(A)) + (y-drift)'*(A\(y-drift));
    %out = sum(log(eigs(A))) + (y-drift)'*(A\(y-drift));
    
    E = nan(N1,1);
    b = exp(-0.5*(H.^2)/(theta^2));
    id0 = 1:N1;
    yixval = nan(N1,1);
    for k = 1:N1
      idout = k;
      idin = find(id0~=k);
      idin = [idin (N1+1):length(y)];
      %c = (driftX(idin,:)'*(A(idin,idin)\driftX(idin,:)))\(driftX(idin,:)'*(A(idin,idin)\y(idin,:)));
      %drift = driftX*c;
      yixval(k) = drift(idout) + b(idout,idin)*(A(idin,idin)\(y(idin)-drift(idin)));
      E(k) = yixval(k) - y(idout);
    endfor
    E = sort(E);
    E = E(2:end-1);
    out = sqrt(mean(E.^2));
    %figure()
    %plot(y(1:N1),yixval,'bo',y(1:N1),y(1:N1),'r-')
    return
endfunction
