function [post, nlZ, dnlZ] = infSMfast(hyp, mean, cov, lik, x, y)
% INFFITCMEX Inference method for Sparse multiscale GPR. Equivalent to FITC
% using covSM.

likstr = lik; if ~ischar(lik), likstr = func2str(lik); end 
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('FITC inference only possible with Gaussian likelihood');
end
cov1 = cov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
if ~strcmp(cov1,'covSM'); error('Only covSM supported.'), end    % check cov
if ~(isempty(mean) || strcmp(func2str(mean{1}), 'meanZero'))
    mean
    error('Only zero-mean supported!');
end
M = cov{2};
post.sW = [];                                                  % unused for FITC
if nargout == 1
    [alpha, L] = infSMmex(M, unwrap(hyp), x, y);
elseif nargout == 2
    [alpha, L, nlZ] = infSMmex(M, unwrap(hyp), x, y);
else
    [alpha, L, nlZ, dnlZ] = infSMmex(M, unwrap(hyp), x, y);
end
post.alpha = alpha;                      % return the posterior parameters
post.L  = L; % Sigma-inv(Kuu)
%dead code to be removed
if nargout<0                                % do we want the marginal likelihood
  nlZ = sum(log(diag(Lu))) + (sum(log(dg)) + n*log(2*pi) + r'*r - be'*be)/2; 
  %= sum(log(diag(Svv/sqrt(sn2)))) +
  %(sum(log(Gamma*sn2))+...+y'(sn2*Gamma)^(-1)*y-beta'beta/sn2)/2
  % appears to be the same as in the Walder paper
  %CHALUPKA - free some memory or it's easy to run out on derivative computations.
  clear Lu Luu V;
  if nargout>2                                         % do we want derivatives?
    dnlZ = hyp;                                 % allocate space for derivatives
    W = Ku./repmat(sqrt(dg)',nu,1); 
    W = chol(Kuu+W*W'+snu2*eye(nu))'\Ku; % inv(K) = inv(G) - inv(G)*W'*W*inv(G);
    % = (Avv/sn2)^(-1/2)*Uvx
    al = (y-m - W'*(W*((y-m)./dg)))./dg;
    % = (y - Uvx'*(Avv/sn2)^(-1)*Uvx*Gamma^(-1)*y)*Gamma^(-1)
    B = iKuu*Ku; 
    % = Upsi^(-1)*Uvx
    clear Ku Kuu iKuu; %KRZ - also the line below moved from above.
    Wdg = W./repmat(dg',nu,1); w = B*al; 
    
    % w = Upsi^(-1)*Uvx*Gamma^(-1/2)*y - Upsi^(-1)*Uvx*Uvx'*v
    %KRZ - free more memory.
    for i = 1:numel(hyp.cov)
      [ddiagKi,dKuui,dKui] = feval(cov{:}, hyp.cov, x, [], i);  % eval cov deriv
      R = 2*dKui-dKuui*B; v = ddiagKi - sum(R.*B,1)';   % diag part of cov deriv
      % R = 2*dUvx-dUpsi*Upsi^(-1)*Uvx
      % v = dGamma?
      dnlZ.cov(i) = (ddiagKi'*(1./dg) +w'*(dKuui*w-2*(dKui*al)) -al'*(v.*al) ...
                         - sum(Wdg.*Wdg,1)*v - sum(sum((R*Wdg').*(B*Wdg'))) )/2;
      % = tr(dGAmma*Gamma^(-1)) + ...
    end 
    clear dKui; %KRZ
    dnlZ.lik = sn2*(sum(1./dg) -sum(sum(W.*W,1)'./(dg.*dg)) -al'*al);
    % since snu2 is a fixed fraction of sn2, there is a covariance-like term in
    % the derivative as well
    dKuui = 2*snu2; R = -dKuui*B; v = -sum(R.*B,1)';   % diag part of cov deriv
    dnlZ.lik = dnlZ.lik + (w'*dKuui*w -al'*(v.*al)...
                         - sum(Wdg.*Wdg,1)*v - sum(sum((R*Wdg').*(B*Wdg'))) )/2; 
    for i = 1:numel(hyp.mean)
      dnlZ.mean(i) = -feval(mean{:}, hyp.mean, x, i)'*al;
    end
  end
end
