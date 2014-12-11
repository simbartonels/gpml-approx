function [post nlZ dnlZ] = infExactDegKernel(hyp, mean, degCov, lik, x, y)

% Exact inference for a GP with Gaussian likelihood with a degenerate
% kernel k(x,y)=psi(x)*psi(y). The implementation follows "Gaussian
% Processes for Machine Learning" p.12 equation (2.12).

likstr = lik; if ~ischar(lik), likstr = func2str(lik); end 
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('Exact inference only possible with Gaussian likelihood');
end
cov1 = degCov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
if ~strcmp(cov1,'covDegenerate'); error('Only covDegenerate supported.'), end    % check cov

 
[n, D] = size(x);
weight_prior = feval(degCov{:}, hyp.cov);
if size(weight_prior, 2) > 1; error('Weight prior must be column vector!'); end
%TODO: it's probably better to use repmat where possible!
%SigmaInv = diag(1./weight_prior);
sigmainv = 1./weight_prior;
sn2 = exp(2*hyp.lik);                               % noise variance of likGauss
%by convention the third argument is []. See covDegenerate.m
Phi = feval(degCov{:}, hyp.cov, [], x);
if size(Phi, 1) > n; error('The feature space dimensionality is greater than the number of inputs!'); end
A = Phi*Phi'; %+sn2*SigmaInv;                      % evaluate covariance matrix
diagidx = logical(eye(size(Phi, 1)));
A(diagidx) = A(diagidx) + sn2*sigmainv;
L = chol(A);
clear A;
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector
Phiy = Phi*(y-m);
post.alpha = solve_chol(L, Phiy);
post.L = solve_chol(L, eye(size(L, 1)))*sn2;%return inverse of A
post.sW = [];
%clear L;

if nargout>1                               % do we want the marginal likelihood?
  %the following formula is taken from Solin (hopefully this is general
  %enough!)
%   Qhat = Phi' * diag(hyp.weight_prior) * Phi + sn2*eye(n);
%   Lhat = chol(Qhat);
%   clear Qhat;
%   nlZ = (y-m)'*solve_chol(Lhat, (y-m))/2 + sum(log(diag(Lhat))) + n*log(2*pi)/2;  % -log marg lik
  
  %this formula is more efficient using Woodbury formula and determinant
  %lemma
  M = L'\Phiy;
  logdetA = 2*sum(log(diag(L)));
  %M'*M = Phiy'*alpha (?)
  %M'*M is Matrix-Matrix-multiplication!!!
  yyMM = ((y-m)'*(y-m)-M'*M)/sn2;
  nlZ = yyMM + logdetA ...
      +sum(log(weight_prior))+n*log(2*pi)+(n-size(L, 1))*(2*hyp.lik);
  nlZ = nlZ/2;
  if nargout>2                                         % do we want derivatives?
    dnlZ = hyp;
    invAPhiy = post.alpha;
    for i = 1:numel(hyp.cov)
        dsigma = feval(degCov{:}, hyp.cov, [], [], i);
        dPhi = feval(degCov{:}, hyp.cov, [], x, i);
        dA = dPhi * Phi' + Phi * dPhi';% - sn2 * SigmaInv * diag(dSigma) * SigmaInv;
        dA(diagidx) = dA(diagidx) - sn2 * sigmainv .* dsigma .* sigmainv;
        dlogdetA = trace(solve_chol(L, dA)); % * detA / detA
        %dlogdetSigma = trace(SigmaInv * diag(dSigma));
        dlogdetSigma = sum(sigmainv .* dsigma);
        %dMM = 2*invAPhiy'*dPhi*y - invAPhiy' * dA * invAPhiy;
        dMM = invAPhiy'*(2*dPhi*y - dA * invAPhiy);
        dnlZ.cov(i) = (-dMM/sn2 + dlogdetA + dlogdetSigma)/2;
    end
    % TODO: can this be more efficient?
    SigmaInv = diag(sigmainv);
    dnlZ.lik = -2*yyMM + ...
       2*invAPhiy'*SigmaInv*invAPhiy + ...
       trace(solve_chol(L, 2*sn2*SigmaInv)) + 2*(n-size(L, 1));
    dnlZ.lik = dnlZ.lik / 2;
    %dnlZ.mean
  end
end
