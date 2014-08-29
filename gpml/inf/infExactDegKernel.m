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
if size(hyp.weight_prior, 2) > 1; error('Weight prior must be column vector!'); end
SigmaInv = diag(1./hyp.weight_prior);
sn2 = exp(2*hyp.lik);                               % noise variance of likGauss
%by convention the third argument is NaN. See covDegenerate.m
Phi = feval(degCov{:}, hyp.cov, NaN, x);
if size(Phi, 1) > n; error('The feature space dimensionality is greater than the number of inputs!'); end
A = 1/sn2*(Phi*Phi')+SigmaInv;                      % evaluate covariance matrix
L = chol(A);
clear A;
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector
Phiy = Phi*(y-m);
post.alpha = solve_chol(L, Phiy) / sn2;
post.L = solve_chol(L, eye(size(L, 1)));%return inverse of A
post.sW = ones(size(L, 1),1)/sqrt(sn2);                  % sqrt of noise precision vector
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
  nlZ = ((y-m)'*(y-m)-M'*M/sn2)/sn2 +2*sum(log(diag(L))) ...
      +sum(log(hyp.weight_prior))+n*(log(sn2)+log(2*pi));
  nlZ = nlZ/2;
  if nargout>2                                         % do we want derivatives?
    
    error('Computing derivatives not supported!');
  end
end
