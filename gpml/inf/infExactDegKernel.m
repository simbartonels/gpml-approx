function [post nlZ dnlZ] = infExactDegKernel(hyp, mean, basis_funcs, lik, x, y)

% Exact inference for a GP with Gaussian likelihood with a degenerate
% kernel k(x,y)=psi(x)*psi(y). The implementation follows "Gaussian
% Processes for Machine Learning" p.12 equation (2.12).

likstr = lik; if ~ischar(lik), likstr = func2str(lik); end 
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('Exact inference only possible with Gaussian likelihood');
end
cov1 = basis_funcs{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
if ~strcmp(cov1,'covDegenerate'); error('Only covDegenerate supported.'), end    % check cov

 
[n, D] = size(x);
if size(hyp.weight_prior, 2) > 1; error('Weight prior must be vector!'); end
SigmaInv = diag(1./hyp.weight_prior);
sn2 = exp(2*hyp.lik);                               % noise variance of likGauss
Phi = feval(basis_funcs{:}, hyp.cov, NaN, x);
if size(Phi, 1) > n; error('The feature space dimensionality is greater than the number of inputs!'); end
A = 1/sn2*(Phi*Phi')+SigmaInv;                      % evaluate covariance matrix
L = chol(A);
clear A;
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector
post.alpha = solve_chol(L, Phi*(y-m));
post.L = solve_chol(L, eye(size(L, 1)));%return inverse of A
post.sW = ones(size(L, 1),1)/sqrt(sn2);                  % sqrt of noise precision vector

if nargout>1                               % do we want the marginal likelihood?
  error('TODO: implement computation of marginal likelihood!');
  nlZ = (y-m)'*alpha/2 + sum(log(diag(L))) + n*log(2*pi*sn2)/2;  % -log marg lik
  if nargout>2                                         % do we want derivatives?
    error('Computing derivatives not supported!');
  end
end
