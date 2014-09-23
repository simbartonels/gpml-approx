function K = covHSMnaive(s, a, b, d, hyp, x_full, z, i)

% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sqrt(sf2)) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% See also COVFUNCTIONS.M.

if nargin<6, K = '(0)'; return; end              % report number of parameters
if nargin<7, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

x = x_full(:, d);
n = size(x, 1);
m = size(s, 1);
phij = @(inp) pi*sum((inp-repmat(a, size(inp, 1), 1))./repmat(b-a, size(inp, 1), 1), 2);
j = 1:m;
phiall = @(inp) sin(repmat(phij(inp), 1, m).*repmat(j, size(inp, 1), 1))';
phi = phiall(x);
if dg                                                               % vector kxx
  K = 2*diag(phi'*(phi.*repmat(s, 1, n)))/(b-a);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = 2*phi'*(phi.*repmat(s, 1, n))/(b-a);
  else                                                   % cross covariances Kxz
    phiz = phiall(z(:, d));
    K = 2*phiz'*(phi.*repmat(s, 1, n))/(b-a);
    K = K';
  end
end
if nargin>7                                                        % derivatives
    error('Optimization of hyperparameters not implemented.')
end