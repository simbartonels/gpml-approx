function K = covHSMnaive(s, a, b, d, hyp, x_full, z_full, i)

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
z = z_full;
if nargin<7, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

x = x_full(:, d);
n = size(x, 1);
m = size(s, 1);
%sf2 = exp(2*hyp(1));                                         % signal variance

phij = @(inp) pi*sum((inp-repmat(a, size(inp, 1), 1))./repmat(b-a, size(inp, 1), 1), 2);
j = 1:m;
%phiall = @(inp) phij(inp, j);
phiall = @(inp) sin(repmat(phij(inp), 1, m).*repmat(j, size(inp, 1), 1))';

%phiDN = (x-repmat(a, 1, n)')./repmat(b-a, 1, n)';
%phiN = sum(phiDN, 2);
%matrix below must be NxM, i.e. phi_ij = phi_j(x_i)
%phi = sin(pi*repmat(j, n, 1).*repmat(phiN, 1, m));
phi = phiall(x);
if dg                                                               % vector kxx
  K = 2*diag(phi'*(phi.*repmat(s, 1, n)))/(b-a);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = 2*phi'*(phi.*repmat(s, 1, n))/(b-a);
  else                                                   % cross covariances Kxz
    %phiDN = (z-repmat(a, 1, sz)')./repmat(b-a, 1, sz)';
    %phiN = sum(phiDN, 2);
    %j = 1:m;
    %phiz = sin(pi*repmat(j, sz, 1).*repmat(phiN, 1, m));
    %phi = phiall(x);
    z = z_full(:, d);
    phiz = phiall(z);
    K = 2*phiz'*(phi.*repmat(s, 1, n))/(b-a);
    K = K';
  end
end
%K = sf2*K; % covariance
if nargin>7                                                        % derivatives
    error('Optimization of hyperparameters not implemented.')
end