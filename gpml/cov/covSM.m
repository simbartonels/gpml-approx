function K = covSM(v, logsigma, hyp, x, z, i)

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

if nargin<4, K = covSEard(); return; end              % report number of parameters
if nargin<5, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,D] = size(x);
%TODO: make sure basis points have same dimension as x
%TODO: make somehow sure basis points are the same as in infSM
m = size(v, 1);
sigma = logsigma; %exp(logsigma);
sf2 = exp(2*hyp(D+1));                                         % signal variance
% precompute squared distances
if dg                                                               % vector kxx
  K = sf2*ones(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    error('This covariance function is not meant to be used to compute covariance matrices!')
  else                                                   % cross covariances Kxz
    K = zeros(m, size(z, 1));
    %TODO: make this more efficient
    for i=1:m
        K(i, :) = covSEard([sigma(i, :), 0], z, v(i, :));
    end
  end
end                                             % covariance
if nargin>5                                                   % derivatives
    error('Optimization of hyperparameters not implemented.')
end