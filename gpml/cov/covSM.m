function K = covSM(M, hyp, x, z, i)

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

if nargin<2, K = sprintf('(2*%d*D+D+1)', M); return; end % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,D] = size(x);
%TODO: make sure basis points have same dimension as x
%TODO: make somehow sure basis points are the same as in infSM
logll = hyp(1:D);                               % characteristic length scale
actsf2 = exp(2*(hyp(2*M*D+D+1)-(log(2*pi)*D+sum(logll))/2));
if dg                                                               % vector kxx
  K = actsf2*ones(n ,1);
else
  if xeqz                                                 % symmetric matrix Kxx
    error('This covariance function is not meant to be used to compute covariance matrices!')
  else                                                   % cross covariances Kxz
    K = zeros(M, size(z, 1));
    sigma = hyp(D+1:M*D+D);
    sigma = reshape(sigma, [M, D]);
    logP = -(D*log(2*pi)+sum(sigma, 2))/2;
    V = hyp(M*D+D+1:2*M*D+D);
    V = reshape(V, [M, D]);
    %TODO: make this more efficient
    for j=1:M
        K(j, :) = covSEard([sigma(j, :), logP(j)], z, V(j, :));
    end
  end
end
if nargin>4                                                   % derivatives
    error('Optimization of hyperparameters not implemented.')
    
end