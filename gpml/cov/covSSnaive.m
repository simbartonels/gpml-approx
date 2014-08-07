function K = covSSnaive(hyp, x, z, i)

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

if nargin<2, K = '(m*D+1)'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,D] = size(x);
m = (length(hyp)-1)/D;
w = 2*pi*reshape(hyp(1:m*D), [m, D])                                 %frequencies
sf2 = exp(2*hyp(m*D+1));                                         % signal variance

% precompute squared distances
if dg                                                               % vector kxx
  K = ones(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    phi = x*w';
    %multiplication with 2*pi already happend in creation of w
    phi = [cos(phi) sin(phi)];
    K = phi*phi'/m;
    %{
    %check against alternative formulation
    for i = 1:n
        for j = i:n
            kij = sum(cos(2*pi*w*((x(i,:)-x(j,:))')))/m;
            if abs(kij - K(i,j)) > 1^-15
                error('Error in computation of kernel matrix')
            end
        end
    end
    %}
  else                                                   % cross covariances Kxz
    phix = x*w';
    %multiplication with 2*pi already happend in creation of w
    phix = [cos(phix) sin(phix)];
    phiz = z*w';
    phiz = [cos(phiz) sin(phiz)];
    K = phix*phiz'/m;
    %{
    %check against alternative formulation
    for i = 1:n
        for j = size(z, 1)
            kij = sum(cos(2*pi*w*((x(i,:)-z(j,:))')))/m;
            if abs(kij - K(i,j)) > 1^-15
                error('Error in computation of cross covariance matrix')
            end
        end
    end
    %}
  end
end
K = sf2*K;                                                  % covariance
if nargin>3                                                        % derivatives
    error('Optimization of hyperparameters not implemented.')
end