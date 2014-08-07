function K = degSS(s, hyp, z, i)

% Implementation of Sparse Spectrum GPR. 
%
% See also deg_covFunctions.M.

if nargin<3, K = '(2)'; return; end              % report number of parameters
[sz, D] = size(z);

ell = exp(hyp(1));                                 % characteristic length scale
sf2 = exp(2*hyp(2));                               % signal variance%j = 1:m;
m = size(s, 1);
K = zeros(sz, 2*m);
w = 2*pi*z*s';
K = [cos(w) sin(w)]';

if nargin>4                                                        % derivatives
    error('Optimization of hyperparameters not implemented.')
end