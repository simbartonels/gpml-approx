function K = degHSM(a, b, hyp, z, i)

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

if nargin<3, K = '(1)'; return; end              % report number of parameters
[sz, D] = size(z);
%TODO: Covariance function needs as hyper-parameter lenghscale and
%amplitude. m MUST NOT be a hyperparameter!
m = hyp(1);
%j = 1:m;
%phij = pi*sum((z-repmat(a, 1, sz)')./repmat(b-a, 1, sz)', 2);
%K = sqrt(2/(b-a))*sin(repmat(phij, 1, m).*repmat(j, sz, 1))';
K = zeros(m.^D, sz);

%create index matrix
J = ones(D, m.^D);
for d = 1:(D-1)
    J(d, :) = repmat(reshape(repmat((1:m)', 1, m.^(D-d))', [m.^(D-d+1), 1]), m.^(d-1), 1);
end
J(D, :) = repmat((1:m)', m.^(D-1), 1);

xMinusAoverBMinusA = (z-repmat(a, sz, 1))./repmat(b-a, sz, 1);
for k=1:m.^D
    j = J(:, k)';
    K(k, :) = (prod(sqrt(2./(b-a)), 2) * prod(sin(pi * repmat(j, sz, 1) .* xMinusAoverBMinusA), 2))';
%    K(idx, :) = sin(pi*(z-repmat(a, sz, 1))./repmat(b-a, sz, 1))*j;
end
%K = prod(sqrt(2./(b-a)), 2) * K;

%phij = pi*j.*(z-repmat(a, 1, sz)')./repmat(b-a, 1, sz)';
%K = prod(sin(repmat(phij, 1, m).*repmat(j, sz, 1)), 2)';


if nargin>4                                                        % derivatives
    error('Optimization of hyperparameters not implemented.')
end