function K = degFastFood(s, gpi, b, hyp, z_org, i)

% Implementation of Fast Food.
% Args:
% s - the (diagonal) scale matrix (as vector)
% gpi - the product of G and Pi (also a diagonal therefore as vector)
% b - the (diagonal) random binary matrix (as vector)
% See also deg_covFunctions.M.

if nargin<4, K = '(2)'; return; end              % report number of parameters
[sz, d] = size(z_org);
D = 2^nextpow2(d);
%TODO: might be this padding is not necessary since fwht does it by itself!
z = [z_org zeros(sz, D-d)];
mD = size(s, 1);
m = mD/D;
ell = exp(hyp(1));                                 % characteristic length scale
sf2 = exp(2*hyp(2));                               % signal variance%j = 1:m;
W = zeros(mD, sz);
for j = 1:m
    idx = (1+(j-1)*D):(j*D);
    w = fwht(diag(b(idx))*z');
    w = fwht(diag(gpi(idx))*w);
    W(idx, :) = diag(s(idx))*w;
end
% W2 = W;
% W = fwht(diag(b)*repmat(z', [m, 1]));
% W = fwht(diag(gpi)*W);
% W = diag(s) * W;
% W2 - W
K = [cos(W); sin(W)];

if nargin>5                                              % derivatives
    error('Optimization of hyperparameters not implemented.')
end
end