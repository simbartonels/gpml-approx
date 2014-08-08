function K = degFastFood(s, gpi, b, hyp, z_org, i)

% Implementation of Fast Food.
%
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
W = zeros(sz, mD);
for j = 1:m
    idx = (1+(j-1)*D):(j*D);
    w = fwht(z*diag(b(idx)));
    w = fwht(w*diag(gpi(idx)));
    W(:, idx) = w*diag(s(idx));
end
K = [cos(W) sin(W)]';

if nargin>5                                                        % derivatives
    error('Optimization of hyperparameters not implemented.')
end
end