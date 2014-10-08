function K = degFastFood(s, gpi, b, hyp, z_org, di)

% Implementation of Fast Food.
% Args:
% s - the (diagonal) scale matrix (as vector)
% gpi - the product of G and Pi (also a diagonal therefore as vector)
% b - the (diagonal) random binary matrix (as vector)
% Hyper-parameters:
% [log(signal variance); log(lengthscale);]
% 
% See also deg_covFunctions.M.

if nargin<4, K = '(2)'; return; end              % report number of parameters
sf2 = exp(2*hyp(1));                               % signal variance
mD = size(s, 1);
if nargin == 4
   % return weight prior
   K = sf2*ones(2*mD, 1)/mD;
   return;
elseif nargin == 5
    % compute phi(z)
    W = multiplyW(z_org, mD, s, gpi, b, hyp);
    K = [cos(W); sin(W)];
elseif nargin==6                                              % derivatives
    %error('Optimization of hyperparameters not implemented.')
    if isempty(z_org)
        % compute derivative of weight prior
        if di == 1
            K = 2 * sf2 * ones(2*mD, 1)/mD;
        else
            K = zeros(2*mD, 1);
        end
    else
        % derivatives of phi(z)
        if di == 2
            W = multiplyW(z_org, mD, s, gpi, b, hyp);
            K = [(sin(W).*W); (-cos(W).*W)];
        else
            K = zeros([2*mD, size(z_org, 1)]);
        end
    end
end
end

function W = multiplyW(z_org, mD, s, gpi, b, hyp)
    [sz, d] = size(z_org);
    D = 2^nextpow2(d);
    ell = exp(hyp(2));
    %TODO: might be this padding is not necessary since fwht does it by itself!
    z = [z_org/ell zeros(sz, D-d)];
    m = mD/D;
    W = zeros(mD, sz);
    for j = 1:m
        idx = (1+(j-1)*D):(j*D);
        %it appears the hadamard transform in matlab is scaled
        w = fwht(diag(b(idx))*z', D, 'hadamard')*D;
        %w = hadamard(D) * diag(b(idx))* z';
        w = fwht(diag(gpi(idx))*w, D, 'hadamard')*D;
        %w = hadamard(D) * diag(gpi(idx))*w;
        W(idx, :) = diag(s(idx))*w;
        %W(idx, :) = diag(s(idx)./ell)*w;
    end
end