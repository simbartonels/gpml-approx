function K = degFastFood(s, g, randpi, b, hyp, z_org, di)

% Implementation of Fast Food.
% Args:
% s - the (diagonal) scale matrix (as vector)
% g - the (diagonal) Gaussian random matrix (as vector)
% randpi - the permutation matrix (as vector)
% b - the (diagonal) random binary matrix (as vector)
% Hyper-parameters:
% [log(signal variance); log(lengthscale1); ...; log(lengthscaleD)]
% 
% See also deg_covFunctions.M.

if nargin<meta_args(), K = '(D+1)'; return; end              % report number of parameters
sf2 = exp(2*hyp(1));                               % signal variance
mD = size(s, 1);
if nargin == meta_args()
   % return weight prior
   K = sf2*ones(2*mD, 1)/mD;
   return;
elseif nargin == meta_args()+1
    % compute phi(z)
    W = multiplyW(z_org, mD, s, g, randpi, b, hyp);
    %TODO: W is padded!
    K = [cos(W); sin(W)];
elseif nargin==meta_args()+2                                             % derivatives
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
        if di >= 2
            W = multiplyW(z_org, mD, s, g, randpi, b, hyp);
            z2 = zeros(size(z_org));
            z2(:, di-1) = z_org(:, di-1);
            W2 = multiplyW(z2, mD, s, g, randpi, b, hyp);
            K = [(sin(W).*W2); (-cos(W).*W2)];
        else
            K = zeros([2*mD, size(z_org, 1)]);
        end
    end
end
end

function N = meta_args()
    N = 5;
end

function W = multiplyW(z_org, mD, s, g, randpi, b, hyp)
    [sz, d] = size(z_org);
    D = 2^nextpow2(d);
    ell = [exp(hyp(2:d+1)); ones([D-d, 1])];
    %TODO: might be this padding is not necessary since fwht does it by itself!
    z = [z_org zeros(sz, D-d)];
    %somehow it won't give the same result when the division is applied
    %later. This works better.
    z = z./repmat(ell', [sz, 1]);
    m = mD/D;
    W = zeros(mD, sz);
    for j = 1:m
        idx = (1+(j-1)*D):(j*D);
        %it appears the hadamard transform in matlab is scaled
        w = fwht(diag(b(idx))*z', D, 'hadamard')*D;
        %w = hadamard(D) * diag(b(idx))* z';
        P = eye(D);
        P = P(randpi(idx), :);
        w = fwht(diag(g(idx))*P*w , D, 'hadamard')*D;
        %w = hadamard(D) * diag(gpi(idx))*w;
        W(idx, :) = diag(s(idx))*w;
        %W(idx, :) = diag(s(idx)./ell)*w;
    end
end