function K = degHSM2(M, L, J, lambda, hyp, z, di)
% DEGHSM Degenerate covariance function proposed in "Hilbert Space Methods
% for Reduced Rank Gaussian Process Regression" by Solin and Särkkä in
% 2014 augmented with automatic relevance determination (ARD).
% In contrast to degHSM ARD is applied in a less naive way and more
% efficient.
% M - the number of basis functions
% L - a vector describing the range of the inputs, i.e. x in [-L, L]
% J - an index matrix created in initHSM.
% lambda - the eigenvalues of the kernel corresponding to the negative 
%   LaPlace operator
% Hyperparameters are the same as for covSEard and in the same order.

if nargin<5, K = '(D+1)'; return; end              % report number of parameters
D = size(L, 2);
if nargin==5
   %return weight prior
   K = getWeightPrior(J, M, D, L, hyp);
   return;
elseif nargin==6
    ls = exp(hyp(1:D));
    K = computePhi(z, D, M, ls, L);
elseif nargin==7                                                        % derivatives
    %error('Optimization of hyperparameters not implemented.')
    if isempty(z)
        %gradients of the weight prior
        if di <= D
            K = getWeightPriorLengthScaleGradient(J, M, D, L, hyp, di);
        elseif di == D+1
            %the derivative is just the weight prior itself
            K = getWeightPrior(J, M, D, L, hyp);
        else
            error('Unknown hyper-parameter!');
        end
    else
        sz = size(z, 1);
        K = zeros(M^D, sz); 
    end
end
end

function K = computePhi(z, D, M, ls, L)
    sz = size(z, 1);
    Phi = zeros(D, M, sz);
    m = 1:M;
    m = m';
    m = pi*m/2;
    %computing the eigenfunction values for D=1
    for d = 1:D
        Phi(d, :, :) = 1/sqrt(L(d)) * sin( m*(z(:, d)'+L(d))/L(d) );
    end
    
    K = kronPower(Phi);
end

function K = kronPower(Phi)
%KRONPOWER Computes the kronecker power of Phi to the D.
    %in the multidimensional case we need to mutiply all combinations
    % we can't use squeeze here since it messes things up in special cases
    % where e.g. M=1=D
    sz = size(Phi, 3);
    M = size(Phi, 2);
    D = size(Phi, 1);
    temp = reshape(Phi(1, :, :), [M, sz]);
    Md = M;
    for d = 2:D
        t2 = zeros(Md*M, sz);
        for m = 1:M
            idx = (m-1)*Md+(1:Md);
            t2(idx, :) = temp * diag(squeeze(Phi(d, m, :)));
        end
        temp = t2;
        Md = Md * M;
    end
    K = temp;
end


function K = getWeightPrior(J, M, D, L, hyp)
    sf = exp(hyp(D+1));
    ell = exp(2 * hyp(1:D));
    K = zeros(M^D, 1);
    for k = 1:M^D
        lambda = pi^2*((J(:, k)'./(2*L)).^2.*ell');
        K(k) = spectralDensity(sum(lambda), D, sf, ell);
    end
end

function K = getWeightPriorLengthScaleGradient(J, M, D, L, hyp, di)
    sf = exp(hyp(D+1));
    ell = exp(2 * hyp(1:D));
    K = zeros(M^D, 1);
    for k = 1:M^D
        lambda = pi^2*((J(:, k)'./(2*L)).^2.*ell');
        K(k) = spectralDensity(sum(lambda), D, sf, ell) * (1 - pi^2*((J(di, k)/(2*L(di)))^2)*ell(di));
    end
end

function s = spectralDensity(rSqrd, D, sf, ell)
    %see Rasmussen p.154 above (7.11)
    s = sf*sqrt((2*pi)^D*prod(ell))*exp(-rSqrd/2);
end