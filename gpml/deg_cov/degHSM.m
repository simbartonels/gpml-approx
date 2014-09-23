function K = degHSM(M, L, J, lambda, hyp, z, di)
% DEGHSM Degenerate covariance function proposed in "Hilbert Space Methods
% for Reduced Rank Gaussian Process Regression" by Solin and Särkkä in
% 2014. 
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
   K = getWeightPrior(lambda, M, D, hyp);
   return;
elseif nargin==6
    sz = size(z, 1);
    ls = exp(hyp(1:D));
    z = z./repmat(ls'/LS(), [sz, 1]);
    K = zeros(M^D, sz);
    xMinusAoverBMinusA = (z+repmat(L, sz, 1))./repmat(2*L, sz, 1);
    for k=1:M^D
        j = J(:, k)';
        K(k, :) = (prod(sqrt(1./L), 2) * prod(sin(pi * repmat(j, sz, 1) .* xMinusAoverBMinusA), 2))';
    end
elseif nargin==7                                                        % derivatives
    %error('Optimization of hyperparameters not implemented.')
    if isempty(z)
        %gradients of the weight prior
        if di <= D
            error('To do: implement!');
        elseif di == D+1
            %the derivative is just the weight prior itself
            K = getWeightPrior(lambda, M, D, hyp);
        else
            error('Unknown hyper-parameter!');
        end
    else
        %basis function gradients
        sz = size(z, 1);
        K = zeros(M^D, sz);
    end
end
end

function K = getWeightPrior(lambda, M, D, hyp)
    sf = exp(hyp(D+1));
    K = zeros(M^D, 1);
    for k = 1:M^D
         K(k) = spectralDensity(lambda(k), D, sf);
    end
end

function s = spectralDensity(rSqrd, D, sf)
    %see Rasmussen p.154 above (7.11)
    s = sf*sqrt(2*pi)*(LS()^D)*exp(-LS()^2*rSqrd/2);
end

function retval = LS()
    retval = 0.1;
end