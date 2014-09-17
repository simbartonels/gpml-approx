function K = degHSM(M, L, J, lambda, hyp, z, di)
if nargin<5, K = '(2)'; return; end              % report number of parameters
D = size(L, 1);
if nargin==5
   %return weight prior
   K = getWeightPrior(lambda, M, D, hyp);
   return;
elseif nargin==6
    sz = size(z, 1);
    K = zeros(M^D, sz);
    xMinusAoverBMinusA = (z-repmat(L, sz, 1))./repmat(2*L, sz, 1);
    for k=1:M^D
        j = J(:, k)';
        K(k, :) = (prod(sqrt(1./L), 2) * prod(sin(pi * repmat(j, sz, 1) .* xMinusAoverBMinusA), 2))';
    end
elseif nargin==7                                                        % derivatives
    %error('Optimization of hyperparameters not implemented.')
    if isempty(z)
        %gradients of the weight prior
        if di == 1
            ls2 = exp(2*hyp(1));
            K = getWeightPrior(lambda, M, D, hyp).*(D-ls2*lambda);
        elseif di == 2
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
    ls = exp(hyp(1));
    sf = exp(hyp(2));
    K = zeros(M^D, 1);
    for k = 1:M^D
         K(k) = spectralDensity(lambda(k), D, ls, sf);
    end
end

function s = spectralDensity(rSqrd, D, ls, sf)
    %see Rasmussen p.154 above (7.11)
    s = sf*sqrt(2*pi)*(ls^D)*exp(-ls^2*rSqrd/2);
end