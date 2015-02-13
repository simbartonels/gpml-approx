function K = degHSM(M, L, J, lambda, hyp, z, di)
% DEGHSM Degenerate covariance function proposed in "Hilbert Space Methods
% for Reduced Rank Gaussian Process Regression" by Solin and Särkkä in
% 2014 augmented with automatic relevance determination.
% M - D^M number of basis functions
% L - a vector describing the range of the inputs, i.e. x in [-L, L]
% J - an index matrix created in initHSM.
% lambda - the eigenvalues of the kernel corresponding to the negative 
%   LaPlace operator
% Hyperparameters are the same as for covSEard and in the same order.
% In this implementation the length scales are applied on the inputs. Thus
% the gradients of the basis functions become non-trivial. Also the
% hyper-parameters are not quadratic.

%TODO: change to quadratic length scale!!!
if nargin<5, K = '(D+1)'; return; end              % report number of parameters
D = size(L, 2);
if nargin==5
   %return weight prior
   K = getWeightPrior(lambda, M, D, hyp);
   return;
elseif nargin==6
    ls = exp(hyp(1:D));
    K = computePhi(z, D, M, ls, L);
elseif nargin==7                                                        % derivatives
    %error('Optimization of hyperparameters not implemented.')
    if isempty(z)
        %gradients of the weight prior
        if di <= D
            %error('To do: implement!');
            K =  zeros([M^D, 1]);
        elseif di == D+1
            %the derivative is just the weight prior itself
            K = getWeightPrior(lambda, M, D, hyp);
        else
            error('Unknown hyper-parameter!');
        end
    else
        sz = size(z, 1);
        if di == D+1
            K = zeros(M^D, sz);
        elseif di <= D
            ls = exp(hyp(1:D));
            K = computePhidlls(z, D, M, ls, L, di);
        else
            error('Unknown hyper-parameter!');
        end 
    end
end
end

function K = computePhi(z, D, M, ls, L)
    z = z*diag(LS()./ls');
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

function dK = computePhidlls(z, D, M, ls, L, di)
%COMPUTEPHIDLLS Computes the gradients of Phi with respect to the length
% scales on a log scale.
    z = z*diag(LS()./ls');
    sz = size(z, 1);
    Phi = zeros(D, M, sz);
    m = 1:M;
    m = m';
    m = pi*m/2;
    %computing the eigenfunction values for D=1
    for d = 1:D
        if d == di    
            Phi(di, :, :) = 1/sqrt(L(di))*diag(m)*cos( m*(z(:, di)'+L(di))/L(di) )/L(di);
        else
            Phi(d, :, :) = 1/sqrt(L(d)) * sin( m*(z(:, d)'+L(d))/L(d) );
        end
    end
    
    dK = kronPower(Phi);
    dK = dK*diag(-z(:, di));
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


function K = getWeightPrior(lambda, M, D, hyp)
    sf = exp(hyp(D+1));
    K = zeros(M^D, 1);
    for k = 1:M^D
        % What happens here is the same as in degHSM2.
        % lambda will be multiplied by LS()^2 in the spectral density.
        K(k) = spectralDensity(lambda(k), D, sf);
    end
end

function s = spectralDensity(rSqrd, D, sf)
    %see Rasmussen p.154 above (7.11)
    s = sf*(sqrt(2*pi)^D)*(LS()^D)*exp(-LS()^2*rSqrd/2);
end

function retval = LS()
    retval = 0.1;
end

function dK = testDPhi(z, D, M, ls, L, di)
%TESTDPHI Function to test whether the gradients of Phi are computed
%correctly. To enable just replace the respective call.
    options = optimoptions(@fmincon,'Algorithm','interior-point',...
        'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);
    k = floor(rand(1)*M^D+1);
    l = floor(rand(1)*size(z, 1)+1);
    % actual optimization function
    actoptfunc = @(sls) optfunc(exp(sls), z, D, M, ls, L, di, k, l);
    %derivative check
    try
        fmincon(actoptfunc,...
                   log(ls(di)),[],[],[],[],[],[],@unitdisk,options);
    catch
        di
        error('Gradientcheck in degHSM failed');
    end
    dK = computePhidlls(z, D, M, ls, L, di);
end

function [sK, sdK] = optfunc(sls, z, D, M, ls, L, di, k, l)
    ls(di) = sls;
    K = computePhi(z, D, M, ls, L);
    dK = computePhidlls(z, D, M, ls, L, di);
    sK = K(k, l);
    sdK = dK(k, l);
end