function [K, Upsi, Uvx] = covSM(M, hyp, x, z, di)

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

if nargin<2, K = sprintf('(2*%d*D+D+1)', M); return; end % report number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,D] = size(x);
%TODO: make sure basis points have same dimension as x
%TODO: make somehow sure basis points are the same as in infSM
logll = hyp(1:D);                               % characteristic length scale
lsf = hyp(2*M*D+D+1);
sf2 = exp(2*lsf);
actsf2 = exp(2*computeLogRootSignalVariance(lsf, logll'));
if dg                                                               % vector kxx
    K = actsf2*ones(n ,1);
else
    sigma = hyp(D+1:M*D+D);
    sigma = reshape(sigma, [M, D]);
    if any(exp(2*sigma) < repmat(exp(2*logll')/2, [M, 1]))
        disp('All inducing input length scales must be longer than half the corresponding length scale!');
        error('All inducing input length scales must be longer than half the corresponding length scale!');
        K = zeros([n, 1]);
        Upsi = eye(M);
        Uvx = zeros([M, n]);
        if nargin > 4
            % gradients are all 0
            Upsi = zeros([M, M]);
        end
        return
    end

    logP = computeLogRootSignalVariance(0, sigma);

    V = hyp(M*D+D+1:2*M*D+D);
    V = reshape(V, [M, D]);
    if xeqz                                                 
        K = actsf2*ones(n ,1);
        %TODO: write in a more efficient way
        Upsi = zeros(M, M);
        Uvx = zeros(M, n);
        for i=1:M
           for j = 1:M
               %division and multiplication with 2 because are dealing with
               %square roots.
               temp = log(exp(2*sigma(i, :))+exp(2*sigma(j, :))-exp(2*logll'))/2;
               Upsi(i, j) = covSEard([temp, computeLogRootSignalVariance(0, temp)], V(i, :), V(j, :));
           end
           for j = 1:n
               Uvx(i, j) = covSEard([sigma(i, :), logP(i)], x(j, :), V(i, :));
           end
        end
        Upsi = Upsi / sf2;
    else                                            % cross covariances Kxz
        K = zeros(M, size(z, 1));
        %TODO: make this more efficient
        for j=1:M
            K(j, :) = covSEard([sigma(j, :), logP(j)], z, V(j, :));
        end
    end
end
if nargin>4                                                   % derivatives
    if dg
        K = dKd(K, di, D, M);
    else
        if xeqz
            K = dKd(K, di, D, M);
            %lengthscale derivatives?
            if di <= D
                Uvx = zeros(size(Uvx));
                
                d = di;
                p2 = exp(2*logll(d));
                for k=1:M
                    for l=1:M
                        p = exp(2*sigma(k, d))+exp(2*sigma(l, d))-p2;
                        u = ((V(k, d) - V(l, d))/p)^2-1/p;
                        u = u * Upsi(k, l)/2;
                        % chain rule
                        u = 2 * (-p2) * u;
                        Upsi(k, l) = u;
                    end
                end
            elseif di >= D+1 && di <= M*D+D
                %TODO: could this be easier with logical indices?
                [d, j] = getDimensionAndInducingPoint(di, D, M);
                Uvx = dUdl(Uvx, hyp(di), d, j, x, V);
                
                p2 = exp(2*sigma(j, d));
                p = p2+exp(2*sigma(:, d))-exp(2*logll(d));
                dUpsi = dAdl(Upsi, p, d, V, V, j);
                % chain rule
                dUpsi(j, :) = 2 * p2 * dUpsi(j, :);
                dUpsi(:, j) = dUpsi(j, :);
                dUpsi(j, j) = 2 * dUpsi(j, j);
                Upsi = dUpsi;
            elseif di >= M*D+D+1 && di <=2*M*D+D
                %inducing point derivatives
            elseif di == 2*M*D+D+1
                Uvx = zeros(size(Uvx));
                %chain rule because sf2 is square root and log 
                Upsi = -2*Upsi;
            end
        else
            if di >= D+1 && di <= M*D+D
                [d, j] = getDimensionAndInducingPoint(di, D, M);
                dUvz = dUdl(Uvz, hyp(di), d, j, z, V);
                Uvx = dUvz;
            end
        end
    end
end
end

function lsf2 = computeLogRootSignalVariance(lsf2, logll)
    % COMPUTELOGROOTSIGNALVARIANCE Computes the signal variance depending on
    % the length scales and the given length scale parameter for the SEard.
    % lsf2 is the log square root signal variance.
    % logll are log length scales.
    

    % ARDse squares the signal variance. Therefore we need to divide TWICE 
    % by 2. Also the sum needs to be multiplied with two since ARDse works 
    % with square root length scales.
    D = size(logll, 2);
    lsf2 = lsf2 - (log(2*pi)*D+sum(logll, 2)*2)/4;
end

function dK = dKd(K, di, D, M)
        if di <= D
            %derivatives of the length scales
            %the chain rule part cancels with the 1/sigma
            dK = -K;
        elseif di >=D+1 && di <= 2*M*D+D
            %derivatives for the inducing points and corresponding length
            %scales
            dK = 0;
        else
            %derivative of the amplitude
            dK = 2 * K;
        end
end

function dUvx = dUdl(Uvx, logl, d, j, x, V)
    % DUDL Computes the derivative of Uvx with respect to the inducing
    % points' length scales.
    % Uvx: the matrix
    % logl: log square root length scale
    % d: the dimension
    % j: index of the corresponding basis vector
    % x: input matrix
    % V: inducing input matrix
    
    % the parameter
    p = exp(2*logl);
    dUvx = dAdl(Uvx, p, d, x, V, j);
    % need to apply chain rule since the parameter is optimized
    % on a log scale and is a square root
    dUvx = dUvx * 2 * p;
end

function dA = dAdl(A, p, d, x, z, i)
    % DADL Computes the derivative of A with respect to length scale
    % parameter p where [A]ij = g(xj, zi, [Sigma]i) and [Sigma]id=p.
    
    %TODO: could this be easier with logical indices?
    dA = zeros(size(A));
    dA(i, :) = ((((z(i, d) - x(:, d))./p).^2-1./p))'.*A(i, :)/2;
end

function [d, j] = getDimensionAndInducingPoint(di, D, M)
    % the corresponding inducing point
    j = mod(di-D-1, M)+1;
    % which dimension the parameter belongs to
    d = (di-D-j)/M+1;
end