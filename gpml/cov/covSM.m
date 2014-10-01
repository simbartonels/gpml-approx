function [K, Upsi, Uvx] = covSM(M, hyp, x, z, di)
% Covariance function for infFITC. Efficient implementation of 
% "Sparse Multiscale Gaussian Process regression" as
% described in the paper by Walder, Kim and Schölkopf in 2008.
% Let g be Walder's ARD SE. The covariance function is parameterized as:
%
% k(x,z) = delta(x, z) * g(x, z, [S[0], sf2]) + ...
%               (1 - delta(x, z)) * u(V,x)'*inv(Upsi)*u(V, z)
%
% where V is a matrix of inducing points, u(V, x)[i] = g(x, V[i], S[i]),S 
% is a matrix of length scales and sf is the signal variance.
% To describe the hyperparameters let M the number of inducing points, 
% s = [ log(ell_1),
%       log(ell_2),
%        .
%       log(ell_D) ]
% S = [ s0; s1; s2; ... sM], v an M-dimensional vector and V = [v1; v2; ...
% vM]. Then the hyperparameters are:
% 
% hyp = [flat(S), flat(V), log(sqrt(sf2))]
%
% ATTENTION: It is assumed that all vectors in the union of x and z are 
% pairwise disjunct!
%
% See also COVFUNCTIONS.M.

if nargin<2, K = sprintf('(2*%d*D+D+1)', M); return; end % report number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,D] = size(x);
%TODO: make sure basis points have same dimension as x
logll = hyp(1:D);                               % characteristic length scale
lsf = hyp(2*M*D+D+1);
sf2 = exp(2*lsf);
actsf2 = exp(2*computeLogRootSignalVariance(lsf, logll'));
if dg                                                               % vector kxx
    K = actsf2*ones(n ,1);
else
    sigma = hyp(D+1:M*D+D);
    sigma = reshape(sigma, [M, D]);
    % TODO: instead of this check it would be better to set optimization
    % boundaries.
    if any(any(exp(2*sigma) < repmat(exp(2*logll')/2, [M, 1])))
        [K, Upsi, Uvx] = performErrorHandling();
        return
    end

    logP = computeLogRootSignalVariance(0, sigma);

    V = hyp(M*D+D+1:2*M*D+D);
    V = reshape(V, [M, D]);
    if xeqz                                                 
        K = actsf2*ones(n ,1);
        
        ell = exp(2*logll');
        %TODO: write in a more efficient way
        Upsi = zeros(M, M);
        Uvx = zeros(M, n);
        for i=1:M
           for j = 1:M
               %division and multiplication with 2 because are dealing with
               %square roots.
               temp = log(exp(2*sigma(i, :))+exp(2*sigma(j, :))-ell)/2;
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
                [d, j] = getDimensionAndIndex(di, D, M);
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
                [d, j] = getDimensionAndIndex(di, D, M);
                dUvx = zeros(size(Uvx));
                sig = exp(2*sigma(j, d));
                dUvx(j, :) = (-V(j, d) + x(:, d))/sig .* Uvx(j, :)';
                Uvx = dUvx;
                
                dUpsi = zeros(size(Upsi));
                p2 = exp(2*sigma(j, d));
                p = p2+exp(2*sigma(:, d))-exp(2*logll(d));
                dUpsi(j, :) = (-V(j, d) + V(:, d)) .* Upsi(j, :)' ./p;
                dUpsi(:, j) = dUpsi(j, :);
                Upsi = dUpsi;
            elseif di == 2*M*D+D+1
                Uvx = zeros(size(Uvx));
                %chain rule because sf2 is square root and log 
                Upsi = -2*Upsi;
            end
        else
            if di <= D
                K = zeros(size(K));
            elseif di >= D+1 && di <= M*D+D
                [d, j] = getDimensionAndIndex(di, D, M);
                K = dUdl(K, hyp(di), d, j, z, V);
            elseif di >= M*D+D+1 && di <=2*M*D+D
                %inducing point derivatives
                [d, j] = getDimensionAndIndex(di, D, M);
                dUvz = zeros(size(K));
                sig = exp(2*sigma(j, d));
                dUvz(j, :) = (-V(j, d) + z(:, d))/sig .* K(j, :)';
                K = dUvz;
            elseif di == 2*M*D+D+1
                K = zeros(size(K));
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
            dK = zeros(size(K));
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

function [d, j] = getDimensionAndIndex(di, D, M)
    % GETDIMENSIONANDINEX Returns the dimension of the parameter and the
    % index that is between between 1 and M. This works for the inducing
    % points as well as for the corresponding length scales.
    % the corresponding inducing point
    j = mod(di-D-1, M)+1;
    % which dimension the parameter belongs to
    d = (di-D-j)/M+1;
    d = mod(d-1, D)+1;
end

function [K, Upsi, Uvx] = performErrorHandling()
    disp('All inducing input length scales must be longer than half the corresponding length scale!');
    %error('All inducing input length scales must be longer than half the corresponding length scale!');
    if xeqz
        K = zeros([n, 1]);
    else
        K = zeros([M, size(z, 1)]);
    end
    Upsi = eye(M);
    Uvx = zeros([M, n]);
    if nargin > 4
        % gradients are all 0
        Upsi = zeros([M, M]);
    end
end