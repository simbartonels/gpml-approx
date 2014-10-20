function [K, Upsi, Uvx] = covSM(M, hyp, x, z, di)
% Covariance function for infFITC. Efficient implementation of 
% "Sparse Multiscale Gaussian Process regression" as
% described in the paper by Walder, Kim and Schölkopf in 2008.
% Let g be Walder's ARD SE. The covariance function is parameterized as:
%
% k(x,z) = sf * delta(x, z) * g(x, z, S[0]) + ...
%               (1 - delta(x, z)) * u(V,x)'*inv(Upsi)*u(V, z)
%
% where V is a matrix of inducing points, u(V, x)[i] = g(x, V[i], S[i]),S 
% is a matrix of length scales and sf is the signal variance.
% To describe the hyperparameters let M the number of inducing points, 
% s = [ log(ell_1),
%       log(ell_2),
%        .
%       log(ell_D) ]
% s_j = log( S[j]-ell/2 )
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

% TODO: it's probably the best to make a mex file...

[n,D] = size(x);
%TODO: make sure basis points have same dimension as x
logll = hyp(1:D);                               % characteristic length scale
lsf = hyp(2*M*D+D+1);
sf = exp(lsf);
ell = exp(logll');
%actsf2 = sf/sqrt(prod(ell)*(2*pi)^D);
actsf2 = exp(lsf - (sum(logll)+D*log(2*pi))/2);
if dg                                                               % vector kxx
    K = actsf2*ones(n ,1);
else
    sigma = hyp(D+1:M*D+D);
    sigma = exp(reshape(sigma, [M, D])) + repmat(ell/2, [M, 1]);

    V = hyp(M*D+D+1:2*M*D+D);
    V = reshape(V, [M, D]);
    if xeqz                                                 
        K = actsf2*ones(n ,1);
        
        %TODO: write in a more efficient way
        Upsi = zeros(M, M);
        Uvx = zeros(M, n);
        for i=1:M
           for j = 1:M
               Upsi(i, j) = g(V(i, :), V(j, :), sigma(i, :)+sigma(j, :)-ell, D);
           end
           for j = 1:n
               Uvx(i, j) = g(x(j, :), V(i, :), sigma(i, :), D);
           end
        end
        Upsi = Upsi / sf;
    else                                            % cross covariances Kxz
        K = zeros(M, size(z, 1));
        %TODO: make this more efficient
        for j=1:M
            K(j, :) = g(z, V(j, :), sigma(j, :), D);
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
                d = di;
                % this is what we add during initialization to sigma
                p2 = ell(d)/2;
                for k=1:M
                    p = sigma(k, d);
                    for l=1:n
                        u = ((V(k, d) - x(l, d))/p)^2-1/p;
                        u = u * Uvx(k, l)/2;
                        % chain rule
                        u = p2 * u;
                        Uvx(k, l) = u;
                    end
                end
                
                % In the computation of Upsi ell actually cancels out. It
                % is added during initialization and substracted when
                % computing temp. Therefore the gradient is 0.
                Upsi = zeros(size(Upsi));
            elseif di >= D+1 && di <= M*D+D
                [d, j] = getDimensionAndIndex(di, D, M);
                p2 = sigma(j, d);
                Uvx = dAdl(Uvx, p2, d, x, V, j);
                
                p = p2+sigma(:, d)-ell(d);
                dUpsi = dAdl(Upsi, p, d, V, V, j);
                
                % chain rule
                p2 = p2 - ell(d)/2; % that half has no influence on the gradient
                Uvx = p2 * Uvx;
                dUpsi(j, :) = p2 * dUpsi(j, :);
                dUpsi(:, j) = dUpsi(j, :);
                dUpsi(j, j) = 2 * dUpsi(j, j);
                Upsi = dUpsi;
            elseif di >= M*D+D+1 && di <=2*M*D+D
                %inducing point derivatives
                [d, j] = getDimensionAndIndex(di, D, M);
                dUvx = zeros(size(Uvx));
                sig = sigma(j, d);
                dUvx(j, :) = (-V(j, d) + x(:, d))/sig .* Uvx(j, :)';
                Uvx = dUvx;
                
                dUpsi = zeros(size(Upsi));
                p2 = sigma(j, d);
                p = p2+sigma(:, d)-ell(d);
                dUpsi(j, :) = (-V(j, d) + V(:, d)) .* Upsi(j, :)' ./p;
                dUpsi(:, j) = dUpsi(j, :);
                Upsi = dUpsi;
            elseif di == 2*M*D+D+1
                Uvx = zeros(size(Uvx));
                %chain rule because sf2 is square root and log 
                Upsi = -Upsi;
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
                sig = sigma(j, d);
                dUvz(j, :) = (-V(j, d) + z(:, d))/sig .* K(j, :)';
                K = dUvz;
            elseif di == 2*M*D+D+1
                K = zeros(size(K));
            end
        end
    end
end
end

function z = g(x, y, s, D)
    z = sq_dist(diag(1./sqrt(s))*x',diag(1./sqrt(s))*y');
    z = exp(-z/2);
    z = z/sqrt(prod(s)*(2*pi)^D);
end

function lsf = computeLogRootSignalVariance(lsf, logll)
    % COMPUTELOGROOTSIGNALVARIANCE Computes the signal variance depending on
    % the length scales and the given length scale parameter for the SEard.
    % lsf2 is the log square root signal variance.
    % logll are log length scales.
    

    % ARDse squares the signal variance. Therefore we need to divide TWICE 
    % by 2. Also the sum needs to be multiplied with two since ARDse works 
    % with square root length scales.
    D = size(logll, 2);
    lsf = lsf - (log(2*pi)*D+2*sum(logll, 2))/4;
end

function dK = dKd(K, di, D, M)
        if di <= D
            dK = -K/2;
        elseif di >=D+1 && di <= 2*M*D+D
            %derivatives for the inducing points and corresponding length
            %scales
            dK = zeros(size(K));
        else
            %derivative of the amplitude
            dK = K;
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
    p = exp(logl);
    dUvx = dAdl(Uvx, p, d, x, V, j);
    % need to apply chain rule since the parameter is optimized
    % on a log scale and is a square root
    dUvx = dUvx * p;
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

function [K, Upsi, Uvx] = performErrorHandling(n, M, z, xeqz, gradients)
    disp('All inducing input length scales must be longer than half the corresponding length scale!');
    %error('All inducing input length scales must be longer than half the corresponding length scale!');
    if xeqz
        K = zeros([n, 1]);
    else
        K = zeros([M, size(z, 1)]);
    end
    Upsi = eye(M);
    Uvx = zeros([M, n]);
    if gradients
        % gradients are all 0
        Upsi = zeros([M, M]);
    end
end