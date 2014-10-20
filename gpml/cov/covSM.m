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
if nargin>4        
        K = dKd(K, di, D, M);
        %lengthscale derivatives?
        if di <= D
            d = di;
            % this is what we add during initialization to sigma
            p2 = ell(d)/2;
            dUvx = zeros(size(Uvx));
            for k = 1:M
                %dAdl returns actually only a vector. the rest is 0
                dUvx = dUvx + dAdl(Uvx, sigma(k, d), d, x, V, k);
            end
            % chain rule
            Uvx = p2 * dUvx;


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
end
end

function z = g(x, y, s, D)
    z = sq_dist(diag(1./sqrt(s))*x',diag(1./sqrt(s))*y');
    z = exp(-z/2);
    z = z/sqrt(prod(s)*(2*pi)^D);
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