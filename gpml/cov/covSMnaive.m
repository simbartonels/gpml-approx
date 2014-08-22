function K = covSMnaive(M, hyp, x, z, i)
% Covariance function for "Sparse Multiscale Gaussian Process regression" as
% described in the paper by Walder, Kim and Schölkopf in 2008.
% Let g the ARD SE. The covariance function is parameterized as:
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
if nargin<3, K = sprintf('(2*%d*D+D+1)', M); return; end              % report number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,D] = size(x);                                       % signal variance
sz = size(z, 1);
logll = hyp(1:D);                               % characteristic length scale
lsf2 = hyp(2*M*D+D+1);

if dg                                                               % vector kxx
    K = covSEard([logll; lsf2], x, 'diag');
else
    S = hyp(D+1:M*D+D);
    S = reshape(S, [M, D]);
    V = hyp(M*D+D+1:2*M*D+D);
    V = reshape(V, [M, D]);
    Upsi = zeros(M, M);
    Uvx = zeros(M, n);
    if ~xeqz
        Uvz = zeros(M, sz);
    end
    for i=1:M
       for j = 1:M
           temp = log(exp(S(i, :))+exp(S(j, :))-exp(logll'));
           Upsi(i, j) = covSEard([temp, 0], V(i, :), V(j, :));
       end
       for j = 1:n
           Uvx(i, j) = covSEard([S(i, :), 0], x(j, :), V(i, :));
       end
       if ~xeqz
           for j = 1:sz
              Uvz(i, j) = covSEard([S(i, :), 0], z(j, :), V(i, :));
           end
       end
    end
    Upsi = Upsi / exp(2*lsf2);
    Lpsi = chol(Upsi);
    clear Upsi;
    if xeqz, Uvz = Uvx; end
    K = Uvx'*solve_chol(Lpsi, Uvz);
    %set diagonal to that of what SE ARD would produce
    if xeqz
        K(logical(eye(size(K)))) = 0;
        K = K + diag(covSEard([logll; lsf2], x, 'diag')); 
    end
end

if nargin>4                                                        % derivatives
    error('Hyperparameter optimization not supported yet!')
end