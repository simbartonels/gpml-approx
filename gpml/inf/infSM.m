function [post nlZ dnlZ] = infSM(hyp, mean, cov, lik, x, y)

% Exact inference for a GP with Gaussian likelihood. Compute a parametrization
% of the posterior, the negative log marginal likelihood and its derivatives
% w.r.t. the hyperparameters. See also "help infMethods".
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-12-22
%
% See also INFMETHODS.M.
likstr = lik; if ~ischar(lik), likstr = func2str(lik); end 
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('Sparse multi-scale inference only possible with Gaussian likelihood');
end
cov1 = cov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
if ~strcmp(cov1,'covSM'); error('Only covSM supported.'), end    % check cov

[n, D] = size(x);
v = hyp.v;
if ~(size(v, 2) == D); error('Training data and basis vectors must have same dimenion!'), end
M = size(v, 1);
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector
sigma = hyp.logsigma; %exp(hyp.logsigma);
if ~(size(sigma) == size(v)); error('There must be one length-scale vector for basis vector with the same dimensionality!'), end
ell = hyp.cov(1:D);%exp(hyp.cov(1:D));
sn2 = exp(2*hyp.lik);                               % noise variance of likGauss
lsf2 = hyp.cov(D+1);
%TODO: write in a more efficient way
Upsi = zeros(M, M);
Uvx = zeros(M, n);
%Uvv = zeros(m, m);
for i=1:M
   for j = 1:M
       %Upsi(i, j) = covSEard([sigma(i, :)+sigma(j, :)-ell', 0], v(i, :), v(j, :));
       %Upsi(i, j) = covSEard([sigma(i, :).*sigma(j, :)./ell', 0], v(i, :), v(j, :));
       temp = log(exp(sigma(i, :))+exp(sigma(j, :))-exp(ell'));
       Upsi(i, j) = covSEard([temp, 0], v(i, :), v(j, :));
       %Uvv(i, j) = covSEard([sigma(i, :), 0], v(j, :), v(i, :));
   end
   for j = 1:n
       Uvx(i, j) = covSEard([sigma(i, :), 0], x(j, :), v(i, :));
   end
end
Upsi = Upsi / exp(2*lsf2);
Lpsi = chol(Upsi);
%lambda = zeros(m, 1);
lambda = zeros(n, 1);
%TODO: is this equivalent to lambda(1:m) = Uvv(:, 1:m)' * Lpsi\Uvv(:, 1:m);
%for i=1:m
    %lambda(i) = Uvv(:, i)' * solve_chol(Lpsi, Uvv(:, i));
for i=1:n
    lambda(i) = Uvx(:, i)' * solve_chol(Lpsi, Uvx(:, i));
end
lambda = covSEard(hyp.cov, x, 'diag') - lambda;
%LambdaSnInv = diag(1./(covSEard(hyp.cov, v, 'diag') - lambda + sn2));
LambdaSnInv = diag(1./(lambda + sn2));
%clear lambda
Q = Upsi + Uvx * LambdaSnInv * Uvx';
LQ = chol(Q);
%TODO: make this more efficient
%maybe it is possible to use the alternative parameterization in infExact!
%return -Upsi^- + Q^-
%post.L = chol(solve_chol(chol(solve_chol(Lpsi, eye(M)) - solve_chol(LQ, eye(M))), eye(M)));
post.L = chol(solve_chol(chol(solve_chol(Lpsi, eye(M)) - solve_chol(LQ, eye(M))), eye(M))/sn2);
%post.L is the cholesky of the inverse of (Upsi^-1 - Q^-1)
clear Q
clear Upsi
mean_vec = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector
post.alpha = solve_chol(LQ, Uvx * LambdaSnInv * (y - mean_vec));
post.sW = ones(M,1)/sqrt(sn2);                  % sqrt of noise precision vector

clear LambdSnInv
%clear Lpsi
clear LQ


if nargout>1                               % do we want the marginal likelihood?
    %error('Marginal likelihood not implemented yet!');
 
    % this is what Walder proposed
    gamma = lambda./sn2+ones(n, 1);
    %need to solve for the transpose of Lpsi due to solve_chol
    V = Lpsi'\Uvx;
    S = chol(sn2*eye(M)+V*diag(1./gamma)*V');
    beta = S'\(V*((y-m)./gamma));
    
    p = n; %?
    q = M; %?
    nlZ = log(prod(gamma))+log(prod(diag(S).^2))+(p-q)*log(sn2)+...
        ((y-m)'*((y-m)./gamma)-beta'*beta)/sn2;
    %nlZ = nlZ + M*log(2*pi);
    nlZ = nlZ + n*log(2*pi);
    nlZ=nlZ/2;
    
    %this is what Walder wrote as psi
%     K = Uvx'*solve_chol(Lpsi, Uvx)+diag(lambda);
%     L = chol(K+sn2*eye(n));
%     phi = log(prod(diag(L).^2))+(y-m)'*solve_chol(L, (y-m));
%     %phi = D*log(2*pi) + phi;
%     phi = M*log(2*pi) + phi;
%     phi = phi/2;
%     phi - nlZ
    if nargout>2                                         % do we want derivatives?
        error('Derivatives not implemented yet!');    
    end
end
