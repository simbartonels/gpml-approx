function [post nlZ dnlZ] = infSM(hyp, mean, cov, lik, x, y)

% Exact inference for a GP using Sparse Multiscale approximation.
%
% ATTENTION: The length scales affect the signal variance!
%
% See also INFMETHODS.M.
likstr = lik; if ~ischar(lik), likstr = func2str(lik); end 
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('Sparse multi-scale inference only possible with Gaussian likelihood');
end
cov1 = cov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
if ~strcmp(cov1,'covSM'); error('Only covSM supported.'), end    % check cov

[n, D] = size(x);
M = hyp.M;
logll = hyp.cov(1:D);                               % characteristic length scale
lsf2 = hyp.cov(2*M*D+D+1);
sf2 = exp(2*lsf2);
%Walder uses a slightly different ARD SE where the sigmas influence the 
%length scale 
actlsf2 = lsf2-(log(2*pi)*D+sum(logll))/2;
sigma = hyp.cov(D+1:M*D+D);
sigma = reshape(sigma, [M, D]);
logP = -(D*log(2*pi)+sum(sigma, 2))/2;
V = hyp.cov(M*D+D+1:2*M*D+D);
V = reshape(V, [M, D]);
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector
if ~(size(sigma) == size(V)); error('There must be one length-scale vector for basis vector with the same dimensionality!'), end
sn2 = exp(2*hyp.lik);                               % noise variance of likGauss

%TODO: write in a more efficient way
Upsi = zeros(M, M);
Uvx = zeros(M, n);
for i=1:M
   for j = 1:M
       temp = log(exp(sigma(i, :))+exp(sigma(j, :))-exp(logll'));
       Upsi(i, j) = covSEard([temp, -(log(2*pi)*D+sum(temp))/2], V(i, :), V(j, :));
   end
   for j = 1:n
       Uvx(i, j) = covSEard([sigma(i, :), logP(i)], x(j, :), V(i, :));
   end
end
Upsi = Upsi / sf2;
Lpsi = chol(Upsi);
lambda = zeros(n, 1);
for i=1:n
    lambda(i) = Uvx(:, i)' * solve_chol(Lpsi, Uvx(:, i));
end
lambda = covSEard([logll; actlsf2], x, 'diag') - lambda;
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
    Vvx = Lpsi'\Uvx;
    S = chol(sn2*eye(M)+Vvx*diag(1./gamma)*Vvx');
    beta = S'\(Vvx*((y-m)./gamma));
    
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
        %error('Derivatives not implemented yet!');    
        %lengthscale derivatives
        if i <= M*D+D
            if i > D
                %optimize the length scales of each basis functions
                
                %TODO: could this be easier with logical indices?
                %which dimension the parameter belongs to
                d = mod(i, D);
                %the corresponding basis vector
                j = (i-D-d)/M;
                dUvx = zeros([n, 1]);
                for k = 1:n
                    dUvx(k) = ((V(j, d) - x(k, d))^2/hyp.cov(i)^2 ... 
                        - 1/hyp.cov(i))/2*Uvx(j, k);
                end
            end
        end
    end
end
