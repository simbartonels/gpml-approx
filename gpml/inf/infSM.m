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
M = cell2mat(cov(2));
sigma = hyp.cov(D+1:M*D+D);
sigma = reshape(sigma, [M, D]);

V = hyp.cov(M*D+D+1:2*M*D+D);
V = reshape(V, [M, D]);
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector
if ~(size(sigma) == size(V)); error('There must be one length-scale vector for basis vector with the same dimensionality!'), end
sn2 = exp(2*hyp.lik);                               % noise variance of likGauss

[K, Upsi, Uvx] = covSM(M, hyp.cov, x);
Lpsi = chol(Upsi);
lambda = zeros(n, 1);
for i=1:n
    lambda(i) = Uvx(:, i)' * solve_chol(Lpsi, Uvx(:, i));
end
%TODO: the step above could be more efficient using following:
%Vvx = Lpsi'\Uvx;
%lambda = sum(Vvx.*Vvx, 1)';

lambda = K - lambda;
%TODO: is it really necessary to create the diagonal? if it is always
%multiplied with a vector it is not necessary!
LambdaSnInv = diag(1./(lambda + sn2));
%clear lambda
Q = Upsi + Uvx * LambdaSnInv * Uvx';
LQ = chol(Q);

%return -Upsi^- + Q^-
%solvedTODO: make this more efficient
%post.L = chol(solve_chol(chol(solve_chol(Lpsi, eye(M)) - solve_chol(LQ, eye(M))), eye(M))/sn2);
%infFITC does something like the following!
post.L = -solve_chol(Lpsi, eye(M)) + solve_chol(LQ, eye(M));


clear Q
clear Upsi
mean_vec = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector
post.alpha = solve_chol(LQ, Uvx * (LambdaSnInv * (y - mean_vec)));
post.sW = ones(M,1)/sqrt(sn2);                  % sqrt of noise precision vector
%post.sW = [];
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
    %nlZ = sum(log(gamma))+sum(2*log(diag(S)))+(p-q)*log(sn2)+...
    %    ((y-m)'*((y-m)./gamma)-beta'*beta)/sn2;
    %nlZ = nlZ + M*log(2*pi);
    nlZ = nlZ + n*log(2*pi);
    nlZ = nlZ/2;
    
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
        for i = D:M*D+D
                %optimize the length scales of each basis functions
                
        end
    end
end
