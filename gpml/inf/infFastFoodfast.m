function [post, nlZ, dnlZ, g, pi, b] = infFastFoodfast(hyp, mean, cov, lik, x, y)

likstr = lik; if ~ischar(lik), likstr = func2str(lik); end 
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('FITC inference only possible with Gaussian likelihood');
end
cov1 = cov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
if ~strcmp(cov1,'degFastFood'); error('Only degFastFood supported.'), end    % check cov
if ~(isempty(mean) || strcmp(func2str(mean{1}), 'meanZero'))
    mean
    error('Only zero-mean supported!');
end
D = size(x, 2);
M = (size(hyp.cov) - (D+1))/4;
post.sW = [];                                                  % unused for FITC
if nargout == 1
    [alpha, L] = infFastFoodmex(M, unwrap(hyp), x, y);
elseif nargout == 2
    [alpha, L, nlZ] = infFastFoodmex(M, unwrap(hyp), x, y);
elseif nargout == 3
    [alpha, L, nlZ, dnlZ] = infFastFoodmex(M, unwrap(hyp), x, y);
else
    [alpha, L, nlZ, s, g, pi, b] = infFastFoodmex(m, unwrap(hyp), x, y);
    dnlZ = s;
end
post.alpha = alpha;                      % return the posterior parameters
post.L  = L; % Sigma-inv(Kuu)
end
