%TODO: copy&paste code!!!
function [post, nlZ, dnlZ] = infSolinfast(hyp, mean, cov, lik, x, y)
% INFFITCMEX Inference method for Sparse multiscale GPR. Equivalent to FITC
% using covSM.

likstr = lik; if ~ischar(lik), likstr = func2str(lik); end 
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('FITC inference only possible with Gaussian likelihood');
end
cov1 = cov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
if ~strcmp(cov1,'degHSM2'); error('Only degHSM2 supported.'), end    % check cov
if ~(isempty(mean) || strcmp(func2str(mean{1}), 'meanZero'))
    mean
    error('Only zero-mean supported!');
end
M = cov{2};
post.sW = [];                                                  % unused for FITC
if nargout == 1
    [alpha, L] = infSolinmex(M, unwrap(hyp), x, y);
elseif nargout == 2
    [alpha, L, nlZ] = infSolinmex(M, unwrap(hyp), x, y);
else
    [alpha, L, nlZ, dnlZ] = infSolinmex(M, unwrap(hyp), x, y);
end
post.alpha = alpha;                      % return the posterior parameters
post.L  = L; % Sigma-inv(Kuu)
end
