%TODO: copy&paste code!!!
function [post, nlZ, dnlZ] = infSolinfast(hyp, mean, cov, lik, x, y)

likstr = lik; if ~ischar(lik), likstr = func2str(lik); end 
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('Inference method supports only Gaussian likelihood');
end
cov1 = cov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
if strcmp(cov1,'covDegFast');
    cov = cov{2};
else
    error('You MUST use covDegFast to use this inference method.');
end
cov1 = cov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
if ~strcmp(cov1,'degHSM2'); error('Only degHSM2 supported.'), end    % check cov
if ~(isempty(mean) || strcmp(func2str(mean{1}), 'meanZero'))
    mean
    error('Only zero-mean supported!');
end
%TODO: this is a hack!
M_intern = cov{2};
M = size(x, 2)^M_intern;
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
