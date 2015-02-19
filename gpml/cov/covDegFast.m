function K = covDegFast(bf, seed, M, hyp, x, z, i)
% Wrapper for basis functions. Works like covDegenerate but calls a mex
% function.
if nargin<4, K = feval(bf{:}); return; end              % report number of parameters

if nargin == 4
    %return the weight prior
    K = feval(bf{:}, hyp);
    return;
end

% determine mode
if nargin < 6, z = []; end
dg = strcmp(z,'diag') && numel(z)>0;       

n = size(x, 1);
if nargin <= 6
    if dg
      % gp.m will make this call
      K = zeros(n, 1);
    else
        bf1 = bf{1}; if isa(bf1, 'function_handle'), bf1 = func2str(bf1); end
        %ignore x
        K = bfmex(convertDegCovNameToLibGPMethodName(bf1), seed, M, unwrap(hyp), z);
    end
else                                                        % derivatives
    %TODO call mex function but check that we actually need that
    K = feval(bf{:}, hyp, z, i);
end
end

function name = convertDegCovNameToLibGPMethodName(name)
    if strcmp(name, 'degHSM2')
        name = 'Solin';
    elseif strcmp(name, 'degFastFood')
        name = 'FastFood';
    end
end
