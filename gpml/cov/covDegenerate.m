function K = covDegenerate(bf, hyp, x, z, i)
% Wrapper for basis functions.
% Basis functions are supposed are called feval(bf{:}, hyp, z); and must
% return an m x n matrix where n = size(z, 1) and m the
% number of basis functions.
if nargin<2, K = feval(bf{:}); return; end              % report number of parameters

if nargin == 2
    %return the weight prior
    K = feval(bf{:}, hyp);
    return;
end

% determine mode
if nargin < 4, z = []; end
dg = strcmp(z,'diag') && numel(z)>0;       

[n,D] = size(x);
if nargin <= 4
    if dg
      % gp.m will make this call
      K = zeros(n, 1);
    else
        %ignore x
        K = feval(bf{:}, hyp, z);
    end
else                                                        % derivatives
    K = feval(bf{:}, hyp, z, i);
end