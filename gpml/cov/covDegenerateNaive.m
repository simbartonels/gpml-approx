function K = covDegenerateNaive(bf, hyp, x, z, i)
% Wrapper for basis functions.
% Basis functions are supposed are called feval(bf{:}, hyp, z); and must
% return an m x n matrix where n = size(z, 1) and m the
% number of basis functions.

%TODO: make sure bf is a basis function (i.e. degCov)
if nargin<2, K = feval(bf{:}); return; end              % report number of parameters

% determine mode
if nargin < 4, z = []; end
dg = strcmp(z,'diag') && numel(z)>0;       

[n,D] = size(x);
%the weight prior
sigma = diag(feval(bf{:}, hyp));
bfx = feval(bf{:}, hyp, x);
bfz = feval(bf{:}, hyp, z);
if dg
    K = diag(bfx'*sigma*bfx);
else
    if xeqz
      bfz = bfx;
    else
      bfz = feval(bf{:}, hyp, z);
    end
    K = bfx'*sigma*bfz;
end
if nargin > 4                                                        % derivatives
    error('gradients not implemented');
end
