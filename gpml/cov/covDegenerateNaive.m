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

if nargin <= 4
    if dg
      % gp.m will make this call
      K = diag(bfx'*sigma*bfx);
    else
      K = bfx'*sigma*bfx;
    end
else                                                        % derivatives
    bfz = feval(bf{:}, hyp, z);
    K = bfx'*sigma*bfz;
end