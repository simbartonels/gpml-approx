function K = covDegenerate(bf, hyp, x, z, i)
% Wrapper for basis functions.
% Basis functions are supposed are called feval(bf{:}, hyp, z); and must
% return an m x n matrix where n = size(z, 1) and m the
% number of basis functions.
if nargin<2, K = feval(bf{:}); return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,D] = size(x);
m = hyp(1);

%phiall = @(inp) phij(inp, j);
if dg                                                               % vector kxx
  K = zeros(n, 1);
else
  if xeqz                                                 % symmetric matrix Kxx
    error('Degenerate covariance functions are not meant to compute kernel matrices!');
  else
    %ignore x
    K = feval(bf{:}, hyp, z);
  end
end                                                 % covariance
if nargin>4                                                        % derivatives
    error('Optimization of hyperparameters not implemented.')
end