function [out1, out2, out3] = gp_sod_mine(logtheta, opt_iters, covfunc, likfunc, x, varargin)

% gpr_sod - Gaussian Process regression, using the Subset of Data 
% approximation. 
% 
% usage: [loghyper sod theta_over_time] = gpr_sod(logtheta, covfunc, likfunc, x, y, N, method,splitLen, 'split')
%    or: [mu S2 nlZ] = gpr_sod(logtheta, covfunc, likfunc, x, y, N, method, xstar)
%    or: [mu S2 nlZ] = gpr_sod(logtheta, covfunc, likfunc, x, y, sod, method, xstar)
%
% where:
%
%   logtheta is a (column) vector of log hyperparameters
%   covfunc  is the covariance function
%   x        is a n by D matrix of training inputs
%   y        is a (column) vector (of size n) of targets
%   xstar    is a nn by D matrix of test inputs
%   N        is the number of points to be included in the subset of data
%   method   is the method to be used for sod selection. Can be 'r'andom,
%                'e'ntropy maximization or 'c'lustering. If 'g' then assume
%                the indices of sod are 'g'iven in the sod argument
%   splitLen is only used with method 'c'lustering to indicate that the cluster center
%               list needs to be split into pieces not to keep the whole N x C matrix
%               in memory, where N is #data and C is #clusters. Note that setting splitLen
%               to dim(x) is reasonable as the whole training data array is assumed to be kept
%               in memory so it's a feasible size.
%   sod      is a (row) vector of indices of the subset of data
%   loghyper is a vector of optimized hyperparameters
%   mu       is a (column) vector (of size nn) of prediced means
%   S2       is a (column) vector (of size nn) of predicted variances

% Krzysztof Chalupka, University of Edinburgh 2011.

if nargin == 8

    y = varargin{1};
    N = varargin{2};
    method = varargin{3};
    sod = indPoints(x, N, method, covfunc, logtheta);

    % Optimize the hyperparameters.
    [out1, ~, ~, out3] = minimize(logtheta, @gp, opt_iters, @infExact, [], covfunc, likfunc, x(sod,:), y(sod,:));
    out2 = sod;

elseif nargin == 9

  method = varargin{3};

  if method == 'g'

    y = varargin{1};
    sod = varargin{2};
    xstar = varargin{4};

    [out1, out2, ~, ~, out3] = gp(logtheta, @infExact, [], covfunc, likfunc, x(sod,:), y(sod,:), xstar);

  else

    y = varargin{1};
    N = varargin{2};
    method = varargin{3};
    xstar = varargin{4};

    % Choose a sod.
    sod = indPoints(x, N, method, covfunc, logtheta);
    % Do the regression.
    [out1, out2, ~, ~, out3] = gp(logtheta, @infExact, [], covfunc, likfunc, x(sod,:), y(sod,:), xstar);
  end
elseif nargin==10

    y = varargin{1};
    N = varargin{2};
    method = varargin{3};
    splitLen = varargin{4};
    sod = indPoints(x, N, method, covfunc, logtheta, splitLen);

    % Optimize the hyperparameters.
    [out1, ~, ~, out3] = minimize(logtheta, @gp, opt_iters, @infExact, [], covfunc, likfunc, x(sod,:), y(sod,:));
    out2 = sod;
end

