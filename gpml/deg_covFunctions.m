%TODO: update documentation

% degenerate covariance functions to be used by Gaussian process functions. 
% These functions are meant to represent kernels of the form
% k(x,y)=phi(x)*Sigma*phi(y). Degenerate covariance functions are used in
% the same locations as "normal" covariance functions but are expected to
% behave differently. In particular they must return a null vector if
% called as k(x, 'diag'), phi(z) for a call k(x, z) and should ignore/fail 
% calls of the form k(x, x).
% Naming convention: all degenerate covariance functions are named "deg/deg*.m".
%
% See also covFunctions.m.