function S = initSS( m, D, ls )
%INITSS Initializes all parameters for Sparse Spectrum GPR.
%   m - the number of basis functions
%   D - the input dimensionality
%   ls - length scale (not ls^2!!!)
    % for parameter optimization it is preferable to divide later by ls!
    S = randn(m, D); % / ls;
end