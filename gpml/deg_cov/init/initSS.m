function S = initSS( m, D )
%INITSS Initializes all parameters for Sparse Spectrum GPR.
%   m - the number of basis functions
%   D - the input dimensionality
    % for parameter optimization it is preferable to divide later by ls!
    S = randn(m, D);
end