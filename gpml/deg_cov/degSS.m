function K = degSS(s, hyp, z, di)

% Implementation of Sparse Spectrum GPR. 
%
% See also deg_covFunctions.M.

if nargin<2, K = '(2)'; return; end              % report number of parameters

m = size(s, 1);                      % signal variance
if nargin == 2
    %return weight prior
    sf2 = exp(2*hyp(2));         
    K = sf2*ones(2*m, 1)/m;
    return;
elseif nargin == 3
    ell = exp(hyp(1));
    w = computeWz(s, z, ell);
    K = [cos(w) sin(w)]';
else  %nargin>3                                                        % derivatives
    if isempty(z)
        % weight prior derivative
        if di == 2
            sf2 = exp(2*hyp(2));
            K = 2 * sf2 * ones(2*m, 1)/m;
        else
            K = zeros(2*m, 1);
        end
    else
        % derivatives of phi(z)
        if di == 1
            ell = exp(hyp(1));                                 % characteristic length scale
            w = computeWz(s, z, ell);
            K = [(sin(w).*w) (-cos(w).*w)]'; %/ell;
        else
            sz = size(z, 1);
            K = zeros([2*m, sz]);
        end
    end
end
end

function w = computeWz(s, z, ell)
    w = 2*pi*z*s'/ell;
end