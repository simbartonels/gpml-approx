function [K, Upsi, Uvx] = covSM(M, hyp, x, z, di)

% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sqrt(sf2)) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% See also COVFUNCTIONS.M.

if nargin<2, K = sprintf('(2*%d*D+D+1)', M); return; end % report number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,D] = size(x);
%TODO: make sure basis points have same dimension as x
%TODO: make somehow sure basis points are the same as in infSM
logll = hyp(1:D);                               % characteristic length scale
lsf = hyp(2*M*D+D+1);
sf2 = exp(2*lsf);
actsf2 = exp(2*computeLogRootSignalVariance(lsf, logll'));
if dg                                                               % vector kxx
    K = actsf2*ones(n ,1);
else
    sigma = hyp(D+1:M*D+D);
    sigma = reshape(sigma, [M, D]);
    logP = computeLogRootSignalVariance(0, sigma);

    V = hyp(M*D+D+1:2*M*D+D);
    V = reshape(V, [M, D]);
    if xeqz                                                 
        K = actsf2*ones(n ,1);
        %TODO: write in a more efficient way
        Upsi = zeros(M, M);
        Uvx = zeros(M, n);
        for i=1:M
           for j = 1:M
               %division and multiplication with 2 because are dealing with
               %square roots.
               temp = log(exp(2*sigma(i, :))+exp(2*sigma(j, :))-exp(2*logll'))/2;
               Upsi(i, j) = covSEard([temp, computeLogRootSignalVariance(0, temp)], V(i, :), V(j, :));
           end
           for j = 1:n
               Uvx(i, j) = covSEard([sigma(i, :), logP(i)], x(j, :), V(i, :));
           end
        end
        Upsi = Upsi / sf2;
    else                                            % cross covariances Kxz
        K = zeros(M, size(z, 1));
        %TODO: make this more efficient
        for j=1:M
            K(j, :) = covSEard([sigma(j, :), logP(j)], z, V(j, :));
        end
    end
end
if nargin>4                                                   % derivatives
    if dg
        error('Not implemented yet!');
        %K = feval(cov{:},hyp,x,'diag',i);
    else
        if xeqz
            %K   = feval(cov{:},hyp,x,'diag',i);
            %Kuu = feval(cov{:},hyp,xu,[],i);
            %Ku  = feval(cov{:},hyp,xu,x,i);
            
            %lengthscale derivatives?
            if di >= D+1 && di <= M*D+D
                %TODO: could this be easier with logical indices?
                % the corresponding inducing point
                j = mod(di-D-1, M)+1;
                % which dimension the parameter belongs to
                d = (di-D-j)/M+1;
                dUvx = zeros([n, 1]);
                % the parameter
                p = exp(2*hyp(di));
                for k = 1:n
                     dUvx(k) = (((V(j, d) - x(k, d))/p).^2 ... 
                        - 1/p)* Uvx(j, k)/2;
                    %need to apply chain rule since the parameter is optimized
                    %on a log scale and a square root
                    dUvx(k) = dUvx(k) * 2 * p;
                end
            end
            %TODO: this costs speed! (how much?)
            Uvx = dUvx;
        else
            error('Not implemented yet!');
          %K = feval(cov{:},hyp,xu,z,i);
        end
    end
end
end

function lsf2 = computeLogRootSignalVariance(lsf2, logll)
    % COMPUTELOGROOTSIGNALVARIANCE Computes the signal variance depending on
    % the length scales and the given length scale parameter for the SEard.
    % lsf2 is the log square root signal variance.
    % logll are log length scales.
    

    % ARDse squares the signal variance. Therefore we need to divide TWICE 
    % by 2. Also the sum needs to be multiplied with two since ARDse works 
    % with square root length scales.
    D = size(logll, 2);
    lsf2 = lsf2 - (log(2*pi)*D+sum(logll, 2)*2)/4;
end