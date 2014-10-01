function K = covSMnaive(M, logIndNoise, hyp, x, z, i)
    % Covariance function for "Sparse Multiscale Gaussian Process regression" as
    % described in the paper by Walder, Kim and Schölkopf in 2008.
    % M - the number of inducing inputs
    % logIndNoise - noise on the inducing variables on a log scale
    % Let g be Walder's ARD SE. The covariance function is parameterized as:
    %
    % k(x,z) = delta(x, z) * g(x, z, [S[0], sf2]) + ...
    %               (1 - delta(x, z)) * u(V,x)'*inv(Upsi)*u(V, z)
    %
    % where V is a matrix of inducing points, u(V, x)[i] = g(x, V[i], S[i]),S 
    % is a matrix of length scales and sf is the signal variance.
    % To describe the hyperparameters let M the number of inducing points, 
    % s = [ log(ell_1),
    %       log(ell_2),
    %        .
    %       log(ell_D) ]
    % S = [ s0; s1; s2; ... sM], v an M-dimensional vector and V = [v1; v2; ...
    % vM]. Then the hyperparameters are:
    % 
    % hyp = [flat(S), flat(V), log(sqrt(sf2))]
    %
    % ATTENTION: It is assumed that all vectors in the union of x and z are 
    % pairwise disjunct!
    %
    % See also COVFUNCTIONS.M.
    if nargin<=numMetaArgs(), K = sprintf('(2*%d*D+D+1)', M); return; end              % report number of parameters
    if nargin<=numMetaArgs()+2, z = []; end                                   % make sure, z exists
    xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

    [n,D] = size(x);                                       % signal variance
    sz = size(z, 1);
    logll = hyp(1:D);                               % characteristic length scale
    lsf = hyp(2*M*D+D+1);
    %Walder uses a slightly different ARD SE where the sigmas influence the 
    %length scale
    actlsf2 = lsf-(log(2*pi)*D+sum(logll)*2)/4;
    if dg                                                               % vector kxx
        K = covSEard([logll; actlsf2], x, 'diag');
    else
        [~, Upsi, Uvx] = covSM(M, hyp, x);
        % add noise for the inducing inputs
        indNoise = exp(logIndNoise);
        Upsi = Upsi + indNoise * eye(M);
        Lpsi = chol(Upsi);
        clear Upsi;
        if xeqz, 
            %K2 = Uvx'*solve_chol(Lpsi, Uvx);
            %this is to ensure that K is positive definite
            K = Lpsi'\Uvx;
            K = K'*K;
            %set diagonal to what Walder's ARD SE would produce
            K(logical(eye(size(K)))) = 0;
            K = K + diag(covSEard([logll; actlsf2], x, 'diag')); 
        else
            Uvz = covSM(M, hyp, x, z);
            K = Uvx'*solve_chol(Lpsi, Uvz);
        end
    end

    if nargin>numMetaArgs()+3                                                        % derivatives
        error('Hyperparameter optimization not supported yet!')
        %[dK, Upsi, Uvx] = covSM(M, hyp, x, z, i);
    end
end

function n = numMetaArgs()
    n = 2;
end