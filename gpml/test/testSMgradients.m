function testSMgradients()
    sd = floor(rand(1) * 32000)
    rng(sd);
    
    me = mfilename;                                            % what is my filename
    mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
    addpath([mydir,'util'])

    testdK();
    
    testdUpsi();

    testdUvx();
end

function testdUpsi()
n = 5;
D = 3;
M = 3;
[x, smhyp] = initStuff(n, D, M);
for i = 1:D
    disp('Checking length scales derivatives of Upsi');
    for k = 1:M
        for j = 1:M
            checkLsGradient(M, D, smhyp, x, 2, i, j, k);
        end
    end
end
for i = D+1:M*D+D
    disp('Checking inducing length scales derivatives of Upsi');
    for k = 1:M
        for j = 1:M
            checkInducingLsGradient(M, D, smhyp, x, 2, i, j, k);
        end
    end
end
for i = M*D+D+1:2*M*D+D+1
    disp('Checking inducing inputs and amplitude derivatives of Upsi');
    for k = 1:M
        for j = 1:M
            checkGradientRandomx0(M, D, smhyp, x, 2, i, j, k);
        end
    end
end
end

function testdUvx()
n = 5;
D = 3;
M = n;
[x, smhyp] = initStuff(n, D, M);
for i = 1:D
    for k = 1:M
        for j = 1:n
            checkLsGradient(M, D, smhyp, x, 3, i, j, k);
        end
    end
end
for i = D+1:M*D+D
    for k = 1:M
        for j = 1:n
            checkInducingLsGradient(M, D, smhyp, x, 3, i, j, k);
        end
    end
end
for i = D+M*D+1:D+2*M*D+1
    for k = 1:M
        for j = 1:n
            checkGradientRandomx0(M, D, smhyp, x, 3, i, j, k);
        end
    end
end
end

function testdK()
n = 5;
D = 3;
M = 3;
[x, smhyp] = initStuff(n, D, M);
for i = 1:D
    for k = 1:n
        checkLsGradient(M, D, smhyp, x, 1, i, 1, k);
    end
end
for i = D+1:M*D+D
    for k = 1:n
        checkInducingLsGradient(M, D, smhyp, x, 1, i, 1, k);
    end
end
for i = D+M*D+1:D+2*M*D+1
    for k = 1:n
        checkGradientRandomx0(M, D, smhyp, x, 1, i, 1, k);
    end
end
end

function checkGradient(M, D, smhyp, x, z, mode, i, j, k, x0)
            optimfunc = @(s) caller(mode, smhyp, M, x, z, i, j, k, s);
            %make sure the method can be called
            %[optimfunc(x0); unitdisk(x0)]
            %rng(0,'twister'); 
            options = optimoptions(@fmincon,'Algorithm','interior-point',...
                'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);
            fmincon(optimfunc,...
               x0,[],[],[],[],[],[],@unitdisk,options);
end

function checkGradientRandomx0(M, D, smhyp, x, mode, i, j, k, z)
            x0 = randn(1);
            if nargin < 9, z = []; end
            checkGradient(M, D, smhyp, x, z, mode, i, j, k, x0);
end

function checkLsGradient(M, D, smhyp, x, mode, i, j, k, z)
            sigma = reshape(smhyp.cov(D+1:M*D+D), [M, D]);
            cap = min(sigma(:, i));
            cap = exp(2*cap)/2;
            %make sure length scales are shorter than half of the inducing
            %length scales
            x0 = log(rand(1) * cap)/2;
            if nargin < 9, z = []; end
            checkGradient(M, D, smhyp, x, z, mode, i, j, k, x0);
end

function checkInducingLsGradient(M, D, smhyp, x, mode, i, j, k, z)
            x0 = randn(1, 1);
            d = mod(i-D-1, M)+1;
            d = (i-D-d)/M+1;
            logell = smhyp.cov(1:D);
            x0 = log(exp(2*x0)+exp(2*logell(d))/2)/2;
            if nargin < 9, z = []; end
            checkGradient(M, D, smhyp, x, z, mode, i, j, k, x0);
end

function [u, du] = caller(mode, hyp, M, x, z, i, j, k, s)
    hyp.cov(i) = s;
    if numel(z) == 0
        [K, Upsi, Uvx] = covSM(M, hyp.cov, x, z);
        switch mode
            case 1
                u = K(j);
            case 2
                u = Upsi(j, k);
            case 3
                u = Uvx(j, k);
        end
        if nargout > 1
            [dK, dUpsi, dUvx] = covSM(M, hyp.cov, x, z, i);
            switch mode
                case 1
                    du = dK(j);
                case 2
                    du = dUpsi(j, k);
                case 3
                    du = dUvx(j, k);
            end
        end
    elseif numel(z) > 0
        K = covSM(M, hyp.cov, x, z);
        u = K(j);
        if nargout > 1
            dK = covSM(M, hyp.cov, x, z, i);
            du = dK(j);
        end
    end
end

function [U, dU] = computeUvx(hyp, mean, x, y, di)
    M = hyp.M;
    [n, D] = size(x);
    [K, Upsi, Uvx] = covSM(M, hyp.cov, x);
    %the corresponding inducing point
    j = mod(di-D-1, M)+1;
    U = Uvx(j, :);
    if nargout > 1
        [dK, dUpsi, dUvx] = covSM(M, hyp.cov, x, [], di);
        dU = dUvx;
    end
end

function [x, smhyp] = initStuff(n, D, M)
x = randn(n, D);

logell = randn(D, 1);
lsf2 = log(randn(1)^2);
V = randn(M, D);
%TODO: make this more random (but make sure that sigma/2>=ell)
logsigma = repmat(logell', M, 1)+rand([M, D]);
smhyp.lik = log(randn(1)^2);
smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
    reshape(V, [M*D, 1]); lsf2];
end