function testSMgradients()
    sd = floor(rand(1) * 32000)
    rng(sd);
    
    me = mfilename;                                            % what is my filename
    mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
    addpath([mydir,'util'])

    testdK();
    
    testUpsi();
    %testdUpsi();

    testUvx();
    %testdUvx();
end

function testdUpsi()
n = 5;
D = 3;
M = n-1;
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
M = n;
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
[z, smhyp] = initStuff(n, D, M);
for i = 1:D
    for k = 1:n
        checkLsGradient(M, D, smhyp, x, 1, i, 1, k, z);
    end
end
for i = D+1:M*D+D
    for k = 1:n
        checkInducingLsGradient(M, D, smhyp, x, 1, i, 1, k, z);
    end
end
for i = D+M*D+1:D+2*M*D+1
    for k = 1:n
        checkGradientRandomx0(M, D, smhyp, x, 1, i, 1, k, z);
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

function testUpsi()
    M = 4;
    [sd, n, D, x, y, xs, logell, lsf2, lsn2] = initEnv();
    rng(sd);
    logsigma = randn(M, D);
    logsigma = log(exp(2*logsigma)+repmat(exp(2*logell')/2, [M, 1]))/2;
    V = randn([M, D]);
    smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
        reshape(V, [M*D, 1]); lsf2];
    [~, Upsi, ~] = covSM(M, smhyp.cov, x);
    sigma = exp(2*logsigma);
    ell = exp(2*logell);
    U = zeros([M, M]);
    for i = 1:M
        for j = 1:M
            temp = sigma(i, :)+sigma(j, :)-ell';
            K = (V(i, :)-V(j, :))*diag(1./temp)*(V(i, :)-V(j, :))';
            u = exp(-K/2);
            f1 = -(log(2*pi)*D+sum(log(temp)/2, 2)*2)/4;
            f2 = exp(2*f1);
            f1 = 1/(sqrt(prod(temp))*sqrt((2*pi)^D));
            u = u * f1;
            U(i, j) = u;
        end
    end
    U = U/exp(2*lsf2);
    if max(max((U - Upsi).^2)) > 1e-15
        error('Something is wrong in the computation of Upsi');
    end
end

function testUvx()
    testUvxSimple();
    testUvxSimple2();
    testUvxRandom();
end
function testUvxSimple()
    %first tests the computation of Uvx for a special case in D=2
    %x = 0, V=1, sigma=1 => should deal 1/(2*pi*e)
    n = 5;
    M = n;
    D = 2;
    x = zeros(n, D);
    logell = zeros(D, 1);
    logsigma = repmat(logell', M, 1);
    %length scales are all 1

    lsf2 = 0;
    V = ones(n, D);
    %so V-x= 1
    
    smhyp.lik = log(randn(1)^2);
    smhyp.M = M;
    smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
    reshape(V, [M*D, 1]); lsf2];
    U = computeUvx(smhyp, [], x, [], D+1);
    for l=1:n
        if (U(l) - 1/(2*pi*exp(1)))^2 > 1e-30
            error('Testing computation of Uvx failed.');
        end
    end
end

function testUvxSimple2()
    %first tests the computation of Uvx for a special case in D=2
    %x = 0, V=1, sigma=1 => should deal 1/(2*pi*e)
    n = 5;
    M = n;
    D = 2;
    x = zeros(n, D);
    sigma = 2 * ones(D, 1);
    logell = log(sigma)/2;
    logsigma = repmat(logell', M, 1);
    %length scales are all 1

    lsf2 = 0;
    V = ones(n, D);
    %so V-x= 1
    
    smhyp.lik = log(randn(1)^2);
    smhyp.M = M;
    smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
    reshape(V, [M*D, 1]); lsf2];
    U = computeUvx(smhyp, [], x, [], D+1);
    for l = 1:n
        if (U(l) - exp(-sum(1./sigma)/2)/sqrt((2*pi)^D*prod(sigma)))^2 > 1e-30
            error('Testing computation of Uvx failed.');
        end
    end
end

function testUvxRandom()
    %now a random example
    D = 2;
    x = randn(1, D);
    V = randn(1, D);
    M = size(V, 1);
    logell = randn(D, 1)/2;
    sigma = exp(2*logell);
    logsigma = repmat(logell', M, 1);
    lsf = randn(1);
    %sf2 = exp(2*lsf);
    
    smhyp.lik = log(randn(1)^2);
    smhyp.M = M;
    smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
        reshape(V, [M*D, 1]); lsf];
    factor = 1/sqrt((2*pi)^D*prod(sigma));
    u = computeUvx(smhyp, [], x, [], D+1);
    %sf2 plays no role in Uvx
    K = (x-V)*diag(1./sigma)*(x-V)';
    us = factor*exp(-K/2);
    if (u - us)^2 > 1e-30
        error('Testing computation of Uvx failed.');
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