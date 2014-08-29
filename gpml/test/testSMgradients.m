function testSMgradients()
    sd = floor(rand(1) * 32000)
    rng(sd);
    testUvx()
    testdUvx();
end

function testdUvx()
n = 5;
D = 3;
x = randn(n, D);
y = randn(n, 1);

logell = randn(D, 1);
lsf2 = log(randn(1)^2);
V = randn(n, D);
M = size(V, 1);
%TODO: make this more random (but make sure that sigma/2>=ell)
logsigma = repmat(logell', M, 1)+rand([M, D]);
smhyp.lik = log(randn(1)^2);
smhyp.M = M;
smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
    reshape(V, [M*D, 1]); lsf2];

me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
addpath([mydir,'util'])

%rng(0,'twister'); 
options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);

for i = D+1:M*D
    for l = 1:n
        i
        l
        optimfunc = @(s) caller(@computeUvx, smhyp, [], x, y, i, l, s);
        x0 = randn(1, 1);
        %fplot(optimfunc, [x0-0.5, x0+0.5]);
        [xstar fval exitflag output] = fmincon(optimfunc,...
           x0,[],[],[],[],[],[],@unitdisk,options)
    end
end
end

function [u, du] = caller(f, hyp, mean, x, y, i, l, s)
    hyp.cov(i) = s;
    if nargout > 1
        [U, dU] = f(hyp, mean, x, y, i);
        u = U(l);
        du = dU(l);
    else
        U = f(hyp, mean, x, y, i);
        u = U(l);
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