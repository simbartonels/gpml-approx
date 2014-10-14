function testSM()
    testUpsi();
    testUvx();
    %testAgainstFullGP();
    testFITCimplAgainstNaive();
    %testGradients();
    %testFailures();
end

function testUpsi()
    M = 4;
    [x, ~, ~, hyp] = initEnv();
    D = size(x, 2);
    logell = hyp.cov(1:D);
    lsf = hyp.cov(D+1);
    logsigma = randn(M, D);
    logsigma = log(exp(2*logsigma)+repmat(exp(2*logell')/2, [M, 1]))/2;
    V = randn([M, D]);
    smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
        reshape(V, [M*D, 1]); lsf];
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
    U = U/exp(2*lsf);
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
    [~, ~, U] = covSM(M, smhyp.cov, x);
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
    [~, ~, U] = covSM(M, smhyp.cov, x);
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
    [~, ~, u] = covSM(M, smhyp.cov, x);
    %sf2 plays no role in Uvx
    K = (x-V)*diag(1./sigma)*(x-V)';
    us = factor*exp(-K/2);
    if (u - us)^2 > 1e-30
        error('Testing computation of Uvx failed.');
    end
    
end


function testFailures()
    testTooSmallInducingLengthScales();
end

function testTooSmallInducingLengthScales()
% Tests what happens if the algorithm is parameterized wrong.
% Should deal a warning message
    [x, y, xs, hyp] = initEnv();
    lsn2 = 0;
    lsf2 = 0;

    [n, D] = size(x);
    M = n - 2;
    V = randn([M, D]);
    %length scale parameters are shorter than half of the original ls
    logell = hyp.cov(1:D);
    logsigma = log(repmat(exp(2*logell)', M, 1)/3)/2;
    smhyp.lik = lsn2;
    %smhyp.M = M;
    smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
        reshape(V, [M*D, 1]); lsf2+(log(2*pi)*D+sum(logell)*2)/4];
    %should yield a warning message
    gp(smhyp, @infFITC, [], {@covSM, M}, @likGauss, x, y, xs);
    nlZ = gp(smhyp, @infFITC, [], {@covSM, M}, @likGauss, x, y);
    if nlZ > 0, error('Model has positive llh even though it s parameters are wrong!'); end

    options = optimoptions(@fmincon,'Algorithm','interior-point',...
        'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);
    optfunc = @(hypx) optimfunc(hypx, smhyp, @infFITC, [], {@covSM, M}, @likGauss, x, y);

    %derivative check
    fmincon(optfunc,...
               unwrap(smhyp),[],[],[],[],[],[],@unitdisk,options);
end

function testAgainstFullGP()
% Assert that if using M=n inducing points with U=X that we have then the
% original GP.
    % seed that appears to break this test: sd = 1894
    % sd = 26071
    [x, y, xs, hyp] = initEnv();

    [ymuE, ys2E] = gp(hyp, @infExact, [], @covSEard, @likGauss, x, y, xs);
    nlZtrue = gp(hyp, @infExact, [], @covSEard, @likGauss, x, y);

    V = x;
    M = size(V, 1);
    D = size(x, 2);
    logell = hyp.cov(1:D);
    lsf2 = hyp.cov(D+1);
    logsigma = repmat(logell', M, 1);
    smhyp.lik = hyp.lik;
    smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
        reshape(V, [M*D, 1]); lsf2+(log(2*pi)*D+sum(logell)*2)/4];
    
    M = size(V, 1);
    %then let's make sure the naive implementation is correct
    indNoise = -Inf;
    [ymu, ys2, ~, ~] = gp(smhyp, @infExact, [], {@covSMnaive, M, indNoise}, @likGauss, x, y, xs);
    nlZN = gp(smhyp, @infExact, [], {@covSMnaive, M, indNoise}, @likGauss, x, y);
    %nlZN = 0;
    %should deal the same output as the GP
    worst_dev_naive = [max((ymu-ymuE).^2), max((ys2-ys2E).^2), nlZtrue - nlZN]
    if any(abs(worst_dev_naive) > 1e-13), error('Naive Implementation appears broken!'); end
end

function testFITCimplAgainstNaive()
    % If the length scales of the individual basis functions are equal to the 
    % GPs length scales the method becomes equivalent Snelson's SPGP/FITC 
    % method.
    %sd = 31594;
    [x, y, xs, smhyp] = initEnvConcrete(31594);
    D = size(x, 2);
    M = (numel(smhyp.cov)-1-D)/D/2;

    logIndNoise = log(1e-6 * exp(2*smhyp.lik));
    [ymu, ys2] = gp(smhyp, @infExact, [], {@covSMnaive, M, logIndNoise}, @likGauss, x, y, xs);
    nlZ = gp(smhyp, @infExact, [], {@covSMnaive, M, logIndNoise}, @likGauss, x, y);


    [ymuS, ys2S] = gp(smhyp, @infFITC, [], {@covSM, M}, @likGauss, x, y, xs);
    nlZS = gp(smhyp, @infFITC, [], {@covSM, M}, @likGauss, x, y);
    diff_impl_and_fitc = [max((ymu-ymuS).^2), max((ys2-ys2S).^2), nlZS - nlZ]
    if any(diff_impl_and_fitc > 1e-14), error('FITC Implementation appears broken!'); end
end

function testGradients()
    [x, y, ~, smhyp] = initEnvConcrete();
    D = size(x, 2);
    M = (numel(smhyp.cov)-1-D)/D/2;
    options = optimoptions(@fmincon,'Algorithm','interior-point',...
        'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);
    optfunc = @(hypx) optimfunc(hypx, smhyp, @infFITC, [], {@covSM, M}, @likGauss, x, y);

    %derivative check
    fmincon(optfunc,...
               unwrap(smhyp),[],[],[],[],[],[],@unitdisk,options);
end

function [x, y, xs, smhyp] = initEnvConcrete(sd)
    if nargin == 0
        [x, y, xs, smhyp] = initEnv();
    else
        [x, y, xs, smhyp] = initEnv(sd);
    end
    [n, D] = size(x);
    logell = smhyp.cov(1:D);
    lsf2 = smhyp.cov(D+1);
    % Now let's check that the naive and actual implementation agree.
    M = n - 2;
    V = randn([M, D]);
    %make sure length scale parameters are larger than half of the original ls
    logsigma = log(exp(2*randn([M, D]).^2)+repmat(exp(2*logell)', M, 1)/2)/2;

    smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
        reshape(V, [M*D, 1]); lsf2+(log(2*pi)*D+sum(logell)*2)/4];
end