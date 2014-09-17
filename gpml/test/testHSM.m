function testHSM()
    testToyExample();
    testGradients();
    testAgainstNaiveImplementation();
end

function testGradients()
    [sd, n, D, x, y, xs, logell, lsf2, lsn2] = initEnv();
    sd
    rng(sd);
    hyp.lik = lsn2;
    logell = logell(1);
    hyp.cov = [logell; lsf2];
    M = 2;
    L = rand(1, D);
    [J, lambda] = initHSM(M, D, L);

    cov_deg = {@covDegenerate, {@degHSM, M, L, J, lambda}};

    options = optimoptions(@fmincon,'Algorithm','interior-point',...
        'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);
    optfunc = @(hypx) optimfunc(hypx, hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y);

    %derivative check
    fmincon(optfunc,...
               unwrap(hyp),[],[],[],[],[],[],@unitdisk,options);
end

function testAgainstNaiveImplementation()
    error('Not implemented!');
end

function testToyExample()
D = 2;
M = 32;
n = 2*M^D;
z = 5;
%L = 3 * rand(1, D);
L = ones(1, D);
x = rand(n, D) / 2;
y = randn(n, 1);
xs = rand(z, D) / 2;
hyp.lik = 0;
logls = log(0.1);
logsf2 = 0;
hyp.cov = [logls; logsf2];

[J, lambda] = initHSM(M, D, L);

cov2 = {@covDegenerate, {@degHSM, M, L, J, lambda}};
[mF s2F] = gp(hyp, @infExactDegKernel, [], cov2, @likGauss, x, y, xs);

[mFo s2Fo] = gp(hyp, @infExact, [], @covSEiso, @likGauss, x, y, xs);

[mF - mFo, s2F - s2Fo]
end