function testHSMiso()
    testToyExample();
    testGradients();
    testAgainstNaiveImplementation();
end

function testGradients()
    [x, y, ~, hyp] = initEnv();
    hyp.cov = [randn(1); randn(1)];
    M = 2;
    D = size(x, 2);
    L = 1.2 * max(abs(x));%rand(1, D);
    [J, lambda] = initHSM(M, D, L);

    cov_deg = {@covDegenerate, {@degHSMiso, M, L, J, lambda}};

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
    D = 1;
    M = 64;
    n = 1;
    z = 1;
    %L = 3 * rand(1, D);
    L = ones(1, D);
    x = rand(n, D) / 2;
    xs = rand(z, D) / 2;
    hyp.lik = 0;
    logls = log(0.1);
    logsf2 = 0;
    hyp.cov = [logls; logsf2];

    [J, lambda] = initHSM(M, D, L);

    phix = degHSMiso(M, L, J, lambda, hyp.cov, x);
    phixs = degHSMiso(M, L, J, lambda, hyp.cov, xs);
    weight_prior = degHSMiso(M, L, J, lambda, hyp.cov);

    diff = phix'*diag(weight_prior)*phixs - covSEiso(hyp.cov, x, xs)
    if abs(diff) > 1e-15, error('Toy example appears broken.'); end
end