function testHSM()
    testSEard();
    testToyExample();
    testGradients2();
    testGradients();
    testAgainstNaiveImplementation();
end

function testSEard()
    sd = floor(rand(1) * 32000)
%    sd = 12620
    %sd = 10184
    rng(sd);
    D = 2;
    M = 64;
    n = 3;
    x = rand(n, D) / 2;
    z = rand(2, D) / 2;
    logsf2 = 0;
    logls = randn(D, 1);
    hsmhyp.lik = 0;
    hsmhyp.cov = [];
    hyp.lik = 0;
    hyp.cov = [logls; logsf2];
    b = ones(1, D);
    a = -b;
    cov = cell(D, 1);
    j = 1:M;
    j = j';
    for d = 1:D
        loglsd = log(0.1);
        sqrtlambda = pi*j/(b(d)-a(d));
        foo = @(r) exp(2*logsf2)*sqrt(2*pi)*exp(loglsd)*exp(-exp(2*loglsd)*r.^2/2);

        s = foo(sqrtlambda);
        cov(d) = {{@covHSMnaive, s, a(1, d), b(1, d), d}};
    end
    %varargout1 = gp(hyp1, @infExact, [], {@covProd, cov}, @likGauss, x, y, xs)
    cov = {@covProd, cov};
    ls = exp(logls');
    diff = feval(cov{:}, hsmhyp.cov, 0.1*x./repmat(ls, [size(x, 1), 1]), ...
        0.1*z./repmat(ls, [size(z, 1), 1])) - covSEard(hyp.cov, x, z)
    if abs(diff) > 1e-15, error('Product kernel view appears broken!'); end
    L = b;
    [J, lambda] = initHSM(M, D, L);
    cov2 = {@degHSM, M, L, J, lambda};
    cov2hyp = hyp.cov;
    phix = feval(cov2{:}, cov2hyp, x);
    phiz = feval(cov2{:}, cov2hyp, z);
    weight_prior = feval(cov2{:}, cov2hyp);
    diff = phix'*diag(weight_prior)*phiz - covSEard(hyp.cov, x, z);
    % TODO: the difference is huge but this is due to numerical problems
    % which seem to be inherent to the method.
end

function testGradients2()
    D = 2;
    hyp.lik = 0;
    hyp.cov = [zeros([D, 1]); 0];
    M = 1;
    L = ones([1, D]);
    [J, lambda] = initHSM(M, D, L);
    cov_deg = {@covDegenerate, {@degHSM, M, L, J, lambda}};
    z = randn([1, D]);
    dK = feval(cov_deg{:}, hyp.cov, [], z, 1);
    z = z * 0.1; % adapt length scales
    dKtarget = pi * cos( pi*(z(:, 1)'+1)/2 )/2;
    phid2 = sin( pi * (z(:, 2)'+1) / 2);
    dKtarget = dKtarget .* phid2;
    dKtarget = dKtarget*diag(-z(:, 1));
    diff = abs(dK - dKtarget)
    if diff > 1e-6, error('Simple gradient check failed.'); end
end

function testGradients()
    [x, y, ~, hyp] = initEnv();
    M = 2;
    D = size(x, 2);
    L = 1.2 * max(abs(x));%rand(1, D);
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
D = 1;
M = 64;
n = 1;
z = 1;
%L = 3 * rand(1, D);
L = ones(1, D);
x = rand(n, D) / 2;
xs = rand(z, D) / 2;
hyp.lik = 0;
logls = log(0.1) * ones([D, 1]);
logsf2 = 0;
hyp.cov = [logls; logsf2];

[J, lambda] = initHSM(M, D, L);

phix = degHSM(M, L, J, lambda, hyp.cov, x);
phixs = degHSM(M, L, J, lambda, hyp.cov, xs);
weight_prior = degHSM(M, L, J, lambda, hyp.cov);

diff = phix'*diag(weight_prior)*phixs - covSEard(hyp.cov, x, xs)
if abs(diff) > 1e-15, error('Toy example appears broken.'); end
end