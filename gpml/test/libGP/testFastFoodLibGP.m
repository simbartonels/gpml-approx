function testFastFoodLibGP()
    sd = 11853;
    [x, y, z, hyp] = initEnv(sd);
    hyp.mean = [];
    D = size(x, 2);
    m = 1;
    intD = 2^nextpow2(D)
    M = m*intD %not *2 because we only need it for the matrices
    [alpha, L, nlZ, dnlZ, s, g, randpi, b] = infFastFoodmex(2*M, unwrap(hyp), x, y);
    %something's awfully wrong...
    %L = solve_chol(L, eye(size(L, 1)))
    if any(any(isnan(alpha)))
	alpha
	error('alpha contains NaN!');
    end
    L = L'\(L\eye(size(L, 1)));
    L = L * exp(2*hyp.lik);
    s = reshape(s', [M, 1]);
    g = reshape(g', [M, 1]);
    randpi = reshape(randpi', [M, 1]);
    randpi = randpi + 1;
    b = reshape(b', [M, 1]);
    fastFoodBF = {@degFastFood, s, g, randpi, b};
    degCov = {@covDegenerate, fastFoodBF};
    ff_hyp = hyp;
    ff_hyp.cov = [hyp.cov(D+1); hyp.cov(1:D)];
    [post, nlZ_o, dnlZ_o] = infExactDegKernel(ff_hyp, {@meanZero}, degCov, @likGauss, x, y);
    dnlZ_o.cov = [dnlZ_o.cov(2:D+1); dnlZ_o.cov(1)];
    dnlZ_o = unwrap(dnlZ_o);
    alpha_o = post.alpha;
    L_o = post.L;
    diff = max(max(abs(L_o - L)./L_o));
    if diff > 1e-10
	%L_o
        diff = L_o - L
        error('Check computation of L matrix!');
    end
    diff = max(abs((alpha_o - alpha)./alpha_o));
    if diff > 1e-10
        error('Check computation of alpha!');
    end
    diff = abs(nlZ_o - nlZ);
    if diff > 1e-10
        diff
        error('Check computation of nlZ!');
    end
    diff = max(abs((dnlZ_o - dnlZ)./dnlZ_o));
    dnlZ
    if diff > 1e-10
	abs(dnlZ_o - dnlZ)
	error('Check computation of gradients!');
    end
end

function testApproximationQuality()
    D = 4;
    m = 4;
    x = randn([1, D]);
    z = randn([1, D]);
    hyp.mean = [];
    intD = 2^nextpow2(D)
    M = m*intD %not *2 because we only need it for the matrices
    [~, ~, ~, ~, s, g, randpi, b] = infFastFoodmex(2*M, unwrap(hyp), x, y);
    s = reshape(s', [M, 1]);
    g = reshape(g', [M, 1]);
    randpi = reshape(randpi', [M, 1]);
    randpi = randpi + 1;
    b = reshape(b', [M, 1]);
    fastFoodBF = {@degFastFood, s, g, randpi, b};
    degCov = {@covDegenerate, fastFoodBF};
    ff_hyp = hyp;
    ff_hyp.cov = [hyp.cov(D+1); hyp.cov(1:D)];

    phix = feval(fastFoodBF{:}, ff_hyp.cov, x);
    sigma = diag(feval(fastFoodBF{:}, ff_hyp.cov));
    phiz = feval(fastFoodBF{:}, ff_hyp.cov, z);
    khat = phix' * sigma * phiz;
    k = covSEard(hyp, x, z);
    approx_quality = max(abs(khat-k)./k)
    disp('Test complete.');
end
