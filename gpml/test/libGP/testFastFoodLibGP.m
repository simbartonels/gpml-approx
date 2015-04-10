function testFastFoodLibGP()
	testComplianceWithMATLAB()
	testFastFoodApproximationQuality()
end

function testComplianceWithMATLAB()
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


function testFastFoodApproximationQuality()
    seed = randi(32000)
    %seed = 28033
    %seed = 1924
    %seed = 23076
    rng(seed);
    D = 10;
    m = 1024;
    n = 4000;
    nt = 1;
    hyp.cov = [randn(D+1, 1)];
    hyp.cov(D+1) = 0;
    hyp.lik = randn(1);
    x = rand(n, D);
    %y = randn(n, 1);
    bf = {'FastFood'};
    intD = 2^nextpow2(D);
    M = m*intD; %not *2 because we only need it for the matrices
    iSigma = covDegFast(bf, seed, 2*M, D, unwrap(hyp));
    Sigma = diag(1./diag(iSigma));
    phix = covDegFast(bf, seed, 2*M, D, unwrap(hyp), [], x);
    %phiz = covDegFast(bf, seed, 2*M, D, unwrap(hyp), [], z);
    phiz = phix;
    Ktarget = covSEard(hyp.cov, x);
    quality = mean(mean(abs(Ktarget - phix'*Sigma*phiz)))
    if quality > 0.025, error('Quality is worse than what authors report in article.'); end
end

function testFastFoodApproximationQuality_old()
    seed = randi(32000)
    %seed = 28033
    %seed = 1924
    %seed = 23076
    rng(seed);
    D = 2;
    m = 200;
    n = 3;
    nt = 1;
    hyp.cov = [randn(D+1, 1)];
    hyp.cov(D+1) = 0;
    hyp.lik = randn(1);
    x = randn(n, D);
    y = randn(n, 1);
    z = randn(nt, D);

    lambda = exp(2*hyp.lik);
    %Random Kitchen Sinks
	 % Training
	w = randn(m*D, D);
	Z = [cos(w*x'); sin(w*x')]; %exp(i*w*x');
	alpha = (eye(2*m*D)*lambda+Z*Z')\(Z*y(:));

	 % testing
	mF4 = alpha(:)'*[cos(w*z'); sin(w*z')];

    [mF, s2F, ~, ~, nlZ, post] = gp(hyp, @infExact, [], {@covSEard}, @likGauss, x, y, z);
    checkError(mF, mF4, 'fullGP', 'Random Kitchen Sinks', 'mean');
    %z = x;
    [s, g, randpi, b] = initFastFood(m, D, []);

    fastFoodBF = {@degFastFood, s, g, randpi, b};
    degCov = {@covDegenerateNaive, fastFoodBF};
    ff_hyp = hyp;
    ff_hyp.cov = [hyp.cov(D+1); hyp.cov(1:D)];
    [mF3, s2F3] = gp(ff_hyp, @infExact, [], degCov, @likGauss, 2*pi*x, y, 2*pi*z);

    checkError(mF, mF3, 'fullGP', 'FastFood', 'mean');

    Dintern = 2^nextpow2(D);
    shape = [Dintern, m];
    extra = [reshape(s, shape); reshape(g, shape); reshape(b, shape)];
    %TODO is this M correct?
    [alpha2, L2, nlZ2, mF2, s2F2] = infLibGPmex(x, y, z, 'degenerate', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), 2*m*Dintern, 'FastFood', seed, extra);

    checkError(mF, mF2, 'GPML', 'LibGP', 'mean');
    checkError(s2F, s2F2, 'GPML', 'LibGP', 'variance');
end
