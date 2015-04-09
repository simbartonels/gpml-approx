function testDegCovsLibGP()
%TESTDEGCOVS This function compares what the matlab implementation
%covariance functions produce against what the libGP implementations do.
    testFastFoodApproximationQuality();
    testFastFood();
	return;
    testSolin();
    testFIC();
    testSM();
    disp('Test passed sucessfully.');
end

function testSolin()
    seed = 0;
    [z, ~, ~, hyp] = initEnv();
    [n, D] = size(z);
    L = 4 * ones([1, D])
    M = floor((n-1)^(1/D));
    [J, lambda] = initHSM(M, D, L);
    bf = {@degHSM2, M, L, J, lambda};
    iSigma_o = diag(1./feval(bf{:}, hyp.cov));
    iSigma = covDegFast(bf, seed, M^D, D, unwrap(hyp));
    phi_o = feval(bf{:}, hyp.cov, z);
    phi = covDegFast(bf, seed, M^D, D, unwrap(hyp), [], z);
    checkError(phi_o, phi, 'GPML', 'LibGP', 'basis function');
    checkError(iSigma_o, iSigma, 'GPML', 'LibGP', 'weight prior');
end

function testSM()
    seed = 0;
    M = 10;
    [z, ~, ~, hyp] = initEnvSM(M);
    snu2 = 1e-6 * exp(2 * hyp.lik);
    D = size(z, 2);
    bf = {'SparseMultiScaleGP'};
    phi_o = covSM(M, hyp.cov, z, z); %yields Uvz
    phi = covDegFast(bf, seed, M, D, unwrap(hyp), [], z);
    checkError(phi_o, phi, 'GPML', 'LibGP', 'basis function');
    [~, iSigma_o, ~] = covSM(M, hyp.cov, z);
    %bfmex will return the inverse with inducing noise added
    iSigma_o = iSigma_o+snu2*eye(M);
    iSigma = covDegFast(bf, seed, M, D, unwrap(hyp));
    checkError(iSigma_o, iSigma, 'GPML', 'LibGP', 'weight prior');
    for p=1:size(hyp.cov)
	[~, iSigma_o, ~] = covSM(M, hyp.cov, z, [], p);
	iSigma = covDegFast(bf, seed, M, D, unwrap(hyp), [], [], p);
	checkError(iSigma_o, iSigma, 'GPML', 'LibGP', sprintf('weight prior gradient %d', p));
	[~, ~, phi_o] = covSM(M, hyp.cov, z, [], p);
    	phi = covDegFast(bf, seed, M, D, unwrap(hyp),[], z, p);
	checkError(phi_o, phi, 'GPML', 'LibGP', sprintf('basis function gradient %d', p));
    end
end

function testFIC()
    seed = 0;
    [z, ~, ~, hyp2] = initEnv();
    snu2 = 1e-6 * exp(2 * hyp2.lik);
    D = size(z, 2);
    M = 4;
    U = randn(M, D);
    logell = hyp2.cov(1:D);
    lsf2 = hyp2.cov(D+1);
    hyp = hyp2;
    hyp.cov = [logell; lsf2; reshape(U, [M*D, 1])];
    bf = {'FIC'};
    phi_o = covFITC({@covSEard}, U, hyp2.cov, z, z);
    phi = covDegFast(bf, seed, M, D, unwrap(hyp), [], z);
    checkError(phi_o, phi, 'GPML', 'LibGP', 'basis function');
    [~, iSigma_o, ~] = covFITC({@covSEard}, U, hyp2.cov, z);
    %bfmex will return the inverse with inducing noise added
    iSigma_o = iSigma_o+snu2*eye(M);
    iSigma = covDegFast(bf, seed, M, D, unwrap(hyp));
    checkError(iSigma_o, iSigma, 'GPML', 'LibGP', 'weight prior');

    for p=1:size(hyp2.cov)
	[~, ~, phi_o] = covFITC({@covSEard}, U, hyp2.cov, z, [], p);
    	phi = covDegFast(bf, seed, M, D, unwrap(hyp), [], z, p);
	checkError(phi_o, phi, 'GPML', 'LibGP', sprintf('basis function gradient %d', p));
	[~, iSigma_o, ~] = covFITC({@covSEard}, U, hyp2.cov, z, [], p);
	iSigma = covDegFast(bf, seed, M, D, unwrap(hyp), [], [], p);
	checkError(iSigma_o, iSigma, 'GPML', 'LibGP', sprintf('weight prior gradient %d', p));
    end
end

function testFastFood()
    seed = randi(32000)
    [z, ~, y, hyp] = initEnv(seed);
    [n, D] = size(z)
    bf = {'FastFood'};
    m = 2;
    intD = 2^nextpow2(D)
    M = m*intD %not *2 because we only need it for the matrices
    iSigma = covDegFast(bf, seed, 2*M, D, unwrap(hyp));
    phi = covDegFast(bf, seed, 2*M, D, unwrap(hyp), [], z);
    [alpha, L, nlZ, dnlZ, s, g, randpi, b] = infFastFoodmex(2*M, unwrap(hyp), z, y, seed);
    s = reshape(s', [M, 1]);
    g = reshape(g', [M, 1]);
    randpi = reshape(randpi', [M, 1]);
    randpi = randpi + 1;
    b = reshape(b', [M, 1]);
    hyp.cov = [hyp.cov(D+1); hyp.cov(1:D)];
    bf = {@covDegenerate, {@degFastFood, s, g, randpi, b}};
    iSigma_o = 1./feval(bf{:}, hyp.cov);
    checkError(iSigma_o, diag(iSigma), 'GPML', 'LibGP', 'weight prior');
    phi_o = feval(bf{:}, hyp.cov, [], z);
    checkError(phi_o, phi, 'GPML', 'LibGP', 'basis function');
end

function testFastFoodApproximationQuality()
    %seed = randi(32000)
    seed = 28033
    rng(seed);
    D = 4;
    m = 32;
    n = 3;
    nt = 2;
    hyp.cov = [randn(D+1, 1)];
    hyp.lik = randn(1);
    x = randn(n, D);
    y = randn(n, 1);
    z = randn(nt, D);
    %z = x;
    [s, g, randpi, b] = initFastFood(m, D, []);

    fastFoodBF = {@degFastFood, s, g, randpi, b};
    degCov = {@covDegenerateNaive, fastFoodBF};
    ff_hyp = hyp;
    ff_hyp.cov = [hyp.cov(D+1); hyp.cov(1:D)];
    phi = feval(fastFoodBF{:}, ff_hyp.cov, x)'
    [mF3, s2F3, ~, ~, nlZ3, post3] = gp(ff_hyp, @infExact, [], degCov, @likGauss, x, y, z);

    [mF, s2F, ~, ~, nlZ, post] = gp(hyp, @infExact, [], {@covSEard}, @likGauss, x, y, z);
    checkError(mF, mF3, 'fullGP', 'FastFood', 'mean');

    Dintern = 2^nextpow2(D);
    shape = [Dintern, m];
    extra = [reshape(s, shape); reshape(g, shape); reshape(b, shape)];
    %TODO is this M correct?
    [alpha2, L2, nlZ2, mF2, s2F2] = infLibGPmex(x, y, z, 'degenerate', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), 2*m*Dintern, 'FastFood', seed, extra);

    checkError(mF, mF2, 'GPML', 'LibGP', 'mean');
    checkError(s2F, s2F2, 'GPML', 'LibGP', 'variance');
end
