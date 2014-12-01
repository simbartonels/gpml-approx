function testFastFoodLibGP()
    [x, y, ~, hyp] = initEnv();
    hyp.mean = [];
    D = size(x, 2);
    m = 1;
    intD = 2^nextpow2(D);
    M = m*intD; %not *2 because we only need it for the matrices
    [alpha, L, nlZ, s, g, randpi, b] = infFastFoodmex(2*M, unwrap(hyp), x, y);
    L = solve_chol(L, eye(size(L, 1)))*exp(2*hyp.lik);
    s = reshape(s', [M, 1]);
    g = reshape(g', [M, 1]);
    randpi = reshape(randpi', [M, 1]);
    randpi = randpi + 1;
    b = reshape(b', [M, 1]);
    degCov = {@covDegenerate, {@degFastFood, s, g, randpi, b}};
    ff_hyp = hyp;
    ff_hyp.cov = [hyp.cov(D+1); hyp.cov(1:D)];
    [post, nlZ_o] = infExactDegKernel(ff_hyp, {@meanZero}, degCov, @likGauss, x, y);
    alpha_o = post.alpha;
    L_o = post.L;
    diff = max(max(abs(L_o - L)));
    if diff > 1e-10
        error('Check computation of L matrix!');
    end
    diff = max(abs(alpha_o - alpha));
    if diff > 1e-10
        error('Check computation of alpha!');
    end
    diff = abs(nlZ_o - nlZ);
    if diff > 1e-10
        diff
        error('Check computation of nlZ!');
    end
    disp('Test complete.');
end
