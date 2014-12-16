function testSolinLibGP()
    sd = 19023
    [x, y, ~, hyp] = initEnv(sd);
    [n, D] = size(x);
    actualM = n - 1;
    M = floor(actualM^(1/D));

    L = 1.2 * ones([1, D]);
    [J, lambda] = initHSM(M, D, L);
    [nlZ_o, dnlZ_o, post] = gp(hyp, @infExactDegKernel, [],{@covDegenerate, {@degHSM2, M, L, J, lambda}}, @likGauss, x, y);
    alpha_o = post.alpha;
    L_o = post.L;
    %[ymuS, ys2S, ~, ~, ~, post] = gp(smhyp, @infSMfast, [], {@covSM, M}, @likGauss, x, y, xs);
    [nlZ, dnlZ, post] = gp(hyp, @infSolinfast, [], {@degHSM2, actualM}, @likGauss, x, y);
    post.L = post.L(1:M^D, 1:M^D);
    L = post.L;
    L = L'\(L\eye(size(L)))*exp(2*hyp.lik);
    post.L = L;
    %post.L = solve_chol(post.L, eye(size(post.L, 1)))*exp(2*hyp.lik);
    L_o
    post.L
    %TODO: check computation of L. Too many NaN!
    diff = max(max(abs(L_o - post.L)));
    if diff > 1e-10
        diff
        error('Check computation of L matrix!');
    end
    diff = max(abs(alpha_o - post.alpha));
    if diff > 1e-10
        error('Check computation of alpha!');
    end
    diff = abs(nlZ_o - nlZ);
    if diff > 1e-10
        diff
        error('Check computation of nlZ!');
    end
    [diff, arg] = max(abs(unwrap(dnlZ_o)-unwrap(dnlZ)));
    if diff > 1e-10
        diff
        arg
        error('Check gradients of nlZ!');
    end
end