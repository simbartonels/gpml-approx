function testSolinLibGP()
    %sd = 19023
    [x, y, xs, hyp] = initEnv();
    [n, D] = size(x);
    actualM = n - 1;
    M = floor(actualM^(1/D));

    l = 4 * 2.2 * ones([1, D]) / 3;
    [J, lambda] = initHSM(M, D, l);
    [nlZ_o, dnlZ_o, post] = gp(hyp, @infExactDegKernel, [], {@covDegenerate, {@degHSM2, M, l, J, lambda}}, @likGauss, x, y);
    alpha_o = post.alpha;
    L_o = post.L;
    %[ymuS, ys2S, ~, ~, ~, post] = gp(smhyp, @infSMfast, [], {@covSM, M}, @likGauss, x, y, xs);
    [nlZ, dnlZ, post] = gp(hyp, @infSolinfast, [], {@covDegFast, {@degHSM2, M, l, J, lambda}, 0, M^D}, @likGauss, x, y);
    post.alpha = post.alpha(1:M^D);
    post.L = post.L(1:M^D, 1:M^D);
    %post.L = solve_chol(post.L, eye(size(post.L, 1)))*exp(2*hyp.lik);
    diff = max(max(abs((L_o - post.L)./L_o)));
    if diff > 1e-10
        diff
        disp('Compare the value of L here and in the libGP implementation.');
        error('Check computation of L matrix!');
    end
    diff = max(abs((alpha_o - post.alpha)./alpha_o));
    if diff > 1e-10
        diff
        error('Check computation of alpha!');
    end
    diff = abs((nlZ_o - nlZ)./nlZ_o);
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
    
    [mF_o, s2F_o] = gp(hyp, @infExactDegKernel, [], {@covDegenerate, {@degHSM2, M, l, J, lambda}}, @likGauss, x, y, xs);
    [mF, s2F] = gp(hyp, @infSolinfast, [], {@covDegFast, {@degHSM2, M, l, J, lambda}, 0, actualM}, @likGauss, x, y, xs);
    diff = max(max(abs((mF_o - mF)./mF_o)));
    if diff > 1e-5, error('Mean predictions disagree!'); end
    diff = max(max(abs((s2F - s2F_o)./s2F_o)));
    if diff > 1e-5, error('Variance predictions disagree!'); end

    disp('Test completed succesfully.');
end
