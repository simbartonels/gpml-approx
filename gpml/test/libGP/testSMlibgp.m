function testSMlibgp()
    %sd = 21988 %numerically very unstable scenario
    %sd = 8903 %llh breaks!
    %sd = 21371 %problems with L matrix?!
    [x, y, ~, smhyp] = initEnvSM(sd);
    D = size(x, 2);
    M = (numel(smhyp.cov)-1-D)/D/2;

    %logIndNoise = log(0);
    %[ymu, ys2] = gp(smhyp, @infExact, [], {@covSMnaive, M, logIndNoise}, @likGauss, x, y, xs);
    %[ymuS, ys2S, ~, ~, ~, post] = gp(smhyp, @infFITC, [], {@covSM, M}, @likGauss, x, y, xs);
    [nlZ_o, dnlZ_o, post] = gp(smhyp, @infFITC, [], {@covSM, M}, @likGauss, x, y);
    alpha_o = post.alpha;
    L_o = post.L;
    %[ymuS, ys2S, ~, ~, ~, post] = gp(smhyp, @infSMfast, [], {@covSM, M}, @likGauss, x, y, xs);
    [nlZ, dnlZ, post] = gp(smhyp, @infSMfast, [], {@covSM, M}, @likGauss, x, y);
    diff = max(max(abs(L_o - post.L)));
    if diff > 1e-10
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