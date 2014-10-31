function testSMlibgp()
    %sd = 21988 %numerically very unstable scenario
    %sd = 8903 %llh breaks!
    [x, y, xs, smhyp] = initEnvConcrete();
    D = size(x, 2);
    M = (numel(smhyp.cov)-1-D)/D/2;

    %logIndNoise = log(0);
    %[ymu, ys2] = gp(smhyp, @infExact, [], {@covSMnaive, M, logIndNoise}, @likGauss, x, y, xs);
    %[ymuS, ys2S, ~, ~, ~, post] = gp(smhyp, @infFITC, [], {@covSM, M}, @likGauss, x, y, xs);
    [nlZ_o, ~, post] = gp(smhyp, @infFITC, [], {@covSM, M}, @likGauss, x, y);
    alpha_o = post.alpha;
    L_o = post.L;
    %[ymuS, ys2S, ~, ~, ~, post] = gp(smhyp, @infSMfast, [], {@covSM, M}, @likGauss, x, y, xs);
    [nlZ, ~, post] = gp(smhyp, @infSMfast, [], {@covSM, M}, @likGauss, x, y);
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
end

function [x, y, xs, smhyp] = initEnvConcrete(sd)
    if nargin == 0
        [x, y, xs, smhyp] = initEnv();
    else
        [x, y, xs, smhyp] = initEnv(sd);
    end
    [n, D] = size(x);
    logell = smhyp.cov(1:D);
    lsf2 = smhyp.cov(D+1);

    M = D+1;
    V = randn([M, D]);
    %make sure length scale parameters are larger than half of the original ls
    logsigma = log(exp(2*randn([M, D]).^2)+repmat(exp(2*logell)', M, 1)/2)/2;

    smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
        reshape(V, [M*D, 1]); lsf2+(log(2*pi)*D+sum(logell)*2)/4];
end