% Tests for the Sparse Multiscale GP regression implementation.
n = 5;
D = 3;
x = rand(n, D);
y = randn(n, 1);
xs = rand(2, D);

% Assert that if using M=n inducing points with U=X that we have then the
% original GP.
hyp.lik = log(randn(1)^2);
logell = randn(D, 1);
lsf2 = log(randn(1)^2);
hyp.cov = [logell; lsf2];
[ymuE, ys2E, fmuE, fs2E, ls, postE] = gp(hyp, @infExact, [], @covSEard, @likGauss, x, y, xs);

smhyp.lik = hyp.lik;
smhyp.v = x;
smhyp.logsigma = repmat(logell', size(smhyp.v, 1), 1);
smhyp.cov = hyp.cov;
concreteCov = {@covSM, smhyp.v, smhyp.logsigma};
%should deal the same output as the full GP
[ymu, ys2, fmu, fs2] = gp(smhyp, @infSM, [], concreteCov, @likGauss, x, y, xs);

worst_dev = max((ymu-ymuE).^2)
worst_dev = max((ys2-ys2E).^2)

nlZtrue = gp(hyp, @infExact, [], @covSEard, @likGauss, x, y);
nlZ = gp(smhyp, @infSM, [], concreteCov, @likGauss, x, y);
diff_in_llh = nlZtrue - nlZ


% Now let's check that the naive and actual implementation agree.
M = n - 2;
%TODO: make this random again
smhyp.lik = hyp.lik;
smhyp.v = randn([M, D]);
%make sure length scale parameters are larger than half of the original ls
smhyp.logsigma = log(exp(randn([M, D]))+repmat(exp(logell)', M, 1)/2);
smhyp.cov = hyp.cov;
concreteCov = {@covSM, smhyp.v, smhyp.logsigma};
[ymu, ys2, fmu, fs2] = gp(smhyp, @infSM, [], concreteCov, @likGauss, x, y, xs);
nlZ = gp(smhyp, @infSM, [], concreteCov, @likGauss, x, y);

naive.lik = smhyp.lik;
naive.cov = [logell; reshape(smhyp.logsigma, [M*D, 1]); ...
    reshape(smhyp.v, [M*D, 1]); lsf2];
[ymuN, ys2N, fmuN, fs2N, ls, postN] = gp(naive, @infExact, [], {@covSMnaive, M}, @likGauss, x, y, xs);

dev_to_naive = max((ymu-ymuN).^2)
dev_to_naive = max((ys2-ys2N).^2)

nlZnaive = gp(naive, @infExact, [], {@covSMnaive, M}, @likGauss, x, y);
diff_in_llh = nlZnaive - nlZ

%postE.L - postN.L