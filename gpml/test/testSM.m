% Tests for the Sparse Multiscale GP regression implementation.
n = 5;
D = 3;
x = rand(n, D);
y = randn(n, 1);
xs = rand(2, D);

% Assert that if using M=n inducing points with U=X that we have then the
% original GP.
hyp.lik = randn(1);
logell = randn(D, 1);
lsf2 = log(randn(1)^2);
hyp.cov = [logell; lsf2];
[ymuE, ys2E, fmuE, fs2E, ls, postE] = gp(hyp, @infExact, [], @covSEard, @likGauss, x, y, xs);
nlZtrue = gp(hyp, @infExact, [], @covSEard, @likGauss, x, y);

V = x;
M = size(V, 1);
logsigma = repmat(logell', M, 1);
smhyp.lik = hyp.lik;
smhyp.M = M;
smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
    reshape(V, [M*D, 1]); lsf2+(log(2*pi)*D+sum(logell))/2];

%first let's make sure the naive implementation is correct
[ymu, ys2, fmu, fs2] = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y, xs);
%should deal the same output as the GP
worst_dev = max((ymu-ymuE).^2)
worst_dev = max((ys2-ys2E).^2)
nlZN = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y);
diff_in_llh = nlZtrue - nlZN


%should deal the same output as the GP
[ymu, ys2, fmu, fs2] = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y, xs);

worst_dev = max((ymu-ymuE).^2)
worst_dev = max((ys2-ys2E).^2)

nlZ = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y);
diff_in_llh = nlZtrue - nlZ


% Now let's check that the naive and actual implementation agree.
M = n - 2;
V = randn([M, D]);
%make sure length scale parameters are larger than half of the original ls
logsigma = log(exp(randn([M, D]))+repmat(exp(logell)', M, 1)/2);

smhyp.lik = hyp.lik;
smhyp.M = M;
smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
    reshape(V, [M*D, 1]); lsf2];
[ymu, ys2, fmu, fs2] = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y, xs);
nlZ = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y);

%naive.lik = smhyp.lik;
%naive.cov = [logell; reshape(logsigma, [M*D, 1]); ...
%    reshape(V, [M*D, 1]); lsf2];
[ymuN, ys2N, fmuN, fs2N, ls, postN] = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y, xs);

dev_to_naive = max((ymu-ymuN).^2)
dev_to_naive = max((ys2-ys2N).^2)

nlZnaive = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y);
diff_in_llh = nlZnaive - nlZ

%postE.L - postN.L