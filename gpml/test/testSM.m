sd = floor(rand(1) * 32000)
%sd = 16594 %difference for llh is largern than e-15 probably due to large
%values of llh
%sd = 21128;
rng(sd);
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
[ymuE, ys2E, fmuE, fs2E, ~, postE] = gp(hyp, @infExact, [], @covSEard, @likGauss, x, y, xs);
nlZtrue = gp(hyp, @infExact, [], @covSEard, @likGauss, x, y);
%nlZtrue = 0;

V = x;
M = size(V, 1);
logsigma = repmat(logell', M, 1);
smhyp.lik = hyp.lik;
smhyp.M = M;
smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
    reshape(V, [M*D, 1]); lsf2+(log(2*pi)*D+sum(logell)*2)/4];

%first check against SPGP/FITC
% TODO: the noise on the inducing inputs makes this test fail
[ymuS, ys2S, fmuS, fs2S] = gp(hyp, @infFITC, [], {@covFITC, {@covSEard}, V}, @likGauss, x, y, xs);
nlZS = gp(hyp, @infFITC, [], {@covFITC, {@covSEard}, V}, @likGauss, x, y);
%nlZS = 0;
worst_dev_fitc = [max((ymuE-ymuS).^2), max((ys2E-ys2S).^2), nlZtrue - nlZS]

%then let's make sure the naive implementation is correct
[ymu, ys2, fmu, fs2] = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y, xs);
nlZN = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y);
%nlZN = 0;
%should deal the same output as the GP
worst_dev_naive = [max((ymu-ymuE).^2), max((ys2-ys2E).^2), nlZtrue - nlZN]

%now the actual implementation
[ymu, ys2, fmu, fs2] = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y, xs);
nlZ = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y);

worst_dev_impl = [max((ymu-ymuE).^2), max((ys2-ys2E).^2), nlZtrue - nlZ]


% Now let's check that the naive and actual implementation agree.
M = n - 2;
V = randn([M, D]);
%make sure length scale parameters are larger than half of the original ls
%logsigma = log(exp(randn([M, D]))+repmat(exp(logell)', M, 1)/2);
logsigma = repmat(logell', M, 1);

smhyp.lik = hyp.lik;
smhyp.M = M;
smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
    reshape(V, [M*D, 1]); lsf2+(log(2*pi)*D+sum(logell)*2)/4];
%smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
%    reshape(V, [M*D, 1]); lsf2];
[ymu, ys2, fmu, fs2] = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y, xs);
nlZ = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y);
%nlZ = 0;
%naive.lik = smhyp.lik;
%naive.cov = [logell; reshape(logsigma, [M*D, 1]); ...
%    reshape(V, [M*D, 1]); lsf2];
[ymuN, ys2N, fmuN, fs2N, ls, postN] = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y, xs);
nlZnaive = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y);

dev_impl_naive = [max((ymu-ymuN).^2) ,max((ys2-ys2N).^2), nlZnaive - nlZ]

% If the length scales of the individual basis functions are equal to the 
% GPs length scales the method becomes equivalent Snelson's SPGP/FITC 
% method.
% TODO: FITC adds noise on the inducing inputs!
[ymuS, ys2S, fmuS, fs2S] = gp(hyp, @infFITC, [], {@covFITC, {@covSEard}, V}, @likGauss, x, y, xs);
nlZS = gp(hyp, @infFITC, [], {@covFITC, {@covSEard}, V}, @likGauss, x, y);
dev_impl_fitc = [max((ymu-ymuS).^2), max((ys2-ys2S).^2), nlZS - nlZ]
clear