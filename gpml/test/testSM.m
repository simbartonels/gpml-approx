function testSM()
    %testAgainstFullGP();
    %testImplAgainstFITC();
    %testImplAgainstNaive();
    testImplAndFITC();
end

function testAgainstFullGP()
[sd, n, D, x, y, xs, logell, lsf2, lsn2] = initEnv();
rng(sd);

% Assert that if using M=n inducing points with U=X that we have then the
% original GP.
hyp.lik = lsn2;
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
end

function testFullAgainstFITC(hyp, V, x, y, xs, ymuE, ys2E, nlZtrue)
%first check against SPGP/FITC
% TODO: the noise on the inducing inputs makes this test fail
[ymuS, ys2S, fmuS, fs2S] = gp(hyp, @infFITC, [], {@covFITC, {@covSEard}, V}, @likGauss, x, y, xs);
nlZS = gp(hyp, @infFITC, [], {@covFITC, {@covSEard}, V}, @likGauss, x, y);
%nlZS = 0;
worst_dev_fitc = [max((ymuE-ymuS).^2), max((ys2E-ys2S).^2), nlZtrue - nlZS]
end

function testFullAgainstNaive(smhyp, V, x, y, xs, ymuE, ys2E, nlZtrue)
M = size(V, 1);
%then let's make sure the naive implementation is correct
[ymu, ys2, fmu, fs2] = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y, xs);
nlZN = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y);
%nlZN = 0;
%should deal the same output as the GP
worst_dev_naive = [max((ymu-ymuE).^2), max((ys2-ys2E).^2), nlZtrue - nlZN]
end

function testFullAgainstImpl(smhyp, V, x, y, xs, ymuE, ys2E, nlZtrue)
M = size(V, 1);
%now the actual implementation
[ymu, ys2, fmu, fs2] = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y, xs);
nlZ = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y);

worst_dev_impl = [max((ymu-ymuE).^2), max((ys2-ys2E).^2), nlZtrue - nlZ]
end

function testImplAgainstFITC()
[sd, n, D, x, y, xs, logell, lsf2, lsn2] = initEnv();
rng(sd);

% Now let's check that the naive and actual implementation agree.
M = n - 2;
V = randn([M, D]);
logsigma = repmat(logell', M, 1);

smhyp.lik =lsn2;
smhyp.M = M;
smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
    reshape(V, [M*D, 1]); lsf2+(log(2*pi)*D+sum(logell)*2)/4];
[ymu, ys2, fmu, fs2] = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y, xs);
nlZ = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y);

[ymuN, ys2N, fmuN, fs2N, ls, postN] = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y, xs);
nlZnaive = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y);

dev_impl_naive = [max((ymu-ymuN).^2) ,max((ys2-ys2N).^2), nlZnaive - nlZ]

% If the length scales of the individual basis functions are equal to the 
% GPs length scales the method becomes equivalent Snelson's SPGP/FITC 
% method.
% TODO: FITC adds noise on the inducing inputs!
hyp.lik = lsn2;
hyp.cov = [logell; lsf2];
[ymuS, ys2S, fmuS, fs2S] = gp(hyp, @infFITC, [], {@covFITC, {@covSEard}, V}, @likGauss, x, y, xs);
nlZS = gp(hyp, @infFITC, [], {@covFITC, {@covSEard}, V}, @likGauss, x, y);
dev_impl_fitc = [max((ymu-ymuS).^2), max((ys2-ys2S).^2), nlZS - nlZ]
end

function testImplAgainstNaive()
[sd, n, D, x, y, xs, logell, lsf2, lsn2] = initEnv();
rng(sd);

M = n - 2;
V = randn([M, D]);
%make sure length scale parameters are larger than half of the original ls
logsigma = log(exp(randn([M, D]))+repmat(exp(logell)', M, 1)/2);

smhyp.lik = lsn2;
smhyp.M = M;
smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
    reshape(V, [M*D, 1]); lsf2+(log(2*pi)*D+sum(logell)*2)/4];
[ymu, ys2, fmu, fs2] = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y, xs);
nlZ = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y);

[ymuN, ys2N, fmuN, fs2N, ls, postN] = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y, xs);
nlZnaive = gp(smhyp, @infExact, [], {@covSMnaive, M}, @likGauss, x, y);

dev_impl_naive = [max((ymu-ymuN).^2) ,max((ys2-ys2N).^2), nlZnaive - nlZ]
end

function testImplAndFITC()
[sd, n, D, x, y, xs, logell, lsf2, lsn2] = initEnv();
rng(sd);

% Now let's check that the naive and actual implementation agree.
M = n - 2;
V = randn([M, D]);
%make sure length scale parameters are larger than half of the original ls
logsigma = log(exp(randn([M, D]).^2)+repmat(exp(logell)', M, 1)/2);

smhyp.lik = lsn2;
smhyp.M = M;
smhyp.cov = [logell; reshape(logsigma, [M*D, 1]); ...
    reshape(V, [M*D, 1]); lsf2+(log(2*pi)*D+sum(logell)*2)/4];
[ymu, ys2, fmu, fs2] = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y, xs);
nlZ = gp(smhyp, @infSM, [], {@covSM, M}, @likGauss, x, y);

% If the length scales of the individual basis functions are equal to the 
% GPs length scales the method becomes equivalent Snelson's SPGP/FITC 
% method.
% TODO: FITC adds noise on the inducing inputs!
[ymuS, ys2S, fmuS, fs2S] = gp(smhyp, @infFITC, [], {@covSM, M}, @likGauss, x, y, xs);
nlZS = gp(smhyp, @infFITC, [], {@covSM, M}, @likGauss, x, y);
dev_impl_and_fitc = [max((ymu-ymuS).^2), max((ys2-ys2S).^2), nlZS - nlZ]
end

function [sd, n, D, x, y, xs, logell, lsf2, lsn2] = initEnv()
sd = floor(rand(1) * 32000)
n = 5;
D = 3;
x = rand(n, D);
y = randn(n, 1);
xs = rand(2, D);
logell = randn(D, 1);
lsf2 = log(randn(1)^2);
lsn2 = log(rand(1)^2);
end