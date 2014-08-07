%TODO: compare against original implementation!

n = 64;
m = 20;
D = 3;

ls = 1;
noise = 1;
sf2 = 2;
x = randn(n, D);
y = randn(n, 1);
xs = randn(2, D);
hyp.lik = log(noise);
hyp.cov = [log(ls); log(sf2)/2];
S = randn(m, D) / ls;

cov = {@covSSnaive, S};

[ymu ys2 fmu fs2] = gp(hyp, @infExact, [], cov, @likGauss, x, y, xs);

hyp.weight_prior = sf2*ones(2*m, 1)/m;

cov_deg = {@covDegenerate, {@degSS, S}};
[ymuE ys2E fmuE fs2E] = gp(hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y, xs);
ymu - ymuE
ys2 - ys2E
fmu - fmuE
fs2 - fs2E

[nlZ] = gp(hyp, @infExact, [], cov, @likGauss, x, y);
[nlZE] = gp(hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y);
nlZ - nlZE

