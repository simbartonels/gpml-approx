D = 2;
m = 3;
n = 2*m.^D;
z = 5;

ls = exp(randn(1));
noise = exp(randn(1));
sf2 = exp(randn(1));

x = randn(n, D);
y = randn(n, 1);
xs = randn(2, D);
hyp.lik = log(noise)/2;
hyp.cov = [log(ls); log(sf2)/2];

hyp.weight_prior = sf2*ones(2*m, 1)/m;
[s, gpi, b] = initFastFood(m, D, hyp.cov);
cov_deg = {@covDegenerate, {@degFastFood, s, gpi, b}};
covSEiso(hyp.cov, x, 'diag') - feval(cov_deg{:}, hyp.cov, x, 'diag')
%covSEiso(hyp.cov, x, xs) - 
feval(cov_deg{:}, hyp.cov, x, xs)
%[ymuE ys2E fmuE fs2E] = gp(hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y, xs)
