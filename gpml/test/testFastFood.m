D = 16;
m = 64;
n = 1; %m*D+1;
z = 1;

ls = 1; %exp(randn(1));
noise = 1; %exp(randn(1));
sf2 = 1; %exp(randn(1));

x = rand(n, D) / 2;
y = randn(size(x, 1), 1);
xs = rand(z, D);
hyp.lik = log(noise)/2;
hyp.cov = [log(ls); log(sf2)/2];

d = 2^nextpow2(D);
hyp.weight_prior = sf2*ones(2*m*d, 1)/(m*d);
[s, gpi, b] = initFastFood(m, D, hyp.cov);
cov_deg = {@covDegenerate, {@degFastFood, s, gpi, b}};
phi = feval(cov_deg{:}, hyp.cov, NaN, x);
phiz = feval(cov_deg{:}, hyp.cov, NaN, xs);
K = phi' * diag(hyp.weight_prior) * phiz
covSEiso(hyp.cov, x, xs)
sum(sum(abs((covSEiso(hyp.cov, x, xs) - K).^2))/(n*z))

s = ones([m*d, 1]) / sqrt(d*ls);
cov_deg = {@covDegenerate, {@degFastFood, s, gpi, b}};
phi = feval(cov_deg{:}, hyp.cov, NaN, x);
phiz = feval(cov_deg{:}, hyp.cov, NaN, xs);
K = phi' * diag(hyp.weight_prior) * phiz
sum(sum(abs((covSEiso(hyp.cov, x, xs) - K).^2))/(n*z))

%covSEiso(hyp.cov, x, xs) - 
%feval(cov_deg{:}, hyp.cov, x, xs)
%[ymuE ys2E fmuE fs2E] = gp(hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y, xs)
