%TODO: compare against original implementation! Especially likelihood!

n = 64;
m = 20;
D = 3;

ls = exp(randn(1));
noise = exp(randn(1));
sf2 = exp(randn(1));
x = randn(n, D);
y = randn(n, 1);
xs = randn(2, D);
hyp.lik = log(noise)/2;
hyp.cov = [log(ls); log(sf2)/2];
S = randn(m, D) / ls;

cov = {@covSSnaive, S};

[ymu ys2 fmu fs2] = gp(hyp, @infExact, [], cov, @likGauss, x, y, xs);

hyp.weight_prior = sf2*ones(2*m, 1)/m;

cov_deg = {@covDegenerate, {@degSS, S}};
[ymuE ys2E fmuE fs2E] = gp(hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y, xs)
ymu - ymuE
ys2 - ys2E
fmu - fmuE
fs2 - fs2E

[nlZ] = gp(hyp, @infExact, [], cov, @likGauss, x, y);
[nlZE] = gp(hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y)
nlZ - nlZE

me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
addpath([mydir,'testSS'])

params = [log(ls)*ones(D, 1); log(sf2)/2; log(noise)/2; 2*pi*ls*reshape(S, [m*D, 1])];
[mu S2] = ssgprfixed(params, x, y, xs);
mu - ymuE
S2 - ys2E
nlZO = ssgprfixed(params, x, y);
nlZ - nlZO