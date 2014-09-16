function testSS()
sd = 2423
sd = floor(rand(1) * 32000)
rng(sd);

n = 64;
m = 20;
D = 3;

ls = 1; exp(randn(1))
noise = 1; exp(randn(1))
sf2 = 1; %exp(randn(1));
x = randn(n, D);
y = randn(n, 1);
xs = randn(2, D);
hyp.lik = log(noise)/2;
hyp.cov = [log(ls); log(sf2)/2];

S = initSS(m, D, ls);

cov = {@covSSnaive, S};

[ymu ys2 fmu fs2] = gp(hyp, @infExact, [], cov, @likGauss, x, y, xs);
[nlZ] = gp(hyp, @infExact, [], cov, @likGauss, x, y);


cov_deg = {@covDegenerate, {@degSS, S}};
[ymuE ys2E fmuE fs2E] = gp(hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y, xs);
[nlZE] = gp(hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y);

diff_naive_impl = [ymu - ymuE; ys2 - ys2E; nlZ - nlZE]

me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
addpath([mydir,'testSS'])

params = [log(ls)*ones(D, 1); log(sf2)/2; log(noise)/2; 2*pi*ls*reshape(S, [m*D, 1])];
[mu S2] = ssgprfixed(params, x, y, xs);
nlZO = ssgprfixed(params, x, y);
diff_impl_orig = [mu - ymuE; S2 - ys2E; nlZE - nlZO]


options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);
optfunc = @(hypx) optimfunc(hypx, hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y);

%derivative check
fmincon(optfunc,...
           unwrap(hyp),[],[],[],[],[],[],@unitdisk,options);
end

function [fx, dx] = optimfunc(hypx, hyp0, inf, mean, cov, lik, x, y)
if nargout > 1
    [fx, dL] = gp(rewrap(hyp0, hypx), inf, mean, cov, lik, x, y);
    dx = unwrap(dL);
else
    fx = gp(rewrap(hyp0, hypx), inf, mean, cov, lik, x, y);
end
end