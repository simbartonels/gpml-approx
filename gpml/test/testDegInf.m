D = 1; %MUST be 1
M = 32;
n = 2*M.^D;
z = 5;
b = ones(1, D);
a = -b;

x = rand(n, D) / 2;
y = randn(n, 1);
xs = rand(z, D) / 2;
hyp.lik = randn(1);
hyp1.lik = hyp.lik;
hyp2.lik = hyp.lik;
logls = log(0.1);
logsf2 = 0;
hyp.cov = [logls; logsf2];
hyp1.cov = [];

j = 1:M;
j = j';

%see Rasmussen p.154 above (7.11)
S = @(r) exp(logsf2)*sqrt(2*pi)*(exp(logls).^D)*exp(-(exp(logls)*r).^2/2);

sqrtlambda = pi*j/(b-a);
s = S(sqrtlambda);
cov = {@covHSMnaive, s, a, b, 1};
[ymu ys2 fmu fs2] = gp(hyp1, @infExact, [], cov, @likGauss, x, y, xs);

%create index matrix
m = M;
J = zeros(D, M.^D);
for d = 1:(D-1)
    J(d, :) = repmat(reshape(repmat((1:m)', 1, m.^(D-d))', [m.^(D-d+1), 1]), m.^(d-1), 1);
end
J(D, :) = repmat((1:m)', m.^(D-1), 1);
s = zeros(m.^D, 1);
for k = 1:m.^D
    sqrtlambda = pi*sqrt(sum((J(:, k)'./(b-a)).^2));
    s(k) = S(sqrtlambda);
end
hyp2.cov = [m];
hyp2.weight_prior = s;
cov2 = {@covDegenerate, {@degHSM, a, b}};
[ymuE ys2E fmuE fs2E] = gp(hyp2, @infExactDegKernel, [], cov2, @likGauss, x, y, xs);
ymu - ymuE
ys2 - ys2E
fmu - fmuE
fs2 - fs2E

[nlZ] = gp(hyp1, @infExact, [], cov, @likGauss, x, y);
[nlZE] = gp(hyp2, @infExactDegKernel, [], cov2, @likGauss, x, y);
nlZ - nlZE
%dnlZ - dnlZE