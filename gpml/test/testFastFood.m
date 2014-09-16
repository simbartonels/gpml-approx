function testFastFood()
    testHadamardMultiplication();
    testApproximationQuality();
    testGradients();
    disp('All tests succesful.');
end

function testGradients()

end

function testApproximationQuality()
D = 4;
m = 256;
n = 1; %m*D+1;
z = 1;

ls = 1; %exp(randn(1));
noise = 1; %exp(randn(1));
sf2 = 1; %exp(randn(1));

x = rand(n, D)
xs = rand(z, D)
hyp.lik = log(noise)/2;
hyp.cov = [log(ls); log(sf2)/2];

d = 2^nextpow2(D);
[s, gpi, b] = initFastFood(m, D, hyp.cov);
%s = ones([m*d, 1])/sqrt(D);
cov_deg = {@covDegenerate, {@degFastFood, s, gpi, b}};
phi = feval(cov_deg{:}, hyp.cov, NaN, x);
phiz = feval(cov_deg{:}, hyp.cov, NaN, xs);
K = phi' * diag(hyp.weight_prior) * phiz
Korig = covSEiso(hyp.cov, x, xs)
diff = sum(sum(abs((covSEiso(hyp.cov, x, xs) - K).^2))/(n*z))

% s = ones([m*d, 1]) / sqrt(d*ls);
% cov_deg = {@covDegenerate, {@degFastFood, s, gpi, b}};
% phi = feval(cov_deg{:}, hyp.cov, NaN, x);
% phiz = feval(cov_deg{:}, hyp.cov, NaN, xs);
% K = phi' * diag(hyp.weight_prior) * phiz
% sum(sum(abs((covSEiso(hyp.cov, x, xs) - K).^2))/(n*z))

%covSEiso(hyp.cov, x, xs) - 
%feval(cov_deg{:}, hyp.cov, x, xs)
%[ymuE ys2E fmuE fs2E] = gp(hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y, xs)
end

function testHadamardMultiplication()
n = 1;
D = 2;
m = 1;

lls = 0; %exp(randn(1));
lnoise = 1; %exp(randn(1));
lsf = 0; %exp(randn(1));

sf2 = exp(2*lsf);
ls = exp(2*lls);

x = rand(n, D) / 2;
hyp.lik = lnoise;
hyp.cov = [lls; lsf];
[s, gpi, b] = initFastFood(m, D, hyp.cov);
cov_deg = {@covDegenerate, {@degFastFood, s, gpi, b}};
Kimpl = feval(cov_deg{:}, hyp.cov, [], x);

H = ones(D);
H(4) = -1;
W = diag(s)*H*diag(gpi)*H*diag(b)*x';
K = [cos(W); sin(W)];
if sum((K - Kimpl).^2) > 1e-30, ...
    error('Something is wrong with Hadamard multiplication'); end
end
