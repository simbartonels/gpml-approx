function testFastFood()
    testGradients();
    testHadamardMultiplication();
    testApproximationQuality();
    disp('All tests succesful.');
end

function testGradients()
    [sd, n, D, x, y, xs, logell, lsf2, lsn2] = initEnv();
    sd
    rng(sd);
    hyp.lik = lsn2;
    logell = logell(1);
    hyp.cov = [logell; lsf2];
    m = 1;
    [s, gpi, b] = initFastFood(m, D, hyp.cov);
    %s = ones([m*d, 1])/sqrt(D);
    cov_deg = {@covDegenerate, {@degFastFood, s, gpi, b}};

    options = optimoptions(@fmincon,'Algorithm','interior-point',...
        'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);
    optfunc = @(hypx) optimfunc(hypx, hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y);

    %derivative check
    fmincon(optfunc,...
               unwrap(hyp),[],[],[],[],[],[],@unitdisk,options);
end

function testApproximationQuality()
D = 4;
m = 256;
n = 1; %m*D+1;
z = 1;

ls = exp(randn(1));
noise = exp(randn(1));
sf2 = exp(randn(1));

x = rand(n, D);
xs = rand(z, D);
hyp.lik = log(noise)/2;
hyp.cov = [log(ls); log(sf2)/2];

[s, gpi, b] = initFastFood(m, D, hyp.cov);
%s = ones([m*d, 1])/sqrt(D);
cov_deg = {@covDegenerate, {@degFastFood, s, gpi, b}};
weight_prior = feval(cov_deg{:}, hyp.cov);
phi = feval(cov_deg{:}, hyp.cov, [], x);
phiz = feval(cov_deg{:}, hyp.cov, [], xs);
K = phi' * diag(weight_prior) * phiz
Korig = covSEiso(hyp.cov, x, xs)
diff = sum(sum(abs((covSEiso(hyp.cov, x, xs) - K).^2))/(n*z))
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
