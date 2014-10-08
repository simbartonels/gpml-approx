function testFastFood()
    testHadamardMultiplication();
    testBasisFunctionWorksProperly();
    testMatrixGradients();
    testGradients();
    testApproximationQuality();
    disp('All tests succesful.');
end

function testBasisFunctionWorksProperly()
% Asserts that degCov returns indeed what it's supposed to.
    [x, ~, ~, hyp] = initEnv();
    M = 2;
    [n, D] = size(x);
    % for fast food the 1st parameter is the length scale!
    ell = exp(hyp.cov(2:D+1));
    [s, gpi, b] = initFastFood(M, D, hyp.cov);
    d = D;
    D = 2^nextpow2(D);
    target = zeros(M*D, n);
    z = [x./repmat(ell', [n, 1]) zeros(n, D-d)];
    for j = 1:M
        idx = (1+(j-1)*D):(j*D);
        Z = diag(s(idx))*hadamard(D)*diag(gpi(idx))*hadamard(D)*diag(b(idx));
        target(idx, :) = Z*z';
    end
    target = [cos(target); sin(target)];
    actual = degFastFood(s, gpi, b, hyp.cov, x);
    if any(any(abs(target - actual) > 1e-14))
        %target - actual
        max_diff = max(max(abs(target-actual)))
        error('Fast Food implementation is wrong.');
    end
end

function testMatrixGradients()
    [x, ~, ~, hyp] = initEnv();
    [n, D] = size(x);
    n = 1;
    x = randn([n, D]);
    m = 1;
    [s, gpi, b] = initFastFood(m, D, hyp.cov);
    for di = 1:D+1
        fun = @(h) matrixGradientCheckFunction(h, x, s, gpi, b, hyp.cov, di);
        checkMatrixGradients(fun, hyp.cov(di), m, n);
    end
    disp('matrix gradients check successful');
end

function [K, dK] = matrixGradientCheckFunction(h, z, s, gpi, b, hyp, di)
    hyp(di) = h;
    K = degFastFood(s, gpi, b, hyp, z);
    dK = degFastFood(s, gpi, b, hyp, z, di);
end

function testGradients()
    [x, y, ~, hyp] = initEnv(17465);
    D = size(x, 2);
    m = 2;
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
    sd = floor(rand(1)*32000)
    %sd = 15909
    %sd = 24362
    %sd = 22965
    %sd = 13719
    rng(sd);
    D = 4;
    m = 256;
    n = 1; %m*D+1;
    z = 1;

    ls = exp(randn(1)*ones([D,1]));
    ls = exp(randn([D, 1]));
    noise = exp(randn(1));
    sf2 = 1; exp(2*randn(1));

    x = rand(n, D);
    xs = rand(z, D);
    ffhyp.lik = log(noise)/2;
    ffhyp.cov = [log(sf2)/2; log(ls)];
    hyp.lik = ffhyp.lik;
    hyp.cov = [log(ls); log(sf2)/2];

    [s, gpi, b] = initFastFood(m, D, ffhyp.cov);
    %s = ones([m*d, 1])/sqrt(D);
    cov_deg = {@covDegenerate, {@degFastFood, s, gpi, b}};
    weight_prior = feval(cov_deg{:}, ffhyp.cov);
    phi = feval(cov_deg{:}, ffhyp.cov, [], x);
    phiz = feval(cov_deg{:}, ffhyp.cov, [], xs);
    K = phi' * diag(weight_prior) * phiz;
    Korig = covSEard(hyp.cov, x, xs);
    diff = sum(sum(abs((covSEard(hyp.cov, x, xs) - K).^2))/(n*z))
    if abs(diff) > 1e-5, error('Approximation quality is too bad!'); end
end

function testHadamardMultiplication()
n = 1;
D = 2;
m = 1;

lls = zeros([D, 1]); %exp(randn(1));
lnoise = 1; %exp(randn(1));
lsf = 0; %exp(randn(1));

sf2 = exp(2*lsf);
ls = exp(2*lls);

x = rand(n, D) / 2;
hyp.lik = lnoise;
hyp.cov = [lsf; lls];
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
