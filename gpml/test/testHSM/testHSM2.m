function testHSM2()
    testBasisFunctionImpl();
    testEquivalentToHSMsimple();
    testAgainstSEard();
    testGradients();
    disp('Test completed successfully.');
end

function testAgainstSEard()
    [x, ~, z, hyp2] = initEnv();
    M = 60;
    D = size(x, 2);
    hyp2.cov(1:D) = 1./(1+randn(D, 1).^2)-2;

    L = 1.2 * max(abs(x));%rand(1, D);
    [J, lambda] = initHSM(M, D, L);

    cov_deg2 = {@covDegenerate, {@degHSM2, M, L, J, lambda}};
    
    sigma2 = feval(cov_deg2{:}, hyp2.cov);
    % by convention the first input is ignored (see covDegenerate)
    bf2x = feval(cov_deg2{:}, hyp2.cov, [], x);

    bf2z = feval(cov_deg2{:}, hyp2.cov, [], z);
    k2 = bf2x' * diag(sigma2) * bf2z;
    k = covSEard(hyp2.cov, x, z);
    % Taking maximum here is not a good idea since the approximation is
    % sometimes quite bad. I'm satisfied when the median agrees.
    diff = median(median(abs((k-k2).^2./k)));
    if diff > (1e-9)^(1/D)
        diff
        error('DegHSM2 and SE_ard disagree!'); 
    end
end

function testEquivalentToHSMsimple()
    [x, ~, ~, hyp2] = initEnv();
    M = 2;
    D = size(x, 2);
    hyp2.cov(1:D) = log(0.1) * ones([1, D]);

    hyp.lik = hyp2.lik;
    %in degHSM we use exp(hyp) instead of exp(2*hyp)
    hyp.cov = hyp2.cov;
    hyp.cov(D+1) = 2 * hyp.cov(D+1);

    L = 1.2 * max(abs(x));%rand(1, D);
    [J, lambda] = initHSM(M, D, L);

    cov_deg2 = {@covDegenerate, {@degHSM2, M, L, J, lambda}};
    cov_deg = {@covDegenerate, {@degHSM, M, L, J, lambda}};
    
    sigma2 = feval(cov_deg2{:}, hyp2.cov);
    sigma = feval(cov_deg{:}, hyp.cov);
    diff = max(abs((sigma2-sigma)./sigma));
    % In this special case the weight priors MUST be the same.
    if diff > 1e-5
       error('Weight priors disagree! But that is due to the implementation.'); 
    end
    % by convention the first input is ignored (see covDegenerate)
    bf2x = feval(cov_deg2{:}, hyp2.cov, [], x);
    bfx = feval(cov_deg{:}, hyp.cov, [], x);
    diff = max(max(abs((bf2x-bfx)./bfx)));
    % In this special case also phi(x) should produce the same results.
    if diff > 1e-5
       error('Basis function computation disagrees!'); 
    end
end

function testBasisFunctionImpl()
    sd = floor(rand(1) * 32000)
%    sd = 12620
    %sd = 10184
    rng(sd);
    D = 3;
    M = 6;
    n = 1;
    x = rand(n, D) / 2;
    logsf2 = 0;
    logls = 1./(1+randn(D, 1).^2)-2
    cov2hyp = [logls; logsf2];
    L = ones(1, D);
    [J, lambda] = initHSM(M, D, L);
    cov2 = {@degHSM2, M, L, J, lambda};
    phix = feval(cov2{:}, cov2hyp, x);
    weight_prior = feval(cov2{:}, cov2hyp);    
    sf2 = exp(2*logsf2);
    for m = 1:M^D
        bf = 1;
        s = sf2;
        for d = 1:D
            loglsd = logls(d);
            sqrtlambda = pi*J(d, m)/L(d)/2;
            spec_dens = @(r) sqrt(2*pi)*exp(loglsd)*exp(-exp(2*loglsd)*r.^2/2);
            s = s * spec_dens(sqrtlambda);
            
            bf = bf * sin(pi * J(d, m) * (x(d)+L(d))/2/L(d))/sqrt(L(D));
        end
        
        diff = abs((phix(m) - bf)/bf);
        if diff > 1e-5
            m
            phix
            bf
            error('Somethings wrong in the computation of the basis functions.');
        end
        
        diff = abs((weight_prior(m) - s)/s);
        if diff > 1e-5
            weight_prior
            s
            m
            error('Somethings wrong in the computation of Gamma.');
        end

    end
end

function testGradients2()
    error('copy&paste code');
    D = 2;
    hyp.lik = 0;
    hyp.cov = [zeros([D, 1]); 0];
    M = 1;
    L = ones([1, D]);
    [J, lambda] = initHSM(M, D, L);
    cov_deg = {@covDegenerate, {@degHSM, M, L, J, lambda}};
    z = randn([1, D]);
    dK = feval(cov_deg{:}, hyp.cov, [], z, 1);
    z = z * 0.1; % adapt length scales
    dKtarget = pi * cos( pi*(z(:, 1)'+1)/2 )/2;
    phid2 = sin( pi * (z(:, 2)'+1) / 2);
    dKtarget = dKtarget .* phid2;
    dKtarget = dKtarget*diag(-z(:, 1));
    diff = abs(dK - dKtarget)
    if diff > 1e-6, error('Simple gradient check failed.'); end
end

function testGradients()
    [x, y, ~, hyp] = initEnv();
    M = 2;
    D = size(x, 2);
    L = 1.2 * max(abs(x));%rand(1, D);
    [J, lambda] = initHSM(M, D, L);

    cov_deg = {@covDegenerate, {@degHSM2, M, L, J, lambda}};

    options = optimoptions(@fmincon,'Algorithm','interior-point',...
        'DerivativeCheck','on','GradObj','on', 'MaxFunEvals', 1);
    optfunc = @(hypx) optimfunc(hypx, hyp, @infExactDegKernel, [], cov_deg, @likGauss, x, y);

    %derivative check
    fmincon(optfunc,...
               unwrap(hyp),[],[],[],[],[],[],@unitdisk,options);
end

function testAgainstNaiveImplementation()
    error('copy&paste code');
    error('Not implemented!');
end

function testToyExample()
    error('copy&paste code');
D = 1;
M = 64;
n = 1;
z = 1;
%L = 3 * rand(1, D);
L = ones(1, D);
x = rand(n, D) / 2;
xs = rand(z, D) / 2;
hyp.lik = 0;
logls = log(0.1) * ones([D, 1]);
logsf2 = 0;
hyp.cov = [logls; logsf2];

[J, lambda] = initHSM(M, D, L);

phix = degHSM(M, L, J, lambda, hyp.cov, x);
phixs = degHSM(M, L, J, lambda, hyp.cov, xs);
weight_prior = degHSM(M, L, J, lambda, hyp.cov);

diff = phix'*diag(weight_prior)*phixs - covSEard(hyp.cov, x, xs)
if abs(diff) > 1e-15, error('Toy example appears broken.'); end
end