function testDegCovsLibGP()
%TESTDEGCOVS This function compares what the matlab implementation
%covariance functions produce against what the libGP implementations do.
    testSolin();
    testSM();
end

function testSolin()
    seed = 0;
    [z, ~, ~, hyp] = initEnv();
    [n, D] = size(z);
    L = 1.2 * ones([1, D]);
    M = floor((n-1)^(1/D));
    [J, lambda] = initHSM(M, D, L);
    bf = {@degHSM2, M, L, J, lambda};
    phi_o = feval(bf{:}, hyp.cov, z);
    phi = covDegFast(bf, seed, D^M, hyp.cov, [], z);
    diff = max(max(abs(phi-phi_o)./abs(phi_o)));
    if(diff > 1e-5)
	error('Basis Function computations disagree!');
    end
end

function testSM()
    seed = 0;
    [z, ~, ~, hyp] = initEnvSM();
    D = size(z, 2);
    M = (numel(smhyp.cov)-1-D)/D/2;
    bf = {'SparseMultiScaleGP'};
    phi_o = covSM(M, hyp.cov, [], z);
    phi = covDegFast(bf, seed, M, hyp.cov, [], z);
    diff = max(max(abs(phi-phi_o)./abs(phi_o)));
    if(diff > 1e-5)
	error('Basis Function computations disagree!');
    end
end
