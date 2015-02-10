function testDegCovsLibGP()
%TESTDEGCOVS This function compares what the matlab implementation
%covariance functions produce against what the libGP implementations do.
    testSolin();
end

function testSolin()
    [z, ~, ~, hyp] = initEnv();
    [J, lambda] = initHSM(m, D, L);
    bf = {@degHSM2, M, L, J, lambda};
    phi_o = feval(bf, hyp.cov, z)
    phi = covDegFast(bf, M, seed, hyp.cov, [], z)
end