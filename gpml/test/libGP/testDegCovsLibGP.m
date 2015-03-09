function testDegCovsLibGP()
%TESTDEGCOVS This function compares what the matlab implementation
%covariance functions produce against what the libGP implementations do.
    testSolin();
    testSM();
    testFIC();
    disp('Test passed sucessfully.');
end

function testSolin()
    seed = 0;
    [z, ~, ~, hyp] = initEnv();
    [n, D] = size(z);
    L = 4 * ones([1, D])
    M = floor((n-1)^(1/D));
    [J, lambda] = initHSM(M, D, L);
    bf = {@degHSM2, M, L, J, lambda};
    Sigma_o = diag(feval(bf{:}, hyp.cov));
    Sigma = covDegFast(bf, seed, M^D, D, unwrap(hyp));
    phi_o = feval(bf{:}, hyp.cov, z);
    phi = covDegFast(bf, seed, M^D, D, unwrap(hyp), [], z);
    checkError(phi_o, phi, 'GPML', 'LibGP', 'basis function');
    checkError(Sigma_o, Sigma, 'GPML', 'LibGP', 'weight prior');
end

function testSM()
    seed = 0;
    M = 10;
    [z, ~, ~, hyp] = initEnvSM(M);
    snu2 = 1e-6 * exp(2 * hyp.lik);
    D = size(z, 2);
    bf = {'SparseMultiScaleGP'};
    phi_o = covSM(M, hyp.cov, z, z); %yields Uvz
    phi = covDegFast(bf, seed, M, D, unwrap(hyp), [], z);
    checkError(phi_o, phi, 'GPML', 'LibGP', 'basis function');
    [~, Sigma_o, ~] = covSM(M, hyp.cov, z);
    %bfmex will return the inverse with inducing noise added
    Sigma_o = chol(Sigma_o+snu2*eye(M));
    Sigma_o = solve_chol(Sigma_o, eye(M));
    Sigma = covDegFast(bf, seed, M, D, unwrap(hyp));
    checkError(Sigma_o, Sigma, 'GPML', 'LibGP', 'weight prior');
end

function testFIC()
    seed = 0;
    [z, ~, ~, hyp2] = initEnv();
    snu2 = 1e-6 * exp(2 * hyp2.lik);
    D = size(z, 2);
    M = 4;
    U = randn(M, D);
    logell = hyp2.cov(1:D);
    lsf2 = hyp2.cov(D+1);
    hyp = hyp2;
    hyp.cov = [logell; lsf2; reshape(U, [M*D, 1])];
    bf = {'FIC'};
    phi_o = covFITC({@covSEard}, U, hyp2.cov, z, z);
    phi = covDegFast(bf, seed, M, D, unwrap(hyp), [], z);
    checkError(phi_o, phi, 'GPML', 'LibGP', 'basis function');
    [~, Sigma_o, ~] = covFITC({@covSEard}, U, hyp2.cov, z);
    %bfmex will return the inverse with inducing noise added
    Sigma_o = chol(Sigma_o+snu2*eye(M));
    Sigma_o = solve_chol(Sigma_o, eye(M));
    Sigma = covDegFast(bf, seed, M, D, unwrap(hyp));
    checkError(Sigma_o, Sigma, 'GPML', 'LibGP', 'weight prior');
end
