function [EXPERIMENT, times, theta_over_time, mF, s2F, nlZ, gradientNorms, mFT] = FIC(EXPERIMENT, trainX, trainY, testX, trial_id)
    D = size(trainX, 2);
    M = EXPERIMENT.M;
    sn = 0.25; hyp.lik = log(sn);
    disp('Executing FPC');
    %[~, U] = indPoints(trainX, M, 'c');
    sod = indPoints(trainX, M, 'c');
    U = trainX(sod, :);
    disp('Done.');
    size(U)
    ell = zeros(D,1);
    sf2 = 0;
    seed = 0;
    hyp.cov = [ell; sf2];
    [alpha2, L2, nlZ2, mF2, s2F2] = infLibGPmex(trainX, trainY, testX, 'OptFIC', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), EXPERIMENT.M, 'FICfixed', seed, U);

    U = reshape(U, [M*D], 1);
    hyp.cov = [ell; sf2; U];
    [alpha, L, nlZ, mF, s2F] = infLibGPmex(trainX, trainY, testX, 'OptFIC', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), EXPERIMENT.M, 'FIC', seed, '');
    U = trainX(sod, :);
    U = reshape(U', [M*D, 1]);
    hyp.cov = [ell; sf2; U];
    [alpha3, L3, nlZ3, mF3, s2F3] = infLibGPmex(trainX, trainY, testX, 'OptFIC', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), EXPERIMENT.M, 'FIC', seed, '');
    max(max(abs(L2-L3)))
    checkError(L, L2, 'FIC', 'FICfixed', 'L');
    checkError(alpha, alpha2, 'FIC', 'FICfixed', 'alpha');
    checkError(nlZ, nlZ2, 'FIC', 'FICfixed', 'nlZ');
    checkError(mF, mF2, 'FIC', 'FICfixed', 'mF');
    checkError(s2F, s2F2, 'FIC', 'FICfixed', 's2F');
error('Test finished succesfully.');
end
