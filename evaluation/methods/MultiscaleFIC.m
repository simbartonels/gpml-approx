function [EXPERIMENT, times, theta_over_time, mF, s2F, nlZ, gradientNorms, mFT] = FIC(EXPERIMENT, trainX, trainY, testX, trial_id)
    %Uses Multiscale but fixes the length scales.
    D = size(trainX, 2);
    M = EXPERIMENT.M;
    sn = 0.25; hyp.lik = log(sn);
    disp('Executing FPC');
    %[~, U] = indPoints(trainX, M, 'c');
    sod = indPoints(trainX, M, 'c');
    U = trainX(sod, :);
    disp('Done.');
    size(U)
    U = reshape(U, [M*D, 1]);
    ell = zeros(D,1);
    sf2 = 0;
    
    hyp.cov = [ell; sf2; U];
    hyp = unwrap(hyp);    
    hyp = FIC_params_to_Multiscale(hyp, D, M);
    EXPERIMENT.EXTRA = [1e-6, 1.0; 1.0, 1.0];
    [times, theta_over_time, mF, s2F, nlZ, gradientNorms, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, 'OptMultiscale', 'CovSum (CovSEard, CovNoise)', hyp, 'SparseMultiScaleGP');
    EXPERIMENT.EXTRA = 1e-6;
end
