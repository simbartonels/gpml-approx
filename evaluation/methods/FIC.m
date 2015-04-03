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
    U = reshape(U, [M*D, 1]);
    ell = zeros(D,1);
    sf2 = 0;
    
    hyp.cov = [ell; sf2; U];
    hyp = unwrap(hyp);    
    [times, theta_over_time, mF, s2F, nlZ, gradientNorms, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, 'OptFIC', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), 'FIC');
end
