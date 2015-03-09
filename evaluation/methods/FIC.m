function [times, theta_over_time, mF, s2F, nlZ, mFT] = FIC(EXPERIMENT, trainX, trainY, testX)
    D = size(trainX, 2);
    M = EXPERIMENT.M;
    sn = 0.25; hyp.lik = log(sn);
    disp('Executing FPC');
    %[~, U] = indPoints(trainX, M, 'c');
    sod = indPoints(trainX, M, 'c');
    U = trainX(sod, :);
    disp('Done.');
    size(U)
    U = reshape(U', [M*D, 1]);
    ell = zeros(D,1);
    sf2 = 0;
    
    hyp.cov = [ell; sf2; U];
    hyp = unwrap(hyp);    
    [times, theta_over_time, mF, s2F, nlZ, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, 'FIC', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), 'FIC');
end