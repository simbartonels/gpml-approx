function [times, theta_over_time, mF, s2F, nlZ, mFT] = HSM(EXPERIMENT, trainX, trainY, testX)
    D = size(trainX, 2);
    M = EXPERIMENT.M;
    sn = 0.25; hyp.lik = log(sn);
    disp('Executing FPC');
    [~, U] = indPoints(trainX, M, 'c');
    hyp.cov = [zeros(D,1); reshape(U, [M*D, 1]); zeros(M*D, 1); 0];
    
    [times, theta_over_time, mF, s2F, nlZ, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, 'FIC', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), 'SparseMultiScaleGP');
end
