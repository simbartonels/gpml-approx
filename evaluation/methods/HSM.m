function [EXPERIMENT, times, theta_over_time, mF, s2F, nlZ, mFT] = HSM(EXPERIMENT, trainX, trainY, testX, trial_id)
    D = size(trainX, 2);
    hyp.cov = [zeros(D,1);0];
    sn = 0.25; hyp.lik = log(sn);
    [times, theta_over_time, mF, s2F, nlZ, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, 'Solin', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), 'Solin');
end
