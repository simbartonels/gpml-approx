function [EXPERIMENT, times, theta_over_time, mF, s2F, nlZ, gradNorms, mFT] = FastFood(EXPERIMENT, trainX, trainY, testX, trial_id)
    D = size(trainX, 2);
    hyp.cov = [zeros(D,1);0];
    sn = 0.25; hyp.lik = log(sn);
    m = floor(EXPERIMENT.M/2/D);
    [s, g, ~, b] = initFastFood(m, D, []);
    Dintern = 2^nextpow2(D);
    shape = [Dintern, m];
    EXPERIMENT.EXTRA = [reshape(s, shape); reshape(g, shape); reshape(b, shape)];
    [times, theta_over_time, mF, s2F, nlZ, gradNorms, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, 'degenerate', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), 'FastFood');
end
