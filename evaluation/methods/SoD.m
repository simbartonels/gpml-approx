function [times, theta_over_time, mF, s2F, nlZ, mFT] = SoD(EXPERIMENT, trainX, trainY, testX, trial_id)
D = size(trainX, 2);
M = EXPERIMENT.M;
hyp.cov = [zeros(D,1);0];
likfunc = @likGauss; sn = 0.25; hyp.lik = log(sn);
disp('Executing FPC');
sod = indPoints(trainX, M, 'c');
disp('Done');
[times, theta_over_time, mF, s2F, nlZ, ~] = libgpMexCall(EXPERIMENT, trainX(sod, :), trainY(sod, :), testX, 'full', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), '');
[~, ~, ~, mFT, ~] = infLibGPmex(trainX(sod, :), trainY(sod, :), trainX, 'full', 'CovSum (CovSEard, CovNoise)', theta_over_time(:, size(times)));
end
