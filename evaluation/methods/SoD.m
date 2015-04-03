function [EXPERIMENT, times, theta_over_time, mF, s2F, nlZ, gradNorms, mFT] = SoD(EXPERIMENT, trainX, trainY, testX, trial_id)
error('Make sure to test this method first!');
D = size(trainX, 2);
M = EXPERIMENT.M;
hyp.cov = [zeros(D,1);0];
likfunc = @likGauss; sn = 0.25; hyp.lik = log(sn);
disp('Executing FPC');
sod = indPoints(trainX, M, 'c');
if ~exist('EXPERIMENT.SOD') 
	EXPERIMENT.SOD{trial_id} = sod;
elseif size(EXPERIMENT.SOD, 2) < trial_id
	EXPERIMENT.SOD{trial_id} = sod;
else
	sod = EXPERIMENT.SOD{trial_id};
end
disp('Done');
[times, theta_over_time, mF, s2F, nlZ, gradNorms, ~] = libgpMexCall(EXPERIMENT, trainX(sod, :), trainY(sod, :), testX, 'full', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), '');
%[~, ~, ~, mFT, ~] = infLibGPmex(trainX(sod, :), trainY(sod, :), trainX, 'full', 'CovSum (CovSEard, CovNoise)', theta_over_time(:, size(times)));
mFT = [];
end
