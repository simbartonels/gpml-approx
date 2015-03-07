function [times, theta_over_time, mF, s2F, nlZ, mFT] = SoD(EXPERIMENT, trainX, trainY, testX)
D = size(trainX, 2);
M = EXPERIMENT.M;
hyp.cov = [zeros(D,1);0];
likfunc = @likGauss; sn = 0.25; hyp.lik = log(sn);
disp('Executing FPC');
sod = indPoints(trainX, M, 'c');
disp('Done');
%if strcmp(EXPERIMENT.EXTRA, 'clustering')
%	[hyp, sod, theta_over_time] = gp_sod_mine(hyp, EXPERIMENT.NUM_HYPER_OPT_ITERATIONS, {@covSEard}, likfunc, trainX, trainY, m, 'c', D, 'split');
%else
%	[hyp, sod, theta_over_time] = gp_sod_mine(hyp, EXPERIMENT.NUM_HYPER_OPT_ITERATIONS, {@covSEard}, likfunc, trainX, trainY, m, 'r');
%end
[times, theta_over_time, mF, s2F, nlZ, ~] = libgpMexCall(EXPERIMENT, trainX(sod, :), trainY(sod, :), testX, 'full', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), '');
[~, ~, ~, mFT, ~] = infLibGPmex(trainX(sod, :), trainY(sod, :), trainX, 'full', 'CovSum (CovSEard, CovNoise)', theta_over_time(:, size(times)));
return;
retvals = {sod, likfunc};
times = theta_over_time(:, 1);
actual_iters = size(theta_over_time, 2);
theta_over_time = theta_over_time(:, 2:actual_iters);
test_n = size(testX, 1);
mF = -ones([test_n, actual_iters]);
s2F = -ones([test_n, actual_iters]);
nlZ = -ones([1, actual_iters]);
mFT = -ones([size(trainX, 1), actual_iters]);
for i=1:size(theta_over_time, 2)
    if theta_over_time(1, i) < 0, break, end
    [mFi, s2Fi, nlZi] = gp_sod_mine(hyp, [], {@covSEard}, retvals{2}, trainX, trainY, retvals{1}, 'g', testX);
    mF(:, i) = mFi;
    s2F(:, i) = s2Fi;
    nlZ(i) = nlZi;
    mFTi = gp_sod_mine(hyp, [], {@covSEard}, retvals{2}, trainX, trainY, retvals{1}, 'g', trainX);
    mFT(:, i) = mFTi;
end
end