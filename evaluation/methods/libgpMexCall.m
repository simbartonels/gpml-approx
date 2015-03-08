function [times, theta_over_time, mF, s2F, nlZ, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, approximation, covName, initialhypers, bfname)
	iters = abs(EXPERIMENT.NUM_HYPER_OPT_ITERATIONS);
	[times, theta_over_time, mF, s2F, nlZ] = rpropmex(EXPERIMENT.SEED, iters, trainX, trainY, testX, approximation, covName, initialhypers, EXPERIMENT.M, bfname);
	%workaround
	%mFT = zeros(size(trainX, 1), iters);    
    %[~, ~, ~, mFT, ~] = infLibGPmex(trainX, trainY, trainX, approximation, covName, theta_over_time(:, size(times)), EXPERIMENT.M, bfname);
    [~, ~, ~, mFT, ~] = infLibGPmex(trainX, trainY, testX, approximation, covName, theta_over_time(:, size(times)), EXPERIMENT.M, bfname);
end
