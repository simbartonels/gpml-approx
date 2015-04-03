function [times, theta_over_time, mF, s2F, nlZ, gradNorm, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, approximation, covName, initialhypers, bfname)
	current_trial = EXPERIMENT.LAST_TRIAL;
	if isfield(EXPERIMENT, 'LAST_HYPERS')
		if size(EXPERIMENT.LAST_HYPERS, 2) >= current_trial
			initialhypers = EXPERIMENT.LAST_HYPERS{current_trial};
			disp('Using hyper-parameters from previous run.');
		end
	end
	iters = abs(EXPERIMENT.NUM_HYPER_OPT_ITERATIONS);
	[times, theta_over_time, mF, s2F, nlZ, gradNorm] = rpropmex(EXPERIMENT.SEED{current_trial}, iters, trainX, trainY, testX, approximation, covName, initialhypers, EXPERIMENT.M, bfname, EXPERIMENT.CAP_TIME, EXPERIMENT.EXTRA);
	%workaround
	mFT = zeros(size(trainX, 1), iters);    
    %[~, ~, ~, mFT, ~] = infLibGPmex(trainX, trainY, trainX, approximation, covName, theta_over_time(:, size(times)), EXPERIMENT.M, bfname);
    %[~, ~, ~, mFT, ~] = infLibGPmex(trainX, trainY, testX, approximation, covName, theta_over_time(:, size(times)), EXPERIMENT.M, bfname);
end
