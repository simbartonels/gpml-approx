function [times, theta_over_time, mF, s2F, nlZ, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, approximation, covName, initialhypers, bfname)
	[times, theta_over_time, mF, s2F, nlZ, mFT] = rpropmex(EXPERIMENT.SEED, abs(EXPERIMENT.NUM_HYPER_OPT_ITERATIONS), trainX, trainY, testX, approximation, covName, initialhypers, EXPERIMENT.M, bfname);
end