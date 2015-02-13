function [mF, s2F] = SoD_mine_predict(EXPERIMENT, hyp, trainX, trainY, testX, retvals)
[mF, s2F] = gp_sod_mine(hyp, EXPERIMENT.NUM_HYPER_OPT_ITERATIONS, {@covSEard}, retvals{2}, trainX, trainY, retvals{1}, 'g', testX);
end