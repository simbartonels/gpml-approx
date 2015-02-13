function [hyp, theta_over_time, retvals] = SoD_mine_hyper_opt(EXPERIMENT, trainX, trainY, m, D)
hyp.cov = [zeros(D,1);0];
likfunc = @likGauss; sn = 0.25; hyp.lik = log(sn);
if strcmp(EXPERIMENT.EXTRA, 'clustering')
	[hyp, sod, theta_over_time] = gp_sod_mine(hyp, EXPERIMENT.NUM_HYPER_OPT_ITERATIONS, {@covSEard}, likfunc, trainX, trainY, m, 'c', D, 'split');
else
	[hyp, sod, theta_over_time] = gp_sod_mine(hyp, EXPERIMENT.NUM_HYPER_OPT_ITERATIONS, {@covSEard}, likfunc, trainX, trainY, m, 'r');
end
retvals = {sod, likfunc};
end