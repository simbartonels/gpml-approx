function [hyp, theta_over_time, retvals] = HSM_hyper_opt(EXPERIMENT, trainX, trainY, M, D)
    hyp.cov = [zeros(D,1);0];
    likfunc = @likGauss; 
    sn = 0.25; hyp.lik = log(sn);
    infMethod = @infExact;
    covfunc = {@covSEard};
    [nhyp, theta_over_time] = rpropmex(EXPERIMENT.SEED, abs(EXPERIMENT.NUM_HYPER_OPT_ITERATIONS), trainX, trainY, 'full', 'CovSum (CovSEard, CovNoise)', unwrap(hyp));
    hyp = rewrap(hyp, nhyp);
    
    retvals = {infMethod, covfunc, likfunc};
end
