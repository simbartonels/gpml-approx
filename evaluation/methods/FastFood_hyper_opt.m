function [hyp, theta_over_time, retvals] = HSM_hyper_opt(EXPERIMENT, trainX, trainY, M, D)
    hyp.cov = [zeros(D,1);0];
    likfunc = @likGauss; 
    sn = 0.25; hyp.lik = log(sn);
    %TODO: choose another L
    infMethod = @infFastFoodfast;
    covfunc = {@covDegFast, {'degFastFood'}, EXPERIMENT.SEED, M};
    [nhyp, theta_over_time] = rpropmex(EXPERIMENT.SEED, abs(EXPERIMENT.NUM_HYPER_OPT_ITERATIONS), trainX, trainY, 'degenerate', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), M, 'FastFood');
    hyp = rewrap(hyp, nhyp);
    
    retvals = {infMethod, covfunc, likfunc};
end
