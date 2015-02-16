function [hyp, theta_over_time, retvals] = HSM_hyper_opt(EXPERIMENT, trainX, trainY, m, D)
    %TODO: choose another L
    L = 1.2;
    infMethod = @infSolinfast;
    %covfunc = {@covDegFast, {@degHSM2, floor(m^(1/D)), L, J, lambda}, EXPERIMENT.SEED, m};
    covfunc = {@covDegFast, {'degHSM2', floor(D^(1/m))}, EXPERIMENT.SEED, m};
    hyp.cov = [zeros(D,1);0];
    likfunc = @likGauss; 
    sn = 0.25; hyp.lik = log(sn);
    %[hyp, ~, ~, theta_over_time] = minimize(hyp, @gp, EXPERIMENT.NUM_HYPER_OPT_ITERATIONS, {infMethod}, [], covfunc, likfunc, trainX, trainY);
    [nhyp, theta_over_time] = rpropmex(EXPERIMENT.SEED, abs(EXPERIMENT.NUM_HYPER_OPT_ITERATIONS), trainX, trainY, 'degenerate', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), m, 'Solin');
    hyp = rewrap(hyp, nhyp);
    retvals = {infMethod, covfunc, likfunc};
end
