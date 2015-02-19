function [hyp, theta_over_time, retvals] = HSM_hyper_opt(EXPERIMENT, trainX, trainY, M, D)
    m = floor(M^(1/D));    
    hyp.cov = [zeros(D,1);0];
    likfunc = @likGauss; 
    sn = 0.25; hyp.lik = log(sn);
    %TODO: choose another L
    L = 1.2 * ones([1, D]);
    [J, lambda] = initHSM(m, D, L);
    M
    size(lambda)
    infMethod = @infExactDegKernel;
    covfunc = {@covDegenerate, {@degHSM2, m, L, J, lambda}};
    %[hyp, ~, ~, theta_over_time] = minimize(hyp, @gp, EXPERIMENT.NUM_HYPER_OPT_ITERATIONS, {infMethod}, [], covfunc, likfunc, trainX, trainY);
    infMethod = @infSolinfast;
    covfunc = {@covDegFast, {'degHSM2', m}, EXPERIMENT.SEED, M};
    [nhyp, theta_over_time] = rpropmex(EXPERIMENT.SEED, abs(EXPERIMENT.NUM_HYPER_OPT_ITERATIONS), trainX, trainY, 'degenerate', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), m, 'Solin');
    hyp = rewrap(hyp, nhyp);
    
    retvals = {infMethod, covfunc, likfunc};
end
