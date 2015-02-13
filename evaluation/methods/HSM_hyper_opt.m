function [hyp, theta_over_time, retvals] = HSM_hyper_opt(EXPERIMENT, trainX, trainY, m, D)
    %TODO: choose another L
    %L = 1.2;
    infMethod = @infSolinfast;
    %covfunc = {@covDegFast, {@degHSM2, m, L, J, lambda}, 0, m};
    covfunc = {@covDegFast, {'degHSM2', m}, 0, D^m};
    hyp.cov = [zeros(D,1);0];
    likfunc = @likGauss; sn = 0.25; hyp.lik = log(sn);
    [hyp, ~, ~, theta_over_time] = minimize(hyp, @gp, EXPERIMENT.NUM_HYPER_OPT_ITERATIONS, {infMethod}, [], covfunc, likfunc, trainX, trainY);
    retvals = {infMethod, covfunc, likfunc};
end