function [times, theta_over_time, mF, s2F, nlZ, mFT] = Multiscale(EXPERIMENT, trainX, trainY, testX)
    filename = sprintf('resultsSoD_%s_fold%d.mat', EXPERIMENT.DATASET, EXPERIMENT.DATASET_FOLD);
    resultsSoD = load([EXPERIMENT.RESULTS_DIR filename]);
    resultsSoD = resultsSoD.resultsSoD;
    D = size(trainX, 2);
    M = EXPERIMENT.M;
    sn = 0.25; hyp.lik = log(sn);
    disp('Executing FPC');
    %[~, U] = indPoints(trainX, M, 'c');
    sod = indPoints(trainX, M, 'c');
    U = trainX(sod, :);
    disp('Done.');
    size(U)
    U = reshape(U', [M*D, 1]);
    ellU = -ones([M*D, 1]);
    ell = zeros(D,1);
    sf2 = 0;
    
    SoDhyp = resultsSoD.hyp_over_time{1};
    ind = size(SoDhyp, 2);
    ell = SoDhyp(1:D, ind);
    sf2 = SoDhyp(D+1, ind);
    hyp.lik = SoDhyp(D+2, ind);

    hyp.cov = [ell; U; ellU; sf2];
    hyp = unwrap(hyp);    
    [times, theta_over_time, mF, s2F, nlZ, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, 'FIC', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), 'SparseMultiScaleGP');
end
