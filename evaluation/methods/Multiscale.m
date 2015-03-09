function [times, theta_over_time, mF, s2F, nlZ, mFT] = Multiscale(EXPERIMENT, trainX, trainY, testX)
    filename = sprintf('resultsFIC_%s_fold%d.mat', EXPERIMENT.DATASET, EXPERIMENT.DATASET_FOLD);
    resultsFIC = load([EXPERIMENT.RESULTS_DIR filename]);
    resultsFIC = resultsFIC.resultsFIC;
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
    
    FIChyp = resultsFIC.hyp_over_time{1};
    ind = size(FIChyp, 2);
    ind = 50 %TODO: remove
    sf2 = FIChyp(D+1, ind);
    ell = FIChyp(1:D, ind);
    ellU = repmat(log(exp(2*ell)/2), [M, 1]);
    U = FIChyp((D+2):(D+1+D*M), ind);
    U = reshape(U, [M, D]); %little sanity check %TODO: Remove
    U = reshape(U, [M*D, 1]);
    hyp.lik = FIChyp(size(FIChyp, 1), ind);

    hyp.cov = [ell; U; ellU; sf2];
    hyp = unwrap(hyp);
    [times, theta_over_time, mF, s2F, nlZ, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, 'FIC', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), 'SparseMultiScaleGP');
end
