[~, ~, nlZ, mF, s2F, t] = infLibGPmex(trainX, trainY, trainX, gpname, 'CovSum (CovSEard, CovNoise)', resultOut.hyp_over_time{trial_id}(:, end), EXPERIMENT.M, bfname, EXPERIMENT.SEED{trial_id}, EXPERIMENT.EXTRA);
smse = mean((mF-trainY).^2)/var(trainY)
