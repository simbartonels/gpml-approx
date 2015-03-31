function [EXPERIMENT, times, theta_over_time, mF, s2F, nlZ, mFT] = FIC(EXPERIMENT, trainX, trainY, testX, trial_id)
    D = size(trainX, 2);
    M = EXPERIMENT.M;
    sn = 0.25; hyp.lik = log(sn);
    ell = zeros(D,1);
    sf2 = 0;
    
    hyp.cov = [ell; sf2];
    hyp = unwrap(hyp); 
    disp('Executing FPC');
    %[~, U] = indPoints(trainX, M, 'c');
    sod = indPoints(trainX, M, 'c');
	current_trial = trial_id;
	if isfield(EXPERIMENT, 'LAST_HYPERS')
		if size(EXPERIMENT.LAST_HYPERS, 2) >= current_trial
			hyp = EXPERIMENT.LAST_HYPERS{current_trial};
			sod = EXPERIMENT.SOD{trial_id};
			disp('Using hyper-parameters from previous run.');
		end
	end
    EXPERIMENT.SOD{current_trial} = sod;
    U = trainX(sod, :);
    disp('Done.');
   
    [times, theta_over_time, mF, s2F, nlZ, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, 'FIC', 'CovSum (CovSEard, CovNoise)', unwrap(hyp), 'FICfixed', U);
end
