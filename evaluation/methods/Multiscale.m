function [EXPERIMENT, times, theta_over_time, mF, s2F, nlZ, gradNorms, mFT] = Multiscale(EXPERIMENT, trainX, trainY, testX, trial_id)
    D = size(trainX, 2);
    M = EXPERIMENT.M;
    sn = 0.25; hyp.lik = log(sn);
%    disp('Executing FPC');
%    %[~, U] = indPoints(trainX, M, 'c');
%    %U = U';
%    sod = indPoints(trainX, M, 'c');
%    U = trainX(sod, :);
%    disp('Done.');
%    size(U)
%    U = reshape(U, [M*D, 1]);
%    ell = zeros(D,1);
%    ellU = log(ones([M*D, 1])/2);
%    sf2 = 0;
%    hyp = [ell; ellU; U; sf2; log(sn)];
%    [times, theta_over_time, mF, s2F, nlZ, gradNorms, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, 'OptMultiscale', 'CovSum (CovSEard, CovNoise)', hyp, 'SparseMultiScaleGP');
%return;

    filename = sprintf('%s%s%sresultsFIC_fold%d_M%d.mat', ...
	EXPERIMENT.RESULTS_DIR, EXPERIMENT.DATASET, filesep, EXPERIMENT.DATASET_FOLD, EXPERIMENT.M)
    resultsFIC = load(filename);
    resultsFIC = resultsFIC.resultsFIC;
    
    FIChyp = resultsFIC.hyp_over_time{trial_id};
    ind = floor(size(FIChyp, 2)/2);
%    ind = 25 %TODO: remove
    FIChyp = FIChyp(:, ind);
    time_offset = resultsFIC.hyp_time{trial_id}(ind);
%    time_offset = time_offset(ind);
    disp('Setting noise to low value for FIC and Multiscale compliance.');
    FIChyp(size(FIChyp, 1)) = 1e-150;
    hyp = FIC_params_to_Multiscale(FIChyp, D, M);
    cap = EXPERIMENT.CAP_TIME;
	%are we continuing an experiment? ...
    	if isfield(EXPERIMENT, 'LAST_HYPERS')
		if size(EXPERIMENT.LAST_HYPERS, 2) >= trial_id
			% in that case the captime was taken care of in EVAL_METHOD
			time_offset = 0;
		end
	end
    EXPERIMENT.CAP_TIME = cap - time_offset;
    [times, theta_over_time, mF, s2F, nlZ, gradNorms, mFT] = libgpMexCall(EXPERIMENT, trainX, trainY, testX, 'OptMultiscale', 'CovSum (CovSEard, CovNoise)', hyp, 'SparseMultiScaleGP');
    times(times>0) = times(times>0) + time_offset;
    EXPERIMENT.CAP_TIME = cap;
end
