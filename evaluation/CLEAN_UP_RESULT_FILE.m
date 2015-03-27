error('Make sure to test this method first.');
resultVarName = sprintf('results%s', EXPERIMENT.METHOD);
results_file = sprintf('%s%s%s%s%s_fold%d_M%d.mat', EXPERIMENT.RESULTS_DIR, '', EXPERIMENT.DATASET, filesep, resultVarName, EXPERIMENT.DATASET_FOLD, EXPERIMENT.M)
if exist(results_file, 'file') == 2
	disp('WARNING: An experiment results file already exists for this configuration. Attempting continuation of the experiment.');
	load(results_file);
	resultsOut = eval(resultVarName);
        [trials_old, len_old] = size(resultOut.('hyp_time'));
	disp('The following Experiment description was loaded and is used...');
	EXPERIMENT = resultsOut.EXPERIMENT;
	EXPERIMENT.NUM_HYPER_OPT_ITERATIONS = iters;
%	if len_old ~= len
%		error('The current number of hyper-parameter optimization steps (%d) is different from the previous number of steps (%d). This is not allowed yet.', len, len_old);
%	else
	if len_old >= abs(iters)
		first_trial_id = trials_old + 1;
	end
%	end
end
if first_trial_id <= EXPERIMENT.NUM_TRIALS
    m = EXPERIMENT.M;
    for trial_id = first_trial_id:EXPERIMENT.NUM_TRIALS
	if trial_id > size(resultOut.hyp_time, 1)
		len_old = 0;
	else
		EXPERIMENT.LAST_HYPERS{trial_id} = resoutOut.hyp_over_time{triald_id}(:, end); 
	end
	seed = EXPERIMENT.SEED;
    	resultsOut.('seeds'){trial_id} = seed;

        for i=1:size(times)
            if times(i) < 0, break, end
	    resultOut.('hyp_time')(trial_id, i) = times(i);
	    resultOut.('hyp_over_time'){trial_id}(:, i) = theta_over_time(:, i);
            %disp(sprintf('Calculating MSE for iteration %d', i)); 
            resultOut.('msll')(trial_id, i) = mnlp(mF(:, i),testY,s2F(:, i), meanTest, varTest);
            resultOut.('mse')(trial_id, i)  = mse(mF(:, i),testY, meanTest, varTest);
            resultOut.('llh')(trial_id, i) = nlZ(i);   
        end
	resultOut.('EXPERIMENT') = EXPERIMENT;
	resultOut
        eval(sprintf('%s=resultOut;', resultVarName));
        %save(results_file, resultVarName);
    end
end
