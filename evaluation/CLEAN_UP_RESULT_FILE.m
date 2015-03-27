for d = 1:10
EXPERIMENT.METHOD = 'SoD';
EXPERIMENT.DATASET = 'PRECIPITATION';
EXPERIMENT.DATASET_FOLD = d;
EXPERIMENT.M = 1940;
EXPERIMENT.NUM_TRIALS = 0;
EXPERIMENT.RESULTS_DIR = './results/'; 
resultVarName = sprintf('results%s', EXPERIMENT.METHOD);
results_file = sprintf('%s%s%s%s%s_fold%d_M%d.mat', EXPERIMENT.RESULTS_DIR, '', EXPERIMENT.DATASET, filesep, resultVarName, EXPERIMENT.DATASET_FOLD, EXPERIMENT.M)
if exist(results_file, 'file') == 2
	disp('WARNING: An experiment results file already exists for this configuration. Attempting continuation of the experiment.');
	a = load(results_file);
	resultOut = eval(['a.' resultVarName]);
    [trials_old, len_old] = size(resultOut.('hyp_time'));
	disp('The following Experiment description was loaded and is used...');
    if exist('resultOut.EXPERIMENT')
        EXPERIMENT = resultOut.EXPERIMENT; 
    else
        EXPERIMENT.NUM_TRIALS = trials_old;
    end
	EXPERIMENT.NUM_HYPER_OPT_ITERATIONS = len_old;
else
    error('Result file not found.');
end
if EXPERIMENT.NUM_TRIALS > 1, error('Can not clean this yet'); end
first_trial_id = 1;
if first_trial_id <= EXPERIMENT.NUM_TRIALS
    m = EXPERIMENT.M;
    for trial_id = first_trial_id:EXPERIMENT.NUM_TRIALS
        times = resultOut.hyp_time(trial_id, :);
        theta_over_time = resultOut.hyp_over_time{trial_id};
        msll = resultOut.('msll')(trial_id, :);
        mse = resultOut.('mse')(trial_id, :);
        nlZ = resultOut.('llh')(trial_id, :);
        try
            seed = resultOut.('seeds'){trial_id};
        catch
            try
                seed = resultOut.('seeds')(trial_id);
            catch
                disp('Defaulting seed to 0');
                seed = 0;
            end
        end
        clear resultOut;
    	EXPERIMENT.SEED{trial_id} = seed;
        resultOut.seeds{trial_id} = seed;

        for i=1:size(times, 2)
            if times(i) < 0, break, end
    	    resultOut.('hyp_time'){trial_id}(i) = times(i);
            resultOut.('hyp_over_time'){trial_id}(:, i) = theta_over_time(:, i);
            %disp(sprintf('Calculating MSE for iteration %d', i)); 
            resultOut.('msll'){trial_id}(i) = msll(i);
            resultOut.('mse'){trial_id}(i)  = mse(i);
            resultOut.('llh'){trial_id}(i) = nlZ(i);   
        end
        
        EXPERIMENT.LAST_HYPERS{trial_id} = resultOut.hyp_over_time{trial_id}(:, end)
        resultOut.('EXPERIMENT') = EXPERIMENT;
        EXPERIMENT
        EXPERIMENT.LAST_HYPERS{trial_id}
        resultOut
        eval(sprintf('%s=resultOut;', resultVarName));
        prompt = 'Do you want to overwrite the result file? y/[n]: ';
        answer = input(prompt, 's');
        if answer == 'y'
            save(results_file, resultVarName); 
            disp('Result file overwritten.');
        else
            disp('Taking no action.');
        end
    end
end
clear EXPERIMENT;
clear resultOut;
end