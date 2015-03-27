error('Make sure to test this method first.');
global testTime;
varTest=var(testY);
meanTest=mean(testY);
varTrain=var(trainY);
meanTrain=mean(trainY);

iters = EXPERIMENT.NUM_HYPER_OPT_ITERATIONS;
len = abs(iters);
num_trials = EXPERIMENT.NUM_TRIALS
for trial_id = 1:num_trials
	seed = floor(rand(1) * 32000);
	EXPERIMENT.SEED{trial_id} = seed;
end

%check if there already some results

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
	EXPERIMENT.NUM_TRIALS = num_trials;
%	if len_old ~= len
%		error('The current number of hyper-parameter optimization steps (%d) is different from the previous number of steps (%d). This is not allowed yet.', len, len_old);
%	else
	if len_old >= abs(iters)
		first_trial_id = trials_old + 1;
	end
	for trial_id = trials_old + 1:EXPERIMENT.NUM_TRIALS
		seed = floor(rand(1) * 32000);
		EXPERIMENT.SEED{trial_id} = seed;
	end
%	end
else
	len_old = 0;
	%resultOut.('msll') = zeros(EXPERIMENT.NUM_TRIALS, len);
	%resultOut.('mse') = zeros(EXPERIMENT.NUM_TRIALS, len);;
	%resultOut.('llh') = zeros(EXPERIMENT.NUM_TRIALS, len);;
	%resultOut.('tmse') = zeros(EXPERIMENT.NUM_TRIALS, len);;

	%resultOut.('hyp_time') = -ones(EXPERIMENT.NUM_TRIALS, len);;
	%resultOut.('train_time') = zeros(EXPERIMENT.NUM_TRIALS, len);;
	%resultOut.('test_time') = zeros(EXPERIMENT.NUM_TRIALS, len);;
	
	resultOut.('msll') = zeros(1, 1);
	resultOut.('mse') = zeros(1, 1);;
	resultOut.('llh') = zeros(1, 1);;
	resultOut.('tmse') = zeros(1, 1);;

	resultOut.('hyp_time') = -ones(1, 1);;
	resultOut.('train_time') = zeros(1, 1);;
	resultOut.('test_time') = zeros(1, 1);;
	resultOut.N_train = length(trainY);
	resultOut.N_test = length(testY);
	%resultOut.('seeds') = zeros(EXPERIMENT.NUM_TRIALS, 1);
	resultOut.('hyp_over_time') = {};
        first_trial_id = 1;

        save(results_file, 'resultOut'); %test if saving works
        delete(results_file);
end
if first_trial_id <= EXPERIMENT.NUM_TRIALS
    m = EXPERIMENT.M;
    for trial_id = first_trial_id:EXPERIMENT.NUM_TRIALS
	if trial_id > size(resultOut.hyp_time, 1)
		len_old = 0;
	else
		EXPERIMENT.LAST_HYPERS{trial_id} = resoutOut.hyp_over_time{triald_id}(:, end); 
	end
	seed = EXPERIMENT.SEED{trial_id};
	rng('default');
	rng(seed);
    	resultsOut.('seeds'){trial_id} = seed;

        %----------------------------------------
        % Optimize hyperparameters.
        %----------------------------------------
	EXPERIMENT.LAST_TRIAL = trial_id;
	EXPERIMENT.NUM_HYPER_OPT_ITERATIONS = len - len_old;
        [times, theta_over_time, mF, s2F, nlZ, ~] = feval(EXPERIMENT.METHOD, EXPERIMENT, trainX, trainY, testX, trial_id);
	EXPERIMENT.NUM_HYPER_OPT_ITERATIONS = iters;
        %----------------------------------------
        % Save data.
        %----------------------------------------
        
        % I just assignt the same prediction time
        %resultOut.('train_time')(trial_id, :) = 0; %predTime-testTime;
        %resultOut.('test_time')(trial_id, :) = 0; %testTime;
        %resultOut.('hyps'){trial_id} = [resultOut.('hyps'){trial_id} theta_over_time]; %[resultOut.('hyps'){trial_id} rewrap(theta_over_time];
        for i=1:size(times)
            if times(i) < 0, break, end
	    resultOut.('hyp_time')(trial_id, i) = times(i);
	    resultOut.('hyp_over_time'){trial_id}(:, i) = theta_over_time(:, i);
            %disp(sprintf('Calculating MSE for iteration %d', i)); 
            resultOut.('msll')(trial_id, i) = mnlp(mF(:, i),testY,s2F(:, i), meanTest, varTest);
            resultOut.('mse')(trial_id, i)  = mse(mF(:, i),testY, meanTest, varTest);
            %resultOut.('tmse')(trial_id, i)  = mse(mFT(:, i),trainY, meanTrain, varTrain);
	    resultOut.('llh')(trial_id, i) = nlZ(i);    
        end
        disp('Last test error: ');
        %if size(mFT, 2) > 1, mFT = mFT(:, size(times, 1)); end
        %last_train_error = mse(mFT, trainY, meanTrain, varTrain)
        last_test_error = mse(mF(:, size(times, 1)), testY, meanTest, varTest)
        disp('NaNs or Infs: ');
        any(isnan(mFT) | isinf(abs(mFT)))

	resultOut.('EXPERIMENT') = EXPERIMENT;
        eval(sprintf('%s=resultOut;', resultVarName));
        save(results_file, resultVarName);
    end
end
