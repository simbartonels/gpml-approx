global testTime;
varTest=var(testY);
meanTest=mean(testY);
varTrain=var(trainY);
meanTrain=mean(trainY);

first_trial_id = 1;
iters = EXPERIMENT.NUM_HYPER_OPT_ITERATIONS;
len = abs(iters);
num_trials = EXPERIMENT.NUM_TRIALS
folds = EXPERIMENT.DATASET_FOLDS;
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
	resultOut = eval(resultVarName);
        trials_old = size(resultOut.('hyp_time'), 2);
	disp('The following Experiment description was loaded and is used...');
	EXPERIMENT = resultOut.EXPERIMENT;
	EXPERIMENT.NUM_HYPER_OPT_ITERATIONS = iters;
	EXPERIMENT.NUM_TRIALS = num_trials;
	EXPERIMENT
%	if len_old ~= len
%		error('The current number of hyper-parameter optimization steps (%d) is different from the previous number of steps (%d). This is not allowed yet.', len, len_old);
%	else
        
	time_offset = zeros([EXPERIMENT.NUM_TRIALS, 1]);
	for trial_id = 1:trials_old
		time_offset(trial_id) = resultOut.hyp_time{trial_id}(end);
	end
	for trial_id = (trials_old+1):EXPERIMENT.NUM_TRIALS
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
	resultOut.N_train = length(trainY);
	resultOut.N_test = length(testY);
	%resultOut.('seeds') = zeros(EXPERIMENT.NUM_TRIALS, 1);
	resultOut.('hyp_over_time') = {};
        first_trial_id = 1;

        save(results_file, 'resultOut'); %test if saving works
        delete(results_file);
	clear resultOut;
	resultOut.hyp_time = {}
	time_offset = zeros([EXPERIMENT.NUM_TRIALS, 1]);
end
EXPERIMENT.DATASET_FOLDS = folds;
if first_trial_id <= EXPERIMENT.NUM_TRIALS
    m = EXPERIMENT.M;
    for trial_id = first_trial_id:EXPERIMENT.NUM_TRIALS
	trial_id
	if trial_id > size(resultOut.hyp_time, 2)
	    len_old = 0;
        else
            len_old = size(resultOut.hyp_over_time{trial_id}, 2);
	    EXPERIMENT.LAST_HYPERS{trial_id} = resultOut.hyp_over_time{trial_id}(:, end); 
        end

	if len - len_old > 0
	seed = EXPERIMENT.SEED{trial_id};
	rng('default');
	rng(seed);
        resultOut.('seeds'){trial_id} = seed;

        %----------------------------------------
        % Optimize hyperparameters.
        %----------------------------------------
	EXPERIMENT.LAST_TRIAL = trial_id;
	EXPERIMENT.NUM_HYPER_OPT_ITERATIONS = len - len_old;
	[EXPERIMENT, times, theta_over_time, mF, s2F, nlZ, gradNorms, ~] = feval(EXPERIMENT.METHOD, EXPERIMENT, trainX, trainY, testX, trial_id);
	EXPERIMENT.NUM_HYPER_OPT_ITERATIONS = iters;
        %----------------------------------------
        % Save data.
        %----------------------------------------
        
        % I just assignt the same prediction time
        %resultOut.('train_time')(trial_id, :) = 0; %predTime-testTime;
        %resultOut.('test_time')(trial_id, :) = 0; %testTime;
        %resultOut.('hyps'){trial_id} = [resultOut.('hyps'){trial_id} theta_over_time]; %[resultOut.('hyps'){trial_id} rewrap(theta_over_time];
	len_old = abs(len_old);        
	for i=1:size(times, 1)
            if times(i) < 0, break, end
		    resultOut.('hyp_time'){trial_id}(i+len_old) = times(i) + time_offset(trial_id);
	        resultOut.('hyp_over_time'){trial_id}(:, i+len_old) = theta_over_time(:, i);
            resultOut.('msll'){trial_id}(i+len_old) = mnlp(mF(:, i),testY,s2F(:, i), meanTest, varTest);
            resultOut.('mse'){trial_id}(i+len_old)  = mse(mF(:, i),testY, meanTest, varTest);
            %resultOut.('tmse')(trial_id, i)  = mse(mFT(:, i),trainY, meanTrain, varTrain);
			resultOut.('llh'){trial_id}(i+len_old) = nlZ(i);
			resultOut.('grad_norms'){trial_id}(i+len_old) = gradNorms(i);
        end
        %disp('Last test error: ');
        %if size(mFT, 2) > 1, mFT = mFT(:, size(times, 1)); end
        %last_train_error = mse(mFT, trainY, meanTrain, varTrain)
        %last_test_error = mse(mF(:, size(times, 1)), testY, meanTest, varTest)
        disp('NaNs or Infs: ');
        any(any(isnan(mF) | isinf(abs(mF))))

        resultOut.('EXPERIMENT') = EXPERIMENT;
        eval(sprintf('%s=resultOut;', resultVarName));
        save(results_file, resultVarName);
	end

    end
end
disp('Done.');
