global testTime;
varTest=var(testY);
meanTest=mean(testY);
varTrain=var(trainY);
meanTrain=mean(trainY);

len = abs(EXPERIMENT.NUM_HYPER_OPT_ITERATIONS);
resultOut.('msll') = zeros(EXPERIMENT.NUM_TRIALS, len);
resultOut.('mse') = zeros(EXPERIMENT.NUM_TRIALS, len);;
resultOut.('llh') = zeros(EXPERIMENT.NUM_TRIALS, len);;
resultOut.('tmse') = zeros(EXPERIMENT.NUM_TRIALS, len);;

resultOut.('hyp_time') = zeros(EXPERIMENT.NUM_TRIALS, len);;
resultOut.('train_time') = zeros(EXPERIMENT.NUM_TRIALS, len);;
resultOut.('test_time') = zeros(EXPERIMENT.NUM_TRIALS, len);;
resultOut.N_train = length(trainY);
resultOut.N_test = length(testY);
resultOut.('seeds') = zeros(EXPERIMENT.NUM_TRIALS, 1);
resultOut.('hyp_over_time') = {};

    m = EXPERIMENT.M;

    for trial_id = 1:EXPERIMENT.NUM_TRIALS
	seed = floor(rand(1) * 32000);
	EXPERIMENT.SEED = seed;
	rng('default');
	rng(seed);

        %----------------------------------------
        % Optimize hyperparameters.
        %----------------------------------------
        [times, theta_over_time, mF, s2F, nlZ, mFT] = feval(EXPERIMENT.METHOD, EXPERIMENT, trainX, trainY, testX, trial_id);
        %----------------------------------------
        % Save data.
        %----------------------------------------
        resultOut.('hyp_time')(trial_id, :) = times;
        
        % I just assignt the same prediction time
        resultOut.('train_time')(trial_id, :) = 0; %predTime-testTime;
        resultOut.('test_time')(trial_id, :) = 0; %testTime;
        %resultOut.('hyps'){trial_id} = [resultOut.('hyps'){trial_id} theta_over_time]; %[resultOut.('hyps'){trial_id} rewrap(theta_over_time];
        resultOut.('hyp_over_time'){trial_id} = theta_over_time;
        for i=1:size(times)
            if times(i) < 0, break, end
            disp(sprintf('Calculating MSE for iteration %d', i)); 
            resultOut.('msll')(trial_id, i) = mnlp(mF(:, i),testY,s2F(:, i), meanTest, varTest);
            resultOut.('mse')(trial_id, i)  = mse(mF(:, i),testY, meanTest, varTest);
            %resultOut.('tmse')(trial_id, i)  = mse(mFT(:, i),trainY, meanTrain, varTrain);
        end
        disp('Training error: ');
        if size(mFT, 2) > 1, mFT = mFT(:, size(times, 1)); end
        last_train_error = mse(mFT, trainY, meanTrain, varTrain)
        last_test_error = mse(mF(:, size(times, 1)), testY, meanTest, varTest)
        disp('NaNs or Infs: ');
        any(isnan(mFT) | isinf(abs(mFT)))

        resultOut.('llh')(trial_id, :) = nlZ;    
        resultVarName = sprintf('results%s', EXPERIMENT.METHOD);
        eval(sprintf('%s=resultOut;', resultVarName));
        save(sprintf('%s%s_%s_fold%d', EXPERIMENT.RESULTS_DIR, resultVarName, EXPERIMENT.DATASET, EXPERIMENT.DATASET_FOLD), resultVarName);
    end
