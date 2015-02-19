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

    m = EXPERIMENT.M;

    for trial_id = 1:EXPERIMENT.NUM_TRIALS
	seed = floor(rand(1) * 32000);
	EXPERIMENT.SEED = seed;
	rng('default');
	rng(seed);

        resultOut.('hyps'){trial_id}=[];
        resultOut.('sod'){trial_id} = zeros(m, 1);

        disp(sprintf('testSoD: m = %d, trial %d.', m, trial_id));

        %----------------------------------------
        % Optimize hyperparameters.
        %----------------------------------------
        hypTic=tic;
        [hyp, theta_over_time, retvals] = feval([EXPERIMENT.METHOD '_hyper_opt'], EXPERIMENT, trainX, trainY, m, D);
        hypTime = toc(hypTic);

        %----------------------------------------
        % Compute predictive mean and variance.
        %----------------------------------------
        predTic=tic;
        %I just want the time
        %TODO: replace with function
        methodName = [EXPERIMENT.METHOD '_predict'];
        [~, ~] = feval(methodName, EXPERIMENT, hyp, trainX, trainY, testX, retvals);
        predTime=toc(predTic);
        
        %----------------------------------------
        % Save data.
        %----------------------------------------
        resultOut.('hyp_time')(trial_id, :) = theta_over_time(1, :);
        num_hyps = size(unwrap(hyp), 1) + 1;
        
        % I just assignt the same prediction time
        resultOut.('train_time')(trial_id, :) = predTime-testTime;
        resultOut.('test_time')(trial_id, :) = testTime;
        for i=1:size(theta_over_time, 2)
            if theta_over_time(1, i) < 0, break, end
            disp(sprintf('Calculating MSE for iteration %d', i)); 
            resultOut.('hyps'){trial_id} = [resultOut.('hyps'){trial_id} theta_over_time(2:num_hyps, i)];
            [mF, s2F, nlZ] = feval(methodName, EXPERIMENT, rewrap(hyp, theta_over_time(2:num_hyps, i)), trainX, trainY, testX, retvals);
            resultOut.('msll')(trial_id, i) = mnlp(mF,testY,s2F, meanTest, varTest);
            resultOut.('mse')(trial_id, i)  = mse(mF,testY, meanTest, varTest);
            resultOut.('llh')(trial_id, i) = nlZ;
            mF = feval(methodName, EXPERIMENT, rewrap(hyp, theta_over_time(2:num_hyps, i)), trainX, trainY, trainX, retvals);
            resultOut.('tmse')(trial_id, i)  = mse(mF,trainY, meanTrain, varTrain);

        end
        resultVarName = sprintf('results%s', EXPERIMENT.METHOD);
        eval(sprintf('%s=resultOut;', resultVarName));
        save(sprintf('%s%s_%s_fold%d', EXPERIMENT.RESULTS_DIR, resultVarName, EXPERIMENT.DATASET, EXPERIMENT.DATASET_FOLD), resultVarName);
    end
