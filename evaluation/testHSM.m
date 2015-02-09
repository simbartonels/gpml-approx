% testLocal: test the Local GP approximation's error vs time
% performance.
%
% REQUIRES:
% The script uses the EXPERIMENT structure containing 
% experiment parameters, as explained in RUN_EXPERIMENTS.m.
%
% EXPERIMENT.M    -- contains a list of the cluster sizes used
%
% EXPERIMENT.EXTRA should be set to one of:
%     'separate'  -- GP hyperparameters will be estimated 
%                    separately for each cluster. 
%     'joint'     -- the likelihood will sum over all 
%                    clusters and the hypers will be shared.
%
% RETURNS:
% After running the experiments, variable resultsHSM will
% contain a structure with the results of the tests: 
%    resultsHSM.msll   -- mean standardized log loss,
%    resultsHSM.mse    -- standardized mean squared error, 
%    resultsHSM.times  -- computation times,
%    resultsHSM.hyps   -- the hyperparameters chosen by Local GP,
%    resultsHSM.cis    -- the indices of the clusters to which each 
%                           training point belongs.
%    resultsHSM.N_test -- number of test points in the dataset.
%    resultsHSM.N_train      -- number of training points.
%
% This variable is saved in EXPERIMENT.RESULTS_DIR directory.
%
% Krzysztof Chalupka, University of Edinburgh 
% and California Institute of Technology, 2012
% kjchalup@caltech.edu

% Load the data.
%loadData

% Global variable to keep track of the test time. 
% See ../code/project/local/gp_local.m.
global testTime;

varTest = var(testY,1);
meanTest = mean(testY);

len = abs(EXPERIMENT.NUM_HYPER_OPT_ITERATIONS)+1;
resultsHSM.('msll') = zeros(EXPERIMENT.NUM_TRIALS, len);
resultsHSM.('mse') = zeros(EXPERIMENT.NUM_TRIALS, len);
resultsHSM.('hyp_time') = zeros(EXPERIMENT.NUM_TRIALS, len);
resultsHSM.('train_time') = zeros(EXPERIMENT.NUM_TRIALS, len);
resultsHSM.('test_time') = zeros(EXPERIMENT.NUM_TRIALS, len);
resultsHSM.N_train = length(trainY);
resultsHSM.N_test = length(testY);
L = 1.2 * max(abs(trainX));
    m = EXPERIMENT.M;
    %resultsHSM.('hyps'){m_id} = [];
    for trial_id = 1:EXPERIMENT.NUM_TRIALS
        resultsHSM.('hyps'){trial_id} = [];

        disp(sprintf('testHSM: m = %d, trial %d.', m, trial_id));
        %----------------------------------------
        % Initialize hyperparameters.
        %----------------------------------------
        meanfunc = []; 
        D = size(trainX, 2);
        [J, lambda] = initHSM(m, D, L);
        covfunc = {@covDegenerate, {@degHSM2, m, L, J, lambda}}; 
        %covfunc = {@degHSM2, m, L, J, lambda};
        hyp = []; hyp.cov = zeros(D+1,1);
        likfunc = @likGauss; sn = 0.25; hyp.lik = log(sn);

        %----------------------------------------
        % Optimize hyperparameters 
        % (including clustering!)
        %----------------------------------------
        hypTic=tic;
        [hyp, ~, ~, theta_over_time] = minimize(hyp, @gp, EXPERIMENT.NUM_HYPER_OPT_ITERATIONS, {@infSolinfast}, meanfunc, covfunc, likfunc, trainX, trainY);
        hypTime = toc(hypTic);

        %----------------------------------------
        % Compute predictive mean and variance.
        %----------------------------------------
        predStart=tic;
        [mF s2F] = gp(hyp, @infSolinfast, meanfunc, covfunc, likfunc, trainX, trainY, testX);
        predTime=toc(predStart);
        
        %----------------------------------------
        % Save data.
        %----------------------------------------
        %resultsHSM.('msll')(trial_id, m_id) = mnlp(mF,testY,s2F, meanTest, varTest);
        %resultsHSM.('mse')(trial_id, m_id)  = mse(mF,testY, meanTest, varTest);
        %resultsHSM.('hyp_time')(trial_id, m_id) = hypTime;
        %resultsHSM.('train_time')(trial_id, m_id) = predTime-testTime;
        %resultsHSM.('test_time')(trial_id, m_id) = testTime;
        %resultsHSM.('hyps'){trial_id} = [hyp.cov; hyp.lik];
        
        resultsHSM.('hyp_time')(trial_id, :) = theta_over_time(1, :);
        num_hyps = size(unwrap(hyp), 1) + 1;
        for i=1:size(theta_over_time, 2)
            if theta_over_time(1, i) < 0, break, end
            resultsHSM.('hyps'){trial_id} = [resultsHSM.('hyps'){trial_id} theta_over_time(2:num_hyps, i)];
            [mF, s2F] = gp(rewrap(theta_over_time(2, i), hyp), @infExactDegKernel, meanfunc, covfunc, likfunc, trainX, trainY, testX);
            resultsHSM.('mse')(trial_id, i) = mse(mF, testY, meanTest, varTest);
            resultsHSM.('msll')(trial_id, i) = mnlp(mF, testY, s2F, meanTest, varTest);
        end
        save(sprintf('%sresultsHSM_%s_fold%d', EXPERIMENT.RESULTS_DIR, EXPERIMENT.DATASET, EXPERIMENT.DATASET_FOLD), 'resultsHSM');
    end