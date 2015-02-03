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
% After running the experiments, variable resultsLocal will
% contain a structure with the results of the tests: 
%    resultsLocal.msll   -- mean standardized log loss,
%    resultsLocal.mse    -- standardized mean squared error, 
%    resultsLocal.times  -- computation times,
%    resultsLocal.hyps   -- the hyperparameters chosen by Local GP,
%    resultsLocal.cis    -- the indices of the clusters to which each 
%                           training point belongs.
%    resultsLocal.N_test -- number of test points in the dataset.
%    resultsLocal.N_train      -- number of training points.
%
% This variable is saved in EXPERIMENT.RESULTS_DIR directory.
%
% Krzysztof Chalupka, University of Edinburgh 
% and California Institute of Technology, 2012
% kjchalup@caltech.edu


addpath(genpath('../code/project'));
addpath(genpath('../code/gpml'));
maxNumCompThreads(1);
startup

% Load the data.
loadData

% Global variable to keep track of the test time. 
% See ../code/project/local/gp_local.m.
global testTimeBig;

varTest = var(testY,1);
meanTest = mean(testY);

resultsLocal.('msll') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));
resultsLocal.('mse') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsLocal.('hyp_time') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsLocal.('train_time') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsLocal.('test_time') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsLocal.N_train = length(trainY);
resultsLocal.N_test = length(testY);

for m_id = 1:length(EXPERIMENT.M)
    m = EXPERIMENT.M(m_id);
    resultsLocal.('hyps'){m_id} = [];
    resultsLocal.('cis'){m_id}  = [];
    for trial_id = 1:EXPERIMENT.NUM_TRIALS
        disp(sprintf('testLocal: m = %d, trial %d.', m, trial_id));
        %----------------------------------------
        % Initialize hyperparameters.
        %----------------------------------------
        meanfunc = []; 
        covfunc = {@covSEard}; hyp = []; hyp.cov = zeros(D+1,1);
        likfunc = @likGauss; sn = 0.25; hyp.lik = log(sn);

        %----------------------------------------
        % Optimize hyperparameters 
        % (including clustering!)
        %----------------------------------------
        hypTic=tic;
        [ci splits splitAxes] = rpClust(trainX, m);
        hyp = gp_local(hyp, covfunc, meanfunc, likfunc, trainX, trainY, ci, EXPERIMENT.EXTRA);
        hypTime = toc(hypTic);

        %----------------------------------------
        % Compute predictive mean and variance.
        %----------------------------------------
        predStart=tic;
        [mF s2F] = gp_local(hyp, covfunc, meanfunc, likfunc, trainX, trainY, ci, splits, splitAxes, testX);
        predTime=toc(predStart);

        %----------------------------------------
        % Save data.
        %----------------------------------------
        resultsLocal.('msll')(trial_id, m_id) = mnlp(mF,testY,s2F, meanTest, varTest);
        resultsLocal.('mse')(trial_id, m_id)  = mse(mF,testY, meanTest, varTest);
        resultsLocal.('hyp_time')(trial_id, m_id) = hypTime;
        resultsLocal.('train_time')(trial_id, m_id) = predTime-testTimeBig;
        resultsLocal.('test_time')(trial_id, m_id) = testTimeBig;
        if strcmp(EXPERIMENT.EXTRA, 'joint')
            resultsLocal.('hyps'){m_id} = [resultsLocal.('hyps'){m_id} [hyp.cov; hyp.lik]];
        elseif strcmp(EXPERIMENT.EXTRA, 'separate')
            resultsLocal.('hyps'){m_id} = [resultsLocal.('hyps'){m_id} {hyp}];
        end
        resultsLocal.('cis'){m_id}(trial_id, :) = ci;
        save(sprintf('%sresultsLocal_%s', EXPERIMENT.RESULTS_DIR, EXPERIMENT.DATASET), 'resultsLocal');
    end
end
