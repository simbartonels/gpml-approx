% testHybrid: test the hybrid SoD/FITC approximation's error vs time
% performance.
%
% REQUIRES:
% The script uses the EXPERIMENT structure containing 
% experiment parameters, as explained in RUN_TESTS.m.
%
% EXPERIMENT.M    -- contains a list of the inducing point set sizes used
%
% EXPERIMENT.EXTRA should be set to one of:
%    'clustering' -- the data subset to be used will
%                    be chosen as means of the Farthest Point
%                    Clustering algorithm.
%    'random'     -- the data subset to be used will be
%                    chosen at random.
%
% RETURNS:
% After running the experiments, variable resultsHybrid will
% contain a structure with the results of the tests: 
%    resultsHybrid.msll   -- mean standardized log loss,
%    resultsHybrid.mse    -- standardized mean squared error, 
%    resultsHybrid.times  -- computation times,
%    resultsHybrid.hyps   -- the hyperparameters chosen by SoD,
%    resultsHybrid.N_test -- number of test points in the dataset.
%    resultsHybrid.N_train      -- number of training points.
%    resultsHybrid.sod    -- the indices of the datapoints used 
%                       in the approximation.
%
% This variable is saved in EXPERIMENT.RESULTS_DIR directory.
%
% Krzysztof Chalupka, University of Edinburgh 
% and California Institute of Technology, 2012
% kjchalup@caltech.edu

addpath('../code/gpml')
addpath('../code/figtree-0.9.3/matlab')
addpath(genpath('../code/project'))
maxNumCompThreads(1); % Use only one core, for fair comparison.
startup

% The global variable testTime should store 
% the test time of the algorithm, i.e. after
% the hyperparameters are estimated and the 
% covariance matrix is inverted, to compute the
% predictive mean & variances. See ../code/gpml/gp.m
% which I modified to keep track of this variable.
global testTime;
varTest=var(testY);
meanTest=mean(testY);

resultsHybrid.('msll') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));
resultsHybrid.('mse') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsHybrid.('hyp_time') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsHybrid.('train_time') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsHybrid.('test_time') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsHybrid.N_train = length(trainY);
resultsHybrid.N_test = length(testY)

for m_id = 1:length(EXPERIMENT.M)
    m = EXPERIMENT.M(m_id);
    resultsHybrid.('hyps'){m_id}=[];
    resultsHybrid.('sod'){m_id} = zeros(EXPERIMENT.NUM_TRIALS, m);

    for trial_id = 1:EXPERIMENT.NUM_TRIALS
        disp(sprintf('testHybrid: m = %d, trial %d.', m, trial_id));
        %----------------------------------------
        % Initialize hyperparameters.
        %----------------------------------------
        meanfunc = []; 
        hyp.cov = [zeros(D,1);0];
        likfunc = @likGauss; sn = 0.25; hyp.lik = log(sn);

        %----------------------------------------
        % Optimize hyperparameters using SoD.
        %----------------------------------------
        hypTic=tic;
        if strcmp(EXPERIMENT.EXTRA, 'clustering')
            [hyp, sod] = gp_sod(hyp, {@covSEard}, likfunc, trainX, trainY, m, 'c', D, 'split');
        else
            [hyp, sod] = gp_sod(hyp, {@covSEard}, likfunc, trainX, trainY, m, 'r');
        end 
        hypTime = toc(hypTic);
        
        %-------------------------------------------------
        % Compute predictive mean and variance using FITC.
        %-------------------------------------------------
        indPts = trainX(sod,:);
        covfuncFITC = {@covFITC, {@covSEard}, indPts};
        predTic=tic;
        [mF s2F] = gp(hyp, @infFITC, meanfunc, covfuncFITC, likfunc, trainX, trainY, testX);
        predTime=toc(predTic);
        
        %----------------------------------------
        % Save data.
        %----------------------------------------
        resultsHybrid.('msll')(trial_id, m_id) = mnlp(mF,testY,s2F, meanTest, varTest);
        resultsHybrid.('mse')(trial_id, m_id)  = mse(mF,testY, meanTest, varTest);
        resultsHybrid.('hyp_time')(trial_id, m_id) = hypTime;
        resultsHybrid.('train_time')(trial_id, m_id) = predTime-testTime;
        resultsHybrid.('test_time')(trial_id, m_id) = testTime;
        resultsHybrid.('hyps'){m_id} = [resultsHybrid.('hyps'){m_id} [hyp.cov; hyp.lik]];
        resultsHybrid.('sod'){m_id}(trial_id, :) = sod;
        save(sprintf('%sresultsHybrid_%s', EXPERIMENT.RESULTS_DIR, EXPERIMENT.DATASET), 'resultsHybrid');
    end
end
