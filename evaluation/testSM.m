% testFITC: test the FITC approximation's error vs time
% performance.
%
% REQUIRES:
% The script uses the EXPERIMENT structure containing 
% experiment parameters, as explained in RUN_EXPERIMENT.m.
%
% EXPERIMENT.M    -- contains a list of the inducing point set sizes used
%
% EXPERIMENT.EXTRA should be set to one of:
%    'clustering' -- the inducing points' set will
%                    be chosen as means of the Farthest Point
%                    Clustering algorithm.
%    'random'     -- the inducing points will be
%                    chosen at random.
%
% RETURNS:
% After running the experiments, variable resultsFITC will
% contain a structure with the results of the tests: 
%    resultsFITC.msll   -- mean standardized log loss,
%    resultsFITC.mse    -- standardized mean squared error, 
%    resultsFITC.times  -- computation times,
%    resultsFITC.hyps   -- the hyperparameters chosen by FITC,
%    resultsFITC.N_test -- number of test points in the dataset.
%    resultsFITC.N_train      -- number of training points.
%    resultsFITC.indPts -- the indices of the datapoints used 
%                          in the approximation.
%
% This variable is saved in EXPERIMENT.RESULTS_DIR directory.
%
% Krzysztof Chalupka, University of Edinburgh 
% and California Institute of Technology, 2012
% kjchalup@caltech.edu

addpath('../code/gpml')
addpath('../code/project/inducing')
maxNumCompThreads(1); % Use one core.
startup

% Load the data.
loadData

% The global variable testTime should store 
% the test time of the algorithm, i.e. after
% the hyperparameters are estimated and the 
% covariance matrix is inverted, to compute the
% predictive mean & variances. See ../code/gpml/gp.m
% which I modified to keep track of this variable.
global testTime
varTest = var(testY,1);
meanTest = mean(testY);

resultsFITC.('msll') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));
resultsFITC.('mse') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsFITC.('hyp_time') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsFITC.('train_time') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsFITC.('test_time') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsFITC.N_train = length(trainY);
resultsFITC.N_test = length(testY)

for m_id = 1:length(EXPERIMENT.M)
    m = EXPERIMENT.M(m_id);
    resultsFITC.('hyps'){m_id}={};
    for trial_id = 1:EXPERIMENT.NUM_TRIALS
        disp(sprintf('testFITC: m = %d, trial %d.', m, trial_id));

        %----------------------------------------
        % Initialize hyperparameters.
        %----------------------------------------
        meanfunc = []; 
	[rx, ci, cc, np, cr] = KCenterClustering(D, n, trainX', m);
        indPts = cc';
        hyp.cov = [0.5*log((max(trainX)-min(trainX))'/2); 0];
        likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
        
        %----------------------------------------
        % Optimize hyperparameters.
        %----------------------------------------
        hypTic=tic;

        covfuncFITC = {@covSM, m};
        try
            hyp = minimize(hyp, @gp, -100, {@infFITC}, meanfunc, covfuncFITC, likfunc,trainX, trainY);
            hypTime = toc(hypTic);
            
            %----------------------------------------
            % Compute predictive mean and variance.
            %----------------------------------------
            predTic=tic;
            [mF s2F] = gp(hyp, @infFITC, meanfunc, covfuncFITC, likfunc, trainX, trainY, testX);
            predTime=toc(predTic);
            
            %----------------------------------------
            % Save data.
            %----------------------------------------
            resultsFITC.('msll')(trial_id, m_id) = mnlp(mF,testY,s2F, meanTest, varTest);
            resultsFITC.('mse')(trial_id, m_id)  = mse(mF,testY, meanTest, varTest);
            resultsFITC.('hyp_time')(trial_id, m_id) = hypTime;
            resultsFITC.('train_time')(trial_id, m_id) = predTime-testTime;
            resultsFITC.('test_time')(trial_id, m_id) = testTime;
            resultsFITC.('hyps'){m_id} = [resultsFITC.('hyps'){m_id} {[hyp.cov; hyp.lik]}];
        
        catch % FITC doesn't seem perfectly stable. If something goes wrong,
              % don't stop the test, just put infinities in the results.
            resultsFITC.('msll')(trial_id, m_id) = inf;
            resultsFITC.('mse')(trial_id, m_id)  = inf;
            resultsFITC.('hyp_time')(trial_id, m_id) = inf;
            resultsFITC.('train_time')(trial_id, m_id) = inf;
            resultsFITC.('test_time')(trial_id, m_id) = inf;
            resultsFITC.('hyps'){m_id} = [resultsFITC.('hyps'){m_id} {[hyp.cov*0; hyp.lik*0]}];
        end
        save(sprintf('%sresultsSM_%s_fold%d.mat', EXPERIMENT.RESULTS_DIR, EXPERIMENT.DATASET, EXPERIMENT.DATASET_FOLD),'resultsFITC');
    end
end
