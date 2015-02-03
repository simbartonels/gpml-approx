% testSoD: test the SoD approximation's error vs time
% performance.
%
% REQUIRES:
% The script uses the EXPERIMENT structure containing 
% experiment parameters, as explained in RUN_TESTS.m.
%
% EXPERIMENT.M    -- contains a list of the subset sizes used
%
% EXPERIMENT.EXTRA should be set to one of:
%    'clustering' -- the data subset to be used will
%                    be chosen as means of the Farthest Point
%                    Clustering algorithm.
%    'random'     -- the data subset to be used will be
%                    chosen at random.
%
% RETURNS:
% After running the experiments, variable resultsSoD will
% contain a structure with the results of the tests: 
%    resultsSoD.msll   -- mean standardized log loss,
%    resultsSoD.mse    -- standardized mean squared error, 
%    resultsSoD.times  -- computation times,
%    resultsSoD.hyps   -- the hyperparameters chosen by SoD,
%    resultsSoD.N_test -- number of test points in the dataset.
%    resultsSoD.N_train      -- number of training points.
%    resultsSoD.sod    -- the indices of the datapoints used 
%                       in the approximation.
%
% This variable is saved in EXPERIMENT.RESULTS_DIR directory.
%
% Krzysztof Chalupka, University of Edinburgh 
% and California Institute of Technology, 2012
% kjchalup@caltech.edu

addpath('../code/gpml')
maxNumCompThreads(1); % Use only one core, for fair comparison.
startup

loadData;

% The global variable testTime should store 
% the test time of the algorithm, i.e. after
% the hyperparameters are estimated and the 
% covariance matrix is inverted, to compute the
% predictive mean & variances. See ../code/gpml/gp.m
% which I modified to keep track of this variable.
global testTime;
varTest=var(testY);
meanTest=mean(testY);

resultsSoD.('msll') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));
resultsSoD.('mse') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsSoD.('hyp_time') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsSoD.('train_time') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsSoD.('test_time') = zeros(EXPERIMENT.NUM_TRIALS, length(EXPERIMENT.M));;
resultsSoD.N_train = length(trainY);
resultsSoD.N_test = length(testY)

m = 1;
m_id = 1;
resultsSoD.('hyps'){m_id}=[];
resultsSoD.('sod'){m_id} = zeros(EXPERIMENT.NUM_TRIALS, m);

for trial_id = 1:EXPERIMENT.NUM_TRIALS
disp(sprintf('testFullMatern: m = %d, trial %d.', m, trial_id));
%----------------------------------------
% Initialize hyperparameters.
%----------------------------------------
meanfunc = []; 
hyp.cov = [0;0];
likfunc = @likGauss; sn = 0.25; hyp.lik = log(sn);

%----------------------------------------
% Optimize hyperparameters.
%----------------------------------------
hypTic=tic;
%[hyp, sod] = gp_sod(hyp, {@covSEard}, likfunc, trainX, trainY, m, 'r');
disp('Optimizing hyper-parameters.');
hyp = minimize(hyp, @gp, -100, @infExact, [], {@covMaterniso, 5}, likfunc, trainX, trainY);
hypTime = toc(hypTic);

%----------------------------------------
% Compute predictive mean and variance.
%----------------------------------------
predTic=tic;
disp('Predicting for test inputs.');
[mF s2F] = gp(hyp, @infExact, [], {@covMaterniso, 5}, likfunc, trainX, trainY, testX);
predTime=toc(predTic);

%----------------------------------------
% Save data.
%----------------------------------------
resultsSoD.('msll')(trial_id, m_id) = mnlp(mF,testY,s2F, meanTest, varTest);
resultsSoD.('mse')(trial_id, m_id)  = mse(mF,testY, meanTest, varTest);
resultsSoD.('hyp_time')(trial_id, m_id) = hypTime;
resultsSoD.('train_time')(trial_id, m_id) = predTime-testTime;
resultsSoD.('test_time')(trial_id, m_id) = testTime;
resultsSoD.('hyps'){m_id} = [resultsSoD.('hyps'){m_id} [hyp.cov; hyp.lik]];
save(sprintf('%sresultsFullMatern_%s_fold%d', EXPERIMENT.RESULTS_DIR, EXPERIMENT.DATASET, EXPERIMENT.DATASET_FOLD), 'resultsSoD');
end
