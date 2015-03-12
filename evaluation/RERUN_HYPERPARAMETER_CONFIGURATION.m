datasets = 'PUMADYN';
EXPERIMENT.DATASET_FOLD = 4
approximation = 'degenerate';
%bfname = 'SparseMultiScaleGP';
bfname = 'FastFood';
EXPERIMENT.METHOD = 'FastFood';
EXPERIMENT.M = 2048;
m = 20;


% Choose the approximation to use: SoD, Local, FITC.
% To use a custom method, create a test[method_name].m script 
% analogous to out testSoD.m etc.
EXPERIMENT.EXTRA = '';
EXPERIMENT.NUM_HYPER_OPT_ITERATIONS = -50;
EXPERIMENT.NUM_TRIALS = 1; % Number of experiment repetitions.

addpath(genpath('../gpml'));
addpath(genpath('./methods'));
addpath('../project/sod');
addpath(genpath('../project'));
disp('Adding KCenterClustering path');
addpath(genpath('../figtree-0.9.3/matlab'));

maxNumCompThreads(1);
startup
EXPERIMENT.DATASET = datasets;
EXPERIMENT.DATASET_FOLDS = 1; %total number of folds will be determined in load_data
% For SoD, FITC and Local EXPERIMENT.M is a list of the 
% subset sizes, inducing point set sizes and cluster sizes resp.
%EXPERIMENT.M = [32 64 128 256 512 1024 2048]; 

% Save raw experiment results here.
EXPERIMENT.RESULTS_DIR = './results/'; 

%----------------------------------------
% Run the experiment.
%----------------------------------------

loadData;
EXPERIMENT.DATASET_FOLDS = 1;
%if the data set is evaluated using cross validation we do that here
loadData;
EVAL_METHOD;
varTest=var(testY);
meanTest=mean(testY);
varTrain=var(trainY);
meanTrain=mean(trainY);
load(sprintf('%sresults%s_%s_fold%s_M%d', RESULTS_DIR, EXPERIMENT.METHOD, EXPERIMENT.DATASET, EXPERIMENT.DATASET_FOLD, EXPERIMENT.M));
results = eval(sprintf('results%s', EXPERIMENT.METHOD));
%EXPERIMENT.SEED = results.seeds(m);
seed = 0; %TODO: use upper line
rng(seed);
covName = 'CovSum (CovSEard, CovNoise)';
[~, ~, ~, mFT, ~] = infLibGPmex(trainX, trainY, trainX, approximation, covName, theta_over_time(:, m), EXPERIMENT.M, bfname);
[~, ~, ~, mF, ~] = infLibGPmex(trainX, trainY, testX, approximation, covName, theta_over_time(:, m), EXPERIMENT.M, bfname);

disp('Training error: ');
last_train_error = mse(mFT, trainY, meanTrain, varTrain)
last_test_error = mse(mF, testY, meanTest, varTest)
disp('NaNs or Infs: ');
any(isnan(mFT) | isinf(abs(mFT)))
