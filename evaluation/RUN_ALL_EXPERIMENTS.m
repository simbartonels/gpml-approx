% 
%datasets = {'PRECIPITATION'};%, 'CHEM', 'SARCOS'};
datasets = {'PRECIPITATION'};

% Choose the approximation to use: SoD, Local, FITC.
% To use a custom method, create a test[method_name].m script 
% analogous to out testSoD.m etc.
EXPERIMENT.METHOD = 'Multiscale';
EXPERIMENT.EXTRA = '';
EXPERIMENT.M = 50;
EXPERIMENT.NUM_HYPER_OPT_ITERATIONS = -500;
EXPERIMENT.NUM_TRIALS = 1; % Number of experiment repetitions.

addpath(genpath('../gpml'));
addpath(genpath('./methods'));
addpath('../project/sod');
addpath(genpath('../project'));
disp('Adding KCenterClustering path');
addpath(genpath('../figtree-0.9.3/matlab'));

maxNumCompThreads(1);
startup

for d=1:length(datasets)
EXPERIMENT.DATASET = datasets{d};
EXPERIMENT.DATASET_FOLD = 1;
EXPERIMENT.DATASET_FOLDS = 1; %total number of folds will be determined in load_data
% For SoD, FITC and Local EXPERIMENT.M is a list of the 
% subset sizes, inducing point set sizes and cluster sizes resp.
%EXPERIMENT.M = [32 64 128 256 512 1024 2048]; 

% Save raw experiment results here.
EXPERIMENT.RESULTS_DIR = './results/'; 

%----------------------------------------
% Run the experiment.
%----------------------------------------
fprintf('Running %d %s experiments on fold %d of %d for %s dataset.\nThis might take a while...\n', EXPERIMENT.NUM_TRIALS, EXPERIMENT.METHOD, EXPERIMENT.DATASET_FOLD, ... 
    EXPERIMENT.DATASET_FOLDS, EXPERIMENT.DATASET);
%TODO: remove
disp('REMOVE THE LINE BELOW!!!');
EXPERIMENT.DATASET_FOLD = 4
loadData;
%eval(['test' EXPERIMENT.METHOD]);
EVAL_METHOD;
%TODO: remove
disp('REMOVE THE LINE BELOW!!!');
EXPERIMENT.DATASET_FOLDS = 1
%if the data set is evaluated using cross validation we do that here
for f=2:EXPERIMENT.DATASET_FOLDS
    EXPERIMENT.DATASET_FOLD = f;
    fprintf('Running %d %s experiments on fold %d of %d for %s dataset.\nThis might take a while...\n', EXPERIMENT.NUM_TRIALS, EXPERIMENT.METHOD, EXPERIMENT.DATASET_FOLD, ... 
        EXPERIMENT.DATASET_FOLDS, EXPERIMENT.DATASET);
    loadData;
    %eval(['test' EXPERIMENT.METHOD]);
    EVAL_METHOD;
end
%----------------------------------------
% Load the results.
% results struct contains the following fields:
% results.msll - each of length(EXPERIMENT.M) columns contains
%                the mean standardized log losses returned by the 
%                EXPERIMENT.NUM_TRIALS  experiment repetitions for each
%                value of EXPERIMENT.M
%                
% results.mse  - same as results.msll, but shows mean squared errors.
% results.hyp_time - same as results.msll, but shows time used to train
%                    the hyperparameters.
% results.train_time - same as results.msll, but shows time used to train
%                      the parameters (the inverted covariance matrix mostly).
% results.test_time - same as results.msll, but shows the test times, i.e.
%                     time to compute predictive variances and means.
% results.hyps      - each of length(EXPERIMENT.M) cells contains 
%                     EXPERIMENT.NUM_TRIALS matrices. Each column
%                     of each matrix contains the hyperparameters
%                     chosen by the approximation. First D rows
%                     contain log(lengthscale),
%                     (D+1)st row contains log(amplitude) and the
%                     last row contains log(noise std dev).
%
% Any additional entries should contain approximation-specific
% information, for example indices of inducing points used, cluster
% structure, etc.
%----------------------------------------------------------------------
end
