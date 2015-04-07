clear;
EXPERIMENT.PREPROCESS_DATASET = true
% 
%datasets = {'PRECIPITATION'};%, 'CHEM', 'SARCOS'};
%datasets = {'PUMADYN'}
datasets = {'CT_SLICES'}
EXPERIMENT.CAP_TIME = 150 + 1*60*60;
EXPERIMENT.METHOD = 'FastFood'
% Choose the approximation to use: SoD, Local, FITC.
% To use a custom method, create a test[method_name].m script 
% analogous to out testSoD.m etc.
EXPERIMENT.EXTRA = [1e-6]; %, 0.0; 0.0, 0.0];
EXPERIMENT.M = 2048;
EXPERIMENT.NUM_HYPER_OPT_ITERATIONS = -10;
EXPERIMENT.NUM_TRIALS = 4; % Number of experiment repetitions.

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
loadData;
EVAL_METHOD;
%if the data set is evaluated using cross validation we do that here
for f=2:EXPERIMENT.DATASET_FOLDS
    EXPERIMENT.DATASET_FOLD = f;
    fprintf('Running %d %s experiments on fold %d of %d for %s dataset.\nThis might take a while...\n', EXPERIMENT.NUM_TRIALS, EXPERIMENT.METHOD, EXPERIMENT.DATASET_FOLD, ... 
        EXPERIMENT.DATASET_FOLDS, EXPERIMENT.DATASET);
    loadData;
    EVAL_METHOD;
end
end
