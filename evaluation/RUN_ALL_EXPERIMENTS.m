clear;
EXPERIMENT.PREPROCESS_DATASET = true
% 
%datasets = {'CPU'}
%datasets = {'PRECIPITATION'};%, 'CHEM', 'SARCOS'};
datasets = {'PUMADYN'}
%datasets = {'CT_SLICES'}
%datasets = {'INSURANCE'}
EXPERIMENT.METHOD = 'FullSE'
if size(datasets, 2) > 1, error('Several datasets are no longer supported.'); end
EXPERIMENT.NUM_HYPER_OPT_ITERATIONS = -80000;
EXPERIMENT.EXTRA = [1e-6]; %, 1.0; 1.0, 1.0];
method_is_random = false;
if strcmp(datasets{1}, 'INSURANCE')
	EXPERIMENT.CAP_TIME = 3*60*60;
	if strcmp(EXPERIMENT.METHOD, 'FastFood')
		EXPERIMENT.M = 2048;
		method_is_random = true;
	elseif strcmp(EXPERIMENT.METHOD, 'SoD')
		EXPERIMENT.M = 1700;
		method_is_random = true;
	elseif strcmp(EXPERIMENT.METHOD, 'FIC')
		EXPERIMENT.M = 100;
		method_is_random = true;
	elseif strcmp(EXPERIMENT.METHOD, 'Multiscale')
		EXPERIMENT.M = 100;
		method_is_random = true;
	else
		error('Unknown method %s', EXPERIMENT.METHOD);
	end
elseif strcmp(datasets{1}, 'PRECIPITATION')
	EXPERIMENT.CAP_TIME = 150+1*60*60;
	if strcmp(EXPERIMENT.METHOD, 'FastFood')
		EXPERIMENT.M = 2048;
		method_is_random = true;
	elseif strcmp(EXPERIMENT.METHOD, 'HSM')
		EXPERIMENT.M = 2048;
	elseif strcmp(EXPERIMENT.METHOD, 'Multiscale')
		EXPERIMENT.M = 1500;
		method_is_random = true;
	elseif strcmp(EXPERIMENT.METHOD, 'SoD')
		EXPERIMENT.M = 1940;
		method_is_random = true;
	elseif strcmp(EXPERIMENT.METHOD, 'FIC')
		EXPERIMENT.M = 1940;
		method_is_random = true;
	elseif strcmp(EXPERIMENT.METHOD, 'FullSE')
		EXPERIMENT.M = 1;
		EXPERIMENT.CAP_TIME = Inf;
	else
		error('Unknown method %s', EXPERIMENT.METHOD);
	end
elseif strcmp(datasets{1}, 'CT_SLICES')
	EXPERIMENT.CAP_TIME = 2*60*60;
	if strcmp(EXPERIMENT.METHOD, 'FICfixed')
		EXPERIMENT.M = 1440;
		method_is_random = true;
	elseif strcmp(EXPERIMENT.METHOD, 'SoD')
		EXPERIMENT.M = 1440;
		method_is_random = true;
	elseif strcmp(EXPERIMENT.METHOD, 'Multiscale')
		EXPERIMENT.M = 40;
		method_is_random = true;
	elseif strcmp(EXPERIMENT.METHOD, 'FIC')
		EXPERIMENT.M = 40
		method_is_random = true;
	else
		error('Unknown method %s', EXPERIMENT.METHOD);
	end
elseif strcmp(datasets{1}, 'PUMADYN')
	EXPERIMENT.CAP_TIME = 4*60*60;
	if strcmp(EXPERIMENT.METHOD, 'FullSE')
		EXPERIMENT.M = 1;
		EXPERIMENT.CAP_TIME = Inf;
	elseif strcmp(EXPERIMENT.METHOD, 'FIC')
		EXPERIMENT.M = 120;
		method_is_random = true;
	elseif strcmp(EXPERIMENT.METHOD, 'FastFood')
		EXPERIMENT.M = 198;
		method_is_random = true;
	else
		error('Unknown method %s', EXPERIMENT.METHOD);	
	end
else
	error('Unknown dataset %s', datasets{1});
end
% Choose the approximation to use: SoD, Local, FITC.
% To use a custom method, create a test[method_name].m script 
% analogous to out testSoD.m etc.


if method_is_random
	EXPERIMENT.NUM_TRIALS = 5; % Number of experiment repetitions.
else
	EXPERIMENT.NUM_TRIALS = 1;
end
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
